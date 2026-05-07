"""
Train GNN-based Bundle Adjustment with noisy GT poses (no VO model dependency).

Adds Gaussian noise to GT relative poses, triangulates 3D points from noisy
absolute poses, and trains the GNN to recover clean GT poses and points.

Optimized: triangulation runs inside DataLoader workers (parallel), GT point
triangulation is pre-computed once.

Usage:
    python train_gnn_ba_noisy.py --sequence 03 --num_epochs 100
"""

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from tqdm import tqdm

from datasets.kitti_gnn import KITTIFeatureDataset
from timesformer.models.gnn_ba import GNNBAOptimizer
from utils.gnn_ba import (
    triangulate_all_points,
    triangulate_tracks_no_filter,
    build_heterogeneous_graph,
    project_points_torch,
    get_obs_depths_batch,
    huber_loss,
    compute_reprojection_error,
    post_processing,
    recover_trajectory_and_poses,
    pose_6dof_to_dq,
    euler_to_rotation,
)
from utils.dq import (
    dq_mult_np,
    dq_conj_np,
    dq_conj_torch,
    dq_mult_torch,
    dq_normalize_torch,
    dq_geodesic_loss,
    dq_transform_point_torch,
    relative_poses_to_absolute_dq_torch,
    dq_to_matrix_torch,
    matrix_to_dq_np,
    dq_identity_np,
    dq_identity_torch,
)


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def relative_poses_to_absolute_torch(relative_dq, first_abs_dq, device):
    return relative_poses_to_absolute_dq_torch(relative_dq, first_abs_dq, device)


def add_noise_to_relative_poses(rel_poses_dq, trans_noise, rot_noise):
    """Add noise to DQ relative poses by converting to 6-DOF, adding noise, converting back."""
    noisy_dq = []
    for dq in rel_poses_dq:
        R, t = dq_to_rt_np_noisy(dq)
        t += np.random.randn(3) * trans_noise
        from scipy.spatial.transform import Rotation
        rot_vec = np.random.randn(3) * rot_noise
        R_noise = Rotation.from_rotvec(rot_vec).as_matrix()
        R_new = R_noise @ R
        T = np.eye(4)
        T[:3, :3] = R_new
        T[:3, 3] = t
        noisy_dq.append(matrix_to_dq_np(T))
    return np.array(noisy_dq)


def accumulate_relative_poses(relative_poses_dq):
    """Accumulate relative DQ poses into absolute DQ poses."""
    abs_dq = [dq_identity_np()]
    for rel in relative_poses_dq:
        abs_dq.append(dq_mult_np(abs_dq[-1], rel))
    return np.array(abs_dq)


def dq_to_rt_np_noisy(dq):
    """Extract R and t from DQ (numpy)."""
    from utils.dq import _quat_to_rotmat_torch as _q2r, dq_extract_translation_torch
    import torch as _t
    dq_t = _t.as_tensor(dq, dtype=_t.float32).unsqueeze(0)
    R_t = _q2r(dq_t[..., :4]).squeeze(0)
    t_t = dq_extract_translation_torch(dq_t).squeeze(0)
    return R_t.numpy(), t_t.numpy()


def absolute_4x4_to_6dof(abs_poses):
    """Convert absolute 4x4 poses to DQ representation."""
    return np.array([matrix_to_dq_np(p) for p in abs_poses])


# =============================================================================
# Wrapper dataset: noise + triangulation inside DataLoader workers
# =============================================================================

class NoisyWrapperDataset(torch.utils.data.Dataset):
    """
    Wraps KITTIFeatureDataset.  Base samples + GT points are cached once.
    Noise and triangulation are refreshed at the start of each epoch via
    refresh_epoch() so the GNN sees different noise every epoch (no overfitting).
    """

    def __init__(self, base_dataset, trans_noise, rot_noise, window_size):
        self.base = base_dataset
        self.trans_noise = trans_noise
        self.rot_noise = rot_noise
        self.window_size = window_size

        n = len(self.base)
        print(f"Loading {n} base samples + GT triangulation (once)...")
        self._base = []      # noise-independent data per sample
        self._noisy = [None] * n  # noise-dependent cache, refreshed each epoch

        for idx in tqdm(range(n), desc="Loading base"):
            sample = self.base[idx]
            tracks = sample["tracks"]
            K = sample["K"]
            gt_rel = sample["global_poses"]
            if isinstance(gt_rel, torch.Tensor):
                gt_rel = gt_rel.cpu().numpy()
            gt_abs = sample["abs_poses"]
            images = sample["images"]
            img_h, img_w = images[0].shape[:2]

            # GT point triangulation (noise-independent)
            if len(tracks) >= 2:
                T_inv = np.linalg.inv(gt_abs[0])
                gt_abs_rel = [T_inv @ p for p in gt_abs]
                gt_abs_dq = [matrix_to_dq_np(p) for p in gt_abs_rel]
                all_indices = list(range(len(tracks)))
                gt_pts_all = triangulate_tracks_no_filter(
                    tracks, all_indices, gt_abs_dq, K, device="cpu"
                )
            else:
                gt_pts_all = torch.zeros(0, 3)

            self._base.append({
                "tracks": tracks,
                "K": K,
                "gt_rel": gt_rel,
                "gt_abs": gt_abs,
                "img_h": img_h,
                "img_w": img_w,
                "gt_pts_all": gt_pts_all,
            })

        self._valid_count = sum(
            1 for b in self._base if len(b["tracks"]) >= 10
        )
        print(f"  {self._valid_count} samples with >=10 tracks")

    def refresh_epoch(self):
        """Regenerate noise + triangulate all samples (call once per epoch)."""
        for idx, b in enumerate(self._base):
            tracks = b["tracks"]
            K = b["K"]
            gt_rel = b["gt_rel"]
            img_h, img_w = b["img_h"], b["img_w"]
            gt_pts_all = b["gt_pts_all"]

            if len(tracks) < 10:
                self._noisy[idx] = {
                    "n_pts": 0,
                    "points_3d_norm": torch.zeros(0, 3),
                    "camera_feats": np.zeros((self.window_size, 8)),
                    "graph_obs": [],
                    "pt_mean": torch.zeros(3),
                    "pt_std": torch.ones(3),
                    "img_width": img_w,
                    "img_height": img_h,
                    "n_cam": self.window_size,
                    "gt_rel_dq": gt_rel,
                    "noisy_rel_dq": np.zeros((self.window_size - 1, 8)),
                    "gt_points_3d": torch.zeros(0, 3),
                    "K": K,
                }
                continue

            # Add noise + triangulate
            noisy_rel_dq = add_noise_to_relative_poses(gt_rel, self.trans_noise, self.rot_noise)
            noisy_abs_dq = accumulate_relative_poses(noisy_rel_dq)

            points_3d, graph_obs, valid_tracks = triangulate_all_points(
                tracks, noisy_abs_dq, K
            )

            if len(points_3d) < 10:
                self._noisy[idx] = {
                    "n_pts": 0,
                    "points_3d_norm": torch.zeros(0, 3),
                    "camera_feats": np.zeros((self.window_size, 8)),
                    "graph_obs": [],
                    "pt_mean": torch.zeros(3),
                    "pt_std": torch.ones(3),
                    "img_width": img_w,
                    "img_height": img_h,
                    "n_cam": self.window_size,
                    "gt_rel_dq": gt_rel,
                    "noisy_rel_dq": np.zeros((self.window_size - 1, 8)),
                    "gt_points_3d": torch.zeros(0, 3),
                    "K": K,
                }
                continue

            # Slice GT points
            if len(valid_tracks) > 0 and len(gt_pts_all) > max(valid_tracks):
                gt_points_3d = gt_pts_all[valid_tracks]
            else:
                gt_points_3d = torch.zeros(0, 3)

            # Normalize points
            if not torch.is_tensor(points_3d):
                points_3d = torch.tensor(points_3d, dtype=torch.float32)
            pt_mean = points_3d.mean(dim=0)
            pt_std = points_3d.std(dim=0).clamp_min(1e-6)
            points_3d_norm = (points_3d - pt_mean) / pt_std

            self._noisy[idx] = {
                "n_pts": points_3d.shape[0],
                "points_3d_norm": points_3d_norm,
                "camera_feats": noisy_abs_dq,
                "graph_obs": graph_obs,
                "pt_mean": pt_mean,
                "pt_std": pt_std,
                "img_width": img_w,
                "img_height": img_h,
                "n_cam": self.window_size,
                "gt_rel_dq": gt_rel,
                "noisy_rel_dq": noisy_rel_dq,
                "gt_points_3d": gt_points_3d,
                "K": K,
            }

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._noisy[idx]


def noisy_collate_fn(batch):
    """Collate — keeps per-sample data as lists, no PyG objects transferred."""
    collated = {}
    for key in batch[0].keys():
        if key in ("K",):
            collated[key] = [sample[key] for sample in batch]
        elif key in ("points_3d_norm", "pt_mean", "pt_std", "gt_points_3d"):
            collated[key] = [sample[key] for sample in batch]
        elif key in ("camera_feats", "noisy_rel_dq"):
            collated[key] = [torch.tensor(sample[key], dtype=torch.float32) for sample in batch]
        elif key == "graph_obs":
            collated[key] = [sample[key] for sample in batch]
        elif key in ("gt_rel_dq", "gt_rel", "global_poses"):
            collated[key] = [sample[key] for sample in batch]
        else:
            collated[key] = [sample[key] for sample in batch]
    return collated


# =============================================================================
# Training
# =============================================================================

def train_epoch(gnn_model, train_loader, optimizer, epoch, args, device):
    """Train GNN for one epoch.  All heavy lifting is in the DataLoader workers."""
    gnn_model.train()
    running_loss = 0
    running_reproj_loss = 0
    running_pose_loss = 0
    running_point_loss = 0
    num_loss_samples = 0
    epoch_loss = 0
    num_batches = 0

    mean_angles_cpu = torch.tensor([1.7061e-5, 9.5582e-4, -5.5258e-5], dtype=torch.float32)
    std_angles_cpu = torch.tensor([2.8256e-3, 1.7771e-2, 3.2326e-3], dtype=torch.float32)
    mean_t_cpu = torch.tensor([-8.6736e-5, -1.6038e-2, 9.0033e-1], dtype=torch.float32)
    std_t_cpu = torch.tensor([2.5584e-2, 1.8545e-2, 3.0352e-1], dtype=torch.float32)

    window_size = args["window_size"]
    reproj_weight = args.get("reproj_weight", 0.01)
    pose_weight = args.get("pose_weight", 1.0)
    point_weight = args.get("point_weight", 0.1)
    k = args.get("weighted_loss", 3.0)
    loss_clip = args.get("loss_clip", 500.0)

    n_edges_per_sample = window_size - 1

    with tqdm(train_loader, unit="batch", dynamic_ncols=True) as tepoch:
        for batch_data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            optimizer.zero_grad()

            batch_size_eff = len(batch_data["n_pts"])

            # ---- Build graphs in main process (no PyG serialization) ----
            graphs = []
            for i in range(batch_size_eff):
                n_pts = batch_data["n_pts"][i]
                n_cam = batch_data["n_cam"][i]
                img_width = batch_data["img_width"][i]
                img_height = batch_data["img_height"][i]
                points_3d_norm = batch_data["points_3d_norm"][i]
                camera_feats = batch_data["camera_feats"][i]
                graph_obs = batch_data["graph_obs"][i]
                noisy_rel_dq = batch_data["noisy_rel_dq"][i]

                if n_pts < 10:
                    graphs.append(None)
                    continue

                if not torch.is_tensor(points_3d_norm):
                    points_3d_norm = torch.tensor(points_3d_norm, dtype=torch.float32)

                data = build_heterogeneous_graph(
                    n_cam, points_3d_norm, graph_obs, img_height, img_width
                )
                data["camera"].x = camera_feats

                # c2c edges: use DQ noisy relative pose
                c2c_index = torch.stack([
                    torch.arange(0, n_cam - 1), torch.arange(1, n_cam)
                ], dim=0)
                data["camera", "temporal", "camera"].edge_index = c2c_index
                data["camera", "temporal", "camera"].edge_attr = noisy_rel_dq
                data = data.to(device)
                graphs.append(data)
            # ---- Batched GNN forward ----
            valid_graphs = [g for g in graphs if g is not None]
            if not valid_graphs:
                continue
            gnn_batch = Batch.from_data_list(valid_graphs).to(device)
            pred_rel_poses_all, point_refined_all = gnn_model(gnn_batch, output_mode="relative")

            # Per-sample loss
            batch_loss = None
            valid_samples_in_batch = 0
            valid_count = 0  # index into pred_rel_poses_all / point_refined_all
            pt_offset = 0    # cumulative n_pts for valid samples

            for i in range(batch_size_eff):
                n_pts = batch_data["n_pts"][i]
                if n_pts < 10:
                    if graphs[i] is not None:
                        valid_count += 1
                    continue

                img_width = batch_data["img_width"][i]
                img_height = batch_data["img_height"][i]
                K = batch_data["K"][i]
                gt_points_3d = batch_data["gt_points_3d"][i].to(device)

                edge_start = valid_count * n_edges_per_sample
                pred_rel_poses = pred_rel_poses_all[
                    edge_start : edge_start + n_edges_per_sample
                ]

                # Denormalize GNN DQ output and compose absolute DQ
                first_pose_dq = dq_identity_torch((), device)
                abs_dq = relative_poses_to_absolute_dq_torch(
                    pred_rel_poses, first_pose_dq, device
                )

                # Denormalize GNN-refined points
                pt_mean = batch_data["pt_mean"][i].to(device)
                pt_std = batch_data["pt_std"][i].to(device)
                points_3d_refined = point_refined_all[pt_offset : pt_offset + n_pts]
                points_3d_refined = points_3d_refined * pt_std + pt_mean

                # Reprojection error
                edge_index = graphs[i]["camera", "observes", "point"].edge_index
                edge_attr = graphs[i]["camera", "observes", "point"].edge_attr
                u_obs = edge_attr[:, 0] * img_width
                v_obs = edge_attr[:, 1] * img_height

                projected = project_points_torch(points_3d_refined, abs_dq, K, device)
                cam_indices = edge_index[0]
                pt_indices = edge_index[1]
                proj_x = projected[cam_indices, pt_indices, 0]
                proj_y = projected[cam_indices, pt_indices, 1]
                errors = torch.sqrt((proj_x - u_obs) ** 2 + (proj_y - v_obs) ** 2)

                obs_depths = get_obs_depths_batch(abs_dq, points_3d_refined, cam_indices, pt_indices)
                valid_errors = errors[obs_depths > 0.1]

                # Pose supervision: geodesic SE(3) loss
                gt_rel_dq = batch_data["gt_rel_dq"][i]
                if isinstance(gt_rel_dq, np.ndarray) and gt_rel_dq.ndim == 3:
                    gt_rel_dq = gt_rel_dq[0]
                gt_rel_dq_tensor = torch.as_tensor(gt_rel_dq, dtype=torch.float32, device=device)

                pose_loss = dq_geodesic_loss(pred_rel_poses, gt_rel_dq_tensor, k=k)

                # Point supervision
                if gt_points_3d.numel() > 0 and points_3d_refined.shape[0] == gt_points_3d.shape[0]:
                    point_loss = huber_loss(points_3d_refined - gt_points_3d, delta=1.0).mean()
                else:
                    point_loss = torch.tensor(0.0, device=device)

                if valid_errors.numel() == 0:
                    valid_count += 1
                    pt_offset += n_pts
                    continue

                reproj_loss = huber_loss(valid_errors, delta=5.0).mean()
                mean_pose_reg = 0.01 * pred_rel_poses.mean(dim=0).norm()
                loss = (
                    reproj_loss * reproj_weight
                    + pose_loss * pose_weight
                    + point_loss * point_weight
                    + mean_pose_reg
                )

                if not torch.isfinite(loss) or loss.item() > loss_clip:
                    valid_count += 1
                    pt_offset += n_pts
                    continue

                if batch_loss is None:
                    batch_loss = loss
                else:
                    batch_loss = batch_loss + loss

                running_loss += loss.item()
                running_reproj_loss += reproj_loss.item()
                running_pose_loss += pose_loss.item()
                running_point_loss += point_loss.item()
                num_loss_samples += 1
                valid_samples_in_batch += 1
                valid_count += 1
                pt_offset += n_pts

            if valid_samples_in_batch > 0:
                batch_loss = batch_loss / valid_samples_in_batch
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += batch_loss.item()
                num_batches += 1

            tepoch.set_postfix(
                avg_loss=running_loss / max(num_loss_samples, 1),
                avg_reproj=running_reproj_loss / max(num_loss_samples, 1),
                avg_pose=running_pose_loss / max(num_loss_samples, 1),
                avg_point=running_point_loss / max(num_loss_samples, 1),
            )

    return {
        "total": epoch_loss / max(num_batches, 1),
        "reproj": running_reproj_loss / max(num_loss_samples, 1),
        "pose": running_pose_loss / max(num_loss_samples, 1),
        "point": running_point_loss / max(num_loss_samples, 1),
    }


# =============================================================================
# Validation
# =============================================================================

def validate(gnn_model, val_loader, args, device):
    """Validate GNN using noisy GT poses."""
    gnn_model.eval()
    total_error_initial = 0
    total_error_optimized = 0
    num_samples = 0

    window_size = args["window_size"]

    mean_angles_cpu = torch.tensor([1.7061e-5, 9.5582e-4, -5.5258e-5], dtype=torch.float32)
    std_angles_cpu = torch.tensor([2.8256e-3, 1.7771e-2, 3.2326e-3], dtype=torch.float32)
    mean_t_cpu = torch.tensor([-8.6736e-5, -1.6038e-2, 9.0033e-1], dtype=torch.float32)
    std_t_cpu = torch.tensor([2.5584e-2, 1.8545e-2, 3.0352e-1], dtype=torch.float32)

    with torch.no_grad():
        for batch_data in val_loader:
            batch_size_eff = len(batch_data["n_pts"])

            for i in range(batch_size_eff):
                n_pts = batch_data["n_pts"][i]
                if n_pts < 10:
                    continue

                K = batch_data["K"][i]
                img_height = batch_data["img_height"][i]
                img_width = batch_data["img_width"][i]
                graph_obs = batch_data["graph_obs"][i]
                camera_feats = batch_data["camera_feats"][i]
                noisy_rel_dq = batch_data["noisy_rel_dq"][i]
                points_3d_norm = batch_data["points_3d_norm"][i]
                pt_mean = batch_data["pt_mean"][i].to(device)
                pt_std = batch_data["pt_std"][i].to(device)

                if not torch.is_tensor(points_3d_norm):
                    points_3d_norm = torch.tensor(points_3d_norm, dtype=torch.float32)

                # Build graph
                data = build_heterogeneous_graph(
                    window_size, points_3d_norm, graph_obs, img_height, img_width
                )
                data["camera"].x = camera_feats
                # c2c edges: use DQ noisy relative pose
                c2c_index = torch.stack([
                    torch.arange(0, window_size - 1), torch.arange(1, window_size)
                ], dim=0)
                if not torch.is_tensor(noisy_rel_dq):
                    noisy_rel_dq = torch.tensor(noisy_rel_dq, dtype=torch.float32)
                data["camera", "temporal", "camera"].edge_index = c2c_index
                data["camera", "temporal", "camera"].edge_attr = noisy_rel_dq
                data = data.to(device)

                pred_rel_poses, point_refined = gnn_model(data, output_mode="relative")

                # Compose absolute DQ
                first_pose_dq = dq_identity_torch((), device)
                opt_abs_dq = relative_poses_to_absolute_dq_torch(pred_rel_poses, first_pose_dq, device)
                opt_abs_np = opt_abs_dq.cpu().numpy()  # (M, 8) DQ

                # Denormalize points
                point_refined_denorm = point_refined * pt_std + pt_mean
                opt_points_np = point_refined_denorm.cpu().numpy()

                # Reconstruct noisy absolute DQ from camera_feats
                if isinstance(camera_feats, torch.Tensor):
                    cf = camera_feats.cpu().numpy()
                else:
                    cf = camera_feats

                # Initial points: denormalize the noisy triangulation
                points_3d_np = (points_3d_norm * pt_std.cpu() + pt_mean.cpu()).cpu().numpy()

                initial_error = compute_reprojection_error(
                    cf, points_3d_np, graph_obs, K, img_height, img_width
                )
                optimized_error = compute_reprojection_error(
                    opt_abs_np, opt_points_np, graph_obs, K, img_height, img_width
                )

                total_error_initial += initial_error
                total_error_optimized += optimized_error
                num_samples += 1

    return {
        "initial_error": total_error_initial / max(num_samples, 1),
        "optimized_error": total_error_optimized / max(num_samples, 1),
    }


# =============================================================================
# Full-sequence prediction
# =============================================================================

def predict_full_sequence(gnn_model, dataset, args, device):
    """Predict full sequence trajectory via GNN denoising."""
    gnn_model.eval()

    window_size = args["window_size"]
    overlap = args["overlap"]
    trans_noise = args.get("trans_noise", 0.1)
    rot_noise = args.get("rot_noise", 0.02)

    opt_relative_poses_list = []
    num_success = 0

    for idx in tqdm(range(len(dataset)), desc="GNN optimization"):
        sample = dataset[idx]

        K = sample["K"]
        tracks = sample["tracks"]
        gt_rel = sample["global_poses"]
        if isinstance(gt_rel, torch.Tensor):
            gt_rel = gt_rel.cpu().numpy()

        img_height, img_width = sample["images"][0].shape[:2]

        if len(tracks) < 10:
            opt_relative_poses_list.append(np.zeros((window_size - 1, 8)))
            continue

        noisy_rel_dq = add_noise_to_relative_poses(gt_rel, trans_noise, rot_noise)
        noisy_abs_dq = accumulate_relative_poses(noisy_rel_dq)
        camera_feats = noisy_abs_dq

        points_3d, graph_obs, _ = triangulate_all_points(tracks, noisy_abs_dq, K)
        if len(points_3d) < 10:
            opt_relative_poses_list.append(np.zeros((window_size - 1, 8)))
            continue

        if not torch.is_tensor(points_3d):
            points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
        pt_mean = points_3d.mean(dim=0)
        pt_std = points_3d.std(dim=0).clamp_min(1e-6)
        points_3d_norm = (points_3d - pt_mean) / pt_std

        data = build_heterogeneous_graph(
            window_size, points_3d_norm, graph_obs, img_height, img_width
        )
        data["camera"].x = torch.tensor(camera_feats, dtype=torch.float32)
        # c2c edges: use DQ noisy relative pose
        c2c_index = torch.stack([
            torch.arange(0, window_size - 1), torch.arange(1, window_size)
        ], dim=0)
        data["camera", "temporal", "camera"].edge_index = c2c_index
        data["camera", "temporal", "camera"].edge_attr = torch.tensor(noisy_rel_dq, dtype=torch.float32)
        data = data.to(device)

        with torch.no_grad():
            pred_rel_poses, _ = gnn_model(data, output_mode="relative")

        num_success += 1
        opt_relative_poses_list.append(pred_rel_poses.detach().cpu().numpy())

    print(f"\nGNN Optimization: {num_success}/{len(dataset)} windows successful")

    opt_relative_poses = np.array(opt_relative_poses_list)
    processed = post_processing(opt_relative_poses, window_size, overlap)
    opt_poses, _ = recover_trajectory_and_poses(processed, use_dq=True)
    return opt_poses


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GNN-BA with noisy GT poses")
    parser.add_argument("--sequence", type=str, default="03")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weighted_loss", type=float, default=3.0)
    parser.add_argument("--reproj_weight", type=float, default=0.01)
    parser.add_argument("--pose_weight", type=float, default=1.0)
    parser.add_argument("--point_weight", type=float, default=0.1)
    parser.add_argument("--loss_clip", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non_deterministic", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="data/sequences_jpg")
    parser.add_argument("--gt_path", type=str, default="data/poses")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="gnn_ba_output")
    parser.add_argument("--trans_noise", type=float, default=0.1)
    parser.add_argument("--rot_noise", type=float, default=0.02)

    args_ns = parser.parse_args()
    if args_ns.non_deterministic:
        args_ns.deterministic = False
    else:
        args_ns.deterministic = True

    set_seed(args_ns.seed, deterministic=args_ns.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args_ns.save_dir, exist_ok=True)

    print(f"Noise: trans={args_ns.trans_noise:.3f}m, rot={args_ns.rot_noise:.3f}rad")
    print(f"Sequence: {args_ns.sequence}")

    # ---- Load base dataset ----
    base_dataset = KITTIFeatureDataset(
        data_path=args_ns.data_path,
        gt_path=args_ns.gt_path,
        sequence=args_ns.sequence,
        window_size=args_ns.window_size,
        overlap=args_ns.overlap,
        max_points=200,
    )

    if args_ns.debug:
        print("Debug mode: limiting to 100 windows")
        base_dataset = torch.utils.data.Subset(
            base_dataset, list(range(min(100, len(base_dataset))))
        )

    print(f"Base dataset size: {len(base_dataset)} windows")

    gt_poses = (
        base_dataset.dataset.gt_poses
        if isinstance(base_dataset, torch.utils.data.Subset)
        else base_dataset.gt_poses
    )

    # ---- Wrap with noise + triangulation in __getitem__ ----
    noisy_dataset = NoisyWrapperDataset(
        base_dataset, args_ns.trans_noise, args_ns.rot_noise, args_ns.window_size
    )

    # ---- Train/val split ----
    num_val = max(1, int(len(noisy_dataset) * args_ns.val_split))
    train_size = len(noisy_dataset) - num_val
    split_gen = torch.Generator().manual_seed(args_ns.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        noisy_dataset, [train_size, num_val], generator=split_gen
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args_ns.batch_size, shuffle=True,
        num_workers=args_ns.num_workers, collate_fn=noisy_collate_fn,
        persistent_workers=(args_ns.num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args_ns.batch_size, shuffle=False,
        num_workers=args_ns.num_workers, collate_fn=noisy_collate_fn,
        persistent_workers=(args_ns.num_workers > 0),
    )

    # ---- GNN model ----
    gnn_model = GNNBAOptimizer(
        hidden_dim=args_ns.hidden_dim, num_layers=args_ns.num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(
        gnn_model.parameters(), lr=args_ns.lr, weight_decay=args_ns.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args_ns.num_epochs, eta_min=args_ns.lr * 0.1
    )

    print(f"\nTraining for {args_ns.num_epochs} epochs...")
    print(f"  Train batches: {len(train_loader)} × {args_ns.batch_size}")
    best_val_error = float("inf")

    train_args = {
        "window_size": args_ns.window_size,
        "batch_size": args_ns.batch_size,
        "overlap": args_ns.overlap,
        "weighted_loss": args_ns.weighted_loss,
        "reproj_weight": args_ns.reproj_weight,
        "pose_weight": args_ns.pose_weight,
        "point_weight": args_ns.point_weight,
        "loss_clip": args_ns.loss_clip,
        "trans_noise": args_ns.trans_noise,
        "rot_noise": args_ns.rot_noise,
    }

    for epoch in range(1, args_ns.num_epochs + 1):
        # Fresh noise each epoch — prevents overfitting to a single noise pattern
        noisy_dataset.refresh_epoch()

        train_metrics = train_epoch(
            gnn_model, train_loader, optimizer, epoch, train_args, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch}: Train Total = {train_metrics['total']:.4f}, "
            f"Train Reproj = {train_metrics['reproj']:.4f}, "
            f"Train Pose = {train_metrics['pose']:.4f}"
        )

        if epoch % 2 == 0:
            val_metrics = validate(gnn_model, val_loader, train_args, device)
            print(
                f"Epoch {epoch}: Train Loss = {train_metrics['total']:.4f}, "
                f"Val Init = {val_metrics['initial_error']:.2f}px, "
                f"Val Opt  = {val_metrics['optimized_error']:.2f}px"
            )

            if val_metrics["optimized_error"] < best_val_error:
                best_val_error = val_metrics["optimized_error"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": gnn_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_error": best_val_error,
                    },
                    os.path.join(args_ns.save_dir, "gnn_ba_noisy_best.pth"),
                )
                print(f"Saved best model: {best_val_error:.2f}px")

    print("\nTraining completed!")

    # ---- Final trajectory ----
    print("\nGenerating full trajectory...")
    opt_poses = predict_full_sequence(gnn_model, base_dataset, train_args, device)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt_pos = np.array([p[:3, 3] for p in gt_poses])
        opt_pos = np.array([p[:3, 3] for p in opt_poses])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(gt_pos[:, 0], gt_pos[:, 2], "b-", label="GT", lw=2)
        ax.plot(opt_pos[:, 0], opt_pos[:, 2], "g-.", label="GNN", lw=1.5, alpha=0.7)
        ax.scatter(gt_pos[0, 0], gt_pos[0, 2], c="b", s=100, marker="o", zorder=5)
        ax.scatter(gt_pos[-1, 0], gt_pos[-1, 2], c="b", s=100, marker="x", zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Trajectory — Seq {args_ns.sequence} (noisy GT → GNN)")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")
        sp = os.path.join(args_ns.save_dir, f"traj_noisy_{args_ns.sequence}.pdf")
        plt.savefig(sp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot: {sp}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
