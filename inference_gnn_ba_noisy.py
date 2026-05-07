"""
Inference script for noisy-GT-trained GNN-BA model.

Loads a trained GNN model, adds noise to GT poses, runs denoising,
and outputs the recovered trajectory compared to ground truth.

Usage:
    python inference_gnn_ba_noisy.py --checkpoint gnn_ba_output/gnn_ba_noisy_best.pth --sequence 03
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets.kitti_gnn import KITTIFeatureDataset
from timesformer.models.gnn_ba import GNNBAOptimizer
from utils.gnn_ba import (
    triangulate_all_points,
    build_heterogeneous_graph,
    compute_reprojection_error,
    post_processing,
    recover_trajectory_and_poses,
)
from utils.dq import (
    dq_mult_np,
    dq_conj_np,
    dq_identity_np,
    dq_to_matrix_torch,
    relative_poses_to_absolute_dq_torch,
    matrix_to_dq_np,
)


def add_noise_to_relative_poses(rel_poses_dq, trans_noise, rot_noise):
    """Add noise to DQ relative poses."""
    noisy_dq = []
    for dq in rel_poses_dq:
        from utils.dq import _quat_to_rotmat_torch as _q2r, dq_extract_translation_torch
        import torch as _t
        from scipy.spatial.transform import Rotation
        dq_t = _t.as_tensor(dq, dtype=_t.float32).unsqueeze(0)
        R = _q2r(dq_t[..., :4]).squeeze(0).numpy()
        t = dq_extract_translation_torch(dq_t).squeeze(0).numpy()
        t += np.random.randn(3) * trans_noise
        R_noise = Rotation.from_rotvec(np.random.randn(3) * rot_noise).as_matrix()
        R_new = R_noise @ R
        T = np.eye(4)
        T[:3, :3] = R_new
        T[:3, 3] = t
        noisy_dq.append(matrix_to_dq_np(T))
    return np.array(noisy_dq)


def accumulate_relative_poses(relative_poses_dq):
    abs_dq = [dq_identity_np()]
    for rel in relative_poses_dq:
        abs_dq.append(dq_mult_np(abs_dq[-1], rel))
    return np.array(abs_dq)


def absolute_4x4_to_6dof(abs_poses):
    result = []
    for pose in abs_poses:
        R = pose[:3, :3]
        t = pose[:3, 3]
        euler = rotation_to_euler(R, seq="zyx")
        result.append(np.concatenate([euler, t]))
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description="Inference with noisy-GT-trained GNN-BA")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained GNN checkpoint (gnn_ba_noisy_best.pth)")
    parser.add_argument("--sequence", type=str, default="03")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--data_path", type=str, default="data/sequences_jpg")
    parser.add_argument("--gt_path", type=str, default="data/poses")
    parser.add_argument("--save_dir", type=str, default="gnn_ba_output")
    parser.add_argument("--trans_noise", type=float, default=0.1)
    parser.add_argument("--rot_noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for reproducible noise)")
    args_ns = parser.parse_args()

    np.random.seed(args_ns.seed)
    torch.manual_seed(args_ns.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load GNN model ----
    print(f"\nLoading GNN model from: {args_ns.checkpoint}")
    gnn_model = GNNBAOptimizer(
        hidden_dim=args_ns.hidden_dim, num_layers=args_ns.num_layers
    ).to(device)
    ckpt = torch.load(args_ns.checkpoint, map_location=device)
    gnn_model.load_state_dict(ckpt["model_state_dict"])
    gnn_model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ---- Load dataset ----
    print(f"\nLoading KITTI sequence {args_ns.sequence}...")
    dataset = KITTIFeatureDataset(
        data_path=args_ns.data_path,
        gt_path=args_ns.gt_path,
        sequence=args_ns.sequence,
        window_size=args_ns.window_size,
        overlap=args_ns.overlap,
        max_points=200,
    )
    gt_poses = dataset.gt_poses
    print(f"  {len(dataset)} windows, {len(gt_poses)} GT poses")

    # ---- Run GNN on each window ----
    print(f"\nRunning GNN denoising (noise: trans={args_ns.trans_noise}m, rot={args_ns.rot_noise}rad)...")

    window_size = args_ns.window_size
    overlap = args_ns.overlap
    opt_relative_poses_list = []
    num_success = 0
    num_failed = 0

    for idx in tqdm(range(len(dataset)), desc="GNN denoising"):
        sample = dataset[idx]

        K = sample["K"]
        tracks = sample["tracks"]
        gt_rel = sample["global_poses"]
        if isinstance(gt_rel, torch.Tensor):
            gt_rel = gt_rel.cpu().numpy()

        img_height, img_width = sample["images"][0].shape[:2]

        if len(tracks) < 10:
            num_failed += 1
            opt_relative_poses_list.append(np.zeros((window_size - 1, 8)))
            continue

        # Add noise to GT relative DQ poses
        noisy_rel_dq = add_noise_to_relative_poses(gt_rel, args_ns.trans_noise, args_ns.rot_noise)
        noisy_abs_dq = accumulate_relative_poses(noisy_rel_dq)

        # Triangulate from noisy poses
        points_3d, graph_obs, _ = triangulate_all_points(tracks, noisy_abs_dq, K)
        if len(points_3d) < 10:
            num_failed += 1
            opt_relative_poses_list.append(np.zeros((window_size - 1, 8)))
            continue

        # Normalize points
        if not torch.is_tensor(points_3d):
            points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
        pt_mean = points_3d.mean(dim=0)
        pt_std = points_3d.std(dim=0).clamp_min(1e-6)
        points_3d_norm = (points_3d - pt_mean) / pt_std

        # Build graph
        data = build_heterogeneous_graph(
            window_size, points_3d_norm, graph_obs, img_height, img_width
        )
        data["camera"].x = torch.tensor(noisy_abs_dq, dtype=torch.float32)
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

    print(f"\n  Success: {num_success}/{len(dataset)}, Failed: {num_failed}")

    # ---- Merge windows and recover trajectory ----
    print("Post-processing...")
    opt_relative_poses = np.array(opt_relative_poses_list)
    processed = post_processing(opt_relative_poses, window_size, overlap)
    opt_poses, opt_trajectory = recover_trajectory_and_poses(processed, use_dq=True)

    print(f"  Recovered {len(opt_poses)} optimized poses")

    # ---- Compute ATE (absolute trajectory error) ----
    gt_pos = np.array([p[:3, 3] for p in gt_poses])
    opt_pos = np.array([p[:3, 3] for p in opt_poses])

    min_len = min(len(gt_pos), len(opt_pos))
    gt_pos = gt_pos[:min_len]
    opt_pos = opt_pos[:min_len]

    rmse = np.sqrt(np.mean(np.sum((gt_pos - opt_pos) ** 2, axis=1)))
    print(f"\nATE RMSE: {rmse:.4f}m over {min_len} frames")

    # ---- Save results ----
    os.makedirs(args_ns.save_dir, exist_ok=True)

    # Save optimized poses
    pose_path = os.path.join(args_ns.save_dir, f"poses_{args_ns.sequence}_noisy_opt.txt")
    with open(pose_path, "w") as f:
        for pose in opt_poses:
            flat = np.concatenate([pose[:3, :3].flatten(), pose[:3, 3]])
            f.write(" ".join(f"{v:.6f}" for v in flat) + "\n")
    print(f"  Saved poses: {pose_path}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(gt_pos[:, 0], gt_pos[:, 2], "b-", label="Ground Truth", lw=2)
        ax.plot(opt_pos[:, 0], opt_pos[:, 2], "g-.", label=f"GNN Denoised (ATE={rmse:.2f}m)", lw=1.5, alpha=0.7)
        ax.scatter(gt_pos[0, 0], gt_pos[0, 2], c="b", s=100, marker="o", zorder=5, label="Start")
        ax.scatter(gt_pos[-1, 0], gt_pos[-1, 2], c="b", s=100, marker="x", zorder=5, label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Trajectory — Seq {args_ns.sequence} (Noisy GT → GNN)")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

        plot_path = os.path.join(args_ns.save_dir, f"traj_noisy_infer_{args_ns.sequence}.pdf")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {plot_path}")
    except ImportError:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
