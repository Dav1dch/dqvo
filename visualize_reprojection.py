"""
Visualize feature matching and pose reprojection.

Evaluation protocol:
1. Triangulate 3D points from frame 1 & 3 (indices 0 & 2) using ORB feature matches and GT poses.
2. Reproject these points to frame 3 (index 2) using:
   - VO pose (before optimization)
    - GNN pose reconstructed from direct relative-pose predictions
3. Visualize side-by-side:
   - Left: Frame 2 with matched feature points (from ORB matching between frame 1 & 2)
    - Right: Frame 3 with reprojected points (VO and GNN) compared to observed keypoints

Evaluation modes:
- gt_points: evaluate all methods on GT-triangulated points
- both: print GT/VO/GNN pose comparison using GT-triangulated points

Usage:
    python visualize_reprojection.py --vo_checkpoint checkpoints/Exp51/checkpoint_best.pth --gnn_checkpoint gnn_ba_output/gnn_ba_best.pth --sequence 03
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from build_model import build_model
from datasets.kitti_gnn import KITTIFeatureDataset
from timesformer.models.gnn_ba import GNNBAOptimizer
from utils.gnn_ba import (
    KITTI_MEAN_ANGLES,
    KITTI_MEAN_T,
    KITTI_STD_ANGLES,
    KITTI_STD_T,
    build_heterogeneous_graph,
    denormalize_poses,
    euler_to_rotation,
    triangulate_all_points,
    rotation_to_euler,
)


preprocess = transforms.Compose(
    [
        transforms.Resize((224, 672)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183],
        ),
    ]
)


def enrich_graph_for_new_gnn(data, relative_pose_features):
    """Add explicit point->camera and camera->camera edges expected by new GNN."""
    obs_edge_type = ("camera", "observes", "point")
    if obs_edge_type in data.edge_types:
        cam_point_edge_index = data[obs_edge_type].edge_index
        cam_point_edge_attr = data[obs_edge_type].edge_attr
        data["point", "observed_by", "camera"].edge_index = torch.stack(
            [cam_point_edge_index[1], cam_point_edge_index[0]], dim=0
        )
        data["point", "observed_by", "camera"].edge_attr = cam_point_edge_attr.clone()

    num_cameras = data["camera"].x.shape[0]
    num_c2c_edges = max(num_cameras - 1, 0)

    if num_c2c_edges == 0:
        c2c_edge_index = torch.zeros(2, 0, dtype=torch.long)
        c2c_edge_attr = torch.zeros(0, 6, dtype=torch.float32)
    else:
        src = torch.arange(0, num_c2c_edges, dtype=torch.long)
        dst = src + 1
        c2c_edge_index = torch.stack([src, dst], dim=0)

        rel_pose_tensor = torch.as_tensor(relative_pose_features, dtype=torch.float32)
        if rel_pose_tensor.ndim == 3:
            rel_pose_tensor = rel_pose_tensor[0]

        c2c_edge_attr = torch.zeros(num_c2c_edges, 6, dtype=torch.float32)
        usable = min(num_c2c_edges, rel_pose_tensor.shape[0])
        if usable > 0:
            c2c_edge_attr[:usable] = rel_pose_tensor[:usable, :6]

    data["camera", "temporal", "camera"].edge_index = c2c_edge_index
    data["camera", "temporal", "camera"].edge_attr = c2c_edge_attr
    return data


def visualize_side_by_side(
    img_frame2,
    img_frame3,
    matches_frame2,
    gt_errors,
    vo_errors,
    opt_errors,
    window_indices,
    save_path,
):
    """
    Create side-by-side visualization:
    - Left: Frame 2 with matched feature points (from ORB matching between frame 1 & 2)
    - Right: Frame 3 with reprojected points (GT, VO, and GNN)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Prepare images
    if len(img_frame2.shape) == 2:
        img2_display = cv2.cvtColor(img_frame2, cv2.COLOR_GRAY2RGB)
    else:
        img2_display = img_frame2.copy()

    if len(img_frame3.shape) == 2:
        img3_display = cv2.cvtColor(img_frame3, cv2.COLOR_GRAY2RGB)
    else:
        img3_display = img_frame3.copy()

    # === Left: Frame 2 with matched feature points ===
    ax1.imshow(cv2.cvtColor(img2_display, cv2.COLOR_BGR2RGB))
    ax1.set_title(
        f"Frame {window_indices[1]} - Matched Features", fontsize=12, fontweight="bold"
    )

    # Plot matched points (from triangulation)
    if matches_frame2:
        matches_array = np.array(matches_frame2)
        ax1.scatter(
            matches_array[:, 0],
            matches_array[:, 1],
            c="lime",
            s=20,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label="Matched features",
        )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.axis("off")

    # === Right: Frame 3 with reprojection results ===
    ax2.imshow(cv2.cvtColor(img3_display, cv2.COLOR_BGR2RGB))
    ax2.set_title(
        f"Frame {window_indices[2]} - Reprojection", fontsize=12, fontweight="bold"
    )

    # Plot observed keypoints (ground truth positions)
    if vo_errors:
        obs_points = np.array([[e["u_obs"], e["v_obs"]] for e in vo_errors])
        ax2.scatter(
            obs_points[:, 0],
            obs_points[:, 1],
            c="yellow",
            s=15,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label="Observed",
            alpha=0.7,
        )

    # Plot GT projections
    if gt_errors:
        gt_proj = np.array([[e["u_proj"], e["v_proj"]] for e in gt_errors])
        ax2.scatter(
            gt_proj[:, 0],
            gt_proj[:, 1],
            c="blue",
            s=15,
            marker="x",
            label="GT",
            alpha=0.8,
        )
        # Draw lines connecting observed to GT projections
        for e in gt_errors:
            ax2.plot(
                [e["u_obs"], e["u_proj"]],
                [e["v_obs"], e["v_proj"]],
                "b-",
                linewidth=0.5,
                alpha=0.3,
            )

    # Plot VO projections
    if vo_errors:
        vo_proj = np.array([[e["u_proj"], e["v_proj"]] for e in vo_errors])
        ax2.scatter(
            vo_proj[:, 0],
            vo_proj[:, 1],
            c="red",
            s=15,
            marker="x",
            label="VO",
            alpha=0.8,
        )
        # Draw lines connecting observed to VO projections
        for e in vo_errors:
            ax2.plot(
                [e["u_obs"], e["u_proj"]],
                [e["v_obs"], e["v_proj"]],
                "r-",
                linewidth=0.5,
                alpha=0.3,
            )

    # Plot GNN projections
    if opt_errors:
        opt_proj = np.array([[e["u_proj"], e["v_proj"]] for e in opt_errors])
        ax2.scatter(
            opt_proj[:, 0],
            opt_proj[:, 1],
            c="green",
            s=15,
            marker="x",
            label="GNN",
            alpha=0.8,
        )
        # Draw lines connecting observed to GNN projections
        for e in opt_errors:
            ax2.plot(
                [e["u_obs"], e["u_proj"]],
                [e["v_obs"], e["v_proj"]],
                "g-",
                linewidth=0.5,
                alpha=0.3,
            )

    ax2.legend(loc="upper right", fontsize=9)
    ax2.axis("off")

    # Add error statistics
    gt_avg = np.mean([e["error"] for e in gt_errors]) if gt_errors else 0
    vo_avg = np.mean([e["error"] for e in vo_errors]) if vo_errors else 0
    opt_avg = np.mean([e["error"] for e in opt_errors]) if opt_errors else 0
    improvement_vo = (gt_avg - vo_avg) / gt_avg * 100 if gt_avg > 0 else 0
    improvement_opt = (gt_avg - opt_avg) / gt_avg * 100 if gt_avg > 0 else 0

    plt.suptitle(
        f"Reprojection: GT ({gt_avg:.2f}px) → VO ({vo_avg:.2f}px) → GNN ({opt_avg:.2f}px)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize feature matching and reprojection"
    )
    parser.add_argument(
        "--vo_checkpoint",
        type=str,
        default="checkpoints/Exp51/checkpoint_best.pth",
        help="Path to VO model checkpoint",
    )
    parser.add_argument(
        "--gnn_checkpoint",
        type=str,
        default="gnn_ba_output/gnn_ba_best.pth",
        help="Path to GNN model checkpoint",
    )
    parser.add_argument("--sequence", type=str, default="03", help="Sequence number")
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="GNN hidden dimension"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="gnn_ba_output/reprojection_vis",
        help="Output directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sequences_jpg",
        help="Path to KITTI sequences",
    )
    parser.add_argument(
        "--gt_path", type=str, default="data/poses", help="Path to KITTI poses"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="gt_points",
        choices=["gt_points", "both"],
        help="Evaluation mode for reprojection error",
    )
    parser.add_argument("--num_layers", type=int, default=6, help="Number of GNN layers")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load VO model
    print(f"\nLoading VO model from: {args.vo_checkpoint}")
    model_params = {
        "dim": 384,
        "image_size": (224, 672),
        "patch_size": 16,
        "attention_type": "divided_space_time",
        "num_frames": 3,
        "num_classes": 12,
        "depth": 16,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.2,
        "ff_dropout": 0.2,
        "time_only": False,
    }
    model_args = {
        "checkpoint": os.path.basename(args.vo_checkpoint),
        "checkpoint_path": (
            os.path.dirname(args.vo_checkpoint)
            if os.path.dirname(args.vo_checkpoint)
            else "."
        ),
        "pretrained_ViT": False,
    }
    vo_model, _ = build_model(model_args, model_params)
    vo_model.eval()
    print("VO model loaded")

    # Load GNN model
    print(f"\nLoading GNN model from: {args.gnn_checkpoint}")
    gnn_model = GNNBAOptimizer(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    checkpoint = torch.load(args.gnn_checkpoint, map_location=device)
    load_result = gnn_model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        print("Warning: loaded GNN checkpoint with non-strict key matching")
        if load_result.missing_keys:
            print(f"  Missing keys: {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")
    gnn_model.eval()
    print(f"GNN model loaded (epoch {checkpoint.get('epoch', '?')})")

    # Load dataset
    print(f"\nLoading KITTI dataset for sequence {args.sequence}...")
    dataset = KITTIFeatureDataset(
        data_path=args.data_path,
        gt_path=args.gt_path,
        sequence=args.sequence,
        window_size=3,
        overlap=2,
        max_points=200,
    )
    print(f"Dataset size: {len(dataset)} windows")

    # Process samples
    print(f"\nProcessing {args.num_samples} samples...")
    for sample_idx in tqdm(range(min(args.num_samples, len(dataset)))):
        sample = dataset[sample_idx]

        images = sample["images"]
        tracks = sample["tracks"]
        K = sample["K"]
        window_indices = sample["window_indices"]
        gt_abs_poses = sample["abs_poses"]

        if len(tracks) < 10:
            continue

        # Preprocess images for VO model
        pil_images = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img)
            pil_images.append(preprocess(pil_img))

        imgs_batch = torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)

        # Get VO poses
        with torch.no_grad():
            vo_output = vo_model(imgs_batch)
        vo_output_np = vo_output.reshape(-1, 2, 6).detach().cpu().numpy()[0]

        vo_poses = denormalize_poses(
            vo_output,
            3,
            torch.tensor(KITTI_MEAN_ANGLES, device=device),
            torch.tensor(KITTI_STD_ANGLES, device=device),
            torch.tensor(KITTI_MEAN_T, device=device),
            torch.tensor(KITTI_STD_T, device=device),
        )[0]

        # === Step 1: Triangulate 3D points from frames 0 & 2 using GT poses ===
        K_matrix = np.array(
            [[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]], dtype=np.float64
        )

        # Build w2c (world-to-camera) poses from c2w (camera-to-world) poses
        # KITTI poses are c2w, so we need to invert to get w2c for projection matrices
        def pose_c2w_to_projection(pose_c2w, K):
            pose_w2c = np.linalg.inv(pose_c2w)
            Rt_w2c = pose_w2c[:3, :]
            P = K @ Rt_w2c
            return P

        P0 = pose_c2w_to_projection(gt_abs_poses[0], K_matrix)
        P2 = pose_c2w_to_projection(gt_abs_poses[2], K_matrix)

        points_3d_gt = []
        observations_frame2 = []
        matches_on_frame2 = []

        for track in tracks:
            frames_in_track = [obs[0] for obs in track]
            if 0 in frames_in_track and 2 in frames_in_track:
                obs0 = next(obs for obs in track if obs[0] == 0)
                obs2 = next(obs for obs in track if obs[0] == 2)
                u0, v0 = obs0[1], obs0[2]
                u2, v2 = obs2[1], obs2[2]

                try:
                    obs_h = np.array([[u0], [v0]], dtype=np.float64), np.array(
                        [[u2], [v2]], dtype=np.float64
                    )

                    point_4d = cv2.triangulatePoints(P0, P2, obs_h[0], obs_h[1])
                    X_3d = (point_4d[:3] / point_4d[3]).flatten()

                    if X_3d[2] < 0.1 or np.isnan(X_3d[2]):
                        continue

                    points_3d_gt.append(X_3d)
                    observations_frame2.append((len(points_3d_gt) - 1, 2, u2, v2))
                    matches_on_frame2.append((u2, v2))
                except Exception:
                    continue

        if len(points_3d_gt) < 10:
            continue

        points_3d_gt = np.array(points_3d_gt, dtype=np.float32)

        # === Step 2: Get optimized poses from GNN direct relative predictions ===
        # Triangulate using VO for GNN input (as in training)
        points_vo, graph_obs_vo, _ = triangulate_all_points(tracks, vo_poses, K)
        if len(points_vo) < 10:
            continue

        # Build graph from VO triangulation
        img_height, img_width = images[0].shape[:2]
        data = build_heterogeneous_graph(3, points_vo, graph_obs_vo, img_height, img_width)

        # Use accumulated absolute VO poses as GNN camera-node input.
        # The model predicts relative poses between consecutive frames.
        abs_poses_6dof = []
        for pose in vo_poses:
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu().numpy()
            R = pose[:3, :3]
            t = pose[:3, 3]
            euler = rotation_to_euler(R, seq='zyx')
            abs_poses_6dof.append(np.concatenate([euler, t]))
        abs_poses_6dof = np.array(abs_poses_6dof)  # denorm

        # Use denormalized absolute poses directly as GNN input
        camera_feats_tensor = torch.tensor(abs_poses_6dof, dtype=torch.float32)
        data["camera"].x = camera_feats_tensor
        data = enrich_graph_for_new_gnn(data, vo_output_np)
        data = data.to(device)

        with torch.no_grad():
            pred_rel_poses, _ = gnn_model(data, output_mode="relative")

        # Denormalize GNN output and compose optimized absolute poses.
        pred_rel_np = pred_rel_poses.detach().cpu().numpy()  # normalized
        # Denormalize
        pred_rel_np_denorm = pred_rel_np.copy()
        pred_rel_np_denorm[:, :3] = pred_rel_np_denorm[:, :3] * KITTI_STD_ANGLES + KITTI_MEAN_ANGLES
        pred_rel_np_denorm[:, 3:] = pred_rel_np_denorm[:, 3:] * KITTI_STD_T + KITTI_MEAN_T

        first_pose = vo_poses[0]
        if isinstance(first_pose, torch.Tensor):
            first_pose = first_pose.cpu().numpy()

        opt_poses = [first_pose.copy()]
        for rel_pose in pred_rel_np_denorm:
            R_rel = euler_to_rotation(rel_pose[:3], seq="zyx")
            T_rel = np.eye(4)
            T_rel[:3, :3] = R_rel
            T_rel[:3, 3] = rel_pose[3:]
            opt_poses.append(opt_poses[-1] @ T_rel)

        # === Step 3: Compute reprojection errors on frame 2 (index 2) ===
        K_matrix = np.array(
            [[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]], dtype=np.float64
        )

        def compute_errors(poses, points, observations, target_frame=2):
            errs = []
            for pt_idx, frame_idx, u_obs, v_obs in observations:
                if frame_idx != target_frame:
                    continue
                if pt_idx >= len(points):
                    continue
                X_h = np.append(points[pt_idx], 1.0)
                pose_w2c = np.linalg.inv(poses[target_frame])
                X_cam2 = pose_w2c @ X_h
                if X_cam2[2] < 0.1:
                    continue
                proj = K_matrix @ X_cam2[:3]
                proj = proj[:2] / proj[2]
                error = np.sqrt((proj[0] - u_obs) ** 2 + (proj[1] - v_obs) ** 2)
                errs.append(
                    {
                        "point_idx": pt_idx,
                        "frame_idx": 2,
                        "error": error,
                        "u_obs": u_obs,
                        "v_obs": v_obs,
                        "u_proj": proj[0],
                        "v_proj": proj[1],
                    }
                )
            return errs

        gt_errors = compute_errors(gt_abs_poses, points_3d_gt, observations_frame2)
        vo_errors = compute_errors(vo_poses, points_3d_gt, observations_frame2)
        opt_errors_gt_points = compute_errors(opt_poses, points_3d_gt, observations_frame2)

        if args.eval_mode == "gt_points":
            opt_errors = opt_errors_gt_points
        else:
            opt_errors = opt_errors_gt_points

        # Print errors
        if gt_errors:
            gt_avg = np.mean([e["error"] for e in gt_errors])
            print(f"  GT avg reprojection error: {gt_avg:.4f} px")
        if vo_errors:
            vo_avg = np.mean([e["error"] for e in vo_errors])
            print(f"  VO avg reprojection error: {vo_avg:.4f} px")
        if opt_errors_gt_points:
            opt_avg_gt = np.mean([e["error"] for e in opt_errors_gt_points])
            print(f"  GNN avg reprojection error (gt_points): {opt_avg_gt:.4f} px")
        if not opt_errors:
            print("  GNN reprojection error: no valid observations for selected eval mode")

        # === Step 4: Visualize side-by-side ===
        save_path = os.path.join(args.save_dir, f"reprojection_sample_{sample_idx}.png")
        visualize_side_by_side(
            images[1],  # Frame 2 (index 1)
            images[2],  # Frame 3 (index 2)
            matches_on_frame2,  # Matched points on frame 2
            gt_errors,
            vo_errors,
            opt_errors,
            window_indices,
            save_path,
        )

    print(f"\nAll visualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
