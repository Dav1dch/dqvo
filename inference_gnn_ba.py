"""
Simple inference script for GNN-based Bundle Adjustment.

This script loads a trained GNN model and predicts optimized relative poses
for each window, then merges overlapping windows into a full trajectory.
"""

import argparse
import os

import cv2
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
    KITTI_STD_ANGLES,
    KITTI_MEAN_T,
    KITTI_STD_T,
    triangulate_all_points,
    build_heterogeneous_graph,
    rotation_to_euler,
    post_processing,
    recover_trajectory_and_poses,
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



def plot_trajectories(gt_poses, initial_poses, optimized_poses, save_path):
    import matplotlib.pyplot as plt

    gt_positions = np.array([p[:3, 3] for p in gt_poses])
    initial_positions = np.array([p[:3, 3] for p in initial_poses])
    opt_positions = np.array([p[:3, 3] for p in optimized_poses])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(
        gt_positions[:, 0], gt_positions[:, 2], "b-", label="Ground Truth", linewidth=2
    )
    ax.plot(
        initial_positions[:, 0],
        initial_positions[:, 2],
        "r--",
        label="Initial VO",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.plot(
        opt_positions[:, 0],
        opt_positions[:, 2],
        "g-.",
        label="GNN Optimized",
        linewidth=1.5,
        alpha=0.7,
    )

    ax.scatter(
        gt_positions[0, 0], gt_positions[0, 2], c="b", s=100, marker="o", zorder=5
    )
    ax.scatter(
        gt_positions[-1, 0], gt_positions[-1, 2], c="b", s=100, marker="x", zorder=5
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Camera Trajectory Comparison")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved to: {save_path}")


def predict_sequence_simple(vo_model, gnn_model, dataset, args, device):
    """
    Inference using GNN-BA with proper post-processing of relative poses.

    The GNN directly predicts optimized relative poses between consecutive frames.
    These relative poses are normalized and merged across overlapping windows
    to produce a smooth continuous trajectory.
    """
    gnn_model.eval()
    vo_model.eval()

    window_size = args.window_size
    overlap = args.overlap
    step_size = window_size - overlap
    num_windows = len(dataset)

    # Step 1: Collect raw VO outputs (normalized relative poses) for all windows
    print("Step 1: Computing VO trajectory...")
    vo_relative_poses = []
    for idx in tqdm(range(num_windows), desc="Collecting VO outputs"):
        sample = dataset[idx]
        pil_images = []
        for img in sample["images"]:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img)
            pil_images.append(preprocess(pil_img))
        imgs_batch = torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            vo_output = vo_model(imgs_batch)
        vo_output_np = vo_output.cpu().numpy()[0]  # (window_size-1, 6) normalized
        vo_relative_poses.append(vo_output_np)

    # Post-process VO relative poses to get initial absolute trajectory
    raw_vo_array = np.array(vo_relative_poses)  # (num_windows, window_size-1, 6)
    processed_vo = post_processing(raw_vo_array, window_size, overlap)
    vo_poses, vo_trajectory = recover_trajectory_and_poses(processed_vo, norm=True)
    print(f"VO trajectory: {len(vo_poses)} poses")

    # Step 2: Run GNN on each window, collect optimized relative poses
    print("\nStep 2: Running GNN optimization...")
    opt_relative_poses_list = []
    rel_pose_norms = []
    num_success = 0
    num_triangulation_failed = 0
    num_track_count_low = 0

    for idx in tqdm(range(num_windows), desc="GNN optimization"):
        sample = dataset[idx]

        # Prepare images
        pil_images = []
        for img in sample["images"]:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img)
            pil_images.append(preprocess(pil_img))
        imgs_batch = torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)

        K = sample["K"]
        tracks = sample["tracks"]
        # Reformat tracks to (frame_idx, u, v)
        rel_tracks = []
        for track in tracks:
            if len(track) >= 2:
                rel_tracks.append([(obs[0], obs[1], obs[2]) for obs in track])
        tracks = rel_tracks

        # Get absolute poses for this window from global VO trajectory
        window_start = idx * step_size
        if window_start >= len(vo_poses):
            window_abs_poses = [vo_poses[-1]] * window_size
        elif window_start + window_size > len(vo_poses):
            window_abs_poses = vo_poses[window_start:]
            while len(window_abs_poses) < window_size:
                window_abs_poses.append(vo_poses[-1])
        else:
            window_abs_poses = vo_poses[window_start : window_start + window_size]

        # Convert absolute poses to 6-DOF (denorm), then normalize for GNN
        abs_poses_6dof = []
        for pose in window_abs_poses:
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu().numpy()
            R = pose[:3, :3]
            t = pose[:3, 3]
            euler = rotation_to_euler(R, seq="zyx")
            abs_poses_6dof.append(np.concatenate([euler, t]))
        abs_poses_6dof = np.array(abs_poses_6dof)  # denorm

        # Build graph and run GNN if enough tracks
        if len(tracks) >= 10:
            points_3d, graph_obs, _ = triangulate_all_points(tracks, window_abs_poses, K)
            if len(points_3d) >= 10:
                # Normalize points for GNN input
                if not torch.is_tensor(points_3d):
                    points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
                pt_mean = points_3d.mean(dim=0)
                pt_std = points_3d.std(dim=0).clamp_min(1e-6)
                points_3d_norm = (points_3d - pt_mean) / pt_std

                img_height, img_width = sample["images"][0].shape[:2]
                data = build_heterogeneous_graph(
                    window_size, points_3d_norm, graph_obs, img_height, img_width
                )
                # Use denormalized absolute poses directly as GNN input.
                # GNN resolves c2c edges internally from camera.x (same as
                # validate() in train_gnn_ba.py and predict_full_sequence()),
                # computing (abs[i+1]-abs[i]) / std — consistent with training.
                camera_feats_tensor = torch.tensor(abs_poses_6dof, dtype=torch.float32)
                data["camera"].x = camera_feats_tensor
                data = data.to(device)

                with torch.no_grad():
                    pred_rel_poses, _ = gnn_model(data, output_mode="relative")
                rel_pose_norms.append(torch.norm(pred_rel_poses).item())

                # GNN output is already normalized; use directly for post-processing
                pred_rel_np = pred_rel_poses.detach().cpu().numpy()
                opt_relative_poses_list.append(pred_rel_np)
                num_success += 1
            else:
                # Fallback: use VO relative poses for this window
                num_triangulation_failed += 1
                opt_relative_poses_list.append(vo_relative_poses[idx])
        else:
            # Fallback: use VO relative poses for this window
            num_track_count_low += 1
            opt_relative_poses_list.append(vo_relative_poses[idx])

    # Step 3: Merge optimized relative poses and recover absolute trajectory
    opt_relative_poses_array = np.array(
        opt_relative_poses_list
    )  # (num_windows, window_size-1, 6)
    processed_opt_relative = post_processing(
        opt_relative_poses_array, window_size, overlap
    )
    opt_poses, opt_trajectory = recover_trajectory_and_poses(
        processed_opt_relative, norm=True
    )
    print(f"Optimized trajectory: {len(opt_poses)} poses")
    if rel_pose_norms:
        print(
            f"Relative pose norms - mean: {np.mean(rel_pose_norms):.4f}, max: {np.max(rel_pose_norms):.4f}"
        )

    # Print debug statistics
    print(f"\nGNN Optimization Debug Stats:")
    print(
        f"  Success: {num_success}/{num_windows} ({100*num_success/num_windows:.1f}%)"
    )
    print(f"  Triangulation failed: {num_triangulation_failed}")
    print(f"  Track count low: {num_track_count_low}")
    if num_success > 0:
        print(
            f"  Avg relative pose norm: {np.mean(rel_pose_norms):.6f}"
        )

    return vo_poses, opt_poses


def main():
    parser = argparse.ArgumentParser(description="GNN-BA Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Exp51/checkpoint_best.pth",
        help="Path to VO model checkpoint",
    )
    parser.add_argument(
        "--gnn_model",
        type=str,
        default="gnn_ba_output/gnn_ba_best.pth",
        help="Path to trained GNN model",
    )
    parser.add_argument("--sequence", type=str, default="03", help="Sequence number")
    parser.add_argument("--debug", action="store_true", help="Debug mode: use first 100 windows only")
    parser.add_argument("--window_size", type=int, default=3, help="Window size")
    parser.add_argument(
        "--overlap", type=int, default=2, help="Overlap between windows"
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
        "--save_dir", type=str, default="gnn_ba_output", help="Output directory"
    )
    parser.add_argument("--num_layers", type=int, default=6, help="Number of GNN layers")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load VO model
    print(f"\nLoading VO model from: {args.checkpoint}")
    model_params = {
        "dim": 384,
        "image_size": (224, 672),
        "patch_size": 16,
        "attention_type": "divided_space_time",
        "num_frames": args.window_size,
        "num_classes": 6 * (args.window_size - 1),
        "depth": 16,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.2,
        "ff_dropout": 0.2,
        "time_only": False,
    }

    model_args = {
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_path": (
            os.path.dirname(args.checkpoint)
            if os.path.dirname(args.checkpoint)
            else "."
        ),
        "pretrained_ViT": False,
    }

    vo_model, _ = build_model(model_args, model_params)
    vo_model.eval()
    print("VO model loaded successfully")

    # Load GNN model
    print(f"\nLoading GNN model from: {args.gnn_model}")
    gnn_model = GNNBAOptimizer(hidden_dim=128, num_layers=args.num_layers).to(device)
    checkpoint = torch.load(args.gnn_model, map_location=device, weights_only=False)
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
    print("GNN model loaded successfully")

    # Load dataset
    print(f"\nLoading KITTI dataset for sequence {args.sequence}...")
    dataset = KITTIFeatureDataset(
        data_path=args.data_path,
        gt_path=args.gt_path,
        sequence=args.sequence,
        window_size=args.window_size,
        overlap=args.overlap,
        max_points=200,
    )
    if args.debug:
        print(f"Debug mode: limiting dataset to first 100 windows")
        dataset = torch.utils.data.Subset(dataset, list(range(min(100, len(dataset)))))
    print(f"Dataset size: {len(dataset)} windows")

    # Run inference
    print("\nRunning inference...")
    vo_poses, opt_poses = predict_sequence_simple(
        vo_model, gnn_model, dataset, args, device
    )

    # Save poses
    seq_vo_path = os.path.join(args.save_dir, f"poses_{args.sequence}_vo.txt")
    seq_opt_path = os.path.join(args.save_dir, f"poses_{args.sequence}_opt.txt")

    with open(seq_vo_path, "w") as f:
        for pose in vo_poses:
            row = pose.flatten()[:12]
            f.write(" ".join([str(v) for v in row]) + "\n")

    with open(seq_opt_path, "w") as f:
        for pose in opt_poses:
            row = pose.flatten()[:12]
            f.write(" ".join([str(v) for v in row]) + "\n")

    print(f"\nVO poses saved to: {seq_vo_path}")
    print(f"Optimized poses saved to: {seq_opt_path}")

    # Plot trajectory comparison
    gt_poses = dataset.dataset.gt_poses if isinstance(dataset, torch.utils.data.Subset) else dataset.gt_poses
    gt_poses_4x4 = []
    for pose_3x4 in gt_poses[: len(opt_poses)]:
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :4] = pose_3x4
        gt_poses_4x4.append(pose_4x4)

    plot_trajectories(
        gt_poses_4x4,
        vo_poses,
        opt_poses,
        os.path.join(args.save_dir, "trajectory_comparison.png"),
    )

    print(f"\nTrajectory plot saved to: {args.save_dir}/trajectory_comparison.png")

    # Print comparison
    print("\n=== Trajectory Comparison (first 5 frames) ===")
    print("Frame | VO t | Optimized t | Diff norm")
    for i in range(min(5, len(vo_poses), len(opt_poses))):
        vo_t = vo_poses[i][:3, 3]
        opt_t = opt_poses[i][:3, 3]
        diff = np.linalg.norm(opt_t - vo_t)
        print(
            f"  {i}   | ({vo_t[0]:.3f}, {vo_t[1]:.3f}, {vo_t[2]:.3f}) | ({opt_t[0]:.3f}, {opt_t[1]:.3f}, {opt_t[2]:.3f}) | {diff:.3f}m"
        )


if __name__ == "__main__":
    main()
