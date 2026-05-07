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
    euler_to_rotation,
    poses_to_camera_features,
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

    mean_angles_np = KITTI_MEAN_ANGLES.copy()
    std_angles_np = KITTI_STD_ANGLES.copy()
    mean_t_np = KITTI_MEAN_T.copy()
    std_t_np = KITTI_STD_T.copy()

    for idx in tqdm(range(num_windows), desc="GNN optimization"):
        sample = dataset[idx]

        K = sample["K"]
        tracks = sample["tracks"]
        # Reformat tracks to (frame_idx, u, v)
        rel_tracks = []
        for track in tracks:
            if len(track) >= 2:
                rel_tracks.append([(obs[0], obs[1], obs[2]) for obs in track])
        tracks = rel_tracks

        if len(tracks) >= 10:
            # Build absolute poses from per-window VO (match training, NOT merged global VO)
            vo_out = vo_relative_poses[idx]  # (window_size-1, 6) normalized
            rel_poses_denorm = []
            for i in range(vo_out.shape[0]):
                euler = vo_out[i, :3] * std_angles_np + mean_angles_np
                t = vo_out[i, 3:] * std_t_np + mean_t_np
                rel_poses_denorm.append(np.concatenate([euler, t]))

            abs_poses_4x4 = [np.eye(4)]
            for rel_pose in rel_poses_denorm:
                R = euler_to_rotation(rel_pose[:3], seq="zyx")
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = rel_pose[3:]
                abs_poses_4x4.append(abs_poses_4x4[-1] @ T_rel)

            # Convert to 6-DOF for camera features
            abs_poses_6dof = []
            for pose in abs_poses_4x4:
                R = pose[:3, :3]
                t = pose[:3, 3]
                euler = rotation_to_euler(R, seq="zyx")
                abs_poses_6dof.append(np.concatenate([euler, t]))
            abs_poses_6dof = np.array(abs_poses_6dof)  # denorm

            # Triangulate 3D points using per-window VO poses
            points_3d, graph_obs, _ = triangulate_all_points(tracks, abs_poses_4x4, K)
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
                # Use per-window normalized absolute poses as GNN input (match training)
                cam_mean = abs_poses_6dof.mean(axis=0)
                cam_std = abs_poses_6dof.std(axis=0).clip(min=1e-6)
                abs_poses_6dof_norm = (abs_poses_6dof - cam_mean) / cam_std
                camera_feats_tensor = torch.tensor(abs_poses_6dof_norm, dtype=torch.float32)
                data["camera"].x = camera_feats_tensor
                # c2c edges: use exact VO relative pose (already normalized)
                c2c_index = torch.stack([
                    torch.arange(0, window_size - 1), torch.arange(1, window_size)
                ], dim=0)
                c2c_attr = torch.tensor(vo_relative_poses[idx], dtype=torch.float32)
                data["camera", "temporal", "camera"].edge_index = c2c_index
                data["camera", "temporal", "camera"].edge_attr = c2c_attr
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
    parser.add_argument("--sequence", type=str, default=None, help="Single sequence (overridden by --test_seqs)")
    parser.add_argument("--test_seqs", type=str, default=None,
                        help="Comma-separated test sequences, e.g. 01,03,05,07,10")
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

    # Determine test sequences
    if args.test_seqs is not None:
        test_seq_list = [s.strip() for s in args.test_seqs.split(",")]
    elif args.sequence is not None:
        test_seq_list = [args.sequence]
    else:
        test_seq_list = ["03"]

    print(f"\nEvaluating on sequences: {test_seq_list}")

    all_ate = {}
    for seq in test_seq_list:
        print(f"\n--- Sequence {seq} ---")

        # Load dataset
        dataset = KITTIFeatureDataset(
            data_path=args.data_path,
            gt_path=args.gt_path,
            sequence=seq,
            window_size=args.window_size,
            overlap=args.overlap,
            max_points=200,
        )
        if args.debug:
            print(f"Debug mode: limiting dataset to first 100 windows")
            dataset = torch.utils.data.Subset(dataset, list(range(min(100, len(dataset)))))
        print(f"Dataset size: {len(dataset)} windows")

        # Run inference
        print("Running inference...")
        vo_poses, opt_poses = predict_sequence_simple(
            vo_model, gnn_model, dataset, args, device
        )

        # Compute ATE
        gt_raw = dataset.dataset.gt_poses if isinstance(dataset, torch.utils.data.Subset) else dataset.gt_poses
        gt_pos = np.array([p[:3, 3] for p in gt_raw])
        vo_pos = np.array([p[:3, 3] for p in vo_poses])
        opt_pos = np.array([p[:3, 3] for p in opt_poses])
        n = min(len(gt_pos), len(vo_pos), len(opt_pos))
        ate = np.sqrt(np.mean(np.sum((gt_pos[:n] - opt_pos[:n]) ** 2, axis=1)))
        ate_vo = np.sqrt(np.mean(np.sum((gt_pos[:n] - vo_pos[:n]) ** 2, axis=1)))
        all_ate[seq] = (ate_vo, ate)

        # Save poses
        pred_dir = os.path.join(args.save_dir, "pred_pose")
        os.makedirs(pred_dir, exist_ok=True)
        vo_path = os.path.join(args.save_dir, f"poses_{seq}_vo.txt")
        opt_path = os.path.join(pred_dir, f"{seq}.txt")
        with open(vo_path, "w") as f:
            for p in vo_poses:
                vals = p[:3, :4]
                f.write(" ".join(f"{vals[i, j]:.6f}" for i in range(3) for j in range(4)) + "\n")
        with open(opt_path, "w") as f:
            for p in opt_poses:
                vals = p[:3, :4]
                f.write(" ".join(f"{vals[i, j]:.6f}" for i in range(3) for j in range(4)) + "\n")

        # Plot trajectory
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            gt_poses_4x4 = []
            for pose_3x4 in gt_raw[: n]:
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :4] = pose_3x4
                gt_poses_4x4.append(pose_4x4)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(gt_pos[:, 0], gt_pos[:, 2], "b-", label="GT", lw=1.5)
            ax.plot(vo_pos[:, 0], vo_pos[:, 2], "r--", label=f"VO (ATE={ate_vo:.2f}m)", lw=1, alpha=0.5)
            ax.plot(opt_pos[:, 0], opt_pos[:, 2], "g-", label=f"GNN (ATE={ate:.2f}m)", lw=1.5)
            ax.set_title(f"Seq {seq} — VO={ate_vo:.2f}m  GNN={ate:.2f}m")
            ax.legend(); ax.grid(True); ax.axis("equal")
            plt.savefig(os.path.join(args.save_dir, f"traj_{seq}.png"), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        print(f"  VO ATE: {ate_vo:.4f}m  |  GNN ATE: {ate:.4f}m")

    # Summary table
    print("\n" + "=" * 55)
    print("ATE SUMMARY")
    print("=" * 55)
    print(f"  {'Seq':>5}  {'VO ATE':>8}  {'GNN ATE':>8}  {'Δ':>8}")
    print("  " + "-" * 40)
    ate_vo_list, ate_gnn_list = [], []
    for seq, (vo_a, gnn_a) in all_ate.items():
        delta = vo_a - gnn_a
        print(f"  {seq:>5}  {vo_a:>8.4f}  {gnn_a:>8.4f}  {delta:>+8.4f}")
        ate_vo_list.append(vo_a)
        ate_gnn_list.append(gnn_a)
    print("  " + "-" * 40)
    print(f"  Mean:  {np.mean(ate_vo_list):>8.4f}  {np.mean(ate_gnn_list):>8.4f}  {np.mean(ate_vo_list)-np.mean(ate_gnn_list):>+8.4f}")
    print(f"  Median:{np.median(ate_vo_list):>8.4f}  {np.median(ate_gnn_list):>8.4f}  {np.median(ate_vo_list)-np.median(ate_gnn_list):>+8.4f}")



if __name__ == "__main__":
    main()
