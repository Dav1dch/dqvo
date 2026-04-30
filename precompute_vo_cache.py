"""
Pre-compute VO poses and triangulation points for GNN-BA training.

This script runs the frozen VO model on all windows and pre-computes:
- VO camera features (normalized 6-DoF poses for GNN input)
- VO absolute 4x4 poses (for reprojection loss)
- VO triangulated 3D points
- Graph observations (point_idx, frame_idx, u, v)
- GT triangulated 3D points (for point supervision loss)

Saves everything to a single .pt file for fast loading during training.

Usage:
    python precompute_vo_cache.py --checkpoint checkpoints/Exp51/checkpoint_best.pth --sequence 03
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
from utils.gnn_ba import (
    KITTI_MEAN_ANGLES,
    KITTI_STD_ANGLES,
    KITTI_MEAN_T,
    KITTI_STD_T,
    triangulate_all_points,
    triangulate_tracks_no_filter,
    rotation_to_euler,
    euler_to_rotation,
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


def precompute_window(vo_model, dataset, idx, window_size, device):
    """Pre-compute VO poses and triangulation for a single window."""
    sample = dataset[idx]
    tracks = sample["tracks"]
    images = sample["images"]
    K = sample["K"]
    gt_abs_poses = sample["abs_poses"]
    gt_relative_poses = sample["global_poses"]
    img_height, img_width = images[0].shape[:2]

    # Make GT poses relative to first camera frame (match VO coordinate frame)
    T_inv = np.linalg.inv(gt_abs_poses[0])
    gt_abs_poses_rel = [T_inv @ p for p in gt_abs_poses]

    if len(tracks) < 10:
        return None

    # Preprocess images for VO model
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.squeeze(0).numpy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_images.append(preprocess(Image.fromarray(img)))

    vo_imgs = torch.stack(pil_images, dim=1).unsqueeze(0).to(device)

    # VO forward pass
    with torch.no_grad():
        vo_output = vo_model(vo_imgs)  # (1, window_size-1, 6)
    vo_out = vo_output.cpu().numpy()[0]  # (window_size-1, 6)

    # Denormalize relative poses and accumulate to absolute 4x4
    mean_angles_np = KITTI_MEAN_ANGLES.copy()
    std_angles_np = KITTI_STD_ANGLES.copy()
    mean_t_np = KITTI_MEAN_T.copy()
    std_t_np = KITTI_STD_T.copy()

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

    # Convert absolute 4x4 poses to 6-DoF for GNN camera features
    abs_poses_6dof = []
    for pose in abs_poses_4x4:
        R = pose[:3, :3]
        t = pose[:3, 3]
        euler = rotation_to_euler(R, seq="zyx")
        abs_poses_6dof.append(np.concatenate([euler, t]))
    camera_feats = np.array(abs_poses_6dof)

    # VO triangulation
    points_3d, graph_obs, valid_tracks = triangulate_all_points(
        tracks, abs_poses_4x4, K
    )
    if len(points_3d) < 10:
        return None

    if not torch.is_tensor(points_3d):
        points_3d = torch.tensor(points_3d, dtype=torch.float32)
    else:
        points_3d = points_3d.cpu()

    # GT triangulation (same tracks, no cheirality filter)
    gt_points_3d = triangulate_tracks_no_filter(
        tracks, valid_tracks, gt_abs_poses_rel, K, device="cpu"
    )
    if not torch.is_tensor(gt_points_3d):
        gt_points_3d = torch.tensor(gt_points_3d, dtype=torch.float32)
    else:
        gt_points_3d = gt_points_3d.cpu()

    return {
        "vo_camera_feats": camera_feats,             # (window_size, 6)
        "vo_abs_poses_4x4": abs_poses_4x4,            # list of 4x4 numpy arrays
        "points_3d": points_3d,                       # (N, 3)
        "graph_obs": graph_obs,                       # list of (point_idx, frame_idx, u, v)
        "valid_tracks": valid_tracks,                 # list of track indices
        "gt_points_3d": gt_points_3d,                 # (N, 3)
        "gt_relative_poses": gt_relative_poses,       # (window_size-1, 6)
        "img_height": img_height,
        "img_width": img_width,
        "n_cam": window_size,
        "n_pts": points_3d.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VO poses and triangulation cache")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Exp51/checkpoint_best.pth",
        help="Path to VO model checkpoint",
    )
    parser.add_argument("--sequence", type=str, default="03", help="Sequence number")
    parser.add_argument("--window_size", type=int, default=3, help="Window size")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between windows")
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
        "--output",
        type=str,
        default=None,
        help="Output cache file path (default: cache/vo_cache_{sequence}.pt)",
    )
    parser.add_argument("--max_points", type=int, default=200, help="Max ORB features per frame")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    vo_model.to(device)
    print("VO model loaded successfully")

    # Load dataset
    print(f"\nLoading KITTI dataset for sequence {args.sequence}...")
    dataset = KITTIFeatureDataset(
        data_path=args.data_path,
        gt_path=args.gt_path,
        sequence=args.sequence,
        window_size=args.window_size,
        overlap=args.overlap,
        max_points=args.max_points,
    )
    print(f"Dataset size: {len(dataset)} windows")

    # Pre-compute all windows
    print("\nPre-computing VO poses and triangulation...")
    cache = {}
    valid_count = 0
    skipped_count = 0

    for idx in tqdm(range(len(dataset)), desc="Pre-computing"):
        result = precompute_window(vo_model, dataset, idx, args.window_size, device)
        if result is not None:
            cache[idx] = result
            valid_count += 1
        else:
            skipped_count += 1

    print(f"\nPre-computation complete:")
    print(f"  Valid windows: {valid_count}/{len(dataset)}")
    print(f"  Skipped windows: {skipped_count}")

    # Save cache
    if args.output is None:
        os.makedirs("cache", exist_ok=True)
        output_path = f"cache/vo_cache_{args.sequence}.pt"
    else:
        output_path = args.output

    torch.save(cache, output_path)
    print(f"\nCache saved to: {output_path}")


if __name__ == "__main__":
    main()
