"""
Train GNN-based Bundle Adjustment using KITTI dataset.

This script uses the KITTI dataset for:
- Multi-frame feature extraction and matching using ORB
- Ground truth poses for building 3D point tracks
- VO model for initial pose estimation
- GNN-based optimization to refine poses

Architecture Overview:
1. KITTIFeatureDataset: Loads KITTI images, extracts ORB features, builds 2D->3D tracks via optical flow
2. GNNBAOptimizer: Graph neural network that alternates messages between camera and point nodes
3. Training loop: VO model provides initial poses, GNN refines them by minimizing reprojection error

Key Functions:
- triangulate_all_points(): DLT triangulation from 2D observations + camera poses -> 3D points
- build_heterogeneous_graph(): Creates PyTorch Geometric HeteroData (camera, point, observation edges)
- train_epoch(): Main training loop - forward pass through VO, triangulate, GNN optimization
- predict_full_sequence(): Full sequence prediction with overlapping window merging

Usage:
    python train_gnn_ba.py --checkpoint Exp51/checkpoint_best --sequence 03
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    poses_to_camera_features,
    denormalize_poses,
    euler_to_rotation_torch,
    project_points_torch,
    huber_loss,
    compute_reprojection_error,
    post_processing,
    recover_trajectory_and_poses,
    rotation_to_euler,
    euler_to_rotation,
    pose_6dof_to_matrix_torch,
    camera_features_to_poses,
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


def train_epoch(model, gnn_model, train_loader, optimizer, epoch, args, device):
    """
    Train GNN-BA for one epoch.

    Training pipeline for each batch (window):
    1. Get images -> run through VO model -> get initial relative poses
    2. Convert relative poses to absolute poses (accumulation)
    3. Triangulate 3D points from tracks using initial poses
    4. Build heterogeneous graph (camera nodes + point nodes + edges)
    5. GNN forward pass -> predict optimized pose
    8. Backprop Huber loss on reprojection error -> update GNN

    Note: VO model is frozen (eval mode, no gradient). Only GNN is trained.
    """
    model.eval()  # Freeze VO model
    gnn_model.train()  # Train GNN
    epoch_loss = 0
    num_batches = 0
    running_loss = 0

    # Copy normalization stats as numpy arrays (for triangulation functions)
    mean_angles_np = KITTI_MEAN_ANGLES.copy()
    std_angles_np = KITTI_STD_ANGLES.copy()
    mean_t_np = KITTI_MEAN_T.copy()
    std_t_np = KITTI_STD_T.copy()

    # Torch tensors for GPU computation
    mean_angles = torch.tensor(KITTI_MEAN_ANGLES, dtype=torch.float32, device=device)
    std_angles = torch.tensor(KITTI_STD_ANGLES, dtype=torch.float32, device=device)
    mean_t = torch.tensor(KITTI_MEAN_T, dtype=torch.float32, device=device)
    std_t = torch.tensor(KITTI_STD_T, dtype=torch.float32, device=device)

    # KITTI original resolution -> model input resolution
    orig_width, orig_height = 1241, 376
    target_width, target_height = 672, 224
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height

    with tqdm(train_loader, unit="batch", dynamic_ncols=True) as tepoch:
        for batch_data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            observations = batch_data["observations"][0]
            global_poses = batch_data["global_poses"][0]
            K = batch_data["K"]
            tracks = batch_data["tracks"][0]
            images = batch_data["images"][0]

            # Scale intrinsics to match image resolution used by model
            K_scaled = {
                "fx": K["fx"] * scale_x,
                "fy": K["fy"] * scale_y,
                "cx": K["cx"] * scale_x,
                "cy": K["cy"] * scale_y,
            }

            window_size = args["window_size"]

            # Preprocess images for VO model (resize, normalize)
            pil_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    img = img.squeeze(0).numpy()
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(img)
                pil_images.append(preprocess(pil_img))

            imgs_batch = (
                torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)
            )

            # Use original image dimensions from the first image (for graph normalization)
            img_height, img_width = images[0].shape[:2]

            # ---- Step 1: Get initial relative poses from VO model ----
            with torch.no_grad():
                vo_output = model(imgs_batch)  # Shape: (1, window_size-1, 6) - normalized relative pose
            
            # vo_output is normalized relative pose (6-DOF: euler_norm, t_norm)
            # For triangulation, we need absolute poses
            vo_output_np = vo_output.cpu().numpy()[0]  # (window_size-1, 6) normalized
            
            # Denormalize: actual = normalized * std + mean
            rel_poses_6dof = []
            for i in range(vo_output_np.shape[0]):
                euler = vo_output_np[i, :3] * std_angles_np + mean_angles_np  # in radians
                t = vo_output_np[i, 3:] * std_t_np + mean_t_np  # in meters
                rel_poses_6dof.append(np.concatenate([euler, t]))
            
            # Accumulate relative poses to get absolute poses (for triangulation)
            abs_poses_4x4 = [np.eye(4)]  # First frame is identity
            for rel_pose in rel_poses_6dof:
                R = euler_to_rotation(rel_pose[:3], seq='zyx')
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = rel_pose[3:]
                abs_pose = abs_poses_4x4[-1] @ T_rel
                abs_poses_4x4.append(abs_pose)
            
            # GNN input: accumulated absolute poses in 6-DOF (denormalized: euler in radians, t in meters)
            # GNN predicts delta in denormalized space (euler in radians, t in meters)
            # optimized_pose = vo_pose + delta_pose (all denormalized)
            abs_poses_6dof = []
            for pose in abs_poses_4x4:
                R = pose[:3, :3]
                t = pose[:3, 3]
                euler = rotation_to_euler(R, seq='zyx')
                abs_poses_6dof.append(np.concatenate([euler, t]))
            camera_feats_full = np.array(abs_poses_6dof)  # (window_size, 6) accumulated absolute poses
            
            # Prepare GT relative poses for loss computation (already in 6-DOF from dataset)
            gt_relative_poses = global_poses
            if isinstance(gt_relative_poses, torch.Tensor):
                gt_relative_poses = gt_relative_poses.cpu().numpy()
            
            # Skip windows with too few tracks
            if len(tracks) < 10:
                print("tracks < 10")
                continue

            # ---- Step 2: Triangulate 3D points from tracks (using VO absolute poses) ----
            points_3d, graph_obs = triangulate_all_points(tracks, abs_poses_4x4, K)

            if len(points_3d) < 10:
                continue

            # ---- Step 3: Build heterogeneous graph ----
            # Graph structure: window_size camera nodes (first=reference, rest=relative poses)
            data = build_heterogeneous_graph(
                window_size, points_3d, graph_obs, img_height, img_width
            )

            # GNN input: accumulated absolute poses (denormalized: euler in radians, t in meters)
            data["camera"].x = torch.tensor(camera_feats_full, dtype=torch.float32)

            data = data.to(device)

            optimizer.zero_grad()

            # ---- Step 4: GNN forward pass - predict delta pose ----
            # GNN with return_delta=True outputs correction delta to be added to input pose
            camera_delta, point_delta = gnn_model(data, return_delta=True)

            # ---- Step 5: Apply delta to get refined poses ----
            # GNN input is denormalized, GNN delta is in denormalized space
            # optimized_pose = vo_pose + delta_pose (all denormalized)
            camera_feats_tensor = torch.tensor(camera_feats_full, dtype=torch.float32, device=device)
            refined_poses_denorm = camera_feats_tensor + camera_delta  # (window_size, 6) denormalized
            point_params = data['point'].x + point_delta  # (N, 3) refined point positions

            # Poses are already denormalized (euler in radians, t in meters)
            euler = refined_poses_denorm[:, :3]
            t = refined_poses_denorm[:, 3:]

            # ---- Step 6: Convert refined absolute poses to 4x4 matrices (NO accumulation) ----
            # refined_poses_denorm already contains absolute poses (denormalized)
            # Just convert each to 4x4 matrix (matching inference)
            Rs = euler_to_rotation_torch(euler, seq="zyx")  # (window_size, 3, 3)

            pose_matrices_list = []
            for i in range(window_size):
                T = torch.eye(4).to(device)
                T[:3, :3] = Rs[i]
                T[:3, 3] = t[i]
                pose_matrices_list.append(T)

            pose_matrices = torch.stack(pose_matrices_list, dim=0)  # (window_size, 4, 4)

            # ---- Step 7: Compute reprojection error ----
            edge_index = data["camera", "observes", "point"].edge_index
            edge_attr = data["camera", "observes", "point"].edge_attr

            # Convert normalized edge coords back to pixel coords
            u_obs = edge_attr[:, 0] * img_width
            v_obs = edge_attr[:, 1] * img_height

            # Project 3D points using refined poses
            projected = project_points_torch(point_params, pose_matrices, K, device)

            # Get projected coords for each observation
            cam_indices = edge_index[0]
            pt_indices = edge_index[1]

            proj_x = projected[cam_indices, pt_indices, 0]
            proj_y = projected[cam_indices, pt_indices, 1]

            # Euclidean error in pixels
            errors = torch.sqrt((proj_x - u_obs) ** 2 + (proj_y - v_obs) ** 2)

            # Filter to valid observations (positive depth)
            # Invert poses to get world-to-camera for depth transform
            poses_w2c = torch.inverse(pose_matrices)
            points_h = torch.cat(
                [point_params, torch.ones(point_params.shape[0], 1, device=device)],
                dim=1,
            )
            points_cam = torch.matmul(poses_w2c[:, :3, :], points_h.T)
            depths = points_cam[:, 2, :]
            obs_depths = depths[cam_indices, pt_indices]
            valid_mask = obs_depths > 0.1

            valid_errors = errors[valid_mask]

            # ---- Step 7b: Compute pose loss on delta (correction) poses ----
            # GNN input is accumulated absolute poses (camera_feats_full)
            # GT is given as relative poses - accumulate to get GT absolute poses
            gt_rel = gt_relative_poses
            if isinstance(gt_rel, np.ndarray) and gt_rel.ndim == 3:
                gt_rel = gt_rel[0]
            
            gt_abs_poses_4x4 = [np.eye(4)]
            for i in range(len(gt_rel)):
                R = euler_to_rotation(gt_rel[i, :3], seq='zyx')
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = gt_rel[i, 3:]
                gt_abs_poses_4x4.append(gt_abs_poses_4x4[-1] @ T_rel)
            
            gt_abs_poses_6dof = []
            for pose in gt_abs_poses_4x4:
                R = pose[:3, :3]
                t = pose[:3, 3]
                euler = rotation_to_euler(R, seq='zyx')
                gt_abs_poses_6dof.append(np.concatenate([euler, t]))
            gt_abs_poses_6dof = np.array(gt_abs_poses_6dof)
            
            gt_delta = gt_abs_poses_6dof[1:] - camera_feats_full[1:]
            gt_delta_euler = torch.tensor(gt_delta[:, :3], dtype=torch.float32, device=device)
            gt_delta_t = torch.tensor(gt_delta[:, 3:], dtype=torch.float32, device=device)
            
            gnn_delta_euler = camera_delta[1:, :3]
            gnn_delta_t = camera_delta[1:, 3:]
            
            k = args.get("weighted_loss", 100.0)
            loss_angles = k * F.mse_loss(gnn_delta_euler, gt_delta_euler)
            loss_translation = k * F.mse_loss(gnn_delta_t, gt_delta_t)
            # print(loss_angles, loss_translation, valid_errors.mean())
            pose_loss = loss_angles + loss_translation

            # ---- Step 8: Backprop and update ----
            if valid_errors.numel() > 0:
                # Combined loss: reprojection loss + pose loss
                loss = huber_loss(valid_errors, delta=1.0).mean() * 0.05 + pose_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                running_loss += loss.item()

            # Update tqdm display
            tepoch.set_postfix(avg_loss=running_loss / max(num_batches, 1))

    return epoch_loss / max(num_batches, 1)


def validate(model, gnn_model, val_loader, args, device):
    """
    Validate GNN-BA on validation set.

    Computes reprojection error before and after GNN optimization
    to measure improvement.
    """
    model.eval()
    gnn_model.eval()
    total_error_initial = 0
    total_error_optimized = 0
    num_samples = 0

    mean_angles = KITTI_MEAN_ANGLES
    std_angles = KITTI_STD_ANGLES
    mean_t = KITTI_MEAN_T
    std_t = KITTI_STD_T

    # Resolution scaling
    orig_width, orig_height = 1241, 376
    target_width, target_height = 672, 224
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height

    with torch.no_grad():
        for batch_data in val_loader:
            observations = batch_data["observations"][0]
            global_poses = batch_data["global_poses"][0]
            K = batch_data["K"]
            tracks = batch_data["tracks"][0]
            images = batch_data["images"][0]

            window_size = args["window_size"]

            # Preprocess images
            pil_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    img = img.squeeze(0).numpy()
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(img)
                pil_images.append(preprocess(pil_img))

            imgs_batch = (
                torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)
            )

            # Original image dimensions for graph construction
            img_height, img_width = images[0].shape[:2]

            # Get VO poses
            vo_output = model(imgs_batch)
            vo_poses = denormalize_poses(
                vo_output, window_size, mean_angles, std_angles, mean_t, std_t
            )[0]

            # tracks from DataLoader is a flat list: [track1, track2, ...]
            if len(tracks) < 10:
                continue

            # Triangulate 3D points using original K
            points_3d, graph_obs = triangulate_all_points(tracks, vo_poses, K)

            if len(points_3d) < 10:
                print("points3d < 10")
                continue

            # Build graph
            data = build_heterogeneous_graph(
                window_size, points_3d, graph_obs, img_height, img_width
            )

            # Convert 4x4 absolute poses to 6-DOF (denormalized: euler in radians, t in meters)
            abs_poses_6dof = []
            for pose in vo_poses:
                if isinstance(pose, torch.Tensor):
                    pose = pose.cpu().numpy()
                R = pose[:3, :3]
                t = pose[:3, 3]
                euler = rotation_to_euler(R, seq='zyx')
                abs_poses_6dof.append(np.concatenate([euler, t]))
            abs_poses_6dof = np.array(abs_poses_6dof)

            # Use denormalized 6-DOF poses directly as GNN input
            camera_feats_tensor = torch.tensor(abs_poses_6dof, dtype=torch.float32)
            data["camera"].x = camera_feats_tensor
            data = data.to(device)

            # GNN predicts delta pose in denormalized space - add to original VO pose
            camera_delta, point_delta = gnn_model(data, return_delta=True)

            # Apply delta: optimized_pose = vo_pose + delta_pose (all denormalized)
            opt_camera_denorm = camera_feats_tensor.to(device) + camera_delta
            opt_points_np = (data['point'].x + point_delta).cpu().numpy()

            # Optimized poses are already in denormalized space (euler in radians, t in meters)
            opt_camera_np = opt_camera_denorm.cpu().numpy()
            euler = opt_camera_np[:, :3]
            t = opt_camera_np[:, 3:]

            opt_poses = []
            for i in range(window_size):
                R = euler_to_rotation(euler[i], seq="zyx")
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t[i]
                opt_poses.append(pose)

            # Compute reprojection errors (before vs after optimization) using original K
            initial_error = compute_reprojection_error(
                vo_poses, points_3d, graph_obs, K
            )
            optimized_error = compute_reprojection_error(
                opt_poses, opt_points_np, graph_obs, K
            )

            total_error_initial += initial_error
            total_error_optimized += optimized_error
            num_samples += 1

    return {
        "initial_error": total_error_initial / max(num_samples, 1),
        "optimized_error": total_error_optimized / max(num_samples, 1),
    }


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


def predict_full_sequence(vo_model, gnn_model, dataset, args, device):
    """
    Predict poses for the full KITTI sequence using VO + GNN optimization.

    This function generates a complete trajectory by:
    1. Running the VO model on all overlapping windows to get initial relative poses
    2. Post-processing to merge overlapping windows (averaging)
    3. Accumulating relative poses into absolute trajectory (VO baseline)
    4. Running GNN optimization on each window to refine poses
    5. Post-processing and accumulating GNN-optimized poses

    The GNN optimization is applied per-window then merged the same way as VO poses,
    ensuring consistency with the sliding window approach used during training.
    """
    gnn_model.eval()
    vo_model.eval()

    window_size = args["window_size"]
    overlap = args["overlap"]

    # Resolution scaling
    orig_width, orig_height = 1241, 376
    target_width, target_height = 672, 224
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height

    # ---- Step 1: Collect raw VO outputs for all windows ----
    raw_vo_outputs = []

    print("Step 1: Collecting VO model outputs...")
    for idx in tqdm(range(len(dataset)), desc="Collecting outputs"):
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

        # Reshape to (1, window_size-1, 6) for consistency
        vo_output = vo_output.reshape(1, window_size - 1, 6)
        raw_vo_outputs.append(vo_output.cpu().detach().numpy())

    # Stack: shape (num_windows, window_size-1, 6)
    raw_vo_outputs = np.concatenate(raw_vo_outputs, axis=0)
    print(f"Collected {len(raw_vo_outputs)} windows of raw outputs")

    # ---- Step 2: Post-process VO outputs (merge overlapping windows) ----
    print("Step 2: Applying post-processing...")
    processed_poses = post_processing(raw_vo_outputs, window_size, overlap)
    print(f"Processed poses shape: {processed_poses.shape}")

    # ---- Step 3: Accumulate to get absolute VO poses ----
    print("Step 3: Recovering trajectory...")
    vo_poses, vo_trajectory = recover_trajectory_and_poses(processed_poses, norm=True)
    print(f"Recovered {len(vo_poses)} absolute poses")

    # ---- Step 4: GNN optimization on each window ----
    print("Step 4: Running GNN optimization...")

    opt_relative_poses_list = []

    # Debug counters
    num_triangulation_failed = 0
    num_track_count_low = 0
    num_success = 0
    total_delta_norm = 0.0

    for idx in tqdm(range(len(dataset)), desc="GNN optimization"):
        sample = dataset[idx]

        pil_images = []
        for img in sample["images"]:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img)
            pil_images.append(preprocess(pil_img))

        imgs_batch = torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)

        K = sample["K"]
        # Use original K (no scaling) for triangulation
        tracks = sample["tracks"]
        window_indices = sample["window_indices"]

        # Reformat tracks to (rel_frame_idx, u, v) format
        rel_tracks = []
        for track in tracks:
            if len(track) >= 2:
                rel_tracks.append([(obs[0], obs[1], obs[2]) for obs in track])
        tracks = rel_tracks

        # Get absolute poses for this window from the global VO trajectory
        window_start = idx * (window_size - overlap)
        # Handle boundary cases
        if window_start >= len(vo_poses):
            window_abs_poses = [vo_poses[-1]] * window_size
        elif window_start + window_size > len(vo_poses):
            window_abs_poses = vo_poses[window_start:]
            while len(window_abs_poses) < window_size:
                window_abs_poses.append(vo_poses[-1])
        else:
            window_abs_poses = vo_poses[window_start : window_start + window_size]

        if len(tracks) >= 10:
            # Triangulate 3D points using original K
            points_3d, graph_obs = triangulate_all_points(tracks, window_abs_poses, K)

            if len(points_3d) >= 10:
                # Original image dimensions for graph normalization
                img_height, img_width = sample["images"][0].shape[:2]
                data = build_heterogeneous_graph(
                    window_size, points_3d, graph_obs, img_height, img_width
                )

                # Use accumulated absolute poses as GNN input (like validate)
                # Convert 4x4 absolute poses to 6-DOF (denormalized: euler in radians, t in meters)
                abs_poses_6dof = []
                for pose in window_abs_poses:
                    if isinstance(pose, torch.Tensor):
                        pose = pose.cpu().numpy()
                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    euler = rotation_to_euler(R, seq='zyx')
                    abs_poses_6dof.append(np.concatenate([euler, t]))
                abs_poses_6dof = np.array(abs_poses_6dof)

                camera_feats_tensor = torch.tensor(abs_poses_6dof, dtype=torch.float32)
                data["camera"].x = camera_feats_tensor
                data = data.to(device)

                with torch.no_grad():
                    camera_delta, point_delta = gnn_model(data, return_delta=True)

                # Apply delta: optimized_pose = vo_pose + delta_pose (all denormalized)
                opt_camera_denorm = camera_feats_tensor.to(device) + camera_delta

                # Track delta magnitude for debugging
                delta_norm = torch.norm(camera_delta).item()
                total_delta_norm += delta_norm

                num_success += 1

                # Optimized poses are already in denormalized space (euler in radians, t in meters)
                opt_camera_np = opt_camera_denorm.cpu().numpy()
                euler = opt_camera_np[:, :3]
                t = opt_camera_np[:, 3:]

                # Convert to 4x4 pose matrices
                opt_poses = []
                for i in range(window_size):
                    R = euler_to_rotation(euler[i], seq="zyx")
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = t[i]
                    opt_poses.append(pose)

                # Convert absolute optimized poses to relative poses (for post-processing which expects normalized)
                for i in range(len(opt_poses) - 1):
                    T_rel = np.linalg.inv(opt_poses[i]) @ opt_poses[i + 1]
                    R_rel = T_rel[:3, :3]
                    t_rel = T_rel[:3, 3]
                    euler_rel = rotation_to_euler(R_rel, seq="zyx")
                    # Normalize relative pose for post-processing
                    euler_norm = (euler_rel - KITTI_MEAN_ANGLES) / KITTI_STD_ANGLES
                    t_norm = (t_rel - KITTI_MEAN_T) / KITTI_STD_T
                    opt_relative_poses_list.append(np.concatenate([euler_norm, t_norm]))
            else:
                # Fallback: use VO output if triangulation fails
                num_triangulation_failed += 1
                for i in range(window_size - 1):
                    opt_relative_poses_list.append(raw_vo_outputs[idx, i, :])
        else:
            # Fallback: use VO output if too few tracks
            num_track_count_low += 1
            for i in range(window_size - 1):
                opt_relative_poses_list.append(raw_vo_outputs[idx, i, :])

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} windows")

    # Print debug statistics
    print(f"\nGNN Optimization Debug Stats:")
    print(
        f"  Success: {num_success}/{len(dataset)} ({100*num_success/len(dataset):.1f}%)"
    )
    print(f"  Triangulation failed: {num_triangulation_failed}")
    print(f"  Track count low: {num_track_count_low}")
    print(
        f"  Avg delta norm (when successful): {total_delta_norm/max(num_success,1):.6f}"
    )

    # ---- Step 5: Merge optimized poses ----
    print("Step 5: Merging optimized poses...")
    opt_relative_poses = np.array(opt_relative_poses_list).reshape(
        -1, window_size - 1, 6
    )
    print(f"Optimized relative poses shape: {opt_relative_poses.shape}")

    # Post-process and accumulate
    processed_opt_poses = post_processing(opt_relative_poses, window_size, overlap)
    opt_poses_full, opt_trajectory = recover_trajectory_and_poses(
        processed_opt_poses, norm=True
    )
    print(f"Recovered {len(opt_poses_full)} optimized absolute poses")

    return vo_poses, opt_poses_full


def main():
    parser = argparse.ArgumentParser(description="Train GNN-based BA")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Exp51/checkpoint_best.pth",
        help="Path to VO model checkpoint",
    )
    parser.add_argument("--sequence", type=str, default="03", help="Sequence number")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--window_size", type=int, default=3, help="Window size")
    parser.add_argument(
        "--overlap", type=int, default=2, help="Overlap between windows"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sequences_jpg",
        help="Path to KITTI sequences",
    )
    parser.add_argument(
        "--gt_path", type=str, default="data/poses", help="Path to KITTI poses"
    )
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument(
        "--save_dir", type=str, default="gnn_ba_output", help="Output directory"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

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

    vo_model, model_args = build_model(model_args, model_params)
    vo_model.eval()
    print("VO model loaded successfully")

    print(f"\nLoading KITTI dataset for sequence {args.sequence}...")
    full_dataset = KITTIFeatureDataset(
        data_path=args.data_path,
        gt_path=args.gt_path,
        sequence=args.sequence,
        window_size=args.window_size,
        overlap=args.overlap,
        max_points=200,
    )

    print(f"Dataset size: {len(full_dataset)} windows")

    gt_poses = full_dataset.gt_poses

    num_val = int(len(full_dataset) * args.val_split)
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     full_dataset, [len(full_dataset) - num_val, num_val]
    # )
    train_dataset, val_dataset = full_dataset, full_dataset

    def collate_fn(batch):
        """Custom collate function to handle variable-size observations and tracks."""
        collated = {}
        for key in batch[0].keys():
            if key in ['observations', 'tracks', 'images', 'keypoints', 'abs_poses']:
                collated[key] = [sample[key] for sample in batch]
            elif key == 'window_indices':
                collated[key] = [sample[key] for sample in batch]
            elif key == 'K':
                collated[key] = batch[0][key]
            elif isinstance(batch[0][key], np.ndarray):
                collated[key] = torch.stack([torch.tensor(sample[key]) for sample in batch], dim=0)
            else:
                collated[key] = [sample[key] for sample in batch]
        return collated

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=collate_fn
    )

    gnn_model = GNNBAOptimizer(hidden_dim=args.hidden_dim, num_layers=3).to(device)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)

    print(f"\nTraining for {args.num_epochs} epochs...")
    best_val_error = float("inf")

    train_args = {
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "overlap": args.overlap,
    }

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(
            vo_model, gnn_model, train_loader, optimizer, epoch, train_args, device
        )

        if epoch % 10 == 0:
            val_metrics = validate(vo_model, gnn_model, val_loader, train_args, device)
            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f'Val Initial Error = {val_metrics["initial_error"]:.2f}px, '
                f'Val Optimized Error = {val_metrics["optimized_error"]:.2f}px'
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
                    os.path.join(args.save_dir, "gnn_ba_best.pth"),
                )
                print(f"Saved best model with error: {best_val_error:.2f}px")

    print("\nTraining completed! Running final evaluation...")

    final_metrics = validate(vo_model, gnn_model, val_loader, train_args, device)
    print(f'Final Initial Error: {final_metrics["initial_error"]:.2f}px')
    print(f'Final Optimized Error: {final_metrics["optimized_error"]:.2f}px')

    print("\nGenerating trajectory comparison...")

    test_sample = full_dataset[0]
    window_indices = test_sample["window_indices"]

    pil_images = []
    for img in test_sample["images"]:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img)
        pil_images.append(preprocess(pil_img))

    sample_images = torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        vo_output = vo_model(sample_images)

    vo_poses = denormalize_poses(
        vo_output,
        args.window_size,
        KITTI_MEAN_ANGLES,
        KITTI_STD_ANGLES,
        KITTI_MEAN_T,
        KITTI_STD_T,
    )[0]

    tracks = test_sample["tracks"]
    K = test_sample["K"]

    # Use original K and actual image dimensions
    img_height, img_width = test_sample["images"][0].shape[:2]

    points_3d, graph_obs = triangulate_all_points(tracks, vo_poses, K)
    data = build_heterogeneous_graph(
        args.window_size, points_3d, graph_obs, img_height, img_width
    )

    # Convert 4x4 absolute poses to 6-DOF (denormalized: euler in radians, t in meters)
    # MATCHES validate() and train_epoch() - use DENORMALIZED poses as GNN input
    abs_poses_6dof = []
    for pose in vo_poses:
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        R = pose[:3, :3]
        t = pose[:3, 3]
        euler = rotation_to_euler(R, seq='zyx')
        abs_poses_6dof.append(np.concatenate([euler, t]))
    abs_poses_6dof = np.array(abs_poses_6dof)

    camera_feats_tensor = torch.tensor(abs_poses_6dof, dtype=torch.float32)
    data["camera"].x = camera_feats_tensor
    data = data.to(device)

    gnn_model.eval()
    with torch.no_grad():
        camera_delta, point_delta = gnn_model(data, return_delta=True)

    # Apply delta: optimized_pose = vo_pose + delta_pose (all denormalized)
    opt_camera_denorm = camera_feats_tensor.to(device) + camera_delta
    opt_points_np = (data['point'].x + point_delta).cpu().numpy()

    # Optimized poses are already in denormalized space (euler in radians, t in meters)
    opt_camera_np = opt_camera_denorm.cpu().numpy()
    euler = opt_camera_np[:, :3]
    t = opt_camera_np[:, 3:]

    opt_poses = []
    for i in range(args.window_size):
        R = euler_to_rotation(euler[i], seq="zyx")
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t[i]
        opt_poses.append(pose)

    gt_window_poses = test_sample["abs_poses"]  # Use absolute poses from dataset

    plot_trajectories(
        gt_window_poses,
        vo_poses,
        opt_poses,
        os.path.join(args.save_dir, "trajectory_comparison.png"),
    )

    output_poses_path = os.path.join(args.save_dir, f"poses_test.txt")
    with open(output_poses_path, "w") as f:
        for pose in opt_poses:
            row = pose.flatten()[:12]
            f.write(" ".join([str(v) for v in row]) + "\n")
    print(f"Poses saved to: {output_poses_path}")

    print("\nGenerating full sequence trajectory...")

    full_sequence_vo_poses, full_sequence_opt_poses = predict_full_sequence(
        vo_model, gnn_model, full_dataset, train_args, device
    )

    print(f"Full sequence: {len(full_sequence_opt_poses)} poses predicted")

    seq_vo_path = os.path.join(args.save_dir, f"poses_{args.sequence}_vo.txt")
    seq_opt_path = os.path.join(args.save_dir, f"poses_{args.sequence}_opt.txt")

    with open(seq_vo_path, "w") as f:
        for pose in full_sequence_vo_poses:
            row = pose.flatten()[:12]
            f.write(" ".join([str(v) for v in row]) + "\n")

    with open(seq_opt_path, "w") as f:
        for pose in full_sequence_opt_poses:
            row = pose.flatten()[:12]
            f.write(" ".join([str(v) for v in row]) + "\n")

    print(f"VO poses saved to: {seq_vo_path}")
    print(f"Optimized poses saved to: {seq_opt_path}")

    # Convert gt_poses from 3x4 to 4x4 for plotting
    gt_poses_4x4 = []
    for pose_3x4 in gt_poses[: len(full_sequence_opt_poses)]:
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :4] = pose_3x4
        gt_poses_4x4.append(pose_4x4)

    plot_trajectories(
        gt_poses_4x4,
        full_sequence_vo_poses,
        full_sequence_opt_poses,
        os.path.join(args.save_dir, "trajectory_comparison.png"),
    )


if __name__ == "__main__":
    main()
