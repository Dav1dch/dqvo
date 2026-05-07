"""
GNN-based Bundle Adjustment Single-Window Test.

This script tests GNN-BA on a single window of KITTI sequences to debug
reprojection error changes. It follows the data flow and conventions from
train_gnn_ba.py and uses utility functions from utils/gnn_ba.py.

Usage:
    python gnn_ba_test.py --kitti_root data --sequence 00 --num_epochs 100
"""

import argparse
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from utils.gnn_ba import (
    triangulate_all_points,
    build_heterogeneous_graph,
    project_points_torch,
    compute_reprojection_error,
    huber_loss,
)
from utils.dq import (
    matrix_to_dq_np,
    dq_to_matrix_np,
    dq_identity_torch,
    dq_identity_np,
    dq_mult_torch,
    dq_mult_np,
    dq_conj_torch,
    dq_conj_np,
    dq_transform_point_torch,
    relative_poses_to_absolute_dq_torch,
)
from timesformer.models.gnn_ba import GNNBAOptimizer


# =============================================================================
# KITTI Data Loading
# =============================================================================

def load_kitti_intrinsics(calib_path: str) -> dict:
    """Load camera intrinsics from KITTI calib.txt file."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    p0_line = lines[0].strip().split()
    fx = float(p0_line[1])
    cx = float(p0_line[3])
    fy = float(p0_line[6])
    cy = float(p0_line[7])
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}


def load_kitti_poses(poses_path: str, num_frames: int = None) -> list:
    """Load ground truth poses from KITTI poses file."""
    poses = []
    with open(poses_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if num_frames and i >= num_frames:
            break
        values = [float(x) for x in line.strip().split()]
        pose = np.array(values).reshape(3, 4)
        poses.append(pose)
    return poses


def load_kitti_images(image_dir: str, num_frames: int = 5) -> list:
    """Load KITTI images from directory."""
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))[:num_frames]
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


# =============================================================================
# Feature Extraction and Matching
# =============================================================================

def extract_and_match_features(images: list, max_points: int = 500) -> dict:
    """Extract ORB features and match between consecutive frames."""
    orb = cv2.ORB_create(nfeatures=max_points)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    keypoints = []
    descriptors = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
    matches_per_pair = []
    for i in range(len(images) - 1):
        if descriptors[i] is None or descriptors[i+1] is None:
            matches_per_pair.append(np.array([]))
            continue
        knn_matches = bf.knnMatch(descriptors[i], descriptors[i+1], k=2)
        good_matches = []
        for m_n in knn_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        matches_per_pair.append(good_matches)
    return {
        'keypoints': keypoints,
        'matches': matches_per_pair,
        'num_frames': len(images)
    }


def build_track_matches(images: list, match_data: dict) -> list:
    """Build feature tracks across multiple frames."""
    keypoints = match_data['keypoints']
    matches = match_data['matches']
    num_frames = match_data['num_frames']
    tracks = []
    for kp_idx, kp in enumerate(keypoints[0]):
        tracks.append([(0, kp_idx, kp.pt[0], kp.pt[1])])
    for frame_idx in range(1, num_frames):
        frame_matches = matches[frame_idx - 1]
        prev_to_curr = {}
        for m in frame_matches:
            prev_to_curr[m.queryIdx] = m.trainIdx
        new_tracks = []
        for track in tracks:
            last_frame, last_kp_idx, last_u, last_v = track[-1]
            if last_kp_idx in prev_to_curr:
                curr_kp_idx = prev_to_curr[last_kp_idx]
                curr_kp = keypoints[frame_idx][curr_kp_idx]
                track.append((frame_idx, curr_kp_idx, curr_kp.pt[0], curr_kp.pt[1]))
                new_tracks.append(track)
            else:
                if len(track) >= 2:
                    new_tracks.append(track)
        tracks = new_tracks
    valid_tracks = [t for t in tracks if len(t) >= 2]
    return valid_tracks


# =============================================================================
# Pose Utilities (used by add_noise_to_poses)
# =============================================================================

def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix using Rodrigues formula."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


def add_noise_to_poses(poses: list, trans_noise: float = 0.2,
                       rot_noise: float = 0.05) -> list:
    """Add noise to ground truth poses to simulate VO initialization. Returns DQ list."""
    noisy_dqs = []
    for pose in poses:
        t_noise = np.random.randn(3) * trans_noise
        new_t = pose[:3, 3] + t_noise
        rot_noise_vec = np.random.randn(3) * rot_noise
        R_noise = axis_angle_to_rotation_matrix(rot_noise_vec)
        new_R = R_noise @ pose[:3, :3]
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_R
        new_pose[:3, 3] = new_t
        noisy_dqs.append(matrix_to_dq_np(new_pose))
    return noisy_dqs


# =============================================================================
# Main Test Function
# =============================================================================

def run_gnn_ba(kitti_root: str, sequence: str = '00', num_frames: int = 5,
               num_epochs: int = 100, hidden_dim: int = 32, learning_rate: float = 0.001,
               save_plot: str = None, reproj_weight: float = 1.0, pose_weight: float = 0.01):
    """
    Single-window GNN-BA test.
    """
    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print(f"Loading KITTI sequence {sequence}, frames 0-{num_frames-1}.")
    seq_dir = os.path.join(kitti_root, 'sequences_jpg', sequence)
    image_dir = os.path.join(seq_dir, 'image_0')
    calib_path = os.path.join(seq_dir, 'calib.txt')
    poses_path = os.path.join(kitti_root, 'poses', f'{sequence}.txt')
    
    for p, name in [(image_dir, 'image_dir'), (calib_path, 'calib'), (poses_path, 'poses')]:
        if not os.path.exists(p):
            print(f"Error: {name} not found: {p}")
            return
    
    images = load_kitti_images(image_dir, num_frames)
    if len(images) < 2:
        print("Error: Not enough images loaded")
        return
    
    K = load_kitti_intrinsics(calib_path)
    gt_poses = load_kitti_poses(poses_path, num_frames)
    img_height, img_width = images[0].shape
    
    print(f"Loaded {len(images)} images. Intrinsics: fx={K['fx']:.1f}, fy={K['fy']:.1f}, cx={K['cx']:.1f}, cy={K['cy']:.1f}")
    
    # Feature extraction and tracks
    print("\nExtracting ORB features and building tracks...")
    match_data = extract_and_match_features(images, max_points=500)
    tracks = build_track_matches(images, match_data)
    print(f"Built {len(tracks)} tracks across {num_frames} frames.")
    
    # Create noisy initial poses (DQ)
    print("\nAdding noise to ground truth poses...")
    initial_dqs = add_noise_to_poses(gt_poses, trans_noise=0.2, rot_noise=0.05)
    initial_poses_4x4 = [dq_to_matrix_np(dq) for dq in initial_dqs]
    
    # Triangulate 3D points using initial poses (DQ)
    print("Triangulating 3D points...")
    points_3d, observations, _ = triangulate_all_points(tracks, initial_dqs, K)
    if len(points_3d) < 10:
        print("Error: Not enough 3D points after triangulation")
        return
    print(f"Triangulated {len(points_3d)} points with {len(observations)} observations.")
    
    # Normalize points for GNN input
    if not torch.is_tensor(points_3d):
        points_3d = torch.tensor(points_3d, dtype=torch.float32)
    pt_mean = points_3d.mean(dim=0)
    pt_std = points_3d.std(dim=0).clamp_min(1e-6)
    points_3d_norm = (points_3d - pt_mean) / pt_std

    # Build heterogeneous graph (only observation edges)
    data = build_heterogeneous_graph(num_frames, points_3d_norm, observations, img_height, img_width)
    
    # Set camera node features: DQ absolute poses
    data['camera'].x = torch.tensor(np.array(initial_dqs), dtype=torch.float32)

    # Compute relative DQ for c2c edges: d_rel = conj(d_i) * d_{i+1}
    rel_dqs_np = []
    for i in range(num_frames - 1):
        d_inv = dq_conj_np(initial_dqs[i])
        rel = dq_mult_np(d_inv, initial_dqs[i+1])
        rel_dqs_np.append(rel)

    c2c_index = torch.stack([torch.arange(0, num_frames - 1), torch.arange(1, num_frames)], dim=0)
    data['camera', 'temporal', 'camera'].edge_index = c2c_index
    data['camera', 'temporal', 'camera'].edge_attr = torch.tensor(np.array(rel_dqs_np), dtype=torch.float32)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    points_3d = points_3d.to(device)
    
    # Compute initial reprojection error
    initial_error = compute_reprojection_error(
        initial_dqs, points_3d.cpu().numpy(), observations, K, img_height, img_width
    )
    print(f"\nInitial reprojection error (normalized): {initial_error:.4f}")
    
    # GNN model
    model = GNNBAOptimizer(hidden_dim=hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Monitoring
    error_history = []
    delta_norm_history = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        pred_rel_dq, point_refined = model(data, output_mode="relative")
        
        # Compose absolute DQ
        first_dq = dq_identity_torch((), device)
        abs_dqs = relative_poses_to_absolute_dq_torch(pred_rel_dq, first_dq, device)
        
        # Reprojection loss
        proj = project_points_torch(points_3d, abs_dqs, K, device)
        edge_index = data['camera', 'observes', 'point'].edge_index
        edge_attr = data['camera', 'observes', 'point'].edge_attr
        u_obs = edge_attr[:, 0] * img_width
        v_obs = edge_attr[:, 1] * img_height
        
        proj_x = proj[edge_index[0], edge_index[1], 0]
        proj_y = proj[edge_index[0], edge_index[1], 1]
        errors = torch.sqrt((proj_x - u_obs)**2 + (proj_y - v_obs)**2)
        
        # Depth validity mask
        dqs_w2c = dq_conj_torch(abs_dqs)
        depths = []
        for ci, pi in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            X_cam = dq_transform_point_torch(dqs_w2c[ci:ci+1], points_3d[pi:pi+1])
            depths.append(X_cam[0, 2])
        obs_depths = torch.tensor(depths, device=device)
        valid_mask = obs_depths > 0.1
        valid_errors = errors[valid_mask]
        
        if valid_errors.numel() > 0:
            reproj_loss = huber_loss(valid_errors, delta=5.0).mean()
        else:
            reproj_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Pose regularization (encourage small corrections)
        pose_reg = torch.norm(pred_rel_dq)**2
        
        loss = reproj_loss * reproj_weight + pose_reg * pose_weight
        
        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss at epoch {epoch}")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logging
        if epoch % 10 == 0:
            with torch.no_grad():
                abs_dq_np = abs_dqs.detach().cpu()
                abs_np = []
                for dq_t in abs_dq_np:
                    T = dq_to_matrix_np(dq_t.numpy())
                    abs_np.append(T)
                current_error = compute_reprojection_error(
                    abs_np, points_3d.cpu().numpy(), observations, K, img_height, img_width
                )
                delta_norm = torch.norm(pred_rel_dq).item()
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f}, RMSE = {current_error:.4f}, delta_norm = {delta_norm:.6f}")
            error_history.append(current_error)
            delta_norm_history.append(delta_norm)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_rel_final, _ = model(data, output_mode="relative")
        abs_dq_final = relative_poses_to_absolute_dq_torch(pred_rel_final, dq_identity_torch((), device), device)
        abs_dq_final_np = abs_dq_final.detach().cpu()
        final_abs_np = [dq_to_matrix_np(dq_t.numpy()) for dq_t in abs_dq_final_np]
        final_error = compute_reprojection_error(
            final_abs_np, points_3d.cpu().numpy(), observations, K, img_height, img_width
        )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Initial reprojection error: {initial_error:.4f}")
    print(f"Final reprojection error:   {final_error:.4f}")
    if initial_error > 0:
        reduction = (initial_error - final_error) / initial_error * 100
        print(f"Reduction: {reduction:.1f}%")
    
    # Plot training curve
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 4))
        epochs = list(range(0, len(error_history)*10, 10))
        ax1.plot(epochs, error_history, 'b-o', label='RMSE')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE (norm)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(epochs, delta_norm_history, 'r--', alpha=0.7, label='||delta||')
        ax2.set_ylabel('||delta||', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title('GNN-BA Single Window Training')
        fig.tight_layout()
        if save_plot:
            plt.savefig(save_plot, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_plot}")
        else:
            plt.savefig('gnn_ba_test_debug.png', dpi=150, bbox_inches='tight')
            print(f"Plot saved to: gnn_ba_test_debug.png")
        plt.close()
    except ImportError:
        print("\nMatplotlib not available, skipping plot.")
    
    print("\nTest completed!")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN-BA single window test')
    parser.add_argument('--kitti_root', type=str, default='data',
                        help='Path to KITTI root (containing sequences_jpg/ and poses/)')
    parser.add_argument('--sequence', type=str, default='00',
                        help='Sequence number')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames (window size)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='GNN hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--reproj_weight', type=float, default=1.0,
                        help='Weight for reprojection loss')
    parser.add_argument('--pose_weight', type=float, default=0.01,
                        help='Weight for pose regularization')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Path to save training curve plot')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of GNN layers')
    
    args = parser.parse_args()
    
    run_gnn_ba(
        kitti_root=args.kitti_root,
        sequence=args.sequence,
        num_frames=args.num_frames,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        save_plot=args.save_plot,
        reproj_weight=args.reproj_weight,
        pose_weight=args.pose_weight,
    )
