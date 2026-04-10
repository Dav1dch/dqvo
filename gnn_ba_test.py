"""
GNN-based Bundle Adjustment Test Script using KITTI Dataset

This script implements a minimal test for GNN-based local Bundle Adjustment
optimization using the KITTI Odometry dataset.

Usage:
    python gnn_ba_test.py --kitti_root /path/to/kitti/odometry/dataset --sequence 00

Requirements:
    pip install torch torch_geometric numpy opencv-python scipy
"""

import argparse
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import HeteroData
from scipy.spatial.transform import Rotation


# =============================================================================
# KITTI Data Loading Functions
# =============================================================================

def load_kitti_intrinsics(calib_path: str) -> dict:
    """
    Load camera intrinsics from KITTI calib.txt file.
    
    Args:
        calib_path: Path to calib.txt file
        
    Returns:
        Dictionary with fx, fy, cx, cy
    """
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    # P0 is the left gray camera (or left RGB)
    # Format: P0: fx 0 cx 0 0 fy cy 0 0 0 1
    p0_line = lines[0].strip().split()
    fx = float(p0_line[1])
    cx = float(p0_line[3])
    fy = float(p0_line[6])
    cy = float(p0_line[7])
    
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}


def load_kitti_poses(poses_path: str, num_frames: int = None) -> list:
    """
    Load ground truth poses from KITTI poses file.
    
    Args:
        poses_path: Path to poses/.txt file
        num_frames: Number of frames to load (None for all)
        
    Returns:
        List of 3x4 transformation matrices
    """
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
    """
    Load KITTI images from directory.
    
    Args:
        image_dir: Path to image directory (e.g., sequence/00/image_0/)
        num_frames: Number of frames to load
        
    Returns:
        List of grayscale images
    """
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
    """
    Extract ORB features and match between consecutive frames.
    
    Args:
        images: List of grayscale images
        max_points: Maximum number of features per frame
        
    Returns:
        Dictionary with matches info
    """
    orb = cv2.ORB_create(nfeatures=max_points)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Store keypoints and descriptors for each image
    keypoints = []
    descriptors = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
    
    # Match between consecutive frames
    matches_per_pair = []
    for i in range(len(images) - 1):
        if descriptors[i] is None or descriptors[i + 1] is None:
            matches_per_pair.append(np.array([]))
            continue
            
        knn_matches = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)
        
        # Apply ratio test
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
    """
    Build feature tracks across multiple frames.
    
    Args:
        images: List of images
        match_data: Dictionary with keypoints and matches
        
    Returns:
        List of tracks, each track is a list of (frame_idx, point_idx, x, y)
    """
    keypoints = match_data['keypoints']
    matches = match_data['matches']
    num_frames = match_data['num_frames']
    
    # Initialize tracks with first frame features
    tracks = []  # List of dicts: {'frame': frame_idx, 'pt': (u, v), 'kp_idx': kp_idx}
    
    # First frame: add all keypoints as new tracks
    for kp_idx, kp in enumerate(keypoints[0]):
        tracks.append([(0, kp_idx, kp.pt[0], kp.pt[1])])
    
    # Process subsequent frames
    for frame_idx in range(1, num_frames):
        # Get matches from previous frame to current frame
        frame_matches = matches[frame_idx - 1]
        
        # Create a map from previous kp_idx to current kp_idx
        prev_to_curr = {}
        for m in frame_matches:
            prev_to_curr[m.queryIdx] = m.trainIdx
        
        # Update tracks
        new_tracks = []
        for track in tracks:
            last_frame, last_kp_idx, last_u, last_v = track[-1]
            
            if last_kp_idx in prev_to_curr:
                # This track continues
                curr_kp_idx = prev_to_curr[last_kp_idx]
                curr_kp = keypoints[frame_idx][curr_kp_idx]
                track.append((frame_idx, curr_kp_idx, curr_kp.pt[0], curr_kp.pt[1]))
                new_tracks.append(track)
            else:
                # Track lost - keep it if it has at least 2 observations
                if len(track) >= 2:
                    new_tracks.append(track)
        
        tracks = new_tracks
    
    # Filter tracks with at least 2 observations
    valid_tracks = [t for t in tracks if len(t) >= 2]
    
    return valid_tracks


# =============================================================================
# Triangulation Functions
# =============================================================================

def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix using Rodrigues formula.
    
    Args:
        axis_angle: 3D axis-angle vector (angle * axis)
        
    Returns:
        3x3 rotation matrix
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.eye(3)
    
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        3D axis-angle vector
    """
    rot = Rotation.from_matrix(R)
    angle_axis = rot.as_rotvec()
    return angle_axis


def triangulate_dlt(points_2d: list, poses: list, K: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points using Direct Linear Transform (DLT).
    
    Args:
        points_2d: List of (u, v) pixel coordinates
        poses: List of 3x4 camera pose matrices
        K: 3x3 camera intrinsic matrix
        
    Returns:
        3D point in homogeneous coordinates (4D)
    """
    A = []
    for (u, v), pose in zip(points_2d, poses):
        # Convert to normalized coordinates
        pt = np.array([u, v, 1.0])
        pt_norm = np.linalg.inv(K) @ pt
        
        # Build the projection matrix
        P = pose  # 3x4
        
        # DLT equation: cross product of projection and point gives 0
        A.append(pt_norm[0] * P[2, :] - P[0, :])
        A.append(pt_norm[1] * P[2, :] - P[1, :])
    
    A = np.array(A)
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    return X / X[3]  # Normalize


def triangulate_all_points(tracks: list, poses: list, K: dict, 
                          min_depth: float = 0.1) -> tuple:
    """
    Triangulate all 3D points from feature tracks.
    
    Args:
        tracks: List of feature tracks
        poses: List of camera poses (3x4 matrices)
        K: Camera intrinsics dictionary
        min_depth: Minimum depth threshold
        
    Returns:
        Tuple of (points_3d, observations)
            - points_3d: Nx3 array of 3D points
            - observations: List of (point_idx, frame_idx, u, v) observations
    """
    # Build intrinsic matrix
    K_matrix = np.array([[K['fx'], 0, K['cx']],
                         [0, K['fy'], K['cy']],
                         [0, 0, 1]])
    
    points_3d = []
    observations = []
    
    for track_idx, track in enumerate(tracks):
        if len(track) < 2:
            continue
        
        # Get 2D points and poses for this track
        pts_2d = [(t[2], t[3]) for t in track]
        frame_indices = [t[0] for t in track]
        track_poses = [poses[i] for i in frame_indices]
        
        # Triangulate
        try:
            X = triangulate_dlt(pts_2d, track_poses, K_matrix)
        except:
            continue
        
        # Check depth
        X_3d = X[:3] / X[3]
        
        # Convert to camera frame to check depth
        valid = True
        for pose in track_poses:
            cam_coords = pose @ X
            if cam_coords[2] < min_depth:
                valid = False
                break
        
        if valid and X_3d[2] > 0:
            points_3d.append(X_3d)
            
            # Add observations
            for t in track:
                frame_idx, kp_idx, u, v = t
                observations.append((len(points_3d) - 1, frame_idx, u, v))
    
    return np.array(points_3d), observations


# =============================================================================
# Graph Construction
# =============================================================================

def build_heterogeneous_graph(num_cameras: int, points_3d: np.ndarray, 
                              observations: list, K: dict, 
                              img_width: int = 1241, img_height: int = 376) -> HeteroData:
    """
    Build heterogeneous graph for GNN-based BA.
    
    Nodes:
        - camera nodes: represent camera poses (parameterized by axis-angle + translation)
        - point nodes: represent 3D points
        
    Edges:
        - observation edges: connect camera to point
    
    Args:
        num_cameras: Number of camera nodes
        points_3d: Nx3 array of 3D points
        observations: List of (point_idx, frame_idx, u, v) observations
        K: Camera intrinsics
        img_width: Image width for normalization
        img_height: Image height for normalization
        
    Returns:
        HeteroData object
    """
    data = HeteroData()
    
    # Normalization constants
    fx, fy = K['fx'], K['fy']
    cx, cy = K['cx'], K['cy']
    
    # Camera node features: axis-angle (3) + translation (3) = 6
    # We'll optimize over the 6 DoF
    camera_feats = torch.zeros(num_cameras, 6)
    data['camera'].x = camera_feats
    
    # Point node features: 3D coordinates (3)
    point_feats = torch.tensor(points_3d, dtype=torch.float32)
    data['point'].x = point_feats
    
    # Build edge indices
    cam_indices = []
    point_indices = []
    edge_features = []
    
    for point_idx, frame_idx, u, v in observations:
        cam_indices.append(frame_idx)
        point_indices.append(point_idx)
        
        # Normalized pixel coordinates (0-1 range)
        u_norm = u / img_width
        v_norm = v / img_height
        edge_features.append([u_norm, v_norm])
    
    # Edge index for camera -> point (observation)
    data['camera', 'observes', 'point'].edge_index = torch.tensor(
        [cam_indices, point_indices], dtype=torch.long
    )
    
    # Edge features: normalized pixel coordinates
    data['camera', 'observes', 'point'].edge_attr = torch.tensor(
        edge_features, dtype=torch.float32
    )
    
    return data


# =============================================================================
# GNN Model
# =============================================================================

class BAMessagePassing(nn.Module):
    """
    Message Passing Network for Bundle Adjustment.
    
    Updates both camera poses and 3D points to minimize reprojection errors.
    """
    
    def __init__(self, hidden_dim: int = 32, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Message MLP for edge features
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),  # src + dst + edge feat
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Forward pass through GNN layers.
        
        Args:
            x_dict: Dictionary of node features {'camera': ..., 'point': ...}
            edge_index_dict: Dictionary of edge indices
            edge_attr_dict: Dictionary of edge features
            
        Returns:
            Updated node features
        """
        for layer in range(self.num_layers):
            # Process camera nodes
            if 'camera' in x_dict and ('camera', 'observes', 'point') in edge_index_dict:
                x_dict['camera'] = self.update_camera_nodes(
                    x_dict, edge_index_dict, edge_attr_dict
                )
            
            # Process point nodes
            if 'point' in x_dict and ('camera', 'observes', 'point') in edge_index_dict:
                x_dict['point'] = self.update_point_nodes(
                    x_dict, edge_index_dict, edge_attr_dict
                )
        
        return x_dict
    
    def update_camera_nodes(self, x_dict, edge_index_dict, edge_attr_dict):
        """Update camera node features via message passing."""
        edge_index = edge_index_dict[('camera', 'observes', 'point')]
        edge_attr = edge_attr_dict[('camera', 'observes', 'point')]
        
        # Get source (camera) and target (point) features
        src_feats = x_dict['camera'][edge_index[0]]  # Source: camera
        dst_feats = x_dict['point'][edge_index[1]]   # Target: point
        
        # Build messages
        msg_input = torch.cat([src_feats, dst_feats, edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)
        
        # Aggregate messages (sum)
        num_cameras = x_dict['camera'].shape[0]
        aggregated = torch.zeros(num_cameras, self.hidden_dim, device=messages.device)
        aggregated.index_add_(0, edge_index[0], messages)
        
        # Update: residual connection
        update_input = torch.cat([x_dict['camera'], aggregated], dim=-1)
        update = self.update_mlp(update_input)
        
        return x_dict['camera'] + update[:, :self.hidden_dim]
    
    def update_point_nodes(self, x_dict, edge_index_dict, edge_attr_dict):
        """Update point node features via message passing."""
        # Point nodes receive messages from camera nodes (reverse direction)
        # For simplicity, we use the same edges but swap direction
        edge_index = edge_index_dict[('camera', 'observes', 'point')]
        edge_attr = edge_attr_dict[('camera', 'observes', 'point')]
        
        # Messages from cameras to points
        src_feats = x_dict['camera'][edge_index[0]]
        dst_feats = x_dict['point'][edge_index[1]]
        
        msg_input = torch.cat([src_feats, dst_feats, edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)
        
        # Aggregate messages
        num_points = x_dict['point'].shape[0]
        aggregated = torch.zeros(num_points, self.hidden_dim, device=messages.device)
        aggregated.index_add_(0, edge_index[1], messages)
        
        # Update
        update_input = torch.cat([x_dict['point'], aggregated], dim=-1)
        update = self.update_mlp(update_input)
        
        return x_dict['point'] + update[:, :self.hidden_dim]


class GNNBAOptimizer(nn.Module):
    """
    GNN-based Bundle Adjustment Optimizer.
    
    Learns to optimize camera poses and 3D points to minimize reprojection errors.
    """
    
    def __init__(self, hidden_dim: int = 32, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature embedding layers
        self.camera_embed = nn.Linear(6, hidden_dim)  # 6 DoF -> hidden
        self.point_embed = nn.Linear(3, hidden_dim)   # 3D point -> hidden
        self.edge_embed = nn.Linear(2, hidden_dim)    # 2D coord -> hidden
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection layers (hidden -> original dimensions)
        # Initialize to zero to preserve identity mapping (delta = 0 initially)
        self.camera_out_proj = nn.Linear(hidden_dim, 6)
        self.point_out_proj = nn.Linear(hidden_dim, 3)
        
        # Zero initialization for output layers
        nn.init.zeros_(self.camera_out_proj.weight)
        nn.init.zeros_(self.camera_out_proj.bias)
        nn.init.zeros_(self.point_out_proj.weight)
        nn.init.zeros_(self.point_out_proj.bias)
    
    def forward(self, data: HeteroData, initial_poses: list, 
                points_3d: np.ndarray, K: dict) -> tuple:
        """
        Forward pass to optimize poses and points.
        
        Args:
            data: HeteroData graph
            initial_poses: List of initial 4x4 pose matrices
            points_3d: Initial 3D points (N, 3)
            K: Camera intrinsics
            
        Returns:
            Tuple of (optimized_poses, optimized_points)
        """
        # Embed initial features
        x_dict = {
            'camera': self.camera_embed(data['camera'].x),
            'point': self.point_embed(data['point'].x)
        }
        
        # Embed edge features
        edge_attr_dict = {
            ('camera', 'observes', 'point'): 
                self.edge_embed(data['camera', 'observes', 'point'].edge_attr)
        }
        
        # Message passing
        edge_index_dict = {
            ('camera', 'observes', 'point'): 
                data['camera', 'observes', 'point'].edge_index
        }
        
        for layer in self.mp_layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
        
        # Project back to original dimensions
        delta_camera = self.camera_out_proj(x_dict['camera'])
        delta_point = self.point_out_proj(x_dict['point'])
        
        # Apply deltas to initial values
        optimized_poses = []
        for i, (pose, delta) in enumerate(zip(initial_poses, delta_camera)):
            # delta is axis-angle (3) + translation (3)
            axis_angle = delta[:3].detach().cpu().numpy()
            translation = delta[3:].detach().cpu().numpy()
            
            # Update rotation
            R = axis_angle_to_rotation_matrix(axis_angle)
            new_R = R @ pose[:3, :3]
            
            # Update translation
            new_t = pose[:3, 3] + translation
            
            # Combine
            new_pose = np.eye(4)
            new_pose[:3, :3] = new_R
            new_pose[:3, 3] = new_t
            optimized_poses.append(new_pose)
        
        # Update 3D points
        optimized_points = points_3d + delta_point.detach().cpu().numpy()
        
        return optimized_poses, optimized_points


class MessagePassingLayer(nn.Module):
    """Single message passing layer."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Message function
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """One round of message passing."""
        new_x_dict = {}
        
        # Process camera -> point edges
        edge_index = edge_index_dict[('camera', 'observes', 'point')]
        edge_attr = edge_attr_dict[('camera', 'observes', 'point')]
        
        # Aggregate from cameras to points (for point updates)
        if 'point' in x_dict:
            src = x_dict['camera'][edge_index[0]]
            dst = x_dict['point'][edge_index[1]]
            
            msg = self.msg_net(torch.cat([src, dst, edge_attr], dim=-1))
            
            # Simple sum aggregation
            num_points = x_dict['point'].shape[0]
            agg = torch.zeros(num_points, msg.shape[-1], device=msg.device)
            agg.index_add_(0, edge_index[1], msg)
            
            # Update point features
            point_update = self.update_net(torch.cat([x_dict['point'], agg], dim=-1))
            new_x_dict['point'] = x_dict['point'] + point_update
        
        # Aggregate from points to cameras (for camera updates)
        if 'camera' in x_dict:
            src = x_dict['point'][edge_index[1]]
            dst = x_dict['camera'][edge_index[0]]
            
            msg = self.msg_net(torch.cat([src, dst, edge_attr], dim=-1))
            
            # Simple sum aggregation
            num_cameras = x_dict['camera'].shape[0]
            agg = torch.zeros(num_cameras, msg.shape[-1], device=msg.device)
            agg.index_add_(0, edge_index[0], msg)
            
            # Update camera features
            cam_update = self.update_net(torch.cat([x_dict['camera'], agg], dim=-1))
            new_x_dict['camera'] = x_dict['camera'] + cam_update
        
        return new_x_dict


# =============================================================================
# Projection and Loss Functions
# =============================================================================

def project_points_numpy(points_3d: np.ndarray, pose: np.ndarray, K: dict) -> np.ndarray:
    """
    Project 3D points to 2D pixel coordinates (numpy version for evaluation).
    
    Args:
        points_3d: Nx3 array of 3D points
        pose: 4x4 camera pose matrix (world to camera)
        K: Camera intrinsics
        
    Returns:
        Nx2 array of projected 2D points
    """
    if len(points_3d) == 0:
        return np.array([])
    
    # Convert to homogeneous coordinates
    points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # Transform to camera frame
    cam_points = (pose @ points_h.T).T
    
    # Project to image plane
    K_matrix = np.array([[K['fx'], 0, K['cx']],
                         [0, K['fy'], K['cy']],
                         [0, 0, 1]])
    
    # Divide by depth
    depths = cam_points[:, 2:3]
    valid = depths.flatten() > 0.1
    
    proj_points = np.zeros((len(points_3d), 2))
    if valid.any():
        proj = K_matrix @ cam_points[:, :3].T
        proj_points[valid] = (proj[:2, valid] / depths[valid]).T
    
    return proj_points


def axis_angle_to_rotation_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle to rotation matrix using Rodrigues formula (differentiable).
    
    Args:
        axis_angle: (..., 3) axis-angle vector
        
    Returns:
        (..., 3, 3) rotation matrix
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-10)
    axis = axis_angle / angle
    
    # Build skew-symmetric matrix
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    
    # Rodrigues formula: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    sin_angle = torch.sin(angle)
    cos_angle = torch.cos(angle)
    
    R = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    R = R.unsqueeze(0).expand(angle.shape[0], -1, -1)
    R = R + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
    
    return R.squeeze(0) if R.shape[0] == 1 else R


def project_points_torch(points_3d: torch.Tensor, poses: torch.Tensor, K: dict, device: torch.device) -> torch.Tensor:
    """
    Project 3D points to 2D pixel coordinates (differentiable PyTorch version).
    
    Args:
        points_3d: (N, 3) 3D points in world frame
        poses: (M, 4, 4) camera poses (world to camera)
        K: Camera intrinsics dictionary
        device: torch device
        
    Returns:
        (M, N, 2) projected 2D points for each camera
    """
    N = points_3d.shape[0]
    M = poses.shape[0]
    
    # Build intrinsic matrix
    K_matrix = torch.tensor([[K['fx'], 0, K['cx']],
                              [0, K['fy'], K['cy']],
                              [0, 0, 1]], dtype=torch.float32, device=device)
    
    # Add homogeneous coordinate: (N, 4)
    points_h = torch.cat([points_3d, torch.ones(N, 1, device=device)], dim=1)
    
    # Transform to camera frame: poses @ points_h.T
    # poses: (M, 4, 4), points_h.T: (4, N) -> (M, 4, N)
    points_cam = torch.matmul(poses[:, :3, :], points_h.T)  # (M, 3, N)
    
    # Extract depths
    depths = points_cam[:, 2, :].unsqueeze(1).clamp(min=0.1)  # (M, 1, N)
    
    # Project to image plane: K @ cam_points
    # K_matrix: (3, 3), points_cam: (M, 3, N) -> (M, 3, N)
    proj_h = torch.matmul(K_matrix, points_cam)  # (M, 3, N)
    
    # Divide by depth: (M, 2, N)
    proj_2d = proj_h[:, :2, :] / depths
    
    # Transpose to (M, N, 2)
    proj_2d = proj_2d.permute(0, 2, 1)
    
    return proj_2d


def compute_reprojection_error(poses: list, points_3d: np.ndarray, 
                               observations: list, K: dict) -> float:
    """
    Compute total reprojection error.
    
    Args:
        poses: List of camera poses
        points_3d: Nx3 array of 3D points
        observations: List of (point_idx, frame_idx, u, v)
        K: Camera intrinsics
        
    Returns:
        RMSE of reprojection error in pixels
    """
    errors = []
    
    for point_idx, frame_idx, u_obs, v_obs in observations:
        if point_idx >= len(points_3d):
            continue
        
        # Project point
        point_3d = points_3d[point_idx:point_idx+1]
        proj = project_points_numpy(point_3d, poses[frame_idx], K)
        
        if len(proj) > 0:
            error = np.sqrt((proj[0, 0] - u_obs)**2 + (proj[0, 1] - v_obs)**2)
            errors.append(error)
    
    if len(errors) == 0:
        return 0.0
    
    return np.sqrt(np.mean(np.array(errors)**2))


def huber_loss(errors: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss for robust estimation."""
    abs_errors = torch.abs(errors)
    quadratic = torch.clamp(abs_errors, max=delta)
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


# =============================================================================
# Main Training Function
# =============================================================================

def add_noise_to_poses(poses: list, trans_noise: float = 0.2, 
                       rot_noise: float = 0.05) -> list:
    """
    Add noise to ground truth poses to simulate VO initialization.
    
    Args:
        poses: List of ground truth poses
        trans_noise: Translation noise in meters
        rot_noise: Rotation noise in radians
        
    Returns:
        List of noisy poses
    """
    noisy_poses = []
    
    for pose in poses:
        # Add translation noise
        t_noise = np.random.randn(3) * trans_noise
        new_t = pose[:3, 3] + t_noise
        
        # Add rotation noise (axis-angle)
        rot_noise_vec = np.random.randn(3) * rot_noise
        R_noise = axis_angle_to_rotation_matrix(rot_noise_vec)
        new_R = R_noise @ pose[:3, :3]
        
        # Combine
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_R
        new_pose[:3, 3] = new_t
        noisy_poses.append(new_pose)
    
    return noisy_poses


def run_gnn_ba(kitti_root: str, sequence: str = '00', num_frames: int = 5,
               num_epochs: int = 100, hidden_dim: int = 32, learning_rate: float = 0.001,
               save_plot: str = None):
    """
    Main function to run GNN-based Bundle Adjustment.
    
    Args:
        kitti_root: Path to KITTI odometry dataset
        sequence: Sequence number (e.g., '00')
        num_frames: Number of frames to process
        num_epochs: Number of training epochs
        hidden_dim: Hidden dimension for GNN
        learning_rate: Learning rate for optimizer
    """
    print(f"Loading KITTI sequence {sequence}, frames 0-{num_frames-1}.")
    
    # Paths
    seq_dir = os.path.join(kitti_root, 'sequences_jpg', sequence)
    image_dir = os.path.join(seq_dir, 'image_0')
    calib_path = os.path.join(seq_dir, 'calib.txt')
    poses_path = os.path.join(kitti_root, 'poses', f'{sequence}.txt')
    
    # Check if paths exist
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        print("Please ensure KITTI dataset is properly downloaded.")
        return
    
    if not os.path.exists(calib_path):
        print(f"Error: Calib file not found: {calib_path}")
        return
    
    if not os.path.exists(poses_path):
        print(f"Error: Poses file not found: {poses_path}")
        return
    
    # Load data
    images = load_kitti_images(image_dir, num_frames)
    if len(images) < 2:
        print("Error: Not enough images loaded")
        return
    
    K = load_kitti_intrinsics(calib_path)
    gt_poses = load_kitti_poses(poses_path, num_frames)
    
    print(f"Loaded {len(images)} images.")
    print(f"Camera intrinsics: fx={K['fx']}, fy={K['fy']}, cx={K['cx']}, cy={K['cy']}")
    
    # Get image dimensions
    img_height, img_width = images[0].shape
    
    # Feature extraction and matching
    print("\nExtracting features and matching...")
    match_data = extract_and_match_features(images)
    
    # Build tracks
    tracks = build_track_matches(images, match_data)
    print(f"Built {len(tracks)} feature tracks across {num_frames} frames.")
    
    # Add noise to ground truth for initial poses
    print("\nAdding noise to ground truth poses...")
    noisy_poses = add_noise_to_poses(gt_poses)
    
    # Triangulate 3D points
    print("Triangulating 3D points...")
    points_3d, observations = triangulate_all_points(tracks, noisy_poses, K)
    print(f"Triangulated {len(points_3d)} valid 3D points with {len(observations)} observations.")
    
    if len(points_3d) < 10:
        print("Error: Not enough 3D points triangulated")
        return
    
    # Build heterogeneous graph
    print("Building heterogeneous graph...")
    data = build_heterogeneous_graph(
        num_cameras=num_frames,
        points_3d=points_3d,
        observations=observations,
        K=K,
        img_width=img_width,
        img_height=img_height
    )
    
    # Initialize camera node features from noisy poses
    camera_feats = []
    for pose in noisy_poses:
        # Convert to axis-angle + translation
        R = pose[:3, :3]
        t = pose[:3, 3]
        axis_angle = rotation_matrix_to_axis_angle(R)
        feat = np.concatenate([axis_angle, t])
        camera_feats.append(feat)
    camera_feats = np.array(camera_feats)  # Convert to numpy array to avoid warning
    data['camera'].x = torch.tensor(camera_feats, dtype=torch.float32)
    
    print(f"Graph: {num_frames} camera nodes, {len(points_3d)} point nodes, "
          f"{len(observations)} observation edges.")
    
    # Compute initial reprojection error (with noisy poses, before any GNN optimization)
    initial_error = compute_reprojection_error(noisy_poses, points_3d, observations, K)
    
    # Also compute initial error using the graph's initial camera params (should be same as noisy)
    initial_camera_params = data['camera'].x.numpy()
    init_poses_from_graph = []
    for i in range(num_frames):
        axis_angle = initial_camera_params[i, :3]
        trans = initial_camera_params[i, 3:]
        R = axis_angle_to_rotation_matrix(axis_angle)
        t = trans
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        init_poses_from_graph.append(pose)
    initial_error_from_graph = compute_reprojection_error(init_poses_from_graph, points_3d, observations, K)
    print(f"Initial error verification (from graph): {initial_error_from_graph:.2f} pixels")
    print(f"\nInitial reprojection error (RMSE): {initial_error:.2f} pixels")
    
    # Set up GNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNBAOptimizer(hidden_dim=hidden_dim, num_layers=3).to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        # Initialize with current noisy poses and triangulated points
        x_dict = {
            'camera': model.camera_embed(data['camera'].x),
            'point': model.point_embed(data['point'].x)
        }
        
        edge_attr_dict = {
            ('camera', 'observes', 'point'): 
                model.edge_embed(data['camera', 'observes', 'point'].edge_attr)
        }
        
        edge_index_dict = {
            ('camera', 'observes', 'point'): 
                data['camera', 'observes', 'point'].edge_index
        }
        
        # Message passing
        for layer in model.mp_layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
        
        # Get updates
        delta_camera = model.camera_out_proj(x_dict['camera'])
        delta_point = model.point_out_proj(x_dict['point'])
        
        # Apply updates to get optimized parameters
        # Camera: axis-angle + translation
        camera_params = data['camera'].x + delta_camera  # Shape: (num_cameras, 6)
        
        # Points: 3D coordinates
        point_params = data['point'].x + delta_point  # Shape: (num_points, 3)
        
        # Compute reprojection loss using differentiable operations
        # Get edge information
        edge_index = data['camera', 'observes', 'point'].edge_index
        edge_attr = data['camera', 'observes', 'point'].edge_attr
        
        # Denormalize edge features to get pixel coordinates
        u_obs = edge_attr[:, 0] * img_width
        v_obs = edge_attr[:, 1] * img_height
        
        # Build camera poses from optimized parameters
        # camera_params: (num_cameras, 6) -> axis-angle(3) + translation(3)
        axis_angles = camera_params[:, :3]  # (M, 3)
        translations = camera_params[:, 3:]  # (M, 3)
        
        # Get rotation matrices for all cameras at once
        Rs = []
        for i in range(axis_angles.shape[0]):
            R = axis_angle_to_rotation_matrix_torch(axis_angles[i:i+1]).squeeze(0)
            Rs.append(R)
        Rs = torch.stack(Rs)  # (M, 3, 3)
        
        # Build pose matrices
        pose_matrices = torch.eye(4).unsqueeze(0).repeat(axis_angles.shape[0], 1, 1).to(device)
        pose_matrices[:, :3, :3] = Rs
        pose_matrices[:, :3, 3] = translations
        
        # Get all 3D points
        all_points = point_params  # (N, 3)
        
        # Project all points with all cameras using vectorized operation
        # all_points: (N, 3), pose_matrices: (M, 4, 4) -> (M, N, 2)
        projected = project_points_torch(all_points, pose_matrices, K, device)
        
        # Gather observations
        cam_indices = edge_index[0]
        pt_indices = edge_index[1]
        
        # Get projected coordinates for each observation
        proj_x = projected[cam_indices, pt_indices, 0]
        proj_y = projected[cam_indices, pt_indices, 1]
        
        # Compute reprojection error
        errors = torch.sqrt((proj_x - u_obs)**2 + (proj_y - v_obs)**2)
        
        # Filter out invalid projections (behind camera)
        # Compute depths: (R @ p + t)[2] for each camera-point pair
        # points_cam = pose @ point_h
        points_h = torch.cat([all_points, torch.ones(all_points.shape[0], 1, device=device)], dim=1)
        points_cam = torch.matmul(pose_matrices[:, :3, :], points_h.T)  # (M, 3, N)
        depths = points_cam[:, 2, :]  # (M, N)
        
        # Get depths for each observation
        obs_depths = depths[cam_indices, pt_indices]
        valid_mask = obs_depths > 0.1
        
        valid_errors = errors[valid_mask]
        
        if valid_errors.numel() > 0:
            loss = huber_loss(valid_errors, delta=1.0).mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            with torch.no_grad():
                # Extract optimized poses for RMSE computation
                opt_camera = camera_params.detach().cpu().numpy()
                opt_pts = point_params.detach().cpu().numpy()
                
                # Build optimized poses
                opt_poses_list = []
                for i in range(num_frames):
                    axis_angle = opt_camera[i, :3]
                    trans = opt_camera[i, 3:]
                    R = axis_angle_to_rotation_matrix(axis_angle)
                    t = trans  # Use translation directly from optimized params
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = t
                    opt_poses_list.append(pose)
                
                current_error = compute_reprojection_error(
                    opt_poses_list, opt_pts, observations, K
                )
                print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.4f}, "
                      f"RMSE = {current_error:.2f} pixels")
    
    # Final evaluation - extract optimized poses
    model.eval()
    with torch.no_grad():
        # Forward pass to get optimized parameters
        x_dict = {
            'camera': model.camera_embed(data['camera'].x),
            'point': model.point_embed(data['point'].x)
        }
        
        edge_attr_dict = {
            ('camera', 'observes', 'point'): 
                model.edge_embed(data['camera', 'observes', 'point'].edge_attr)
        }
        
        edge_index_dict = {
            ('camera', 'observes', 'point'): 
                data['camera', 'observes', 'point'].edge_index
        }
        
        # Message passing
        for layer in model.mp_layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
        
        # Get updates
        delta_camera = model.camera_out_proj(x_dict['camera'])
        delta_point = model.point_out_proj(x_dict['point'])
        
        # Apply deltas to get optimized parameters
        opt_camera_params = data['camera'].x + delta_camera
        opt_points_3d = data['point'].x + delta_point
    
    # Convert optimized camera parameters to poses
    # opt_camera_params already contains the full optimized parameters (initial + delta)
    optimized_poses = []
    for i in range(num_frames):
        # Get the optimized camera parameters directly
        axis_angle = opt_camera_params[i, :3].cpu().numpy()
        trans = opt_camera_params[i, 3:].cpu().numpy()
        
        # Convert axis-angle to rotation matrix
        R = axis_angle_to_rotation_matrix(axis_angle)
        
        # Build pose directly from optimized params
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = trans
        optimized_poses.append(new_pose)
    
    optimized_points = opt_points_3d.cpu().numpy()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    final_error = compute_reprojection_error(optimized_poses, optimized_points, observations, K)
    print(f"Initial reprojection error (RMSE): {initial_error:.2f} pixels")
    print(f"Final reprojection error (RMSE): {final_error:.2f} pixels")
    if initial_error > 0:
        print(f"Error reduction: {(initial_error - final_error) / initial_error * 100:.1f}%")
    
    # Compute camera position error vs ground truth
    errors_trans = []
    for i, (gt_pose, noisy_pose, opt_pose) in enumerate(zip(gt_poses, noisy_poses, optimized_poses)):
        # Translation error
        t_gt = gt_pose[:3, 3]
        t_noisy = noisy_pose[:3, 3]
        t_opt = opt_pose[:3, 3]
        
        err_noisy = np.linalg.norm(t_noisy - t_gt)
        err_opt = np.linalg.norm(t_opt - t_gt)
        errors_trans.append((err_noisy, err_opt))
    
    avg_err_noisy = np.mean([e[0] for e in errors_trans])
    avg_err_opt = np.mean([e[1] for e in errors_trans])
    
    print(f"\nCamera position error vs ground truth:")
    print(f"  Initial (noisy): {avg_err_noisy:.3f} m")
    print(f"  After GNN: {avg_err_opt:.3f} m")
    
    # Plot trajectories
    try:
        import matplotlib.pyplot as plt
        
        # Extract positions
        gt_positions = np.array([p[:3, 3] for p in gt_poses])
        noisy_positions = np.array([p[:3, 3] for p in noisy_poses])
        opt_positions = np.array([p[:3, 3] for p in optimized_poses])
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot XZ trajectory (top-down view)
        ax.plot(gt_positions[:, 0], gt_positions[:, 2], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(noisy_positions[:, 0], noisy_positions[:, 2], 'r--', label='Noisy (Initial)', linewidth=1.5, alpha=0.7)
        ax.plot(opt_positions[:, 0], opt_positions[:, 2], 'g-.', label='GNN Optimized', linewidth=1.5, alpha=0.7)
        
        # Mark start and end
        ax.scatter(gt_positions[0, 0], gt_positions[0, 2], c='b', s=100, marker='o', zorder=5)
        ax.scatter(gt_positions[-1, 0], gt_positions[-1, 2], c='b', s=100, marker='x', zorder=5)
        
        ax.scatter(noisy_positions[0, 0], noisy_positions[0, 2], c='r', s=80, marker='o', zorder=4)
        ax.scatter(noisy_positions[-1, 0], noisy_positions[-1, 2], c='r', s=80, marker='x', zorder=4)
        
        ax.scatter(opt_positions[0, 0], opt_positions[0, 2], c='g', s=80, marker='o', zorder=4)
        ax.scatter(opt_positions[-1, 0], opt_positions[-1, 2], c='g', s=80, marker='x', zorder=4)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'Camera Trajectory Comparison\nInitial RMSE: {initial_error:.2f}px, Final RMSE: {final_error:.2f}px')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # Save plot
        if save_plot:
            plt.savefig(save_plot, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_plot}")
        else:
            plt.savefig('gnn_ba_trajectory.png', dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: gnn_ba_trajectory.png")
        plt.close()
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot.")
    
    print("\nGNN-BA optimization completed!")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN-based Bundle Adjustment Test')
    parser.add_argument('--kitti_root', type=str, 
                        default='/path/to/kitti/odometry/dataset',
                        help='Path to KITTI odometry dataset')
    parser.add_argument('--sequence', type=str, default='00',
                        help='Sequence number (e.g., 00, 02)')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames to process')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension for GNN')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Path to save trajectory plot')
    
    args = parser.parse_args()
    
    run_gnn_ba(
        kitti_root=args.kitti_root,
        sequence=args.sequence,
        num_frames=args.num_frames,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        save_plot=args.save_plot
    )
