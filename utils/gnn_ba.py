"""
Utility functions for GNN-based Bundle Adjustment.

Includes:
- Triangulation: DLT-based 3D point estimation from 2D observations
- Pose conversion: Euler/Rotation matrix transformations, normalization
- Graph building: PyTorch Geometric HeteroData construction
- Losses: Huber loss for reprojection error
- Post-processing: Window merging for sequence prediction
"""

import queue

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import HeteroData


# KITTI normalization statistics for VO model outputs
KITTI_MEAN_ANGLES = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
KITTI_STD_ANGLES = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
KITTI_MEAN_T = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
KITTI_STD_T = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])


# =============================================================================
# Rotation and Pose Conversions
# =============================================================================

def rotation_to_euler(R, seq='zyx'):
    """Convert rotation matrix to Euler angles."""
    rot = Rotation.from_matrix(R)
    return rot.as_euler(seq, degrees=False)


def euler_to_rotation(euler, seq='zyx'):
    """Convert Euler angles to rotation matrix."""
    rot = Rotation.from_euler(seq, euler, degrees=False)
    return rot.as_matrix()


def euler_to_rotation_torch(euler, seq='zyx'):
    """
    Convert euler angles (in radians) to rotation matrices using PyTorch.
    Fully differentiable implementation.

    Args:
        euler: tensor of shape (N, 3) with euler angles in radians [z, y, x] for seq='zyx'
        seq: rotation sequence string like 'zyx'
    """
    if seq != 'zyx':
        raise NotImplementedError(f"Only seq='zyx' is implemented, got {seq}")

    # euler shape: (N, 3) = (z, y, x) angles
    z = euler[:, 0]
    y = euler[:, 1]
    x = euler[:, 2]

    # Rotation matrices for each axis
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)

    # Rz(yaw) @ Ry(pitch) @ Rx(roll)
    # R = Rz * Ry * Rx
    R = torch.zeros(euler.shape[0], 3, 3, dtype=euler.dtype, device=euler.device)
    R[:, 0, 0] = cz * cy
    R[:, 0, 1] = cz * sy * sx - sz * cx
    R[:, 0, 2] = cz * sy * cx + sz * sx
    R[:, 1, 0] = sz * cy
    R[:, 1, 1] = sz * sy * sx + cz * cx
    R[:, 1, 2] = sz * sy * cx - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cy * cx

    return R


def axis_angle_to_rotation_matrix(axis_angle):
    """Convert axis-angle to rotation matrix (numpy)."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


def axis_angle_to_rotation_matrix_torch(axis_angle):
    """Convert axis-angle to rotation matrix (torch)."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-10)
    axis = axis_angle / angle
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    sin_angle = torch.sin(angle)
    cos_angle = torch.cos(angle)
    R = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    R = R.unsqueeze(0).expand(angle.shape[0], -1, -1)
    R = R + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
    return R.squeeze(0) if R.shape[0] == 1 else R


def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle."""
    rot = Rotation.from_matrix(R)
    return rot.as_rotvec()


# =============================================================================
# Pose Feature Conversion
# =============================================================================

def pose_6dof_to_matrix(pose_6dof, seq='zyx'):
    """
    Convert 6-DOF pose to 4x4 transformation matrix.

    Args:
        pose_6dof: (6,) array of [euler(3), t(3)] or (N, 6) array of multiple poses
        seq: rotation sequence for euler angles

    Returns:
        4x4 transformation matrix or list of 4x4 matrices
    """
    if isinstance(pose_6dof, np.ndarray) and pose_6dof.ndim == 1:
        euler = pose_6dof[:3]
        t = pose_6dof[3:]
        R = euler_to_rotation(euler, seq=seq)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        return pose
    elif isinstance(pose_6dof, np.ndarray) and pose_6dof.ndim == 2:
        poses = []
        for i in range(pose_6dof.shape[0]):
            poses.append(pose_6dof_to_matrix(pose_6dof[i], seq))
        return poses
    elif isinstance(pose_6dof, torch.Tensor):
        return pose_6dof_to_matrix(pose_6dof.cpu().numpy(), seq)
    return None


def pose_6dof_to_matrix_torch(pose_6dof, seq='zyx'):
    """
    Convert 6-DOF pose to 4x4 transformation matrix using PyTorch.

    Args:
        pose_6dof: (N, 6) tensor of [euler(3), t(3)]
        seq: rotation sequence (only 'zyx' supported)

    Returns:
        (N, 4, 4) tensor of transformation matrices
    """
    euler = pose_6dof[:, :3]  # (N, 3)
    t = pose_6dof[:, 3:]     # (N, 3)
    Rs = euler_to_rotation_torch(euler, seq=seq)  # (N, 3, 3)

    N = pose_6dof.shape[0]
    pose_matrices = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(pose_6dof.device)
    pose_matrices[:, :3, :3] = Rs
    pose_matrices[:, :3, 3] = t
    return pose_matrices


def poses_to_camera_features(poses, mean_angles, std_angles, mean_t, std_t):
    """
    Convert camera poses to normalized camera feature vectors.

    Supports both 4x4 transformation matrices and 6-DOF vectors as input.
    Converts rotation matrix to Euler angles, then normalizes both
    euler angles and translation using dataset statistics.

    Args:
        poses: 4x4 transformation matrices (N, 4, 4) or 6-DOF vectors (N, 6)
               or list of 4x4 matrices, or single 4x4 matrix (4, 4)
        mean_angles, std_angles: normalization parameters for euler angles
        mean_t, std_t: normalization parameters for translation

    Returns:
        camera_feats: (N, 6) array of [euler_norm(3), trans_norm(3)]
    """
    # Handle 6-DOF input directly
    if isinstance(poses, np.ndarray):
        if poses.ndim == 2 and poses.shape[1] == 6:
            # Already 6-DOF format, just normalize
            euler = poses[:, :3]
            t = poses[:, 3:]
            euler_norm = (euler - mean_angles) / std_angles
            t_norm = (t - mean_t) / std_t
            return np.concatenate([euler_norm, t_norm], axis=1)
        elif poses.ndim == 1 and poses.shape[0] == 6:
            # Single 6-DOF pose
            euler = poses[:3]
            t = poses[3:]
            euler_norm = (euler - mean_angles) / std_angles
            t_norm = (t - mean_t) / std_t
            return np.concatenate([euler_norm, t_norm])
        elif poses.ndim == 2 and poses.shape == (4, 4):
            poses = [poses]
        elif poses.ndim == 3 and poses.shape[1:] == (4, 4):
            poses = list(poses)
    elif isinstance(poses, torch.Tensor):
        poses = poses.cpu().numpy()
        if poses.ndim == 2 and poses.shape[1] == 6:
            # Already 6-DOF format, just normalize
            euler = poses[:, :3]
            t = poses[:, 3:]
            euler_norm = (euler - mean_angles) / std_angles
            t_norm = (t - mean_t) / std_t
            return np.concatenate([euler_norm, t_norm], axis=1)
        elif poses.ndim == 1 and poses.shape[0] == 6:
            euler = poses[:3]
            t = poses[3:]
            euler_norm = (euler - mean_angles) / std_angles
            t_norm = (t - mean_t) / std_t
            return np.concatenate([euler_norm, t_norm])
        elif poses.ndim == 2 and poses.shape == (4, 4):
            poses = [poses]
        elif poses.ndim == 3 and poses.shape[1:] == (4, 4):
            poses = list(poses)

    camera_feats = []
    for pose in poses:
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        R = pose[:3, :3]
        t = pose[:3, 3]
        euler = rotation_to_euler(R, seq='zyx')
        euler_norm = (euler - mean_angles) / std_angles
        t_norm = (t - mean_t) / std_t
        feat = np.concatenate([euler_norm, t_norm])
        camera_feats.append(feat)
    return np.array(camera_feats)


def camera_features_to_poses(camera_params, mean_angles, std_angles, mean_t, std_t):
    """
    Convert normalized camera features back to 4x4 poses.

    Inverse of poses_to_camera_features: denormalizes and converts
    euler angles back to rotation matrix.

    Args:
        camera_params: (N, 6) array of [euler_norm(3), trans_norm(3)]
        mean_angles, std_angles: denormalization params for euler
        mean_t, std_t: denormalization params for translation

    Returns:
        poses: list of 4x4 transformation matrices
    """
    poses = []
    for i in range(camera_params.shape[0]):
        euler_norm = camera_params[i, :3]
        t_norm = camera_params[i, 3:]
        euler = euler_norm * std_angles + mean_angles
        t = t_norm * std_t + mean_t
        R = euler_to_rotation(euler, seq='zyx')
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        poses.append(pose)
    return poses


def denormalize_poses(vo_output, window_size, mean_angles, std_angles, mean_t, std_t):
    """
    Convert VO model output (normalized relative poses) to absolute 4x4 poses.

    The VO model outputs normalized relative poses between consecutive frames.
    This function:
    1. Denormalizes each relative pose
    2. Accumulates them to get absolute poses for the window

    Args:
        vo_output: (batch, window_size-1, 6) raw model output
        window_size: number of frames per window
        mean_angles, std_angles, mean_t, std_t: KITTI normalization statistics

    Returns:
        poses: list of list of 4x4 absolute poses per batch
    """
    vo_output = vo_output.reshape(-1, window_size - 1, 6)
    # Convert tensors to numpy if needed
    if isinstance(std_angles, torch.Tensor):
        std_angles = std_angles.cpu().numpy()
        mean_angles = mean_angles.cpu().numpy()
        std_t = std_t.cpu().numpy()
        mean_t = mean_t.cpu().numpy()
    if isinstance(vo_output, torch.Tensor):
        vo_output = vo_output.cpu().numpy()
    poses = []
    for batch_idx in range(vo_output.shape[0]):
        batch_poses = []
        # Denormalize each relative pose
        for frame_idx in range(vo_output.shape[1]):
            euler_norm = vo_output[batch_idx, frame_idx, :3]
            t_norm = vo_output[batch_idx, frame_idx, 3:]
            euler = euler_norm * std_angles + mean_angles
            t = t_norm * std_t + mean_t
            R = euler_to_rotation(euler, seq='zyx')
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            batch_poses.append(pose)

        # Accumulate relative poses to get absolute poses
        abs_poses = []
        abs_pose = np.eye(4)
        abs_poses.append(abs_pose.copy())  # First frame is identity
        for rel_pose in batch_poses:
            abs_pose = abs_pose @ rel_pose  # Compose poses: new_abs = current_abs * rel_pose
            abs_poses.append(abs_pose.copy())

        poses.append(abs_poses)
    return poses


# =============================================================================
# Triangulation
# =============================================================================

def triangulate_dlt(points_2d, poses, K, device='cpu'):
    """
    Triangulate a 3D point from 2D observations using Direct Linear Transform (DLT).
    PyTorch implementation with CUDA support and differentiable.

    For each observation (u, v) and camera pose P:
        - x = K^{-1} * [u, v, 1]  (normalized coordinates)
        - Add equations: x[0]*P[2,:] - P[0,:] = 0
                        x[1]*P[2,:] - P[1,:] = 0
    Solve using SVD - the solution is the last row of Vt.

    Args:
        points_2d: list of (u, v) observations
        poses: list of 4x4 camera poses (camera to world)
        K: camera intrinsics matrix (3x3)
        device: torch device for computation

    Returns:
        X: 4D homogeneous 3D point (X, Y, Z, W) as torch tensor
    """
    dtype = points_2d[0].dtype if isinstance(points_2d[0], torch.Tensor) else torch.float32
    A = []
    for (u, v), pose in zip(points_2d, poses):
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float().to(K.device)
        elif pose.device != K.device:
            pose = pose.to(K.device)
        pt = torch.tensor([u, v, 1.0], dtype=torch.float32, device=K.device)
        K_inv = torch.inverse(K)
        pt_norm = K_inv @ pt
        P = pose
        A.append(pt_norm[0] * P[2, :] - P[0, :])
        A.append(pt_norm[1] * P[2, :] - P[1, :])
    A = torch.stack(A)
    _, _, Vt = torch.svd(A)
    X = Vt[-1]
    return X / X[3]


def triangulate_tracks_no_filter(tracks, track_indices, poses_c2w, K, device='cpu'):
    """
    Triangulate 3D points for specific tracks WITHOUT cheirality filtering.

    Guarantees one output point per input track index, so point counts always match.
    Used for GT triangulation where we need the same number of points as VO triangulation.

    Args:
        tracks: full list of tracks
        track_indices: indices of tracks to triangulate (from VO triangulation valid_tracks)
        poses_c2w: List of 4x4 camera poses (camera to world)
        K: camera intrinsics dict
        device: torch device

    Returns:
        points_3d: (N, 3) tensor with exactly len(track_indices) points
    """
    if isinstance(K['fx'], torch.Tensor):
        K_matrix = torch.tensor([[K['fx'].item(), 0, K['cx'].item()],
                                  [0, K['fy'].item(), K['cy'].item()],
                                  [0, 0, 1]], dtype=torch.float32, device=device)
    else:
        K_matrix = torch.tensor([[K['fx'], 0, K['cx']],
                                  [0, K['fy'], K['cy']],
                                  [0, 0, 1]], dtype=torch.float32, device=device)

    poses_w2c = []
    for pose in poses_c2w:
        if isinstance(pose, np.ndarray):
            poses_w2c.append(np.linalg.inv(pose))
        else:
            poses_w2c.append(torch.inverse(pose))

    points_3d = []
    for track_idx in track_indices:
        track = tracks[track_idx]
        if len(track[0]) == 4:
            pts_2d = [(t[2].item() if hasattr(t[2], 'item') else t[2],
                       t[3].item() if hasattr(t[3], 'item') else t[3]) for t in track]
            frame_indices = [t[0].item() if hasattr(t[0], 'item') else t[0] for t in track]
        else:
            pts_2d = [(t[1].item() if hasattr(t[1], 'item') else t[1],
                       t[2].item() if hasattr(t[2], 'item') else t[2]) for t in track]
            frame_indices = [t[0].item() if hasattr(t[0], 'item') else t[0] for t in track]

        track_poses = [poses_w2c[i] for i in frame_indices]

        try:
            X = triangulate_dlt(pts_2d, track_poses, K_matrix, device=device)
            X_3d = X[:3] / X[3]
            points_3d.append(X_3d)
        except Exception:
            points_3d.append(torch.zeros(3, dtype=torch.float32, device=device))

    return torch.stack(points_3d)


def triangulate_all_points(tracks, poses_c2w, K, min_depth=0.1, device='cpu'):
    """
    Triangulate all 3D points from tracks.
    PyTorch implementation with CUDA support and differentiable.

    For each track (series of 2D observations across frames):
    1. Extract 2D points and corresponding camera poses
    2. Run DLT triangulation
    3. Validate: all camera depths must be positive (Cheirality check)

    Args:
        tracks: List of tracks, each track is [(frame_idx, u, v), ...] or [(frame_idx, kp_idx, u, v), ...]
        poses_c2w: List of 4x4 camera poses (camera to world) as torch tensors or numpy arrays
        K: camera intrinsics dict with fx, fy, cx, cy
        min_depth: minimum valid depth (default 0.1m)
        device: torch device for computation

    Returns:
        points_3d: (N, 3) torch tensor of triangulated 3D points
        observations: list of [point_idx, frame_idx, u, v] for graph edges
        valid_track_indices: list of original track indices that passed triangulation
    """
    if isinstance(K['fx'], torch.Tensor):
        K_matrix = torch.tensor([[K['fx'].item(), 0, K['cx'].item()], 
                                  [0, K['fy'].item(), K['cy'].item()], 
                                  [0, 0, 1]], dtype=torch.float32, device=device)
    else:
        K_matrix = torch.tensor([[K['fx'], 0, K['cx']], 
                                  [0, K['fy'], K['cy']], 
                                  [0, 0, 1]], dtype=torch.float32, device=device)

    # Convert c2w poses to w2c for triangulation and depth checks
    poses_w2c = []
    for pose in poses_c2w:
        if isinstance(pose, np.ndarray):
            poses_w2c.append(np.linalg.inv(pose))
        else:
            # torch.Tensor
            poses_w2c.append(torch.inverse(pose))
    points_3d = []
    observations = []
    valid_track_indices = []

    for track_idx, track in enumerate(tracks):
        if len(track) < 2:
            continue

        if len(track[0]) == 4:
            pts_2d = [(t[2].item() if hasattr(t[2], 'item') else t[2],
                       t[3].item() if hasattr(t[3], 'item') else t[3]) for t in track]
            frame_indices = [t[0].item() if hasattr(t[0], 'item') else t[0] for t in track]
        else:
            pts_2d = [(t[1].item() if hasattr(t[1], 'item') else t[1],
                       t[2].item() if hasattr(t[2], 'item') else t[2]) for t in track]
            frame_indices = [t[0].item() if hasattr(t[0], 'item') else t[0] for t in track]

        track_poses = [poses_w2c[i] for i in frame_indices]

        try:
            X = triangulate_dlt(pts_2d, track_poses, K_matrix, device=device)
        except Exception:
            continue

        X_3d = X[:3] / X[3]

        valid = True
        for pose in track_poses:
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose).float().to(device)
            elif pose.device != device:
                pose = pose.to(device)
            cam_coords = pose @ X
            if cam_coords[2] < min_depth:
                valid = False
                break

        if valid and X_3d[2] > 0:
            points_3d.append(X_3d)
            for t in track:
                if len(t) == 4:
                    frame_idx = t[0].item() if hasattr(t[0], 'item') else t[0]
                    u = t[2].item() if hasattr(t[2], 'item') else t[2]
                    v = t[3].item() if hasattr(t[3], 'item') else t[3]
                else:
                    frame_idx = t[0].item() if hasattr(t[0], 'item') else t[0]
                    u = t[1].item() if hasattr(t[1], 'item') else t[1]
                    v = t[2].item() if hasattr(t[2], 'item') else t[2]
                observations.append((len(points_3d) - 1, frame_idx, u, v))
            valid_track_indices.append(track_idx)

    if points_3d:
        return torch.stack(points_3d), observations, valid_track_indices
    else:
        return torch.zeros(0, 3, device=device, dtype=torch.float32), observations, []


def triangulate_from_graph_obs(edge_index, edge_attr, poses_c2w, K, device='cpu', min_depth=0.1):
    """
    Differentiable batched triangulation from GNN-predicted poses using graph observations.

    Re-triangulates 3D points from camera poses and observation edges.
    Used during training to compute reprojection loss with GNN-predicted poses.
    Fully differentiable - no .item() calls that break the computation graph.

    Args:
        edge_index: (2, M) tensor [cam_indices, point_indices] from ('camera', 'observes', 'point')
        edge_attr: (M, 2) tensor [u_norm, v_norm] normalized pixel coordinates
        poses_c2w: (N_cam, 4, 4) tensor of camera-to-world poses
        K: camera intrinsics dict with fx, fy, cx, cy
        device: torch device
        min_depth: minimum valid depth for cheirality check

    Returns:
        points_3d: (N_pts, 3) tensor of triangulated 3D points
    """
    if isinstance(K['fx'], torch.Tensor):
        fx, fy, cx, cy = K['fx'].item(), K['fy'].item(), K['cx'].item(), K['cy'].item()
    else:
        fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']

    img_width = edge_attr[:, 0].max().item() if edge_attr.numel() > 0 else 672
    img_height = edge_attr[:, 1].max().item() if edge_attr.numel() > 0 else 224

    u_all = edge_attr[:, 0] * img_width
    v_all = edge_attr[:, 1] * img_height

    cam_indices = edge_index[0]
    pt_indices = edge_index[1]

    N_pts = pt_indices.max().item() + 1 if pt_indices.numel() > 0 else 0

    if N_pts == 0:
        return torch.zeros(0, 3, device=device, dtype=torch.float32)

    K_matrix = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    K_inv = torch.inverse(K_matrix)

    points_3d_list = []

    for pt_idx in range(N_pts):
        mask = pt_indices == pt_idx
        n_obs = mask.sum().item()
        if n_obs < 2:
            continue

        u_obs = u_all[mask]
        v_obs = v_all[mask]
        cam_idx = cam_indices[mask].long()

        A_rows = []
        for j in range(n_obs):
            u, v = u_obs[j], v_obs[j]
            ci = cam_idx[j]
            P = poses_c2w[ci]

            pt = torch.stack([u, v, torch.tensor(1.0, device=device)])
            pt_norm = K_inv @ pt

            A_rows.append(pt_norm[0] * P[2, :] - P[0, :])
            A_rows.append(pt_norm[1] * P[2, :] - P[1, :])

        if len(A_rows) < 4:
            continue

        A = torch.stack(A_rows)
        _, _, Vt = torch.linalg.svd(A)
        X = Vt[-1]

        if X[3].abs() < 1e-10:
            continue

        X_3d = X[:3] / X[3]
        points_3d_list.append(X_3d)

    if len(points_3d_list) == 0:
        return torch.zeros(0, 3, device=device, dtype=torch.float32)

    return torch.stack(points_3d_list)


# =============================================================================
# Graph Construction
# =============================================================================

def build_heterogeneous_graph(num_cameras, points_3d, observations, img_height, img_width):
    """
    Build a PyTorch Geometric HeteroData graph for bundle adjustment.

    Graph structure:
    - Camera nodes: one per frame in window (stores 6-DoF pose as feature)
    - Point nodes: triangulated 3D points (stores 3D position as feature)
    - Edges: camera observes point (from triangulated tracks)

    Args:
        num_cameras: number of camera nodes (typically = window_size)
        points_3d: (N, 3) triangulated 3D points
        observations: (M, 4) array of [point_idx, frame_idx, u, v]
        img_height, img_width: image dimensions for normalizing pixel coords

    Returns:
        data: PyTorch Geometric HeteroData object
    """
    data = HeteroData()

    # Initialize node features with zeros (will be set by caller)
    data['camera'].x = torch.zeros(num_cameras, 6)  # 6 = euler(3) + trans(3)
    data['point'].x = points_3d.clone().detach() if isinstance(points_3d, torch.Tensor) else torch.tensor(points_3d, dtype=torch.float32)

    # Build edges: which camera observes which point
    cam_indices = []
    point_indices = []
    edge_features = []

    for point_idx, frame_idx, u, v in observations:
        cam_indices.append(frame_idx)
        point_indices.append(point_idx)
        # Normalize pixel coordinates to [0, 1] range
        u_norm = u / img_width
        v_norm = v / img_height
        edge_features.append([u_norm, v_norm])

    # Edge direction: ('camera', 'observes', 'point')
    data['camera', 'observes', 'point'].edge_index = torch.tensor([cam_indices, point_indices], dtype=torch.long)
    data['camera', 'observes', 'point'].edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return data


# =============================================================================
# Projection and Losses
# =============================================================================

def project_points_torch(points_3d, poses_c2w, K, device):
    """
    Project 3D points to 2D pixel coordinates using camera poses (camera-to-world) and intrinsics.

    Args:
        points_3d: (N, 3) tensor of 3D points in world frame
        poses_c2w: (M, 4, 4) tensor of camera poses (camera to world)
        K: intrinsics dict with fx, fy, cx, cy
        device: torch device

    Returns:
        proj_2d: (M, N, 2) tensor of projected pixel coordinates for each camera
    """
    N = points_3d.shape[0]
    M = poses_c2w.shape[0]

    if isinstance(K['fx'], torch.Tensor):
        fx, fy, cx, cy = K['fx'].item(), K['fy'].item(), K['cx'].item(), K['cy'].item()
    else:
        fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    K_matrix = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)

    # Convert points to homogeneous coordinates
    points_h = torch.cat([points_3d, torch.ones(N, 1, device=device)], dim=1)

    # Invert poses to get world-to-camera
    poses_w2c = torch.inverse(poses_c2w)

    # Transform points to camera frame: X_cam = P_w2c * X_world
    points_cam = torch.matmul(poses_w2c[:, :3, :], points_h.T)

    # Clamp depths to avoid division by zero
    depths = points_cam[:, 2, :].unsqueeze(1).clamp(min=0.1)

    # Project to pixel coordinates
    proj_h = torch.matmul(K_matrix, points_cam)
    proj_2d = proj_h[:, :2, :] / depths  # Divide by depth

    # Return shape: (M, N, 2) = (num_cameras, num_points, 2)
    return proj_2d.permute(0, 2, 1)


def huber_loss(errors, delta=1.0):
    """
    Huber loss (smooth L1 loss) - less sensitive to outliers than L2.

    For |error| <= delta: 0.5 * error^2
    For |error| > delta: delta * |error| - 0.5 * delta^2

    Args:
        errors: tensor of reprojection errors
        delta: threshold separating L2 and L1 regions

    Returns:
        tensor of Huber losses
    """
    abs_errors = torch.abs(errors)
    quadratic = torch.clamp(abs_errors, max=delta)
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def compute_reprojection_error(poses_c2w, points_3d, observations, K, img_height, img_width):
    """
    Compute RMS reprojection error in normalized coordinates [0,1].

    Args:
        poses_c2w: list or array of 4x4 camera poses (camera to world)
        points_3d: (N, 3) array of 3D points
        observations: (M, 4) array of [point_idx, frame_idx, u_obs, v_obs]
        K: camera intrinsics dict
        img_height, img_width: image dimensions for normalization

    Returns:
        RMS error in normalized coordinate space [0,1]
    """
    errors = []
    if isinstance(K['fx'], torch.Tensor):
        K_matrix = np.array([[K['fx'].item(), 0, K['cx'].item()], [0, K['fy'].item(), K['cy'].item()], [0, 0, 1]])
    else:
        K_matrix = np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]])

    # Convert c2w poses to w2c
    poses_w2c = []
    for pose in poses_c2w:
        if isinstance(pose, np.ndarray):
            poses_w2c.append(np.linalg.inv(pose))
        else:
            poses_w2c.append(torch.inverse(pose).cpu().numpy())

    for point_idx, frame_idx, u_obs, v_obs in observations:
        if point_idx >= len(points_3d):
            continue
        point_3d = points_3d[point_idx:point_idx+1]
        point_h = np.hstack([point_3d, np.ones((1, 1))])
        pose_w2c = poses_w2c[frame_idx]
        if isinstance(pose_w2c, np.ndarray):
            cam_coords = pose_w2c @ point_h.T
        else:
            cam_coords = pose_w2c.cpu().numpy() @ point_h.T
        if cam_coords[2] < 0.1:
            continue
        # Project: K * cam_coords -> pixel coordinates
        proj = K_matrix @ cam_coords[:3]
        proj = (proj[:2] / proj[2]).flatten()
        error = np.sqrt((proj[0] - u_obs)**2 + (proj[1] - v_obs)**2)
        errors.append(error)

    return np.sqrt(np.mean(np.array(errors)**2)) if errors else 0.0


# =============================================================================
# Post-processing and Trajectory Recovery
# =============================================================================

def post_processing(pred_poses, window_size, overlap):
    """
    Post-process predicted poses by averaging overlapping windows.

    Args:
        pred_poses: numpy array of shape (num_windows, window_size-1, 6)
        window_size: size of each window
        overlap: number of overlapping frames between consecutive windows

    Returns:
        numpy array of processed poses
    """
    if window_size == 2:
        pred_poses = pred_poses.squeeze(1)
        return np.asarray(pred_poses)

    num_batchs = pred_poses.shape[0]

    # get poses in overlaped frames
    q = queue.Queue(window_size - 1)
    idx = 0
    poses = []

    while not q.full():
        q.put(pred_poses[idx, :, :])
        idx = idx + 1

    while idx < num_batchs:
        # process first full queue
        if idx == (window_size - 1):
            poses.append(q.queue[0][0, :])

            # implemented for specific case window_size = 3 and overlap = 2
            avg_pose = (q.queue[0][1, :] + q.queue[1][0, :]) / 2
            poses.append(avg_pose)

            if window_size == 4:
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :]) / 3
                poses.append(avg_pose)

        elif idx < (num_batchs - 1):
            if window_size == 3:
                # implemented for specific case window_size = 3 and overlap = 2
                avg_pose = (q.queue[0][1, :] + q.queue[1][0, :]) / 2
                poses.append(avg_pose)

            elif window_size == 4:
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :]) / 3
                poses.append(avg_pose)

        # process last full queue (idx == num_batchs-1)
        else:
            if window_size == 3:
                # implemented for specific case window_size = 3 and overlap = 2
                poses.append(q.queue[1][1, :])

            elif window_size == 4:
                # implemented for specific case window_size = 4 and overlap = 2
                avg_pose = (q.queue[1][2, :] + q.queue[2][1, :]) / 2
                poses.append(avg_pose)
                poses.append(q.queue[2][2, :])

            idx = idx + 1

        # update queue
        if idx < (num_batchs - 1):
            idx = idx + 1
            first = q.get()  # dequeue first element
            q.put(pred_poses[idx, :, :])

    return np.asarray(poses)


def recover_trajectory_and_poses(poses, norm=True):
    """
    Recover absolute trajectory from relative poses.

    Args:
        poses: array of shape (N, 6) with [euler_angles, translation]
        norm: whether to undo normalization

    Returns:
        predicted_poses: list of 4x4 transformation matrices
        predicted_trajectory: list of translation vectors
    """
    predicted_poses = []
    predicted_trajectory = []

    # Undo normalization
    mean_angles = KITTI_MEAN_ANGLES
    std_angles = KITTI_STD_ANGLES
    mean_t = KITTI_MEAN_T
    std_t = KITTI_STD_T

    T = np.eye(4)  # Initialize T at the start

    for i in range(len(poses) - 1):
        angles = poses[i, :3]
        t = poses[i, 3:]

        if norm:
            euler = np.multiply(angles, std_angles) + mean_angles
            t = np.multiply(t, std_t) + mean_t
        else:
            euler = angles

        R = np.asarray(euler_to_rotation(euler, seq="zyx"))

        T_r = np.concatenate(
            (
                np.concatenate([R, np.reshape(t, (3, 1))], axis=1),
                [[0.0, 0.0, 0.0, 1.0]],
            ),
            axis=0,
        )
        T_abs = np.dot(T, T_r)
        T = T_abs

        predicted_poses.append(T)
        predicted_trajectory.append(T_abs[:3, 3])

    return predicted_poses, predicted_trajectory
