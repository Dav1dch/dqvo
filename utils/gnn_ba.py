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

from utils.dq import (
    matrix_to_dq_np,
    dq_to_matrix_np,
    dq_to_rt_np,
    dq_mult_np,
    dq_conj_np,
    dq_inverse_np,
    dq_mult_torch,
    dq_conj_torch,
    dq_inverse_torch,
    dq_transform_point_torch,
    dq_transform_points_batch_torch,
    dq_normalize_torch,
    dq_identity_np,
    dq_identity_torch,
    dq_extract_translation_torch,
    dq_geodesic_loss,
    dq_to_matrix_torch,
    _quat_to_rotmat_torch,
    _rotmat_to_quat_np,
    _quat_mult_np,
    dq_transform_point,
)


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


def pose_6dof_to_dq(euler_zyx, translation):
    """
    Convert denormalized 6-DOF [euler_z, euler_y, euler_x, tx, ty, tz] to dual quaternion.

    Args:
        euler_zyx: (3,) euler angles in radians, seq='zyx'
        translation: (3,) translation in meters

    Returns:
        dq: (8,) dual quaternion [qw,qx,qy,qz, q'w,q'x,q'y,q'z]
    """
    R = euler_to_rotation(euler_zyx, seq='zyx')
    q_r = _rotmat_to_quat_np(R)
    if q_r.ndim == 2:
        q_r = q_r[0]
    t_q = np.array([0.0, translation[0], translation[1], translation[2]])
    q_d = 0.5 * _quat_mult_np(t_q, q_r)
    return np.concatenate([q_r, q_d])


def poses_to_camera_features(poses):
    """
    Convert camera poses to DQ camera feature vectors (no normalization).

    Supports 4x4 transformation matrices as input.
    Returns absolute DQ poses for each frame.

    Args:
        poses: 4x4 transformation matrices (N, 4, 4) or list of 4x4, or DQ arrays

    Returns:
        camera_feats: (N, 8) array of DQ absolute poses
    """
    import numpy as np

    if isinstance(poses, np.ndarray):
        if poses.ndim == 2 and poses.shape[1] == 8:
            return poses  # Already DQ
        elif poses.ndim == 1 and poses.shape[0] == 8:
            return poses[np.newaxis, :]
        elif poses.ndim == 2 and poses.shape == (4, 4):
            poses = [poses]
        elif poses.ndim == 3 and poses.shape[1:] == (4, 4):
            poses = list(poses)
    elif isinstance(poses, list):
        pass

    camera_feats = []
    for pose in poses:
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        dq = matrix_to_dq_np(pose)
        camera_feats.append(dq)
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

def _precompute_camera_proj(poses_c2w_dq, K_matrix, device):
    """
    Pre-compute camera projection data from DQ poses.
    
    Returns:
        proj_rows_u: (M, 4) — constraint rows for u coordinate
        proj_rows_v: (M, 4) — constraint rows for v coordinate
        proj_rows_z: (M, 4) — R[2,:] + t[2] row
        poses_w2c: (M, 8) — world-to-camera DQs
        K_inv: (3, 3) — inverse intrinsics
    """
    if isinstance(poses_c2w_dq, np.ndarray):
        poses_c2w = torch.from_numpy(poses_c2w_dq).float().to(device)
    elif isinstance(poses_c2w_dq, list):
        poses_c2w = torch.tensor(np.stack(poses_c2w_dq), dtype=torch.float32, device=device)
    else:
        poses_c2w = poses_c2w_dq.to(device)
    
    M = poses_c2w.shape[0]
    poses_w2c = dq_inverse_torch(poses_c2w)
    q_r = poses_w2c[..., :4]
    
    R = _quat_to_rotmat_torch(q_r)  # (M, 3, 3)
    t = dq_extract_translation_torch(poses_w2c)  # (M, 3)
    
    fx, cx = K_matrix[0, 0], K_matrix[0, 2]
    fy, cy = K_matrix[1, 1], K_matrix[1, 2]
    
    proj_rows_u = torch.empty(M, 4, dtype=torch.float32, device=device)
    proj_rows_u[:, :3] = fx * R[:, 0, :] + cx * R[:, 2, :]
    proj_rows_u[:, 3] = fx * t[:, 0] + cx * t[:, 2]
    
    proj_rows_v = torch.empty(M, 4, dtype=torch.float32, device=device)
    proj_rows_v[:, :3] = fy * R[:, 1, :] + cy * R[:, 2, :]
    proj_rows_v[:, 3] = fy * t[:, 1] + cy * t[:, 2]
    
    proj_rows_z = torch.empty(M, 4, dtype=torch.float32, device=device)
    proj_rows_z[:, :3] = R[:, 2, :]
    proj_rows_z[:, 3] = t[:, 2]
    
    K_inv = torch.inverse(K_matrix)
    
    return proj_rows_u, proj_rows_v, proj_rows_z, poses_w2c, K_inv


def _dq_c2w_to_w2c_rt_torch(dq, device=None):
    """
    Extract world-to-camera rotation and translation from camera-to-world DQ.
    """
    d_w2c = dq_inverse_torch(dq)
    q_r = d_w2c[..., :4]
    q_d = d_w2c[..., 4:]
    R = _quat_to_rotmat_torch(q_r)
    t = dq_extract_translation_torch(d_w2c)
    return R, t


def triangulate_dlt(points_2d, poses_dq, K, device='cpu'):
    """
    Triangulate a 3D point from 2D observations using DLT.
    """
    A = []
    for (u, v), dq in zip(points_2d, poses_dq):
        if isinstance(dq, np.ndarray):
            dq = torch.from_numpy(dq).float().to(K.device)
        elif dq.device != K.device:
            dq = dq.to(K.device)
        R_w2c, t_w2c = _dq_c2w_to_w2c_rt_torch(dq.unsqueeze(0))
        R_w2c, t_w2c = R_w2c.squeeze(0), t_w2c.squeeze(0)
        pt = torch.tensor([u, v, 1.0], dtype=torch.float32, device=K.device)
        K_inv = torch.inverse(K)
        pt_norm = K_inv @ pt
        P_row0 = K[0, 0] * R_w2c[0, :] + K[0, 2] * R_w2c[2, :]
        P_row1 = K[1, 1] * R_w2c[1, :] + K[1, 2] * R_w2c[2, :]
        P_row2 = R_w2c[2, :]
        p0 = K[0, 0] * t_w2c[0] + K[0, 2] * t_w2c[2]
        p1 = K[1, 1] * t_w2c[1] + K[1, 2] * t_w2c[2]
        p2 = t_w2c[2]
        row0 = torch.empty(4, dtype=torch.float32, device=K.device)
        row0[:3] = P_row0; row0[3] = p0
        row1 = torch.empty(4, dtype=torch.float32, device=K.device)
        row1[:3] = P_row1; row1[3] = p1
        row2 = torch.empty(4, dtype=torch.float32, device=K.device)
        row2[:3] = P_row2; row2[3] = p2
        A.append(pt_norm[0] * row2 - row0)
        A.append(pt_norm[1] * row2 - row1)
    A = torch.stack(A)
    _, _, Vt = torch.linalg.svd(A)
    X = Vt[-1]
    return X / X[3]


def _triangulate_track_gpu(frame_indices, pts_2d, proj_rows_u, proj_rows_v, proj_rows_z, K_inv):
    """
    GPU-optimised DLT for a single track using pre-computed camera projection data.
    
    Args:
        frame_indices: (K,) int tensor of camera indices
        pts_2d: (K, 2) float tensor of (u, v) pixel coordinates
        proj_rows_*: (M, 4) pre-computed projection rows K@[R|t] for all cameras
    Returns:
        X_3d: (3,) triangulated 3D point
    """
    K = frame_indices.shape[0]
    
    # Index pre-computed projection rows
    p_u = proj_rows_u[frame_indices]  # (K, 4) — K@[R|t] row 0
    p_v = proj_rows_v[frame_indices]  # (K, 4) — K@[R|t] row 1
    p_z = proj_rows_z[frame_indices]  # (K, 4) — K@[R|t] row 2 = [R[2], t[2]]
    
    # Build A matrix: (2*K, 4)
    # Constraint: u * P_z - P_u = 0,  v * P_z - P_v = 0
    A = torch.empty(2 * K, 4, dtype=pts_2d.dtype, device=pts_2d.device)
    A[0::2] = pts_2d[:, 0:1] * p_z - p_u
    A[1::2] = pts_2d[:, 1:2] * p_z - p_v
    
    _, _, Vt = torch.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_all_points(tracks, poses_c2w_dq, K, min_depth=0.1, device='cpu'):
    """
    GPU-optimised triangulation of all 3D points from tracks.
    
    Pre-computes camera projection data once, then runs per-track DLT
    with GPU tensor indexing and batched cheirality checks.
    """
    if isinstance(K['fx'], torch.Tensor):
        K_matrix = torch.tensor([[K['fx'].item(), 0, K['cx'].item()],
                                  [0, K['fy'].item(), K['cy'].item()],
                                  [0, 0, 1]], dtype=torch.float32, device=device)
    else:
        K_matrix = torch.tensor([[K['fx'], 0, K['cx']],
                                  [0, K['fy'], K['cy']],
                                  [0, 0, 1]], dtype=torch.float32, device=device)

    proj_rows_u, proj_rows_v, proj_rows_z, poses_w2c, K_inv = \
        _precompute_camera_proj(poses_c2w_dq, K_matrix, device)
    
    M = poses_w2c.shape[0]
    points_3d = []
    observations = []
    valid_track_indices = []
    
    # Pre-convert all track data to GPU tensors for fast processing
    for track_idx, track in enumerate(tracks):
        if len(track) < 2:
            continue
        
        if len(track[0]) == 4:
            frame_idx_list = [t[0].item() if hasattr(t[0], 'item') else t[0] for t in track]
            u_list = [t[2].item() if hasattr(t[2], 'item') else t[2] for t in track]
            v_list = [t[3].item() if hasattr(t[3], 'item') else t[3] for t in track]
        else:
            frame_idx_list = [t[0].item() if hasattr(t[0], 'item') else t[0] for t in track]
            u_list = [t[1].item() if hasattr(t[1], 'item') else t[1] for t in track]
            v_list = [t[2].item() if hasattr(t[2], 'item') else t[2] for t in track]
        
        frame_indices = torch.tensor(frame_idx_list, dtype=torch.long, device=device)
        pts_2d = torch.tensor([[u, v] for u, v in zip(u_list, v_list)], dtype=torch.float32, device=device)
        
        try:
            X_3d = _triangulate_track_gpu(frame_indices, pts_2d, proj_rows_u, proj_rows_v, proj_rows_z, K_inv)
        except Exception:
            continue
        
        # Batch cheirality check: transform point through all relevant cameras at once
        dqs_cam = poses_w2c[frame_indices]  # (K, 8)
        X_expanded = X_3d.unsqueeze(0).expand(frame_indices.shape[0], -1)  # (K, 3)
        X_cam = dq_transform_point_torch(dqs_cam, X_expanded)  # (K, 3) — uses broadcasting
        depths = X_cam[:, 2]
        if (depths < min_depth).any():
            continue
        
        if X_3d[2] <= 0:
            continue
        
        points_3d.append(X_3d)
        for t in track:
            if len(t) == 4:
                fi = t[0].item() if hasattr(t[0], 'item') else t[0]
                u = t[2].item() if hasattr(t[2], 'item') else t[2]
                v = t[3].item() if hasattr(t[3], 'item') else t[3]
            else:
                fi = t[0].item() if hasattr(t[0], 'item') else t[0]
                u = t[1].item() if hasattr(t[1], 'item') else t[1]
                v = t[2].item() if hasattr(t[2], 'item') else t[2]
            observations.append((len(points_3d) - 1, fi, u, v))
        valid_track_indices.append(track_idx)
    
    if points_3d:
        return torch.stack(points_3d), observations, valid_track_indices
    else:
        return torch.zeros(0, 3, device=device, dtype=torch.float32), observations, []


def triangulate_tracks_no_filter(tracks, track_indices, poses_c2w_dq, K, device='cpu'):
    """
    GPU-optimised triangulation for specific tracks WITHOUT cheirality filtering.
    Uses pre-computed camera projection data.
    """
    if isinstance(K['fx'], torch.Tensor):
        K_matrix = torch.tensor([[K['fx'].item(), 0, K['cx'].item()],
                                  [0, K['fy'].item(), K['cy'].item()],
                                  [0, 0, 1]], dtype=torch.float32, device=device)
    else:
        K_matrix = torch.tensor([[K['fx'], 0, K['cx']],
                                  [0, K['fy'], K['cy']],
                                  [0, 0, 1]], dtype=torch.float32, device=device)

    proj_rows_u, proj_rows_v, proj_rows_z, _, K_inv = \
        _precompute_camera_proj(poses_c2w_dq, K_matrix, device)

    points_3d = []
    for track_idx in track_indices:
        track = tracks[track_idx]
        if len(track[0]) == 4:
            fi = [t[0].item() if hasattr(t[0], 'item') else t[0] for t in track]
            u_list = [t[2].item() if hasattr(t[2], 'item') else t[2] for t in track]
            v_list = [t[3].item() if hasattr(t[3], 'item') else t[3] for t in track]
        else:
            fi = [t[0].item() if hasattr(t[0], 'item') else t[0] for t in track]
            u_list = [t[1].item() if hasattr(t[1], 'item') else t[1] for t in track]
            v_list = [t[2].item() if hasattr(t[2], 'item') else t[2] for t in track]

        frame_indices = torch.tensor(fi, dtype=torch.long, device=device)
        pts_2d = torch.tensor([[u, v] for u, v in zip(u_list, v_list)], dtype=torch.float32, device=device)

        try:
            X_3d = _triangulate_track_gpu(frame_indices, pts_2d, proj_rows_u, proj_rows_v, proj_rows_z, K_inv)
            points_3d.append(X_3d)
        except Exception:
            points_3d.append(torch.zeros(3, dtype=torch.float32, device=device))

    return torch.stack(points_3d)


def triangulate_from_graph_obs(edge_index, edge_attr, poses_c2w_dq, K, device='cpu', min_depth=0.1):
    """
    Differentiable batched triangulation from GNN-predicted poses using graph observations.

    Args:
        edge_index: (2, M) tensor [cam_indices, point_indices] from ('camera', 'observes', 'point')
        edge_attr: (M, 2) tensor [u_norm, v_norm] normalized pixel coordinates
        poses_c2w_dq: (N_cam, 8) tensor of camera-to-world DQ poses
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
            dq = poses_c2w_dq[ci]
            R_w2c, t_w2c = _dq_c2w_to_w2c_rt_torch(dq.unsqueeze(0))
            R_w2c, t_w2c = R_w2c.squeeze(0), t_w2c.squeeze(0)

            P_row0 = fx * R_w2c[0, :] + cx * R_w2c[2, :]
            P_row1 = fy * R_w2c[1, :] + cy * R_w2c[2, :]
            P_row2 = R_w2c[2, :]
            p0 = fx * t_w2c[0] + cx * t_w2c[2]
            p1 = fy * t_w2c[1] + cy * t_w2c[2]
            p2 = t_w2c[2]

            # Build 4-vector rows: u * P_z - P_u = 0, v * P_z - P_v = 0
            A_rows.append(u * torch.cat([P_row2, p2.unsqueeze(0)]) - torch.cat([P_row0, p0.unsqueeze(0)]))
            A_rows.append(v * torch.cat([P_row2, p2.unsqueeze(0)]) - torch.cat([P_row1, p1.unsqueeze(0)]))

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
    data['camera'].x = torch.zeros(num_cameras, 8)  # 8 = dual quaternion
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

def project_points_torch(points_3d, poses_c2w_dq, K, device):
    """
    Project 3D points to 2D pixel coordinates using DQ camera poses and intrinsics.
    Fully batched over both cameras and points.

    Args:
        points_3d: (N, 3) tensor of 3D points in world frame
        poses_c2w_dq: (M, 8) tensor of camera-to-world DQ poses
        K: intrinsics dict with fx, fy, cx, cy
        device: torch device

    Returns:
        proj_2d: (M, N, 2) tensor of projected pixel coordinates for each camera
    """
    N = points_3d.shape[0]
    M = poses_c2w_dq.shape[0]

    if isinstance(K['fx'], torch.Tensor):
        fx, fy, cx, cy = K['fx'].item(), K['fy'].item(), K['cx'].item(), K['cy'].item()
    else:
        fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    K_matrix = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)

    poses_w2c_dq = dq_inverse_torch(poses_c2w_dq)  # (M, 8)
    X_cam = dq_transform_points_batch_torch(poses_w2c_dq, points_3d)  # (M, N, 3)

    depths = X_cam[..., 2].clamp(min=0.1)  # (M, N)
    proj_h = torch.einsum('ij,mnj->mni', K_matrix, X_cam)  # (M, N, 3)
    proj_2d = proj_h[..., :2] / depths.unsqueeze(-1)  # (M, N, 2)

    return proj_2d


def get_obs_depths_batch(abs_dq_c2w, points_3d, cam_indices, pt_indices):
    """
    Compute depths for observation edges in a fully batched way.

    Args:
        abs_dq_c2w: (M, 8) absolute camera-to-world DQ poses
        points_3d: (N, 3) 3D points
        cam_indices: (E,) camera indices per observation
        pt_indices: (E,) point indices per observation

    Returns:
        obs_depths: (E,) tensor of depths at each observation
    """
    dqs_w2c = dq_inverse_torch(abs_dq_c2w)  # (M, 8)
    X_cam_all = dq_transform_points_batch_torch(dqs_w2c, points_3d)  # (M, N, 3)
    obs_depths = X_cam_all[cam_indices, pt_indices, 2]  # (E,)
    return obs_depths


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


def compute_reprojection_error(poses_c2w_dq, points_3d, observations, K, img_height, img_width):
    """
    Compute RMS reprojection error in normalized coordinates [0,1].

    Args:
        poses_c2w_dq: list or array of (8,) DQ camera poses (camera to world)
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

    # Convert c2w DQ to w2c
    poses_w2c_dq = []
    for dq in poses_c2w_dq:
        if isinstance(dq, torch.Tensor):
            dq = dq.cpu().numpy()
        if dq.size < 8 or np.any(np.isnan(dq)):
            poses_w2c_dq.append(dq_identity_np())  # placeholder
        else:
            poses_w2c_dq.append(dq_inverse_np(dq))

    for point_idx, frame_idx, u_obs, v_obs in observations:
        if point_idx >= len(points_3d):
            continue
        point_3d = points_3d[point_idx]
        d_w2c = poses_w2c_dq[frame_idx]
        X_cam_np = dq_transform_point(d_w2c, point_3d)
        if X_cam_np[2] < 0.1:
            continue
        proj = K_matrix @ np.array([X_cam_np[0], X_cam_np[1], X_cam_np[2], 1.0])[:3]
        proj = (proj[:2] / proj[2])
        error = np.sqrt((proj[0] - u_obs)**2 + (proj[1] - v_obs)**2)
        errors.append(error)

    return np.sqrt(np.mean(np.array(errors)**2)) if errors else 0.0


# =============================================================================
# Post-processing and Trajectory Recovery
# =============================================================================

def post_processing(pred_poses, window_size, overlap):
    """
    Post-process predicted relative poses by averaging overlapping windows.

    Args:
        pred_poses: numpy array of shape (num_windows, window_size-1, D) where D is feature dim (6 or 8)
        window_size: size of each window
        overlap: number of overlapping frames between consecutive windows

    Returns:
        numpy array of processed poses shape (N, D)
    """
    if window_size == 2:
        pred_poses = pred_poses.squeeze(1)
        return np.asarray(pred_poses)

    is_dq = pred_poses.shape[-1] == 8

    def _avg(pose_list):
        if is_dq:
            return _dq_average_np(list(pose_list))
        return sum(pose_list) / len(pose_list)

    num_batchs = pred_poses.shape[0]
    q = queue.Queue(window_size - 1)
    idx = 0
    poses = []

    while not q.full():
        q.put(pred_poses[idx, :, :])
        idx = idx + 1

    while idx < num_batchs:
        if idx == (window_size - 1):
            poses.append(q.queue[0][0, :])
            poses.append(_avg([q.queue[0][1, :], q.queue[1][0, :]]))
            if window_size == 4:
                poses.append(_avg([q.queue[0][2, :], q.queue[1][1, :], q.queue[2][0, :]]))
        elif idx < (num_batchs - 1):
            if window_size == 3:
                poses.append(_avg([q.queue[0][1, :], q.queue[1][0, :]]))
            elif window_size == 4:
                poses.append(_avg([q.queue[0][2, :], q.queue[1][1, :], q.queue[2][0, :]]))
        else:
            if window_size == 3:
                poses.append(q.queue[1][1, :])
            elif window_size == 4:
                poses.append(_avg([q.queue[1][2, :], q.queue[2][1, :]]))
                poses.append(q.queue[2][2, :])
            idx = idx + 1

        if idx < (num_batchs - 1):
            idx = idx + 1
            first = q.get()
            q.put(pred_poses[idx, :, :])

    return np.asarray(poses)


def recover_trajectory_and_poses(poses, use_dq=False, norm=True):
    """
    Recover absolute trajectory from relative poses.

    Args:
        poses: array of shape (N, D)
               D=6: [euler_angles(3), translation(3)] (legacy 6-DOF mode)
               D=8: dual quaternion [qw,qx,qy,qz, q'w,q'x,q'y,q'z] (DQ mode, use_dq=True)
        use_dq: if True, poses are DQ format and trajectory is composed via DQ multiplication

    Returns:
        predicted_poses: list of 4x4 transformation matrices
        predicted_trajectory: list of translation vectors
    """
    predicted_poses = []
    predicted_trajectory = []

    if use_dq:
        d_abs = dq_identity_np()
        for i in range(len(poses)):
            d_rel = _dq_normalize_np(poses[i])  # normalize before composing
            d_abs = dq_mult_np(d_abs, d_rel)
            d_abs = _dq_normalize_np(d_abs)
            T = dq_to_matrix_np(d_abs)
            predicted_poses.append(T)
            predicted_trajectory.append(T[:3, 3])
        return predicted_poses, predicted_trajectory

    # Legacy 6-DOF mode
    mean_angles = KITTI_MEAN_ANGLES
    std_angles = KITTI_STD_ANGLES
    mean_t = KITTI_MEAN_T
    std_t = KITTI_STD_T

    T = np.eye(4)
    for i in range(len(poses)):
        angles = poses[i, :3]
        t = poses[i, 3:]

        if norm:
            euler = np.multiply(angles, std_angles) + mean_angles
            t = np.multiply(t, std_t) + mean_t
        else:
            euler = angles

        R = np.asarray(euler_to_rotation(euler, seq="zyx"))
        T_r = np.concatenate(
            (np.concatenate([R, np.reshape(t, (3, 1))], axis=1),
             [[0.0, 0.0, 0.0, 1.0]]),
            axis=0,
        )
        T_abs = np.dot(T, T_r)
        T = T_abs

        predicted_poses.append(T)
        predicted_trajectory.append(T_abs[:3, 3])

    return predicted_poses, predicted_trajectory


def _dq_normalize_np(d):
    """Numpy version of DQ normalization: unit quaternion + orthogonal dual."""
    q_r = d[:4]
    q_d = d[4:]
    q_norm = np.linalg.norm(q_r)
    if q_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_r_norm = q_r / q_norm
    dot = np.dot(q_d, q_r_norm)
    q_d_orth = q_d - dot * q_r_norm
    return np.concatenate([q_r_norm, q_d_orth])


def _dq_average_np(dq_list):
    """
    Average multiple DQs properly on the SE(3) manifold.
    Extracts R,t from each, averages in log-space, re-encodes.
    """
    if len(dq_list) == 1:
        return dq_list[0]
    
    # Filter out NaN entries
    valid = [dq for dq in dq_list if not np.any(np.isnan(dq))]
    if len(valid) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if len(valid) == 1:
        return valid[0]
    
    # Extract R and t from each valid DQ
    Rs, ts = [], []
    for dq in valid:
        dq_n = _dq_normalize_np(dq)
        R, t = dq_to_rt_np(dq_n)
        Rs.append(R)
        ts.append(t)
    
    # Average rotation: use scipy's Rotation.mean() on rotation vectors
    from scipy.spatial.transform import Rotation
    rots = Rotation.from_matrix(Rs)
    mean_rot = rots.mean()
    R_avg = mean_rot.as_matrix()
    
    # Average translation: simple mean
    t_avg = np.mean(ts, axis=0)
    
    # Build DQ from averaged R,t
    q_r = _rotmat_to_quat_np(R_avg)
    if q_r.ndim == 2:
        q_r = q_r[0]
    t_q = np.array([0.0, t_avg[0], t_avg[1], t_avg[2]])
    q_d = 0.5 * _quat_mult_np(t_q, q_r)
    return np.concatenate([q_r, q_d])
    """Numpy version of DQ normalization: unit quaternion + orthogonal dual."""
    q_r = d[:4]
    q_d = d[4:]
    q_r_norm = q_r / (np.linalg.norm(q_r) + 1e-10)
    dot = np.dot(q_d, q_r_norm)
    q_d_orth = q_d - dot * q_r_norm
    return np.concatenate([q_r_norm, q_d_orth])
