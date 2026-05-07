"""
Dual Quaternion utilities for SE(3) pose representation.

Convention: d = q_r + ε q_d
  - q_r: unit quaternion [w, x, y, z] representing the rotation
  - q_d: dual part = 0.5 * t_q ⊗ q_r  where t_q = [0, tx, ty, tz]
  - DQ stored as [qw, qx, qy, qz, q'w, q'x, q'y, q'z] (8 elements)

Point transformation: X' = d ⊗ [0, X] ⊗ d*
  where d* = q_r* - ε q_d* is the DQ conjugate (= inverse for unit DQ)

Translation extraction: t = 2 * vector_part(q_d ⊗ q_r*)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


# =============================================================================
# Numpy helpers
# =============================================================================

def _quat_mult_np(q1, q2):
    """Quaternion multiplication: q = q1 ⊗ q2."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def _quat_conj_np(q):
    """Quaternion conjugate: q* = [w, -x, -y, -z]."""
    c = q.copy()
    c[..., 1:] = -c[..., 1:]
    return c


def _rotmat_to_quat_np(R):
    """Rotation matrix (3x3) to unit quaternion [w, x, y, z]."""
    rot = Rotation.from_matrix(np.atleast_2d(R))
    q = rot.as_quat()  # scipy returns [x, y, z, w]
    if q.ndim == 1:
        q = np.array([q[3], q[0], q[1], q[2]])
    else:
        q = q[:, [3, 0, 1, 2]]
    return q


def _quat_to_rotmat_np(q):
    """Unit quaternion [w, x, y, z] to rotation matrix (3x3)."""
    q_scipy = np.atleast_2d(q)[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(q_scipy.squeeze())
    return rot.as_matrix()


# =============================================================================
# NumPy DQ functions
# =============================================================================

def matrix_to_dq_np(T):
    """Convert 4x4 transform to dual quaternion [qw,qx,qy,qz, q'w,q'x,q'y,q'z]."""
    R = T[:3, :3]
    t = T[:3, 3]
    q_r = _rotmat_to_quat_np(R)
    if q_r.ndim == 2:
        q_r = q_r[0]
    t_q = np.array([0.0, t[0], t[1], t[2]])
    q_d = 0.5 * _quat_mult_np(t_q, q_r)
    return np.concatenate([q_r, q_d])


def dq_to_matrix_np(d):
    """Convert dual quaternion [qw,qx,qy,qz, q'w,q'x,q'y,q'z] to 4x4 transform."""
    q_r = d[:4]
    q_d = d[4:]
    R = _quat_to_rotmat_np(q_r)
    t = 2.0 * _quat_mult_np(q_d, _quat_conj_np(q_r))[1:]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def dq_to_rt_np(d):
    """DQ to (R_3x3, t_3) tuple."""
    q_r = d[:4]
    q_d = d[4:]
    R = _quat_to_rotmat_np(q_r)
    t = 2.0 * _quat_mult_np(q_d, _quat_conj_np(q_r))[1:]
    return R, t


def dq_mult_np(d1, d2):
    """DQ multiplication: d_out = d1 ⊗ d2."""
    q1_r, q1_d = d1[..., :4], d1[..., 4:]
    q2_r, q2_d = d2[..., :4], d2[..., 4:]
    out_r = _quat_mult_np(q1_r, q2_r)
    out_d = _quat_mult_np(q1_r, q2_d) + _quat_mult_np(q1_d, q2_r)
    return np.concatenate([out_r, out_d], axis=-1)


def dq_conj_np(d):
    """NumPy DQ conjugate (NOT the SE(3) inverse)."""
    q_r = d[..., :4]
    q_d = d[..., 4:]
    conj_r = _quat_conj_np(q_r)
    conj_d = -_quat_conj_np(q_d)
    return np.concatenate([conj_r, conj_d], axis=-1)


def dq_inverse_np(d):
    """
    NumPy DQ inverse (= inverse SE(3) transform).
    Extracts R,t, inverts, re-encodes using pure numpy.
    """
    d = np.asarray(d)
    if d.ndim > 1 and d.shape[0] > 0 and d.shape[-1] == 8:
        return np.array([dq_inverse_np(dd) for dd in d])
    if d.size < 8:
        return dq_identity_np()
    
    R, t = dq_to_rt_np(d)
    R_inv = R.T
    t_inv = -R_inv @ t
    return dq_from_rt_np(R_inv, t_inv)


def dq_from_rt_np(R, t):
    """
    Build dual quaternion from rotation matrix R (3,3) and translation t (3,).
    Returns (8,) DQ.
    """
    q_r = _rotmat_to_quat_np(R)
    if q_r.ndim == 2:
        q_r = q_r[0]
    t_q = np.array([0.0, float(t[0]), float(t[1]), float(t[2])])
    q_d = 0.5 * _quat_mult_np(t_q, q_r)
    return np.concatenate([q_r, q_d])


def _quat_from_rotmat_torch(R):
    """Rotation matrix (..., 3, 3) to unit quaternion [..., w,x,y,z]."""
    # Shepperd's method, adapted from dq_from_rt_torch
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    qw = torch.zeros_like(trace)
    qx = torch.zeros_like(trace)
    qy = torch.zeros_like(trace)
    qz = torch.zeros_like(trace)
    mask = trace > 0
    if mask.any():
        s = 0.5 / torch.sqrt(trace[mask] + 1.0)
        qw[mask] = 0.25 / s
        qx[mask] = (m21[mask] - m12[mask]) * s
        qy[mask] = (m02[mask] - m20[mask]) * s
        qz[mask] = (m10[mask] - m01[mask]) * s
    mask = (m00 > m11) & (m00 > m22) & ~(trace > 0)
    if mask.any():
        s = 2.0 * torch.sqrt(1.0 + m00[mask] - m11[mask] - m22[mask])
        qw[mask] = (m21[mask] - m12[mask]) / s
        qx[mask] = 0.25 * s
        qy[mask] = (m01[mask] + m10[mask]) / s
        qz[mask] = (m02[mask] + m20[mask]) / s
    mask = (m11 > m00) & (m11 > m22) & ~(trace > 0)
    if mask.any():
        s = 2.0 * torch.sqrt(1.0 + m11[mask] - m00[mask] - m22[mask])
        qw[mask] = (m02[mask] - m20[mask]) / s
        qx[mask] = (m01[mask] + m10[mask]) / s
        qy[mask] = 0.25 * s
        qz[mask] = (m12[mask] + m21[mask]) / s
    mask = (m22 > m00) & (m22 > m11) & ~(trace > 0)
    if mask.any():
        s = 2.0 * torch.sqrt(1.0 + m22[mask] - m00[mask] - m11[mask])
        qw[mask] = (m10[mask] - m01[mask]) / s
        qx[mask] = (m02[mask] + m20[mask]) / s
        qy[mask] = (m12[mask] + m21[mask]) / s
        qz[mask] = 0.25 * s
    return torch.stack([qw, qx, qy, qz], dim=-1)


def dq_transform_point_np(d, X):
    """
    Transform 3D point(s) via dual quaternion.
    Extracts R and t from DQ, then does R @ X + t.
    """
    q_r = d[..., :4]
    q_d = d[..., 4:]
    R = _quat_to_rotmat_np(q_r)
    t = 2.0 * _quat_mult_np(q_d, _quat_conj_np(q_r))[..., 1:]
    if R.ndim == 2 and X.ndim == 1:
        return R @ X + t
    return (R @ X[..., np.newaxis])[..., 0] + t


def dq_identity_np(batch=()):
    """Identity DQ: [1, 0, 0, 0, 0, 0, 0, 0] with optional batch dims."""
    d = np.zeros(batch + (8,))
    d[..., 0] = 1.0
    return d


# =============================================================================
# PyTorch helpers
# =============================================================================

def _quat_mult_torch(q1, q2):
    """PyTorch quaternion multiplication."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_conj_torch(q):
    """PyTorch quaternion conjugate."""
    c = q.clone()
    c[..., 1:] = -c[..., 1:]
    return c


def _quat_to_rotmat_torch(q):
    """
    PyTorch unit quaternion [..., w,x,y,z] to 3x3 rotation matrix.
    Handles shapes (4,), (N,4), (B,N,4) etc.
    """
    # Ensure at least 2D: (N, 4)
    orig_shape = q.shape
    if q.dim() == 1:
        q = q.unsqueeze(0)  # (1, 4)
    elif q.dim() > 2:
        q = q.reshape(-1, 4)

    N = q.shape[0]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(N, 3, 3, dtype=q.dtype, device=q.device)
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R[:, 0, 0] = w2 + x2 - y2 - z2
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = w2 - x2 + y2 - z2
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = w2 - x2 - y2 + z2

    # Restore batch dimensions
    if len(orig_shape) == 1:
        R = R.squeeze(0)
    elif len(orig_shape) > 2:
        R = R.reshape(*orig_shape[:-1], 3, 3)
    return R


# =============================================================================
# PyTorch DQ functions
# =============================================================================

def dq_mult_torch(d1, d2):
    """PyTorch DQ multiplication: d_out = d1 ⊗ d2."""
    q1_r, q1_d = d1[..., :4], d1[..., 4:]
    q2_r, q2_d = d2[..., :4], d2[..., 4:]
    out_r = _quat_mult_torch(q1_r, q2_r)
    out_d = _quat_mult_torch(q1_r, q2_d) + _quat_mult_torch(q1_d, q2_r)
    return torch.cat([out_r, out_d], dim=-1)


def dq_conj_torch(d):
    """PyTorch DQ conjugate (NOT inverse for SE(3) DQs)."""
    q_r = d[..., :4]
    q_d = d[..., 4:]
    conj_r = _quat_conj_torch(q_r)
    conj_d = -_quat_conj_torch(q_d)
    return torch.cat([conj_r, conj_d], dim=-1)


def dq_inverse_torch(d):
    """
    PyTorch DQ inverse (= inverse SE(3) transform).
    
    For d = q_r + ε (½ t ⊗ q_r) representing T = [R | t]:
    d⁻¹ = q_r* + ε (½ (-R⁻¹ t) ⊗ q_r*)
    which is computed by extracting R,t, inverting, and re-encoding.
    """
    q_r = d[..., :4]
    q_d = d[..., 4:]
    R = _quat_to_rotmat_torch(q_r)
    t = 2.0 * _quat_mult_torch(q_d, _quat_conj_torch(q_r))[..., 1:]
    R_inv = R.transpose(-2, -1)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    return dq_from_rt_torch(R_inv, t_inv)


def dq_from_rt_torch(R, t):
    """
    Build dual quaternion from rotation matrix R and translation t.
    R: (..., 3, 3), t: (..., 3)
    Returns: (..., 8) DQ
    """
    q_r = _quat_from_rotmat_torch(R)
    t_q = torch.zeros(*t.shape[:-1], 4, dtype=t.dtype, device=t.device)
    t_q[..., 1:] = t
    q_d = 0.5 * _quat_mult_torch(t_q, q_r)
    return torch.cat([q_r, q_d], dim=-1)


def _quat_from_rotmat_torch(R):
    """Rotation matrix (..., 3, 3) to unit quaternion [..., w,x,y,z].
    Falls back to identity [1,0,0,0] for degenerate matrices."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    qw = torch.zeros_like(trace)
    qx = torch.zeros_like(trace)
    qy = torch.zeros_like(trace)
    qz = torch.zeros_like(trace)

    # Case: trace > 0
    mask = trace > 0
    if mask.any():
        s = 0.5 / torch.sqrt(trace[mask] + 1.0)
        qw[mask] = 0.25 / s
        qx[mask] = (m21[mask] - m12[mask]) * s
        qy[mask] = (m02[mask] - m20[mask]) * s
        qz[mask] = (m10[mask] - m01[mask]) * s

    # Case: m00 is largest diagonal
    mask = (m00 >= m11) & (m00 >= m22) & ~(trace > 0) & (trace > -1)
    if mask.any():
        s = 2.0 * torch.sqrt(1.0 + m00[mask] - m11[mask] - m22[mask])
        qw[mask] = (m21[mask] - m12[mask]) / s
        qx[mask] = 0.25 * s
        qy[mask] = (m01[mask] + m10[mask]) / s
        qz[mask] = (m02[mask] + m20[mask]) / s

    # Case: m11 is largest diagonal
    mask = (m11 >= m00) & (m11 >= m22) & ~(trace > 0) & ~((m00 >= m11) & (m00 >= m22) & (trace > -1))
    if mask.any():
        s = 2.0 * torch.sqrt(1.0 + m11[mask] - m00[mask] - m22[mask])
        qw[mask] = (m02[mask] - m20[mask]) / s
        qx[mask] = (m01[mask] + m10[mask]) / s
        qy[mask] = 0.25 * s
        qz[mask] = (m12[mask] + m21[mask]) / s

    # Case: m22 is largest (or all zero — degenerate, fallback to identity)
    mask = ~(trace > 0) & ~((m00 >= m11) & (m00 >= m22) & (trace > -1)) & ~((m11 >= m00) & (m11 >= m22) & (trace > -1))
    if mask.any():
        s = 2.0 * torch.sqrt(1.0 + m22[mask] - m00[mask] - m11[mask] + 1e-10)
        qw[mask] = (m10[mask] - m01[mask]) / (s + 1e-10)
        qx[mask] = (m02[mask] + m20[mask]) / (s + 1e-10)
        qy[mask] = (m12[mask] + m21[mask]) / (s + 1e-10)
        qz[mask] = 0.25 * s

    # Fallback: set zero quaternions to identity
    norm_sq = qw*qw + qx*qx + qy*qy + qz*qz
    zero_mask = norm_sq < 1e-10
    qw[zero_mask] = 1.0
    qx[zero_mask] = 0.0
    qy[zero_mask] = 0.0
    qz[zero_mask] = 0.0

    q_r = torch.stack([qw, qx, qy, qz], dim=-1)
    return F.normalize(q_r, p=2, dim=-1)


def dq_normalize_torch(d):
    """
    Project an 8-D vector to the valid DQ manifold.
    Normalizes real part to unit quaternion, orthogonalizes dual part.
    Returns identity DQ if input has near-zero real part.
    """
    q_r = d[..., :4]
    q_d = d[..., 4:]
    q_r_norm = F.normalize(q_r, p=2, dim=-1, eps=1e-10)
    dot = (q_d * q_r_norm).sum(dim=-1, keepdim=True)
    q_d_orth = q_d - dot * q_r_norm
    return torch.cat([q_r_norm, q_d_orth], dim=-1)


def dq_transform_point_torch(d, X):
    """
    Transform 3D point(s) via dual quaternion.
    Extracts R and t from DQ, then does R @ X + t.

    Args:
        d: (..., 8) DQ. Supports (8,), (B, 8), (B, C, 8) etc.
        X: (..., 3) 3D point(s). Must broadcast with d.
    Returns:
        (..., 3) transformed point(s)
    """
    q_r = d[..., :4]
    q_d = d[..., 4:]
    R = _quat_to_rotmat_torch(q_r)  # (..., 3, 3)
    t = 2.0 * _quat_mult_torch(q_d, _quat_conj_torch(q_r))[..., 1:]  # (..., 3)

    # Handle broadcasting: if d is (8,) and X is (N, 3), R is (3,3)
    if R.dim() == 2 and X.dim() == 1:
        return R @ X + t
    if R.dim() == 2 and X.dim() == 2:
        return (R @ X.T).T + t  # (N, 3, 3) @ (N, 3)^T ?? no
    if R.dim() == 2 and X.dim() >= 2:
        return torch.einsum('ij,...j->...i', R, X) + t
    return torch.einsum('...ij,...j->...i', R, X) + t


def dq_transform_points_batch_torch(d, X):
    """
    Transform N points by M camera DQs in a fully batched way.
    Uses R,t extraction for guaranteed correctness.

    Args:
        d: (M, 8) DQ representing w2c transforms
        X: (N, 3) points in world frame
    Returns:
        X_cam: (M, N, 3) transformed points
    """
    q_r = d[:, :4]           # (M, 4)
    q_d = d[:, 4:]           # (M, 4)
    R = _quat_to_rotmat_torch(q_r)          # (M, 3, 3)
    t = 2.0 * _quat_mult_torch(q_d, _quat_conj_torch(q_r))[..., 1:]  # (M, 3)
    X_exp = X.unsqueeze(0)   # (1, N, 3)
    R_exp = R.unsqueeze(1)   # (M, 1, 3, 3)
    t_exp = t.unsqueeze(1)   # (M, 1, 3)
    return torch.einsum('mnij,mnj->mni', R_exp, X_exp) + t_exp  # (M, N, 3)


def dq_extract_translation_torch(d):
    """Extract translation vector from DQ: t = 2 * vector_part(q_d ⊗ q_r*)."""
    q_r = d[..., :4]
    q_d = d[..., 4:]
    return 2.0 * _quat_mult_torch(q_d, _quat_conj_torch(q_r))[..., 1:]


def dq_identity_torch(batch=(), device='cpu', dtype=torch.float32):
    """Identity DQ tensor: [1, 0, 0, 0, 0, 0, 0, 0]."""
    d = torch.zeros(*batch, 8, device=device, dtype=dtype)
    d[..., 0] = 1.0
    return d


def dq_geodesic_loss(d_pred, d_gt, k=3.0):
    """
    Geodesic SE(3) distance between predicted and GT dual quaternions.
    """
    d_pred = dq_normalize_torch(d_pred)  # ensure valid DQ before inverse
    d_pred_inv = dq_inverse_torch(d_pred)
    d_err = dq_mult_torch(d_pred_inv, d_gt)
    d_err = dq_normalize_torch(d_err)
    q_err = d_err[..., :4]
    t_err = dq_extract_translation_torch(d_err)
    qw = torch.clamp(torch.abs(q_err[..., 0]), 0.0, 1.0 - 1e-7)
    theta = 2.0 * torch.acos(qw)
    loss_rot = (theta ** 2).mean()
    loss_trans = (t_err ** 2).sum(-1).mean()
    return k * loss_rot + loss_trans


def relative_poses_to_absolute_dq_torch(relative_dqs, first_abs_dq, device):
    """
    Compose relative DQ poses into absolute DQ poses.

    Args:
        relative_dqs: (N, 8) tensor of relative DQs
        first_abs_dq: (8,) tensor first absolute DQ
    Returns:
        (N+1, 8) tensor of absolute DQs (includes first frame)
    """
    abs_list = [first_abs_dq.unsqueeze(0)]
    for i in range(relative_dqs.shape[0]):
        abs_list.append(dq_mult_torch(abs_list[-1], relative_dqs[i:i+1]))
    return torch.cat(abs_list, dim=0)


def dq_to_matrix_torch(d):
    """PyTorch DQ to 4x4 matrix."""
    q_r = d[..., :4]
    R = _quat_to_rotmat_torch(q_r)
    t = dq_extract_translation_torch(d)
    N = d.shape[0]
    T = torch.eye(4, device=d.device, dtype=d.dtype).unsqueeze(0).repeat(N, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


# =============================================================================
# Wrapper: auto-dispatch numpy / torch
# =============================================================================

def _is_tensor(x):
    return isinstance(x, torch.Tensor)


def dq_mult(d1, d2):
    if _is_tensor(d1) or _is_tensor(d2):
        return dq_mult_torch(
            torch.as_tensor(d1, dtype=torch.float32) if not _is_tensor(d1) else d1,
            torch.as_tensor(d2, dtype=torch.float32) if not _is_tensor(d2) else d2,
        )
    return dq_mult_np(d1, d2)


def dq_conj(d):
    if _is_tensor(d):
        return dq_conj_torch(d)
    return dq_conj_np(d)


def dq_transform_point(d, X):
    if _is_tensor(d) or _is_tensor(X):
        if not _is_tensor(d):
            d = torch.as_tensor(d, dtype=torch.float32)
        if not _is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)
        return dq_transform_point_torch(d, X)
    return dq_transform_point_np(d, X)
