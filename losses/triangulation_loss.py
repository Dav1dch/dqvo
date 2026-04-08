import torch
import torch.nn as nn


def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    qw = qw.unsqueeze(-1) if qw.dim() == 1 else qw
    qx = qx.unsqueeze(-1) if qx.dim() == 1 else qx
    qy = qy.unsqueeze(-1) if qy.dim() == 1 else qy
    qz = qz.unsqueeze(-1) if qz.dim() == 1 else qz

    R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    R[..., 0, 1] = 2 * (qx * qy - qw * qz)
    R[..., 0, 2] = 2 * (qx * qz + qw * qy)
    R[..., 1, 0] = 2 * (qx * qy + qw * qz)
    R[..., 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    R[..., 1, 2] = 2 * (qy * qz - qw * qx)
    R[..., 2, 0] = 2 * (qx * qz - qw * qy)
    R[..., 2, 1] = 2 * (qy * qz + qw * qx)
    R[..., 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)
    return R


def dual_quaternion_to_pose_torch(dq):
    q = dq[..., :4]
    t = dq[..., 4:]

    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    R = quaternion_to_rotation_matrix(q)

    T = torch.eye(4, device=dq.device, dtype=dq.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def triangulate_points_dlt(pts1, pts2, K, T1, T2):
    K = K.to(dtype=pts1.dtype)
    P1 = K @ T1[:3, :4]
    P2 = K @ T2[:3, :4]

    num_points = pts1.shape[0]
    
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]

    A = torch.zeros(num_points, 4, 4, device=pts1.device, dtype=pts1.dtype)
    A[:, 0, :] = x1[:, None] * P1[2, :] - P1[0, :]
    A[:, 1, :] = y1[:, None] * P1[2, :] - P1[1, :]
    A[:, 2, :] = x2[:, None] * P2[2, :] - P2[0, :]
    A[:, 3, :] = y2[:, None] * P2[2, :] - P2[1, :]

    _, _, Vh = torch.linalg.svd(A)
    X = Vh[:, -1, :]
    X = X / (X[:, 3:4].clamp(min=1e-8))

    points_3d = X[:, :3]

    return points_3d


class TriangulationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pts_list, pred_dqs, K, normalize_coords=False, use_median=False):
        if len(pts_list) < 2:
            return torch.tensor(0.0, device=pred_dqs[0].device)

        pt1 = pts_list[0]
        pt2 = pts_list[1]

        if pt1.shape[0] < 4:
            return torch.tensor(0.0, device=pred_dqs[0].device)

        pred_dq = pred_dqs[0]
        T_pred = dual_quaternion_to_pose_torch(pred_dq.squeeze(0))

        T1 = torch.eye(4, device=pt1.device, dtype=pt1.dtype)
        T2 = T_pred

        try:
            points_3d = triangulate_points_dlt(pt1, pt2, K, T1, T2)
            depths = points_3d[:, 2]

            valid_mask = (depths > 0) & (depths < 100)
            if not valid_mask.any():
                return torch.tensor(0.0, device=pt1.device)

            depths_valid = depths[valid_mask]

            if use_median:
                loss = torch.median(torch.abs(depths_valid))
            else:
                loss = torch.mean(depths_valid)

            return loss
        except Exception:
            return torch.tensor(0.0, device=pt1.device)


class ResidualTriangulationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pts_list, pred_dqs, gt_dqs, K, normalize_coords=False):
        
        if len(pts_list) < 2:
            return torch.tensor(0.0, device=pred_dqs[0].device)

        pt1 = pts_list[0]
        pt2 = pts_list[1]

        if pt1.shape[0] < 4:
            return torch.tensor(0.0, device=pred_dqs[0].device)

        T1 = torch.eye(4, device=pt1.device, dtype=pt1.dtype)

        T_pred = dual_quaternion_to_pose_torch(pred_dqs[0].squeeze(0))
        T_gt = dual_quaternion_to_pose_torch(gt_dqs[0].squeeze(0))

        T_pred = T_pred.to(dtype=pt1.dtype)
        T_gt = T_gt.to(dtype=pt1.dtype)

        try:
            points_3d_pred = triangulate_points_dlt(pt1, pt2, K, T1, T_pred)
            points_3d_gt = triangulate_points_dlt(pt1, pt2, K, T1, T_gt)

            depths_pred = points_3d_pred[:, 2]
            depths_gt = points_3d_gt[:, 2]

            valid_mask = (depths_pred > 0) & (depths_pred < 500) & (depths_gt > 0) & (depths_gt < 500)

            if not valid_mask.any():
                return torch.tensor(0.0, device=pt1.device)

            residual = torch.abs(depths_pred[valid_mask] - depths_gt[valid_mask])
            loss = torch.mean(residual)

            return loss
        except Exception:
            return torch.tensor(0.0, device=pt1.device)


class DualQuaternionVOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_dq, target_dq):
        pred_q = pred_dq[..., :4]
        target_q = target_dq[..., :4]
        pred_t = pred_dq[..., 4:]
        target_t = target_dq[..., 4:]

        pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-8)
        target_q = target_q / (torch.norm(target_q, dim=-1, keepdim=True) + 1e-8)

        dot = torch.sum(pred_q * target_q, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        angle = 2 * torch.acos(torch.abs(dot))
        rot_loss = torch.mean(angle ** 2)

        trans_loss = torch.mean((pred_t - target_t) ** 2)

        return rot_loss + trans_loss
