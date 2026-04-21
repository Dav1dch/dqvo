"""
Test script to evaluate the re-triangulation plan for GNN-BA training.

Tests whether re-triangulating 3D points from GNN-predicted poses (instead of VO-triangulated points)
actually reduces reprojection error when the GNN learns correct poses.

Key insight: In train_gnn_ba.py, the VO model outputs normalized relative poses that are
denormalized and accumulated to absolute poses. These VO poses are used for triangulation,
but the GT relative poses are used for pose supervision loss.

Scenarios tested:
1. VO poses + VO-triangulated points (current baseline)
2. GT poses + GT-triangulated points (ideal lower bound)
3. GT poses + VO-triangulated points (current mismatch - why reproj stays high)
4. GT poses + GT-retriangulated points (proposed fix)
5. Simulated "GNN poses" (interpolated between VO and GT) + re-triangulated points

Usage:
    python test_retriangulation.py --sequence 03 --num_windows 10
"""

import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from build_model import build_model
from datasets.kitti_gnn import KITTIFeatureDataset
from utils.gnn_ba import (
    KITTI_MEAN_ANGLES,
    KITTI_STD_ANGLES,
    KITTI_MEAN_T,
    KITTI_STD_T,
    triangulate_all_points,
    compute_reprojection_error,
    euler_to_rotation,
    rotation_to_euler,
)


preprocess = transforms.Compose([
    transforms.Resize((224, 672)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183],
    ),
])


def get_vo_poses(vo_model, images, K, window_size, device):
    """
    Get VO absolute poses by running VO model on images.
    This matches the logic in train_gnn_ba.py lines 199-212.
    """
    pil_images = []
    for img in images:
        if len(img.shape) == 2:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_images.append(preprocess(Image.fromarray(img)))
    
    imgs_batch = torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        vo_output = vo_model(imgs_batch)
    
    vo_output = vo_output.reshape(1, window_size - 1, 6)
    vo_output_np = vo_output.cpu().numpy()[0]
    
    mean_angles = KITTI_MEAN_ANGLES
    std_angles = KITTI_STD_ANGLES
    mean_t = KITTI_MEAN_T
    std_t = KITTI_STD_T
    
    abs_poses_4x4 = [np.eye(4)]
    for i in range(vo_output_np.shape[0]):
        euler = vo_output_np[i, :3] * std_angles + mean_angles
        t = vo_output_np[i, 3:] * std_t + mean_t
        R = euler_to_rotation(euler, seq='zyx')
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t
        abs_poses_4x4.append(abs_poses_4x4[-1] @ T_rel)
    
    return abs_poses_4x4


def gt_3x4_to_4x4(gt_poses_3x4):
    """Convert GT 3x4 poses to 4x4."""
    return [np.vstack([p, [0, 0, 0, 1]]) for p in gt_poses_3x4]


def interpolate_poses(vo_poses, gt_poses, alpha):
    """
    Interpolate between VO and GT poses.
    alpha=0 -> VO, alpha=1 -> GT
    """
    interp_poses = []
    for vo, gt in zip(vo_poses, gt_poses):
        t = (1 - alpha) * vo[:3, 3] + alpha * gt[:3, 3]
        R = (1 - alpha) * vo[:3, :3] + alpha * gt[:3, :3]
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        interp_poses.append(T)
    return interp_poses


def triangulate_from_poses_numpy(poses, observations, K, min_depth=0.1):
    """
    Re-triangulate 3D points from poses and observations.
    """
    if isinstance(K['fx'], torch.Tensor):
        K_matrix = np.array([[K['fx'].item(), 0, K['cx'].item()], 
                             [0, K['fy'].item(), K['cy'].item()], 
                             [0, 0, 1]])
    else:
        K_matrix = np.array([[K['fx'], 0, K['cx']], 
                             [0, K['fy'], K['cy']], 
                             [0, 0, 1]])
    
    K_inv = np.linalg.inv(K_matrix)
    
    obs_by_point = {}
    for pt_idx, frame_idx, u, v in observations:
        if pt_idx not in obs_by_point:
            obs_by_point[pt_idx] = []
        obs_by_point[pt_idx].append((frame_idx, u, v))
    
    points_3d = []
    for pt_idx in sorted(obs_by_point.keys()):
        obs_list = obs_by_point[pt_idx]
        if len(obs_list) < 2:
            continue
        
        A_rows = []
        for frame_idx, u, v in obs_list:
            pose = poses[frame_idx]
            pt = np.array([u, v, 1.0])
            pt_norm = K_inv @ pt
            A_rows.append(pt_norm[0] * pose[2, :] - pose[0, :])
            A_rows.append(pt_norm[1] * pose[2, :] - pose[1, :])
        
        if len(A_rows) < 4:
            continue
        
        A = np.array(A_rows)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        if abs(X[3]) < 1e-10:
            continue
        
        X_3d = X[:3] / X[3]
        
        valid = True
        for frame_idx, u, v in obs_list:
            pose_w2c = np.linalg.inv(poses[frame_idx])
            X_h = np.append(X_3d, 1.0)
            cam_coords = pose_w2c @ X_h
            if cam_coords[2] < min_depth:
                valid = False
                break
        
        if valid and X_3d[2] > 0:
            points_3d.append(X_3d)
    
    if len(points_3d) == 0:
        return np.zeros((0, 3))
    return np.array(points_3d)


def compute_reproj_error_numpy(poses, points_3d, observations, K):
    """Compute mean reprojection error."""
    if isinstance(K['fx'], torch.Tensor):
        K_matrix = np.array([[K['fx'].item(), 0, K['cx'].item()], 
                             [0, K['fy'].item(), K['cy'].item()], 
                             [0, 0, 1]])
    else:
        K_matrix = np.array([[K['fx'], 0, K['cx']], 
                             [0, K['fy'], K['cy']], 
                             [0, 0, 1]])
    
    errors = []
    for pt_idx, frame_idx, u, v in observations:
        if pt_idx >= len(points_3d):
            continue
        pose_w2c = np.linalg.inv(poses[frame_idx])
        X_h = np.append(points_3d[pt_idx], 1.0)
        cam_coords = pose_w2c @ X_h
        if cam_coords[2] < 0.1:
            continue
        proj = K_matrix @ cam_coords[:3]
        proj = proj[:2] / proj[2]
        err = np.sqrt((proj[0] - u)**2 + (proj[1] - v)**2)
        errors.append(err)
    
    return np.mean(errors) if errors else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, default='03')
    parser.add_argument('--num_windows', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/Exp51/checkpoint_best.pth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nLoading VO model from: {args.checkpoint}")
    model_params = {
        'dim': 384,
        'image_size': (224, 672),
        'patch_size': 16,
        'attention_type': 'divided_space_time',
        'num_frames': 3,
        'num_classes': 6 * 2,
        'depth': 16,
        'heads': 6,
        'dim_head': 64,
        'attn_dropout': 0.2,
        'ff_dropout': 0.2,
        'time_only': False,
    }
    
    model_args = {
        'checkpoint': os.path.basename(args.checkpoint),
        'checkpoint_path': os.path.dirname(args.checkpoint) if os.path.dirname(args.checkpoint) else '.',
        'pretrained_ViT': False,
    }
    
    vo_model, _ = build_model(model_args, model_params)
    vo_model.eval()
    vo_model.to(device)
    print("VO model loaded successfully")
    
    print(f"\nLoading KITTI sequence {args.sequence}...")
    dataset = KITTIFeatureDataset(
        data_path="data/sequences_jpg",
        gt_path="data/poses",
        sequence=args.sequence,
        window_size=3,
        overlap=2,
        max_points=200,
    )
    print(f"Dataset loaded: {len(dataset)} windows")
    
    results = {
        'vo_vo': [],
        'gt_gt': [],
        'gt_vo': [],
        'gt_retriang': [],
        'interp_retriang': [],
        'vo_retriang': [],
    }
    
    num_tested = 0
    for idx in range(min(args.num_windows, len(dataset))):
        sample = dataset[idx]
        images = sample["images"]
        tracks = sample["tracks"]
        K = sample["K"]
        
        if len(tracks) < 10:
            continue
        
        window_indices = sample["window_indices"]
        gt_poses_3x4 = [dataset.gt_poses[i] for i in window_indices]
        gt_poses = gt_3x4_to_4x4(gt_poses_3x4)
        
        vo_poses = get_vo_poses(vo_model, images, K, 3, device)
        
        pts_vo, obs_vo, _ = triangulate_all_points(tracks, vo_poses, K)
        if len(pts_vo) < 10:
            continue
        
        pts_gt, obs_gt, _ = triangulate_all_points(tracks, gt_poses, K)
        if len(pts_gt) < 10:
            continue
        
        pts_vo_np = pts_vo.cpu().numpy() if torch.is_tensor(pts_vo) else pts_vo
        pts_gt_np = pts_gt.cpu().numpy() if torch.is_tensor(pts_gt) else pts_gt
        
        e_vo_vo = compute_reproj_error_numpy(vo_poses, pts_vo_np, obs_vo, K)
        results['vo_vo'].append(e_vo_vo)
        
        e_gt_gt = compute_reproj_error_numpy(gt_poses, pts_gt_np, obs_gt, K)
        results['gt_gt'].append(e_gt_gt)
        
        e_gt_vo = compute_reproj_error_numpy(gt_poses, pts_vo_np, obs_vo, K)
        results['gt_vo'].append(e_gt_vo)
        
        pts_gt_retriang = triangulate_from_poses_numpy(gt_poses, obs_gt, K)
        if len(pts_gt_retriang) > 0:
            e_gt_retriang = compute_reproj_error_numpy(gt_poses, pts_gt_retriang, obs_gt, K)
            results['gt_retriang'].append(e_gt_retriang)
        
        pts_vo_retriang = triangulate_from_poses_numpy(vo_poses, obs_vo, K)
        if len(pts_vo_retriang) > 0:
            e_vo_retriang = compute_reproj_error_numpy(vo_poses, pts_vo_retriang, obs_vo, K)
            results['vo_retriang'].append(e_vo_retriang)
        
        interp_poses = interpolate_poses(vo_poses, gt_poses, alpha=0.5)
        pts_interp_retriang = triangulate_from_poses_numpy(interp_poses, obs_vo, K)
        if len(pts_interp_retriang) > 0:
            e_interp_retriang = compute_reproj_error_numpy(interp_poses, pts_interp_retriang, obs_vo, K)
            results['interp_retriang'].append(e_interp_retriang)
        
        num_tested += 1
        print(f"Window {idx}: VO+VO={e_vo_vo:.2f}, GT+GT={e_gt_gt:.2f}, GT+VO={e_gt_vo:.2f}px")
    
    if num_tested == 0:
        print("No valid windows found!")
        return
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for key, label in [
        ('vo_vo', 'VO poses + VO-triangulated (current baseline)'),
        ('gt_gt', 'GT poses + GT-triangulated (ideal lower bound)'),
        ('gt_vo', 'GT poses + VO-triangulated (current mismatch)'),
        ('gt_retriang', 'GT poses + GT-retriangulated (proposed fix)'),
        ('vo_retriang', 'VO poses + VO-retriangulated (sanity check)'),
        ('interp_retriang', '50% VO/GT + re-triangulated'),
    ]:
        vals = results[key]
        if vals:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"  {label}: {mean_val:.2f} ± {std_val:.2f} px")
        else:
            print(f"  {label}: N/A")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    gt_vo_mean = np.mean(results['gt_vo']) if results['gt_vo'] else 0
    vo_vo_mean = np.mean(results['vo_vo']) if results['vo_vo'] else 0
    gt_retriang_mean = np.mean(results['gt_retriang']) if results['gt_retriang'] else 0
    
    if gt_vo_mean > vo_vo_mean:
        print(f"  [CONFIRMED] GT poses with VO-triangulated points has HIGHER error ({gt_vo_mean:.2f}px)")
        print(f"            than VO poses with VO-triangulated points ({vo_vo_mean:.2f}px)")
        print(f"  -> This explains why reproj loss doesn't decrease with pose-only supervision")
    else:
        print(f"  [INFO] GT poses with VO-triangulated: {gt_vo_mean:.2f}px vs VO+VO: {vo_vo_mean:.2f}px")
    
    if gt_retriang_mean < gt_vo_mean:
        print(f"  [CONFIRMED] Re-triangulating from GT poses reduces error from {gt_vo_mean:.2f}px to {gt_retriang_mean:.2f}px")
        print(f"  -> Re-triangulation from GNN poses should help reduce reprojection loss")
    else:
        print(f"  [UNEXPECTED] Re-triangulation didn't help")
    
    print(f"\n  Conclusion: The plan to re-triangulate from GNN-predicted poses is {'VALIDATED' if gt_retriang_mean < gt_vo_mean else 'NOT VALIDATED'}")


if __name__ == "__main__":
    import os
    main()
