"""Debug script to check GNN delta magnitudes."""
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
    denormalize_poses,
    rotation_to_euler,
    euler_to_rotation,
)


preprocess = transforms.Compose([
    transforms.Resize((224, 672)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183],
    )
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/Exp51/checkpoint_best.pth")
    parser.add_argument("--gnn_model", type=str, default="gnn_ba_output/gnn_ba_best.pth")
    parser.add_argument("--sequence", type=str, default="03")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=2)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load VO model
    model_params = {
        "dim": 384, "image_size": (224, 672), "patch_size": 16,
        "attention_type": "divided_space_time", "num_frames": args.window_size,
        "num_classes": 6 * (args.window_size - 1), "depth": 16, "heads": 6,
        "dim_head": 64, "attn_dropout": 0.2, "ff_dropout": 0.2, "time_only": False,
    }
    model_args = {
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_path": os.path.dirname(args.checkpoint) if os.path.dirname(args.checkpoint) else ".",
        "pretrained_ViT": False,
    }
    vo_model, _ = build_model(model_args, model_params)
    vo_model.eval()
    
    # Load GNN
    gnn_model = GNNBAOptimizer(hidden_dim=32, num_layers=3).to(device)
    checkpoint = torch.load(args.gnn_model, map_location=device)
    gnn_model.load_state_dict(checkpoint["model_state_dict"])
    gnn_model.eval()
    
    # Load dataset
    dataset = KITTIFeatureDataset(
        data_path="data/sequences_jpg",
        gt_path="data/poses",
        sequence=args.sequence,
        window_size=args.window_size,
        overlap=args.overlap,
        max_points=200,
    )
    
    # Collect statistics on GNN deltas
    delta_norms = []
    delta_t_norms = []
    delta_euler_norms = []
    
    print("Analyzing GNN deltas on first 50 windows...")
    for idx in tqdm(range(min(50, len(dataset)))):
        sample = dataset[idx]
        
        pil_images = []
        for img in sample["images"]:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img)
            pil_images.append(preprocess(pil_img))
        
        imgs_batch = torch.stack(pil_images, dim=1).squeeze(0).unsqueeze(0).to(device)
        
        K = sample["K"]
        tracks = sample["tracks"]
        
        # Reformat tracks
        rel_tracks = []
        for track in tracks:
            if len(track) >= 2:
                rel_tracks.append([(obs[0], obs[1], obs[2]) for obs in track])
        tracks = rel_tracks
        
        # Get VO poses
        with torch.no_grad():
            vo_output = vo_model(imgs_batch)
        vo_poses = denormalize_poses(
            vo_output, args.window_size, KITTI_MEAN_ANGLES, KITTI_STD_ANGLES, KITTI_MEAN_T, KITTI_STD_T
        )[0]
        
        if len(tracks) < 10:
            continue
            
        points_3d, graph_obs = triangulate_all_points(tracks, vo_poses, K)
        if len(points_3d) < 10:
            continue
        
        img_height, img_width = sample["images"][0].shape[:2]
        data = build_heterogeneous_graph(args.window_size, points_3d, graph_obs, img_height, img_width)
        
        # Convert to 6-DOF
        abs_poses_6dof = []
        for pose in vo_poses:
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu().numpy()
            R = pose[:3, :3]
            t = pose[:3, 3]
            euler = rotation_to_euler(R, seq='zyx')
            abs_poses_6dof.append(np.concatenate([euler, t]))
        abs_poses_6dof = np.array(abs_poses_6dof)
        
        camera_feats = torch.tensor(abs_poses_6dof, dtype=torch.float32)
        data["camera"].x = camera_feats
        data = data.to(device)
        
        with torch.no_grad():
            camera_delta, point_delta = gnn_model(data, return_delta=True)
        
        # Compute delta norms
        delta_norm = torch.norm(camera_delta).item()
        delta_t_norm = torch.norm(camera_delta[:, 3:]).item()
        delta_euler_norm = torch.norm(camera_delta[:, :3]).item()
        
        delta_norms.append(delta_norm)
        delta_t_norms.append(delta_t_norm)
        delta_euler_norms.append(delta_euler_norm)
    
    print(f"\nDelta Statistics (n={len(delta_norms)} windows):")
    print(f"  Total delta norm: mean={np.mean(delta_norms):.4f}, std={np.std(delta_norms):.4f}")
    print(f"  Translation delta norm: mean={np.mean(delta_t_norms):.4f}, std={np.std(delta_t_norms):.4f}")
    print(f"  Euler delta norm: mean={np.mean(delta_euler_norms):.4f}, std={np.std(delta_euler_norms):.4f}")
    
    # Compare with VO pose norms
    print(f"\nFor reference - typical VO translation magnitude: ~0.86m per frame")


if __name__ == "__main__":
    main()
