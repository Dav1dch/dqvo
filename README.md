# GNN-BA-VO: GNN-Optimized Monocular Visual Odometry

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2305.06121-B31B1B.svg)](https://arxiv.org/abs/2305.06121)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/aofrancani/TSformer-VO/blob/main/LICENSE)

A two-stage monocular visual odometry pipeline: a frozen Vision Transformer (VO) provides initial pose estimates, and a Graph Neural Network Bundle Adjustment (GNN-BA) module refines them by minimizing reprojection error over heterogeneous camera-point graphs.

Based on [TSformer-VO](https://arxiv.org/abs/2305.06121).

## Architecture

```
Images ──► VO Model (frozen) ──► Initial Relative Poses ──► Accumulate Absolute Poses
                                                                      │
                                          ORB + Optical Flow ──► 2D Tracks
                                                                      │
                                                        Triangulation (DLT) ──► 3D Points
                                                                      │
                                              Heterogeneous Graph (camera ↔ point edges)
                                                                      │
                                                     GNN-BA Optimizer ──► Refined Relative Poses
                                                                      │
                                              Post-processing & Accumulation ──► Final Trajectory
```

**GNN-BA Module** (`timesformer/models/gnn_ba.py`):
- **Node types**: Camera (6-DoF pose features), Point (3D coordinates)
- **Edge types**: Point→Camera (observation, 2D pixel features), Camera→Camera (temporal, relative pose features)
- **Per-layer update**: Point→Camera message passing updates camera nodes, then Camera→Camera edges are updated from refined camera embeddings
- **Output**: Optimized relative poses read from camera-to-camera edge features (residual on input)

**Loss**: `reproj_weight × Huber(reprojection_error) + pose_weight × SmoothL1(GNN_rel_poses, GT_rel_poses)`

## Contents
1. [Dataset](#1-dataset)
2. [Setup](#2-setup)
3. [Usage](#3-usage)
4. [Evaluation](#4-evaluation)

## 1. Dataset

Download the [KITTI odometry dataset (grayscale)](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Images must be in `.png` format (the default). The GNN pipeline also reads `calib.txt` from each sequence directory for camera intrinsics.

Create a softlink:
```bash
ln -s <path_to_kitti_data> <project_root>/data
```

Expected structure:
```
data/
  sequences_jpg/
    {seq}/
      image_2/*.png      # GNN pipeline uses image_2 (color cam)
      calib.txt          # Intrinsics (P0 line for fx, fy, cx, cy)
    ...
  poses/
    {seq}.txt            # Ground truth 3x4 pose matrices
```

## 2. Setup

```bash
conda create -n gnn-ba-vo python==3.8.0
conda activate gnn-ba-vo
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

PyTorch Geometric is required for the `HeteroData` graph construction and message passing.

## 3. Usage

### 3.1. Train GNN-BA

```bash
python train_gnn_ba.py \
  --checkpoint checkpoints/Exp51/checkpoint_best.pth \
  --sequence 03 \
  --num_epochs 100 \
  --hidden_dim 128 \
  --lr 1e-4 \
  --window_size 3 \
  --overlap 2 \
  --batch_size 16
```

Key arguments (use `--help` for full list):

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/Exp51/checkpoint_best.pth` | Path to pretrained VO model |
| `--sequence` | `03` | KITTI sequence to train on |
| `--hidden_dim` | `128` | GNN hidden dimension |
| `--lr` | `1e-4` | Learning rate |
| `--weighted_loss` | `10.0` | Weight for angle supervision |
| `--reproj_weight` | `0.01` | Reprojection loss weight |
| `--pose_weight` | `1.0` | Pose supervision loss weight |
| `--loss_clip` | `500.0` | Skip batch when loss exceeds this |
| `--debug` | `False` | Limit dataset to first 100 windows |

The VO model is **frozen** (eval mode, no gradients). Only GNN parameters are optimized. Validation runs every 10 epochs. Best model saved as `gnn_ba_output/gnn_ba_best.pth`.

### 3.2. Inference

```bash
python inference_gnn_ba.py \
  --checkpoint checkpoints/Exp51/checkpoint_best.pth \
  --gnn_model gnn_ba_output/gnn_ba_best.pth \
  --sequence 03
```

Outputs:
- `gnn_ba_output/poses_{seq}_vo.txt` — VO-only trajectory
- `gnn_ba_output/poses_{seq}_opt.txt` — GNN-optimized trajectory
- `gnn_ba_output/trajectory_comparison.png` — Plot comparing GT, VO, and GNN trajectories

### 3.3. Single-Window Debug Test

```bash
python gnn_ba_test.py --sequence 00 --num_frames 5 --num_epochs 100
```

Tests GNN-BA on a single window with noisy GT poses (no VO model needed). Useful for debugging reprojection error reduction and training dynamics.

### 3.4. Legacy VO Training/Inference

The original TSformer-VO pipeline is still available:

```bash
python train.py              # Edit args dict inside file
python predict_poses.py      # Edit checkpoint_path, checkpoint_name, sequences at top
python plot_results.py       # Trajectory visualization
```

These use inline Python dicts for configuration (not argparse).

## 4. Evaluation

Use the [KITTI odometry evaluation toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval) to compute translational (%) and rotational (deg/100m) errors from the output `.txt` pose files.

## Repository Structure

```
train_gnn_ba.py           # Main GNN-BA training (argparse)
inference_gnn_ba.py        # GNN-BA inference (argparse)
gnn_ba_test.py             # Single-window debug test (argparse)
timesformer/models/gnn_ba.py   # GNNBAOptimizer model
datasets/kitti_gnn.py      # KITTIFeatureDataset (ORB + optical flow tracks)
utils/gnn_ba.py            # Triangulation, graph building, pose utils, losses
train.py                   # Legacy VO training (inline dict config)
predict_poses.py           # Legacy VO inference
build_model.py             # VO model factory
```

## Citation

```bibtex
@article{Francani2023,
  title={Transformer-based model for monocular visual odometry: a video understanding approach},
  author={Fran{\c{c}}ani, Andr{\'e} O and Maximo, Marcos ROA},
  journal={arXiv preprint arXiv:2305.06121},
  year={2023}
}
```

## References

Code adapted from [TimeSformer](https://github.com/facebookresearch/TimeSformer).
Previous work: [DPT-VO](https://github.com/aofrancani/DPT-VO)
