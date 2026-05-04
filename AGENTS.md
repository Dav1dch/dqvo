# GNN-BA-VO Agent Guidelines

GNN-optimized monocular visual odometry: a frozen Vision Transformer provides initial pose estimates, and a Graph Neural Network Bundle Adjustment (GNN-BA) module refines them by minimizing reprojection error. Based on KITTI dataset.

## Setup

- **Python**: 3.8+ (originally pinned to 3.8.0; currently runs on 3.10)
- **Data**: softlink `data/` → KITTI `sequences_jpg/` and `poses/`. Each sequence dir must contain `image_2/*.png` and `calib.txt`.
- **VO checkpoint**: must exist at `checkpoints/Exp51/checkpoint_best.pth` before VO-based GNN training.
- **PyTorch Geometric**: install `torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html`
- **fp.pickle**: auto-generated on first legacy dataset load. The GNN pipeline does **not** use it.

## Commands

```bash
python train_gnn_ba.py             # VO + GNN training (argparse)
python inference_gnn_ba.py         # VO + GNN inference
python train_gnn_ba_noisy.py       # Noisy-GT GNN training (no VO model)
python inference_gnn_ba_noisy.py   # Noisy-GT GNN inference
python gnn_ba_test.py              # Single-window debug test
python train.py                    # Legacy VO training
python predict_poses.py            # Legacy VO inference
```

## Script Entrypoints

| Script | Purpose | Deps |
|--------|---------|------|
| `train_gnn_ba.py` | VO + GNN training, multi-seq, cache | VO ckpt, KITTI |
| `train_gnn_ba_noisy.py` | Noisy-GT GNN training (no VO) | KITTI only |
| `inference_gnn_ba.py` | VO + GNN full-sequence eval | VO ckpt, GNN ckpt |
| `inference_gnn_ba_noisy.py` | Noisy-GT GNN full-sequence eval | GNN ckpt |
| `gnn_ba_test.py` | Single-window debug (uses `image_0`, ORB matching) | KITTI only |
| `timesformer/models/gnn_ba.py` | `GNNBAOptimizer` — shared by all scripts | |
| `datasets/kitti_gnn.py` | `KITTIFeatureDataset` — shared dataloader | |
| `utils/gnn_ba.py` | triangulation, graph building, losses, post-processing | |

## Critical Architecture Details

### GNN output is residual
`gnn_ba.py` line ~401: `camera_relative = c2c_edge_attr + c2c_delta`. The GNN adds a learned correction to the input c2c edge features. Initialization (gain=0.01) outputs near-zero delta, so prediction starts at the input pose. This is essential for convergence — without it the GNN starts at zero output (≈ mean pose) and cannot learn turns.

### c2c edges must use exact relative poses
c2c edge features must be the **actual relative pose** (6-DOF, normalized), NOT absolute-pose differences. For VO mode: use the VO model's raw normalized output. For noisy mode: use the exact noisy relative pose. The legacy `compute_normalized_c2c_edges()` uses Euler-angle/translation diffs of absolute poses, which are approximate and cause poor convergence.

### Point loss uses `.mean()` not `.sum()`
`train_gnn_ba.py` line ~497: point loss is `huber_loss(…).mean()`. A previous `.sum()` version inflated the loss 100-200× per sample, dominating training and causing loss-clip filtering.

### Validation measures full GNN refinement
Validation optimized error now uses **GNN-refined points** (not the stale VO-triangulated points), so it measures joint pose+point improvement.

## Training Workflows

### VO mode — multi-sequence with cache
```bash
# 1. Build cache (once, ~minutes)
python train_gnn_ba.py --checkpoint checkpoints/Exp51/checkpoint_best.pth \
    --train_seqs 00,02,08,09 --build_cache vo_cache.pt

# 2. Train fast (GPU-only per epoch)
python train_gnn_ba.py --checkpoint checkpoints/Exp51/checkpoint_best.pth \
    --train_seqs 00,02,08,09 --test_seqs 01,03,04,05,06,07,10 \
    --cache vo_cache.pt --num_epochs 20
```

### VO mode — single sequence (no cache)
```bash
python train_gnn_ba.py --checkpoint checkpoints/Exp51/checkpoint_best.pth \
    --sequence 03 --num_epochs 10
```

### Noisy-GT mode (no VO model)
```bash
python train_gnn_ba_noisy.py --sequence 03 --num_epochs 50
```

## DataLoader / Multiprocessing

**Use `num_workers=0` by default.** The environment has two IPC issues:
1. `RuntimeError: received 0 items of ancdata` — file descriptor exhaustion when PyG tensors are transferred across workers
2. `torch_shm_manager: Permission denied` — `file_system` sharing strategy unavailable

The noisy script defaults to `num_workers=0`. The VO script uses 8 workers in the original code but may crash on long runs. If pushing workers, set `persistent_workers=False`.

## Key Defaults

- `window_size=3`, `overlap=2`, `batch_size=16`
- `hidden_dim=128`, `num_layers=6`, `lr=1e-4`
- `reproj_weight=0.01`, `pose_weight=1.0`, `point_weight=0.1`
- `weighted_loss=3.0` (k multiplier on angle MSE), `loss_clip=500`
- VO model: `dim=384`, `depth=16`, `heads=6`, `image_size=(224,672)`, `patch_size=16`
- Pose normalization stats: `KITTI_MEAN_ANGLES`, `KITTI_STD_ANGLES`, `KITTI_MEAN_T`, `KITTI_STD_T` in `utils/gnn_ba.py`. Do not change without retraining.
- Image normalization: mean `[0.3472, 0.3671, 0.3607]`, std `[0.3074, 0.3152, 0.3202]`

## Common Pitfalls

- **VO model is frozen** — only GNN parameters get gradients. Calling `model.train()` is a bug.
- **GNN output layers have no bias** (`bias=False`) and Xavier init with `gain=0.01` for pose outputs.
- **`calib.txt` reads P2 line** (color camera intrinsics). The dataset loads from `image_2/` (color cam). `gnn_ba_test.py` uses `image_0/` (grayscale, different intrinsics — P0 line).
- **Device placement**: camera features are denormalized absolute 6-DOF (on CPU during graph building, moved to GPU via `.to(device)`). Normalization tensors (`mean_angles`, etc.) must match the device of the data they operate on.
- **`gnn_ba_output/` is not gitignored** — avoid committing trained models or output poses.
- **CUDA OOM**: reduce `batch_size`.

## Git Ignore

`data/`, `checkpoints/`, `*.pth`, `*.pdf`, `*.pt`, `Exp53/`, `__pycache__/`, `.idea/` are gitignored.
