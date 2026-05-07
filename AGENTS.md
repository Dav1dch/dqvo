# GNN-BA-VO Agent Guidelines

GNN-optimized monocular visual odometry: a frozen Vision Transformer provides initial 6-DOF pose estimates, which are converted to **dual quaternions** (DQ), and a GNN-BA module refines them by minimizing geodesic SE(3) loss + reprojection error. Based on KITTI dataset.

## Setup

- **Python**: 3.8+ (currently 3.10)
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
python precompute_vo_cache.py      # Standalone cache builder (uses ProcessPoolExecutor)
python train.py                    # Legacy VO training
python predict_poses.py            # Legacy VO inference
```

## Script Entrypoints

| Script | Purpose | Deps |
|--------|---------|------|
| `train_gnn_ba.py` | VO + GNN training, multi-seq, cache, 2-stage | VO ckpt, KITTI |
| `train_gnn_ba_noisy.py` | Noisy-GT GNN training (no VO) | KITTI only |
| `inference_gnn_ba.py` | VO + GNN full-sequence eval, multi-seq, ATE | VO ckpt, GNN ckpt |
| `inference_gnn_ba_noisy.py` | Noisy-GT GNN full-sequence eval | GNN ckpt |
| `gnn_ba_test.py` | Single-window debug (`image_0`, ORB matching) | KITTI only |
| `precompute_vo_cache.py` | Build VO cache standalone (same format as `--build_cache`) | VO ckpt |
| `timesformer/models/gnn_ba.py` | `GNNBAOptimizer` — shared GNN model | |
| `datasets/kitti_gnn.py` | `KITTIFeatureDataset` — shared dataloader | |
| `utils/gnn_ba.py` | triangulation, graph building, losses, post-processing | |
| `utils/dq.py` | **all** DQ algebra (numpy + torch) — `dq_mult`, `dq_inverse`, `dq_geodesic_loss`, `dq_transform_points_batch_torch`, etc. | |

## Dual Quaternion Architecture

The entire GNN pipeline uses dual quaternions (8-D) as the sole pose representation. No 4×4 matrices or Euler angles are used internally for pose operations.

### DQ Convention

`d = q_r + ε q_d` where `q_r ∈ S³` (unit quaternion), `q_d = ½ t_q ⊗ q_r` with `t_q = [0, tx, ty, tz]`. Stored as `[qw, qx, qy, qz, q'w, q'x, q'y, q'z]` (8 elements).

### Data flow (VO mode)

```
VO model → 6-DOF normalized → denormalize → pose_6dof_to_dq() → accumulate DQ
    → camera.x = absolute DQ (8-D, no normalization)
    → c2c edge_attr = relative DQ from VO
    → GNN: embed 8→hidden, output 8 (bias=identity DQ)
    → d_correction = dq_normalize_torch(output)
    → d_output = dq_mult_torch(d_correction, d_input)   # multiplicative residual
    → loss = dq_geodesic_loss(pred, gt) + reproj + point
```

### GT poses in dataset

`datasets/kitti_gnn.py` returns `global_poses` as `(window_size-1, 8)` DQ relative poses (built via `matrix_to_dq_np(inv(T_i) @ T_{i+1})`). `abs_poses` remains as 4×4 for GT triangulation.

## Critical DQ Gotchas

### `dq_inverse` ≠ `dq_conj`

For SE(3) DQs, the conjugate (`q_r* - ε q_d*`) does NOT equal the inverse. The correct inverse is computed by extracting R,t, inverting them, and re-encoding via `dq_inverse_torch()` / `dq_inverse_np()`. Use `dq_inverse` for:
- Computing w2c from c2w (triangulation, projection)
- Computing relative poses from absolute poses (`d_rel = d_i^{-1} ⊗ d_j`)
- The geodesic loss error computation

`dq_conj` is kept available but is rarely needed.

### `torch.linalg.svd` NOT `torch.svd`

`torch.svd` gives numerically wrong results on poorly-conditioned A matrices (common in DLT triangulation). Always use `torch.linalg.svd(A)` for SVD in DLT. Applied in `triangulate_dlt`, `_triangulate_track_gpu`, and `triangulate_from_graph_obs`.

### DLT uses pixel coords, not normalized coords

When building DLT constraints from projection rows `P = K @ [R | t]`, use `u * P[2] - P[0]` (raw pixel coordinates), NOT `(u-cx)/fx * P[2] - P[0]` (normalized coordinates). The latter produces wrong results. Applied in `_triangulate_track_gpu`.

### DQ point transform: extract R,t, don't use quaternion formula

The naive dual quaternion point transform formula (`d ⊗ X_q ⊗ conj(d)`) was buggy. Instead, extract R and t from the DQ, then use `R @ X + t`. Both `dq_transform_point_torch` and `dq_transform_points_batch_torch` use this approach.

## Triangulation Optimization

`triangulate_all_points` and `triangulate_tracks_no_filter` use `_precompute_camera_proj()` to compute all camera projection rows (P_u, P_v, P_z as 4-vectors) once, then run per-track DLT via `_triangulate_track_gpu` which uses GPU tensor indexing (no inner loops). The cheirality check is also batched. This gives ~5× speedup over per-observation DQ extraction.

**Keep triangulation on CPU** — data-transfer overhead (many small tensors per track) makes GPU slower for KITTI-scale data (3 cameras, 6×4 SVD).

## VO Model Constants

The VO model outputs **normalized 6-DOF** `[euler_z, euler_y, euler_x, tx, ty, tz]`. These normalization stats are needed for denormalization before DQ conversion and are inlined in each training/inference script (hardcoded numpy arrays):

```python
mean_angles = [1.7061e-5,  9.5582e-4, -5.5258e-5]
std_angles  = [2.8256e-3,  1.7771e-2,  3.2326e-3]
mean_t      = [-8.6736e-5, -1.6038e-2,  9.0033e-1]
std_t       = [2.5584e-2,  1.8545e-2,  3.0352e-1]
```

Do not change these without retraining the VO model. The DQ GNN does NOT use these stats (DQ is unnormalized).

## GNN Model Details

- **Embedding**: camera 8→hidden, c2c_edge 8→hidden, point 3→hidden, p2c_edge 2→hidden
- **Output heads**: `camera_out_proj` hidden→8, `c2c_pose_out_proj` hidden→8 (both `bias=True` with identity DQ bias `[1,0,0,0,0,0,0,0]`)
- **Weight init**: Xavier `gain=0.001` for DQ output heads (must be small—DQ output is multiplicative, not additive). Point output uses `gain=1.0`.
- **Output normalization**: `dq_normalize_torch()` projects raw output to valid DQ manifold (unit real part, orthogonal dual)
- **Message passing**: configurable `num_layers` (default 3 VO / 6 noisy). Layer order: c2c_edge → camera←point → point←camera → c2c_edge
- **VO model is frozen** — `model.eval()` with `torch.no_grad()`. Only GNN parameters get gradients.

## Training Workflows

### Stage 1 / Stage 2 training (VO mode)

`train_gnn_ba.py` supports two-stage training via `--stage1_epochs N`:
- **Stage 1** (epochs 1–N): DQ output heads (`c2c_pose_out_proj`, `camera_out_proj`) are **frozen** (no gradients). Only point and edge message-passing layers train. `pose_weight` is set to 0.
- **Stage 2** (epochs N+1 to end): all parameters unfrozen, `pose_weight` restored. Optimizer and scheduler are re-created.

Useful for warming up point triangulation before applying pose supervision. Set `--stage1_epochs 0` (default) to skip.

### VO mode — multi-sequence with cache (fast)
```bash
# Build cache (can use either approach):
# A) Inline flag in train_gnn_ba.py
python train_gnn_ba.py --checkpoint checkpoints/Exp51/checkpoint_best.pth \
    --train_seqs 00,02,08,09 --build_cache vo_cache.pt

# B) Standalone script (same output format)
python precompute_vo_cache.py --checkpoint checkpoints/Exp51/checkpoint_best.pth \
    --sequence 03

# Train fast (GPU-only per epoch)
python train_gnn_ba.py --checkpoint checkpoints/Exp51/checkpoint_best.pth \
    --train_seqs 00,02,08,09 --test_seqs 01,03,04,05,06,07,10 \
    --cache vo_cache.pt --num_epochs 20
```

### VO mode — single sequence (slow, no cache)
```bash
python train_gnn_ba.py --checkpoint checkpoints/Exp51/checkpoint_best.pth \
    --sequence 03 --num_epochs 10 --point_weight 0.1
```

### Noisy-GT mode (no VO model)
```bash
python train_gnn_ba_noisy.py --sequence 03 --num_epochs 50
```

## DataLoader / Multiprocessing

**Use `num_workers=0` by default.** Two IPC issues:
1. `RuntimeError: received 0 items of ancdata` — fd exhaustion with PyG tensors across workers
2. `torch_shm_manager: Permission denied` — `file_system` sharing unavailable

The noisy script defaults to `num_workers=0`. The VO cache builder uses `ProcessPoolExecutor` for CPU triangulation (safe — only numpy arrays, no CUDA tensors).

## Key Defaults

- `window_size=3`, `overlap=2`, `batch_size=16`
- `hidden_dim=256` (VO) / `128` (noisy), `num_layers=3` (VO) / `6` (noisy), `lr=5e-5` (VO) / `1e-4` (noisy)
- `reproj_weight=0.01`, `pose_weight=1.0`, `point_weight=0.1`
- `weighted_loss=3.0` (k multiplier on rotation part of geodesic loss), `loss_clip=500`
- VO model: `dim=384`, `depth=16`, `heads=6`, `image_size=(224,672)`, `patch_size=16`
- Image normalization: mean `[0.3472, 0.3671, 0.3607]`, std `[0.3074, 0.3152, 0.3202]`

## Common Pitfalls

- **Training is slow without cache** — the ViT VO model (depth=16, dim=384) dominates. Use `--build_cache` then `--cache` for fast training.
- **GNN output gains must be small** — `gain=0.001` for DQ output heads. DQ residual is multiplicative, not additive like the old 6-DOF code.
- **Post-processing averages DQs on SE(3) manifold** — `_dq_average_np()` extracts R,t, averages in log-space, re-encodes. Simple 8-D vector averaging produces invalid DQs that cause trajectory oscillation.
- **`calib.txt` reads P2 line** (color camera intrinsics). `image_2/` (color). `gnn_ba_test.py` uses `image_0/` (grayscale, different intrinsics — P0 line).
- **Cache builder uses CPU** — `_triangulate_one_worker` runs in `ProcessPoolExecutor`, cannot use CUDA.
- **`gnn_ba_output/` is gitignored** — avoid committing output files.
- **CUDA OOM**: reduce `batch_size`.
- **No formal tests** — validate by running `inference_gnn_ba.py` and checking trajectory plots. No pytest/lint infrastructure exists.

## Git Ignore

`data/`, `checkpoints/`, `*.pth`, `*.pdf`, `*.pt`, `Exp53/`, `__pycache__/`, `.idea/`, `gnn_ba_output/` are gitignored.
