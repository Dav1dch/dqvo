# GNN-BA-VO Agent Guidelines

GNN-optimized monocular visual odometry: a frozen Vision Transformer provides initial pose estimates, and a Graph Neural Network Bundle Adjustment (GNN-BA) module refines them by minimizing reprojection error. Based on KITTI dataset.

## Critical Setup

- **Python version**: exactly 3.8.0 (not 3.8+)
- **PyTorch**: 1.10.1 (pinned in requirements.txt)
- **PyTorch Geometric**: required (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`, `torch-geometric`)
- **Data location**: Create softlink `data/` → KITTI `sequences_jpg/` and `poses/`. The dataloader expects `data/sequences_jpg/` and `data/poses/`. The GNN pipeline also reads `calib.txt` from each sequence directory.
- **First-run artifacts**: `fp.pickle` (feature points) auto-generated in project root on first legacy dataset load (~minutes). Reused across `kitti.py`, `kitti_fp.py`, `kitti_dual.py`. The GNN pipeline (`kitti_gnn.py`) does **not** use it — extracts features on-the-fly.
- **Pretrained VO model**: Required checkpoint (e.g., `checkpoints/Exp51/checkpoint_best.pth`) must exist before GNN training/inference.

## Commands

```bash
conda create -n gnn-ba-vo python==3.8.0 && conda activate gnn-ba-vo
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

python train_gnn_ba.py       # GNN-BA training (argparse)
python inference_gnn_ba.py   # GNN-BA inference (argparse)
python gnn_ba_test.py        # Single-window debug test (argparse)
python train.py              # Legacy VO training (inline dict config)
python predict_poses.py      # Legacy VO inference (inline dict config)
python plot_results.py       # Trajectory visualization (edit paths)
```

## Repository Structure

- `train_gnn_ba.py` — main GNN-BA training entrypoint (uses argparse)
- `inference_gnn_ba.py` — GNN-BA inference on full sequences
- `gnn_ba_test.py` — single-window debug test (no VO model needed)
- `train.py`, `predict_poses.py` — legacy VO-only pipeline (inline dict configs)
- `timesformer/models/gnn_ba.py` — `GNNBAOptimizer` (heterogeneous GNN)
- `datasets/kitti_gnn.py` — `KITTIFeatureDataset` (ORB features + optical flow tracks)
- `utils/gnn_ba.py` — triangulation, graph building, pose utilities, losses
- `build_model.py` — VO model factory (loads frozen backbone)

## Configuration Pattern

- **Most scripts**: inline Python `dict`s (not argparse). Edit `args` and `model_params` directly in `train.py` and `predict_poses.py`.
- **Exception**: `train_gnn_ba.py`, `inference_gnn_ba.py`, and `gnn_ba_test.py` use `argparse`. Run `python train_gnn_ba.py --help` to see options.
- **GNN defaults** (train_gnn_ba.py): `hidden_dim=128`, `num_layers=6`, `lr=1e-4`, `window_size=3`, `overlap=2`, `batch_size=16`, `reproj_weight=0.01`, `pose_weight=1.0`, `point_weight=0.1`, `weighted_loss=3.0`, `save_dir=gnn_ba_output`.
- **VO model config** is hardcoded in `train_gnn_ba.py` (lines 867–880): `dim=384`, `depth=16`, `heads=6`, `image_size=(224, 672)`, `patch_size=16`.
- **Typecheck**: `basedpyright` in standard mode (`pyproject.toml`). Run `basedpyright` to typecheck, though the codebase has no type hints — expect many findings.

## Data Format

Expected directory layout:
```
data/
  sequences_jpg/
    {seq}/image_2/*.png      # GNN pipeline uses image_2 (color cam)
    {seq}/calib.txt          # Intrinsics (P0 line for fx, fy, cx, cy)
  poses/
    {seq}.txt                # Ground truth 3x4 pose matrices
```

## Training Behavior (train_gnn_ba.py)

- **VO model is frozen** (eval mode, no gradients). Only GNN parameters are optimized.
- **Dropout (0.1)** applied to all GNN message/update networks. Output projection layers have **no bias** (`bias=False`).
- **Loss**: `reproj_weight × Huber(reprojection_error) + pose_weight × [k × MSE(angles) + MSE(translation)] + point_weight × Huber(refined_points - GT_triangulated_points) + 0.01 × ||mean(pred_rel_poses)||`. Batches with `loss > loss_clip` are skipped.
- **Validation** runs every 10 epochs. Metrics: initial vs optimized reprojection error (pixels).
- **Best model** saved as `gnn_ba_output/gnn_ba_best.pth` (lowest val error).
- **DataLoader**: `num_workers=16`, custom `collate_fn` for variable-size tracks/observations.
- **Deterministic training** enabled by default (`torch.backends.cudnn.deterministic=True`).

## Two-Stage Training

Use `--stage1_epochs` to run a single training session that transitions from point-only (stage 1) to full training (stage 2) automatically:

```bash
# 30 epochs of point refinement, then 70 epochs of full training
python train_gnn_ba.py --stage1_epochs 30 --num_epochs 100 [other args]
```

Stage 1 freezes pose output layers (`c2c_pose_out_proj` + `camera_out_proj`) and sets `pose_weight=0`, so only point/feature parameters receive gradients. The reprojection loss (with frozen poses) and point triangulation loss drive point refinement. At epoch `stage1_epochs+1`, all layers are unfrozen, the optimizer is re-created, and full training resumes with all losses. The LR scheduler resets its cosine annealing over the remaining epochs.

## Inference Notes

- `inference_gnn_ba.py`: argparse script; outputs `poses_{seq}_vo.txt` and `poses_{seq}_opt.txt` in `gnn_ba_output/`.
- `gnn_ba_test.py`: single-window test with noisy GT poses; useful for debugging reprojection error dynamics.
- `train_gnn_ba.py` after training runs final evaluation and trajectory plot generation automatically.
- Window merging uses `post_processing()` and `recover_trajectory_and_poses()` from `utils/gnn_ba.py`.

## Common Pitfalls

- **CUDA OOM**: Reduce `batch_size` (default 16) in `train_gnn_ba.py`.
- **Missing data**: Ensure `data/` symlink exists and includes `calib.txt` in each sequence; GNN pipeline will fail silently or crash on missing intrinsics.
- **Stale `fp.pickle`**: Delete if you change feature extraction method; regenerated on next load (~minutes).
- **Checkpoint mismatch**: VO checkpoint architecture must match `build_model.py` expectations. GNN loads with `strict=False` only where noted.
- **Empty validation loader**: If dataset too small, `val_loader` may be empty; check `len(val_loader) > 0` before validation.
- **PyTorch Geometric missing**: Install all PyG dependencies from the specified URL; otherwise `ImportError` on `torch_geometric`.
- **GNN requires full KITTI raw**: Unlike legacy VO, GNN needs `calib.txt` per sequence (present in KITTI raw downloads).
- **Normalization stats hardcoded**: `KITTI_MEAN_ANGLES`, `KITTI_STD_ANGLES`, `KITTI_MEAN_T`, `KITTI_STD_T` in `utils/gnn_ba.py`. Do not change without retraining.

## Style & Workflow

- Minimal docstrings, no type hints, little error handling. Follow existing patterns when editing.
- Image normalization: mean `[0.34721234, 0.36705238, 0.36066107]`, std `[0.30737526, 0.31515116, 0.32020183]` (KITTI-specific).
- Pose representation: 6-DoF (Euler angles ZYX + translation), normalized using the above stats.
- Graph construction: `build_heterogeneous_graph()` creates `torch_geometric.data.HeteroData` with camera nodes, point nodes, and observation edges.
- GNN architecture: `num_layers=6` default, `LayerNorm + residual` in each layer. Per-layer update order: `c2c edge → point→camera → c2c edge`.
- Training uses mixed numpy/torch data flow; be careful with device placement and dtype conversions.

## Git Ignore

`data/`, `checkpoints/`, `*.pth`, `*.pdf`, `*.pt`, `Exp53/`, `__pycache__/`, `.idea/` are gitignored. `gnn_ba_output/` is **not** gitignored — avoid committing trained models or outputs there.
