# TSformer-VO Agent Guidelines

PyTorch monocular visual odometry using TimeSformer/Mamba transformers. Based on KITTI dataset.

## Critical Setup

- **Python version**: exactly 3.8.0 (not 3.8+)
- **PyTorch**: 1.10.1 (pinned in requirements.txt)
- **Data location**: Create softlink `data/` → KITTI `sequences_jpg/` and `poses/`. The dataloader expects `data/sequences_jpg/` and `data/poses/` to exist. Do not copy; use symlink.
- **First-run artifact**: `fp.pickle` (feature points) auto-generated on first dataset load (~minutes). Subsequent runs reuse it.

## Commands

```bash
conda create -n tsformer-vo python==3.8.0 && conda activate tsformer-vo
pip install -r requirements.txt

python train.py              # edit args dict inside file; writes args.pkl + args.txt in checkpoint dir
python predict_poses.py       # edit checkpoint_path, checkpoint_name, sequences at top of file
python test.py                # SuperPoint test (hardcoded image path)
python train_tri.py           # triangulation training
python train_gnn_ba.py        # GNN bundle adjustment training
python gnn_ba_test.py         # GNN BA test
python plot_results.py        # trajectory visualization (edit paths in file)
```

## Repository Structure

- `train.py`, `predict_poses.py`, `train_tri.py`, `train_gnn_ba.py` — entrypoints (config via inline Python dicts)
- `build_model.py` — model factory for ViT, CrossViT, Mamba, DeepVO variants
- `datasets/kitti.py` — KITTI loader; generates `fp.pickle`; hardcoded normalization stats
- `timesformer/models/` — model implementations (vit.py, mamba.py, deepvo.py, etc.)
- `superpoint.py` — SuperPoint feature extractor

## Configuration Pattern

All scripts use inline Python dictionaries, not argparse. Edit the `args` or `model_params` dicts directly in each script. Training saves `args.pkl` and `args.txt` alongside checkpoints; inference loads them back.

## Data Format

Expected directory layout after softlink:
```
data/
  sequences_jpg/
    00/image_0/000000.jpg
    00/image_1/...
    ...
  poses/
    00.txt
    01.txt
    ...
```
Each pose file: N x 12 matrix (3x4 rotation+translation per frame).

## Training Behavior

- Validation runs every epoch (`if not epoch % 1:` in train.py:108)
- Best model saved as `checkpoint_best.pth` (lowest val loss)
- Periodic checkpoint every 20 epochs: `checkpoint_e20.pth`, `checkpoint_e40.pth`, …
- Last checkpoint always saved: `checkpoint_last.pth`
- TensorBoard logs written to checkpoint directory
- `num_workers=4` for DataLoader; `batch_size` from `args["bsize"]`
- Uses `torch.no_grad()` only in validation/inference, not training

## Inference Notes

- `predict_poses.py` hardcodes `window_size=3`, `overlap=2` after loading args — may override checkpoint args
- Model loaded with `strict=False` to ignore missing/extra keys
- Output saved as `.npy` per sequence in `<checkpoint_path>/<checkpoint_name>/`
- Inference DataLoader uses `batch_size=4`, `num_workers=10`

## Common Pitfalls

- **CUDA OOM**: Reduce `batch_size` in train.py or predict_poses.py
- **Missing data**: Ensure `data/` symlink exists; dataloader will fail silently on missing files
- **fp.pickle stale**: Delete if you change feature extraction; regenerated on next load
- **Checkpoint mismatch**: `args.pkl` must match model architecture; don't mix checkpoints across experiments
- **Type checking**: `basedpyright` mentioned in older docs but not in requirements; ignore or install separately
- **No validation split**: If dataset too small, val_loader may be empty; check `len(val_loader)` > 0

## Style Reality Check

The codebase does not follow the Python conventions listed in older AGENTS.md (no docstrings, no type hints, minimal error handling). Follow existing file patterns when editing.

## Git Ignore

`data/`, `checkpoints/`, `*.pth`, `*.png`, `*.pdf`, `*.pt`, `Exp53/` are all gitignored. Do not commit trained models or data.
