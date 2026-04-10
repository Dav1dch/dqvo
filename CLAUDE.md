# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TSformer-VO is a monocular Visual Odometry system using Transformer/Mamba architectures for 6-DoF camera pose estimation from video sequences. Based on TimeSformer with modifications for VO tasks.

**Key Reference**: `timesformer/models/mamba_gated.py` contains `CrossVisionMamba` - the primary model architecture used in recent experiments.

## Architecture

### Models (`timesformer/models/`)
- **VisionTransformer** (`vit.py`): Standard TimeSformer with divided space-time attention
- **CrossViT** (`vit_seq.py`): Cross-attention between two frames for VO
- **CrossVisionMamba** (`mamba_gated.py`): Mamba-based Vision Mamba with bidirectional scanning and spiral patch ordering - **currently primary model**
- **DeepVO** (`deepvo.py`): RNN-based baseline
- **superpoint.py**: SuperPoint feature detector for optional feature point guidance

### Loss Functions (`timesformer/models/losses.py`, `datasets/utils.py`)
- MSELoss for Euler angle + translation output
- `dual_quaternion_loss`, `quaternion_loss` for quaternion-based pose representation
- `quaternion_loss_weighted` for weighted rotation/translation loss

### Dataset (`datasets/kitti.py`)
- KITTI Visual Odometry dataset loader
- Window-based frame sampling (`window_size`, `overlap`)
- Outputs: `(images [C T H W], pt1, pt2, y)` where `y` is relative pose
- Uses `fp.pickle` for precomputed feature points

## Commands

```bash
# Environment
conda create -n tsformer-vo python==3.8.0
conda activate tsformer-vo
pip install -r requirements.txt

# Training (edit args/model_params dicts in train.py first)
python train.py

# Inference (edit checkpoint_path, checkpoint_name, sequences in predict_poses.py)
python predict_poses.py

# Visualization
python plot_results.py

# Type checking
basedpyright
```

## Configuration

Configuration uses Python dicts in each script (not argparse). Key parameters in `train.py`:

```python
args = {
    "window_size": 3,      # frames per window
    "overlap": 2,          # overlapping frames between windows
    "use_dual_quaternion": True,  # 7-dim output vs 6-dim Euler+trans
    "weighted_loss": None,  # float to weight rotation vs translation
    "checkpoint_path": "checkpoints/Exp53",
    # ...
}

model_params = {
    "image_size": (224, 672),
    "patch_size": 16,
    "depth": 16,           # transformer depth
    "embed_dim": 192,      # tiny variant (small=384, base=768)
    # ...
}
```

## Data Layout

```
data/
├── sequences_jpg/    # KITTI sequences (00-10)
│   └── {seq}/image_{0,1,2,3}/*.png
└── poses/            # ground truth poses
    └── {seq}.txt
```

## Model Output

- Output shape: `(batch, window_size-1, 6)` for Euler+trans or `(batch, window_size-1, 7)` for dual quaternion
- 6 DoF per frame pair (3 rotation + 3 translation)
- Ground truth is normalized relative pose between consecutive frames
