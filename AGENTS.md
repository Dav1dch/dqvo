# Agent Guidelines for TSformer-VO

PyTorch Visual Odometry project using Transformer architectures (TimeSformer, Mamba) for monocular camera pose estimation.

## Project Overview

- **Language**: Python 3.8+ | **Framework**: PyTorch 1.10.1
- **Key Libraries**: einops, tensorboard, torchvision, numpy, tqdm, pandas, OpenCV

## Workflow Rules

### Before Editing Code
**IMPORTANT**: Before making any code changes, you MUST create a `plan.md` file that outlines:
1. What needs to be changed and why
2. The specific files and locations that will be modified
3. Step-by-step implementation plan
4. How to verify the change works correctly

```markdown
# Plan: [Brief Task Description]

## Objective
[What this task accomplishes]

## Files to Modify
- `file1.py` - [what change]
- `file2.py` - [what change]

## Implementation Steps
1. [First step]
2. [Second step]
3. [Third step]

## Verification
[How to confirm the change works]
```

## Build/Lint/Test Commands

```bash
# Environment Setup
conda create -n tsformer-vo python==3.8.0 && conda activate tsformer-vo
pip install -r requirements.txt

# Type Checking
basedpyright

# Running Code
python train.py              # Training (edit hyperparameters in train.py)
python predict_poses.py       # Inference (edit paths in file)
python test.py                # SuperPoint test script
python gnn_ba_test.py         # GNN Bundle Adjustment test
python train_tri.py           # Triangulation training
python train_gnn_ba.py        # GNN BA training
python plot_results.py        # Trajectory visualization

# Single file execution
python <file_path>.py
```

## Code Style Guidelines

### Import Organization
```python
# Standard library imports
import json
import os
import pickle
from functools import partial

# Third-party imports (alphabetically)
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# Local imports
from build_model import build_model
from datasets.kitti import KITTI
from timesformer.models.losses import dual_quaternion_loss
```

### Formatting
- **Indentation**: 4 spaces | **Line length**: 100-120 target, 150 soft max
- **Spacing**: One blank line between top-level definitions
- **No trailing whitespace**

### Type Annotations
- Use type hints for non-obvious parameters/returns
- `Optional[T]` over `Union[T, None]`, `List[T]`, `Dict[K, V]`, `Tuple[T, ...]`

```python
def train_epoch(model, train_loader, criterion, optimizer, epoch, 
                tensorboard_writer, args) -> float: ...

def compute_loss(y_hat: torch.Tensor, y: torch.Tensor, 
                 criterion, args: Dict) -> torch.Tensor: ...
```

### Naming Conventions
- **Functions/methods**: `snake_case` | **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE` | **Private members**: `_leading_underscore`
- **Torch modules**: `self.conv1`, `self.fc`

```python
class KITTI(torch.utils.data.Dataset):
    def __init__(self, data_path: str, window_size: int = 3):
        self.data_path = data_path
        self._frame_id = 0

MAX_KEYPOINTS = 100
DEFAULT_BATCH_SIZE = 8
```

### Error Handling
- Use try/except for specific expected errors with meaningful messages
- Use `torch.no_grad()` during inference

```python
try:
    checkpoint = torch.load(path, weights_only=True)
except FileNotFoundError as e:
    print(f"Checkpoint not found: {e}")
    raise
```

### GPU/CUDA Handling
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Docstrings
- Triple double quotes `"""` | Args/Returns/Raises for complex functions

```python
def rotationMatrixToQuaternion3(m):
    """
    Converts a 3x3 rotation matrix to a quaternion.
    Args:
        m: 3x3 rotation matrix (np.matrix)
    Returns:
        q: Quaternion as (4,) ndarray [qx, qy, qz, qw]
    """
```

### PyTorch Model Conventions
- Inherit from `nn.Module`, implement `forward()`, call `super().__init__()` first
- Use `nn.ModuleList`/`nn.ModuleDict` for layer collections

```python
class SuperPoint(nn.Module):
    default_config = {'descriptor_dim': 256, 'nms_radius': 4}
    
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
    
    def forward(self, data):
        return {'keypoints': keypoints, 'scores': scores, 'descriptors': descriptors}
```

### Data Loading
- Inherit `torch.utils.data.Dataset`, implement `__len__` and `__getitem__`
- DataLoader with `num_workers=4-8`

```python
class KITTI(torch.utils.data.Dataset):
    def __len__(self): return len(self.windowed_data["w_idx"].unique())
    def __getitem__(self, idx): return imgs, pt1, pt2, y, n_valid
```

### Configuration
- Python dictionaries (not argparse), store args as `.pkl` and `.txt`

```python
args = {"data_dir": "data", "bsize": 8, "window_size": 3, "optimizer": "Adam", "lr": 1e-5}
```

### File Paths
- `os.path.join()` for cross-platform | `Path` from pathlib

```python
path = Path(__file__).parent / 'weights' / 'model.pth'
torch.load(str(path))
```

## Directory Structure
```
dqvo/
├── build_model.py          # Model construction
├── train.py                # Training script
├── test.py                 # SuperPoint test
├── predict_poses.py        # Inference
├── plot_results.py         # Trajectory visualization
├── superpoint.py           # SuperPoint model
├── losses/                  # Custom loss functions
│   └── triangulation_loss.py
├── datasets/               # Dataset loaders
│   ├── kitti.py, kitti_dual.py, kitti_gnn.py, kitti_fp.py
│   └── utils.py
└── timesformer/            # TimeSformer model code
    ├── models/             # Model definitions (vit, mamba, deepvo, etc.)
    └── datasets/           # Data loading utilities
```

## Common Issues
1. **CUDA OOM**: Use `torch.no_grad()` during inference, reduce batch size
2. **DataLoader workers**: Start with `num_workers=4`
3. **Checkpoint loading**: Use `weights_only=True`
4. **Mixed precision**: Not used; use FP32 consistently
5. **Memory leaks**: Clear CUDA cache with `torch.cuda.empty_cache()`
