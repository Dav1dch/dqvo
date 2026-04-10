# Plan: Convert to 6-DOF Pose Representation Throughout GNN-BA Pipeline

## Objective
Convert the GNN-BA system to use 6-DOF (Euler angles + translation) representation consistently:
1. Dataset should output poses as 6-DOF vectors instead of 4x4 matrices
2. GNN model should accept 6-DOF as input and output 6-DOF deltas/poses
3. During loss computation, convert predicted 6-DOF to 3x4 transformation matrix for reprojection

## Current State Analysis

### Existing 6-DOF Usage
- The system already uses 6-DOF (euler3 + t3) as intermediate representation:
  - `poses_to_camera_features()`: converts 4x4 poses → normalized 6-DOF
  - GNN input: 6-DOF features (line 135 in gnn_ba.py)
  - GNN output: 6-DOF predictions (line 161-162 in gnn_ba.py)
  - Training: GNN output converted to 4x4 for reprojection (train_gnn_ba.py lines 212-223)

### Problem
- Dataset returns 4x4 matrices (`global_poses` and `abs_poses`)
- This requires unnecessary conversion before GNN input
- The conversion back-and-forth is already happening but could be streamlined

## Files to Modify

### 1. `datasets/kitti_gnn.py`
**Changes:**
- In `__getitem__`, convert relative poses from 4x4 to 6-DOF vectors
- Return `global_poses` as 6-DOF array: (window_size-1, 6) with [euler_z, euler_y, euler_x, t_x, t_y, t_z]
- Keep `abs_poses` as 4x4 for evaluation (or also convert to 6-DOF if needed)

**Implementation:**
```python
# After computing relative_poses (4x4 matrices):
relative_poses_6dof = []
for rel_pose in relative_poses:
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]
    euler = rotation_to_euler(R, seq='zyx')  # [z, y, x]
    pose_6dof = np.concatenate([euler, t])
    relative_poses_6dof.append(pose_6dof)

return {
    ...
    'global_poses': np.array(relative_poses_6dof),  # (window_size-1, 6)
    ...
}
```

### 2. `train_gnn_ba.py`
**Changes:**
- Remove conversion from 4x4 to 6-DOF since dataset will provide 6-DOF directly
- Update `poses_to_camera_features()` call to accept 6-DOF poses directly (need to modify or bypass)
- Keep GNN output → 4x4 conversion for reprojection (already correct)

**Implementation options:**
- Option A: Modify `poses_to_camera_features()` to accept 6-DOF input directly
- Option B: Skip `poses_to_camera_features()` and normalize 6-DOF directly in training loop

**Recommended (Option A):** Extend `poses_to_camera_features()` to handle both 4x4 and 6-DOF inputs:
```python
def poses_to_camera_features(poses, mean_angles, std_angles, mean_t, std_t):
    # If already 6-DOF, just normalize
    if isinstance(poses, np.ndarray) and poses.ndim == 2 and poses.shape[1] == 6:
        euler = poses[:, :3]
        t = poses[:, 3:]
        euler_norm = (euler - mean_angles) / std_angles
        t_norm = (t - mean_t) / std_t
        return np.concatenate([euler_norm, t_norm], axis=1)
    # Else handle 4x4 matrices as before
    ...

```

### 3. `utils/gnn_ba.py`
**Changes:**
- Update `poses_to_camera_features()` to accept both 4x4 matrices and 6-DOF arrays
- Add helper function `pose_6dof_to_matrix()` to convert 6-DOF to 4x4 (or reuse existing logic)
- Ensure `denormalize_poses()` returns 6-DOF or 4x4 as needed (currently returns 4x4)

**Implementation:**
```python
def pose_6dof_to_matrix(pose_6dof, seq='zyx'):
    """Convert 6-DOF [euler(3), t(3)] to 4x4 transformation matrix."""
    euler = pose_6dof[:3]
    t = pose_6dof[3:]
    R = euler_to_rotation(euler, seq=seq)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose

def poses_to_camera_features(poses, mean_angles, std_angles, mean_t, std_t):
    # Handle 6-DOF input
    if isinstance(poses, np.ndarray):
        if poses.ndim == 2 and poses.shape[1] == 6:
            euler = poses[:, :3]
            t = poses[:, 3:]
            euler_norm = (euler - mean_angles) / std_angles
            t_norm = (t - mean_t) / std_t
            return np.concatenate([euler_norm, t_norm], axis=1)
        # ... existing 4x4 handling ...
```

## Verification Steps

1. **Dataset output shape**: Verify `global_poses` shape is (window_size-1, 6)
2. **GNN input**: Check camera features shape is (N_cam, 6) after normalization
3. **GNN output**: Should be (N_cam, 6) - already correct
4. **Loss computation**: Convert GNN output (6-DOF) to 4x4 matrices using `euler_to_rotation_torch()` and verify reprojection error decreases
5. **Training**: Run a few batches to ensure no shape mismatches
6. **Validation**: Compare reprojection errors before/after GNN optimization

## Expected Benefits

- Cleaner data flow: dataset → 6-DOF → GNN → 6-DOF → convert for reprojection
- Reduced unnecessary conversions (4x4 → 6-DOF → normalize)
- Explicit 6-DOF representation throughout the pipeline
- Easier to modify pose parameterization (e.g., use quaternions) in the future

## Risks and Mitigations

- **Risk**: Other functions (triangulation) expect 4x4 poses
  - **Mitigation**: Keep `abs_poses` as 4x4 for triangulation; only `global_poses` (relative) become 6-DOF
- **Risk**: `denormalize_poses()` currently returns 4x4; may need to adjust if used elsewhere
  - **Mitigation**: Check usage; currently only used in training/validation for VO poses (4x4 needed for accumulation). This function can stay unchanged for VO outputs.