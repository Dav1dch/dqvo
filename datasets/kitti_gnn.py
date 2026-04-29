"""
KITTI dataset for GNN-based Bundle Adjustment.

Loads KITTI images, extracts ORB features, builds 2D->3D tracks via optical flow,
and provides windows of frames with associated camera poses and point tracks.
"""

import glob
import os

import cv2
import numpy as np
import torch

from utils.gnn_ba import rotation_to_euler


def load_kitti_intrinsics(calib_path):
    """Load KITTI calibration file and return intrinsics dict."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    p2_line = lines[2].strip().split()
    fx = float(p2_line[1])
    cx = float(p2_line[3])
    fy = float(p2_line[6])
    cy = float(p2_line[7])
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}


def load_kitti_poses(poses_path):
    """Load KITTI ground truth poses from text file."""
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            poses.append(np.array(values).reshape(3, 4))
    return poses


class KITTIFeatureDataset(torch.utils.data.Dataset):
    """
    KITTI dataset that extracts 2D features and builds tracks for bundle adjustment.

    Data flow:
    1. Load sequence images and calibration/pose data
    2. Create overlapping windows of frames (e.g., windows of 3 frames with overlap=2)
    3. For each window: extract ORB features, track them via optical flow to build tracks
    4. Output: images, keypoints, tracks (2D observations), global_poses, intrinsics

    Track format: List of tracks, each track is [(frame_idx, u, v), ...]
    - frame_idx: which frame in the window (0, 1, 2 for window_size=3)
    - u, v: pixel coordinates
    """
    def __init__(self, data_path, gt_path, sequence='03',
                 window_size=3, overlap=2, max_points=200):
        self.data_path = data_path
        self.gt_path = gt_path
        self.sequence = sequence
        self.window_size = window_size  # Number of frames per window
        self.overlap = overlap            # Overlapping frames between adjacent windows
        self.max_points = max_points     # Max ORB features per frame

        self.seq_dir = os.path.join(data_path, sequence)
        self.image_dir = os.path.join(self.seq_dir, 'image_2')
        self.calib_path = os.path.join(self.seq_dir, 'calib.txt')
        self.poses_path = os.path.join(gt_path, f'{sequence}.txt')

        # Load calibration (intrinsics K) and ground truth poses
        self.K = load_kitti_intrinsics(self.calib_path)
        self.gt_poses = load_kitti_poses(self.poses_path)

        # Get all image paths for this sequence
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.num_frames = len(self.image_paths)

        # Pre-compute window indices (e.g., [0,1,2], [1,2,3], ... for window=3, overlap=2)
        self.windows = self._create_windows()

    def _create_windows(self):
        """
        Create overlapping windows of frame indices.

        Example for window_size=3, overlap=2:
        step = 3 - 2 = 1
        windows = [[0,1,2], [1,2,3], [2,3,4], ...]
        """
        windows = []
        step = self.window_size - self.overlap
        for start in range(0, self.num_frames - self.window_size + 1, step):
            end = start + self.window_size
            windows.append(list(range(start, end)))
        return windows

    def __len__(self):
        """Number of windows in the sequence."""
        return len(self.windows)

    def _extract_features(self, image):
        """Extract ORB features from image (wrapper around cv2.ORB)."""
        orb = cv2.ORB_create(nfeatures=self.max_points)
        kp, des = orb.detectAndCompute(image, None)
        if des is None or len(kp) == 0:
            return np.array([]), np.array([])
        pts = np.array([p.pt for p in kp], dtype=np.float32)
        return pts, des

    def _match_features_knn(self, des1, des2):
        """Match features using KNN with ratio test (Lowe's ratio)."""
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return [], []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn_matches = bf.knnMatch(des1, des2, k=2)
        matches = []
        for m_n in knn_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    matches.append(m)
        return matches

    def _match_features_optical_flow(self, img1, img2, pts1, search_window=21):
        """
        Match features using optical flow (Lucas-Kanade).

        More robust for consecutive video frames than ORB matching.
        Filters out:
        - Static points (flow < 0.5 pixels)
        - Erratic motion (flow > 100 pixels)
        - High error points from LK
        """
        if len(pts1) == 0:
            return np.array([]), np.array([])

        pts1_float = pts1.astype(np.float32)
        pts2, status, err = cv2.calcOpticalFlowPyrLK(
            img1, img2, pts1_float, None,
            winSize=(search_window, search_window),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if status is None:
            return np.array([]), np.array([])

        valid = status.flatten() > 0
        if err is not None:
            err_flat = err.flatten()
            err_threshold = np.percentile(err_flat[valid], 75) if valid.any() else 1000
            valid &= (err_flat < err_threshold)

        if valid.any():
            flow = pts2 - pts1_float
            flow_mag = np.linalg.norm(flow, axis=1)
            flow_threshold = 0.5  # pixels
            valid &= (flow_mag > flow_threshold)
            flow_threshold_max = 100  # pixels
            valid &= (flow_mag < flow_threshold_max)

        return pts1[valid], pts2[valid]

    def _build_tracks_with_flow(self, images, keypoints_list):
        """
        Build 2D->3D tracks by tracking ORB features with optical flow.

        Algorithm:
        1. Initialize tracks from first frame's ORB keypoints
        2. For each subsequent frame, use optical flow to track each point
        3. If tracking fails (status=0), terminate that track
        4. Keep tracks with at least 2 observations (needed for triangulation)

        Track format (returned): [(frame_idx, u, v), ...]
        - frame_idx: relative index within the window (0, 1, 2, ...)
        - u, v: pixel coordinates
        """
        tracks = []
        num_frames = len(images)

        if len(keypoints_list[0]) == 0:
            return []

        # Initialize tracks with first frame keypoints
        for kp_idx, kp in enumerate(keypoints_list[0]):
            tracks.append([(0, float(kp[0]), float(kp[1]))])

        # Track through subsequent frames using optical flow
        for frame_idx in range(1, num_frames):
            # Get previous frame's track endpoints (last observed position)
            prev_pts = np.array([[t[-1][1], t[-1][2]] for t in tracks], dtype=np.float32)

            if len(prev_pts) == 0:
                break

            prev_img = images[frame_idx - 1]
            curr_img = images[frame_idx]

            curr_pts, status = self._track_points_optical_flow(prev_img, curr_img, prev_pts)

            # Update tracks
            new_tracks = []
            track_idx = 0
            for track in tracks:
                if track_idx < len(status) and status[track_idx] > 0:
                    track.append((frame_idx, curr_pts[track_idx][0], curr_pts[track_idx][1]))
                    new_tracks.append(track)
                elif len(track) >= 2:
                    # Track lost but long enough to keep
                    new_tracks.append(track)
                track_idx += 1
            tracks = new_tracks

            if len(tracks) < 5:  # Too few tracks, stop early
                break

        return [t for t in tracks if len(t) >= 2]

    def _track_points_optical_flow(self, img1, img2, pts, search_window=21):
        """Track points using Lucas-Kanade optical flow."""
        pts = pts.astype(np.float32)
        pts2, status, err = cv2.calcOpticalFlowPyrLK(
            img1, img2, pts, None,
            winSize=(search_window, search_window),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        valid = status.flatten() > 0
        if err is not None:
            err_flat = err.flatten()
            valid &= (err_flat < 100)  # Reject high error points

        return pts2, valid

    def _build_tracks(self, keypoints_list, matches_list):
        """
        Build tracks from ORB feature matches (legacy method, replaced by optical flow).
        Kept for reference - prefer _build_tracks_with_flow for better results.
        """
        tracks = []
        num_frames = len(keypoints_list)

        for kp_idx, kp in enumerate(keypoints_list[0]):
            tracks.append([(0, kp_idx, float(kp[0]), float(kp[1]))])

        for frame_idx in range(1, num_frames):
            frame_matches = matches_list[frame_idx - 1]
            prev_to_curr = {}
            for m in frame_matches:
                prev_to_curr[m.queryIdx] = m.trainIdx

            new_tracks = []
            for track in tracks:
                last_frame, last_kp_idx, last_u, last_v = track[-1]
                if int(last_kp_idx) in prev_to_curr:
                    curr_kp_idx = prev_to_curr[int(last_kp_idx)]
                    curr_kp = keypoints_list[frame_idx][curr_kp_idx]
                    track.append((frame_idx, curr_kp_idx, float(curr_kp[0]), float(curr_kp[1])))
                    new_tracks.append(track)
                elif len(track) >= 2:
                    new_tracks.append(track)
            tracks = new_tracks

        return [t for t in tracks if len(t) >= 2]

    def __getitem__(self, idx):
        """
        Get a single window sample for training.

        Returns dict:
        - images: list of cv2 grayscale images (window_size frames)
        - keypoints: list of ORB keypoints per frame
        - tracks: built via optical flow, format [(frame_idx, u, v), ...]
        - observations: flattened tracks for graph building, [(track_idx, frame_idx, u, v), ...]
        - global_poses: 4x4 ground truth poses for each frame in window
        - window_indices: absolute frame indices in original sequence
        - K: camera intrinsics dict
        """
        window_indices = self.windows[idx]

        # Load images for this window
        images = []
        for frame_idx in window_indices:
            img = cv2.imread(self.image_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {self.image_paths[frame_idx]}")
            images.append(img)

        # Extract ORB features for each frame
        keypoints_list = []
        for img in images:
            pts, _ = self._extract_features(img)
            keypoints_list.append(pts)

        # Build 2D tracks using optical flow
        tracks = self._build_tracks_with_flow(images, keypoints_list)

        # Flatten tracks into observation list for graph construction
        observations = []
        for track_idx, track in enumerate(tracks):
            for obs in track:
                frame_idx, u, v = obs
                observations.append((track_idx, frame_idx, u, v))

        # Convert absolute 3x4 poses to 4x4 format
        abs_poses_4x4 = []
        for pose_3x4 in [self.gt_poses[i] for i in window_indices]:
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :4] = pose_3x4
            abs_poses_4x4.append(pose_4x4)

        # Compute RELATIVE poses: T_rel_ij = inv(T_i) @ T_j
        # Returns window_size-1 relative poses between consecutive frames
        # Convert to 6-DOF representation: [euler_z, euler_y, euler_x, t_x, t_y, t_z]
        relative_poses_6dof = []
        for i in range(len(abs_poses_4x4) - 1):
            T_rel = np.linalg.inv(abs_poses_4x4[i]) @ abs_poses_4x4[i + 1]
            R = T_rel[:3, :3]
            t = T_rel[:3, 3]
            euler = rotation_to_euler(R, seq='zyx')  # [z, y, x] in radians
            pose_6dof = np.concatenate([euler, t])  # (6,)
            relative_poses_6dof.append(pose_6dof)

        return {
            'images': images,
            'keypoints': keypoints_list,
            'tracks': tracks,
            'observations': observations,
            'global_poses': np.array(relative_poses_6dof),  # 6-DOF: (window_size-1, 6)
            'abs_poses': abs_poses_4x4,                       # Absolute 4x4 poses for triangulation
            'window_indices': window_indices,
            'K': self.K,
            'sample_idx': idx,
        }
