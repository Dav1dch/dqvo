"""
Visualize feature tracks across frames in a KITTI window.

This script loads a sample from the KITTIFeatureDataset and visualizes
the optical flow tracks showing how feature points are matched across
consecutive frames in a window.

Usage:
    python visualize_tracks.py --sequence 03 --sample_idx 0 --max_tracks 50
"""

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from datasets.kitti_gnn import KITTIFeatureDataset


def draw_tracks_on_image(img, tracks, frame_idx, colors, max_tracks=50):
    """
    Draw tracks on an image for a specific frame.

    Args:
        img: grayscale or color image
        tracks: list of tracks, each track is [(frame_idx, u, v), ...]
        frame_idx: which frame to draw points for
        colors: list of colors for each track
        max_tracks: maximum number of tracks to draw

    Returns:
        color image with tracks drawn
    """
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    drawn = 0
    for track_idx, track in enumerate(tracks):
        if drawn >= max_tracks:
            break

        # Find observation in this frame
        for obs in track:
            if obs[0] == frame_idx:
                u, v = int(obs[1]), int(obs[2])
                color = colors[track_idx % len(colors)]

                # Draw point
                cv2.circle(img_color, (u, v), 4, color, -1)
                cv2.circle(img_color, (u, v), 6, (255, 255, 255), 1)

                # Draw track ID
                # cv2.putText(img_color, str(track_idx), (u+5, v-5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                drawn += 1
                break

    return img_color


def draw_track_lines(img, tracks, frame_idx_prev, frame_idx_curr, colors, max_tracks=50):
    """
    Draw lines connecting track points between two consecutive frames.

    Args:
        img: color image to draw on (should be the current frame)
        tracks: list of tracks
        frame_idx_prev: previous frame index
        frame_idx_curr: current frame index
        colors: list of colors for each track
        max_tracks: maximum tracks to draw

    Returns:
        image with track lines drawn
    """
    img_result = img.copy()
    drawn = 0

    for track_idx, track in enumerate(tracks):
        if drawn >= max_tracks:
            break

        pt_prev = None
        pt_curr = None

        for obs in track:
            if obs[0] == frame_idx_prev:
                pt_prev = (int(obs[1]), int(obs[2]))
            if obs[0] == frame_idx_curr:
                pt_curr = (int(obs[1]), int(obs[2]))

        if pt_prev is not None and pt_curr is not None:
            color = colors[track_idx % len(colors)]
            # Draw line from previous to current position
            cv2.arrowedLine(img_result, pt_prev, pt_curr, color, 2, tipLength=0.3)
            drawn += 1

    return img_result


def visualize_window_tracks(dataset, sample_idx, save_dir, max_tracks=50):
    """
    Visualize tracks for a single window sample.

    Creates a multi-panel figure showing:
    1. All frames in the window with track points
    2. Consecutive frame pairs with flow arrows
    3. Track statistics
    """
    sample = dataset[sample_idx]
    images = sample["images"]
    tracks = sample["tracks"]
    window_indices = sample["window_indices"]
    window_size = len(images)

    # Generate colors for tracks
    random.seed(42)
    colors = [
        (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        for _ in range(max_tracks)
    ]

    # Filter tracks to max_tracks
    display_tracks = tracks[:max_tracks] if len(tracks) > max_tracks else tracks

    # Create figure
    num_subplots = window_size + (window_size - 1)  # frames + flow pairs
    fig_height = 4 * ((num_subplots + 1) // 2)
    fig, axes = plt.subplots(
        (num_subplots + 1) // 2, 2, figsize=(20, fig_height)
    )
    axes = axes.flatten()

    # Plot 1: Frames with track points
    for i in range(window_size):
        img_with_tracks = draw_tracks_on_image(
            images[i], display_tracks, i, colors, max_tracks
        )
        axes[i].imshow(cv2.cvtColor(img_with_tracks, cv2.COLOR_BGR2RGB))
        axes[i].set_title(
            f"Frame {window_indices[i]} (abs) / {i} (rel)\n"
            f"{len([t for t in tracks if any(o[0]==i for o in t)])} points visible"
        )
        axes[i].axis("off")

    # Plot 2: Consecutive frames with flow arrows
    for i in range(window_size - 1):
        img_curr = cv2.cvtColor(images[i + 1], cv2.COLOR_GRAY2BGR)
        img_with_flow = draw_track_lines(
            img_curr, display_tracks, i, i + 1, colors, max_tracks
        )

        # Count tracks visible in both frames
        num_matched = len(
            [
                t
                for t in tracks
                if any(o[0] == i for o in t) and any(o[0] == i + 1 for o in t)
            ]
        )

        axes[window_size + i].imshow(cv2.cvtColor(img_with_flow, cv2.COLOR_BGR2RGB))
        axes[window_size + i].set_title(
            f"Flow: Frame {i} -> {i + 1}\n{num_matched} tracks matched"
        )
        axes[window_size + i].axis("off")

    # Hide unused subplots
    for i in range(num_subplots, len(axes)):
        axes[i].axis("off")

    # Add overall title with statistics
    track_lengths = [len(t) for t in tracks]
    avg_length = np.mean(track_lengths) if track_lengths else 0
    fig.suptitle(
        f"Sample {sample_idx} | Window frames: {window_indices}\n"
        f"Total tracks: {len(tracks)} | "
        f"Avg track length: {avg_length:.1f} | "
        f"Showing: {min(max_tracks, len(tracks))} tracks",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"tracks_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to: {save_path}")

    return save_path


def visualize_track_tracks_on_single_image(dataset, sample_idx, save_dir, max_tracks=30):
    """
    Visualize all track trajectories on a single composite image.

    Creates an image showing the trajectory of each track as colored lines
    across the entire window, useful for understanding track continuity.
    """
    sample = dataset[sample_idx]
    images = sample["images"]
    tracks = sample["tracks"]
    window_indices = sample["window_indices"]
    window_size = len(images)

    # Generate colors
    random.seed(42)
    colors = [
        (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        for _ in range(max_tracks)
    ]

    # Create composite image: stack frames horizontally
    h, w = images[0].shape[:2]
    composite = np.zeros((h, w * window_size, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()
        # Darken the background
        img_color = (img_color * 0.3).astype(np.uint8)
        composite[:, i * w : (i + 1) * w] = img_color

    # Draw tracks as lines across frames
    display_tracks = tracks[:max_tracks] if len(tracks) > max_tracks else tracks

    for track_idx, track in enumerate(display_tracks):
        color = colors[track_idx % len(colors)]

        # Get all points in this track
        points = []
        for obs in track:
            frame_idx, u, v = obs[0], obs[1], obs[2]
            # Offset u by frame position in composite
            u_global = int(u) + frame_idx * w
            points.append((u_global, int(v)))

        # Draw lines connecting consecutive points
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i + 1]
            cv2.line(composite, pt1, pt2, color, 2)
            cv2.circle(composite, pt1, 4, color, -1)

        # Draw last point
        if points:
            cv2.circle(composite, points[-1], 4, color, -1)

    # Add frame separators and labels
    for i in range(1, window_size):
        x = i * w
        cv2.line(composite, (x, 0), (x, h), (255, 255, 255), 2)
        cv2.putText(
            composite,
            f"Frame {window_indices[i]}",
            (x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        composite,
        f"Frame {window_indices[0]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"track_trajectories_{sample_idx}.png")
    cv2.imwrite(save_path, composite)
    print(f"Saved composite to: {save_path}")

    return save_path


def visualize_track_length_distribution(dataset, sample_idx, save_dir):
    """
    Plot distribution of track lengths for a sample.
    """
    sample = dataset[sample_idx]
    tracks = sample["tracks"]

    track_lengths = [len(t) for t in tracks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(track_lengths, bins=range(1, max(track_lengths) + 2), edgecolor="black")
    axes[0].set_xlabel("Track Length (num observations)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Track Length Distribution\nSample {sample_idx}")
    axes[0].grid(True, alpha=0.3)

    # Statistics
    stats_text = (
        f"Total tracks: {len(tracks)}\n"
        f"Mean length: {np.mean(track_lengths):.2f}\n"
        f"Median length: {np.median(track_lengths):.1f}\n"
        f"Max length: {max(track_lengths)}\n"
        f"Min length: {min(track_lengths)}\n"
        f"Length >= 3: {sum(1 for l in track_lengths if l >= 3)}\n"
        f"Length >= 2: {sum(1 for l in track_lengths if l >= 2)}"
    )
    axes[1].text(
        0.1,
        0.5,
        stats_text,
        transform=axes[1].transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
    )
    axes[1].axis("off")
    axes[1].set_title("Statistics")

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"track_lengths_{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to: {save_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Visualize feature tracks")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sequences_jpg",
        help="Path to KITTI sequences",
    )
    parser.add_argument(
        "--gt_path", type=str, default="data/poses", help="Path to KITTI poses"
    )
    parser.add_argument("--sequence", type=str, default="03", help="Sequence number")
    parser.add_argument(
        "--sample_idx", type=int, default=0, help="Sample index to visualize"
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--max_tracks", type=int, default=50, help="Maximum tracks to visualize"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="gnn_ba_output/track_vis",
        help="Output directory",
    )
    parser.add_argument(
        "--window_size", type=int, default=3, help="Window size"
    )
    parser.add_argument(
        "--overlap", type=int, default=2, help="Overlap between windows"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading KITTI dataset for sequence {args.sequence}...")
    dataset = KITTIFeatureDataset(
        data_path=args.data_path,
        gt_path=args.gt_path,
        sequence=args.sequence,
        window_size=args.window_size,
        overlap=args.overlap,
        max_points=200,
    )
    print(f"Dataset size: {len(dataset)} windows")

    # Visualize samples
    for i in range(args.num_samples):
        sample_idx = args.sample_idx + i
        if sample_idx >= len(dataset):
            break

        print(f"\nVisualizing sample {sample_idx}...")

        # Main visualization: frames with track points and flow arrows
        visualize_window_tracks(
            dataset, sample_idx, args.save_dir, args.max_tracks
        )

        # Composite visualization: track trajectories
        visualize_track_tracks_on_single_image(
            dataset, sample_idx, args.save_dir, max(30, args.max_tracks)
        )

        # Track length distribution
        visualize_track_length_distribution(dataset, sample_idx, args.save_dir)

    print(f"\nAll visualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
