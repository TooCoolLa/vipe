# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Merge multiple submaps into a single global coordinate frame using Umeyama alignment.

This module aligns overlapping camera trajectories from consecutive submaps
and outputs a unified pose list in global coordinates.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from rich.logging import RichHandler
from rich.progress import track

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_submap_name(submap_dir: Path) -> Tuple[int, int]:
    """
    Parse submap directory name to extract start and end frame indices.
    
    Example: "0_700" -> (0, 700)
    """
    match = re.match(r"(\d+)_(\d+)", submap_dir.name)
    if not match:
        raise ValueError(f"Invalid submap directory name: {submap_dir.name}")
    return int(match.group(1)), int(match.group(2))


def load_submap_poses(submap_dir: Path) -> np.ndarray:
    """
    Load camera poses from a submap's result directory.
    
    Returns:
        np.ndarray: Poses with shape (N, 4, 4) in c2w format
    """
    pose_file = submap_dir / "result" / "pose" / "images.npz"
    
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    
    data = np.load(pose_file)
    
    # Check if poses are stored as quaternion + translation or as matrices
    if "c2w" in data:
        poses = data["c2w"]  # Shape: (N, 4, 4)
    elif "data" in data:
        # ViPE format: 'data' contains (N, 4, 4) matrices
        poses = data["data"]  # Shape: (N, 4, 4)
    elif "qvec" in data and "tvec" in data:
        # Convert from quaternion + translation to 4x4 matrix
        qvec = data["qvec"]  # (N, 4) - w, x, y, z
        tvec = data["tvec"]  # (N, 3)
        N = len(qvec)
        poses = np.zeros((N, 4, 4))
        
        for i in range(N):
            # Convert quaternion to rotation matrix
            w, x, y, z = qvec[i]
            R = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
            ])
            poses[i, :3, :3] = R
            poses[i, :3, 3] = tvec[i]
            poses[i, 3, 3] = 1.0
    else:
        raise ValueError(f"Unknown pose format in {pose_file}. Available keys: {list(data.keys())}")
    
    return poses


def umeyama_alignment(src_points: np.ndarray, dst_points: np.ndarray, with_scale: bool = False) -> np.ndarray:
    """
    Umeyama algorithm for finding the optimal similarity transformation.
    
    Given two sets of 3D points, compute the transformation matrix T such that:
        dst_points ≈ T @ src_points
    
    Args:
        src_points: Source points (N, 3)
        dst_points: Destination points (N, 3)
        with_scale: If True, compute scale; if False, only rotation and translation
    
    Returns:
        T: 4x4 transformation matrix
    
    Reference:
        Umeyama, S. (1991). Least-squares estimation of transformation parameters
        between two point patterns. IEEE TPAMI, 13(4), 376-380.
    """
    assert src_points.shape == dst_points.shape, "Point sets must have the same shape"
    assert src_points.shape[1] == 3, "Points must be 3D"
    
    N = src_points.shape[0]
    
    # Compute centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    
    # Center the points
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # Compute covariance matrix
    H = src_centered.T @ dst_centered / N
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        logger.warning("Reflection detected, correcting...")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    if with_scale:
        src_var = np.var(src_centered, axis=0).sum()
        scale = np.trace(np.diag(S)) / src_var
    else:
        scale = 1.0
    
    # Compute translation
    t = dst_centroid - scale * R @ src_centroid
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    
    return T


def compute_alignment_error(src_poses: np.ndarray, dst_poses: np.ndarray, T: np.ndarray) -> float:
    """
    Compute alignment error after applying transformation.
    
    Args:
        src_poses: Source poses (N, 4, 4)
        dst_poses: Destination poses (N, 4, 4)
        T: Transformation matrix (4, 4)
    
    Returns:
        RMS error in meters
    """
    N = src_poses.shape[0]
    errors = []
    
    for i in range(N):
        src_transformed = T @ src_poses[i]
        position_error = np.linalg.norm(src_transformed[:3, 3] - dst_poses[i, :3, 3])
        errors.append(position_error)
    
    return np.sqrt(np.mean(np.array(errors) ** 2))


def align_submaps(root_dir: Path, overlap_frames: int = 10, with_scale: bool = False, output_file: Path = None) -> List[Dict]:
    """
    Align all submaps in the root directory into a global coordinate frame.
    
    Args:
        root_dir: Root directory containing submaps (e.g., data1/)
        overlap_frames: Number of overlapping frames between consecutive submaps
        with_scale: If True, estimate scale; if False, assume same scale
        output_file: Optional output JSON file path. If None, no file is written.
    
    Returns:
        List of dicts containing frame information in global coordinates
    """
    # Find all submap directories and sort by starting frame number
    submap_dirs = [d for d in root_dir.iterdir() if d.is_dir() and re.match(r"\d+_\d+", d.name)]
    
    if len(submap_dirs) == 0:
        raise ValueError(f"No submaps found in {root_dir}")
    
    # Sort by starting frame index (numeric, not lexicographic)
    submap_dirs = sorted(submap_dirs, key=lambda d: int(d.name.split('_')[0]))
    
    logger.info(f"Found {len(submap_dirs)} submaps: {[d.name for d in submap_dirs]}")
    
    # Load poses for all submaps
    submap_poses = {}
    submap_ranges = {}
    
    for submap_dir in track(submap_dirs, description="Loading submaps"):
        start_idx, end_idx = parse_submap_name(submap_dir)
        poses = load_submap_poses(submap_dir)
        submap_poses[submap_dir.name] = poses
        submap_ranges[submap_dir.name] = (start_idx, end_idx)
        logger.info(f"Loaded {submap_dir.name}: {len(poses)} poses")
    
    # Initialize global transformation for each submap
    # First submap is the reference (identity transformation)
    global_transforms = {submap_dirs[0].name: np.eye(4)}
    
    # Align consecutive submaps using overlapping frames
    for i in range(len(submap_dirs) - 1):
        curr_submap = submap_dirs[i].name
        next_submap = submap_dirs[i + 1].name
        
        curr_poses = submap_poses[curr_submap]
        next_poses = submap_poses[next_submap]
        
        logger.info(f"\nAligning {next_submap} to {curr_submap}")
        curr_start, curr_end = submap_ranges[curr_submap]
        next_start, next_end = submap_ranges[next_submap]
        
        # Calculate overlap region based on global frame indices
        overlap_start = max(curr_start, next_start)
        overlap_end = min(curr_end, next_end)
        overlap_size = overlap_end - overlap_start
        
        if overlap_size < overlap_frames:
            logger.warning(f"Insufficient overlap between {curr_submap} and {next_submap}: {overlap_size} frames < {overlap_frames}")
            logger.warning(f"  {curr_submap}: {curr_start}-{curr_end}")
            logger.warning(f"  {next_submap}: {next_start}-{next_end}")
            logger.warning(f"  Using identity transformation")
            global_transforms[next_submap] = global_transforms[curr_submap].copy()
            continue
        
        # Head-to-tail alignment: align first N frames of next_submap to last N frames of curr_submap
        # Calculate local indices for the overlap region
        curr_overlap_local_start = overlap_start - curr_start
        curr_overlap_local_end = overlap_end - curr_start
        next_overlap_local_start = overlap_start - next_start
        next_overlap_local_end = overlap_end - next_start
        
        # Extract the actual overlapping poses (local coordinates)
        curr_overlap = curr_poses[curr_overlap_local_start:curr_overlap_local_end]
        next_overlap = next_poses[next_overlap_local_start:next_overlap_local_end]
        
        # Take only the requested number of overlap frames
        if len(curr_overlap) > overlap_frames:
            curr_overlap = curr_overlap[-overlap_frames:]
            next_overlap = next_overlap[-overlap_frames:]
        
        # Extract camera positions from poses
        curr_positions = curr_overlap[:, :3, 3]  # (N, 3) in local coords
        next_positions = next_overlap[:, :3, 3]  # (N, 3) in local coords
        
        # Transform curr_overlap positions to global coordinates using accumulated transform
        curr_positions_global = []
        for pos in curr_positions:
            pos_hom = np.append(pos, 1.0)
            pos_global = (global_transforms[curr_submap] @ pos_hom)[:3]
            curr_positions_global.append(pos_global)
        curr_positions_global = np.array(curr_positions_global)
        
        # Compute transformation: align next_positions (local) to curr_positions (global)
        # This gives us the transformation from next_submap's local coords to global coords
        T_align = umeyama_alignment(next_positions, curr_positions_global, with_scale=with_scale)
        
        # Compute alignment error
        next_positions_transformed = []
        for pos in next_positions:
            pos_hom = np.append(pos, 1.0)
            pos_transformed = (T_align @ pos_hom)[:3]
            next_positions_transformed.append(pos_transformed)
        next_positions_transformed = np.array(next_positions_transformed)
        
        errors = np.linalg.norm(next_positions_transformed - curr_positions_global, axis=1)
        rms_error = np.sqrt(np.mean(errors ** 2))
        logger.info(f"Alignment error: {rms_error:.4f} meters (RMS)")
        
        # Store the global transformation for next submap
        global_transforms[next_submap] = T_align
        
        logger.info(f"Transformation matrix:\n{T_align}")
    
    # Build final output: all frames in global coordinates
    output_frames = []
    global_frame_id = 0
    
    for submap_dir in track(submap_dirs, description="Building output"):
        submap_name = submap_dir.name
        poses = submap_poses[submap_name]
        T_global = global_transforms[submap_name]
        
        for local_idx, pose in enumerate(poses):
            # Transform pose to global frame
            pose_global = T_global @ pose
            
            frame_info = {
                "frameid": global_frame_id,
                "matrix": pose_global.tolist(),
                "datapath": f"{submap_name}/result",
                "indexIndata": local_idx
            }
            output_frames.append(frame_info)
            global_frame_id += 1
    
    logger.info(f"\nTotal frames in global coordinate system: {len(output_frames)}")
    
    # Write to file if output_file is specified
    if output_file is not None:
        logger.info(f"Saving {len(output_frames)} frames to {output_file}")
        with open(output_file, "w") as f:
            json.dump(output_frames, f, indent=2)
        logger.info(f"✓ Saved to {output_file}")
    
    return output_frames


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple submaps into a unified global coordinate frame using Umeyama alignment."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing submaps (e.g., data1/)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: <root_dir>/camera_poses.json)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Number of overlapping frames between consecutive submaps (default: 10)"
    )
    parser.add_argument(
        "--with-scale",
        action="store_true",
        help="Estimate scale factor (use if submaps have different scales)"
    )
    
    args = parser.parse_args()
    
    root_dir = args.root_dir
    if not root_dir.exists():
        logger.error(f"Root directory not found: {root_dir}")
        return
    
    output_path = args.output or (root_dir / "camera_poses.json")
    
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Overlap frames: {args.overlap}")
    logger.info(f"Estimate scale: {args.with_scale}")
    
    # Align submaps
    frames = align_submaps(root_dir, overlap_frames=args.overlap, with_scale=args.with_scale)
    
    # Save to JSON
    logger.info(f"\nSaving {len(frames)} frames to {output_path}")
    with open(output_path, "w") as f:
        json.dump(frames, f, indent=2)
    
    logger.info("✅ Done!")


if __name__ == "__main__":
    main()
