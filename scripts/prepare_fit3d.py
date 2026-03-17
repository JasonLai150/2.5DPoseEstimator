#!/usr/bin/env python3
"""
Fit3D Dataset Preparation Script.

Uses IMAR vision datasets tools for proper data loading and skeleton conversion.
Reference: https://github.com/sminchisescu-research/imar_vision_datasets_tools

Fit3D is a fitness activity dataset with 3D pose ground truth.
Website: https://fit3d.imar.ro/

Usage:
    # First, clone IMAR tools and set up Fit3D data:
    git clone https://github.com/sminchisescu-research/imar_vision_datasets_tools.git external/imar_tools

    # Then run this script:
    python scripts/prepare_fit3d.py --data_root ./data/fit3d --output_dir ./data/fit3d_processed
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Fit3D uses 25-joint COCO-style skeleton
# This maps to H36M 17-joint format for evaluation
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]

# 25-joint COCO skeleton used by Fit3D
COCO25_JOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  # 0-4
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',  # 5-8
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',  # 9-12
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',  # 13-16
    'left_big_toe', 'left_small_toe', 'left_heel',  # 17-19
    'right_big_toe', 'right_small_toe', 'right_heel',  # 20-22
    'thorax', 'pelvis'  # 23-24 (derived joints)
]

# Map from COCO25 to H36M 17-joint
# H36M joints: pelvis, rhip, rknee, rankle, lhip, lknee, lankle,
#              spine, thorax, neck, head, lshoulder, lelbow, lwrist,
#              rshoulder, relbow, rwrist
COCO25_TO_H36M = {
    0: 24,   # pelvis (use COCO pelvis)
    1: 12,   # right_hip
    2: 14,   # right_knee
    3: 16,   # right_ankle
    4: 11,   # left_hip
    5: 13,   # left_knee
    6: 15,   # left_ankle
    7: None, # spine (interpolate)
    8: 23,   # thorax
    9: 0,    # neck/nose
    10: None, # head (interpolate from ears)
    11: 5,   # left_shoulder
    12: 7,   # left_elbow
    13: 9,   # left_wrist
    14: 6,   # right_shoulder
    15: 8,   # right_elbow
    16: 10,  # right_wrist
}


def coco25_to_h36m17(poses_coco25: np.ndarray) -> np.ndarray:
    """
    Convert COCO 25-joint poses to H36M 17-joint format.

    Args:
        poses_coco25: Shape (..., 25, 3)

    Returns:
        poses_h36m: Shape (..., 17, 3)
    """
    shape = poses_coco25.shape[:-2]
    poses_h36m = np.zeros(shape + (17, 3), dtype=poses_coco25.dtype)

    for h36m_idx, coco_idx in COCO25_TO_H36M.items():
        if coco_idx is not None:
            poses_h36m[..., h36m_idx, :] = poses_coco25[..., coco_idx, :]

    # Interpolate spine (midpoint between pelvis and thorax)
    poses_h36m[..., 7, :] = (poses_h36m[..., 0, :] + poses_h36m[..., 8, :]) / 2

    # Interpolate head (from nose and ears)
    poses_h36m[..., 10, :] = (
        poses_coco25[..., 0, :] * 0.4 +  # nose
        poses_coco25[..., 3, :] * 0.3 +  # left_ear
        poses_coco25[..., 4, :] * 0.3    # right_ear
    )

    return poses_h36m


def load_fit3d_sequence(sequence_dir: Path) -> dict | None:
    """
    Load a Fit3D sequence using IMAR format.

    Expected structure:
    sequence_dir/
        joints3d_25/*.json  (3D joint positions per frame)
        cameras/*.json      (camera parameters)
        videos/*.mp4        (optional)
    """
    joints_dir = sequence_dir / "joints3d_25"
    cameras_dir = sequence_dir / "cameras"

    if not joints_dir.exists():
        # Try alternative structure
        joints_dir = sequence_dir

    # Find all joint files
    joint_files = sorted(joints_dir.glob("*.json"))
    if not joint_files:
        return None

    all_poses = []

    for jf in joint_files:
        with open(jf) as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            if "joints3d_25" in data:
                poses = np.array(data["joints3d_25"])
            elif "joints" in data:
                poses = np.array(data["joints"])
            else:
                # Try to find the joints array
                for key in data:
                    if isinstance(data[key], list) and len(data[key]) > 0:
                        poses = np.array(data[key])
                        break
        elif isinstance(data, list):
            poses = np.array(data)
        else:
            continue

        all_poses.append(poses)

    if not all_poses:
        return None

    # Stack frames: (num_frames, num_joints, 3) or (num_frames, num_subjects, num_joints, 3)
    poses_3d = np.stack(all_poses, axis=0)

    # If multi-subject, take first subject for now
    if poses_3d.ndim == 4:
        poses_3d = poses_3d[:, 0, :, :]

    # Load camera parameters if available
    camera_params = None
    if cameras_dir.exists():
        cam_files = list(cameras_dir.glob("*.json"))
        if cam_files:
            with open(cam_files[0]) as f:
                camera_params = json.load(f)

    return {
        "poses_3d": poses_3d,
        "camera_params": camera_params,
        "num_frames": len(poses_3d),
    }


def project_to_2d(poses_3d: np.ndarray, camera_params: dict) -> np.ndarray:
    """Project 3D poses to 2D using camera parameters."""
    fx = camera_params.get("fx", 1000)
    fy = camera_params.get("fy", 1000)
    cx = camera_params.get("cx", 512)
    cy = camera_params.get("cy", 512)

    X = poses_3d[..., 0]
    Y = poses_3d[..., 1]
    Z = np.clip(poses_3d[..., 2], 1e-8, None)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    return np.stack([u, v], axis=-1)


def process_split(data_root: Path, output_dir: Path, split: str) -> list:
    """Process all sequences in a split."""
    split_dir = data_root / split
    output_split = output_dir / split

    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return []

    # Find sequences
    sequences = []
    for item in split_dir.iterdir():
        if item.is_dir():
            sequences.append(item)

    if not sequences:
        # Maybe sequences are directly in split_dir
        if (split_dir / "joints3d_25").exists():
            sequences = [split_dir]

    metadata = []

    for seq_dir in tqdm(sequences, desc=f"Processing {split}"):
        data = load_fit3d_sequence(seq_dir)
        if data is None:
            continue

        poses_3d = data["poses_3d"]

        # Convert to H36M format if needed
        if poses_3d.shape[1] == 25:
            poses_3d_h36m = coco25_to_h36m17(poses_3d)
        elif poses_3d.shape[1] == 17:
            poses_3d_h36m = poses_3d
        else:
            print(f"Unexpected joint count {poses_3d.shape[1]} in {seq_dir.name}")
            continue

        # Project to 2D
        if data["camera_params"]:
            poses_2d = project_to_2d(poses_3d_h36m, data["camera_params"])
        else:
            # Default projection
            poses_2d = project_to_2d(poses_3d_h36m, {"fx": 1000, "fy": 1000, "cx": 512, "cy": 512})

        # Save
        seq_output = output_split / seq_dir.name
        seq_output.mkdir(parents=True, exist_ok=True)

        np.save(seq_output / "poses_3d.npy", poses_3d_h36m.astype(np.float32))
        np.save(seq_output / "poses_2d.npy", poses_2d.astype(np.float32))

        metadata.append({
            "sequence": seq_dir.name,
            "num_frames": len(poses_3d_h36m),
            "original_joints": poses_3d.shape[1],
        })

    # Save metadata
    if metadata:
        with open(output_split / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare Fit3D dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/fit3d",
        help="Path to raw Fit3D data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/fit3d_processed",
        help="Path to output processed data",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Splits to process",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if not data_root.exists():
        print(f"""
Fit3D data not found at {data_root}

Setup instructions:
1. Register at https://fit3d.imar.ro/ and request access
2. Download the dataset after approval
3. Extract to {data_root}

Expected structure:
{data_root}/
    train/
        subject_001/
            joints3d_25/
                000000.json
                000001.json
                ...
            cameras/
                cam_00.json
        subject_002/
        ...
    test/
        ...

Alternatively, clone IMAR tools for additional utilities:
    git clone https://github.com/sminchisescu-research/imar_vision_datasets_tools.git external/imar_tools
""")
        return

    print(f"Processing Fit3D from {data_root}")
    print(f"Output directory: {output_dir}")

    total_sequences = 0
    for split in args.splits:
        metadata = process_split(data_root, output_dir, split)
        print(f"{split}: {len(metadata)} sequences processed")
        total_sequences += len(metadata)

    print(f"\nTotal: {total_sequences} sequences")
    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
