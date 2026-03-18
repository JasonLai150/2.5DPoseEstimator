#!/usr/bin/env python3
"""
MPI-INF-3DHP Dataset Preparation Script.

MPI-INF-3DHP is a 3D human pose dataset captured in indoor/outdoor settings.
Website: http://gvv.mpi-inf.mpg.de/3dhp-dataset/

Features:
- 8 subjects performing various activities
- Multiple camera views (14 cameras, some moving)
- Marker-less motion capture with 3D ground truth
- Both indoor and outdoor sequences

The dataset uses 28 joints, which we map to H36M 17-joint format.

Usage:
    python scripts/prepare_mpi_inf_3dhp.py --data_root ./data/mpi_inf_3dhp --output_dir ./data/mpi_3dhp_processed
"""

import argparse
import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# MPI-INF-3DHP uses 28 joints (17 body + 11 additional)
# We map to H36M 17-joint format
# MPI-INF-3DHP joint order (28 joints):
# 0: spine3, 1: spine4, 2: spine2, 3: spine, 4: pelvis,
# 5: neck, 6: head, 7: head_top, 8: left_clavicle, 9: left_shoulder,
# 10: left_elbow, 11: left_wrist, 12: left_hand, 13: right_clavicle, 14: right_shoulder,
# 15: right_elbow, 16: right_wrist, 17: right_hand, 18: left_hip, 19: left_knee,
# 20: left_ankle, 21: left_foot, 22: left_toe, 23: right_hip, 24: right_knee,
# 25: right_ankle, 26: right_foot, 27: right_toe

# H36M 17 joints:
# 0: pelvis, 1: right_hip, 2: right_knee, 3: right_ankle
# 4: left_hip, 5: left_knee, 6: left_ankle, 7: spine
# 8: thorax, 9: neck/nose, 10: head, 11: left_shoulder
# 12: left_elbow, 13: left_wrist, 14: right_shoulder, 15: right_elbow, 16: right_wrist

MPI_TO_H36M = {
    0: 4,   # pelvis
    1: 23,  # right_hip
    2: 24,  # right_knee
    3: 25,  # right_ankle
    4: 18,  # left_hip
    5: 19,  # left_knee
    6: 20,  # left_ankle
    7: 3,   # spine
    8: 1,   # spine4 -> thorax
    9: 5,   # neck
    10: 7,  # head_top -> head
    11: 9,  # left_shoulder
    12: 10, # left_elbow
    13: 11, # left_wrist
    14: 14, # right_shoulder
    15: 15, # right_elbow
    16: 16, # right_wrist
}


def mpi_to_h36m17(poses_mpi: np.ndarray) -> np.ndarray:
    """
    Convert MPI-INF-3DHP 28-joint poses to H36M 17-joint format.

    Args:
        poses_mpi: Shape (..., 28, 3)

    Returns:
        poses_h36m: Shape (..., 17, 3)
    """
    shape = poses_mpi.shape[:-2]
    poses_h36m = np.zeros(shape + (17, 3), dtype=poses_mpi.dtype)

    for h36m_idx, mpi_idx in MPI_TO_H36M.items():
        poses_h36m[..., h36m_idx, :] = poses_mpi[..., mpi_idx, :]

    return poses_h36m


def load_mpi_sequence(sequence_path: Path, camera_id: int = 0) -> dict | None:
    """
    Load a MPI-INF-3DHP sequence.

    The dataset structure is:
    S{subject_id}/Seq{seq_id}/
        annot.mat (2D/3D annotations)
        camera.calibration (camera parameters)
        imageSequence/ (video frames)
    """
    annot_file = sequence_path / "annot.mat"

    if not annot_file.exists():
        return None

    try:
        # Try loading as HDF5 (MATLAB v7.3)
        try:
            with h5py.File(annot_file, 'r') as f:
                # Get 3D poses (univ_annot3 is world coordinates)
                if 'univ_annot3' in f:
                    poses_3d = np.array(f['univ_annot3'])
                elif 'annot3' in f:
                    poses_3d = np.array(f['annot3'])
                else:
                    return None

                # Get 2D poses
                if 'annot2' in f:
                    poses_2d = np.array(f['annot2'])
                else:
                    poses_2d = None

                # Camera selection
                if poses_3d.ndim == 4:  # (cameras, frames, joints, 3)
                    poses_3d = poses_3d[camera_id]
                    if poses_2d is not None:
                        poses_2d = poses_2d[camera_id]

        except OSError:
            # Fall back to scipy for older MATLAB format
            mat = sio.loadmat(str(annot_file))

            if 'univ_annot3' in mat:
                poses_3d = mat['univ_annot3']
            elif 'annot3' in mat:
                poses_3d = mat['annot3']
            else:
                return None

            poses_2d = mat.get('annot2', None)

            # Handle nested structure
            if isinstance(poses_3d, np.ndarray) and poses_3d.dtype == object:
                poses_3d = poses_3d[camera_id, 0]
            if poses_2d is not None and isinstance(poses_2d, np.ndarray) and poses_2d.dtype == object:
                poses_2d = poses_2d[camera_id, 0]

    except Exception as e:
        print(f"Error loading {annot_file}: {e}")
        return None

    # Ensure correct shape
    if poses_3d.ndim == 2:  # (frames*joints, 3) -> (frames, joints, 3)
        num_joints = 28
        num_frames = poses_3d.shape[0] // num_joints
        poses_3d = poses_3d.reshape(num_frames, num_joints, 3)

    if poses_2d is not None and poses_2d.ndim == 2:
        num_joints = 28
        num_frames = poses_2d.shape[0] // num_joints
        poses_2d = poses_2d.reshape(num_frames, num_joints, 2)

    # Load camera parameters if available
    camera_params = None
    calib_file = sequence_path / "camera.calibration"
    if calib_file.exists():
        try:
            camera_params = parse_camera_calibration(calib_file, camera_id)
        except Exception:
            pass

    return {
        "poses_3d": poses_3d,
        "poses_2d": poses_2d,
        "camera_params": camera_params,
        "num_frames": len(poses_3d),
    }


def parse_camera_calibration(calib_file: Path, camera_id: int) -> dict:
    """Parse MPI-INF-3DHP camera calibration file."""
    with open(calib_file) as f:
        lines = f.readlines()

    # Simple parsing - format varies by version
    # Usually contains intrinsics (fx, fy, cx, cy) and extrinsics
    params = {
        "fx": 1497.693,  # Default values for 3DHP
        "fy": 1497.693,
        "cx": 1024,
        "cy": 1024,
    }

    # Try to parse actual values
    for line in lines:
        if f"camera {camera_id}" in line.lower() or f"cam{camera_id}" in line.lower():
            # Found camera section
            pass
        if "fx" in line.lower():
            try:
                params["fx"] = float(line.split()[-1])
            except Exception:
                pass

    return params


def project_to_2d(poses_3d: np.ndarray, camera_params: dict) -> np.ndarray:
    """Project 3D poses to 2D using camera intrinsics."""
    fx = camera_params.get("fx", 1497.693)
    fy = camera_params.get("fy", 1497.693)
    cx = camera_params.get("cx", 1024)
    cy = camera_params.get("cy", 1024)

    X = poses_3d[..., 0]
    Y = poses_3d[..., 1]
    Z = np.clip(poses_3d[..., 2], 1e-8, None)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    return np.stack([u, v], axis=-1)


def normalize_2d(poses_2d: np.ndarray, width: int = 2048, height: int = 2048) -> np.ndarray:
    """Normalize 2D poses to [-1, 1] range."""
    poses_norm = poses_2d.copy()
    poses_norm[..., 0] = (poses_norm[..., 0] - width / 2) / (width / 2)
    poses_norm[..., 1] = (poses_norm[..., 1] - height / 2) / (height / 2)
    return poses_norm


def center_3d(poses_3d: np.ndarray, pelvis_idx: int = 0) -> np.ndarray:
    """Center 3D poses at pelvis (root-relative coordinates)."""
    pelvis = poses_3d[..., pelvis_idx:pelvis_idx+1, :]
    return poses_3d - pelvis


def process_split(data_root: Path, output_dir: Path, split: str, subjects: list, camera_id: int = 0) -> list:
    """Process sequences for a split."""
    output_split = output_dir / split
    output_split.mkdir(parents=True, exist_ok=True)

    metadata = []

    for subj_id in subjects:
        subj_dir = data_root / f"S{subj_id}"
        if not subj_dir.exists():
            print(f"Subject directory not found: {subj_dir}")
            continue

        # Find all sequences for this subject
        seq_dirs = sorted([d for d in subj_dir.iterdir() if d.is_dir() and d.name.startswith("Seq")])

        for seq_dir in tqdm(seq_dirs, desc=f"S{subj_id}"):
            data = load_mpi_sequence(seq_dir, camera_id)
            if data is None:
                continue

            poses_3d = data["poses_3d"]
            poses_2d = data["poses_2d"]

            # Skip if too short
            if len(poses_3d) < 10:
                continue

            # Convert to H36M format
            if poses_3d.shape[1] == 28:
                poses_3d_h36m = mpi_to_h36m17(poses_3d)
            elif poses_3d.shape[1] == 17:
                poses_3d_h36m = poses_3d
            else:
                print(f"Unexpected joint count {poses_3d.shape[1]} in {seq_dir.name}")
                continue

            # Get 2D poses
            if poses_2d is not None and poses_2d.shape[1] == 28:
                poses_2d_h36m = mpi_to_h36m17(poses_2d)
            elif poses_2d is not None and poses_2d.shape[1] == 17:
                poses_2d_h36m = poses_2d
            elif data["camera_params"]:
                poses_2d_h36m = project_to_2d(poses_3d_h36m, data["camera_params"])
            else:
                poses_2d_h36m = project_to_2d(poses_3d_h36m, {"fx": 1497.693, "fy": 1497.693, "cx": 1024, "cy": 1024})

            # Center 3D poses
            poses_3d_centered = center_3d(poses_3d_h36m)

            # Normalize 2D poses
            poses_2d_norm = normalize_2d(poses_2d_h36m)

            # Save
            seq_name = f"S{subj_id}_{seq_dir.name}"
            seq_output = output_split / seq_name
            seq_output.mkdir(parents=True, exist_ok=True)

            np.save(seq_output / "poses_3d.npy", poses_3d_centered.astype(np.float32))
            np.save(seq_output / "poses_2d.npy", poses_2d_norm.astype(np.float32))

            metadata.append({
                "sequence": seq_name,
                "subject": subj_id,
                "num_frames": len(poses_3d_h36m),
                "camera_id": camera_id,
            })

    # Save metadata
    if metadata:
        with open(output_split / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare MPI-INF-3DHP dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/mpi_inf_3dhp",
        help="Path to raw MPI-INF-3DHP data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/mpi_3dhp_processed",
        help="Path to output processed data",
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera ID to use (0-13)",
    )
    parser.add_argument(
        "--train_subjects",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Subject IDs for training",
    )
    parser.add_argument(
        "--test_subjects",
        type=int,
        nargs="+",
        default=[7, 8],
        help="Subject IDs for testing",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if not data_root.exists():
        print(f"""
MPI-INF-3DHP data not found at {data_root}

Download instructions:
1. Visit: http://gvv.mpi-inf.mpg.de/3dhp-dataset/
2. Register and download the dataset
3. Extract to {data_root}

Expected structure:
{data_root}/
    S1/
        Seq1/
            annot.mat
            camera.calibration
            imageSequence/
        Seq2/
        ...
    S2/
    ...
    S8/

Note: The dataset is ~25GB total
""")
        return

    print(f"Processing MPI-INF-3DHP from {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Camera ID: {args.camera_id}")
    print(f"Train subjects: {args.train_subjects}")
    print(f"Test subjects: {args.test_subjects}")

    # Process train split
    print("\nProcessing training split...")
    train_meta = process_split(data_root, output_dir, "train", args.train_subjects, args.camera_id)
    print(f"Train: {len(train_meta)} sequences")

    # Process test split
    print("\nProcessing test split...")
    test_meta = process_split(data_root, output_dir, "test", args.test_subjects, args.camera_id)
    print(f"Test: {len(test_meta)} sequences")

    print(f"\nTotal: {len(train_meta) + len(test_meta)} sequences")
    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
