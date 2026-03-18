#!/usr/bin/env python3
"""
Process VideoPose3D preprocessed data to our format.

This script handles the common preprocessed NPZ format from VideoPose3D:
- Human3.6M: data_3d_h36m.npz, data_2d_h36m_gt.npz
- MPI-INF-3DHP: data_train_3dhp.npz, data_test_3dhp.npz

Usage:
    python scripts/process_videopose_data.py --data_root ./data --output_dir ./data/processed
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def process_h36m(data_dir: Path, output_dir: Path) -> dict:
    """
    Process Human3.6M data from VideoPose3D format.

    Expected files:
    - data_3d_h36m.npz: 3D positions (32 joints)
    - data_2d_h36m_gt.npz: 2D ground truth (17 joints, 4 cameras)
    """
    h36m_dir = data_dir / "Human3.6M"

    file_3d = h36m_dir / "data_3d_h36m.npz"
    file_2d = h36m_dir / "data_2d_h36m_gt.npz"

    if not file_3d.exists():
        print(f"Human3.6M 3D data not found at {file_3d}")
        return {}

    print("Loading Human3.6M data...")
    data_3d = np.load(file_3d, allow_pickle=True)['positions_3d'].item()

    # 2D data is optional (can project from 3D)
    data_2d = None
    if file_2d.exists():
        data_2d = np.load(file_2d, allow_pickle=True)['positions_2d'].item()

    # H36M 32 to 17 joint mapping
    # Standard 17: pelvis, rhip, rknee, rankle, lhip, lknee, lankle, spine, thorax, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist
    H36M_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    # Train/test split
    train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    test_subjects = ['S9', 'S11']

    metadata = {'train': [], 'test': []}

    for subject, actions in tqdm(data_3d.items(), desc="Processing H36M"):
        split = 'train' if subject in train_subjects else 'test'
        if subject not in train_subjects and subject not in test_subjects:
            continue

        for action, poses_3d in actions.items():
            # Convert 32 joints to 17
            if poses_3d.shape[1] == 32:
                poses_3d_17 = poses_3d[:, H36M_32_TO_17, :]
            else:
                poses_3d_17 = poses_3d

            # Get 2D data (use first camera)
            if data_2d and subject in data_2d and action in data_2d[subject]:
                poses_2d_list = data_2d[subject][action]
                if isinstance(poses_2d_list, list) and len(poses_2d_list) > 0:
                    poses_2d = poses_2d_list[0]  # First camera
                else:
                    poses_2d = poses_2d_list
            else:
                # Project to 2D (simple orthographic for now)
                poses_2d = poses_3d_17[..., :2].copy()

            # Ensure same length
            min_len = min(len(poses_3d_17), len(poses_2d))
            poses_3d_17 = poses_3d_17[:min_len]
            poses_2d = poses_2d[:min_len]

            # Center 3D at pelvis
            pelvis = poses_3d_17[:, 0:1, :]
            poses_3d_centered = poses_3d_17 - pelvis

            # Note: VideoPose3D H36M data is already in meters, no scaling needed

            # Normalize 2D to [-1, 1]
            poses_2d_norm = poses_2d.copy()
            poses_2d_norm[..., 0] = poses_2d_norm[..., 0] / 500.0 - 1.0  # Assuming ~1000px width
            poses_2d_norm[..., 1] = poses_2d_norm[..., 1] / 500.0 - 1.0

            # Clean action name for filename
            action_clean = action.replace(' ', '_').replace('/', '_')
            seq_name = f"{subject}_{action_clean}"

            # Save
            seq_output = output_dir / "h36m" / split / seq_name
            seq_output.mkdir(parents=True, exist_ok=True)

            np.save(seq_output / "poses_3d.npy", poses_3d_centered.astype(np.float32))
            np.save(seq_output / "poses_2d.npy", poses_2d_norm.astype(np.float32))

            metadata[split].append({
                'sequence': seq_name,
                'subject': subject,
                'action': action,
                'num_frames': len(poses_3d_centered),
            })

    # Save metadata
    for split in ['train', 'test']:
        if metadata[split]:
            split_dir = output_dir / "h36m" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            with open(split_dir / "metadata.json", 'w') as f:
                json.dump(metadata[split], f, indent=2)

    return metadata


def process_mpi_inf_3dhp(data_dir: Path, output_dir: Path) -> dict:
    """
    Process MPI-INF-3DHP data from VideoPose3D format.

    Expected files:
    - data_train_3dhp.npz: Training data (8 subjects × 2 sequences × 8 cameras)
    - data_test_3dhp.npz: Test data (6 test sequences)
    """
    mpi_dir = data_dir / "MPI-INF-3DHP"

    file_train = mpi_dir / "data_train_3dhp.npz"
    file_test = mpi_dir / "data_test_3dhp.npz"

    metadata = {'train': [], 'test': []}

    # Process training data
    if file_train.exists():
        print("Loading MPI-INF-3DHP training data...")
        data_train = np.load(file_train, allow_pickle=True)['data'].item()

        for seq_name, seq_data in tqdm(data_train.items(), desc="Processing 3DHP Train"):
            cameras_dict = seq_data[0]
            fps = seq_data[1] if len(seq_data) > 1 else 25

            # Use first available camera
            cam_key = list(cameras_dict.keys())[0]
            cam_data = cameras_dict[cam_key]

            poses_3d = cam_data['data_3d']
            poses_2d = cam_data['data_2d']

            # Center 3D at pelvis
            pelvis = poses_3d[:, 0:1, :]
            poses_3d_centered = poses_3d - pelvis

            # Scale to meters (MPI is in mm)
            poses_3d_centered = poses_3d_centered / 1000.0

            # Normalize 2D to [-1, 1] (assuming 2048x2048)
            poses_2d_norm = poses_2d.copy()
            poses_2d_norm[..., 0] = poses_2d_norm[..., 0] / 1024.0 - 1.0
            poses_2d_norm[..., 1] = poses_2d_norm[..., 1] / 1024.0 - 1.0

            # Clean sequence name
            seq_clean = seq_name.replace(' ', '_')

            # Save
            seq_output = output_dir / "mpi_3dhp" / "train" / seq_clean
            seq_output.mkdir(parents=True, exist_ok=True)

            np.save(seq_output / "poses_3d.npy", poses_3d_centered.astype(np.float32))
            np.save(seq_output / "poses_2d.npy", poses_2d_norm.astype(np.float32))

            metadata['train'].append({
                'sequence': seq_clean,
                'num_frames': len(poses_3d_centered),
                'camera': cam_key,
                'fps': fps,
            })

    # Process test data
    if file_test.exists():
        print("Loading MPI-INF-3DHP test data...")
        data_test = np.load(file_test, allow_pickle=True)['data'].item()

        for seq_name, seq_data in tqdm(data_test.items(), desc="Processing 3DHP Test"):
            poses_3d = seq_data['data_3d']
            poses_2d = seq_data['data_2d']
            valid = seq_data.get('valid', np.ones(len(poses_3d), dtype=bool))

            # Center 3D at pelvis
            pelvis = poses_3d[:, 0:1, :]
            poses_3d_centered = poses_3d - pelvis

            # Scale to meters
            poses_3d_centered = poses_3d_centered / 1000.0

            # Normalize 2D to [-1, 1]
            poses_2d_norm = poses_2d.copy()
            poses_2d_norm[..., 0] = poses_2d_norm[..., 0] / 1024.0 - 1.0
            poses_2d_norm[..., 1] = poses_2d_norm[..., 1] / 1024.0 - 1.0

            # Save
            seq_output = output_dir / "mpi_3dhp" / "test" / seq_name
            seq_output.mkdir(parents=True, exist_ok=True)

            np.save(seq_output / "poses_3d.npy", poses_3d_centered.astype(np.float32))
            np.save(seq_output / "poses_2d.npy", poses_2d_norm.astype(np.float32))
            np.save(seq_output / "valid.npy", valid.astype(bool))

            metadata['test'].append({
                'sequence': seq_name,
                'num_frames': len(poses_3d_centered),
                'valid_frames': int(valid.sum()),
            })

    # Save metadata
    for split in ['train', 'test']:
        if metadata[split]:
            split_dir = output_dir / "mpi_3dhp" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            with open(split_dir / "metadata.json", 'w') as f:
                json.dump(metadata[split], f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Process VideoPose3D format data")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing Human3.6M and MPI-INF-3DHP folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["h36m", "mpi"],
        choices=["h36m", "mpi", "all"],
        help="Datasets to process",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["h36m", "mpi"]

    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Processing datasets: {datasets}")
    print()

    # Process Human3.6M
    if "h36m" in datasets:
        h36m_meta = process_h36m(data_root, output_dir)
        if h36m_meta:
            train_count = len(h36m_meta.get('train', []))
            test_count = len(h36m_meta.get('test', []))
            train_frames = sum(m['num_frames'] for m in h36m_meta.get('train', []))
            test_frames = sum(m['num_frames'] for m in h36m_meta.get('test', []))
            print(f"\nHuman3.6M: {train_count} train sequences ({train_frames:,} frames), {test_count} test sequences ({test_frames:,} frames)")

    # Process MPI-INF-3DHP
    if "mpi" in datasets:
        mpi_meta = process_mpi_inf_3dhp(data_root, output_dir)
        if mpi_meta:
            train_count = len(mpi_meta.get('train', []))
            test_count = len(mpi_meta.get('test', []))
            train_frames = sum(m['num_frames'] for m in mpi_meta.get('train', []))
            test_frames = sum(m['num_frames'] for m in mpi_meta.get('test', []))
            print(f"\nMPI-INF-3DHP: {train_count} train sequences ({train_frames:,} frames), {test_count} test sequences ({test_frames:,} frames)")

    print(f"\nProcessed data saved to {output_dir}")
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    h36m/")
    print(f"      train/")
    print(f"        S1_Directions_1/")
    print(f"          poses_3d.npy  # (T, 17, 3) root-centered, meters")
    print(f"          poses_2d.npy  # (T, 17, 2) normalized [-1, 1]")
    print(f"      test/")
    print(f"    mpi_3dhp/")
    print(f"      train/")
    print(f"      test/")


if __name__ == "__main__":
    main()
