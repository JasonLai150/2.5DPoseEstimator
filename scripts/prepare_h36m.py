#!/usr/bin/env python3
"""
Human3.6M Dataset Preparation Script.

Human3.6M is the standard benchmark for 3D human pose estimation.
Website: http://vision.imar.ro/human3.6m/

Features:
- 11 subjects (S1, S5, S6, S7, S8, S9, S11)
- 15 actions per subject
- 4 camera views
- Marker-based motion capture with 3D ground truth

The dataset uses 32 joints, which we map to standard 17-joint format.

Usage:
    python scripts/prepare_h36m.py --data_root ./data/h36m --output_dir ./data/h36m_processed
"""

import argparse
import cdflib
import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Human3.6M original uses 32 joints, we convert to standard 17
# 32-joint order from H36M:
# 0: Hip (root), 1: RHip, 2: RKnee, 3: RFoot, 4: LHip, 5: LKnee, 6: LFoot,
# 7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head, 11: LShoulder, 12: LElbow,
# 13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist,
# 17-31: Additional joints (fingers, toes, etc.)

# Standard 17-joint H36M format (same as our target):
# 0: pelvis, 1: right_hip, 2: right_knee, 3: right_ankle
# 4: left_hip, 5: left_knee, 6: left_ankle, 7: spine
# 8: thorax, 9: neck/nose, 10: head, 11: left_shoulder
# 12: left_elbow, 13: left_wrist, 14: right_shoulder, 15: right_elbow, 16: right_wrist

H36M_32_TO_17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Camera intrinsics for H36M (default values per camera)
H36M_CAMERAS = {
    "54138969": {"fx": 1145.04, "fy": 1143.78, "cx": 512.54, "cy": 515.45},
    "55011271": {"fx": 1149.67, "fy": 1147.59, "cx": 508.84, "cy": 508.06},
    "58860488": {"fx": 1149.14, "fy": 1148.78, "cx": 519.81, "cy": 501.18},
    "60457274": {"fx": 1145.51, "fy": 1144.77, "cx": 514.96, "cy": 501.88},
}

# Actions in H36M
H36M_ACTIONS = [
    "Directions", "Discussion", "Eating", "Greeting", "Phoning",
    "Photo", "Posing", "Purchases", "Sitting", "SittingDown",
    "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"
]


def h36m_32_to_17(poses_32: np.ndarray) -> np.ndarray:
    """
    Convert Human3.6M 32-joint poses to 17-joint format.

    Args:
        poses_32: Shape (..., 32, 3)

    Returns:
        poses_17: Shape (..., 17, 3)
    """
    return poses_32[..., H36M_32_TO_17, :]


def load_h36m_cdf(cdf_path: Path) -> np.ndarray | None:
    """Load poses from CDF file (original H36M format)."""
    try:
        cdf = cdflib.CDF(str(cdf_path))
        poses = cdf.varget("Pose")
        # Shape: (frames, 96) -> (frames, 32, 3)
        poses = poses.reshape(-1, 32, 3)
        return poses
    except Exception as e:
        print(f"Error loading CDF {cdf_path}: {e}")
        return None


def load_h36m_mat(mat_path: Path) -> np.ndarray | None:
    """Load poses from MAT file."""
    try:
        mat = sio.loadmat(str(mat_path))
        # Try different keys
        for key in ['pose3d', 'poses3d', 'joints3d', 'data']:
            if key in mat:
                poses = mat[key]
                if poses.ndim == 2:
                    num_joints = 32 if poses.shape[1] % 32 == 0 else 17
                    poses = poses.reshape(-1, num_joints, 3)
                return poses
        return None
    except Exception:
        return None


def load_h36m_h5(h5_path: Path) -> dict | None:
    """Load from HDF5 file (common preprocessed format)."""
    try:
        with h5py.File(h5_path, 'r') as f:
            result = {}
            if 'poses_3d' in f:
                result['poses_3d'] = np.array(f['poses_3d'])
            elif 'pose3d' in f:
                result['poses_3d'] = np.array(f['pose3d'])
            if 'poses_2d' in f:
                result['poses_2d'] = np.array(f['poses_2d'])
            elif 'pose2d' in f:
                result['poses_2d'] = np.array(f['pose2d'])
            return result if result else None
    except Exception:
        return None


def load_h36m_npz(npz_path: Path) -> dict | None:
    """Load from NPZ file (common preprocessed format)."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        result = {}
        if 'positions_3d' in data:
            result['poses_3d'] = data['positions_3d']
        elif 'poses_3d' in data:
            result['poses_3d'] = data['poses_3d']
        if 'positions_2d' in data:
            result['poses_2d'] = data['positions_2d']
        elif 'poses_2d' in data:
            result['poses_2d'] = data['poses_2d']
        return result if result else None
    except Exception:
        return None


def project_to_2d(poses_3d: np.ndarray, camera_params: dict) -> np.ndarray:
    """Project 3D poses to 2D using camera intrinsics."""
    fx = camera_params.get("fx", 1145.0)
    fy = camera_params.get("fy", 1145.0)
    cx = camera_params.get("cx", 512.0)
    cy = camera_params.get("cy", 512.0)

    X = poses_3d[..., 0]
    Y = poses_3d[..., 1]
    Z = np.clip(poses_3d[..., 2], 1e-8, None)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    return np.stack([u, v], axis=-1)


def normalize_2d(poses_2d: np.ndarray, width: int = 1000, height: int = 1002) -> np.ndarray:
    """Normalize 2D poses to [-1, 1] range."""
    poses_norm = poses_2d.copy()
    poses_norm[..., 0] = (poses_norm[..., 0] - width / 2) / (width / 2)
    poses_norm[..., 1] = (poses_norm[..., 1] - height / 2) / (height / 2)
    return poses_norm


def center_3d(poses_3d: np.ndarray, pelvis_idx: int = 0) -> np.ndarray:
    """Center 3D poses at pelvis (root-relative coordinates)."""
    pelvis = poses_3d[..., pelvis_idx:pelvis_idx+1, :]
    return poses_3d - pelvis


def scale_to_mm(poses_3d: np.ndarray) -> np.ndarray:
    """H36M is in mm, convert to meters for consistency."""
    return poses_3d / 1000.0


def process_subject_dir(subject_dir: Path, output_dir: Path, camera_id: str = None) -> list:
    """Process all sequences for a subject directory."""
    metadata = []

    # Find all pose files
    pose_files = []
    for ext in ['*.cdf', '*.mat', '*.h5', '*.npz']:
        pose_files.extend(subject_dir.glob(ext))

    # Also check subdirectories (camera folders)
    for subdir in subject_dir.iterdir():
        if subdir.is_dir():
            for ext in ['*.cdf', '*.mat', '*.h5', '*.npz']:
                pose_files.extend(subdir.glob(ext))

    for pf in tqdm(pose_files, desc=subject_dir.name):
        # Determine camera from path or filename
        cam_id = None
        for cam in H36M_CAMERAS.keys():
            if cam in str(pf):
                cam_id = cam
                break

        if camera_id and cam_id and cam_id != camera_id:
            continue

        camera_params = H36M_CAMERAS.get(cam_id, list(H36M_CAMERAS.values())[0])

        # Load data
        if pf.suffix == '.cdf':
            poses_3d = load_h36m_cdf(pf)
            poses_2d = None
        elif pf.suffix == '.mat':
            poses_3d = load_h36m_mat(pf)
            poses_2d = None
        elif pf.suffix == '.h5':
            data = load_h36m_h5(pf)
            poses_3d = data.get('poses_3d') if data else None
            poses_2d = data.get('poses_2d') if data else None
        elif pf.suffix == '.npz':
            data = load_h36m_npz(pf)
            poses_3d = data.get('poses_3d') if data else None
            poses_2d = data.get('poses_2d') if data else None
        else:
            continue

        if poses_3d is None:
            continue

        # Convert to 17 joints if needed
        if poses_3d.shape[-2] == 32:
            poses_3d = h36m_32_to_17(poses_3d)
        elif poses_3d.shape[-2] != 17:
            print(f"Unexpected joint count {poses_3d.shape[-2]} in {pf}")
            continue

        # Skip if too short
        if len(poses_3d) < 10:
            continue

        # Scale to meters
        poses_3d = scale_to_mm(poses_3d)

        # Center at pelvis
        poses_3d_centered = center_3d(poses_3d)

        # Get 2D poses
        if poses_2d is not None:
            if poses_2d.shape[-2] == 32:
                poses_2d = poses_2d[..., H36M_32_TO_17, :]
            poses_2d_norm = normalize_2d(poses_2d)
        else:
            poses_2d_proj = project_to_2d(poses_3d, camera_params)
            poses_2d_norm = normalize_2d(poses_2d_proj)

        # Extract action name
        action = pf.stem
        for a in H36M_ACTIONS:
            if a.lower() in pf.stem.lower():
                action = a
                break

        # Save
        seq_name = f"{subject_dir.name}_{action}"
        if cam_id:
            seq_name += f"_{cam_id}"
        seq_name += f"_{pf.stem}"

        seq_output = output_dir / seq_name
        seq_output.mkdir(parents=True, exist_ok=True)

        np.save(seq_output / "poses_3d.npy", poses_3d_centered.astype(np.float32))
        np.save(seq_output / "poses_2d.npy", poses_2d_norm.astype(np.float32))

        metadata.append({
            "sequence": seq_name,
            "subject": subject_dir.name,
            "action": action,
            "camera": cam_id,
            "num_frames": len(poses_3d),
        })

    return metadata


def process_preprocessed_npz(npz_path: Path, output_dir: Path) -> list:
    """
    Process commonly available preprocessed H36M NPZ files.

    Common format from VideoPose3D and other repos:
    - data_2d_h36m_{detector}.npz: 2D detections
    - data_3d_h36m.npz: 3D ground truth
    """
    metadata = []

    data = np.load(npz_path, allow_pickle=True)

    # VideoPose3D format
    if 'positions_3d' in data:
        positions = data['positions_3d'].item()  # dict of subjects

        for subject, actions in positions.items():
            for action, cameras in actions.items():
                for cam_idx, poses_3d in enumerate(cameras):
                    if poses_3d is None or len(poses_3d) < 10:
                        continue

                    # Scale and center
                    poses_3d = scale_to_mm(poses_3d)
                    poses_3d_centered = center_3d(poses_3d)

                    # Project to 2D
                    cam_id = list(H36M_CAMERAS.keys())[cam_idx % len(H36M_CAMERAS)]
                    camera_params = H36M_CAMERAS[cam_id]
                    poses_2d_proj = project_to_2d(poses_3d, camera_params)
                    poses_2d_norm = normalize_2d(poses_2d_proj)

                    # Determine split
                    split = "train" if subject in ["S1", "S5", "S6", "S7", "S8"] else "test"

                    seq_name = f"{subject}_{action}_{cam_id}"
                    seq_output = output_dir / split / seq_name
                    seq_output.mkdir(parents=True, exist_ok=True)

                    np.save(seq_output / "poses_3d.npy", poses_3d_centered.astype(np.float32))
                    np.save(seq_output / "poses_2d.npy", poses_2d_norm.astype(np.float32))

                    metadata.append({
                        "sequence": seq_name,
                        "subject": subject,
                        "action": action,
                        "camera": cam_id,
                        "num_frames": len(poses_3d),
                        "split": split,
                    })

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare Human3.6M dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/h36m",
        help="Path to raw Human3.6M data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/h36m_processed",
        help="Path to output processed data",
    )
    parser.add_argument(
        "--camera_id",
        type=str,
        default=None,
        choices=list(H36M_CAMERAS.keys()) + [None],
        help="Camera ID to use (None for all cameras)",
    )
    parser.add_argument(
        "--train_subjects",
        type=str,
        nargs="+",
        default=["S1", "S5", "S6", "S7", "S8"],
        help="Subject IDs for training",
    )
    parser.add_argument(
        "--test_subjects",
        type=str,
        nargs="+",
        default=["S9", "S11"],
        help="Subject IDs for testing",
    )
    parser.add_argument(
        "--preprocessed_npz",
        type=str,
        default=None,
        help="Path to preprocessed NPZ file (e.g., from VideoPose3D)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # Check for preprocessed format first
    if args.preprocessed_npz:
        npz_path = Path(args.preprocessed_npz)
        if npz_path.exists():
            print(f"Processing preprocessed file: {npz_path}")
            metadata = process_preprocessed_npz(npz_path, output_dir)
            print(f"Processed {len(metadata)} sequences")
            return

    # Check for common preprocessed files in data_root
    common_preprocessed = [
        data_root / "data_3d_h36m.npz",
        data_root / "h36m_3d.npz",
        data_root / "positions_3d.npz",
    ]
    for npz_path in common_preprocessed:
        if npz_path.exists():
            print(f"Found preprocessed file: {npz_path}")
            metadata = process_preprocessed_npz(npz_path, output_dir)
            print(f"Processed {len(metadata)} sequences")
            return

    if not data_root.exists():
        print(f"""
Human3.6M data not found at {data_root}

Download instructions:
1. Visit: http://vision.imar.ro/human3.6m/
2. Register and request access
3. Download the following:
   - "3D poses: D3 Positions" (for 3D ground truth)
   - Optionally: Videos or frames for visualization

Alternative: Use preprocessed data
Many repos provide preprocessed H36M in NPZ format:
- VideoPose3D: https://github.com/facebookresearch/VideoPose3D
- Download data_3d_h36m.npz and run:
  python scripts/prepare_h36m.py --preprocessed_npz path/to/data_3d_h36m.npz

Expected structure (raw):
{data_root}/
    S1/
        Directions.cdf (or .mat)
        Discussion.cdf
        ...
    S5/
    S6/
    S7/
    S8/
    S9/
    S11/

Expected structure (preprocessed):
{data_root}/
    data_3d_h36m.npz
""")
        return

    print(f"Processing Human3.6M from {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Camera ID: {args.camera_id or 'all'}")
    print(f"Train subjects: {args.train_subjects}")
    print(f"Test subjects: {args.test_subjects}")

    # Process each split
    for split, subjects in [("train", args.train_subjects), ("test", args.test_subjects)]:
        split_output = output_dir / split
        split_output.mkdir(parents=True, exist_ok=True)

        all_metadata = []

        for subject in subjects:
            subject_dir = data_root / subject
            if not subject_dir.exists():
                print(f"Subject directory not found: {subject_dir}")
                continue

            print(f"\nProcessing {subject}...")
            metadata = process_subject_dir(subject_dir, split_output, args.camera_id)
            all_metadata.extend(metadata)

        # Save metadata
        if all_metadata:
            with open(split_output / "metadata.json", "w") as f:
                json.dump(all_metadata, f, indent=2)

        print(f"{split}: {len(all_metadata)} sequences")

    print(f"\nOutput saved to {output_dir}")


if __name__ == "__main__":
    main()
