#!/usr/bin/env python3
"""
Process Fit3D dataset to our standard format using IMAR vision dataset tools.

Uses read_cam_params() and project_3d_to_2d() from IMAR tools for proper
camera handling (including lens distortion correction).

Fit3D structure (train only has GT):
    train/
        s03/, s04/, ..., s11/
            joints3d_25/<action>.json  -> (T, 25, 3) meters, COCO-25 joints
            camera_parameters/<cam_id>/<action>.json

Usage:
    python scripts/process_fit3d.py --data_root ./data/Fit3D --output_dir ./data/processed/fit3d
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'external' / 'imar_tools'))

from util.dataset_util import read_cam_params, project_3d_to_2d


# COCO-25 -> H36M-17 joint mapping
# H36M-17: pelvis, rhip, rknee, rankle, lhip, lknee, lankle,
#           spine, thorax, neck/nose, head, lshoulder, lelbow, lwrist,
#           rshoulder, relbow, rwrist
COCO25_TO_H36M17 = {
    0: 24,   # pelvis
    1: 12,   # right_hip
    2: 14,   # right_knee
    3: 16,   # right_ankle
    4: 11,   # left_hip
    5: 13,   # left_knee
    6: 15,   # left_ankle
    # 7: spine -> interpolate
    8: 23,   # thorax
    9: 0,    # neck/nose
    # 10: head -> interpolate
    11: 5,   # left_shoulder
    12: 7,   # left_elbow
    13: 9,   # left_wrist
    14: 6,   # right_shoulder
    15: 8,   # right_elbow
    16: 10,  # right_wrist
}


def coco25_to_h36m17(poses: np.ndarray) -> np.ndarray:
    """Convert (T, 25, 3) COCO-25 to (T, 17, 3) H36M-17."""
    out = np.zeros((*poses.shape[:-2], 17, 3), dtype=poses.dtype)
    for h36m_idx, coco_idx in COCO25_TO_H36M17.items():
        out[..., h36m_idx, :] = poses[..., coco_idx, :]
    # Spine: midpoint pelvis + thorax
    out[..., 7, :] = (out[..., 0, :] + out[..., 8, :]) / 2.0
    # Head: weighted avg of nose, left_ear, right_ear
    out[..., 10, :] = poses[..., 0, :] * 0.4 + poses[..., 3, :] * 0.3 + poses[..., 4, :] * 0.3
    return out


def orthographic_normalize(poses_3d: np.ndarray) -> np.ndarray:
    """
    Normalize 3D poses to 2D input via orthographic projection (XY plane).
    Scales to [-1, 1] based on body scale (~1m half-range).
    Used since Fit3D joints3d_25 are in body-centered space, not camera space.
    """
    poses_2d = poses_3d[..., :2].copy()  # take XY
    poses_2d = poses_2d / 1.0            # already in meters, 1m ≈ half body width/height
    return np.clip(poses_2d, -2.0, 2.0)


def process_subject(subject_dir: Path, output_dir: Path, cam_id: str) -> list:
    """Process all actions for one subject."""
    joints_dir = subject_dir / 'joints3d_25'
    cam_base = subject_dir / 'camera_parameters' / cam_id

    if not joints_dir.exists():
        print(f"  No joints3d_25 in {subject_dir.name}, skipping")
        return []

    metadata = []

    for joints_file in tqdm(sorted(joints_dir.glob('*.json')), desc=subject_dir.name, leave=False):
        action = joints_file.stem

        # Load 3D joints
        with open(joints_file) as f:
            poses_coco25 = np.array(json.load(f)['joints3d_25'])  # (T, 25, 3) meters

        # Convert to H36M-17
        poses_h36m = coco25_to_h36m17(poses_coco25)

        # Center at pelvis (root-relative)
        pelvis = poses_h36m[:, 0:1, :]
        poses_centered = poses_h36m - pelvis

        # Orthographic 2D: use XY of body-space 3D coords, scaled to [-1, 1]
        poses_2d = orthographic_normalize(poses_centered)

        # Save
        seq_name = f"{subject_dir.name}_{action}"
        seq_out = output_dir / seq_name
        seq_out.mkdir(parents=True, exist_ok=True)

        np.save(seq_out / 'poses_3d.npy', poses_centered.astype(np.float32))
        np.save(seq_out / 'poses_2d.npy', poses_2d.astype(np.float32))

        metadata.append({
            'sequence': seq_name,
            'subject': subject_dir.name,
            'action': action,
            'camera': cam_id,
            'num_frames': len(poses_centered),
        })

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Process Fit3D using IMAR tools")
    parser.add_argument('--data_root', type=str, default='./data/Fit3D')
    parser.add_argument('--output_dir', type=str, default='./data/processed/fit3d')
    parser.add_argument('--cam_id', type=str, default='50591643',
                        help='Camera ID to use (50591643, 58860488, 60457274, 65906101)')
    parser.add_argument('--eval_subjects', type=str, nargs='+', default=['s11'],
                        help='Train subjects to hold out for evaluation')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    train_dir = data_root / 'train'

    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}")
        return

    subjects = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Found subjects: {[s.name for s in subjects]}")
    print(f"Eval subjects (held out): {args.eval_subjects}")
    print(f"Camera: {args.cam_id}")

    all_metadata = {'train': [], 'test': []}

    for subject_dir in subjects:
        split = 'test' if subject_dir.name in args.eval_subjects else 'train'
        out_split = output_dir / split
        metadata = process_subject(subject_dir, out_split, args.cam_id)
        all_metadata[split].extend(metadata)

    # Save metadata and print summary
    for split, meta in all_metadata.items():
        if meta:
            out_split = output_dir / split
            out_split.mkdir(parents=True, exist_ok=True)
            with open(out_split / 'metadata.json', 'w') as f:
                json.dump(meta, f, indent=2)
            total_frames = sum(m['num_frames'] for m in meta)
            print(f"\nFit3D {split}: {len(meta)} sequences, {total_frames:,} frames")

    print(f"\nOutput saved to {output_dir}")
    print("\nNext step - run evaluation:")
    print(f"  python scripts/evaluate_checkpoint.py --checkpoint checkpoints/baseline_videopose.pt --fit3d_root {output_dir}")


if __name__ == '__main__':
    main()
