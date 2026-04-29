"""
Prepare H36M + Fit3D data for ACAE training with 17-joint latent space.

Produces:
  acae_data/poses_train.npy    — (N, J_unified, 3)  training poses
  acae_data/poses_test.npy     — (N, J_unified, 3)  test poses
  acae_data/joint_names.npy    — unified joint name list
  acae_data/h36m_joint_mask.npy — boolean mask (J_unified,) marking which
                                   unified joints are the H36M 17 joints
  acae_data/h36m_joint_indices.npy — int indices into unified skeleton for
                                      H36M's 17 joints (in H36M order)

The unified skeleton is the union of H36M-17 and Fit3D BODY_25, with
duplicate joints merged. Joint names follow the ACAE l/r/center convention.
"""

import json
import tarfile
import os
import sys
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Joint name definitions
# ────────────────────────────────────────────────────────────────────────────

# H36M 17-joint skeleton (after static joint removal) — canonical VDP3D order
H36M_NAMES_17 = [
    'pelvis',       # 0
    'rhip',         # 1
    'rknee',        # 2
    'rankle',       # 3
    'lhip',         # 4
    'lknee',        # 5
    'lankle',       # 6
    'spine',        # 7
    'thorax',       # 8
    'neck',         # 9   (upperneck in some conventions)
    'headtop',      # 10
    'lshoulder',    # 11
    'lelbow',       # 12
    'lwrist',       # 13
    'rshoulder',    # 14
    'relbow',       # 15
    'rwrist',       # 16
]

# OpenPose BODY_25 used by Fit3D joints3d_25
BODY25_NAMES = [
    'nose',         # 0
    'neck',         # 1
    'rshoulder',    # 2
    'relbow',       # 3
    'rwrist',       # 4
    'lshoulder',    # 5
    'lelbow',       # 6
    'lwrist',       # 7
    'pelvis',       # 8   (MidHip)
    'rhip',         # 9
    'rknee',        # 10
    'rankle',       # 11
    'lhip',         # 12
    'lknee',        # 13
    'lankle',       # 14
    'reye',         # 15
    'leye',         # 16
    'rear',         # 17
    'lear',         # 18
    'lbigtoe',      # 19
    'lsmalltoe',    # 20
    'lheel',        # 21
    'rbigtoe',      # 22
    'rsmalltoe',    # 23
    'rheel',        # 24
]


def build_unified_skeleton():
    """
    Merge H36M-17 and BODY_25 into a unified skeleton.
    Returns (unified_names, h36m_to_unified, body25_to_unified).

    h36m_to_unified[i]  = index in unified skeleton for H36M joint i
    body25_to_unified[i] = index in unified skeleton for BODY25 joint i
                           (or -1 if the joint is dropped)
    """
    # Start with all 17 H36M joints (these are the ones VDP3D uses)
    unified = list(H36M_NAMES_17)

    # Map H36M → unified (trivially identity since H36M comes first)
    h36m_to_unified = list(range(17))

    # Build BODY25 → unified, adding new joints as needed
    body25_to_unified = []
    for i, name in enumerate(BODY25_NAMES):
        if name in unified:
            body25_to_unified.append(unified.index(name))
        else:
            body25_to_unified.append(len(unified))
            unified.append(name)

    return unified, h36m_to_unified, body25_to_unified


# ────────────────────────────────────────────────────────────────────────────
# H36M loader
# ────────────────────────────────────────────────────────────────────────────

def load_h36m_3d(npz_path):
    """
    Load H36M 3D poses in the 17-joint reduced skeleton.

    Uses Human36mDataset which strips static joints from the original 32.
    Returns list of arrays, each (T, 17, 3) in meters.
    """
    # We need the VDP3D common module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vdp3d'))
    from common.h36m_dataset import Human36mDataset

    dataset = Human36mDataset(npz_path)
    sequences = []
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            pos = dataset[subject][action]['positions']  # (T, 17, 3) meters
            sequences.append(pos.astype(np.float32))
    return sequences


# ────────────────────────────────────────────────────────────────────────────
# Fit3D loader
# ────────────────────────────────────────────────────────────────────────────

def load_fit3d_3d(tar_path, max_subjects=None):
    """
    Load Fit3D 3D poses from joints3d_25/ in the training archive.

    Returns list of arrays, each (T, 25, 3).
    Coordinates are in meters (Fit3D uses meters natively).
    """
    print(f'Loading Fit3D from {tar_path} …')
    tf = tarfile.open(tar_path, 'r:gz')

    sequences = []
    count = 0
    subjects_seen = set()

    try:
        for member in tf:
            if 'joints3d_25' not in member.name or not member.name.endswith('.json'):
                continue

            parts = member.name.split('/')
            subject = parts[1] if len(parts) >= 2 else 'unknown'
            subjects_seen.add(subject)

            if max_subjects and len(subjects_seen) > max_subjects:
                break

            f = tf.extractfile(member)
            data = json.load(f)
            poses = np.array(data['joints3d_25'], dtype=np.float32)  # (T, 25, 3)
            sequences.append(poses)
            count += 1

            if count % 50 == 0:
                print(f'  Loaded {count} Fit3D sequences from {len(subjects_seen)} subjects …')
    except EOFError:
        print(f'  Warning: tar.gz ended early (possibly truncated). Got {count} sequences.')

    print(f'  Total: {count} sequences from subjects {sorted(subjects_seen)}')
    return sequences


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h36m-path', default='data/data_3d_h36m.npz',
                        help='Path to H36M 3D npz')
    parser.add_argument('--fit3d-path', default='data/fit3d_train.tar.gz',
                        help='Path to Fit3D training tar.gz')
    parser.add_argument('--output-dir', default='acae_data',
                        help='Output directory')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Fraction of data for test split')
    parser.add_argument('--sample-stride', type=int, default=5,
                        help='Subsample temporal sequences every N frames')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build unified skeleton
    unified_names, h36m_to_unified, body25_to_unified = build_unified_skeleton()
    J_unified = len(unified_names)
    print(f'Unified skeleton: {J_unified} joints')
    print(f'  Names: {unified_names}')
    print(f'  H36M→unified mapping: {h36m_to_unified}')
    print(f'  BODY25→unified mapping: {body25_to_unified}')

    # ── Load H36M ────────────────────────────────────────────────────────
    h36m_seqs = load_h36m_3d(args.h36m_path)
    print(f'\nH36M: {len(h36m_seqs)} sequences')

    # Convert to unified skeleton frames
    all_poses = []
    for seq in h36m_seqs:
        # Subsample
        seq = seq[::args.sample_stride]
        for frame in seq:
            unified_pose = np.full((J_unified, 3), np.nan, dtype=np.float32)
            for h_idx, u_idx in enumerate(h36m_to_unified):
                unified_pose[u_idx] = frame[h_idx] * 1000.0  # meters → mm
            all_poses.append(unified_pose)

    n_h36m = len(all_poses)
    print(f'  H36M frames (subsampled ×{args.sample_stride}): {n_h36m}')

    # ── Load Fit3D ───────────────────────────────────────────────────────
    fit3d_seqs = load_fit3d_3d(args.fit3d_path)
    print(f'\nFit3D: {len(fit3d_seqs)} sequences')

    for seq in fit3d_seqs:
        seq_sub = seq[::args.sample_stride]
        for frame in seq_sub:
            unified_pose = np.full((J_unified, 3), np.nan, dtype=np.float32)
            for b_idx, u_idx in enumerate(body25_to_unified):
                if u_idx >= 0:
                    unified_pose[u_idx] = frame[b_idx] * 1000.0  # meters → mm
            all_poses.append(unified_pose)

    n_fit3d = len(all_poses) - n_h36m
    print(f'  Fit3D frames (subsampled ×{args.sample_stride}): {n_fit3d}')
    print(f'  Total frames: {len(all_poses)}')

    # ── Build arrays ────────────────────────────────────────────────────
    all_poses = np.array(all_poses, dtype=np.float32)

    # Replace NaN with 0 for ACAE (it uses 0 as missing-joint sentinel)
    mask = np.isfinite(all_poses).all(axis=-1)
    all_poses = np.nan_to_num(all_poses, nan=0.0)

    # ── Train/test split ────────────────────────────────────────────────
    np.random.seed(42)
    indices = np.random.permutation(len(all_poses))
    split = int((1.0 - args.test_split) * len(all_poses))

    poses_train = all_poses[indices[:split]]
    poses_test  = all_poses[indices[split:]]

    print(f'\nTrain: {poses_train.shape}, Test: {poses_test.shape}')
    print(f'Missing joints (train): {(~mask[indices[:split]]).sum()}')
    print(f'Missing joints (test):  {(~mask[indices[split:]]).sum()}')

    # ── H36M alignment metadata ──────────────────────────────────────────
    # Boolean mask: which unified joints correspond to H36M's 17
    h36m_mask = np.zeros(J_unified, dtype=bool)
    for idx in h36m_to_unified:
        h36m_mask[idx] = True

    h36m_indices = np.array(h36m_to_unified, dtype=np.int64)

    # ── Save ────────────────────────────────────────────────────────────
    np.save(os.path.join(args.output_dir, 'poses_train.npy'), poses_train)
    np.save(os.path.join(args.output_dir, 'poses_test.npy'), poses_test)
    np.save(os.path.join(args.output_dir, 'joint_names.npy'), np.array(unified_names))
    np.save(os.path.join(args.output_dir, 'h36m_joint_mask.npy'), h36m_mask)
    np.save(os.path.join(args.output_dir, 'h36m_joint_indices.npy'), h36m_indices)

    print(f'\nSaved to {args.output_dir}/:')
    print(f'  poses_train.npy      ({poses_train.shape})')
    print(f'  poses_test.npy       ({poses_test.shape})')
    print(f'  joint_names.npy      ({len(unified_names)} joints)')
    print(f'  h36m_joint_mask.npy  ({h36m_mask.sum()} of {J_unified} are H36M)')
    print(f'  h36m_joint_indices.npy ({len(h36m_indices)} indices)')


if __name__ == '__main__':
    main()
