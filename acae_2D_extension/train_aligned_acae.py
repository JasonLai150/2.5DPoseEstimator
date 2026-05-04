"""
ACAE training with 17-joint H36M-aligned latent space.

Key innovation: on H36M samples (where all 17 H36M joints are present),
we add a supervised alignment loss that forces the ACAE's 17 latent points
to reconstruct the H36M 17-joint positions. This means the latent space
is semantically equivalent to H36M joints, so pretrained VDP3D weights
can be fine-tuned directly.

Usage:
    python train_aligned_acae.py [--device cuda] [--epochs 30]

Produces:
    acae_data/acae_aligned_checkpoint.pth  — full checkpoint
    acae_data/result.npz                   — w1/w2 matrices
"""

import csv
import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from run_paths import ensure_artifact_dirs, CHECKPOINT_DIR

# Import the PyTorch ACAE model directly (bypass __init__.py which imports TF)
import importlib.util
_acae_path = os.path.join(os.path.dirname(__file__),
                          'affine_combining_autoencoder', 'acae_2.5d_torch.py')
_spec = importlib.util.spec_from_file_location('acae_torch', _acae_path)
_acae_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acae_mod)

AffineCombiningAutoencoder = _acae_mod.AffineCombiningAutoencoder
normalize_weights = _acae_mod.normalize_weights
block_concat = _acae_mod.block_concat
invert_permutation = _acae_mod.invert_permutation
permute_weights = _acae_mod.permute_weights
splat = _acae_mod.splat
get_lr = _acae_mod.get_lr


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AlignedPoseDataset(Dataset):
    """
    Returns (pose, h36m_target, is_h36m) tuples.

    For H36M samples, h36m_target contains the 17-joint 3D ground truth
    in VDP3D's joint order. For Fit3D samples, h36m_target is zeros and
    is_h36m is False.
    """

    def __init__(self, poses, h36m_joint_indices):
        """
        Args:
            poses: (N, J_unified, 3) — unified joint poses (0 = missing)
            h36m_joint_indices: (17,) — indices into unified skeleton
                                        for H36M's 17 joints in VDP3D order
        """
        self.poses = torch.from_numpy(poses.astype(np.float32))
        self.h36m_indices = torch.from_numpy(h36m_joint_indices.astype(np.int64))

        # A sample is H36M if ALL 17 H36M joints are present (non-zero)
        h36m_subset = self.poses[:, self.h36m_indices]  # (N, 17, 3)
        is_present = (h36m_subset != 0.0).any(dim=-1)   # (N, 17)
        self.is_h36m = is_present.all(dim=-1)            # (N,)
        print(f'  Dataset: {len(self.poses)} samples, '
              f'{self.is_h36m.sum().item()} H36M, '
              f'{(~self.is_h36m).sum().item()} non-H36M')

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx]
        h36m_target = pose[self.h36m_indices]  # (17, 3)
        return pose, h36m_target, self.is_h36m[idx]


# ─── Aligned Loss ─────────────────────────────────────────────────────────────

def compute_aligned_loss(
    pose3d_in,      # (B, J_unified, 3)
    pose3d_pred,    # (B, J_unified, 3) — reconstruction
    latent,         # (B, 17, 3) — encoder output
    h36m_target,    # (B, 17, 3) — ground-truth H36M joints
    is_h36m,        # (B,) — bool mask
    model,
    regul_lambda=0.6,
    align_lambda=1.0,
):
    """
    Combined loss:
    1. Reconstruction loss (projected 2D MAE) — same as original ACAE
    2. Alignment loss — on H36M samples, force latent ≈ H36M joints
    3. L1 weight regularization
    """
    # ── 1. Reconstruction loss ───────────────────────────────────────────
    x_3d = pose3d_in / 1000.0
    y_3d = pose3d_pred / 1000.0

    is_missing = (pose3d_in == 0.0).all(dim=-1, keepdim=True)
    is_valid = ~is_missing

    # 3D MAE for true-3D samples
    diffs_3d = torch.where(is_valid, (x_3d - y_3d).abs(), torch.zeros_like(x_3d))
    n_valid = is_valid.float().sum(dim=[1, 2]) * x_3d.shape[-1] + 1e-6
    loss_3d = diffs_3d.sum(dim=[1, 2]) / n_valid

    # 2D projected MAE
    x_proj, y_proj = splat(pose3d_in, pose3d_pred)
    is_valid_2d = is_valid[..., :1].expand_as(x_proj)
    diffs_2d = torch.where(is_valid_2d, (x_proj - y_proj).abs(), torch.zeros_like(x_proj))
    n_valid_2d = is_valid_2d.float().sum(dim=[1, 2]) * x_proj.shape[-1] + 1e-6
    loss_2d = diffs_2d.sum(dim=[1, 2]) / n_valid_2d

    # Route: true 3D → 3D loss, flat Z → 2D loss
    z_vals = pose3d_in[..., 2]
    z_fill = torch.where(is_missing.squeeze(-1), torch.full_like(z_vals, 1000.0), z_vals)
    z_range = z_fill.max(dim=1).values - z_fill.min(dim=1).values
    is_3d_batch = z_range > 1e-3

    recon_loss = torch.where(is_3d_batch, loss_3d, loss_2d).mean()

    # ── 2. Alignment loss ─────────────────────────────────────────────────
    # On H36M samples, the ACAE's 17 latent points should match the
    # actual H36M 17-joint positions. This is what makes the latent space
    # interpretable by pretrained VDP3D.
    if is_h36m.any():
        latent_h36m = latent[is_h36m]        # (N_h36m, 17, 3)
        target_h36m = h36m_target[is_h36m]   # (N_h36m, 17, 3)

        # target_h36m is in VDP3D order. latent_h36m is in Left-Right-Center order.
        # We must reorder target_h36m to match the LRC latent format.
        lrc_order = [4, 5, 6, 11, 12, 13, 1, 2, 3, 14, 15, 16, 0, 7, 8, 9, 10]
        target_h36m_lrc = target_h36m[:, lrc_order, :]

        # Normalize both to mm scale
        align_diff = (latent_h36m / 1000.0 - target_h36m_lrc / 1000.0).abs()
        align_loss = align_diff.mean()
    else:
        align_loss = torch.tensor(0.0, device=pose3d_in.device)

    # ── 3. Weight regularization ──────────────────────────────────────────
    regul = model.encoder.get_w().abs().mean() + model.decoder.get_w().abs().mean()

    total = recon_loss + align_lambda * align_loss + regul_lambda * regul

    return {
        'loss': total,
        'recon_loss': recon_loss,
        'align_loss': align_loss,
        'regul': regul,
    }


# ─── Training ─────────────────────────────────────────────────────────────────

def train_aligned_acae(
    poses_train, poses_test, joint_names, h36m_joint_indices,
    n_latent_sided=12, n_latent_center=5,
    batch_size=64, regul_lambda=0.6, align_lambda=1.0,
    training_epochs=30, device='cpu', checkpoint_dir='acae_data',
    run_name: str | None = None,
):
    """
    Train ACAE with 17-joint latent space aligned to H36M.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    assert n_latent_sided + n_latent_center == 17, \
        f'Latent must sum to 17, got {n_latent_sided + n_latent_center}'

    # ── Joint grouping ──────────────────────────────────────────────────
    left_ids = [i for i, n in enumerate(joint_names) if n[0] == 'l']
    right_ids = [joint_names.index('r' + n[1:])
                 for i, n in enumerate(joint_names) if n[0] == 'l']
    center_ids = [i for i, n in enumerate(joint_names) if n[0] not in 'lr']
    permutation = left_ids + right_ids + center_ids
    inv_permutation = invert_permutation(permutation)

    print(f'Joint grouping: {len(left_ids)}L + {len(right_ids)}R + {len(center_ids)}C '
          f'= {len(permutation)} total')

    # Permute data
    poses_train_perm = poses_train[:, permutation]
    poses_test_perm = poses_test[:, permutation]

    # Remap h36m_joint_indices through the permutation
    # h36m_joint_indices[i] gives the unified-skeleton index for H36M joint i.
    # After permutation, we need the *permuted* index for each H36M joint.
    # inv_permutation[unified_idx] = permuted_idx
    h36m_indices_permuted = np.array(
        [inv_permutation[idx] for idx in h36m_joint_indices], dtype=np.int64)
    print(f'H36M indices (unified): {h36m_joint_indices}')
    print(f'H36M indices (permuted): {h36m_indices_permuted}')

    # Build data loaders
    train_ds = AlignedPoseDataset(poses_train_perm, h36m_indices_permuted)
    test_ds = AlignedPoseDataset(poses_test_perm, h36m_indices_permuted)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(test_ds, batch_size=batch_size,
                            shuffle=False, drop_last=False, num_workers=0)

    # ── Model ───────────────────────────────────────────────────────────
    model = AffineCombiningAutoencoder(
        n_sided_joints=len(left_ids) + len(right_ids),
        n_center_joints=len(center_ids),
        n_latent_points_sided=n_latent_sided,
        n_latent_points_center=n_latent_center,
        chiral=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=get_lr(0))

    print(f'\nModel: {sum(p.numel() for p in model.parameters())} parameters')
    print(f'Latent: {n_latent_sided}S + {n_latent_center}C = '
          f'{n_latent_sided + n_latent_center} joints')

    # ── Training loop ───────────────────────────────────────────────────
    global_step = 0
    log_rows = []

    for epoch in range(1, training_epochs + 1):
        model.train()
        train_totals = {'loss': 0, 'recon_loss': 0, 'align_loss': 0, 'regul': 0}
        n_batches = 0

        for pose, h36m_target, is_h36m in train_loader:
            pose = pose.to(device)
            h36m_target = h36m_target.to(device)
            is_h36m = is_h36m.to(device)

            optimizer.zero_grad()

            latent = model.encode(pose)     # (B, 17, 3)
            pred = model.decode(latent)     # (B, J_unified, 3)

            losses = compute_aligned_loss(
                pose, pred, latent, h36m_target, is_h36m,
                model, regul_lambda, align_lambda)

            losses['loss'].backward()
            optimizer.step()

            global_step += 1
            new_lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

            for k in train_totals:
                train_totals[k] += losses[k].item()
            n_batches += 1

        train_avg = {k: v / n_batches for k, v in train_totals.items()}

        # Validate
        model.eval()
        val_totals = {'loss': 0, 'recon_loss': 0, 'align_loss': 0, 'regul': 0}
        n_val = 0
        with torch.no_grad():
            for pose, h36m_target, is_h36m in val_loader:
                pose = pose.to(device)
                h36m_target = h36m_target.to(device)
                is_h36m = is_h36m.to(device)

                latent = model.encode(pose)
                pred = model.decode(latent)

                losses = compute_aligned_loss(
                    pose, pred, latent, h36m_target, is_h36m,
                    model, regul_lambda, align_lambda)

                for k in val_totals:
                    val_totals[k] += losses[k].item()
                n_val += 1

        val_avg = {k: v / max(n_val, 1) for k, v in val_totals.items()}

        print(f'Epoch {epoch:3d}/{training_epochs}  '
              f'train={train_avg["loss"]:.5f} (recon={train_avg["recon_loss"]:.5f} '
              f'align={train_avg["align_loss"]:.5f})  '
              f'val={val_avg["loss"]:.5f}  lr={get_lr(global_step):.1e}')

        log_rows.append({
            'epoch': epoch,
            'train_loss': train_avg['loss'],
            'train_recon': train_avg['recon_loss'],
            'train_align': train_avg['align_loss'],
            'val_loss': val_avg['loss'],
            'val_recon': val_avg['recon_loss'],
            'val_align': val_avg['align_loss'],
        })

    # ── Extract final weights ────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        w1 = model.encoder.get_w().cpu().numpy()
        w2 = model.decoder.get_w().cpu().numpy()
    w1, w2 = permute_weights(w1, w2, inv_permutation)

    # ── Save ─────────────────────────────────────────────────────────────
    prefix = f'{run_name}_' if run_name else ''
    result_name = f'{prefix}result.npz' if run_name else 'result.npz'
    checkpoint_name = f'{prefix}checkpoint.pth' if run_name else 'acae_aligned_checkpoint.pth'
    csv_name = f'{prefix}losses.csv' if run_name else 'losses.csv'

    np.savez(os.path.join(checkpoint_dir, result_name), w1=w1, w2=w2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'epoch': training_epochs,
        'hyperparams': {
            'n_latent_sided': n_latent_sided,
            'n_latent_center': n_latent_center,
            'n_sided_joints': len(left_ids) + len(right_ids),
            'n_center_joints': len(center_ids),
            'chiral': True,
        },
        'permutation': permutation,
        'inv_permutation': inv_permutation,
        'joint_names': joint_names,
        'h36m_joint_indices': h36m_joint_indices.tolist(),
        'w1': w1,
        'w2': w2,
    }, os.path.join(checkpoint_dir, checkpoint_name))

    csv_path = os.path.join(checkpoint_dir, csv_name)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f'\nSaved: {os.path.join(checkpoint_dir, result_name)}  (w1={w1.shape}, w2={w2.shape})')
    print(f'Saved: {os.path.join(checkpoint_dir, checkpoint_name)}')
    return w1, w2, model


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train ACAE with 17-joint H36M-aligned latent space')
    parser.add_argument('--data-dir', default='acae_data')
    parser.add_argument('--checkpoint-dir', default=str(CHECKPOINT_DIR))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--align-lambda', type=float, default=1.0,
                        help='Weight for H36M alignment loss')
    parser.add_argument('--regul-lambda', type=float, default=0.6,
                        help='Weight for L1 weight regularization')
    parser.add_argument('--run-name', default=None,
                        help='Optional prefix for saved checkpoint/result/loss files')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'Device: {device}')
    ensure_artifact_dirs()

    # Load data
    poses_train = np.load(os.path.join(args.data_dir, 'poses_train.npy'))
    poses_test = np.load(os.path.join(args.data_dir, 'poses_test.npy'))
    joint_names = list(np.load(os.path.join(args.data_dir, 'joint_names.npy')))
    h36m_indices = np.load(os.path.join(args.data_dir, 'h36m_joint_indices.npy'))

    print(f'Train: {poses_train.shape}, Test: {poses_test.shape}')
    print(f'Joints: {joint_names}')
    print(f'H36M indices: {h36m_indices}')

    # Count left/right/center to determine latent split
    left = [n for n in joint_names if n[0] == 'l']
    right = [n for n in joint_names if n[0] == 'r']
    center = [n for n in joint_names if n[0] not in 'lr']
    print(f'\nJoint split: {len(left)}L + {len(right)}R + {len(center)}C')

    # H36M 17-joint skeleton has 6L + 6R + 5C
    # Latent must be 17 total. We match H36M's natural split.
    n_latent_sided = 12   # 6 left + 6 right
    n_latent_center = 5

    train_aligned_acae(
        poses_train=poses_train,
        poses_test=poses_test,
        joint_names=joint_names,
        h36m_joint_indices=h36m_indices,
        n_latent_sided=n_latent_sided,
        n_latent_center=n_latent_center,
        batch_size=args.batch_size,
        regul_lambda=args.regul_lambda,
        align_lambda=args.align_lambda,
        training_epochs=args.epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
    )


if __name__ == '__main__':
    main()
