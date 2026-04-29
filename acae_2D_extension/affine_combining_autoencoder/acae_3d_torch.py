# Copyright (C) 2023 István Sárándi (original TensorFlow version)
# PyTorch port of acae_test.py — original 3D-only ACAE for bridging 3D datasets.
# No TF/Keras/fleras dependency.
# Saves:
#   result.npz             — w1, w2 matrices (backward-compatible)
#   acae_3d_checkpoint.pth — full PyTorch checkpoint (for end-to-end use)

import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ─── Utilities ────────────────────────────────────────────────────────────────

def normalize_weights(w: torch.Tensor) -> torch.Tensor:
    """Column-normalize so each latent point's affine weights sum to 1."""
    return w / (w.sum(dim=0, keepdim=True) + 1e-9)


def block_concat(blocks) -> torch.Tensor:
    """Concatenate a list-of-lists of tensors into a block matrix."""
    return torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)


def invert_permutation(perm: list) -> list:
    """Pure-Python equivalent of tf.math.invert_permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def permute_weights(w1: np.ndarray, w2: np.ndarray, inv_perm: list):
    """Restore original joint ordering after training on permuted joints."""
    return w1[inv_perm, :], w2[:, inv_perm]


def splat(x: torch.Tensor, y: torch.Tensor):
    """
    Project 3-D poses to 2-D with a weak-perspective / mean-depth model.
    Matches the original paper's simple splat (no safe-divide, no NaN guard).
    """
    z_mean = x[..., 2:].mean(dim=1, keepdim=True)          # (B, 1, 1)
    x_proj = x[..., :2] / x[..., 2:] * z_mean / 1000.0
    y_proj = y[..., :2] / y[..., 2:] * z_mean / 1000.0
    return x_proj, y_proj


def get_lr(step: int) -> float:
    """Piecewise-constant LR schedule matching the original TF version."""
    if step < 150_000:
        return 3e-2
    elif step < 300_000:
        return 3e-3
    return 3e-4


# ─── Model ────────────────────────────────────────────────────────────────────

class AffineCombinationLayer(nn.Module):
    """
    Original paper's affine-combination layer (no missing-joint masking).

    Builds a block weight matrix W from 5 sub-weights encoding L/R/center symmetry:

        W = | w_s   w_q   w_x |   ← left  joints
            | w_q'  w_s'  w_x'|   ← right joints
            | w_c   w_c'  w_z |   ← center joints

    When chiral=True the primed blocks share the same nn.Parameter as their
    unprimed counterparts (bilateral symmetry).

    Forward pass is a clean einsum('bjc,jJ->bJc') — no masking.
    """

    def __init__(
        self,
        n_sided_points: int,
        n_center_points: int,
        n_latent_points_sided: int,
        n_latent_points_center: int,
        transposed: bool,
        chiral: bool = True,
    ):
        super().__init__()
        self.transposed = transposed
        self.chiral = chiral

        hs = n_sided_points // 2           # half-sided joints
        hl = n_latent_points_sided // 2    # half-latent sided

        def _p(rows, cols):
            return nn.Parameter(torch.empty(rows, cols).uniform_(-0.1, 1.0))

        # Shared (left-side / always-single) blocks
        self.w_s = _p(hs, hl)
        self.w_q = _p(hs, hl)
        self.w_x = _p(hs, n_latent_points_center)
        self.w_c = _p(n_center_points, hl)
        self.w_z = _p(n_center_points, n_latent_points_center)   # always single

        # Independent right-side blocks (only used when chiral=False)
        if not chiral:
            self.w_s_r = _p(hs, hl)
            self.w_q_r = _p(hs, hl)
            self.w_x_r = _p(hs, n_latent_points_center)
            self.w_c_r = _p(n_center_points, hl)

    def _blocks(self):
        w_s_r = self.w_s if self.chiral else self.w_s_r
        w_q_r = self.w_q if self.chiral else self.w_q_r
        w_x_r = self.w_x if self.chiral else self.w_x_r
        w_c_r = self.w_c if self.chiral else self.w_c_r
        return [
            [self.w_s, self.w_q,  self.w_x],    # left joints
            [w_q_r,    w_s_r,     w_x_r   ],    # right joints
            [self.w_c, w_c_r,     self.w_z ],    # center joints
        ]

    def get_w(self) -> torch.Tensor:
        """Return the normalized (optionally transposed) weight matrix."""
        w = block_concat(self._blocks())            # (J_in, J_latent)
        if self.transposed:
            w = w.T                                 # decoder: (J_latent, J_in)
        return normalize_weights(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:   x: (B, J_in, C)    — C is typically 3 (x, y, z)
        Returns:   (B, J_out, C)
        """
        return torch.einsum('bjc,jJ->bJc', x, self.get_w())


class AffineCombiningAutoencoder(nn.Module):
    """
    Encoder → decoder ACAE sandwich (original paper version).
    Can be used standalone or wrapped around VideoPose3D.
    """

    def __init__(
        self,
        n_sided_joints: int,
        n_center_joints: int,
        n_latent_points_sided: int,
        n_latent_points_center: int,
        chiral: bool = True,
    ):
        super().__init__()
        args = (n_sided_joints, n_center_joints,
                n_latent_points_sided, n_latent_points_center)
        self.encoder = AffineCombinationLayer(*args, transposed=False, chiral=chiral)
        self.decoder = AffineCombinationLayer(*args, transposed=True,  chiral=chiral)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_acae_loss(
    pose3d_in: torch.Tensor,
    pose3d_pred: torch.Tensor,
    model: AffineCombiningAutoencoder,
    regul_lambda: float,
    use_projected_loss: bool = True,
) -> dict:
    """
    Original paper loss: projected-2D MAE + L1 weight regularization.
    When use_projected_loss=True, 3D poses are splatted to 2D before computing
    MAE — this is the trick that lets the ACAE bridge datasets with different
    depth scales and camera conventions.
    """
    if use_projected_loss:
        x, y = splat(pose3d_in, pose3d_pred)
    else:
        x, y = pose3d_in / 1000.0, pose3d_pred / 1000.0

    main_loss = (x - y).abs().mean()

    # L1 weight regularization
    regul = model.encoder.get_w().abs().mean() + model.decoder.get_w().abs().mean()
    total = main_loss + regul_lambda * regul

    return {'loss': total, 'main_loss': main_loss, 'regul': regul}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    def __init__(self, poses: np.ndarray):
        self.poses = torch.from_numpy(poses.astype(np.float32))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx]


# ─── Training loop ────────────────────────────────────────────────────────────

def train_acae(
    poses_train: np.ndarray,
    poses_test: np.ndarray,
    joint_names: list,
    n_latent_sided: int = 40,
    n_latent_center: int = 8,
    batch_size: int = 32,
    regul_lambda: float = 6e-1,
    training_epochs: int = 30,
    device: str = 'cpu',
    checkpoint_dir: str = '.',
):
    """
    Standalone ACAE training loop — faithful port of train_acae() in acae_test.py.

    Returns:
        w1    (np.ndarray) — normalized encoder weight matrix  (J, L)
        w2    (np.ndarray) — normalized decoder weight matrix  (L, J)
        model              — trained AffineCombiningAutoencoder
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Joint grouping (identical to original) ──────────────────────────────
    left_ids   = [i for i, n in enumerate(joint_names) if n[0] == 'l']
    right_ids  = [joint_names.index('r' + n[1:])
                  for i, n in enumerate(joint_names) if n[0] == 'l']
    center_ids = [i for i, n in enumerate(joint_names) if n[0] not in 'lr']
    permutation     = left_ids + right_ids + center_ids
    inv_permutation = invert_permutation(permutation)

    poses_train = poses_train[:, permutation]
    poses_test  = poses_test[:, permutation]

    train_loader = DataLoader(
        PoseDataset(poses_train), batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(
        PoseDataset(poses_test), batch_size=batch_size,
        shuffle=False, drop_last=False, num_workers=0)

    # ── Model + optimizer ───────────────────────────────────────────────────
    model = AffineCombiningAutoencoder(
        n_sided_joints=len(left_ids) + len(right_ids),
        n_center_joints=len(center_ids),
        n_latent_points_sided=n_latent_sided,
        n_latent_points_center=n_latent_center,
        chiral=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=get_lr(0))

    # ── Training ────────────────────────────────────────────────────────────
    global_step = 0
    log_rows    = []

    for epoch in range(1, training_epochs + 1):
        model.train()
        train_acc = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred   = model(batch)
            losses = compute_acae_loss(batch, pred, model, regul_lambda,
                                       use_projected_loss=True)
            losses['loss'].backward()
            optimizer.step()

            global_step += 1
            new_lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

            train_acc += losses['loss'].item()

        train_loss = train_acc / len(train_loader)

        # Validate
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(device)
                pred   = model(batch)
                losses = compute_acae_loss(batch, pred, model, regul_lambda,
                                           use_projected_loss=True)
                val_acc += losses['loss'].item()
        val_loss = val_acc / len(val_loader)

        print(f'Epoch {epoch:3d}/{training_epochs}  '
              f'train={train_loss:.5f}  val={val_loss:.5f}  '
              f'lr={get_lr(global_step):.1e}')
        log_rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

    # ── Extract final weight matrices ────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        w1 = model.encoder.get_w().cpu().numpy()
        w2 = model.decoder.get_w().cpu().numpy()
    w1, w2 = permute_weights(w1, w2, inv_permutation)

    # ── Save .npz (backward-compatible with original) ────────────────────────
    np.savez(os.path.join(checkpoint_dir, 'result.npz'), w1=w1, w2=w2)

    # ── Save .pth (for end-to-end integration with VideoPose3D) ──────────────
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step':          global_step,
        'epoch':                training_epochs,
        'hyperparams': {
            'n_latent_sided':  n_latent_sided,
            'n_latent_center': n_latent_center,
            'n_sided_joints':  len(left_ids) + len(right_ids),
            'n_center_joints': len(center_ids),
            'chiral':          True,
        },
        'permutation':     permutation,
        'inv_permutation': inv_permutation,
        'joint_names':     joint_names,
        'w1': w1,
        'w2': w2,
    }, os.path.join(checkpoint_dir, 'acae_3d_checkpoint.pth'))

    # ── Save loss CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(checkpoint_dir, 'losses.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f'\nSaved: {os.path.join(checkpoint_dir, "result.npz")}')
    print(f'Saved: {os.path.join(checkpoint_dir, "acae_3d_checkpoint.pth")}')
    return w1, w2, model


# ─── End-to-end loader ────────────────────────────────────────────────────────

def load_acae_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu',
    freeze: bool = False,
) -> AffineCombiningAutoencoder:
    """
    Load a trained 3D ACAE from a .pth checkpoint.

    Args:
        checkpoint_path: Path to acae_3d_checkpoint.pth
        device:          'cpu' or 'cuda'
        freeze:          If True, freeze all ACAE parameters (use as fixed adapter).

    Example (sandwiching VideoPose3D)::

        acae = load_acae_from_checkpoint('acae_3d_checkpoint.pth', device='cuda')
        latent   = acae.encode(input_3d)          # (B, J_latent, 3)
        vp3d_out = videopose3d_model(latent)       # (B, J_latent, 3)
        output   = acae.decode(vp3d_out)           # (B, J, 3)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp   = ckpt['hyperparams']
    model = AffineCombiningAutoencoder(
        n_sided_joints=hp['n_sided_joints'],
        n_center_joints=hp['n_center_joints'],
        n_latent_points_sided=hp['n_latent_sided'],
        n_latent_points_center=hp['n_latent_center'],
        chiral=hp['chiral'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    n_latent_sided  = 40
    n_latent_center = 8
    batch_size      = 32
    regul_lambda    = 6e-1
    training_epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    poses_train = np.load('poses_train.npy')   # Shape [n_train, n_joints, 3]
    poses_test  = np.load('poses_test.npy')    # Shape [n_test,  n_joints, 3]
    joint_names = list(np.load('joint_names.npy'))

    train_acae(
        poses_train=poses_train,
        poses_test=poses_test,
        joint_names=joint_names,
        n_latent_sided=n_latent_sided,
        n_latent_center=n_latent_center,
        batch_size=batch_size,
        regul_lambda=regul_lambda,
        training_epochs=training_epochs,
        device=device,
        checkpoint_dir='.',
    )


if __name__ == '__main__':
    main()
