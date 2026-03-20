#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on H36M and/or Fit3D.

Usage:
    # H36M only
    python scripts/evaluate_checkpoint.py --checkpoint checkpoints/baseline_videopose.pt

    # H36M + Fit3D
    python scripts/evaluate_checkpoint.py --checkpoint checkpoints/baseline_videopose.pt \
        --fit3d_root ./data/processed/fit3d
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processed_dataset import ProcessedPoseDataset, create_dataloader
from src.models.videopose import VideoPose3D


class SimpleConfig:
    def __init__(self, **kwargs):
        self.model = type('obj', (object,), {})()
        for k, v in kwargs.get('model', {}).items():
            setattr(self.model, k, v)
        self.model.get = lambda key, default=None: getattr(self.model, key, default)


def compute_mpjpe(pred, target, valid=None):
    """Compute MPJPE in mm."""
    pred_mm = pred * 1000
    target_mm = target * 1000
    error = torch.sqrt(((pred_mm - target_mm) ** 2).sum(dim=-1))
    if valid is not None:
        valid_exp = valid.unsqueeze(-1).expand_as(error)
        error = error[valid_exp]
    return error.mean().item()


def compute_p_mpjpe(pred, target, valid=None):
    """Compute Procrustes-aligned MPJPE (batched)."""
    pred_mm = pred * 1000
    target_mm = target * 1000
    B, T, J, _ = pred_mm.shape

    p = pred_mm.reshape(-1, J, 3).cpu().numpy()
    t = target_mm.reshape(-1, J, 3).cpu().numpy()

    p_centered = p - p.mean(axis=1, keepdims=True)
    t_centered = t - t.mean(axis=1, keepdims=True)

    p_scale = np.sqrt((p_centered ** 2).sum(axis=(1, 2), keepdims=True)) + 1e-8
    t_scale = np.sqrt((t_centered ** 2).sum(axis=(1, 2), keepdims=True)) + 1e-8

    p_norm = p_centered / p_scale
    t_norm = t_centered / t_scale

    H = np.einsum('nij,nik->njk', p_norm, t_norm)
    U, S, Vt = np.linalg.svd(H)
    R = np.einsum('nij,nkj->nik', Vt, U)

    det = np.linalg.det(R)
    Vt[det < 0, -1, :] *= -1
    R = np.einsum('nij,nkj->nik', Vt, U)

    t_mean = t.mean(axis=1, keepdims=True)
    p_aligned = np.einsum('nij,nkj->nki', R, p_norm) * t_scale + t_mean
    errors = np.sqrt(((p_aligned - t) ** 2).sum(axis=-1)).mean(axis=-1)

    return errors.mean()


@torch.no_grad()
def evaluate(model, loader, device, name=""):
    model.eval()
    all_mpjpe, all_p_mpjpe = [], []

    for batch in tqdm(loader, desc=f"Evaluating {name}"):
        poses_2d = batch['poses_2d'].to(device)
        poses_3d = batch['poses_3d'].to(device)
        valid = batch.get('valid')
        if valid is not None:
            valid = valid.to(device)

        pred_3d = model(poses_2d)
        all_mpjpe.append(compute_mpjpe(pred_3d, poses_3d, valid))
        all_p_mpjpe.append(compute_p_mpjpe(pred_3d, poses_3d, valid))

    return {'mpjpe': np.mean(all_mpjpe), 'p_mpjpe': np.mean(all_p_mpjpe)}


def make_loader(data_root, dataset, split, batch_size, seq_len):
    ds = ProcessedPoseDataset(
        data_root=data_root,
        dataset=dataset,
        split=split,
        seq_len=seq_len,
        stride=seq_len,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_videopose.pt")
    parser.add_argument("--data_root", type=str, default="./data/processed",
                        help="Root for H36M processed data")
    parser.add_argument("--fit3d_root", type=str, default=None,
                        help="Root for Fit3D processed data (e.g. ./data/processed/fit3d)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=27)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get('config', {})
    seq_len = ckpt_cfg.get('seq_len', args.seq_len)

    cfg = SimpleConfig(model={
        'num_joints': 17, 'input_dim': 2, 'output_dim': 3,
        'seq_len': seq_len,
        'hidden_dim': ckpt_cfg.get('hidden_dim', 1024),
        'num_blocks': ckpt_cfg.get('num_blocks', 4),
        'kernel_size': 3, 'drop_rate': 0.25,
    })

    model = VideoPose3D(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build loaders
    print("\nLoading datasets...")
    loaders = {}

    h36m_loader = make_loader(args.data_root, "h36m", "test", args.batch_size, seq_len)
    print(f"H36M Test:  {len(h36m_loader.dataset)} samples")
    loaders["H36M"] = h36m_loader

    if args.fit3d_root:
        fit3d_loader = make_loader(args.fit3d_root, "", "test", args.batch_size, seq_len)
        print(f"Fit3D Test: {len(fit3d_loader.dataset)} samples")
        loaders["Fit3D"] = fit3d_loader

    # Evaluate
    print("\n" + "="*60)
    results = {}
    for name, loader in loaders.items():
        results[name] = evaluate(model, loader, device, name)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for name, metrics in results.items():
        label = "(in-domain)" if name == "H36M" else "(target domain)"
        print(f"\n{name} {label}:")
        print(f"  MPJPE:   {metrics['mpjpe']:.2f} mm")
        print(f"  P-MPJPE: {metrics['p_mpjpe']:.2f} mm")

    if "H36M" in results and "Fit3D" in results:
        gap = results["Fit3D"]["mpjpe"] - results["H36M"]["mpjpe"]
        print(f"\nDomain gap (H36M -> Fit3D): {gap:.2f} mm")


if __name__ == "__main__":
    main()
