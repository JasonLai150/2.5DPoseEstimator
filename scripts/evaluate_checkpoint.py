#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on H36M and MPI-INF-3DHP.

Usage:
    python scripts/evaluate_checkpoint.py --checkpoint checkpoints/baseline_videopose.pt
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processed_dataset import create_dataloader
from src.models.videopose import VideoPose3D


class SimpleConfig:
    """Simple config object for model initialization."""
    def __init__(self, **kwargs):
        self.model = type('obj', (object,), {})()
        self.data = type('obj', (object,), {})()
        for k, v in kwargs.get('model', {}).items():
            setattr(self.model, k, v)
        self.model.get = lambda key, default=None: getattr(self.model, key, default)


def compute_mpjpe(pred, target, valid=None):
    """Compute MPJPE in mm."""
    pred_mm = pred * 1000
    target_mm = target * 1000
    error = torch.sqrt(((pred_mm - target_mm) ** 2).sum(dim=-1))
    if valid is not None:
        valid = valid.unsqueeze(-1).expand_as(error)
        error = error[valid]
    return error.mean().item()


def compute_p_mpjpe(pred, target, valid=None):
    """Compute Procrustes-aligned MPJPE."""
    pred_mm = pred * 1000
    target_mm = target * 1000
    B, T, J, _ = pred_mm.shape
    pred_flat = pred_mm.reshape(-1, J, 3)
    target_flat = target_mm.reshape(-1, J, 3)

    errors = []
    for i in range(len(pred_flat)):
        p = pred_flat[i].cpu().numpy()
        t = target_flat[i].cpu().numpy()
        p_centered = p - p.mean(axis=0)
        t_centered = t - t.mean(axis=0)
        p_scale = np.sqrt((p_centered ** 2).sum())
        t_scale = np.sqrt((t_centered ** 2).sum())
        p_normalized = p_centered / (p_scale + 1e-8)
        t_normalized = t_centered / (t_scale + 1e-8)
        H = p_normalized.T @ t_normalized
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        p_aligned = (p_normalized @ R) * t_scale + t.mean(axis=0)
        error = np.sqrt(((p_aligned - t) ** 2).sum(axis=-1)).mean()
        errors.append(error)
    return np.mean(errors)


@torch.no_grad()
def evaluate(model, loader, device, dataset_name=""):
    """Evaluate model on a dataset."""
    model.eval()
    all_mpjpe = []
    all_p_mpjpe = []

    for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
        poses_2d = batch['poses_2d'].to(device)
        poses_3d = batch['poses_3d'].to(device)
        valid = batch.get('valid', None)
        if valid is not None:
            valid = valid.to(device)

        pred_3d = model(poses_2d)
        mpjpe = compute_mpjpe(pred_3d, poses_3d, valid)
        p_mpjpe = compute_p_mpjpe(pred_3d, poses_3d, valid)
        all_mpjpe.append(mpjpe)
        all_p_mpjpe.append(p_mpjpe)

    return {
        'mpjpe': np.mean(all_mpjpe),
        'p_mpjpe': np.mean(all_p_mpjpe),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_videopose.pt")
    parser.add_argument("--data_root", type=str, default="./data/processed")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=27)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get config from checkpoint
    ckpt_config = checkpoint.get('config', {})
    hidden_dim = ckpt_config.get('hidden_dim', 1024)
    num_blocks = ckpt_config.get('num_blocks', 4)
    seq_len = ckpt_config.get('seq_len', args.seq_len)

    # Create model
    cfg = SimpleConfig(model={
        'num_joints': 17,
        'input_dim': 2,
        'output_dim': 3,
        'seq_len': seq_len,
        'hidden_dim': hidden_dim,
        'num_blocks': num_blocks,
        'kernel_size': 3,
        'drop_rate': 0.25,
    })

    model = VideoPose3D(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    print("\nLoading datasets...")
    h36m_test_loader = create_dataloader(
        data_root=args.data_root,
        dataset="h36m",
        split="test",
        batch_size=args.batch_size,
        seq_len=seq_len,
        stride=seq_len,
        shuffle=False,
    )
    print(f"H36M Test: {len(h36m_test_loader.dataset)} samples")

    mpi_test_loader = create_dataloader(
        data_root=args.data_root,
        dataset="mpi_3dhp",
        split="test",
        batch_size=args.batch_size,
        seq_len=seq_len,
        stride=seq_len,
        shuffle=False,
    )
    print(f"MPI-INF-3DHP Test: {len(mpi_test_loader.dataset)} samples")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print("\nEvaluating on H36M Test...")
    h36m_metrics = evaluate(model, h36m_test_loader, device, "H36M")

    print("\nEvaluating on MPI-INF-3DHP Test (cross-dataset)...")
    mpi_metrics = evaluate(model, mpi_test_loader, device, "MPI-INF-3DHP")

    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"\nHuman3.6M (in-domain):")
    print(f"  MPJPE:   {h36m_metrics['mpjpe']:.2f} mm")
    print(f"  P-MPJPE: {h36m_metrics['p_mpjpe']:.2f} mm")

    print(f"\nMPI-INF-3DHP (cross-dataset):")
    print(f"  MPJPE:   {mpi_metrics['mpjpe']:.2f} mm")
    print(f"  P-MPJPE: {mpi_metrics['p_mpjpe']:.2f} mm")

    print(f"\nDomain gap: {mpi_metrics['mpjpe'] - h36m_metrics['mpjpe']:.2f} mm")


if __name__ == "__main__":
    main()
