#!/usr/bin/env python3
"""
Baseline Training and Evaluation Script.

Trains VideoPose3D on Human3.6M and evaluates on MPI-INF-3DHP.
This establishes a baseline for cross-dataset generalization.

Usage:
    python scripts/run_baseline.py --epochs 10 --batch_size 256
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processed_dataset import ProcessedPoseDataset, create_dataloader
from src.models.videopose import VideoPose3D, TemporalBlock
from src.config import load_config


class SimpleConfig:
    """Simple config object for model initialization."""
    def __init__(self, **kwargs):
        self.model = type('obj', (object,), kwargs.get('model', {}))()
        self.data = type('obj', (object,), kwargs.get('data', {}))()

    def get(self, key, default=None):
        return getattr(self, key, default)


def compute_mpjpe(pred, target, valid=None):
    """
    Compute Mean Per Joint Position Error (MPJPE) in millimeters.

    Args:
        pred: (B, T, J, 3) predicted 3D poses
        target: (B, T, J, 3) ground truth 3D poses
        valid: (B, T) optional validity mask

    Returns:
        MPJPE in mm
    """
    # Convert meters to mm
    pred_mm = pred * 1000
    target_mm = target * 1000

    # Per-joint error
    error = torch.sqrt(((pred_mm - target_mm) ** 2).sum(dim=-1))  # (B, T, J)

    if valid is not None:
        # Expand valid mask to joint dimension
        valid = valid.unsqueeze(-1).expand_as(error)
        error = error[valid]

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


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        poses_2d = batch['poses_2d'].to(device)
        poses_3d = batch['poses_3d'].to(device)

        optimizer.zero_grad()

        # Forward
        pred_3d = model(poses_2d)

        # Loss (MPJPE in meters)
        loss = criterion(pred_3d, poses_3d)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())

    return total_loss / num_batches


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

        # Forward
        pred_3d = model(poses_2d)

        # Compute metrics
        mpjpe = compute_mpjpe(pred_3d, poses_3d, valid)
        p_mpjpe = compute_p_mpjpe(pred_3d, poses_3d, valid)

        all_mpjpe.append(mpjpe)
        all_p_mpjpe.append(p_mpjpe)

    return {
        'mpjpe': np.mean(all_mpjpe),
        'p_mpjpe': np.mean(all_p_mpjpe),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline training and evaluation")
    parser.add_argument("--data_root", type=str, default="./data/processed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=27)  # Shorter for faster training
    parser.add_argument("--stride", type=int, default=27)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="checkpoints/baseline_videopose.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create config for model
    cfg = SimpleConfig(
        model={
            'num_joints': 17,
            'input_dim': 2,
            'output_dim': 3,
            'seq_len': args.seq_len,
            'hidden_dim': args.hidden_dim,
            'num_blocks': args.num_blocks,
            'kernel_size': 3,
            'drop_rate': 0.25,
        },
        data={
            'num_joints': 17,
            'input_dim': 2,
            'output_dim': 3,
            'seq_len': args.seq_len,
        }
    )

    # Manually set nested attributes
    cfg.model.num_joints = 17
    cfg.model.input_dim = 2
    cfg.model.output_dim = 3
    cfg.model.seq_len = args.seq_len
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.num_blocks = args.num_blocks
    cfg.model.kernel_size = 3
    cfg.model.drop_rate = 0.25
    cfg.model.get = lambda key, default=None: getattr(cfg.model, key, default)

    cfg.data.num_joints = 17
    cfg.data.input_dim = 2
    cfg.data.output_dim = 3
    cfg.data.seq_len = args.seq_len

    # Create model
    print("\nCreating VideoPose3D model...")
    model = VideoPose3D(cfg)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader = create_dataloader(
        data_root=args.data_root,
        dataset="h36m",
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print(f"H36M Train: {len(train_loader.dataset)} samples")

    h36m_test_loader = create_dataloader(
        data_root=args.data_root,
        dataset="h36m",
        split="test",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.seq_len,  # Non-overlapping for eval
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"H36M Test: {len(h36m_test_loader.dataset)} samples")

    mpi_test_loader = create_dataloader(
        data_root=args.data_root,
        dataset="mpi_3dhp",
        split="test",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.seq_len,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"MPI-INF-3DHP Test: {len(mpi_test_loader.dataset)} samples")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Training loop
    print("\n" + "="*60)
    print("TRAINING ON HUMAN3.6M")
    print("="*60)

    best_h36m_mpjpe = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.6f}")

        # Evaluate on H36M test
        h36m_metrics = evaluate(model, h36m_test_loader, device, "H36M Test")
        print(f"H36M Test - MPJPE: {h36m_metrics['mpjpe']:.2f}mm, P-MPJPE: {h36m_metrics['p_mpjpe']:.2f}mm")

        # Save best model
        if h36m_metrics['mpjpe'] < best_h36m_mpjpe:
            best_h36m_mpjpe = h36m_metrics['mpjpe']
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'h36m_mpjpe': h36m_metrics['mpjpe'],
                'config': vars(args),
            }, args.save_path)
            print(f"  -> Saved best model (MPJPE: {best_h36m_mpjpe:.2f}mm)")

        scheduler.step()

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # Load best model
    checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\nEvaluating on H36M Test...")
    h36m_final = evaluate(model, h36m_test_loader, device, "H36M Test")

    print("\nEvaluating on MPI-INF-3DHP Test (cross-dataset)...")
    mpi_final = evaluate(model, mpi_test_loader, device, "MPI-INF-3DHP Test")

    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"\nHuman3.6M (in-domain):")
    print(f"  MPJPE:   {h36m_final['mpjpe']:.2f} mm")
    print(f"  P-MPJPE: {h36m_final['p_mpjpe']:.2f} mm")

    print(f"\nMPI-INF-3DHP (cross-dataset):")
    print(f"  MPJPE:   {mpi_final['mpjpe']:.2f} mm")
    print(f"  P-MPJPE: {mpi_final['p_mpjpe']:.2f} mm")

    print(f"\nModel saved to: {args.save_path}")


if __name__ == "__main__":
    main()
