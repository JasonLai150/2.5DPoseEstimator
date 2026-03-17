#!/usr/bin/env python3
"""
Training entry point for 2.5D pose estimation.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --epochs 200 --lr 1e-4
    python scripts/train.py --lora  # Enable LoRA fine-tuning
"""

import argparse
from pathlib import Path
import torch
import random
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, merge_configs
from src.data.datasets import create_dataloaders
from src.models import create_model
from src.training import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train 2.5D pose estimator")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    args = parser.parse_args()

    # Load config
    cfg = load_config(Path(args.config))

    # Apply CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.lr is not None:
        cfg.training.optimizer.lr = args.lr
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.pretrained is not None:
        cfg.model.pretrained_path = args.pretrained
    if args.lora:
        cfg.model.lora.enabled = True
        cfg.model.lora.rank = args.lora_rank

    set_seed(cfg.get("seed", 42))

    print("=" * 60)
    print("2.5D Pose Estimator Training")
    print("=" * 60)
    print(f"Model: {cfg.model.get('name', 'dstformer')}")
    print(f"LoRA: {'Enabled' if cfg.model.get('lora', {}).get('enabled', False) else 'Disabled'}")
    if cfg.model.get('pretrained_path'):
        print(f"Pretrained: {cfg.model.pretrained_path}")
    print("=" * 60)

    # Create dataloaders
    dataloaders = create_dataloaders(cfg)
    print(f"Train samples: {len(dataloaders.get('train', []))}")
    print(f"Val samples: {len(dataloaders.get('val', []))}")

    # Create model
    model = create_model(cfg)
    print(f"Model parameters: {model.count_parameters():,} (trainable)")
    print(f"Total parameters: {model.count_parameters(trainable_only=False):,}")

    # Create trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=dataloaders.get("train"),
        val_loader=dataloaders.get("val"),
        test_loader=dataloaders.get("test"),
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        print(f"Resumed from {args.resume_from}")

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
