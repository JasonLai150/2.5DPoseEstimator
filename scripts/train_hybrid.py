#!/usr/bin/env python3
"""
Hybrid training entry point: H36M (3D) + Fit3D-train (2D-only) -> DSTformer.

v1 plan (path A): lambda_reproj=0 (reprojection wiring is incorrect; tracked
in memory note "Reprojection loss broken"). Trains with L_3D + L_biomech only.

Usage:
    python scripts/train_hybrid.py --epochs 5 --smoke         # smoke test
    python scripts/train_hybrid.py --epochs 60                # full run
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.processed_dataset import ProcessedPoseDataset
from src.models import create_model
from src.training import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(cfg, smoke: bool):
    """H36M (3D) + Fit3D-train (2D-only weakly-sup) for train; Fit3D s11 for val."""
    seq_len = cfg.data.seq_len
    stride = cfg.data.get("stride", seq_len)
    bs = cfg.data.batch_size
    nw = cfg.data.num_workers

    h36m_train = ProcessedPoseDataset(
        data_root="./data/processed", dataset="h36m", split="train",
        seq_len=seq_len, stride=stride,
    )
    fit3d_train_weak = ProcessedPoseDataset(
        data_root="./data/processed/fit3d", dataset="", split="train",
        seq_len=seq_len, stride=stride, weakly_supervised=True,
    )
    fit3d_test = ProcessedPoseDataset(
        data_root="./data/processed/fit3d", dataset="", split="test",
        seq_len=seq_len, stride=seq_len,
    )

    if smoke:
        # 64 samples per loader is enough for a single-batch sanity epoch
        h36m_train = torch.utils.data.Subset(h36m_train, range(min(64, len(h36m_train))))
        fit3d_train_weak = torch.utils.data.Subset(fit3d_train_weak, range(min(64, len(fit3d_train_weak))))
        fit3d_test = torch.utils.data.Subset(fit3d_test, range(min(64, len(fit3d_test))))

    train_set = ConcatDataset([h36m_train, fit3d_train_weak])
    print(f"Train: {len(train_set)} ({len(h36m_train)} H36M + {len(fit3d_train_weak)} Fit3D-2D)")
    print(f"Val:   {len(fit3d_test)} (Fit3D s11)")

    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        fit3d_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
    )
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument(
        "--pretrained",
        default="checkpoints/motionbert/pretrain/MB_release.bin",
        help="MotionBERT pretrained weights to initialize from",
    )
    ap.add_argument("--lora", action="store_true", default=True)
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="Run a tiny smoke test (64 samples per loader)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:    cfg.training.epochs = args.epochs
    if args.lr is not None:        cfg.training.optimizer.lr = args.lr
    if args.batch_size is not None: cfg.data.batch_size = args.batch_size
    if args.seq_len is not None:   cfg.data.seq_len = args.seq_len; cfg.model.seq_len = args.seq_len
    cfg.model.pretrained_path = args.pretrained
    cfg.model.lora.enabled = args.lora
    if args.lora:
        cfg.model.lora.rank = args.lora_rank

    # Path A: turn off reprojection until the loss is wired correctly.
    cfg.training.loss_weights.reproj = 0.0

    # Disable W&B for batch jobs that don't have wandb auth set up
    cfg.wandb.mode = "disabled"

    # Default skeleton + name fields if missing
    cfg.data.output_skeleton = cfg.data.get("output_skeleton", "h36m_17")
    cfg.model.name = cfg.model.get("name", "dstformer")

    print("=" * 60)
    print(f"Hybrid training (path A: lambda_reproj=0)")
    print(f"  Model:          {cfg.model.name}")
    print(f"  Pretrained:     {cfg.model.pretrained_path}")
    print(f"  LoRA enabled:   {cfg.model.lora.enabled} (rank={cfg.model.lora.get('rank', 8)})")
    print(f"  seq_len/bs:     {cfg.data.seq_len} / {cfg.data.batch_size}")
    print(f"  epochs/lr:      {cfg.training.epochs} / {cfg.training.optimizer.lr}")
    print(f"  loss weights:   l3d={cfg.training.loss_weights.l3d} "
          f"reproj={cfg.training.loss_weights.reproj} "
          f"biomech={cfg.training.loss_weights.biomech}")
    print("=" * 60)

    train_loader, val_loader = build_loaders(cfg, smoke=args.smoke)

    model = create_model(cfg)
    print(f"Trainable params: {model.count_parameters():,} of {model.count_parameters(trainable_only=False):,}")

    trainer = Trainer(cfg=cfg, model=model, train_loader=train_loader, val_loader=val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
