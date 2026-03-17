"""
Training loop for 2.5D pose estimation.

Handles:
- Mixed supervised (H36M) and weakly-supervised (gym) training
- Composite loss optimization
- Checkpointing and logging to W&B
- Validation and evaluation
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Any
import wandb

from ..models.base import PoseEstimatorBase
from ..losses import PoseLoss
from ..metrics import PoseMetrics
from ..config import config_to_dict


class Trainer:
    """Trainer for 2.5D pose estimation models."""

    def __init__(
        self,
        cfg: Any,
        model: PoseEstimatorBase,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss
        self.criterion = PoseLoss(cfg)
        self.criterion.camera.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Metrics
        self.metrics = PoseMetrics(skeleton=cfg.data.output_skeleton)

        # Checkpointing
        self.checkpoint_dir = Path(cfg.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("inf")

        # State
        self.current_epoch = 0
        self.global_step = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_cfg = self.cfg.training.optimizer
        return AdamW(
            self.model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        sched_cfg = self.cfg.training.scheduler
        total_epochs = self.cfg.training.epochs

        # Warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=sched_cfg.warmup_epochs,
        )

        # Main scheduler
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - sched_cfg.warmup_epochs,
            eta_min=sched_cfg.min_lr,
        )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[sched_cfg.warmup_epochs],
        )

    def train(self) -> None:
        """Run full training loop."""
        # Initialize W&B
        wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            config=config_to_dict(self.cfg) if hasattr(self.cfg, 'items') else self.cfg,
            mode=self.cfg.wandb.mode,
        )
        wandb.watch(self.model, log_freq=100)

        for epoch in range(self.current_epoch, self.cfg.training.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self._train_epoch()
            wandb.log({"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}})

            # Validation
            if self.val_loader and (epoch + 1) % self.cfg.training.val_every == 0:
                val_metrics = self._validate()
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

                # Checkpointing
                if self.cfg.training.checkpoint.save_best:
                    monitor = self.cfg.training.checkpoint.monitor.split("/")[-1]
                    current = val_metrics.get(monitor, float("inf"))
                    if current < self.best_metric:
                        self.best_metric = current
                        self.save_checkpoint("best.pt")

            # Periodic checkpoint
            if (epoch + 1) % self.cfg.training.checkpoint.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            self.scheduler.step()

        # Final test evaluation
        if self.test_loader:
            test_metrics = self._evaluate(self.test_loader, "test")
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

        wandb.finish()

    def _train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward
            self.optimizer.zero_grad()
            pred_3d = self.model(batch["poses_2d"], batch.get("mask"))

            # Loss
            losses = self.criterion(pred_3d, batch)

            # Backward
            losses["loss"].backward()

            # Gradient clipping
            if self.cfg.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.gradient_clip,
                )

            self.optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix(loss=losses["loss"].item())

        # Average
        return {k: v / num_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation."""
        return self._evaluate(self.val_loader, "val")

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        split: str,
    ) -> dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        self.metrics.reset()

        for batch in tqdm(loader, desc=f"Evaluating {split}"):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            pred_3d = self.model(batch["poses_2d"], batch.get("mask"))

            if batch.get("has_3d", False) and "poses_3d" in batch:
                self.metrics.update(pred_3d, batch["poses_3d"])

        return self.metrics.compute()

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": config_to_dict(self.cfg) if hasattr(self.cfg, 'items') else self.cfg,
        }, path)

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent N."""
        keep_last = self.cfg.training.checkpoint.keep_last
        checkpoints = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        for ckpt in checkpoints[:-keep_last]:
            ckpt.unlink()
