"""Base class for pose estimation models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any


class PoseEstimatorBase(nn.Module, ABC):
    """
    Abstract base class for 2D-to-3D pose lifting models.

    All pose estimation models should inherit from this class and implement
    the required methods. This ensures consistent interface for training,
    evaluation, and inference.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.num_joints = cfg.model.num_joints
        self.input_dim = cfg.model.input_dim
        self.output_dim = cfg.model.output_dim
        self.seq_len = cfg.model.seq_len

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass: lift 2D poses to 3D.

        Args:
            x: Input 2D keypoints of shape (B, T, J, 2)
               B = batch size, T = sequence length, J = num joints
            mask: Optional visibility mask of shape (B, T, J)
                  1 = visible, 0 = occluded

        Returns:
            3D pose predictions of shape (B, T, J, 3)
            Coordinates are root-relative in camera frame
        """
        pass

    def get_root_relative(self, poses_3d: torch.Tensor, root_idx: int = 0) -> torch.Tensor:
        """Convert absolute 3D poses to root-relative coordinates."""
        root = poses_3d[..., root_idx : root_idx + 1, :]
        return poses_3d - root

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self) -> None:
        """Freeze all parameters except LoRA adapters (if present)."""
        for name, param in self.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
