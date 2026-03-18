"""
VideoPose3D: Temporal Convolutional Network for 3D Pose Estimation.

A simple and effective baseline using dilated temporal convolutions.
Reference: https://github.com/facebookresearch/VideoPose3D

Architecture:
- 1D temporal convolutions with dilation
- Residual connections
- Much simpler than transformer-based approaches
"""

import torch
import torch.nn as nn
from typing import Any, Optional

from .base import PoseEstimatorBase
from .lora import apply_lora_to_model, freeze_non_lora, count_lora_parameters


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with residual connection.

    Uses dilated convolutions to capture long-range temporal dependencies
    without excessive parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 1  # 1x1 conv
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, C, T) - batch, channels, time
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual


class VideoPose3D(PoseEstimatorBase):
    """
    VideoPose3D: 3D pose estimation using temporal convolutions.

    Simple, effective baseline that's easy to train and debug.
    Uses dilated convolutions to capture temporal context.

    Args:
        cfg: Configuration with model parameters
        pretrained_path: Path to pretrained weights
    """

    def __init__(
        self,
        cfg: Any,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__(cfg)

        # Architecture params
        in_features = self.num_joints * self.input_dim  # J * 2
        out_features = self.num_joints * self.output_dim  # J * 3

        hidden_dim = cfg.model.get('hidden_dim', 1024)
        num_blocks = cfg.model.get('num_blocks', 4)
        kernel_size = cfg.model.get('kernel_size', 3)
        dropout = cfg.model.get('drop_rate', 0.25)

        self.hidden_dim = hidden_dim

        # Input projection: (B, T, J*2) -> (B, hidden, T)
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Temporal blocks with increasing dilation
        self.temporal_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            self.temporal_blocks.append(
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout)
            )

        # Output projection: (B, hidden, T) -> (B, T, J*3)
        self.output_proj = nn.Linear(hidden_dim, out_features)

        # Initialize
        self._init_weights()

        # Load pretrained if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)

        # Apply LoRA if configured
        if cfg.model.get('lora', {}).get('enabled', False):
            self._apply_lora(cfg.model.lora)

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _apply_lora(self, lora_cfg):
        """Apply LoRA to linear layers."""
        rank = lora_cfg.get('rank', 8)
        alpha = lora_cfg.get('alpha', 16)
        dropout = lora_cfg.get('dropout', 0.0)
        target_modules = lora_cfg.get('target_modules', ['input_proj', 'output_proj'])

        apply_lora_to_model(self, target_modules, rank, alpha, dropout)
        freeze_non_lora(self)

        total, lora = count_lora_parameters(self)
        print(f"LoRA enabled: {lora:,} trainable params ({100*lora/total:.2f}% of {total:,})")

    def load_pretrained(self, path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location='cpu')

        if 'model_pos' in checkpoint:
            state_dict = checkpoint['model_pos']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Try to load, handling key mismatches
        try:
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: 2D keypoints (B, T, J, 2)
            mask: Visibility mask (B, T, J) - used for masking if provided

        Returns:
            3D poses (B, T, J, 3)
        """
        B, T, J, C = x.shape

        # Flatten joints: (B, T, J, 2) -> (B, T, J*2)
        x = x.reshape(B, T, -1)

        # Apply mask by zeroing out if provided
        if mask is not None:
            mask_flat = mask.reshape(B, T, -1).repeat(1, 1, C)
            x = x * mask_flat

        # Input projection: (B, T, J*2) -> (B, T, hidden)
        x = self.input_proj(x)

        # Temporal convolutions expect (B, C, T)
        x = x.permute(0, 2, 1)

        # Process through temporal blocks
        for block in self.temporal_blocks:
            x = block(x)

        # Back to (B, T, hidden)
        x = x.permute(0, 2, 1)

        # Output projection: (B, T, hidden) -> (B, T, J*3)
        x = self.output_proj(x)

        # Reshape to (B, T, J, 3)
        x = x.reshape(B, T, J, self.output_dim)

        return x
