"""
Wrappers for pretrained 3D pose estimation models.

Supports:
- MotionBERT: https://github.com/Walter0807/MotionBERT
- APTPose: https://github.com/wenwen12321/APTPose

Both models are adapted to work with our unified interface.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any
import subprocess
import sys

from .base import PoseEstimatorBase


def _clone_repo_if_needed(repo_url: str, target_dir: Path) -> None:
    """Clone a git repo if it doesn't exist."""
    if not target_dir.exists():
        print(f"Cloning {repo_url} to {target_dir}...")
        subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            check=True,
        )


def _add_to_path(path: Path) -> None:
    """Add a directory to Python path."""
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class MotionBERTWrapper(PoseEstimatorBase):
    """
    Wrapper for MotionBERT pretrained model.

    MotionBERT uses a Dual-stream Spatio-Temporal transformer (DSTformer)
    pretrained on AMASS via Masked Pose Modeling.

    Reference: https://github.com/Walter0807/MotionBERT
    """

    REPO_URL = "https://github.com/Walter0807/MotionBERT.git"
    CHECKPOINT_URL = "https://github.com/Walter0807/MotionBERT/releases/download/v1.0.0/FT_MB_lite_MB_ft_h36m_global_lite.bin"

    def __init__(
        self,
        cfg: Any,
        checkpoint_path: str | Path | None = None,
        repo_path: str | Path | None = None,
    ):
        super().__init__(cfg)

        self.repo_path = Path(repo_path or "external/MotionBERT")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Clone repo if needed
        _clone_repo_if_needed(self.REPO_URL, self.repo_path)
        _add_to_path(self.repo_path)

        # Import MotionBERT modules
        from lib.model.DSTformer import DSTformer

        # Model config (from MotionBERT defaults)
        self.model = DSTformer(
            dim_in=3,  # MotionBERT expects (x, y, confidence) or (x, y, 1)
            dim_out=3,
            dim_feat=256,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            num_joints=self.num_joints,
            maxlen=self.seq_len,
        )

        # Load weights
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint(self.checkpoint_path)

    def _load_checkpoint(self, path: Path) -> None:
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_pos" in checkpoint:
            state_dict = checkpoint["model_pos"]
        else:
            state_dict = checkpoint

        # Handle possible key mismatches
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded MotionBERT checkpoint from {path}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: 2D keypoints (B, T, J, 2)
            mask: Visibility mask (B, T, J)

        Returns:
            3D poses (B, T, J, 3)
        """
        B, T, J, _ = x.shape

        # MotionBERT expects (B, T, J, 3) with confidence as 3rd dim
        if x.shape[-1] == 2:
            if mask is not None:
                conf = mask.unsqueeze(-1).float()
            else:
                conf = torch.ones(B, T, J, 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, conf], dim=-1)

        # Forward through DSTformer
        output = self.model(x)

        return output


class APTPoseWrapper(PoseEstimatorBase):
    """
    Wrapper for APTPose pretrained model.

    APTPose uses anatomy-aware pretraining for 3D pose estimation,
    incorporating skeletal structure priors.

    Reference: https://github.com/wenwen12321/APTPose
    """

    REPO_URL = "https://github.com/wenwen12321/APTPose.git"

    def __init__(
        self,
        cfg: Any,
        checkpoint_path: str | Path | None = None,
        repo_path: str | Path | None = None,
    ):
        super().__init__(cfg)

        self.repo_path = Path(repo_path or "external/APTPose")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Clone repo if needed
        _clone_repo_if_needed(self.REPO_URL, self.repo_path)
        _add_to_path(self.repo_path)

        # APTPose model initialization
        # Note: Exact imports depend on APTPose repo structure
        # This is a placeholder - adjust based on actual repo
        try:
            from model.aptpose import APTPose as APTPoseModel

            self.model = APTPoseModel(
                num_joints=self.num_joints,
                in_chans=2,
                num_frame=self.seq_len,
                # Additional config as needed
            )

            if self.checkpoint_path and self.checkpoint_path.exists():
                self._load_checkpoint(self.checkpoint_path)

        except ImportError:
            print("Warning: APTPose not yet configured. Check repo structure.")
            self.model = None

    def _load_checkpoint(self, path: Path) -> None:
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded APTPose checkpoint from {path}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: 2D keypoints (B, T, J, 2)
            mask: Visibility mask (B, T, J)

        Returns:
            3D poses (B, T, J, 3)
        """
        if self.model is None:
            raise RuntimeError("APTPose model not initialized. Check installation.")

        output = self.model(x)
        return output


def load_pretrained_model(
    model_name: str,
    cfg: Any,
    checkpoint_path: str | Path | None = None,
) -> PoseEstimatorBase:
    """
    Factory function to load a pretrained model.

    Args:
        model_name: 'motionbert' or 'aptpose'
        cfg: Model configuration
        checkpoint_path: Path to pretrained weights

    Returns:
        Initialized model wrapper
    """
    model_name = model_name.lower()

    if model_name == "motionbert":
        return MotionBERTWrapper(cfg, checkpoint_path)
    elif model_name == "aptpose":
        return APTPoseWrapper(cfg, checkpoint_path)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'motionbert' or 'aptpose'")
