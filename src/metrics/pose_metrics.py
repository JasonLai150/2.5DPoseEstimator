"""
Evaluation metrics for 3D pose estimation.

- MPJPE: Mean Per-Joint Position Error (mm)
- P-MPJPE: Procrustes-aligned MPJPE (removes scale/rotation/translation)
- BLI: Bilateral Length Inconsistency (skeletal realism)
"""

import torch
import numpy as np
from typing import Literal

from ..data.skeleton import SKELETON_CONFIGS


def compute_mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    root_idx: int = 0,
) -> torch.Tensor:
    """
    Compute Mean Per-Joint Position Error.

    Args:
        pred: Predicted poses, shape (B, T, J, 3) or (N, J, 3)
        target: Ground truth poses, same shape as pred
        root_idx: Root joint index for alignment

    Returns:
        MPJPE in the same units as input (typically mm)
    """
    # Root-relative
    pred = pred - pred[..., root_idx : root_idx + 1, :]
    target = target - target[..., root_idx : root_idx + 1, :]

    # Per-joint error
    error = torch.norm(pred - target, dim=-1)

    return error.mean()


def _procrustes_alignment(
    pred: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Procrustes alignment: find optimal scale, rotation, translation.

    Args:
        pred: Shape (J, 3)
        target: Shape (J, 3)

    Returns:
        Aligned prediction
    """
    # Center
    mu_pred = pred.mean(axis=0)
    mu_target = target.mean(axis=0)
    pred_centered = pred - mu_pred
    target_centered = target - mu_target

    # Scale
    scale_pred = np.sqrt((pred_centered ** 2).sum())
    scale_target = np.sqrt((target_centered ** 2).sum())
    pred_scaled = pred_centered / scale_pred
    target_scaled = target_centered / scale_target

    # Rotation via SVD
    H = pred_scaled.T @ target_scaled
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply transformation
    aligned = scale_target * (pred_scaled @ R) + mu_target

    return aligned


def compute_p_mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Procrustes-aligned MPJPE (P-MPJPE).

    Aligns each prediction to ground truth using Procrustes analysis
    before computing error. Removes scale, rotation, and translation.

    Args:
        pred: Predicted poses, shape (B, T, J, 3) or (N, J, 3)
        target: Ground truth poses, same shape as pred

    Returns:
        P-MPJPE in same units as target
    """
    # Flatten to (N, J, 3)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    original_shape = pred_np.shape
    pred_flat = pred_np.reshape(-1, original_shape[-2], original_shape[-1])
    target_flat = target_np.reshape(-1, original_shape[-2], original_shape[-1])

    errors = []
    for p, t in zip(pred_flat, target_flat):
        aligned = _procrustes_alignment(p, t)
        error = np.linalg.norm(aligned - t, axis=-1).mean()
        errors.append(error)

    return torch.tensor(np.mean(errors), device=pred.device)


def compute_bli(
    poses_3d: torch.Tensor,
    skeleton: str = "h36m_17",
) -> torch.Tensor:
    """
    Compute Bilateral Length Inconsistency (BLI).

    Measures variance in bone lengths between left and right sides.
    Lower BLI = more symmetric/realistic skeleton.

    Args:
        poses_3d: 3D poses, shape (B, T, J, 3)
        skeleton: Skeleton format name

    Returns:
        BLI score (variance in mm)
    """
    skeleton_cfg = SKELETON_CONFIGS[skeleton]
    bilateral_pairs = skeleton_cfg.bilateral_pairs

    inconsistencies = []

    for (left_start, left_end), (right_start, right_end) in bilateral_pairs:
        left_bone = poses_3d[..., left_end, :] - poses_3d[..., left_start, :]
        right_bone = poses_3d[..., right_end, :] - poses_3d[..., right_start, :]

        left_length = torch.norm(left_bone, dim=-1)
        right_length = torch.norm(right_bone, dim=-1)

        # Relative difference
        diff = torch.abs(left_length - right_length)
        avg_length = (left_length + right_length) / 2
        relative_diff = diff / avg_length.clamp(min=1e-6)

        inconsistencies.append(relative_diff)

    # Stack and compute variance across all pairs
    all_inconsistencies = torch.stack(inconsistencies, dim=-1)

    return all_inconsistencies.var()


class PoseMetrics:
    """
    Accumulator for pose estimation metrics over batches.
    """

    def __init__(self, skeleton: str = "h36m_17"):
        self.skeleton = skeleton
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self._mpjpe_sum = 0.0
        self._p_mpjpe_sum = 0.0
        self._bli_sum = 0.0
        self._count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """
        Update metrics with a batch.

        Args:
            pred: Predicted poses
            target: Ground truth poses

        Returns:
            Dict of metric values for this batch
        """
        batch_size = pred.shape[0]

        mpjpe = compute_mpjpe(pred, target)
        p_mpjpe = compute_p_mpjpe(pred, target)
        bli = compute_bli(pred, self.skeleton)

        self._mpjpe_sum += mpjpe.item() * batch_size
        self._p_mpjpe_sum += p_mpjpe.item() * batch_size
        self._bli_sum += bli.item() * batch_size
        self._count += batch_size

        return {
            "mpjpe": mpjpe.item(),
            "p_mpjpe": p_mpjpe.item(),
            "bli": bli.item(),
        }

    def compute(self) -> dict[str, float]:
        """Compute aggregated metrics."""
        if self._count == 0:
            return {"mpjpe": 0.0, "p_mpjpe": 0.0, "bli": 0.0}

        return {
            "mpjpe": self._mpjpe_sum / self._count,
            "p_mpjpe": self._p_mpjpe_sum / self._count,
            "bli": self._bli_sum / self._count,
        }
