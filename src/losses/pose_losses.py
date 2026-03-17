"""
Loss functions for 2.5D pose estimation.

Composite loss: L_total = λ1*L_3D + λ2*L_reproj + λ3*L_biomech

- L_3D: MPJPE between predictions and 3D ground truth
- L_reproj: Reprojection error for weakly-supervised samples
- L_biomech: Biomechanical constraints (symmetry + hinge limits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from ..utils.camera import PerspectiveCamera
from ..data.skeleton import SKELETON_CONFIGS, compute_bone_lengths


def mpjpe_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    root_relative: bool = True,
    root_idx: int = 0,
) -> torch.Tensor:
    """
    Mean Per-Joint Position Error (MPJPE).

    Args:
        pred: Predicted 3D poses, shape (B, T, J, 3)
        target: Ground truth 3D poses, shape (B, T, J, 3)
        mask: Optional joint visibility mask, shape (B, T, J)
        root_relative: If True, compute error after root alignment
        root_idx: Index of root joint (pelvis)

    Returns:
        Scalar loss value
    """
    if root_relative:
        pred = pred - pred[..., root_idx : root_idx + 1, :]
        target = target - target[..., root_idx : root_idx + 1, :]

    # Per-joint Euclidean distance
    error = torch.norm(pred - target, dim=-1)  # (B, T, J)

    if mask is not None:
        error = error * mask
        return error.sum() / mask.sum().clamp(min=1)

    return error.mean()


def reprojection_loss(
    pred_3d: torch.Tensor,
    target_2d: torch.Tensor,
    camera: PerspectiveCamera,
    mask: torch.Tensor | None = None,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    Weakly-supervised reprojection loss.

    Projects 3D predictions to 2D and compares with 2D pseudo-labels.

    Args:
        pred_3d: Predicted 3D poses, shape (B, T, J, 3)
        target_2d: 2D pseudo-labels, shape (B, T, J, 2)
        camera: Camera model for projection
        mask: Optional visibility mask, shape (B, T, J)
        loss_type: 'l1', 'l2', or 'smooth_l1'

    Returns:
        Scalar loss value
    """
    # Project 3D -> 2D
    pred_2d = camera(pred_3d)  # (B, T, J, 2)

    # Compute error
    if loss_type == "l1":
        error = torch.abs(pred_2d - target_2d).sum(dim=-1)
    elif loss_type == "l2":
        error = torch.norm(pred_2d - target_2d, dim=-1)
    elif loss_type == "smooth_l1":
        error = F.smooth_l1_loss(pred_2d, target_2d, reduction="none").sum(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if mask is not None:
        error = error * mask
        return error.sum() / mask.sum().clamp(min=1)

    return error.mean()


def bilateral_symmetry_loss(
    poses_3d: torch.Tensor,
    skeleton: str = "h36m_17",
) -> torch.Tensor:
    """
    Enforce equal bone lengths between left and right appendages.

    Penalizes asymmetric skeletons which are anatomically implausible.

    Args:
        poses_3d: 3D poses, shape (B, T, J, 3)
        skeleton: Skeleton format name

    Returns:
        Scalar loss value
    """
    skeleton_cfg = SKELETON_CONFIGS[skeleton]
    bilateral_pairs = skeleton_cfg.bilateral_pairs

    loss = torch.tensor(0.0, device=poses_3d.device, dtype=poses_3d.dtype)

    for (left_start, left_end), (right_start, right_end) in bilateral_pairs:
        left_bone = poses_3d[..., left_end, :] - poses_3d[..., left_start, :]
        right_bone = poses_3d[..., right_end, :] - poses_3d[..., right_start, :]

        left_length = torch.norm(left_bone, dim=-1)
        right_length = torch.norm(right_bone, dim=-1)

        # Penalize difference in bone lengths
        loss = loss + torch.abs(left_length - right_length).mean()

    return loss / len(bilateral_pairs)


def anatomical_hinge_loss(
    poses_3d: torch.Tensor,
    skeleton: str = "h36m_17",
    angle_limits: dict | None = None,
) -> torch.Tensor:
    """
    Penalize physically impossible joint rotations.

    Enforces angle limits on hinge joints (knees, elbows) to prevent
    hyperextension.

    Args:
        poses_3d: 3D poses, shape (B, T, J, 3)
        skeleton: Skeleton format name
        angle_limits: Dict mapping joint name to (min_deg, max_deg)

    Returns:
        Scalar loss value
    """
    if angle_limits is None:
        angle_limits = {
            "knee": (0, 160),
            "elbow": (0, 160),
        }

    skeleton_cfg = SKELETON_CONFIGS[skeleton]

    # Define joint triplets for angle computation
    # (parent, joint, child) - angle at middle joint
    if skeleton == "h36m_17":
        hinge_joints = {
            "left_knee": (4, 5, 6),     # left_hip -> left_knee -> left_ankle
            "right_knee": (1, 2, 3),    # right_hip -> right_knee -> right_ankle
            "left_elbow": (11, 12, 13), # left_shoulder -> left_elbow -> left_wrist
            "right_elbow": (14, 15, 16), # right_shoulder -> right_elbow -> right_wrist
        }
    else:
        return torch.tensor(0.0, device=poses_3d.device)

    loss = torch.tensor(0.0, device=poses_3d.device, dtype=poses_3d.dtype)
    count = 0

    for joint_name, (parent, joint, child) in hinge_joints.items():
        # Get joint type (knee or elbow)
        joint_type = "knee" if "knee" in joint_name else "elbow"
        if joint_type not in angle_limits:
            continue

        min_angle, max_angle = angle_limits[joint_type]

        # Compute vectors
        v1 = poses_3d[..., parent, :] - poses_3d[..., joint, :]
        v2 = poses_3d[..., child, :] - poses_3d[..., joint, :]

        # Compute angle (in degrees)
        v1_norm = F.normalize(v1, dim=-1)
        v2_norm = F.normalize(v2, dim=-1)
        cos_angle = (v1_norm * v2_norm).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_angle) * 180 / torch.pi

        # Penalize angles outside valid range
        min_violation = F.relu(min_angle - angle)
        max_violation = F.relu(angle - max_angle)
        loss = loss + (min_violation + max_violation).mean()
        count += 1

    return loss / max(count, 1)


class PoseLoss(nn.Module):
    """
    Composite loss for 2.5D pose estimation.

    Combines:
    - L_3D: Supervised 3D loss (MPJPE)
    - L_reproj: Weakly-supervised reprojection loss
    - L_biomech: Biomechanical constraints
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg

        # Loss weights
        self.lambda_3d = cfg.training.loss_weights.l3d
        self.lambda_reproj = cfg.training.loss_weights.reproj
        self.lambda_biomech = cfg.training.loss_weights.biomech

        # Biomechanical weights
        self.symmetry_weight = cfg.training.biomech.symmetry_weight
        self.hinge_weight = cfg.training.biomech.hinge_weight
        self.angle_limits = dict(cfg.training.biomech.angle_limits)

        # Camera for reprojection
        self.camera = PerspectiveCamera()

        # Skeleton format
        self.skeleton = cfg.data.output_skeleton

    def forward(
        self,
        pred_3d: torch.Tensor,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Compute composite loss.

        Args:
            pred_3d: Model predictions, shape (B, T, J, 3)
            batch: Dict with keys:
                - poses_3d: Optional GT 3D poses
                - poses_2d: 2D keypoints
                - has_3d: Boolean indicating 3D availability
                - mask: Optional visibility mask

        Returns:
            Dict with 'loss' (total) and individual components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=pred_3d.device)

        mask = batch.get("mask", None)

        # 3D supervised loss
        if batch.get("has_3d", False) and "poses_3d" in batch:
            l3d = mpjpe_loss(pred_3d, batch["poses_3d"], mask=mask)
            losses["l3d"] = l3d
            total_loss = total_loss + self.lambda_3d * l3d

        # Reprojection loss (for samples with 2D pseudo-labels)
        if "poses_2d" in batch:
            l_reproj = reprojection_loss(
                pred_3d, batch["poses_2d"], self.camera, mask=mask
            )
            losses["reproj"] = l_reproj
            total_loss = total_loss + self.lambda_reproj * l_reproj

        # Biomechanical constraints (always applied)
        l_symmetry = bilateral_symmetry_loss(pred_3d, self.skeleton)
        l_hinge = anatomical_hinge_loss(
            pred_3d, self.skeleton, self.angle_limits
        )
        l_biomech = self.symmetry_weight * l_symmetry + self.hinge_weight * l_hinge

        losses["symmetry"] = l_symmetry
        losses["hinge"] = l_hinge
        losses["biomech"] = l_biomech
        total_loss = total_loss + self.lambda_biomech * l_biomech

        losses["loss"] = total_loss

        return losses
