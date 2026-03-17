"""
Skeleton definitions and conversion utilities.

Handles conversion between different skeleton formats (COCO, Human3.6M, etc.)
with proper joint mapping, interpolation, and bone connectivity.
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class SkeletonConfig:
    """Configuration for a skeleton format."""
    name: str
    num_joints: int
    joint_names: list[str]
    bones: list[tuple[int, int]]
    root_idx: int
    # Bilateral pairs: list of ((left_start, left_end), (right_start, right_end))
    bilateral_pairs: list[tuple[tuple[int, int], tuple[int, int]]]


SKELETON_CONFIGS = {
    "coco_17": SkeletonConfig(
        name="coco_17",
        num_joints=17,
        joint_names=[
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ],
        bones=[
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        ],
        root_idx=0,  # nose (no pelvis in COCO)
        bilateral_pairs=[
            ((11, 13), (12, 14)),  # Hip to knee
            ((13, 15), (14, 16)),  # Knee to ankle
            ((5, 7), (6, 8)),      # Shoulder to elbow
            ((7, 9), (8, 10)),     # Elbow to wrist
        ],
    ),
    "h36m_17": SkeletonConfig(
        name="h36m_17",
        num_joints=17,
        joint_names=[
            "pelvis", "right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle", "spine", "thorax",
            "neck_nose", "head", "left_shoulder", "left_elbow", "left_wrist",
            "right_shoulder", "right_elbow", "right_wrist",
        ],
        bones=[
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
            (8, 11), (11, 12), (12, 13),  # Left arm
            (8, 14), (14, 15), (15, 16),  # Right arm
        ],
        root_idx=0,  # pelvis
        bilateral_pairs=[
            ((4, 5), (1, 2)),      # Hip to knee
            ((5, 6), (2, 3)),      # Knee to ankle
            ((11, 12), (14, 15)),  # Shoulder to elbow
            ((12, 13), (15, 16)),  # Elbow to wrist
        ],
    ),
}


class SkeletonConverter:
    """
    Convert between skeleton formats with proper joint mapping and interpolation.
    """

    # COCO index -> H36M index (direct mappings)
    _COCO_TO_H36M_DIRECT = {
        12: 1,   # right_hip
        14: 2,   # right_knee
        16: 3,   # right_ankle
        11: 4,   # left_hip
        13: 5,   # left_knee
        15: 6,   # left_ankle
        0: 9,    # nose -> neck_nose
        5: 11,   # left_shoulder
        7: 12,   # left_elbow
        9: 13,   # left_wrist
        6: 14,   # right_shoulder
        8: 15,   # right_elbow
        10: 16,  # right_wrist
    }

    # H36M joints that need interpolation: h36m_idx -> (coco_indices, weights)
    _COCO_TO_H36M_INTERP = {
        0: ([11, 12], [0.5, 0.5]),          # pelvis from hips
        7: ([11, 12, 5, 6], [0.25] * 4),    # spine from torso center
        8: ([5, 6], [0.5, 0.5]),            # thorax from shoulders
        10: ([0, 1, 2], [0.34, 0.33, 0.33]), # head from nose + eyes
    }

    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        self.source_cfg = SKELETON_CONFIGS[source]
        self.target_cfg = SKELETON_CONFIGS[target]

    def convert(
        self,
        keypoints: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert keypoints from source to target skeleton format.

        Args:
            keypoints: Shape (..., num_joints_source, D) where D is 2 or 3
            confidence: Shape (..., num_joints_source), optional

        Returns:
            converted: Shape (..., num_joints_target, D)
            conf_out: Shape (..., num_joints_target)
        """
        if self.source == "coco_17" and self.target == "h36m_17":
            return self._coco_to_h36m(keypoints, confidence)
        elif self.source == self.target:
            if confidence is None:
                confidence = torch.ones(keypoints.shape[:-1], device=keypoints.device)
            return keypoints, confidence
        else:
            raise NotImplementedError(f"Conversion {self.source} -> {self.target}")

    def _coco_to_h36m(
        self,
        keypoints: torch.Tensor,
        confidence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert COCO 17 to H36M 17."""
        shape = keypoints.shape
        D = shape[-1]
        device = keypoints.device
        dtype = keypoints.dtype

        # Flatten batch dims
        kp_flat = keypoints.reshape(-1, 17, D)
        N = kp_flat.shape[0]

        if confidence is None:
            conf_flat = torch.ones(N, 17, device=device, dtype=dtype)
        else:
            conf_flat = confidence.reshape(-1, 17)

        # Output
        out_kp = torch.zeros(N, 17, D, device=device, dtype=dtype)
        out_conf = torch.zeros(N, 17, device=device, dtype=dtype)

        # Direct mappings
        for coco_idx, h36m_idx in self._COCO_TO_H36M_DIRECT.items():
            out_kp[:, h36m_idx] = kp_flat[:, coco_idx]
            out_conf[:, h36m_idx] = conf_flat[:, coco_idx]

        # Interpolations
        for h36m_idx, (coco_indices, weights) in self._COCO_TO_H36M_INTERP.items():
            weights_t = torch.tensor(weights, device=device, dtype=dtype)
            for coco_idx, w in zip(coco_indices, weights_t):
                out_kp[:, h36m_idx] += w * kp_flat[:, coco_idx]
            # Confidence = min of constituent joints
            out_conf[:, h36m_idx] = torch.min(
                conf_flat[:, coco_indices], dim=-1
            ).values

        # Restore shape
        out_shape = shape[:-2] + (17, D)
        conf_shape = shape[:-2] + (17,)

        return out_kp.reshape(out_shape), out_conf.reshape(conf_shape)


def compute_bone_lengths(
    joints: torch.Tensor,
    skeleton: str | SkeletonConfig,
) -> torch.Tensor:
    """
    Compute bone lengths for a skeleton.

    Args:
        joints: Shape (..., J, 3)
        skeleton: Skeleton name or config

    Returns:
        lengths: Shape (..., num_bones)
    """
    if isinstance(skeleton, str):
        skeleton = SKELETON_CONFIGS[skeleton]

    lengths = []
    for start, end in skeleton.bones:
        bone_vec = joints[..., end, :] - joints[..., start, :]
        length = torch.norm(bone_vec, dim=-1)
        lengths.append(length)

    return torch.stack(lengths, dim=-1)
