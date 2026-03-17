"""
Camera projection utilities for 3D-to-2D reprojection.

Implements perspective camera model for the weakly-supervised reprojection loss.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y


class PerspectiveCamera(nn.Module):
    """
    Differentiable perspective camera for 3D-to-2D projection.

    Projects 3D points in camera coordinates to 2D image coordinates
    using the pinhole camera model.
    """

    def __init__(
        self,
        fx: float = 1000.0,
        fy: float = 1000.0,
        cx: float = 512.0,
        cy: float = 512.0,
    ):
        super().__init__()
        # Register as buffers (not trainable, but move with .to(device))
        self.register_buffer("fx", torch.tensor(fx))
        self.register_buffer("fy", torch.tensor(fy))
        self.register_buffer("cx", torch.tensor(cx))
        self.register_buffer("cy", torch.tensor(cy))

    def forward(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points to 2D.

        Args:
            points_3d: Shape (..., 3) with (X, Y, Z) in camera frame
                       Z is depth (positive = in front of camera)

        Returns:
            points_2d: Shape (..., 2) with (u, v) pixel coordinates
        """
        X = points_3d[..., 0]
        Y = points_3d[..., 1]
        Z = points_3d[..., 2]

        # Avoid division by zero
        Z = Z.clamp(min=1e-8)

        u = self.fx * (X / Z) + self.cx
        v = self.fy * (Y / Z) + self.cy

        return torch.stack([u, v], dim=-1)

    def unproject(
        self,
        points_2d: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unproject 2D points to 3D given depth.

        Args:
            points_2d: Shape (..., 2) with (u, v)
            depth: Shape (...) with Z values

        Returns:
            points_3d: Shape (..., 3)
        """
        u = points_2d[..., 0]
        v = points_2d[..., 1]

        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth

        return torch.stack([X, Y, Z], dim=-1)


def project_to_2d(
    points_3d: torch.Tensor,
    fx: float = 1000.0,
    fy: float = 1000.0,
    cx: float = 512.0,
    cy: float = 512.0,
) -> torch.Tensor:
    """
    Functional interface for perspective projection.

    Args:
        points_3d: Shape (..., 3)
        fx, fy: Focal lengths
        cx, cy: Principal point

    Returns:
        points_2d: Shape (..., 2)
    """
    X = points_3d[..., 0]
    Y = points_3d[..., 1]
    Z = points_3d[..., 2].clamp(min=1e-8)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    return torch.stack([u, v], dim=-1)


def normalize_screen_coordinates(
    coords: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Normalize pixel coordinates to [-1, 1] range.

    Useful for resolution-independent reprojection loss.

    Args:
        coords: Shape (..., 2) in pixel coordinates
        width: Image width
        height: Image height

    Returns:
        normalized: Shape (..., 2) in [-1, 1]
    """
    normalized = coords.clone()
    normalized[..., 0] = 2 * coords[..., 0] / width - 1
    normalized[..., 1] = 2 * coords[..., 1] / height - 1
    return normalized


def denormalize_screen_coordinates(
    coords: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Inverse of normalize_screen_coordinates."""
    denorm = coords.clone()
    denorm[..., 0] = (coords[..., 0] + 1) * width / 2
    denorm[..., 1] = (coords[..., 1] + 1) * height / 2
    return denorm
