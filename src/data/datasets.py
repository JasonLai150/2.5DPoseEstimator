"""
Dataset classes for pose estimation training and evaluation.

Supports:
- Human3.6M: 3D supervised data
- Gym videos: 2D weakly-supervised data with pseudo-labels
- Fit3D: Evaluation data with 3D fitness ground truth
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

from .skeleton import SkeletonConverter


class PoseDataset(Dataset, ABC):
    """Abstract base class for pose datasets."""

    def __init__(
        self,
        cfg: Any,
        split: str = "train",
        seq_len: int = 243,
        stride: int = 1,
    ):
        self.cfg = cfg
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.samples: list[dict] = []

    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset annotations and create sample list."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass


class H36MDataset(PoseDataset):
    """
    Human3.6M dataset for 3D supervised training.

    Provides paired 2D-3D data with camera parameters.
    """

    def __init__(
        self,
        cfg: Any,
        split: str = "train",
        seq_len: int = 243,
        stride: int = 1,
    ):
        super().__init__(cfg, split, seq_len, stride)
        self.data_path = Path(cfg.data.datasets.h36m.path)

        if split == "train":
            self.subjects = cfg.data.datasets.h36m.subjects_train
        else:
            self.subjects = cfg.data.datasets.h36m.subjects_test

        self._load_data()

    def _load_data(self) -> None:
        """
        Load Human3.6M data.

        Expected directory structure:
        h36m/
            S1/
                poses_3d.npy     # (N, 17, 3)
                poses_2d.npy     # (N, 17, 2) - projected
                cameras.pkl      # camera parameters
            S5/
            ...
        """
        self.samples = []

        for subject in self.subjects:
            subject_dir = self.data_path / f"S{subject}"
            if not subject_dir.exists():
                continue

            # Load poses
            poses_3d_file = subject_dir / "poses_3d.npy"
            poses_2d_file = subject_dir / "poses_2d.npy"

            if not poses_3d_file.exists():
                continue

            poses_3d = np.load(poses_3d_file)
            poses_2d = np.load(poses_2d_file) if poses_2d_file.exists() else None

            n_frames = len(poses_3d)

            # Create sequences with stride
            for start in range(0, n_frames - self.seq_len + 1, self.stride):
                self.samples.append({
                    "subject": subject,
                    "start_frame": start,
                    "poses_3d": poses_3d,
                    "poses_2d": poses_2d,
                })

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        start = sample["start_frame"]
        end = start + self.seq_len

        poses_3d = torch.from_numpy(
            sample["poses_3d"][start:end].copy()
        ).float()

        result = {
            "poses_3d": poses_3d,
            "has_3d": True,
            "dataset": "h36m",
        }

        if sample["poses_2d"] is not None:
            poses_2d = torch.from_numpy(
                sample["poses_2d"][start:end].copy()
            ).float()
            result["poses_2d"] = poses_2d

        return result


class GymVideoDataset(PoseDataset):
    """
    In-the-wild gym video dataset for weakly-supervised training.

    Uses pre-extracted 2D keypoints from YOLO-Pose or OpenPose as pseudo-labels.
    No 3D ground truth available.
    """

    def __init__(
        self,
        cfg: Any,
        split: str = "train",
        seq_len: int = 243,
        stride: int = 1,
    ):
        super().__init__(cfg, split, seq_len, stride)
        self.data_path = Path(cfg.data.datasets.gym_videos.path)
        self.detector = cfg.data.datasets.gym_videos.detector
        self.conf_threshold = cfg.data.datasets.gym_videos.confidence_threshold
        self.skeleton_converter = SkeletonConverter("coco_17", "h36m_17")
        self._load_data()

    def _load_data(self) -> None:
        """
        Load gym video 2D detections.

        Expected structure:
        gym_videos/
            video_001/
                keypoints.npy   # (N, 17, 2) COCO format
                confidence.npy  # (N, 17)
            video_002/
            ...
        """
        self.samples = []

        if not self.data_path.exists():
            return

        for video_dir in sorted(self.data_path.iterdir()):
            if not video_dir.is_dir():
                continue

            kp_file = video_dir / "keypoints.npy"
            conf_file = video_dir / "confidence.npy"

            if not kp_file.exists():
                continue

            keypoints = np.load(kp_file)
            confidence = np.load(conf_file) if conf_file.exists() else None

            n_frames = len(keypoints)

            for start in range(0, n_frames - self.seq_len + 1, self.stride):
                self.samples.append({
                    "video": video_dir.name,
                    "start_frame": start,
                    "keypoints": keypoints,
                    "confidence": confidence,
                })

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        start = sample["start_frame"]
        end = start + self.seq_len

        keypoints = torch.from_numpy(
            sample["keypoints"][start:end].copy()
        ).float()

        confidence = None
        if sample["confidence"] is not None:
            confidence = torch.from_numpy(
                sample["confidence"][start:end].copy()
            ).float()

        # Convert COCO -> H36M format
        keypoints_h36m, conf_h36m = self.skeleton_converter.convert(
            keypoints, confidence
        )

        # Mask low-confidence detections
        if confidence is not None:
            mask = conf_h36m > self.conf_threshold
        else:
            mask = torch.ones(keypoints_h36m.shape[:-1], dtype=torch.bool)

        return {
            "poses_2d": keypoints_h36m,
            "confidence": conf_h36m,
            "mask": mask,
            "has_3d": False,
            "dataset": "gym",
        }


class Fit3DDataset(PoseDataset):
    """
    Fit3D dataset for evaluation on fitness movements.

    Provides 3D ground truth for complex fitness poses.
    """

    def __init__(
        self,
        cfg: Any,
        split: str = "test",
        seq_len: int = 243,
        stride: int = 1,
    ):
        super().__init__(cfg, split, seq_len, stride)
        self.data_path = Path(cfg.data.datasets.fit3d.path)
        self._load_data()

    def _load_data(self) -> None:
        """
        Load Fit3D data.

        Expected structure:
        fit3d/
            test/
                sequence_001/
                    poses_3d.npy
                    poses_2d.npy
        """
        self.samples = []
        split_path = self.data_path / self.split

        if not split_path.exists():
            return

        for seq_dir in sorted(split_path.iterdir()):
            if not seq_dir.is_dir():
                continue

            poses_3d = np.load(seq_dir / "poses_3d.npy")
            poses_2d_file = seq_dir / "poses_2d.npy"
            poses_2d = np.load(poses_2d_file) if poses_2d_file.exists() else None

            n_frames = len(poses_3d)

            for start in range(0, n_frames - self.seq_len + 1, self.stride):
                self.samples.append({
                    "sequence": seq_dir.name,
                    "start_frame": start,
                    "poses_3d": poses_3d,
                    "poses_2d": poses_2d,
                })

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        start = sample["start_frame"]
        end = start + self.seq_len

        poses_3d = torch.from_numpy(
            sample["poses_3d"][start:end].copy()
        ).float()

        result = {
            "poses_3d": poses_3d,
            "has_3d": True,
            "dataset": "fit3d",
        }

        if sample["poses_2d"] is not None:
            result["poses_2d"] = torch.from_numpy(
                sample["poses_2d"][start:end].copy()
            ).float()

        return result


def create_dataloaders(cfg: Any) -> dict:
    """Create train/val/test dataloaders from config."""
    from torch.utils.data import DataLoader, ConcatDataset

    seq_len = cfg.data.seq_len
    stride = cfg.data.stride

    # Training datasets
    train_datasets = []

    # H36M (3D supervised)
    h36m_train = H36MDataset(cfg, split="train", seq_len=seq_len, stride=stride)
    if len(h36m_train) > 0:
        train_datasets.append(h36m_train)

    # Gym videos (2D weakly-supervised)
    gym_train = GymVideoDataset(cfg, split="train", seq_len=seq_len, stride=stride)
    if len(gym_train) > 0:
        train_datasets.append(gym_train)

    train_dataset = ConcatDataset(train_datasets) if train_datasets else None

    # Validation
    h36m_val = H36MDataset(cfg, split="val", seq_len=seq_len, stride=seq_len)

    # Test (Fit3D)
    fit3d_test = Fit3DDataset(cfg, split="test", seq_len=seq_len, stride=seq_len)

    dataloaders = {}

    if train_dataset and len(train_dataset) > 0:
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
        )

    if len(h36m_val) > 0:
        dataloaders["val"] = DataLoader(
            h36m_val,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

    if len(fit3d_test) > 0:
        dataloaders["test"] = DataLoader(
            fit3d_test,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

    return dataloaders
