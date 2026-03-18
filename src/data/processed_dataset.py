"""
Dataset class for processed pose data (our format).

Loads data from:
- data/processed/h36m/{train,test}/
- data/processed/mpi_3dhp/{train,test}/
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json


class ProcessedPoseDataset(Dataset):
    """
    Dataset for loading processed pose data.

    Each sequence directory contains:
    - poses_3d.npy: (T, 17, 3) root-centered 3D poses in meters
    - poses_2d.npy: (T, 17, 2) normalized 2D poses in [-1, 1]
    - valid.npy: (T,) optional validity mask
    """

    def __init__(
        self,
        data_root: str | Path,
        dataset: str = "h36m",  # "h36m" or "mpi_3dhp"
        split: str = "train",
        seq_len: int = 243,
        stride: int = 81,
        return_full_sequence: bool = False,
    ):
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.return_full_sequence = return_full_sequence

        self.data_path = self.data_root / dataset / split
        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load all sequences and create sample indices."""
        if not self.data_path.exists():
            print(f"Warning: Data path {self.data_path} does not exist")
            return

        # Load metadata if available
        metadata_file = self.data_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            sequence_names = [m['sequence'] for m in metadata]
        else:
            sequence_names = [d.name for d in sorted(self.data_path.iterdir()) if d.is_dir()]

        for seq_name in sequence_names:
            seq_dir = self.data_path / seq_name
            if not seq_dir.is_dir():
                continue

            poses_3d_file = seq_dir / "poses_3d.npy"
            poses_2d_file = seq_dir / "poses_2d.npy"

            if not poses_3d_file.exists() or not poses_2d_file.exists():
                continue

            # Load data
            poses_3d = np.load(poses_3d_file)
            poses_2d = np.load(poses_2d_file)
            valid = None
            if (seq_dir / "valid.npy").exists():
                valid = np.load(seq_dir / "valid.npy")

            n_frames = len(poses_3d)

            if self.return_full_sequence:
                # Return entire sequence (for evaluation)
                self.samples.append({
                    'sequence': seq_name,
                    'poses_3d': poses_3d,
                    'poses_2d': poses_2d,
                    'valid': valid,
                    'start': 0,
                    'end': n_frames,
                })
            else:
                # Create sliding window samples
                for start in range(0, n_frames - self.seq_len + 1, self.stride):
                    self.samples.append({
                        'sequence': seq_name,
                        'poses_3d': poses_3d,
                        'poses_2d': poses_2d,
                        'valid': valid,
                        'start': start,
                        'end': start + self.seq_len,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start, end = sample['start'], sample['end']

        poses_3d = torch.from_numpy(sample['poses_3d'][start:end].copy()).float()
        poses_2d = torch.from_numpy(sample['poses_2d'][start:end].copy()).float()

        result = {
            'poses_3d': poses_3d,
            'poses_2d': poses_2d,
            'has_3d': True,
            'sequence': sample['sequence'],
        }

        if sample['valid'] is not None:
            result['valid'] = torch.from_numpy(sample['valid'][start:end].copy()).bool()

        return result


def create_dataloader(
    data_root: str | Path,
    dataset: str,
    split: str,
    batch_size: int = 32,
    seq_len: int = 243,
    stride: int = 81,
    shuffle: bool = False,
    num_workers: int = 4,
    return_full_sequence: bool = False,
):
    """Create a DataLoader for processed pose data."""
    from torch.utils.data import DataLoader

    ds = ProcessedPoseDataset(
        data_root=data_root,
        dataset=dataset,
        split=split,
        seq_len=seq_len,
        stride=stride,
        return_full_sequence=return_full_sequence,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train" and not return_full_sequence),
    )
