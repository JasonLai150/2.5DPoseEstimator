#!/usr/bin/env python3
"""
Evaluate pretrained MotionBERT (DSTformer) on H36M and Fit3D.

Loads MB_ft_h36m weights into the in-repo DSTformer and reports
MPJPE / P-MPJPE on the processed H36M and Fit3D test sets.

Usage:
    python scripts/evaluate_motionbert.py \
        --weights checkpoints/motionbert/pose3d/MB_ft_h36m.bin \
        --fit3d_root data/processed/fit3d \
        --output_json outputs/eval/motionbert_baseline.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processed_dataset import ProcessedPoseDataset
from src.models.dstformer import DSTformer


class _Cfg:
    """Minimal cfg shim mirroring DSTformer's expected interface."""

    class _Sub:
        def get(self, k, default=None):
            return getattr(self, k, default)

    def __init__(self, **kwargs):
        self.model = self._Sub()
        self.data = self._Sub()
        for k, v in kwargs.items():
            setattr(self.model, k, v)
            if k in {"num_joints", "input_dim", "output_dim", "seq_len"}:
                setattr(self.data, k, v)


def build_model(weights_path: str, seq_len: int, device: torch.device) -> DSTformer:
    cfg = _Cfg(
        num_joints=17, input_dim=2, output_dim=3, seq_len=seq_len,
        embed_dim=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        lora={"enabled": False},
    )
    model = DSTformer(cfg)
    model.load_pretrained(weights_path)
    return model.to(device).eval()


def compute_mpjpe(pred_m: torch.Tensor, target_m: torch.Tensor) -> float:
    """MPJPE in mm given root-relative meters."""
    return torch.linalg.norm(pred_m - target_m, dim=-1).mean().item() * 1000.0


def compute_p_mpjpe(pred_m: torch.Tensor, target_m: torch.Tensor) -> float:
    """Procrustes-aligned MPJPE in mm (batched, vectorized)."""
    p = (pred_m * 1000).reshape(-1, pred_m.shape[-2], 3).cpu().numpy()
    t = (target_m * 1000).reshape(-1, target_m.shape[-2], 3).cpu().numpy()

    p_c = p - p.mean(axis=1, keepdims=True)
    t_c = t - t.mean(axis=1, keepdims=True)
    p_s = np.sqrt((p_c ** 2).sum(axis=(1, 2), keepdims=True)) + 1e-8
    t_s = np.sqrt((t_c ** 2).sum(axis=(1, 2), keepdims=True)) + 1e-8
    p_n = p_c / p_s
    t_n = t_c / t_s

    H = np.einsum("nij,nik->njk", p_n, t_n)
    U, _, Vt = np.linalg.svd(H)
    R = np.einsum("nji,nkj->nik", Vt, U)
    det = np.linalg.det(R)
    Vt[det < 0, -1, :] *= -1
    R = np.einsum("nji,nkj->nik", Vt, U)

    p_aligned = np.einsum("nij,nkj->nki", R, p_n) * t_s + t.mean(axis=1, keepdims=True)
    return float(np.linalg.norm(p_aligned - t, axis=-1).mean(axis=-1).mean())


@torch.no_grad()
def evaluate(model: DSTformer, loader: DataLoader, device: torch.device, name: str) -> dict:
    mpjpe, p_mpjpe = [], []
    for batch in tqdm(loader, desc=f"Evaluating {name}"):
        poses_2d = batch["poses_2d"].to(device)
        poses_3d = batch["poses_3d"].to(device)
        pred_3d = model(poses_2d)
        # Root-center predictions to match the GT convention (pelvis at origin).
        pred_3d = pred_3d - pred_3d[..., 0:1, :]
        mpjpe.append(compute_mpjpe(pred_3d, poses_3d))
        p_mpjpe.append(compute_p_mpjpe(pred_3d, poses_3d))
    return {"mpjpe": float(np.mean(mpjpe)), "p_mpjpe": float(np.mean(p_mpjpe))}


def make_loader(data_root: str, dataset: str, split: str, batch_size: int, seq_len: int) -> DataLoader:
    ds = ProcessedPoseDataset(
        data_root=data_root, dataset=dataset, split=split,
        seq_len=seq_len, stride=seq_len,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="checkpoints/motionbert/pose3d/MB_ft_h36m.bin")
    p.add_argument("--data_root", default="./data/processed", help="H36M data root")
    p.add_argument("--fit3d_root", default="./data/processed/fit3d")
    p.add_argument("--seq_len", type=int, default=243)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading MotionBERT weights: {args.weights}")
    model = build_model(args.weights, args.seq_len, device)
    print(f"Params: {sum(x.numel() for x in model.parameters()):,}")

    loaders = {
        "H36M":  make_loader(args.data_root, "h36m",  "test", args.batch_size, args.seq_len),
        "Fit3D": make_loader(args.fit3d_root, "",     "test", args.batch_size, args.seq_len),
    }
    for n, l in loaders.items():
        print(f"  {n} test: {len(l.dataset)} windows ({args.seq_len}-frame)")

    results = {n: evaluate(model, l, device, n) for n, l in loaders.items()}

    print("\n" + "=" * 60)
    print("MotionBERT (MB_ft_h36m) zero-shot")
    print("=" * 60)
    for n, m in results.items():
        tag = "(in-domain)" if n == "H36M" else "(target domain)"
        print(f"\n{n} {tag}:")
        print(f"  MPJPE:   {m['mpjpe']:.2f} mm")
        print(f"  P-MPJPE: {m['p_mpjpe']:.2f} mm")
    if "H36M" in results and "Fit3D" in results:
        print(f"\nDomain gap (H36M -> Fit3D): {results['Fit3D']['mpjpe'] - results['H36M']['mpjpe']:.2f} mm")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "model": "DSTformer (MotionBERT MB_ft_h36m)",
            "weights": str(Path(args.weights).resolve()),
            "seq_len": args.seq_len,
            "datasets": {n: {"num_samples": len(loaders[n].dataset), **m} for n, m in results.items()},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "device": str(device),
        }, open(out, "w"), indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
        print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
