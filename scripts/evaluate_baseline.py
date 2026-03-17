#!/usr/bin/env python3
"""
Baseline Evaluation Script.

Evaluates pretrained models (MotionBERT, APTPose) on Fit3D dataset
without any fine-tuning to establish baseline performance.

Usage:
    # Evaluate MotionBERT
    python scripts/evaluate_baseline.py --model motionbert --checkpoint path/to/checkpoint.bin

    # Evaluate APTPose
    python scripts/evaluate_baseline.py --model aptpose --checkpoint path/to/checkpoint.pth

    # Evaluate both
    python scripts/evaluate_baseline.py --model both

    # Custom data path
    python scripts/evaluate_baseline.py --model motionbert --data_path ./data/fit3d_processed/test
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.models import load_pretrained_model
from src.data.datasets import Fit3DDataset
from src.metrics import PoseMetrics, compute_mpjpe, compute_p_mpjpe, compute_bli


def create_minimal_config(
    seq_len: int = 243,
    num_joints: int = 17,
) -> Config:
    """Create minimal config for model initialization."""
    return Config({
        "model": {
            "num_joints": num_joints,
            "input_dim": 2,
            "output_dim": 3,
            "seq_len": seq_len,
        },
        "data": {
            "output_skeleton": "h36m_17",
            "seq_len": seq_len,
            "stride": seq_len,  # Non-overlapping for eval
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "datasets": {
                "fit3d": {
                    "path": "./data/fit3d_processed",
                    "split": "test",
                }
            }
        },
    })


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate a model on the given dataloader.

    Returns:
        Dict with MPJPE, P-MPJPE, BLI metrics
    """
    model.eval()
    metrics = PoseMetrics(skeleton="h36m_17")

    all_mpjpe = []
    all_p_mpjpe = []
    all_bli = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            poses_2d = batch["poses_2d"].to(device)
            poses_3d_gt = batch["poses_3d"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)

            # Forward pass
            poses_3d_pred = model(poses_2d, mask)

            # Compute metrics
            mpjpe = compute_mpjpe(poses_3d_pred, poses_3d_gt)
            p_mpjpe = compute_p_mpjpe(poses_3d_pred, poses_3d_gt)
            bli = compute_bli(poses_3d_pred, skeleton="h36m_17")

            all_mpjpe.append(mpjpe.item())
            all_p_mpjpe.append(p_mpjpe.item())
            all_bli.append(bli.item())

    return {
        "mpjpe": sum(all_mpjpe) / len(all_mpjpe),
        "p_mpjpe": sum(all_p_mpjpe) / len(all_p_mpjpe),
        "bli": sum(all_bli) / len(all_bli),
        "num_samples": len(dataloader.dataset),
    }


def download_checkpoint(model_name: str, output_dir: Path) -> Path:
    """Download pretrained checkpoint if not present."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "motionbert":
        # MotionBERT checkpoint URLs
        checkpoints = {
            "h36m": "https://github.com/Walter0807/MotionBERT/releases/download/v1.0.0/FT_MB_lite_MB_ft_h36m_global_lite.bin",
        }
        ckpt_path = output_dir / "motionbert_h36m.bin"

        if not ckpt_path.exists():
            print(f"Downloading MotionBERT checkpoint...")
            import urllib.request
            urllib.request.urlretrieve(checkpoints["h36m"], ckpt_path)
            print(f"Downloaded to {ckpt_path}")

        return ckpt_path

    elif model_name == "aptpose":
        # APTPose - user needs to download manually
        ckpt_path = output_dir / "aptpose.pth"
        if not ckpt_path.exists():
            print(f"""
APTPose checkpoint not found at {ckpt_path}

Please download from: https://github.com/wenwen12321/APTPose
And place the checkpoint at: {ckpt_path}
""")
            return None
        return ckpt_path

    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models on Fit3D")
    parser.add_argument(
        "--model",
        type=str,
        default="motionbert",
        choices=["motionbert", "aptpose", "both"],
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-downloads if not specified)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/fit3d_processed/test",
        help="Path to processed Fit3D test data",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=243,
        help="Sequence length (frames)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/baseline_results.json",
        help="Path to save results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"""
Fit3D data not found at {data_path}

Please run the preparation script first:
    python scripts/prepare_fit3d.py --data_root ./data/fit3d

Or specify a different path with --data_path
""")
        return

    # Create config and dataloader
    cfg = create_minimal_config(seq_len=args.seq_len)
    cfg.data.datasets.fit3d.path = str(data_path.parent)
    cfg.data.batch_size = args.batch_size

    dataset = Fit3DDataset(cfg, split="test", seq_len=args.seq_len)
    if len(dataset) == 0:
        print(f"No data found in {data_path}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Loaded {len(dataset)} sequences from Fit3D")

    # Determine which models to evaluate
    models_to_eval = ["motionbert", "aptpose"] if args.model == "both" else [args.model]

    results = {
        "timestamp": datetime.now().isoformat(),
        "data_path": str(data_path),
        "seq_len": args.seq_len,
        "models": {},
    }

    for model_name in models_to_eval:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*50}")

        # Get checkpoint
        if args.checkpoint and len(models_to_eval) == 1:
            ckpt_path = Path(args.checkpoint)
        else:
            ckpt_path = download_checkpoint(model_name, Path("checkpoints"))

        if ckpt_path is None or not ckpt_path.exists():
            print(f"Skipping {model_name} - checkpoint not found")
            continue

        try:
            # Load model
            model = load_pretrained_model(model_name, cfg, ckpt_path)
            model.to(device)

            print(f"Model parameters: {model.count_parameters():,}")

            # Evaluate
            metrics = evaluate_model(model, dataloader, device)

            print(f"\nResults for {model_name}:")
            print(f"  MPJPE:   {metrics['mpjpe']:.2f} mm")
            print(f"  P-MPJPE: {metrics['p_mpjpe']:.2f} mm")
            print(f"  BLI:     {metrics['bli']:.4f}")

            results["models"][model_name] = metrics

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print comparison table if both models evaluated
    if len(results["models"]) > 1:
        print("\n" + "="*50)
        print("COMPARISON TABLE")
        print("="*50)
        print(f"{'Model':<15} {'MPJPE (mm)':<12} {'P-MPJPE (mm)':<12} {'BLI':<10}")
        print("-"*50)
        for name, metrics in results["models"].items():
            print(f"{name:<15} {metrics['mpjpe']:<12.2f} {metrics['p_mpjpe']:<12.2f} {metrics['bli']:<10.4f}")


if __name__ == "__main__":
    main()
