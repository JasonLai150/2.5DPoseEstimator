#!/usr/bin/env python3
"""Fine-tune VideoPose3D through an ACAE bridge on H36M plus COCO 2D.

H36M supplies the supervised 3D loss. COCO supplies only a 2D consistency loss:
COCO ViTPose detections are embedded in the unified skeleton, passed through the
same ACAE -> VideoPose3D -> ACAE path, and the decoded x/y joints are nudged to
match the input 2D joints. This does not pretend COCO has 3D ground truth.
"""

from __future__ import annotations

import argparse
import csv
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from finetune_acae_coco_vitpose import load_coco_vitpose_poses
from finetune_h36m_bridge import (
    BridgedTemporalModel,
    bone_symmetry_loss,
    joint_angle_limit_loss,
    load_vdp3d_weights,
    make_h36m_bridge_sequences,
    masked_mpjpe,
    parse_subjects,
)
from run_paths import COCO_2D_DIR, PRETRAINED_H36M_CHECKPOINT_PATH, ensure_artifact_dirs, first_existing_path

import sys

sys.path.insert(0, "vdp3d")

from common.h36m_dataset import Human36mDataset  # noqa: E402
from common.model import TemporalModel  # noqa: E402


class CocoPoseDataset(Dataset):
    def __init__(self, poses_flat_mm: np.ndarray):
        self.targets = torch.from_numpy(poses_flat_mm.astype(np.float32))
        self.inputs = self.targets[..., :2] / 1000.0

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def load_acae_joint_names(checkpoint_path: Path) -> list[str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    names = ckpt.get("joint_names")
    if names is None:
        from prepare_h36m_fit3d import build_unified_skeleton

        names = build_unified_skeleton()[0]
    return [str(name) for name in names]


def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def coco_2d_consistency_loss(pred: torch.Tensor, target_flat_mm: torch.Tensor) -> torch.Tensor:
    target_xy = target_flat_mm[..., :2] / 1000.0
    valid = target_flat_mm.abs().sum(dim=-1) > 1e-5
    if not valid.any():
        return pred.new_tensor(0.0)
    pred_xy = pred[:, 0, :, :2]
    return (pred_xy[valid] - target_xy[valid]).abs().mean()


def train_one_epoch_mixed(
    model: BridgedTemporalModel,
    h36m_sequences: Sequence[Tuple[np.ndarray, np.ndarray]],
    coco_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]] | None,
    optimizer: torch.optim.Optimizer,
    device: str,
    pad: int,
    joint_names: Sequence[str],
    bone_loss_weight: float,
    angle_loss_weight: float,
    coco_loss_weight: float,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "h36m_mpjpe": 0.0, "bone_loss": 0.0, "angle_loss": 0.0, "coco_2d_loss": 0.0}
    n = 0

    for seq_i in np.random.permutation(len(h36m_sequences)):
        pos2d, pos3d = h36m_sequences[seq_i]
        pos2d_padded = np.pad(pos2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        inputs_2d = torch.from_numpy(pos2d_padded).to(device=device, dtype=torch.float32).unsqueeze(0)
        targets_3d = torch.from_numpy(pos3d).to(device=device, dtype=torch.float32).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        pred = model(inputs_2d)
        h36m_mpjpe = masked_mpjpe(pred, targets_3d)
        bone_loss = bone_symmetry_loss(pred, targets_3d, joint_names)
        angle_loss = joint_angle_limit_loss(pred, targets_3d, joint_names)
        loss = h36m_mpjpe + bone_loss_weight * bone_loss + angle_loss_weight * angle_loss

        coco_loss = pred.new_tensor(0.0)
        if coco_iter is not None and coco_loss_weight > 0.0:
            coco_2d, coco_target = next(coco_iter)
            coco_2d = coco_2d.to(device=device, dtype=torch.float32)
            coco_target = coco_target.to(device=device, dtype=torch.float32)
            coco_padded = coco_2d[:, None, :, :].expand(-1, model.receptive_field(), -1, -1)
            coco_pred = model(coco_padded)
            coco_loss = coco_2d_consistency_loss(coco_pred, coco_target)
            coco_angle_loss = joint_angle_limit_loss(coco_pred, coco_target[:, None, :, :], joint_names)
            angle_loss = angle_loss + coco_angle_loss
            loss = loss + coco_loss_weight * coco_loss + angle_loss_weight * coco_angle_loss

        loss.backward()
        optimizer.step()

        totals["loss"] += float(loss.item())
        totals["h36m_mpjpe"] += float(h36m_mpjpe.item())
        totals["bone_loss"] += float(bone_loss.item())
        totals["angle_loss"] += float(angle_loss.item())
        totals["coco_2d_loss"] += float(coco_loss.item())
        n += 1
        if n % 100 == 0:
            print(f"  trained {n}/{len(h36m_sequences)} H36M sequences", flush=True)

    denom = max(n, 1)
    return {key: value / denom for key, value in totals.items()}


@torch.no_grad()
def evaluate_coco_2d(
    model: BridgedTemporalModel,
    loader: DataLoader | None,
    device: str,
    max_batches: int,
) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    losses = []
    for i, (coco_2d, coco_target) in enumerate(loader):
        if i >= max_batches:
            break
        coco_2d = coco_2d.to(device=device, dtype=torch.float32)
        coco_target = coco_target.to(device=device, dtype=torch.float32)
        coco_padded = coco_2d[:, None, :, :].expand(-1, model.receptive_field(), -1, -1)
        pred = model(coco_padded)
        losses.append(float(coco_2d_consistency_loss(pred, coco_target).item()))
    return float(np.mean(losses)) if losses else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h36m-3d", type=Path, default=Path("data/data_3d_h36m.npz"))
    parser.add_argument("--h36m-2d", type=Path, default=Path("data/data_2d_h36m_gt.npz"))
    parser.add_argument("--coco-2d-dir", type=Path, default=COCO_2D_DIR / "vitpose_train2017")
    parser.add_argument("--annotation-file", type=Path, default=Path("data/coco/annotations/person_keypoints_train2017.json"))
    parser.add_argument("--acae-checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED_H36M_CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--bone-loss-weight", type=float, default=0.01)
    parser.add_argument("--angle-loss-weight", type=float, default=0.001)
    parser.add_argument("--coco-loss-weight", type=float, default=0.05)
    parser.add_argument("--coco-batch-size", type=int, default=256)
    parser.add_argument("--max-coco-poses", type=int, default=100000)
    parser.add_argument("--coco-val-fraction", type=float, default=0.05)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--min-valid-joints", type=int, default=6)
    parser.add_argument("--train-subjects", default="S1,S5,S6,S7,S8")
    parser.add_argument("--val-subjects", default="S9,S11")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    acae_checkpoint = first_existing_path(args.acae_checkpoint)
    pretrained = first_existing_path(args.pretrained, "checkpoint/epoch_120.bin")

    print("==========================================", flush=True)
    print("Fine-tuning VideoPose3D through ACAE bridge on H36M + COCO", flush=True)
    print(f"Pretrained VDP3D: {pretrained}", flush=True)
    print(f"ACAE checkpoint: {acae_checkpoint}", flush=True)
    print(f"COCO 2D dir: {args.coco_2d_dir}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print("H36M: supervised 3D loss; COCO: real ViTPose 2D consistency loss", flush=True)
    print("==========================================", flush=True)

    receptive_field = 243
    h36m_dataset = Human36mDataset(str(args.h36m_3d))
    h36m_keypoints_2d = np.load(args.h36m_2d, allow_pickle=True)["positions_2d"].item()
    train_seqs, joint_names, _ = make_h36m_bridge_sequences(
        h36m_dataset,
        h36m_keypoints_2d,
        parse_subjects(args.train_subjects),
        args.sample_stride,
        receptive_field,
    )
    val_seqs, _, _ = make_h36m_bridge_sequences(
        h36m_dataset,
        h36m_keypoints_2d,
        parse_subjects(args.val_subjects),
        args.sample_stride,
        receptive_field,
    )
    print(f"H36M train sequences: {len(train_seqs)}", flush=True)
    print(f"H36M val sequences: {len(val_seqs)}", flush=True)
    if not train_seqs:
        raise RuntimeError("No H36M training sequences found.")

    acae_joint_names = load_acae_joint_names(acae_checkpoint)
    coco_poses = load_coco_vitpose_poses(
        args.coco_2d_dir,
        args.annotation_file,
        acae_joint_names,
        min_score=args.min_score,
        min_valid_joints=args.min_valid_joints,
        max_poses=args.max_coco_poses,
    )
    rng = np.random.default_rng(42)
    order = rng.permutation(len(coco_poses))
    val_size = max(1, int(len(coco_poses) * args.coco_val_fraction))
    coco_val = coco_poses[order[:val_size]]
    coco_train = coco_poses[order[val_size:]]
    coco_train_loader = DataLoader(CocoPoseDataset(coco_train), batch_size=args.coco_batch_size, shuffle=True, drop_last=True, num_workers=0)
    coco_val_loader = DataLoader(CocoPoseDataset(coco_val), batch_size=args.coco_batch_size, shuffle=False, drop_last=False, num_workers=0)
    coco_iter = infinite_loader(coco_train_loader) if len(coco_train_loader) else None
    print(f"COCO train poses: {len(coco_train)}  COCO val poses: {len(coco_val)}", flush=True)

    vdp3d = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024, dense=False)
    load_vdp3d_weights(vdp3d, pretrained)
    model = BridgedTemporalModel(vdp3d, acae_checkpoint, args.device, freeze_acae=True).to(args.device)
    pad = (model.receptive_field() - 1) // 2
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    log_rows = []
    from finetune_h36m_bridge import evaluate as evaluate_h36m

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch_mixed(
            model,
            train_seqs,
            coco_iter,
            optimizer,
            args.device,
            pad,
            joint_names,
            args.bone_loss_weight,
            args.angle_loss_weight,
            args.coco_loss_weight,
        )
        val_stats = evaluate_h36m(model, val_seqs, args.device, pad, joint_names) if val_seqs else {"mpjpe": float("nan"), "bone_loss": float("nan"), "angle_loss": float("nan")}
        coco_val_loss = evaluate_coco_2d(model, coco_val_loader, args.device, max_batches=20)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_h36m_mpjpe": train_stats["h36m_mpjpe"],
            "train_bone_loss": train_stats["bone_loss"],
            "train_angle_loss": train_stats["angle_loss"],
            "train_coco_2d_loss": train_stats["coco_2d_loss"],
            "val_h36m_mpjpe": val_stats["mpjpe"],
            "val_bone_loss": val_stats["bone_loss"],
            "val_angle_loss": val_stats["angle_loss"],
            "val_coco_2d_loss": coco_val_loss,
        }
        log_rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"h36m={row['train_h36m_mpjpe']:.5f} "
            f"angle={row['train_angle_loss']:.5f} "
            f"coco2d={row['train_coco_2d_loss']:.5f} "
            f"val_h36m={row['val_h36m_mpjpe']:.5f} "
            f"val_coco2d={row['val_coco_2d_loss']:.5f}",
            flush=True,
        )

    torch.save(
        {
            "epoch": args.epochs,
            "model_pos": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr": args.learning_rate,
            "source_pretrained": str(pretrained),
            "acae_checkpoint": str(acae_checkpoint),
            "training_data": "H36M supervised 3D + COCO ViTPose 2D consistency",
            "train_subjects": args.train_subjects,
            "val_subjects": args.val_subjects,
            "sample_stride": args.sample_stride,
            "bone_loss_weight": args.bone_loss_weight,
            "angle_loss_weight": args.angle_loss_weight,
            "coco_loss_weight": args.coco_loss_weight,
        },
        args.output,
    )

    log_path = args.output.with_suffix(".csv")
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print("==========================================", flush=True)
    print(f"Saved checkpoint: {args.output}", flush=True)
    print(f"Saved log: {log_path}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
