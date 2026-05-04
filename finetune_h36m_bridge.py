#!/usr/bin/env python3
"""Fine-tune pretrained VideoPose3D through the frozen ACAE bridge on H36M.

This trains the baseline:
  H36M 2D -> ACAE encode -> pretrained VideoPose3D -> ACAE decode -> H36M 3D

Loss is computed in decoded unified-skeleton space, with optional left/right
bone symmetry regularization. ACAE is frozen by default; only VideoPose3D is
fine-tuned unless --train-acae is passed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from run_paths import (
    ACAE_CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    H36M_BRIDGE_CHECKPOINT_PATH,
    PRETRAINED_H36M_CHECKPOINT_PATH,
    ensure_artifact_dirs,
    first_existing_path,
)

sys.path.insert(0, os.path.abspath("vdp3d"))
sys.path.insert(0, os.path.abspath("acae_2D_extension"))

from common.camera import normalize_screen_coordinates, world_to_camera  # noqa: E402
from common.h36m_dataset import Human36mDataset  # noqa: E402
from common.model import TemporalModel  # noqa: E402
from prepare_h36m_fit3d import build_unified_skeleton  # noqa: E402


H36M_TO_LRC = [4, 5, 6, 11, 12, 13, 1, 2, 3, 14, 15, 16, 0, 7, 8, 9, 10]
LRC_TO_H36M = np.argsort(H36M_TO_LRC).tolist()


def load_acae_module():
    acae_path = Path("acae_2D_extension") / "affine_combining_autoencoder" / "acae_2.5d_torch.py"
    spec = importlib.util.spec_from_file_location("acae_torch", acae_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_acae_with_metadata(checkpoint_path: Path, device: str, freeze: bool):
    acae_module = load_acae_module()
    model = acae_module.load_acae_from_checkpoint(str(checkpoint_path), device=device, freeze=freeze)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    permutation = ckpt.get("permutation")
    inv_permutation = ckpt.get("inv_permutation")
    if permutation is None or inv_permutation is None:
        raise RuntimeError(f"ACAE checkpoint lacks permutation metadata: {checkpoint_path}")
    return model, permutation, inv_permutation


def load_vdp3d_weights(model: nn.Module, checkpoint_path: Path) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_pos"] if "model_pos" in checkpoint else checkpoint

    if any(key.startswith("vdp3d.") for key in state):
        state = {key[len("vdp3d.") :]: value for key, value in state.items() if key.startswith("vdp3d.")}
    elif any(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Could not load VideoPose3D weights cleanly from {checkpoint_path}. "
            f"Missing={missing}, unexpected={unexpected}"
        )
    return checkpoint


def load_bridge_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_pos"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def make_h36m_bridge_sequences(
    dataset: Human36mDataset,
    keypoints_2d: Dict,
    subjects: Sequence[str],
    sample_stride: int,
    min_frames: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str], List[int]]:
    unified_names, h36m_to_unified, _ = build_unified_skeleton()
    j_unified = len(unified_names)

    sequences: List[Tuple[np.ndarray, np.ndarray]] = []
    for subject in subjects:
        if subject not in dataset.subjects():
            continue
        cameras = dataset.cameras()[subject]
        for action in dataset[subject].keys():
            pos3d_world = dataset[subject][action]["positions"].astype(np.float32)
            if subject not in keypoints_2d or action not in keypoints_2d[subject]:
                continue

            for cam_idx, pos2d_raw in enumerate(keypoints_2d[subject][action]):
                cam = cameras[cam_idx]
                pos2d = pos2d_raw.astype(np.float32).copy()
                pos2d[..., :2] = normalize_screen_coordinates(pos2d[..., :2], w=cam["res_w"], h=cam["res_h"])
                pos3d_cam = world_to_camera(pos3d_world, R=cam["orientation"], t=cam["translation"]).astype(np.float32)

                n = min(len(pos3d_cam), len(pos2d))
                pos3d = pos3d_cam[:n:sample_stride]
                pos2d = pos2d[:n:sample_stride]
                if len(pos3d) < min_frames:
                    continue

                pos3d = pos3d - pos3d[:, :1, :]

                pos2d_unified = np.zeros((len(pos2d), j_unified, 2), dtype=np.float32)
                pos3d_unified = np.zeros((len(pos3d), j_unified, 3), dtype=np.float32)
                for h_idx, u_idx in enumerate(h36m_to_unified):
                    pos2d_unified[:, u_idx, :] = pos2d[:, h_idx, :]
                    pos3d_unified[:, u_idx, :] = pos3d[:, h_idx, :]

                sequences.append((pos2d_unified, pos3d_unified))

    return sequences, unified_names, h36m_to_unified


class BridgedTemporalModel(nn.Module):
    def __init__(self, vdp3d_model: nn.Module, acae_checkpoint: Path, device: str, freeze_acae: bool = True):
        super().__init__()
        self.acae, permutation, inv_permutation = load_acae_with_metadata(acae_checkpoint, device, freeze=freeze_acae)
        self.vdp3d = vdp3d_model
        self.register_buffer("permutation", torch.as_tensor(permutation, dtype=torch.long))
        self.register_buffer("inv_permutation", torch.as_tensor(inv_permutation, dtype=torch.long))
        self.register_buffer("h36m_to_lrc", torch.as_tensor(H36M_TO_LRC, dtype=torch.long))
        self.register_buffer("lrc_to_h36m", torch.as_tensor(LRC_TO_H36M, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, j, c = x.shape
        x_flat = x.reshape(b * t, j, c)

        x_perm = x_flat.index_select(1, self.permutation)
        latent_lrc = self.acae.encode(x_perm).reshape(b, t, 17, c)

        latent_h36m = latent_lrc.index_select(2, self.lrc_to_h36m)
        pred_h36m = self.vdp3d(latent_h36m)

        pred_lrc = pred_h36m.index_select(2, self.h36m_to_lrc)
        pred_perm = self.acae.decode(pred_lrc.reshape(-1, 17, 3))
        pred_orig = pred_perm.index_select(1, self.inv_permutation)
        return pred_orig.reshape(pred_h36m.shape[0], pred_h36m.shape[1], -1, 3)

    def receptive_field(self) -> int:
        return self.vdp3d.receptive_field()


def masked_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = (target.abs().sum(dim=-1) > 1e-5)
    error = torch.linalg.norm(pred - target, dim=-1)
    return error[valid].mean()


def bone_symmetry_loss(pred: torch.Tensor, target: torch.Tensor, joint_names: Sequence[str]) -> torch.Tensor:
    pairs = [
        ("lhip", "lknee", "rhip", "rknee"),
        ("lknee", "lankle", "rknee", "rankle"),
        ("lshoulder", "lelbow", "rshoulder", "relbow"),
        ("lelbow", "lwrist", "relbow", "rwrist"),
        ("lhip", "lshoulder", "rhip", "rshoulder"),
    ]
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    losses = []
    for la, lb, ra, rb in pairs:
        if not all(name in name_to_idx for name in (la, lb, ra, rb)):
            continue
        idx = [name_to_idx[name] for name in (la, lb, ra, rb)]
        valid = (target[..., idx, :].abs().sum(dim=-1) > 1e-5).all(dim=-1)
        if not valid.any():
            continue
        left_len = torch.linalg.norm(pred[..., idx[0], :] - pred[..., idx[1], :], dim=-1)
        right_len = torch.linalg.norm(pred[..., idx[2], :] - pred[..., idx[3], :], dim=-1)
        losses.append((left_len[valid] - right_len[valid]).abs().mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def joint_angle_limit_loss(pred: torch.Tensor, target: torch.Tensor, joint_names: Sequence[str]) -> torch.Tensor:
    """Softly penalize implausible elbow/knee/hip/shoulder bend angles."""
    angle_specs = [
        ("lhip", "lknee", "lankle", 5.0, 178.0),
        ("rhip", "rknee", "rankle", 5.0, 178.0),
        ("lshoulder", "lelbow", "lwrist", 5.0, 178.0),
        ("rshoulder", "relbow", "rwrist", 5.0, 178.0),
        ("thorax", "lshoulder", "lelbow", 10.0, 175.0),
        ("thorax", "rshoulder", "relbow", 10.0, 175.0),
        ("spine", "lhip", "lknee", 10.0, 175.0),
        ("spine", "rhip", "rknee", 10.0, 175.0),
    ]
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    losses = []
    eps = 1e-8
    for a, b, c, min_deg, max_deg in angle_specs:
        if not all(name in name_to_idx for name in (a, b, c)):
            continue
        ia, ib, ic = [name_to_idx[name] for name in (a, b, c)]
        valid = (target[..., [ia, ib, ic], :].abs().sum(dim=-1) > 1e-5).all(dim=-1)
        if not valid.any():
            continue
        v1 = pred[..., ia, :] - pred[..., ib, :]
        v2 = pred[..., ic, :] - pred[..., ib, :]
        v1 = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + eps)
        v2 = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + eps)
        cos_angle = (v1 * v2).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(cos_angle)
        min_angle = pred.new_tensor(np.deg2rad(min_deg))
        max_angle = pred.new_tensor(np.deg2rad(max_deg))
        losses.append((torch.relu(min_angle - angle[valid]).square() + torch.relu(angle[valid] - max_angle).square()).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_one_epoch(
    model: BridgedTemporalModel,
    sequences: Sequence[Tuple[np.ndarray, np.ndarray]],
    optimizer: torch.optim.Optimizer,
    device: str,
    pad: int,
    joint_names: Sequence[str],
    bone_loss_weight: float,
    angle_loss_weight: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_bone = 0.0
    total_angle = 0.0
    n = 0

    order = np.random.permutation(len(sequences))
    for seq_i in order:
        pos2d, pos3d = sequences[seq_i]
        pos2d_padded = np.pad(pos2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")

        inputs_2d = torch.from_numpy(pos2d_padded).to(device=device, dtype=torch.float32).unsqueeze(0)
        targets_3d = torch.from_numpy(pos3d).to(device=device, dtype=torch.float32).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        pred = model(inputs_2d)
        mpjpe = masked_mpjpe(pred, targets_3d)
        bone_loss = bone_symmetry_loss(pred, targets_3d, joint_names)
        angle_loss = joint_angle_limit_loss(pred, targets_3d, joint_names)
        loss = mpjpe + bone_loss_weight * bone_loss + angle_loss_weight * angle_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_mpjpe += float(mpjpe.item())
        total_bone += float(bone_loss.item())
        total_angle += float(angle_loss.item())
        n += 1

        if n % 100 == 0:
            print(f"  trained {n}/{len(sequences)} sequences", flush=True)

    denom = max(n, 1)
    return {
        "loss": total_loss / denom,
        "mpjpe": total_mpjpe / denom,
        "bone_loss": total_bone / denom,
        "angle_loss": total_angle / denom,
    }


@torch.no_grad()
def evaluate(
    model: BridgedTemporalModel,
    sequences: Sequence[Tuple[np.ndarray, np.ndarray]],
    device: str,
    pad: int,
    joint_names: Sequence[str],
) -> Dict[str, float]:
    model.eval()
    total_mpjpe = 0.0
    total_bone = 0.0
    total_angle = 0.0
    n = 0
    for pos2d, pos3d in sequences:
        pos2d_padded = np.pad(pos2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        inputs_2d = torch.from_numpy(pos2d_padded).to(device=device, dtype=torch.float32).unsqueeze(0)
        targets_3d = torch.from_numpy(pos3d).to(device=device, dtype=torch.float32).unsqueeze(0)
        pred = model(inputs_2d)
        total_mpjpe += float(masked_mpjpe(pred, targets_3d).item())
        total_bone += float(bone_symmetry_loss(pred, targets_3d, joint_names).item())
        total_angle += float(joint_angle_limit_loss(pred, targets_3d, joint_names).item())
        n += 1
    denom = max(n, 1)
    return {"mpjpe": total_mpjpe / denom, "bone_loss": total_bone / denom, "angle_loss": total_angle / denom}


def parse_subjects(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h36m-3d", type=Path, default=Path("data/data_3d_h36m.npz"))
    parser.add_argument("--h36m-2d", type=Path, default=Path("data/data_2d_h36m_gt.npz"))
    parser.add_argument("--acae-checkpoint", type=Path, default=ACAE_CHECKPOINT_PATH)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED_H36M_CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=H36M_BRIDGE_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--bone-loss-weight", type=float, default=0.01)
    parser.add_argument("--angle-loss-weight", type=float, default=0.001)
    parser.add_argument("--train-subjects", default="S1,S5,S6,S7,S8")
    parser.add_argument("--val-subjects", default="S9,S11")
    parser.add_argument("--train-acae", action="store_true", help="Fine-tune ACAE too. Default keeps ACAE frozen.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_artifact_dirs()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    acae_checkpoint = first_existing_path(args.acae_checkpoint, "artifacts/checkpoints/h36_fit_checkpoint.pth")
    pretrained = first_existing_path(args.pretrained, "checkpoint/epoch_120.bin")
    print("==========================================", flush=True)
    print("Fine-tuning VideoPose3D through ACAE bridge on H36M", flush=True)
    print(f"Pretrained VDP3D: {pretrained}", flush=True)
    print(f"ACAE checkpoint: {acae_checkpoint}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Device: {args.device}", flush=True)
    print("==========================================", flush=True)

    receptive_field = 243
    # Human36mDataset mutates its module-level skeleton while removing static
    # joints, so construct it once and slice train/validation from that object.
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
    print(f"Train sequences: {len(train_seqs)}", flush=True)
    print(f"Val sequences: {len(val_seqs)}", flush=True)
    if not train_seqs:
        raise RuntimeError("No H36M training sequences found.")

    vdp3d = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024, dense=False)
    load_vdp3d_weights(vdp3d, pretrained)

    model = BridgedTemporalModel(vdp3d, acae_checkpoint, args.device, freeze_acae=not args.train_acae).to(args.device)
    pad = (model.receptive_field() - 1) // 2
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    log_rows = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_seqs,
            optimizer,
            args.device,
            pad,
            joint_names,
            args.bone_loss_weight,
            args.angle_loss_weight,
        )
        val_stats = evaluate(model, val_seqs, args.device, pad, joint_names) if val_seqs else {"mpjpe": float("nan"), "bone_loss": float("nan"), "angle_loss": float("nan")}
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_mpjpe": train_stats["mpjpe"],
            "train_bone_loss": train_stats["bone_loss"],
            "train_angle_loss": train_stats["angle_loss"],
            "val_mpjpe": val_stats["mpjpe"],
            "val_bone_loss": val_stats["bone_loss"],
            "val_angle_loss": val_stats["angle_loss"],
        }
        log_rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train={train_stats['mpjpe']:.2f}mm "
            f"bone={train_stats['bone_loss']:.2f} "
            f"angle={train_stats['angle_loss']:.5f} "
            f"val={val_stats['mpjpe']:.2f}mm",
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
            "train_subjects": args.train_subjects,
            "val_subjects": args.val_subjects,
            "sample_stride": args.sample_stride,
            "bone_loss_weight": args.bone_loss_weight,
            "angle_loss_weight": args.angle_loss_weight,
            "bridge_ordering": "permute unified -> ACAE LRC latent -> H36M order for VDP3D -> LRC for ACAE decode",
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
