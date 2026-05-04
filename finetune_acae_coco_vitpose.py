#!/usr/bin/env python3
"""Fine-tune the aligned ACAE on extracted COCO ViTPose 2D keypoints.

COCO detections are 2D-only. We embed them as unified skeleton poses with
normalized x/y coordinates scaled by 1000 and a constant z=1000. The existing
ACAE loss recognizes these flat-depth samples and uses its projected 2D
``splat`` loss rather than pretending COCO has real 3D depth.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from run_paths import CHECKPOINT_DIR, COCO_2D_DIR, REPO_ROOT, ensure_artifact_dirs, first_existing_path

sys.path.insert(0, str(REPO_ROOT / "acae_2D_extension"))
from prepare_h36m_fit3d import build_unified_skeleton  # noqa: E402


DEFAULT_BASE_ACAE = CHECKPOINT_DIR / "h36_fit_checkpoint.pth"
DEFAULT_OUTPUT = CHECKPOINT_DIR / "h36_fit_coco_checkpoint.pth"

COCO_TO_UNIFIED = {
    "nose": "nose",
    "left_eye": "leye",
    "right_eye": "reye",
    "left_ear": "lear",
    "right_ear": "rear",
    "left_shoulder": "lshoulder",
    "right_shoulder": "rshoulder",
    "left_elbow": "lelbow",
    "right_elbow": "relbow",
    "left_wrist": "lwrist",
    "right_wrist": "rwrist",
    "left_hip": "lhip",
    "right_hip": "rhip",
    "left_knee": "lknee",
    "right_knee": "rknee",
    "left_ankle": "lankle",
    "right_ankle": "rankle",
}


def load_acae_module():
    acae_path = REPO_ROOT / "acae_2D_extension" / "affine_combining_autoencoder" / "acae_2.5d_torch.py"
    spec = importlib.util.spec_from_file_location("acae_torch", acae_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_xy(xy: np.ndarray, width: float, height: float) -> np.ndarray:
    return xy / width * 2.0 - np.array([1.0, height / width], dtype=np.float32)


def midpoint(poses: np.ndarray, scores: np.ndarray, out_idx: int, a_idx: int, b_idx: int) -> None:
    valid = (scores[:, a_idx] > 0.0) & (scores[:, b_idx] > 0.0)
    if not valid.any():
        return
    poses[valid, out_idx] = (poses[valid, a_idx] + poses[valid, b_idx]) * 0.5
    poses[valid, out_idx, 2] = 1000.0
    scores[valid, out_idx] = np.minimum(scores[valid, a_idx], scores[valid, b_idx])


def weighted_midpoint(
    poses: np.ndarray,
    scores: np.ndarray,
    out_idx: int,
    a_idx: int,
    b_idx: int,
    alpha: float,
) -> None:
    valid = (scores[:, a_idx] > 0.0) & (scores[:, b_idx] > 0.0)
    if not valid.any():
        return
    poses[valid, out_idx] = poses[valid, a_idx] * (1.0 - alpha) + poses[valid, b_idx] * alpha
    poses[valid, out_idx, 2] = 1000.0
    scores[valid, out_idx] = np.minimum(scores[valid, a_idx], scores[valid, b_idx])


def load_image_sizes(annotation_path: Path) -> Dict[int, Tuple[float, float]]:
    with annotation_path.open("r") as f:
        data = json.load(f)
    return {int(img["id"]): (float(img["width"]), float(img["height"])) for img in data["images"]}


def record_iter(index_path: Path) -> Iterable[Dict]:
    with index_path.open("r") as f:
        index = json.load(f)
    yield from index["images"]


def convert_npz_to_unified_poses(
    npz_path: Path,
    image_size: Tuple[float, float],
    unified_names: Sequence[str],
    min_score: float,
    min_valid_joints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    keypoints = np.asarray(data["keypoints"], dtype=np.float32)
    scores_17 = np.asarray(data["scores"], dtype=np.float32)
    names_17 = [str(item) for item in data["keypoint_names"]]
    if keypoints.shape[0] == 0:
        return np.zeros((0, len(unified_names), 3), dtype=np.float32), np.zeros((0, len(unified_names)), dtype=np.float32)

    width, height = image_size
    unified_idx = {name: i for i, name in enumerate(unified_names)}
    poses = np.zeros((keypoints.shape[0], len(unified_names), 3), dtype=np.float32)
    scores = np.zeros((keypoints.shape[0], len(unified_names)), dtype=np.float32)

    for coco_idx, coco_name in enumerate(names_17):
        unified_name = COCO_TO_UNIFIED.get(coco_name)
        if unified_name is None or unified_name not in unified_idx:
            continue
        xy = keypoints[:, coco_idx]
        valid = np.isfinite(xy).all(axis=-1) & (scores_17[:, coco_idx] >= min_score)
        if not valid.any():
            continue
        u_idx = unified_idx[unified_name]
        poses[valid, u_idx, :2] = normalize_xy(xy[valid], width, height) * 1000.0
        poses[valid, u_idx, 2] = 1000.0
        scores[valid, u_idx] = scores_17[valid, coco_idx]

    for out_name, a_name, b_name in [
        ("pelvis", "lhip", "rhip"),
        ("neck", "lshoulder", "rshoulder"),
    ]:
        if all(name in unified_idx for name in (out_name, a_name, b_name)):
            midpoint(poses, scores, unified_idx[out_name], unified_idx[a_name], unified_idx[b_name])

    if all(name in unified_idx for name in ("spine", "pelvis", "neck")):
        weighted_midpoint(poses, scores, unified_idx["spine"], unified_idx["pelvis"], unified_idx["neck"], alpha=0.45)
    if all(name in unified_idx for name in ("thorax", "pelvis", "neck")):
        weighted_midpoint(poses, scores, unified_idx["thorax"], unified_idx["pelvis"], unified_idx["neck"], alpha=0.75)

    keep = np.count_nonzero(scores > 0.0, axis=1) >= min_valid_joints
    return poses[keep], scores[keep]


def load_coco_vitpose_poses(
    coco_2d_dir: Path,
    annotation_path: Path,
    unified_names: Sequence[str],
    min_score: float,
    min_valid_joints: int,
    max_poses: int | None,
) -> np.ndarray:
    index_path = coco_2d_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing COCO ViTPose index: {index_path}")
    sizes = load_image_sizes(annotation_path)

    all_poses: List[np.ndarray] = []
    total_people = 0
    kept_people = 0
    for i, record in enumerate(record_iter(index_path), start=1):
        image_id = int(record["image_id"])
        output = Path(record["output"])
        if not output.is_absolute():
            output = REPO_ROOT / output
        if image_id not in sizes or not output.exists():
            continue

        poses, _scores = convert_npz_to_unified_poses(
            output,
            sizes[image_id],
            unified_names,
            min_score=min_score,
            min_valid_joints=min_valid_joints,
        )
        total_people += int(record.get("num_persons", len(poses)))
        if len(poses):
            all_poses.append(poses)
            kept_people += len(poses)
            if max_poses is not None and kept_people >= max_poses:
                break
        if i % 10000 == 0:
            print(f"  scanned {i} COCO images; kept {kept_people} detected people", flush=True)

    if not all_poses:
        raise RuntimeError("No COCO ViTPose poses survived filtering.")
    poses_all = np.concatenate(all_poses, axis=0)
    if max_poses is not None and len(poses_all) > max_poses:
        poses_all = poses_all[:max_poses]
    print(f"COCO detected people: total={total_people} kept={len(poses_all)}", flush=True)
    return poses_all.astype(np.float32)


class PoseDataset(Dataset):
    def __init__(self, poses: np.ndarray):
        self.poses = torch.from_numpy(poses.astype(np.float32))

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.poses[idx]


def sample_rows(array: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or len(array) == 0:
        return np.zeros((0,) + array.shape[1:], dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(array), size=min(count, len(array)), replace=False)
    return np.asarray(array[indices], dtype=np.float32)


def load_replay_poses(
    replay_data_dir: Path,
    checkpoint_joint_names: Sequence[str],
    permutation: Sequence[int],
    coco_pose_count: int,
    replay_fraction: float,
    max_replay_poses: int,
    retention_eval_poses: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_path = replay_data_dir / "poses_train.npy"
    test_path = replay_data_dir / "poses_test.npy"
    names_path = replay_data_dir / "joint_names.npy"
    if replay_fraction <= 0.0:
        return (
            np.zeros((0, len(checkpoint_joint_names), 3), dtype=np.float32),
            np.zeros((0, len(checkpoint_joint_names), 3), dtype=np.float32),
        )
    if not train_path.exists() or not test_path.exists() or not names_path.exists():
        raise FileNotFoundError(
            f"Replay requested but missing ACAE data files under {replay_data_dir}. "
            "Expected poses_train.npy, poses_test.npy, and joint_names.npy."
        )

    replay_joint_names = [str(name) for name in np.load(names_path, allow_pickle=True)]
    if replay_joint_names != list(checkpoint_joint_names):
        raise RuntimeError(
            "Replay joint names do not match checkpoint joint names. "
            f"Replay={replay_joint_names}, checkpoint={list(checkpoint_joint_names)}"
        )

    train_mem = np.load(train_path, mmap_mode="r")
    test_mem = np.load(test_path, mmap_mode="r")
    replay_count = min(max_replay_poses, int(coco_pose_count * replay_fraction))
    replay_train = sample_rows(train_mem, replay_count, seed=123)
    replay_val = sample_rows(test_mem, retention_eval_poses, seed=456)
    replay_train = replay_train[:, permutation]
    replay_val = replay_val[:, permutation]
    print(
        f"Replay poses: train={len(replay_train)} retention_val={len(replay_val)} "
        f"from {replay_data_dir}",
        flush=True,
    )
    return replay_train.astype(np.float32), replay_val.astype(np.float32)


def masked_mpjpe_mm(batch: torch.Tensor, pred: torch.Tensor) -> float:
    valid = (batch.abs().sum(dim=-1) > 1e-5)
    if not valid.any():
        return float("nan")
    error = torch.linalg.norm((pred - batch) / 1000.0, dim=-1)
    return float(error[valid].mean().item() * 1000.0)


@torch.no_grad()
def evaluate_loader(
    model,
    compute_loss,
    loader: DataLoader | None,
    regul_lambda: float,
    device: str,
    use_projected_loss: bool,
    include_mpjpe: bool = False,
) -> Dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    totals = {"loss": 0.0, "main_loss": 0.0, "regul": 0.0, "mpjpe_mm": 0.0}
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        losses = compute_loss(batch, pred, model, regul_lambda, use_projected_loss=use_projected_loss)
        totals["loss"] += float(losses["loss"].item())
        totals["main_loss"] += float(losses["main_loss"].item())
        totals["regul"] += float(losses["regul"].item())
        if include_mpjpe:
            totals["mpjpe_mm"] += masked_mpjpe_mm(batch, pred)
        batches += 1

    denom = max(batches, 1)
    out = {
        "loss": totals["loss"] / denom,
        "main_loss": totals["main_loss"] / denom,
        "regul": totals["regul"] / denom,
    }
    if include_mpjpe:
        out["mpjpe_mm"] = totals["mpjpe_mm"] / denom
    return out


def train(
    model,
    optimizer: torch.optim.Optimizer,
    compute_loss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    retention_loader: DataLoader | None,
    epochs: int,
    regul_lambda: float,
    device: str,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_main = 0.0
        train_regul = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            losses = compute_loss(batch, pred, model, regul_lambda, use_projected_loss=True)
            losses["loss"].backward()
            optimizer.step()
            train_loss += float(losses["loss"].item())
            train_main += float(losses["main_loss"].item())
            train_regul += float(losses["regul"].item())

        val_metrics = evaluate_loader(
            model,
            compute_loss,
            val_loader,
            regul_lambda,
            device,
            use_projected_loss=True,
            include_mpjpe=False,
        )
        retention_metrics = evaluate_loader(
            model,
            compute_loss,
            retention_loader,
            regul_lambda,
            device,
            use_projected_loss=True,
            include_mpjpe=True,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_projected_loss": train_main / max(len(train_loader), 1),
            "train_regul": train_regul / max(len(train_loader), 1),
            "val_loss": val_metrics["loss"],
            "val_projected_loss": val_metrics["main_loss"],
            "val_regul": val_metrics["regul"],
        }
        if retention_metrics:
            row.update(
                {
                    "retention_loss": retention_metrics["loss"],
                    "retention_recon_loss": retention_metrics["main_loss"],
                    "retention_regul": retention_metrics["regul"],
                    "retention_mpjpe_mm": retention_metrics["mpjpe_mm"],
                }
            )
        rows.append(row)
        msg = (
            f"Epoch {epoch}/{epochs} "
            f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"train_proj={row['train_projected_loss']:.6f} val_proj={row['val_projected_loss']:.6f}"
        )
        if retention_metrics:
            msg += (
                f" retention={row['retention_loss']:.6f} "
                f"retention_mpjpe={row['retention_mpjpe_mm']:.2f}mm"
            )
        print(msg, flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-2d-dir", type=Path, default=COCO_2D_DIR / "vitpose_train2017")
    parser.add_argument("--annotation-file", type=Path, default=Path("data/coco/annotations/person_keypoints_train2017.json"))
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_ACAE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--regul-lambda", type=float, default=6e-1)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--min-valid-joints", type=int, default=6)
    parser.add_argument("--max-poses", type=int, default=200000)
    parser.add_argument("--replay-data-dir", type=Path, default=Path("acae_data_h36_fit"))
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.25,
        help="Original H36M/Fit3D ACAE poses to mix into training as a fraction of COCO poses.",
    )
    parser.add_argument("--max-replay-poses", type=int, default=50000)
    parser.add_argument(
        "--retention-eval-poses",
        type=int,
        default=8192,
        help="Held-out original ACAE poses used only for retention logging.",
    )
    parser.add_argument("--disable-replay", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is not available.")

    acae_module = load_acae_module()
    base_checkpoint = first_existing_path(args.base_checkpoint)
    ckpt = torch.load(base_checkpoint, map_location=args.device, weights_only=False)
    permutation = ckpt.get("permutation")
    inv_permutation = ckpt.get("inv_permutation")
    joint_names = list(ckpt.get("joint_names", build_unified_skeleton()[0]))
    if permutation is None or inv_permutation is None:
        raise RuntimeError(f"Base ACAE checkpoint lacks permutation metadata: {base_checkpoint}")

    print("==========================================", flush=True)
    print("Fine-tuning ACAE on COCO ViTPose 2D keypoints", flush=True)
    print(f"Base checkpoint: {base_checkpoint}", flush=True)
    print(f"COCO 2D dir: {args.coco_2d_dir}", flush=True)
    print(f"Output checkpoint: {args.output}", flush=True)
    print("Input type: real ViTPose 2D, embedded as flat z=1000 for ACAE projected loss", flush=True)
    print("==========================================", flush=True)

    poses = load_coco_vitpose_poses(
        args.coco_2d_dir,
        args.annotation_file,
        joint_names,
        min_score=args.min_score,
        min_valid_joints=args.min_valid_joints,
        max_poses=args.max_poses,
    )
    poses = poses[:, permutation]
    val_size = max(1, int(len(poses) * args.val_fraction))
    rng = np.random.default_rng(42)
    order = rng.permutation(len(poses))
    val_indices = order[:val_size]
    train_indices = order[val_size:]
    coco_train = poses[train_indices]
    coco_val = poses[val_indices]

    replay_fraction = 0.0 if args.disable_replay else args.replay_fraction
    replay_train, replay_val = load_replay_poses(
        args.replay_data_dir,
        joint_names,
        permutation,
        coco_pose_count=len(poses),
        replay_fraction=replay_fraction,
        max_replay_poses=args.max_replay_poses,
        retention_eval_poses=args.retention_eval_poses,
    )
    train_poses = coco_train
    if len(replay_train):
        train_poses = np.concatenate([coco_train, replay_train], axis=0)

    train_loader = DataLoader(PoseDataset(train_poses), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(PoseDataset(coco_val), batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    retention_loader = None
    if len(replay_val):
        retention_loader = DataLoader(PoseDataset(replay_val), batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    print(
        f"COCO train poses: {len(coco_train)}  COCO val poses: {len(coco_val)}  "
        f"Replay train poses: {len(replay_train)}  Retention val poses: {len(replay_val)}",
        flush=True,
    )

    model = acae_module.load_acae_from_checkpoint(str(base_checkpoint), device=args.device, freeze=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rows = train(
        model,
        optimizer,
        acae_module.compute_acae_loss,
        train_loader,
        val_loader,
        retention_loader,
        epochs=args.epochs,
        regul_lambda=args.regul_lambda,
        device=args.device,
    )

    model.eval()
    with torch.no_grad():
        w1_internal = model.encoder.get_w().detach().cpu().numpy()
        w2_internal = model.decoder.get_w().detach().cpu().numpy()
    w1, w2 = acae_module.permute_weights(w1_internal, w2_internal, inv_permutation)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": args.epochs,
            "source_checkpoint": str(base_checkpoint),
            "training_data": "COCO train2017 ViTPose 2D detections",
            "input_representation": "unified skeleton, normalized xy * 1000, flat z=1000",
            "hyperparams": ckpt["hyperparams"],
            "permutation": permutation,
            "inv_permutation": inv_permutation,
            "joint_names": joint_names,
            "w1": w1,
            "w2": w2,
            "args": vars(args),
        },
        args.output,
    )
    np.savez(args.output.with_suffix(".npz"), w1=w1, w2=w2)
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("==========================================", flush=True)
    print(f"Saved checkpoint: {args.output}", flush=True)
    print(f"Saved weights: {args.output.with_suffix('.npz')}", flush=True)
    print(f"Saved log: {args.output.with_suffix('.csv')}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
