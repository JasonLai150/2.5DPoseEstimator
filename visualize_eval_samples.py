#!/usr/bin/env python3
"""Render one comparison image per evaluated sample.

The script runs the current bridge checkpoint on a small set of Fit3D and/or
H36M sequences, then saves one PNG per sample with the GT pose on the left and
the prediction on the right. Each filename and title includes the sample-level
MPJPE so it is easy to compare examples by eye.
"""

from __future__ import annotations

import argparse
import csv
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from eval_fit3d_vitpose import (
    fit3d_joints25_to_unified_3d,
    load_fit3d_targets_and_cameras,
    load_model,
    vitpose_to_unified_2d,
)
from finetune_h36m_bridge import parse_subjects
from prepare_h36m_fit3d import build_unified_skeleton
from run_paths import (
    FIT3D_2D_DIR,
    H36M_BRIDGE_CHECKPOINT_PATH,
    VISUALIZATION_DIR,
    ensure_artifact_dirs,
    first_existing_path,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "vdp3d"))

from common.camera import normalize_screen_coordinates, world_to_camera  # noqa: E402
from common.camera import image_coordinates  # noqa: E402
from common.h36m_dataset import Human36mDataset  # noqa: E402


BONES = [
    ("lshoulder", "lelbow"),
    ("lelbow", "lwrist"),
    ("rshoulder", "relbow"),
    ("relbow", "rwrist"),
    ("lshoulder", "rshoulder"),
    ("lhip", "lknee"),
    ("lknee", "lankle"),
    ("rhip", "rknee"),
    ("rknee", "rankle"),
    ("lhip", "rhip"),
    ("lshoulder", "lhip"),
    ("rshoulder", "rhip"),
    ("pelvis", "thorax"),
    ("thorax", "neck"),
    ("neck", "headtop"),
]


@dataclass
class SampleRecord:
    dataset: str
    label: str
    pos2d: np.ndarray
    pos3d: np.ndarray
    background_image: np.ndarray | None = None
    image_size: tuple[int, int] | None = None
    source_member: str | None = None
    frame_index: int | None = None


def sequence_mpjpe_mm(pred: np.ndarray, target: np.ndarray) -> float:
    valid = np.abs(target).sum(axis=-1) > 1e-5
    if not valid.any():
        return float("nan")
    error = np.linalg.norm(pred - target, axis=-1)
    return float(error[valid].mean() * 1000.0)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_")


def pose_axis_limit(*poses: np.ndarray) -> float:
    finite_values = []
    for pose in poses:
        valid = np.abs(pose).sum(axis=-1) > 1e-5
        if valid.any():
            finite_values.append(np.abs(pose[valid]).reshape(-1))
    if not finite_values:
        return 1.0
    return max(1.0, float(np.nanmax(np.concatenate(finite_values))) * 1.25)


def plot_pose(
    ax: plt.Axes,
    pose: np.ndarray,
    joint_names: Sequence[str],
    title: str,
    color: str,
    lim: float | None = None,
) -> None:
    valid = np.abs(pose).sum(axis=-1) > 1e-5
    ax.scatter(pose[valid, 0], pose[valid, 2], -pose[valid, 1], c=color, s=18, alpha=0.95)

    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for a, b in BONES:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        ia = name_to_idx[a]
        ib = name_to_idx[b]
        if valid[ia] and valid[ib]:
            ax.plot(
                [pose[ia, 0], pose[ib, 0]],
                [pose[ia, 2], pose[ib, 2]],
                [-pose[ia, 1], -pose[ib, 1]],
                c=color,
                linewidth=1.6,
                alpha=0.9,
            )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Depth")
    ax.set_zlabel("Height")
    ax.view_init(elev=18, azim=-70)
    if lim is None:
        lim = pose_axis_limit(pose)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def plot_pose_on_image(
    ax: plt.Axes,
    pose_2d: np.ndarray,
    joint_names: Sequence[str],
    title: str,
    color: str,
    background_image: np.ndarray,
) -> None:
    h, w = background_image.shape[:2]
    ax.imshow(background_image)
    valid = np.abs(pose_2d).sum(axis=-1) > 1e-5
    pixels = image_coordinates(pose_2d, w=w, h=h)
    ax.scatter(pixels[valid, 0], pixels[valid, 1], c=color, s=18, alpha=0.95)
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for a, b in BONES:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        ia = name_to_idx[a]
        ib = name_to_idx[b]
        if valid[ia] and valid[ib]:
            ax.plot(
                [pixels[ia, 0], pixels[ib, 0]],
                [pixels[ia, 1], pixels[ib, 1]],
                c=color,
                linewidth=1.6,
                alpha=0.9,
            )
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def plot_missing_frame(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.text(
        0.5,
        0.5,
        "Fit3D frame unavailable",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.set_axis_off()


def extract_tar_member_to_temp(tf: tarfile.TarFile, member_name: str) -> Path:
    while True:
        try:
            member = tf.next()
        except EOFError as exc:
            raise KeyError(f"Archive member not found before EOF: {member_name}") from exc
        if member is None:
            break
        if member.name != member_name:
            continue
        handle = tf.extractfile(member)
        if handle is None:
            raise RuntimeError(f"Could not open archive member: {member_name}")
        suffix = Path(member_name).suffix or ".mp4"
        fd, tmp_name = tempfile.mkstemp(prefix="fit3d_viz_", suffix=suffix)
        tmp_path = Path(tmp_name)
        with handle, open(fd, "wb", closefd=True) as dst:
            dst.write(handle.read())
        return tmp_path
    raise KeyError(f"Archive member not found before EOF: {member_name}")


def load_fit3d_background_frame(fit3d_path: Path, source_member: str, frame_index: int) -> np.ndarray | None:
    cap = None
    tmp_video: Path | None = None
    try:
        with tarfile.open(fit3d_path, "r|gz") as tf:
            tmp_video = extract_tar_member_to_temp(tf, source_member)
    except Exception as exc:
        print(f"WARNING: could not extract Fit3D video member {source_member}: {exc}", flush=True)
        return None

    try:
        cap = cv2.VideoCapture(str(tmp_video))
        if not cap.isOpened():
            print(f"WARNING: could not open extracted Fit3D video {source_member}", flush=True)
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
        ok, frame = cap.read()
        if not ok:
            print(f"WARNING: could not read Fit3D frame {frame_index} from {source_member}", flush=True)
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            if tmp_video is not None:
                tmp_video.unlink(missing_ok=True)
        except Exception:
            pass


def load_fit3d_records(
    fit3d_path: Path,
    fit3d_2d_dir: Path,
    sample_stride: int,
    min_frames: int,
) -> tuple[list[SampleRecord], list[str], dict[str, int]]:
    targets, camera_params = load_fit3d_targets_and_cameras(fit3d_path)
    unified_names, _, _ = build_unified_skeleton()

    records: list[SampleRecord] = []
    stats = {"seen": 0, "matched": 0, "missing_target": 0, "missing_camera": 0, "too_short": 0}

    for npz_path in sorted(fit3d_2d_dir.rglob("*.npz")):
        stats["seen"] += 1
        npz = np.load(npz_path, allow_pickle=True)
        pos2d, _scores, (split, subject, camera, action) = vitpose_to_unified_2d(npz_path, unified_names)
        target = targets.get((split, subject, action))
        if target is None:
            stats["missing_target"] += 1
            continue
        cam_params = camera_params.get((split, subject, camera, action))
        if cam_params is None:
            stats["missing_camera"] += 1
            continue

        n = min(len(pos2d), len(target))
        pos2d = pos2d[:n:sample_stride]
        target = target[:n:sample_stride]
        if len(pos2d) < min_frames:
            stats["too_short"] += 1
            continue

        source_member = str(npz["source_member"].item() if npz["source_member"].shape == () else npz["source_member"])
        mid_frame_index = (len(pos2d) // 2) * sample_stride
        pos3d = fit3d_joints25_to_unified_3d(target, len(unified_names), cam_params)
        records.append(
            SampleRecord(
                dataset="Fit3D",
                label=f"{split}/{subject}/{action}/{camera}",
                pos2d=pos2d,
                pos3d=pos3d,
                image_size=(int(npz["meta_width"].item()), int(npz["meta_height"].item())),
                source_member=source_member,
                frame_index=mid_frame_index,
            )
        )
        stats["matched"] += 1

    return records, unified_names, stats


def attach_fit3d_backgrounds(records: Sequence[SampleRecord], fit3d_path: Path) -> list[SampleRecord]:
    hydrated = []
    for record in records:
        if record.source_member is None or record.frame_index is None:
            hydrated.append(record)
            continue
        record.background_image = load_fit3d_background_frame(fit3d_path, record.source_member, record.frame_index)
        status = "loaded" if record.background_image is not None else "missing"
        print(
            f"Fit3D background {status}: {record.label} "
            f"member={record.source_member} frame={record.frame_index}",
            flush=True,
        )
        hydrated.append(record)
    return hydrated


def load_h36m_records(
    h36m_3d_path: Path,
    h36m_2d_path: Path,
    subjects: Sequence[str],
    sample_stride: int,
    min_frames: int,
) -> tuple[list[SampleRecord], list[str]]:
    dataset = Human36mDataset(str(h36m_3d_path))
    keypoints_2d = np.load(h36m_2d_path, allow_pickle=True)["positions_2d"].item()
    unified_names, h36m_to_unified, _ = build_unified_skeleton()

    records: list[SampleRecord] = []
    for subject in subjects:
        if subject not in dataset.subjects():
            continue
        cameras = dataset.cameras()[subject]
        for action in dataset[subject].keys():
            if subject not in keypoints_2d or action not in keypoints_2d[subject]:
                continue

            pos3d_world = dataset[subject][action]["positions"].astype(np.float32)
            for cam_idx, pos2d_raw in enumerate(keypoints_2d[subject][action]):
                cam = cameras[cam_idx]
                pos2d = pos2d_raw.astype(np.float32).copy()
                pos2d[..., :2] = normalize_screen_coordinates(pos2d[..., :2], w=cam["res_w"], h=cam["res_h"])
                pos3d_cam = world_to_camera(pos3d_world, R=cam["orientation"], t=cam["translation"]).astype(np.float32)

                n = min(len(pos3d_cam), len(pos2d))
                pos2d = pos2d[:n:sample_stride]
                pos3d = pos3d_cam[:n:sample_stride]
                if len(pos2d) < min_frames:
                    continue

                pos3d = pos3d - pos3d[:, :1, :]
                pos2d_unified = np.zeros((len(pos2d), len(unified_names), 2), dtype=np.float32)
                pos3d_unified = np.zeros((len(pos3d), len(unified_names), 3), dtype=np.float32)
                for h_idx, u_idx in enumerate(h36m_to_unified):
                    pos2d_unified[:, u_idx, :] = pos2d[:, h_idx, :]
                    pos3d_unified[:, u_idx, :] = pos3d[:, h_idx, :]

                records.append(
                    SampleRecord(
                        dataset="H36M",
                        label=f"{subject}/{action}/cam{cam_idx}",
                        pos2d=pos2d_unified,
                        pos3d=pos3d_unified,
                    )
                )

    return records, unified_names


def select_records(records: Sequence[SampleRecord], count: int, seed: int) -> list[SampleRecord]:
    if count <= 0 or not records:
        return []
    rng = np.random.default_rng(seed)
    order = np.arange(len(records))
    rng.shuffle(order)
    return [records[idx] for idx in order[: min(count, len(records))]]


def render_sample(
    record: SampleRecord,
    pred_seq: np.ndarray,
    joint_names: Sequence[str],
    output_path: Path,
) -> float:
    mid = len(record.pos3d) // 2
    gt = record.pos3d[mid]
    pred = pred_seq[mid]
    sample_mpjpe = sequence_mpjpe_mm(pred_seq, record.pos3d)

    if record.dataset == "Fit3D":
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(f"{record.dataset} | {record.label} | MPJPE {sample_mpjpe:.2f} mm", fontsize=14)
        ax_input = fig.add_subplot(1, 3, 1)
        if record.background_image is not None:
            plot_pose_on_image(
                ax_input,
                record.pos2d[mid],
                joint_names,
                "ViTPose 2D Input on Frame",
                "dodgerblue",
                record.background_image,
            )
        else:
            plot_missing_frame(ax_input, "ViTPose 2D Input on Frame")
        lim = pose_axis_limit(gt, pred)
        ax_gt = fig.add_subplot(1, 3, 2, projection="3d")
        plot_pose(ax_gt, gt, joint_names, "Fit3D 3D Target", "green", lim=lim)
        ax_pred = fig.add_subplot(1, 3, 3, projection="3d")
        plot_pose(ax_pred, pred, joint_names, "Prediction", "crimson", lim=lim)
    else:
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(f"{record.dataset} | {record.label} | MPJPE {sample_mpjpe:.2f} mm", fontsize=14)
        lim = pose_axis_limit(gt, pred)
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
        plot_pose(ax_gt, gt, joint_names, "Ground Truth", "green", lim=lim)
        ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
        plot_pose(ax_pred, pred, joint_names, "Prediction", "crimson", lim=lim)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return sample_mpjpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=H36M_BRIDGE_CHECKPOINT_PATH)
    parser.add_argument("--acae-checkpoint", type=Path, default=Path("artifacts/checkpoints/h36_fit_checkpoint.pth"))
    parser.add_argument("--fit3d-path", type=Path, default=Path("data/fit3d_train.tar.gz"))
    parser.add_argument("--fit3d-2d-dir", type=Path, default=FIT3D_2D_DIR / "vitpose_fullframe")
    parser.add_argument("--h36m-3d", type=Path, default=Path("data/data_3d_h36m.npz"))
    parser.add_argument("--h36m-2d", type=Path, default=Path("data/data_2d_h36m_gt.npz"))
    parser.add_argument("--h36m-subjects", default="S9,S11")
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--fit3d-count", type=int, default=3)
    parser.add_argument("--h36m-count", type=int, default=3)
    parser.add_argument("--min-frames", type=int, default=243)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-fit3d", action="store_true")
    parser.add_argument("--skip-h36m", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALIZATION_DIR / "sample_comparisons",
        help="Directory where per-sample PNGs and summary.csv are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    checkpoint = first_existing_path(args.checkpoint)
    acae_checkpoint = first_existing_path(args.acae_checkpoint)
    print("==========================================", flush=True)
    print("Rendering sample comparison figures", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"ACAE checkpoint: {acae_checkpoint}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print("==========================================", flush=True)

    model = load_model(checkpoint, acae_checkpoint, device)
    pad = (model.receptive_field() - 1) // 2

    summary_rows: list[dict[str, str]] = []

    if not args.skip_fit3d:
        fit3d_records, joint_names, stats = load_fit3d_records(
            args.fit3d_path,
            args.fit3d_2d_dir,
            sample_stride=args.sample_stride,
            min_frames=args.min_frames,
        )
        print(
            f"Fit3D records: seen={stats['seen']} matched={stats['matched']} "
            f"missing_target={stats['missing_target']} missing_camera={stats['missing_camera']} "
            f"too_short={stats['too_short']}",
            flush=True,
        )
        chosen = attach_fit3d_backgrounds(select_records(fit3d_records, args.fit3d_count, args.seed), args.fit3d_path)
        for idx, record in enumerate(chosen, start=1):
            pos2d = np.pad(record.pos2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            inputs = torch.from_numpy(pos2d).to(device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = model(inputs)[0].cpu().numpy()

            mpjpe_mm = render_sample(
                record,
                pred,
                joint_names,
                args.output_dir / f"fit3d_{idx:02d}_{safe_name(record.label)}.png",
            )
            summary_rows.append(
                {
                    "dataset": record.dataset,
                    "label": record.label,
                    "mpjpe_mm": f"{mpjpe_mm:.4f}",
                    "output_path": str(args.output_dir / f"fit3d_{idx:02d}_{safe_name(record.label)}.png"),
                }
            )
            print(f"  saved Fit3D sample {idx}/{len(chosen)} -> {mpjpe_mm:.2f} mm", flush=True)

    if not args.skip_h36m:
        h36m_records, joint_names = load_h36m_records(
            args.h36m_3d,
            args.h36m_2d,
            parse_subjects(args.h36m_subjects),
            sample_stride=args.sample_stride,
            min_frames=args.min_frames,
        )
        print(f"H36M records: {len(h36m_records)}", flush=True)
        chosen = select_records(h36m_records, args.h36m_count, args.seed)
        for idx, record in enumerate(chosen, start=1):
            pos2d = np.pad(record.pos2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            inputs = torch.from_numpy(pos2d).to(device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = model(inputs)[0].cpu().numpy()

            mpjpe_mm = render_sample(
                record,
                pred,
                joint_names,
                args.output_dir / f"h36m_{idx:02d}_{safe_name(record.label)}.png",
            )
            summary_rows.append(
                {
                    "dataset": record.dataset,
                    "label": record.label,
                    "mpjpe_mm": f"{mpjpe_mm:.4f}",
                    "output_path": str(args.output_dir / f"h36m_{idx:02d}_{safe_name(record.label)}.png"),
                }
            )
            print(f"  saved H36M sample {idx}/{len(chosen)} -> {mpjpe_mm:.2f} mm", flush=True)

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "label", "mpjpe_mm", "output_path"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("==========================================", flush=True)
    print(f"Wrote {len(summary_rows)} sample figures", flush=True)
    print(f"Summary CSV: {summary_path}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
