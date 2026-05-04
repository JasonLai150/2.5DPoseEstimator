#!/usr/bin/env python3
"""Evaluate the H36M bridge checkpoint on Fit3D ViTPose 2D keypoints.

This is the real-detector Fit3D evaluation path:
  Fit3D video -> ViTPose COCO-17 keypoints -> ACAE unified skeleton
  -> ACAE encode -> VideoPose3D -> ACAE decode -> Fit3D joints3d_25 target.

It intentionally does not create 2D by flattening/projecting Fit3D 3D poses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from run_paths import FIT3D_2D_DIR, H36M_BRIDGE_CHECKPOINT_PATH, REPO_ROOT, ensure_artifact_dirs, first_existing_path

sys.path.insert(0, str(REPO_ROOT / "vdp3d"))
sys.path.insert(0, str(REPO_ROOT / "acae_2D_extension"))

from common.camera import normalize_screen_coordinates  # noqa: E402
from common.h36m_dataset import Human36mDataset  # noqa: E402
from common.model import TemporalModel  # noqa: E402
from prepare_h36m_fit3d import build_unified_skeleton  # noqa: E402
from finetune_h36m_bridge import BridgedTemporalModel, make_h36m_bridge_sequences, parse_subjects  # noqa: E402


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


def iter_tar_members(tf: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    while True:
        try:
            member = tf.next()
        except EOFError:
            print("WARNING: hit EOFError while reading Fit3D archive; using targets loaded so far.", flush=True)
            break
        if member is None:
            break
        yield member


def parse_fit3d_pose_member(member_name: str) -> Tuple[str, str, str] | None:
    path = PurePosixPath(member_name)
    if path.suffix != ".json" or "joints3d_25" not in path.parts:
        return None
    parts = path.parts
    idx = parts.index("joints3d_25")
    if idx < 1:
        return None
    return parts[0], parts[idx - 1], path.stem


def parse_fit3d_camera_member(member_name: str) -> Tuple[str, str, str, str] | None:
    path = PurePosixPath(member_name)
    if path.suffix != ".json" or "camera_parameters" not in path.parts:
        return None
    parts = path.parts
    idx = parts.index("camera_parameters")
    if idx < 1 or idx + 1 >= len(parts):
        return None
    return parts[0], parts[idx - 1], parts[idx + 1], path.stem


def load_fit3d_targets_and_cameras(
    tar_path: Path,
) -> Tuple[Dict[Tuple[str, str, str], np.ndarray], Dict[Tuple[str, str, str, str], Dict]]:
    targets: Dict[Tuple[str, str, str], np.ndarray] = {}
    cameras: Dict[Tuple[str, str, str, str], Dict] = {}
    print(f"Loading Fit3D 3D targets and camera parameters: {tar_path}", flush=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in iter_tar_members(tf):
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            pose_key = parse_fit3d_pose_member(member.name)
            camera_key = parse_fit3d_camera_member(member.name)
            if pose_key is None and camera_key is None:
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            with handle:
                data = json.load(handle)
            if pose_key is not None:
                targets[pose_key] = np.asarray(data["joints3d_25"], dtype=np.float32)
            elif camera_key is not None:
                cameras[camera_key] = data
    print(
        f"Loaded {len(targets)} Fit3D 3D target sequences and {len(cameras)} camera parameter files.",
        flush=True,
    )
    return targets, cameras


def load_fit3d_targets(tar_path: Path) -> Dict[Tuple[str, str, str], np.ndarray]:
    targets, _ = load_fit3d_targets_and_cameras(tar_path)
    return targets


def load_h36m_inputs(h36m_3d_path: Path, h36m_2d_path: Path, subjects: Sequence[str], sample_stride: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], List[str]]:
    dataset = Human36mDataset(str(h36m_3d_path))
    keypoints_2d = np.load(h36m_2d_path, allow_pickle=True)["positions_2d"].item()
    bridged, joint_names, _ = make_h36m_bridge_sequences(
        dataset,
        keypoints_2d,
        subjects,
        sample_stride=sample_stride,
        min_frames=243,
    )
    native: List[Tuple[np.ndarray, np.ndarray]] = []
    for pos2d, pos3d in bridged:
        # Native VideoPose3D uses the 17-joint input; the bridged version keeps the
        # 28-joint unified layout, but both share the same H36M targets.
        native.append((pos2d[:, :17, :], pos3d[:, :17, :]))
    return bridged, native, joint_names


def scalar_string(npz: np.lib.npyio.NpzFile, key: str) -> str:
    value = npz[key]
    return str(value.item() if value.shape == () else value)


def midpoint(values: np.ndarray, scores: np.ndarray, out_idx: int, a_idx: int, b_idx: int) -> None:
    valid = (scores[:, a_idx] > 0.0) & (scores[:, b_idx] > 0.0)
    if not valid.any():
        return
    values[valid, out_idx, :] = (values[valid, a_idx, :] + values[valid, b_idx, :]) * 0.5
    scores[valid, out_idx] = np.minimum(scores[valid, a_idx], scores[valid, b_idx])


def weighted_midpoint(values: np.ndarray, scores: np.ndarray, out_idx: int, a_idx: int, b_idx: int, alpha: float) -> None:
    valid = (scores[:, a_idx] > 0.0) & (scores[:, b_idx] > 0.0)
    if not valid.any():
        return
    values[valid, out_idx, :] = values[valid, a_idx, :] * (1.0 - alpha) + values[valid, b_idx, :] * alpha
    scores[valid, out_idx] = np.minimum(scores[valid, a_idx], scores[valid, b_idx])


def vitpose_to_unified_2d(npz_path: Path, unified_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str, str, str]]:
    data = np.load(npz_path, allow_pickle=True)
    keypoints = np.asarray(data["keypoints"], dtype=np.float32)
    scores_17 = np.asarray(data["scores"], dtype=np.float32)
    names_17 = [str(item) for item in data["keypoint_names"]]
    width = float(np.asarray(data["meta_width"]).item())
    height = float(np.asarray(data["meta_height"]).item())

    t = keypoints.shape[0]
    unified_idx = {name: idx for idx, name in enumerate(unified_names)}
    pose = np.zeros((t, len(unified_names), 2), dtype=np.float32)
    scores = np.zeros((t, len(unified_names)), dtype=np.float32)

    for coco_idx, coco_name in enumerate(names_17):
        unified_name = COCO_TO_UNIFIED.get(coco_name)
        if unified_name is None or unified_name not in unified_idx:
            continue
        u_idx = unified_idx[unified_name]
        xy = keypoints[:, coco_idx, :].copy()
        finite = np.isfinite(xy).all(axis=-1) & (scores_17[:, coco_idx] > 0.0)
        if not finite.any():
            continue
        xy_norm = normalize_screen_coordinates(xy[finite], w=width, h=height)
        pose[finite, u_idx, :] = xy_norm
        scores[finite, u_idx] = scores_17[finite, coco_idx]

    for out_name, a_name, b_name in [
        ("pelvis", "lhip", "rhip"),
        ("neck", "lshoulder", "rshoulder"),
    ]:
        if all(name in unified_idx for name in (out_name, a_name, b_name)):
            midpoint(pose, scores, unified_idx[out_name], unified_idx[a_name], unified_idx[b_name])

    if all(name in unified_idx for name in ("spine", "pelvis", "neck")):
        weighted_midpoint(pose, scores, unified_idx["spine"], unified_idx["pelvis"], unified_idx["neck"], alpha=0.45)
    if all(name in unified_idx for name in ("thorax", "pelvis", "neck")):
        weighted_midpoint(pose, scores, unified_idx["thorax"], unified_idx["pelvis"], unified_idx["neck"], alpha=0.75)

    meta = (
        scalar_string(data, "split"),
        scalar_string(data, "subject"),
        scalar_string(data, "camera"),
        scalar_string(data, "action"),
    )
    return pose, scores, meta


FIT3D25_TO_UNIFIED = {
    # Fit3D joints3d_25 is SMPL/H36M-like, not OpenPose BODY25.
    # Coordinates are z-up; convert to the VideoPose3D camera-like convention:
    # x stays x, y becomes down/up axis (-z), z becomes the horizontal/depth-ish y.
    0: 0,   # pelvis
    1: 1,   # rhip
    2: 2,   # rknee
    3: 3,   # rankle
    4: 4,   # lhip
    5: 5,   # lknee
    6: 6,   # lankle
    7: 7,   # spine
    8: 8,   # thorax
    9: 9,   # neck
    10: 10, # headtop
    11: 14, # rshoulder
    12: 15, # relbow
    13: 16, # rwrist
    14: 11, # lshoulder
    15: 12, # lelbow
    16: 13, # lwrist
}


def fit3d_world_to_vdp3d_camera(seq_fit3d: np.ndarray) -> np.ndarray:
    out = np.empty_like(seq_fit3d, dtype=np.float32)
    out[..., 0] = seq_fit3d[..., 0]
    out[..., 1] = -seq_fit3d[..., 2]
    out[..., 2] = seq_fit3d[..., 1]
    return out


def fit3d_world_to_camera(seq_fit3d: np.ndarray, camera_params: Dict) -> np.ndarray:
    rotation = np.asarray(camera_params["extrinsics"]["R"], dtype=np.float32)
    translation = np.asarray(camera_params["extrinsics"]["T"], dtype=np.float32).reshape(1, 1, 3)
    return (seq_fit3d.astype(np.float32) - translation) @ rotation.T


def fit3d_joints25_to_unified_3d(
    seq_joints25: np.ndarray,
    joint_count: int,
    camera_params: Dict | None = None,
) -> np.ndarray:
    if camera_params is None:
        seq_cam = fit3d_world_to_vdp3d_camera(seq_joints25.astype(np.float32))
    else:
        seq_cam = fit3d_world_to_camera(seq_joints25, camera_params)
    out = np.zeros((len(seq_cam), joint_count, 3), dtype=np.float32)
    mapped_indices = []
    for fit3d_idx, unified_idx in FIT3D25_TO_UNIFIED.items():
        out[:, unified_idx, :] = seq_cam[:, fit3d_idx, :]
        mapped_indices.append(unified_idx)
    pelvis = out[:, :1, :].copy()
    out[:, mapped_indices, :] -= pelvis
    return out


def build_eval_sequences(
    keypoint_dir: Path,
    targets: Dict[Tuple[str, str, str], np.ndarray],
    camera_params: Dict[Tuple[str, str, str, str], Dict],
    sample_stride: int,
    min_frames: int,
    max_sequences: int | None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, int], List[str]]:
    unified_names, _, _ = build_unified_skeleton()
    sequences: List[Tuple[np.ndarray, np.ndarray]] = []
    stats = {"seen": 0, "matched": 0, "missing_target": 0, "missing_camera": 0, "too_short": 0}

    for npz_path in sorted(keypoint_dir.rglob("*.npz")):
        stats["seen"] += 1
        pos2d, scores, (split, subject, camera, action) = vitpose_to_unified_2d(npz_path, unified_names)
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

        pos3d = fit3d_joints25_to_unified_3d(target, len(unified_names), cam_params)
        sequences.append((pos2d, pos3d))
        stats["matched"] += 1
        if max_sequences is not None and len(sequences) >= max_sequences:
            break

    return sequences, stats, unified_names


def load_model(checkpoint_path: Path, acae_checkpoint: Path, device: torch.device) -> BridgedTemporalModel:
    vdp3d = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024, dense=False)
    model = BridgedTemporalModel(vdp3d, acae_checkpoint, str(device), freeze_acae=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_pos"])
    return model


@torch.no_grad()
def evaluate(model: BridgedTemporalModel, sequences: Sequence[Tuple[np.ndarray, np.ndarray]], device: torch.device) -> Dict[str, float]:
    model.eval()
    pad = (model.receptive_field() - 1) // 2
    total_error = 0.0
    total_joints = 0.0
    total_sequences = 0

    for pos2d, pos3d in sequences:
        pos2d_padded = np.pad(pos2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        inputs_2d = torch.from_numpy(pos2d_padded).to(device=device, dtype=torch.float32).unsqueeze(0)
        targets_3d = torch.from_numpy(pos3d).to(device=device, dtype=torch.float32).unsqueeze(0)
        pred = model(inputs_2d)

        valid = (targets_3d.abs().sum(dim=-1) > 1e-5)
        error = torch.linalg.norm(pred - targets_3d, dim=-1)
        total_error += float(error[valid].sum().item())
        total_joints += float(valid.sum().item())
        total_sequences += 1

    mpjpe = total_error / total_joints if total_joints > 0 else float("inf")
    return {"mpjpe": mpjpe, "mpjpe_mm": mpjpe * 1000.0, "sequences": float(total_sequences), "joints": total_joints}


def evaluate_h36m(model: BridgedTemporalModel, sequences: Sequence[Tuple[np.ndarray, np.ndarray]], device: torch.device) -> Dict[str, float]:
    return evaluate(model, sequences, device)


@torch.no_grad()
def evaluate_h36m_native_vdp3d(model: BridgedTemporalModel, sequences: Sequence[Tuple[np.ndarray, np.ndarray]], device: torch.device) -> Dict[str, float]:
    """Evaluate the 17-joint VideoPose3D core without ACAE encode/decode."""
    model.vdp3d.eval()
    pad = (model.receptive_field() - 1) // 2
    total_error = 0.0
    total_joints = 0.0
    total_sequences = 0

    for pos2d, pos3d in sequences:
        pos2d_padded = np.pad(pos2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        inputs_2d = torch.from_numpy(pos2d_padded).to(device=device, dtype=torch.float32).unsqueeze(0)
        targets_3d = torch.from_numpy(pos3d).to(device=device, dtype=torch.float32).unsqueeze(0)
        pred = model.vdp3d(inputs_2d)

        valid = targets_3d.abs().sum(dim=-1) > 1e-5
        error = torch.linalg.norm(pred - targets_3d, dim=-1)
        total_error += float(error[valid].sum().item())
        total_joints += float(valid.sum().item())
        total_sequences += 1

    mpjpe = total_error / total_joints if total_joints > 0 else float("inf")
    return {"mpjpe": mpjpe, "mpjpe_mm": mpjpe * 1000.0, "sequences": float(total_sequences), "joints": total_joints}


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
    parser.add_argument("--skip-fit3d", action="store_true")
    parser.add_argument("--skip-h36m", action="store_true")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    checkpoint = first_existing_path(args.checkpoint)
    acae_checkpoint = first_existing_path(args.acae_checkpoint)
    print("==========================================", flush=True)
    print("Evaluating H36M bridge checkpoint on Fit3D ViTPose 2D", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"ACAE checkpoint: {acae_checkpoint}", flush=True)
    print(f"Fit3D 3D archive: {args.fit3d_path}", flush=True)
    print(f"Fit3D 2D keypoints: {args.fit3d_2d_dir}", flush=True)
    print(f"H36M 3D archive: {args.h36m_3d}", flush=True)
    print(f"H36M 2D keypoints: {args.h36m_2d}", flush=True)
    print("Inputs: real ViTPose 2D detector keypoints, normalized like VideoPose3D", flush=True)
    print("Targets: Fit3D joints3d_25 transformed with per-video camera parameters", flush=True)
    print("==========================================", flush=True)

    model = load_model(checkpoint, acae_checkpoint, device)
    if not args.skip_fit3d:
        targets, camera_params = load_fit3d_targets_and_cameras(args.fit3d_path)
        sequences, stats, _joint_names = build_eval_sequences(
            args.fit3d_2d_dir,
            targets,
            camera_params,
            sample_stride=args.sample_stride,
            min_frames=243,
            max_sequences=args.max_sequences,
        )
        print(
            f"Fit3D 2D files seen={stats['seen']} matched={stats['matched']} "
            f"missing_target={stats['missing_target']} missing_camera={stats['missing_camera']} "
            f"too_short={stats['too_short']}",
            flush=True,
        )
        if not sequences:
            raise RuntimeError("No Fit3D ViTPose sequences matched 3D targets.")

        metrics = evaluate(model, sequences, device)
        print("==========================================", flush=True)
        print(f"Evaluated Fit3D sequences: {int(metrics['sequences'])}", flush=True)
        print(f"Fit3D ViTPose MPJPE: {metrics['mpjpe_mm']:.2f} mm", flush=True)
        print("Dataset: Fit3D", flush=True)
        print("2D input: real ViTPose detector output", flush=True)
        print("Model: H36M-trained VideoPose3D + frozen ACAE bridge", flush=True)
        print("==========================================", flush=True)

    if not args.skip_h36m:
        bridged, native, _ = load_h36m_inputs(args.h36m_3d, args.h36m_2d, parse_subjects(args.h36m_subjects), args.sample_stride)
        print(f"H36M bridged sequences: {len(bridged)}", flush=True)
        print(f"H36M native sequences: {len(native)}", flush=True)
        if not bridged or not native:
            raise RuntimeError("No H36M evaluation sequences could be built.")

        h36m_bridged = evaluate_h36m(model, bridged, device)
        h36m_native = evaluate_h36m_native_vdp3d(model, native, device)
        print("==========================================", flush=True)
        print(f"H36M bridged MPJPE: {h36m_bridged['mpjpe_mm']:.2f} mm", flush=True)
        print(f"H36M native MPJPE: {h36m_native['mpjpe_mm']:.2f} mm", flush=True)
        print("Dataset: H36M", flush=True)
        print("Bridged input: unified 28-joint H36M 2D", flush=True)
        print("Native input: 17-joint H36M 2D through VideoPose3D core only", flush=True)
        print("Model: same checkpoint, native metric bypasses ACAE encode/decode", flush=True)
        print("==========================================", flush=True)


if __name__ == "__main__":
    main()
