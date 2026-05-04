#!/usr/bin/env python3
"""Convert COCO 2017 person keypoints into ACAE unified-skeleton tensors.

COCO is 2D-only. We store x/y as normalized screen coordinates scaled by 1000
and set z=1000 for present joints, which routes the existing ACAE loss through
its projected-2D path. Missing joints remain all-zero, matching the ACAE mask.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from run_paths import ACAE_CHECKPOINT_PATH, COCO_ACAE_DIR, REPO_ROOT, ensure_artifact_dirs, first_existing_path

sys.path.insert(0, str(REPO_ROOT / "acae_2D_extension"))
from prepare_h36m_fit3d import build_unified_skeleton  # noqa: E402


COCO_NAMES = [
    "nose",
    "leye",
    "reye",
    "lear",
    "rear",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
    "lhip",
    "rhip",
    "lknee",
    "rknee",
    "lankle",
    "rankle",
]

DIRECT_COCO_TO_UNIFIED = {
    "nose": "nose",
    "leye": "leye",
    "reye": "reye",
    "lear": "lear",
    "rear": "rear",
    "lshoulder": "lshoulder",
    "rshoulder": "rshoulder",
    "lelbow": "lelbow",
    "relbow": "relbow",
    "lwrist": "lwrist",
    "rwrist": "rwrist",
    "lhip": "lhip",
    "rhip": "rhip",
    "lknee": "lknee",
    "rknee": "rknee",
    "lankle": "lankle",
    "rankle": "rankle",
}


def normalize_xy(xy: np.ndarray, width: float, height: float) -> np.ndarray:
    """Match VideoPose3D-style screen normalization: x in [-1, 1]."""
    return xy / width * 2.0 - np.array([1.0, height / width], dtype=np.float32)


def add_joint(
    pose: np.ndarray,
    scores: np.ndarray,
    unified_index: Dict[str, int],
    name: str,
    xy_norm: np.ndarray,
    score: float,
) -> None:
    if name not in unified_index:
        return
    idx = unified_index[name]
    pose[idx, 0] = xy_norm[0] * 1000.0
    pose[idx, 1] = xy_norm[1] * 1000.0
    pose[idx, 2] = 1000.0
    scores[idx] = score


def midpoint(
    pose: np.ndarray,
    scores: np.ndarray,
    unified_index: Dict[str, int],
    out_name: str,
    left_name: str,
    right_name: str,
) -> None:
    if out_name not in unified_index or left_name not in unified_index or right_name not in unified_index:
        return
    out_idx = unified_index[out_name]
    left_idx = unified_index[left_name]
    right_idx = unified_index[right_name]
    if scores[left_idx] <= 0.0 or scores[right_idx] <= 0.0:
        return
    pose[out_idx] = (pose[left_idx] + pose[right_idx]) * 0.5
    pose[out_idx, 2] = 1000.0
    scores[out_idx] = min(scores[left_idx], scores[right_idx])


def weighted_midpoint(
    pose: np.ndarray,
    scores: np.ndarray,
    unified_index: Dict[str, int],
    out_name: str,
    a_name: str,
    b_name: str,
    alpha: float,
) -> None:
    if out_name not in unified_index or a_name not in unified_index or b_name not in unified_index:
        return
    out_idx = unified_index[out_name]
    a_idx = unified_index[a_name]
    b_idx = unified_index[b_name]
    if scores[a_idx] <= 0.0 or scores[b_idx] <= 0.0:
        return
    pose[out_idx] = pose[a_idx] * (1.0 - alpha) + pose[b_idx] * alpha
    pose[out_idx, 2] = 1000.0
    scores[out_idx] = min(scores[a_idx], scores[b_idx])


def annotation_to_pose(
    ann: Dict,
    image_info: Dict,
    unified_names: List[str],
    visibility: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    keypoints = np.asarray(ann.get("keypoints", []), dtype=np.float32).reshape(-1, 3)
    if keypoints.shape != (17, 3):
        return None, None

    width = float(image_info["width"])
    height = float(image_info["height"])
    if width <= 0 or height <= 0:
        return None, None

    unified_index = {name: i for i, name in enumerate(unified_names)}
    pose = np.zeros((len(unified_names), 3), dtype=np.float32)
    scores = np.zeros((len(unified_names),), dtype=np.float32)

    min_v = 2 if visibility == "visible" else 1
    for coco_idx, coco_name in enumerate(COCO_NAMES):
        x, y, v = keypoints[coco_idx]
        if v < min_v:
            continue
        unified_name = DIRECT_COCO_TO_UNIFIED[coco_name]
        xy_norm = normalize_xy(np.array([x, y], dtype=np.float32), width, height)
        add_joint(pose, scores, unified_index, unified_name, xy_norm, score=float(v) / 2.0)

    midpoint(pose, scores, unified_index, "pelvis", "lhip", "rhip")
    midpoint(pose, scores, unified_index, "neck", "lshoulder", "rshoulder")
    weighted_midpoint(pose, scores, unified_index, "spine", "pelvis", "neck", alpha=0.45)
    weighted_midpoint(pose, scores, unified_index, "thorax", "pelvis", "neck", alpha=0.75)

    if np.count_nonzero(scores > 0.0) == 0:
        return None, None
    return pose, scores


def load_annotations(annotation_path: Path) -> Tuple[Dict[int, Dict], Iterable[Dict]]:
    with annotation_path.open("r") as f:
        data = json.load(f)
    images = {int(img["id"]): img for img in data["images"]}
    return images, data["annotations"]


def load_acae(checkpoint_path: Path, device: str):
    import torch

    acae_path = REPO_ROOT / "acae_2D_extension" / "affine_combining_autoencoder" / "acae_2.5d_torch.py"
    spec = importlib.util.spec_from_file_location("acae_torch", acae_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_acae_from_checkpoint(str(checkpoint_path), device=device, freeze=True)


def encode_with_acae(poses: np.ndarray, checkpoint_path: Path, device: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    model = load_acae(checkpoint_path, device)
    latents = []
    with torch.no_grad():
        for start in range(0, len(poses), batch_size):
            batch = torch.from_numpy(poses[start : start + batch_size].astype(np.float32)).to(device)
            latent = model.encode(batch).detach().cpu().numpy()
            latents.append(latent)
    latent_3d = np.concatenate(latents, axis=0) if latents else np.zeros((0, 17, 3), dtype=np.float32)
    latent_2d = latent_3d[..., :2] / 1000.0
    return latent_3d.astype(np.float32), latent_2d.astype(np.float32)


def save_split(output_dir: Path, name: str, poses: np.ndarray, scores: np.ndarray) -> None:
    np.save(output_dir / f"poses_{name}.npy", poses)
    np.save(output_dir / f"scores_{name}.npy", scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-file", type=Path, default=Path("data/coco/annotations/person_keypoints_train2017.json"))
    parser.add_argument("--output-dir", type=Path, default=COCO_ACAE_DIR)
    parser.add_argument("--visibility", choices=["labeled", "visible"], default="labeled")
    parser.add_argument("--min-joints", type=int, default=8)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encode", action="store_true", help="Also run the trained ACAE encoder.")
    parser.add_argument("--acae-checkpoint", type=Path, default=ACAE_CHECKPOINT_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    ensure_artifact_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    unified_names, h36m_to_unified, _ = build_unified_skeleton()
    images, annotations = load_annotations(args.annotation_file)

    poses = []
    scores = []
    kept = 0
    skipped = 0

    print("==========================================", flush=True)
    print("Converting COCO keypoints to ACAE unified skeleton", flush=True)
    print(f"Annotations: {args.annotation_file}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    print("==========================================", flush=True)

    for ann in annotations:
        if ann.get("iscrowd", 0):
            skipped += 1
            continue
        if ann.get("num_keypoints", 0) < args.min_joints:
            skipped += 1
            continue
        image_info = images.get(int(ann["image_id"]))
        if image_info is None:
            skipped += 1
            continue
        pose, score = annotation_to_pose(ann, image_info, unified_names, args.visibility)
        if pose is None or int((score > 0).sum()) < args.min_joints:
            skipped += 1
            continue
        poses.append(pose)
        scores.append(score)
        kept += 1
        if args.max_instances and kept >= args.max_instances:
            break
        if kept % 50000 == 0:
            print(f"  kept {kept} person instances...", flush=True)

    if not poses:
        raise RuntimeError("No COCO poses were converted. Check annotation path and filters.")

    poses = np.stack(poses).astype(np.float32)
    scores = np.stack(scores).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(poses))
    split = int(round(len(indices) * (1.0 - args.test_split)))
    train_idx = indices[:split]
    test_idx = indices[split:]

    poses_train = poses[train_idx]
    poses_test = poses[test_idx]
    scores_train = scores[train_idx]
    scores_test = scores[test_idx]

    save_split(args.output_dir, "train", poses_train, scores_train)
    save_split(args.output_dir, "test", poses_test, scores_test)
    np.save(args.output_dir / "joint_names.npy", np.asarray(unified_names))
    np.save(args.output_dir / "h36m_joint_indices.npy", np.asarray(h36m_to_unified, dtype=np.int64))

    h36m_mask = np.zeros(len(unified_names), dtype=bool)
    h36m_mask[np.asarray(h36m_to_unified, dtype=np.int64)] = True
    np.save(args.output_dir / "h36m_joint_mask.npy", h36m_mask)

    metadata = {
        "source": str(args.annotation_file),
        "coordinate_space": "VideoPose3D normalized screen coordinates scaled by 1000; z=1000 for present joints",
        "visibility": args.visibility,
        "min_joints": args.min_joints,
        "kept_instances": int(kept),
        "skipped_annotations": int(skipped),
        "train_shape": list(poses_train.shape),
        "test_shape": list(poses_test.shape),
        "joint_names": unified_names,
    }

    if args.encode:
        checkpoint = first_existing_path(args.acae_checkpoint, "artifacts/checkpoints/h36_fit_checkpoint.pth")
        latent_train_3d, latent_train_2d = encode_with_acae(poses_train, checkpoint, args.device, args.batch_size)
        latent_test_3d, latent_test_2d = encode_with_acae(poses_test, checkpoint, args.device, args.batch_size)
        np.save(args.output_dir / "latent_train_3d.npy", latent_train_3d)
        np.save(args.output_dir / "latent_test_3d.npy", latent_test_3d)
        np.save(args.output_dir / "latent_train_2d.npy", latent_train_2d)
        np.save(args.output_dir / "latent_test_2d.npy", latent_test_2d)
        metadata["acae_checkpoint"] = str(checkpoint)
        metadata["latent_train_2d_shape"] = list(latent_train_2d.shape)
        metadata["latent_test_2d_shape"] = list(latent_test_2d.shape)

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print("==========================================", flush=True)
    print(f"Kept instances: {kept}", flush=True)
    print(f"Skipped annotations: {skipped}", flush=True)
    print(f"Train: {poses_train.shape}", flush=True)
    print(f"Test: {poses_test.shape}", flush=True)
    if args.encode:
        print(f"Latent train 2D: {metadata['latent_train_2d_shape']}", flush=True)
    print(f"Saved: {args.output_dir}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
