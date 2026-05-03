#!/usr/bin/env python3
"""Extract Fit3D 2D keypoints with ViTPose using full-frame person boxes.

Fit3D ships videos plus 3D/camera data in this repo, but VideoPose3D expects
2D keypoints as input. This script creates those 2D keypoints from the videos.
The default bbox is the whole frame, matching the ViTPose full-frame demo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from run_paths import FIT3D_2D_DIR, REPO_ROOT, VITPOSE_B_CHECKPOINT_PATH, ensure_artifact_dirs


DEFAULT_VITPOSE_ROOT = REPO_ROOT / "external" / "ViTPose_fitness"
DEFAULT_VITPOSE_CONFIG = (
    DEFAULT_VITPOSE_ROOT
    / "configs"
    / "body"
    / "2d_kpt_sview_rgb_img"
    / "topdown_heatmap"
    / "coco"
    / "ViTPose_base_coco_256x192.py"
)
DEFAULT_OUTPUT_DIR = FIT3D_2D_DIR / "vitpose_fullframe"

COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass
class VideoMember:
    member_name: str
    split: str
    subject: str
    camera: str
    action: str


def parse_csv_filter(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def import_vitpose(vitpose_root: Path):
    root = vitpose_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"ViTPose repo not found: {root}")
    sys.path.insert(0, str(root))

    try:
        from mmpose.apis import inference_top_down_pose_model, init_pose_model
        from mmpose.datasets import DatasetInfo
    except Exception as exc:
        raise RuntimeError(
            "Could not import ViTPose/MMPose. Install the cloned repo and its "
            "OpenMMLab dependencies in your Slurm environment, then rerun."
        ) from exc

    return inference_top_down_pose_model, init_pose_model, DatasetInfo


def validate_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ViTPose-B checkpoint: {path}\n"
            "Place the COCO ViTPose-B 256x192 .pth file there before running."
        )
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Checkpoint file is too small to be valid ({path}, {path.stat().st_size} bytes). "
            "The current file is probably a failed download placeholder."
        )


def parse_fit3d_video_member(member_name: str) -> Optional[VideoMember]:
    path = PurePosixPath(member_name)
    if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        return None

    parts = path.parts
    if "videos" not in parts:
        return None

    idx = parts.index("videos")
    if idx < 1 or idx + 2 >= len(parts):
        return None

    return VideoMember(
        member_name=member_name,
        split=parts[0],
        subject=parts[idx - 1],
        camera=parts[idx + 1],
        action=path.stem,
    )


def selected(record: VideoMember, subjects: Optional[set[str]], cameras: Optional[set[str]], actions: Optional[set[str]]) -> bool:
    return (
        (subjects is None or record.subject in subjects)
        and (cameras is None or record.camera in cameras)
        and (actions is None or record.action in actions)
    )


def output_path_for(record: VideoMember, output_dir: Path) -> Path:
    return output_dir / "videos" / record.split / record.subject / record.camera / f"{record.action}.npz"


def extract_member_to_temp(tf: tarfile.TarFile, member: tarfile.TarInfo, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = PurePosixPath(member.name).suffix or ".mp4"
    handle = tf.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Could not open archive member: {member.name}")

    fd, tmp_name = tempfile.mkstemp(prefix="fit3d_", suffix=suffix, dir=str(tmp_dir))
    tmp_path = Path(tmp_name)
    with handle, open(fd, "wb", closefd=True) as dst:
        shutil.copyfileobj(handle, dst, length=1024 * 1024)
    return tmp_path


def init_pose_context(config_path: Path, checkpoint_path: Path, device: str, vitpose_root: Path):
    inference_top_down_pose_model, init_pose_model, DatasetInfo = import_vitpose(vitpose_root)

    pose_model = init_pose_model(str(config_path), str(checkpoint_path), device=device.lower())
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn("ViTPose config has no dataset_info; continuing with dataset name only.")
    else:
        dataset_info = DatasetInfo(dataset_info)

    return pose_model, dataset, dataset_info, inference_top_down_pose_model


def run_vitpose_on_video(
    video_path: Path,
    pose_model,
    dataset: str,
    dataset_info,
    inference_top_down_pose_model,
    sample_stride: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open extracted video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bbox = np.array([0, 0, width, height], dtype=np.float32)

    keypoints = []
    scores = []
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if sample_stride > 1 and frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frame,
            [{"bbox": bbox.copy()}],
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        if pose_results:
            kpt = np.asarray(pose_results[0]["keypoints"], dtype=np.float32)
            keypoints.append(kpt[:, :2])
            scores.append(kpt[:, 2])
        else:
            keypoints.append(np.full((17, 2), np.nan, dtype=np.float32))
            scores.append(np.zeros((17,), dtype=np.float32))

        frame_idx += 1

    cap.release()

    if not keypoints:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    meta = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames_reported": total_frames,
        "frames_read": frame_idx,
        "frames_saved": len(keypoints),
        "sample_stride": sample_stride,
        "bbox_xyxy": bbox.tolist(),
    }
    return np.stack(keypoints), np.stack(scores), meta


def save_video_keypoints(record: VideoMember, out_path: Path, keypoints: np.ndarray, scores: np.ndarray, meta: Dict[str, float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keypoints=keypoints,
        scores=scores,
        source_member=record.member_name,
        split=record.split,
        subject=record.subject,
        camera=record.camera,
        action=record.action,
        layout_name="coco",
        keypoint_names=np.asarray(COCO_KEYPOINTS),
        **{f"meta_{key}": value for key, value in meta.items()},
    )


def write_index(output_dir: Path, records: Sequence[Dict]) -> None:
    index = {
        "layout_name": "coco",
        "num_joints": 17,
        "keypoint_names": COCO_KEYPOINTS,
        "bbox_mode": "full_frame",
        "format": "one npz per video with keypoints=(T,17,2), scores=(T,17)",
        "videos": list(records),
    }
    index_path = output_dir / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))


def iter_tar_members(tf: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    while True:
        try:
            member = tf.next()
        except EOFError:
            print("WARNING: hit EOFError while reading Fit3D archive; keeping completed outputs.", flush=True)
            break
        if member is None:
            break
        yield member


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit3d-path", type=Path, default=REPO_ROOT / "data" / "fit3d_train.tar.gz")
    parser.add_argument("--vitpose-root", type=Path, default=DEFAULT_VITPOSE_ROOT)
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_VITPOSE_CONFIG)
    parser.add_argument("--pose-checkpoint", type=Path, default=VITPOSE_B_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp") / "fit3d_vitpose_video_extract")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subjects", help="Comma-separated subject filter, e.g. s03,s05")
    parser.add_argument("--cameras", help="Comma-separated camera filter, e.g. 60457274")
    parser.add_argument("--actions", help="Comma-separated action filter, e.g. squat,deadlift")
    parser.add_argument("--max-videos", type=int, help="Limit videos for smoke tests.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Save every Nth frame.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute outputs that already exist.")
    parser.add_argument("--keep-temp-video", action="store_true")
    args = parser.parse_args()

    ensure_artifact_dirs()
    validate_checkpoint(args.pose_checkpoint)
    if not args.fit3d_path.exists():
        raise FileNotFoundError(f"Fit3D archive not found: {args.fit3d_path}")
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    subjects = parse_csv_filter(args.subjects)
    cameras = parse_csv_filter(args.cameras)
    actions = parse_csv_filter(args.actions)

    print("==========================================", flush=True)
    print("Extracting Fit3D 2D keypoints with ViTPose-B", flush=True)
    print(f"Fit3D archive: {args.fit3d_path}", flush=True)
    print(f"ViTPose root: {args.vitpose_root}", flush=True)
    print(f"Pose config: {args.pose_config}", flush=True)
    print(f"Pose checkpoint: {args.pose_checkpoint}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print("BBox mode: full frame", flush=True)
    print("==========================================", flush=True)

    pose_model, dataset, dataset_info, inference_fn = init_pose_context(
        args.pose_config, args.pose_checkpoint, args.device, args.vitpose_root
    )

    processed_records = []
    seen = 0
    matched = 0
    skipped = 0

    with tarfile.open(args.fit3d_path, "r:gz") as tf:
        for member in iter_tar_members(tf):
            if not member.isfile():
                continue
            record = parse_fit3d_video_member(member.name)
            if record is None or not selected(record, subjects, cameras, actions):
                continue

            matched += 1
            out_path = output_path_for(record, args.output_dir)
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {record.member_name} -> {out_path}", flush=True)
                skipped += 1
                processed_records.append({**asdict(record), "output": str(out_path), "status": "skipped_existing"})
                if args.max_videos and matched >= args.max_videos:
                    break
                continue

            print(f"[run] {record.member_name}", flush=True)
            tmp_video = extract_member_to_temp(tf, member, args.tmp_dir)
            try:
                keypoints, scores, meta = run_vitpose_on_video(
                    tmp_video,
                    pose_model,
                    dataset,
                    dataset_info,
                    inference_fn,
                    args.sample_stride,
                )
                save_video_keypoints(record, out_path, keypoints, scores, meta)
                processed_records.append(
                    {**asdict(record), "output": str(out_path), "status": "processed", **meta}
                )
                processed_records.sort(key=lambda item: (item["split"], item["subject"], item["camera"], item["action"]))
                write_index(args.output_dir, processed_records)
                seen += 1
                print(f"[ok] saved {keypoints.shape} -> {out_path}", flush=True)
            finally:
                if not args.keep_temp_video:
                    tmp_video.unlink(missing_ok=True)

            if args.max_videos and matched >= args.max_videos:
                break

    write_index(args.output_dir, processed_records)
    print("==========================================", flush=True)
    print(f"Matched videos: {matched}", flush=True)
    print(f"Processed videos: {seen}", flush=True)
    print(f"Skipped existing: {skipped}", flush=True)
    print(f"Index: {args.output_dir / 'index.json'}", flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
