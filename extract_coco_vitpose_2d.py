#!/usr/bin/env python3
"""Run ViTPose on COCO 2017 train images using COCO person annotations.

This writes one compact `.npz` per image containing the detected person
keypoints. It mirrors the Fit3D extractor style, but uses COCO's person
bounding boxes rather than a full-frame box.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from run_paths import COCO_2D_DIR, REPO_ROOT, VITPOSE_B_CHECKPOINT_PATH, ensure_artifact_dirs


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
DEFAULT_OUTPUT_DIR = COCO_2D_DIR / "vitpose_train2017"

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


def import_vitpose(vitpose_root: Path):
    root = vitpose_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"ViTPose repo not found: {root}")
    sys.path.insert(0, str(root))

    try:
        from mmpose.apis import inference_top_down_pose_model, init_pose_model
        from mmpose.datasets import DatasetInfo
        from xtcocotools.coco import COCO
    except Exception as exc:
        raise RuntimeError(
            "Could not import ViTPose/MMPose. Make sure the ViTPose env is active "
            "and the compatibility stack is installed."
        ) from exc

    return inference_top_down_pose_model, init_pose_model, DatasetInfo, COCO


def validate_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing ViTPose checkpoint: {path}")
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Checkpoint file looks invalid ({path}, {path.stat().st_size} bytes)."
        )


def output_path_for(out_dir: Path, image_id: int, file_name: str) -> Path:
    rel = Path(file_name)
    return out_dir / rel.parent / f"{rel.stem}.npz"


def run_vitpose_on_image(
    pose_model,
    image_path: Path,
    person_results: List[Dict],
    dataset: str,
    dataset_info,
    inference_top_down_pose_model,
):
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        str(image_path),
        person_results,
        bbox_thr=None,
        format="xywh",
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None,
    )
    return pose_results


def save_image_keypoints(
    out_path: Path,
    image_id: int,
    file_name: str,
    pose_results: List[Dict],
    bbox_count: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pose_results:
        keypoints = np.stack([np.asarray(p["keypoints"], dtype=np.float32)[:, :2] for p in pose_results])
        scores = np.stack([np.asarray(p["keypoints"], dtype=np.float32)[:, 2] for p in pose_results])
        bboxes = np.stack([np.asarray(p.get("bbox", np.zeros(4)), dtype=np.float32) for p in pose_results])
    else:
        keypoints = np.zeros((0, 17, 2), dtype=np.float32)
        scores = np.zeros((0, 17), dtype=np.float32)
        bboxes = np.zeros((0, 4), dtype=np.float32)

    np.savez_compressed(
        out_path,
        image_id=np.int64(image_id),
        file_name=np.asarray(file_name),
        layout_name="coco",
        keypoint_names=np.asarray(COCO_KEYPOINTS),
        keypoints=keypoints,
        scores=scores,
        bboxes=bboxes,
        num_persons=np.int64(len(pose_results)),
        num_ann_bboxes=np.int64(bbox_count),
    )


def write_index(out_dir: Path, records: List[Dict]) -> None:
    index = {
        "layout_name": "coco",
        "num_joints": 17,
        "keypoint_names": COCO_KEYPOINTS,
        "bbox_source": "coco person annotations",
        "format": "one npz per image",
        "images": records,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--annotation-file", type=Path, default=Path("data/coco/annotations/person_keypoints_train2017.json"))
    parser.add_argument("--vitpose-root", type=Path, default=DEFAULT_VITPOSE_ROOT)
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_VITPOSE_CONFIG)
    parser.add_argument("--pose-checkpoint", type=Path, default=VITPOSE_B_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ensure_artifact_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_checkpoint(args.pose_checkpoint)

    inference_fn, init_pose_model, DatasetInfo, COCO = import_vitpose(args.vitpose_root)

    print("==========================================", flush=True)
    print("Running ViTPose on COCO train2017", flush=True)
    print(f"Images: {args.coco_root / 'train2017'}", flush=True)
    print(f"Annotations: {args.annotation_file}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    print("BBox source: COCO person annotations", flush=True)
    print("==========================================", flush=True)

    coco = COCO(str(args.annotation_file))
    pose_model = init_pose_model(str(args.pose_config), str(args.pose_checkpoint), device=args.device.lower())
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn("ViTPose config has no dataset_info; continuing with dataset name only.")
    else:
        dataset_info = DatasetInfo(dataset_info)

    img_ids = list(coco.imgs.keys())
    records = []
    processed = 0
    skipped = 0

    for i, image_id in enumerate(img_ids):
        image = coco.loadImgs(image_id)[0]
        image_path = args.coco_root / "train2017" / image["file_name"]
        if not image_path.exists():
            skipped += 1
            continue

        out_path = output_path_for(args.output_dir, image_id, image["file_name"])
        if out_path.exists() and args.skip_existing:
            skipped += 1
            records.append({"image_id": image_id, "file_name": image["file_name"], "output": str(out_path), "status": "skipped_existing"})
            continue

        ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[1], iscrowd=False)
        anns = [coco.anns[ann_id] for ann_id in ann_ids]
        person_results = []
        for ann in anns:
            bbox = np.asarray(ann["bbox"], dtype=np.float32)
            if bbox.shape != (4,):
                continue
            person_results.append({"bbox": bbox})

        pose_results = []
        if person_results:
            pose_results = run_vitpose_on_image(
                pose_model,
                image_path,
                person_results,
                dataset,
                dataset_info,
                inference_fn,
            )

        save_image_keypoints(out_path, image_id, image["file_name"], pose_results, len(person_results))
        records.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "num_persons": int(len(pose_results)),
                "num_ann_bboxes": int(len(person_results)),
                "output": str(out_path),
                "status": "processed",
            }
        )
        processed += 1

        if processed % 1000 == 0:
            print(f"  processed {processed} images...", flush=True)
        if args.max_images and processed >= args.max_images:
            break

    write_index(args.output_dir, records)

    print("==========================================", flush=True)
    print(f"Processed images: {processed}", flush=True)
    print(f"Skipped images: {skipped}", flush=True)
    print(f"Index: {args.output_dir / 'index.json'}", flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
