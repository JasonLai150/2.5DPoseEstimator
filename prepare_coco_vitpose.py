#!/usr/bin/env python3
"""Prepare COCO 2017 keypoints in the layout expected by ViTPose/MMPose.

Expected output layout:
  data/coco/
    train2017/
    annotations/
      person_keypoints_train2017.json

The script is intentionally conservative: it validates the annotation file and
extracts zip archives only when the expected directories/files are missing.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, output_dir: Path, expected_path: Path) -> None:
    if expected_path.exists():
        print(f"[skip] already present: {expected_path}", flush=True)
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing archive: {zip_path}")

    print(f"[extract] {zip_path} -> {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    if not expected_path.exists():
        raise RuntimeError(f"Extraction finished but expected path is missing: {expected_path}")


def validate_coco_keypoints(annotation_path: Path) -> None:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing COCO keypoint annotation file: {annotation_path}")

    print(f"[check] {annotation_path}", flush=True)
    with annotation_path.open("r") as f:
        data = json.load(f)

    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise RuntimeError(f"{annotation_path} does not look like a COCO annotation file; missing '{key}'")

    person_categories = [cat for cat in data["categories"] if cat.get("name") == "person"]
    if not person_categories:
        raise RuntimeError(f"{annotation_path} has no 'person' category")

    keypoints = person_categories[0].get("keypoints", [])
    if len(keypoints) != 17:
        raise RuntimeError(f"Expected 17 COCO body keypoints, found {len(keypoints)}")

    print(
        f"[ok] images={len(data['images'])}, annotations={len(data['annotations'])}, "
        f"person_keypoints={len(keypoints)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--train-zip", type=Path, default=Path("data/coco/train2017.zip"))
    parser.add_argument("--annotations-zip", type=Path, default=Path("data/coco/annotations_trainval2017.zip"))
    parser.add_argument("--skip-images", action="store_true", help="Only validate/extract annotations.")
    args = parser.parse_args()

    train_dir = args.coco_root / "train2017"
    annotation_file = args.coco_root / "annotations" / "person_keypoints_train2017.json"

    print("==========================================", flush=True)
    print("Preparing COCO 2017 for ViTPose/MMPose", flush=True)
    print(f"COCO root: {args.coco_root}", flush=True)
    print("==========================================", flush=True)

    if not args.skip_images:
        extract_zip(args.train_zip, args.coco_root, train_dir)
    extract_zip(args.annotations_zip, args.coco_root, annotation_file)
    validate_coco_keypoints(annotation_file)

    print("==========================================", flush=True)
    print("COCO ViTPose layout ready", flush=True)
    print(f"Images: {train_dir}", flush=True)
    print(f"Annotations: {annotation_file}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
