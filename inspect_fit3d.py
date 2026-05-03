import argparse
import json
import os
import tarfile
from collections import Counter

import numpy as np

import sys

sys.path.append(os.path.abspath("acae_2D_extension"))

from prepare_h36m_fit3d import build_unified_skeleton


def summarize_fit3d(tar_path, sample_count=5):
    unified_names, h36m_to_unified, body25_to_unified = build_unified_skeleton()

    total_members = 0
    total_json = 0
    total_images = 0
    total_videos = 0
    total_pose_entries = 0
    total_frames = 0
    zero_joint_frames = 0
    finite_violations = 0
    subjects = Counter()
    lengths = []
    sample_rows = []
    other_json_samples = []
    candidate_2d_members = []
    candidate_video_members = []
    candidate_other_assets = []

    two_d_tokens = (
        "2d",
        "keypoint",
        "keypoints",
        "pose2d",
        "openpose",
        "coco",
        "detectron",
        "mmpose",
        "alphapose",
        "cpn",
        "bbox",
    )
    video_tokens = (".mp4", ".mov", ".avi", ".mkv")
    image_tokens = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    coord_min = np.full(3, np.inf, dtype=np.float64)
    coord_max = np.full(3, -np.inf, dtype=np.float64)

    with tarfile.open(tar_path, "r:gz") as tf:
        while True:
            try:
                member = tf.next()
            except EOFError:
                print("Warning: Fit3D tar.gz ended early while streaming; reporting partial archive stats.")
                break

            if member is None:
                break

            total_members += 1
            lower_name = member.name.lower()

            if any(token in lower_name for token in image_tokens):
                total_images += 1
            if any(lower_name.endswith(token) for token in video_tokens):
                total_videos += 1

            if any(token in lower_name for token in two_d_tokens):
                if len(candidate_2d_members) < sample_count:
                    candidate_2d_members.append(member.name)
            if any(lower_name.endswith(token) for token in video_tokens):
                if len(candidate_video_members) < sample_count:
                    candidate_video_members.append(member.name)
            if (
                any(token in lower_name for token in two_d_tokens)
                or any(lower_name.endswith(token) for token in video_tokens)
                or any(token in lower_name for token in image_tokens)
            ):
                if len(candidate_other_assets) < sample_count:
                    candidate_other_assets.append(member.name)

            if not member.isfile() or not member.name.endswith(".json"):
                continue

            total_json += 1
            parts = member.name.split("/")
            subject = parts[1] if len(parts) >= 2 else "unknown"
            subjects[subject] += 1

            extracted = tf.extractfile(member)
            if extracted is None:
                continue

            data = json.load(extracted)
            keys = sorted(data.keys())
            if "joints3d_25" not in data:
                if len(other_json_samples) < sample_count:
                    other_json_samples.append((member.name, keys))
                continue

            poses = np.asarray(data["joints3d_25"], dtype=np.float32)
            if poses.ndim != 3 or poses.shape[1] != 25 or poses.shape[2] != 3:
                raise ValueError(
                    f"Unexpected shape in {member.name}: {poses.shape}, expected (T, 25, 3)"
                )

            total_pose_entries += 1
            T = poses.shape[0]
            lengths.append(T)
            total_frames += T

            finite_mask = np.isfinite(poses)
            if not finite_mask.all():
                finite_violations += int((~finite_mask).any(axis=(1, 2)).sum())

            zero_joint_mask = np.all(np.isclose(poses, 0.0), axis=-1)
            zero_joint_frames += int(zero_joint_mask.sum())

            flat = poses.reshape(-1, 3)
            coord_min = np.minimum(coord_min, flat.min(axis=0))
            coord_max = np.maximum(coord_max, flat.max(axis=0))

            if len(sample_rows) < sample_count:
                sample_rows.append(
                    {
                        "member": member.name,
                        "shape": tuple(poses.shape),
                        "keys": keys,
                        "frame0_min": float(poses[0].min()),
                        "frame0_max": float(poses[0].max()),
                        "first_joint": poses[0, 0].round(6).tolist(),
                    }
                )

    lengths_arr = np.asarray(lengths, dtype=np.int64) if lengths else np.array([], dtype=np.int64)
    zero_joint_rate = (zero_joint_frames / (total_frames * 25)) if total_frames else 0.0

    print(f"Archive: {tar_path}")
    if os.path.exists(tar_path):
        print(f"File size: {os.path.getsize(tar_path) / (1024 ** 3):.2f} GiB")
    print(f"Tar members: {total_members}")
    print(f"JSON members: {total_json}")
    print(f"Image members: {total_images}")
    print(f"Video members: {total_videos}")
    print(f"Pose sequences found: {total_pose_entries}")
    print(f"Subjects seen: {dict(subjects)}")
    print(f"Sequence length stats: count={len(lengths)} min={lengths_arr.min() if len(lengths) else 0} max={lengths_arr.max() if len(lengths) else 0} mean={lengths_arr.mean() if len(lengths) else 0:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Coordinate range (meters, as loaded): min={coord_min.tolist()} max={coord_max.tolist()}")
    print(f"Zero-joint frames: {zero_joint_frames} of {total_frames * 25 if total_frames else 0} joint slots ({zero_joint_rate:.6f})")
    print(f"Non-finite frame count: {finite_violations}")
    print("")
    print("2D / media scan:")
    print(f"  candidate 2D-ish members found: {len(candidate_2d_members)}")
    print(f"  candidate video members found: {len(candidate_video_members)}")
    if candidate_2d_members:
        print("  sample 2D-ish member names:")
        for name in candidate_2d_members:
            print(f"    {name}")
    if candidate_video_members:
        print("  sample video member names:")
        for name in candidate_video_members:
            print(f"    {name}")
    if candidate_other_assets:
        print("  sample media-related member names:")
        for name in candidate_other_assets:
            print(f"    {name}")
    print("")
    print("Unified skeleton used by ACAE:")
    print(f"  joint count: {len(unified_names)}")
    print(f"  names: {unified_names}")
    print(f"  H36M -> unified: {h36m_to_unified}")
    print(f"  BODY25 -> unified: {body25_to_unified}")
    print("")
    print("Sample pose files:")
    for row in sample_rows:
        print(f"  {row['member']}")
        print(f"    shape: {row['shape']}")
        print(f"    top-level keys: {row['keys']}")
        print(f"    frame0 range: {row['frame0_min']:.6f} .. {row['frame0_max']:.6f}")
        print(f"    frame0 first joint: {row['first_joint']}")

    if other_json_samples:
        print("")
        print("Other JSON entries encountered:")
        for member_name, keys in other_json_samples:
            print(f"  {member_name}: keys={keys}")

    print("")
    print("Interpretation:")
    print("  - The Fit3D archive in this repo is 3D pose data, not 2D keypoints.")
    print("  - The canonical pose payload is `joints3d_25` with shape (T, 25, 3).")
    print("  - Values are loaded as meters.")
    print("  - Any 2D input for VideoPose3D must come from detections or projection, not from this archive directly.")
    if candidate_2d_members or candidate_video_members:
        print("  - The archive appears to contain media or 2D-related files; inspect the sample names above.")
    else:
        print("  - The archive scan did not find obvious 2D keypoint or video/image assets by filename.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit3d-path",
        default="data/fit3d_train.tar.gz",
        help="Path to the Fit3D tar.gz archive.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of sample sequences and non-pose entries to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summarize_fit3d(args.fit3d_path, sample_count=args.sample_count)


if __name__ == "__main__":
    main()
