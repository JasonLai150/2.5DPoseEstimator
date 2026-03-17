# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 3D human pose estimator bridging the domain gap between clean lab-captured 3D data and high-occlusion in-the-wild fitness video data. Uses a hybrid 2.5D training loop with weakly-supervised reprojection and biomechanical constraints.

## Architecture

**Model**: DSTformer (Dual-Stream Spatio-Temporal Transformer)
- Adapted from MotionBERT architecture
- Dual parallel streams: spatial→temporal (ST) and temporal→spatial (TS)
- Learned attention fusion between streams
- Pre-trained on AMASS via Masked Pose Modeling (MPM) for anatomical priors
- Fine-tuned with LoRA in attention layers (`qkv`, `proj`) to handle gym occlusions while preventing catastrophic forgetting
- Input: temporal sequence of 2D joint coordinates (B, T, J, 2)
- Output: 3D joint coordinates in root-relative camera frame (B, T, J, 3)

## Data Pipeline

**Datasets**:
- Human3.6M: 3D supervised data (geometric ground truth and scale)
- In-the-wild gym videos: 2D weakly-supervised data with YOLO-Pose/OpenPose pseudo-labels
- Fit3D: Evaluation data with 3D ground truth for fitness movements (25-joint COCO → 17-joint H36M)

**Skeleton Mapper**: Critical preprocessing component that reconciles skeleton format differences (e.g., COCO to Human3.6M) before loss calculation. Handles:
- Index reordering between formats
- Dropping incompatible joints (facial keypoints)
- Interpolating missing nodes (pelvis, spine)

## Loss Formulation

`Total Loss = λ1*L_3D + λ2*L_reproj + λ3*L_biomech`

- **L_3D**: MPJPE between predictions and Human3.6M ground truth
- **L_reproj**: Reprojection loss on gym videos - project 3D→2D via perspective camera, compare to 2D pseudo-labels
- **L_biomech**: Regularizer against depth cheating
  - Bilateral symmetry: equal bone lengths left/right
  - Anatomical hinge: penalize impossible rotations (hyperextension)

## Evaluation Metrics

- **MPJPE / P-MPJPE**: Millimetric 3D error (primary metric on Fit3D)
- **BLI (Bilateral Length Inconsistency)**: Variance in bone lengths for skeletal realism

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Baseline evaluation (pretrained models on Fit3D)
python scripts/prepare_fit3d.py --data_root ./data/fit3d
python scripts/evaluate_baseline.py --model motionbert

# Training
python scripts/train.py
python scripts/train.py --epochs 200 --lr 1e-4

# Training with LoRA fine-tuning
python scripts/train.py --pretrained checkpoints/motionbert.bin --lora --lora_rank 8

# Resume from checkpoint
python scripts/train.py --resume_from outputs/checkpoints/best.pt
```

## Project Structure

```
src/
├── models/
│   ├── base.py         # PoseEstimatorBase - abstract model interface
│   ├── dstformer.py    # DSTformer - main model with LoRA support
│   ├── lora.py         # LoRALinear, apply_lora_to_model
│   └── pretrained.py   # MotionBERTWrapper, APTPoseWrapper
├── data/
│   ├── skeleton.py     # SkeletonConverter, SKELETON_CONFIGS
│   └── datasets.py     # H36MDataset, GymVideoDataset, Fit3DDataset
├── losses/pose_losses.py  # PoseLoss composite, mpjpe_loss, reprojection_loss, biomech losses
├── metrics/pose_metrics.py # compute_mpjpe, compute_p_mpjpe, compute_bli
├── utils/camera.py     # PerspectiveCamera for reprojection
├── config.py           # Config loading (YAML-based)
└── training/trainer.py # Trainer class with W&B logging
scripts/
├── train.py            # Training entry point
├── prepare_fit3d.py    # Fit3D data preprocessing (IMAR tools compatible)
└── evaluate_baseline.py # Baseline evaluation
configs/                # YAML configs (model/, data/, training/)
external/MotionBERT/    # Cloned MotionBERT repo
```

## Key Implementation Notes

- `DSTformer` is the main model - use `create_model(cfg)` factory function
- LoRA targets `qkv` and `proj` layers in attention blocks by default
- Skeleton conversion: `SkeletonConverter.convert()` handles COCO→H36M with pelvis interpolation
- Fit3D uses 25-joint COCO format, converted via `coco25_to_h36m17()` in prepare_fit3d.py
- `PoseLoss.forward()` returns dict with individual loss components for logging
- Camera intrinsics in `PerspectiveCamera` default to (fx=fy=1000, cx=cy=512)
