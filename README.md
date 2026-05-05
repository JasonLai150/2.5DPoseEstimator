# 2.5D Pose Estimator

A 2.5D approach to training robust 3D human pose estimators in specialized exercise settings — Georgia Tech CS 7643 final project (Spring 2026).

**Authors:** Candice Chen, Alec Cheng, Jason Lai

## What this project is

3D human pose estimation has matured rapidly in lab-captured settings, but every dataset with millimeter-accurate 3D ground truth (Human3.6M, AMASS) was collected in a controlled studio. The videos people actually want analyzed — pushups, deadlifts, burpees — exist mostly in 2D. We ask: **can a 3D pose estimator be adapted to gym/exercise video using only weakly-supervised 2D signals on the target distribution, while preserving its lab-data accuracy?**

We pursued three parallel hybrid 2.5D investigations of cross-domain adaptation, evaluated on the held-out **Fit3D s11** subject and **Human3.6M test**:

1. **VideoPose3D + Fit3D-2D supervision** — a temporal convolutional lifting backbone fine-tuned with Detectron2-extracted Fit3D 2D pseudo-labels, with an optional ACAE skeleton-preprocessing variant.
2. **MotionBERT + LoRA + KD** — a frozen DSTformer (`MB_ft_h36m`) with rank-2 LoRA adapters and a knowledge-distillation regularizer that exposes a cross-domain Pareto knob.
3. **ACAE + COCO substitute** — a VideoPose3D + frozen ACAE skeleton bridge fed by ViTPose 2D detections from COCO, testing whether a diverse 2D substitute can replace target-domain data entirely.

## Headline results

All numbers are on the held-out **Fit3D s11** subject. *Note:* threads use slightly different Fit3D skeleton mappings, so absolute MPJPE is not strictly comparable across rows — only the *relative* gain over each thread's own baseline is.

| Thread | Fit3D MPJPE (zero-shot → fine-tuned) | Δ Fit3D | H36M cost |
|---|---|---|---|
| VideoPose3D + Fit3D-2D (partial) | 874.7 → 407.7 mm | **−53%** | +2.4 mm MPJPE |
| MotionBERT + LoRA + KD λ=1 | 299.4 → 242.1 mm | **−19.2%** | +32 mm P-MPJPE |
| MotionBERT + KD λ=1000 (Pareto endpoint) | 299.4 → 294.5 mm | −1.6% | +4 mm P-MPJPE |
| ACAE + COCO (no Fit3D 2D) | 406.0 → 310.7 mm | **−23.5%** | +113 mm |

**Key finding:** sweeping the knowledge-distillation weight $\lambda_{kd}$ in the MotionBERT thread traces a clean cross-domain Pareto frontier — a single scalar interpolating between max-Fit3D-adaptation and max-H36M-preservation.

## Repo structure

```
src/
├── models/        # DSTformer, LoRA, MotionBERT wrapper, base interface
├── data/          # H36M / Fit3D / gym-video datasets, skeleton converter
├── losses/        # composite PoseLoss (3D + reproj + biomech + KD)
├── metrics/       # MPJPE, P-MPJPE, BLI
├── utils/         # PerspectiveCamera for reprojection
└── training/      # Trainer with W&B logging
scripts/
├── train.py             # training entry point
├── prepare_fit3d.py     # Fit3D preprocessing (IMAR-compatible)
├── evaluate_baseline.py # zero-shot baselines
└── make_report_figures.py
configs/                 # YAML configs (model / data / training)
external/MotionBERT/     # MotionBERT submodule
figures/                 # report figures
```

## Quick start

```bash
# Install
pip install -r requirements.txt

# Prepare Fit3D evaluation data
python scripts/prepare_fit3d.py --data_root ./data/fit3d

# Zero-shot baseline
python scripts/evaluate_baseline.py --model motionbert

# Fine-tune with LoRA + KD (MotionBERT thread)
python scripts/train.py \
  --pretrained checkpoints/motionbert.bin \
  --lora --lora_rank 2 \
  --kd_weight 1.0 \
  --epochs 15

# Resume from checkpoint
python scripts/train.py --resume_from outputs/checkpoints/best.pt
```

## Final report

See `team_report.tex` / `team_report.md` for the full writeup, hypotheses (H1–H3), ablations, per-action analysis, and discussion.

## Acknowledgments

Built on top of [MotionBERT](https://github.com/Walter0807/MotionBERT) (Zhu et al., ICCV 2023), [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) (Pavllo et al., CVPR 2019), the [Fit3D](https://fit3d.imar.ro/) dataset (Fieraru et al., 2021), and the [Affine Combining Autoencoder](https://github.com/isarandi/affine-combining-autoencoder) (Sárándi et al., WACV 2023). Compute provided by Georgia Tech PACE-ICE.
