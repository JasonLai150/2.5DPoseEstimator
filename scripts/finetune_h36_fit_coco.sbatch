#!/bin/bash
#SBATCH --job-name=h36_fit_coco
#SBATCH --output=artifacts/logs/h36_fit_coco_%j.out
#SBATCH --error=artifacts/logs/h36_fit_coco_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=03:00:00
#SBATCH --partition=coe-gpu

set -euo pipefail

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator
mkdir -p artifacts/logs artifacts/checkpoints

echo "=========================================="
echo "Fine-tuning corrected H36M+Fit3D ACAE on COCO ViTPose 2D"
echo "=========================================="

python -u finetune_acae_coco_vitpose.py \
  --coco-2d-dir artifacts/coco_2d/vitpose_train2017 \
  --annotation-file data/coco/annotations/person_keypoints_train2017.json \
  --base-checkpoint artifacts/checkpoints/h36_fit_checkpoint.pth \
  --output artifacts/checkpoints/h36_fit_coco_checkpoint.pth \
  --epochs 10 \
  --batch-size 512 \
  --learning-rate 0.0003 \
  --regul-lambda 0.6 \
  --min-score 0.2 \
  --min-valid-joints 6 \
  --max-poses 200000 \
  --replay-data-dir acae_data_h36_fit \
  --replay-fraction 0.25 \
  --max-replay-poses 50000 \
  --retention-eval-poses 8192 \
  --device cuda

echo ""
echo "=========================================="
echo "Done. Check artifacts/checkpoints/h36_fit_coco_checkpoint.pth"
echo "=========================================="
