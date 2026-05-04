#!/bin/bash
#SBATCH --job-name=vdp_h36_coco
#SBATCH --output=artifacts/logs/vdp_h36_fit_coco_%j.out
#SBATCH --error=artifacts/logs/vdp_h36_fit_coco_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=05:00:00
#SBATCH --partition=coe-gpu

set -euo pipefail

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator
mkdir -p artifacts/logs artifacts/checkpoints

echo "=========================================="
echo "Fine-tuning pretrained VideoPose3D with corrected H36M+Fit3D+COCO ACAE on H36M + COCO"
echo "=========================================="

python -u finetune_h36m_coco_bridge.py \
  --h36m-3d data/data_3d_h36m.npz \
  --h36m-2d data/data_2d_h36m_gt.npz \
  --coco-2d-dir artifacts/coco_2d/vitpose_train2017 \
  --annotation-file data/coco/annotations/person_keypoints_train2017.json \
  --acae-checkpoint artifacts/checkpoints/h36_fit_coco_checkpoint.pth \
  --pretrained checkpoint/epoch_120.bin \
  --output artifacts/checkpoints/vdp3d_h36_fit_coco_bridge.bin \
  --epochs 20 \
  --sample-stride 2 \
  --learning-rate 0.0001 \
  --bone-loss-weight 0.01 \
  --angle-loss-weight 0.001 \
  --coco-loss-weight 0.05 \
  --coco-batch-size 256 \
  --max-coco-poses 100000 \
  --min-score 0.2 \
  --min-valid-joints 6 \
  --device cuda

echo ""
echo "=========================================="
echo "H36M+Fit3D+COCO ACAE bridge VideoPose3D fine-tuning complete"
echo "=========================================="
