#!/bin/bash
#SBATCH --job-name=viz_h36_coco
#SBATCH --chdir=/home/hice1/acheng324/scratch/2.5DPoseEstimator
#SBATCH --output=/home/hice1/acheng324/scratch/2.5DPoseEstimator/artifacts/logs/viz_h36_fit_coco_%j.out
#SBATCH --error=/home/hice1/acheng324/scratch/2.5DPoseEstimator/artifacts/logs/viz_h36_fit_coco_%j.err
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator

echo "Rendering COCO-added H36+Fit+COCO bridge sample comparisons..."

python -u visualize_eval_samples.py \
  --checkpoint artifacts/checkpoints/vdp3d_h36_fit_coco_bridge.bin \
  --acae-checkpoint artifacts/checkpoints/h36_fit_coco_checkpoint.pth \
  --fit3d-path data/fit3d_train.tar.gz \
  --fit3d-2d-dir artifacts/fit3d_2d/vitpose_fullframe \
  --h36m-3d data/data_3d_h36m.npz \
  --h36m-2d data/data_2d_h36m_gt.npz \
  --output-dir artifacts/visualizations/sample_comparisons_h36_fit_coco \
  --fit3d-count 3 \
  --h36m-count 3 \
  --sample-stride 2

echo "Done."
