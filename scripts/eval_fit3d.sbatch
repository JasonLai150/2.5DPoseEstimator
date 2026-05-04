#!/bin/bash
#SBATCH --job-name=eval_h36_fit
#SBATCH --chdir=/home/hice1/acheng324/scratch/2.5DPoseEstimator
#SBATCH --output=/home/hice1/acheng324/scratch/2.5DPoseEstimator/artifacts/logs/eval_h36_fit_%j.out
#SBATCH --error=/home/hice1/acheng324/scratch/2.5DPoseEstimator/artifacts/logs/eval_h36_fit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=01:00:00
#SBATCH --partition=coe-gpu

set -euo pipefail

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator

echo "=========================================="
echo "Evaluating baseline H36+Fit ACAE bridge + VDP3D"
echo "Dataset: H36M + Fit3D"
echo "Fit3D input: real ViTPose 2D detector output"
echo "H36M input: official H36M 2D keypoints"
echo "=========================================="

python -u eval_fit3d.py \
  --checkpoint artifacts/checkpoints/vdp3d_h36_fit_bridge.bin \
  --acae-checkpoint artifacts/checkpoints/h36_fit_checkpoint.pth \
  --fit3d-path data/fit3d_train.tar.gz \
  --fit3d-2d-dir artifacts/fit3d_2d/vitpose_fullframe \
  --h36m-3d data/data_3d_h36m.npz \
  --h36m-2d data/data_2d_h36m_gt.npz \
  --sample-stride 2 \
  --device cuda

echo "Done."
