#!/bin/bash
#SBATCH --job-name=h36_fit
#SBATCH --output=artifacts/logs/h36_fit_%j.out
#SBATCH --error=artifacts/logs/h36_fit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=05:00:00
#SBATCH --partition=coe-gpu

set -euo pipefail

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator
mkdir -p artifacts/logs artifacts/checkpoints acae_data_h36_fit

echo "=========================================="
echo "Preparing corrected H36M + Fit3D ACAE data"
echo "=========================================="

python -u acae_2D_extension/prepare_h36m_fit3d.py \
  --h36m-path data/data_3d_h36m.npz \
  --fit3d-path data/fit3d_train.tar.gz \
  --output-dir acae_data_h36_fit \
  --sample-stride 5

echo ""
echo "=========================================="
echo "Training corrected H36M + Fit3D ACAE"
echo "=========================================="

python -u acae_2D_extension/train_aligned_acae.py \
  --data-dir acae_data_h36_fit \
  --checkpoint-dir artifacts/checkpoints \
  --run-name h36_fit \
  --device auto \
  --epochs 30 \
  --batch-size 64 \
  --align-lambda 1.0 \
  --regul-lambda 0.6

echo ""
echo "=========================================="
echo "Done. Check artifacts/checkpoints/h36_fit_checkpoint.pth"
echo "=========================================="
