#!/bin/bash
#SBATCH --job-name=acae_aligned
#SBATCH --output=artifacts/logs/acae_aligned_%j.out
#SBATCH --error=artifacts/logs/acae_aligned_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=04:00:00
#SBATCH --partition=coe-gpu

# ============================================================================
# Two-stage pipeline:
#   1. Prepare H36M + Fit3D data into unified skeleton format
#   2. Train ACAE with 17-joint H36M-aligned latent space
# ============================================================================

set -e

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator

echo "=========================================="
echo "Stage 1: Preparing H36M + Fit3D data"
echo "=========================================="

# Data is already prepared in acae_data/ from the previous run
# python acae_2D_extension/prepare_h36m_fit3d.py \
#     --h36m-path data/data_3d_h36m.npz \
#     --fit3d-path data/fit3d_train.tar.gz \
#     --output-dir acae_data \
#     --sample-stride 5

echo ""
echo "=========================================="
echo "Stage 2: Training aligned ACAE"
echo "=========================================="

python acae_2D_extension/train_aligned_acae.py \
    --data-dir acae_data \
    --checkpoint-dir artifacts/checkpoints \
    --device auto \
    --epochs 30 \
    --batch-size 64 \
    --align-lambda 1.0 \
    --regul-lambda 0.6

echo ""
echo "=========================================="
echo "Done! Check artifacts/checkpoints/ for outputs."
echo "=========================================="
echo "  artifacts/checkpoints/acae_aligned_checkpoint.pth"
echo "  artifacts/checkpoints/result.npz"
echo "  artifacts/checkpoints/losses.csv"
