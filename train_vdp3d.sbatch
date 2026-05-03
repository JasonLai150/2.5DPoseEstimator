#!/bin/bash
#SBATCH --job-name=vdp3d_acae
#SBATCH --output=artifacts/logs/vdp3d_acae_%j.out
#SBATCH --error=artifacts/logs/vdp3d_acae_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=08:00:00
#SBATCH --partition=coe-gpu

set -e

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator

echo "=========================================="
echo "Training VideoPose3D with ACAE Wrapper"
echo "=========================================="

# Run VideoPose3D on standard Human3.6M to establish the 17-joint temporal model.
# --subjects-unlabeled "" guarantees that the Weakly-Supervised Reprojection Loss is disabled.
# The BridgedTemporalModel will automatically freeze the ACAE model.
python -u vdp3d/run.py \
    -d h36m \
    -k gt \
    -arc 3,3,3,3,3 \
    -c artifacts/checkpoints \
    --epochs 120 \
    --batch-size 1024 \
    --subjects-unlabeled "" 

echo ""
echo "=========================================="
echo "Training Complete! Model saved in artifacts/checkpoints/"
echo "=========================================="
