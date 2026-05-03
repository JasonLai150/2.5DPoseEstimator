#!/bin/bash
#SBATCH --job-name=inspect_fit3d
#SBATCH --output=artifacts/logs/inspect_fit3d_%j.out
#SBATCH --error=artifacts/logs/inspect_fit3d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=coe-gpu

set -e

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator

echo "=========================================="
echo "Inspecting Fit3D archive format and 2D/media assets"
echo "=========================================="

python -u inspect_fit3d.py --fit3d-path data/fit3d_train.tar.gz

echo ""
echo "=========================================="
echo "Fit3D inspection complete"
echo "=========================================="
