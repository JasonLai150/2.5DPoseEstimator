#!/bin/bash
#SBATCH --job-name=pose_baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=coc-gpu
#SBATCH --account=coc

# PACE/ICE Baseline Training Script
# Usage: sbatch scripts/pace_baseline.sbatch

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

# Load modules
module load anaconda3

# Activate environment
source activate pose_env

# Create logs directory if needed
mkdir -p logs
mkdir -p checkpoints

# Print GPU info
nvidia-smi

# Run baseline training
echo ""
echo "Starting baseline training..."
echo ""

python scripts/run_baseline.py \
    --data_root ./data/processed \
    --epochs 30 \
    --batch_size 512 \
    --seq_len 27 \
    --hidden_dim 1024 \
    --num_blocks 4 \
    --lr 1e-3 \
    --num_workers 8 \
    --save_path checkpoints/baseline_videopose.pt

echo ""
echo "============================================"
echo "End Time: $(date)"
echo "============================================"
