#!/bin/bash
#SBATCH --job-name=finetune_mixed
#SBATCH --output=artifacts/logs/finetune_mixed_%j.out
#SBATCH --error=artifacts/logs/finetune_mixed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=04:00:00
#SBATCH --partition=coe-gpu

set -e

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator

echo "=========================================="
echo "Fine-tuning VideoPose3D on Mixed Dataset (Fit3D + H36M)"
echo "=========================================="

python -u finetune_mixed.py

echo ""
echo "=========================================="
echo "Evaluating Fine-Tuned Model on Human3.6M"
echo "=========================================="
python -u vdp3d/run.py -d h36m -k gt -arc 3,3,3,3,3 --evaluate mixed_finetuned_model.bin --checkpoint artifacts/checkpoints --subjects-unlabeled ""

echo ""
echo "=========================================="
echo "Fine-tuning and Evaluations Complete!"
echo "=========================================="
