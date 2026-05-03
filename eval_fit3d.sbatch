#!/bin/bash
#SBATCH --job-name=eval_fit3d
#SBATCH --output=artifacts/logs/eval_fit3d_%j.out
#SBATCH --error=artifacts/logs/eval_fit3d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=01:00:00
#SBATCH --partition=coe-gpu

set -e

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator

echo "=========================================="
echo "Evaluating mixed fine-tuned wrapper checkpoint"
echo "=========================================="

python -u eval_fit3d.py --checkpoint artifacts/checkpoints/epoch_120.bin
python -u eval_fit3d.py --checkpoint artifacts/checkpoints/mixed_finetuned_model.bin

echo ""
echo "=========================================="
echo "Evaluating pretrained VideoPose3D on Human3.6M (official metric)"
python -u vdp3d/run.py -d h36m -k gt -arc 3,3,3,3,3 --evaluate epoch_120.bin --checkpoint artifacts/checkpoints --subjects-unlabeled ""

echo ""
echo "=========================================="
echo "Evaluating mixed fine-tuned checkpoint on Human3.6M (official metric, bridge bypassed)"
python -u vdp3d/run.py -d h36m -k gt -arc 3,3,3,3,3 --evaluate mixed_finetuned_model.bin --checkpoint artifacts/checkpoints --subjects-unlabeled ""

echo ""
echo "=========================================="
echo "All Evaluations Complete!"
echo "=========================================="
