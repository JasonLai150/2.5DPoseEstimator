#!/bin/bash
#SBATCH --job-name=pose_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=ice-gpu
#SBATCH --account=coc

# Hybrid training: DSTformer + LoRA + composite loss (lambda_reproj=0 for v1).
# Args: $1 = epochs (default 60), $2 = extra args
EPOCHS="${1:-60}"
EXTRA_ARGS="${@:2}"

echo "============================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
echo "Epochs: $EPOCHS  Extra: $EXTRA_ARGS"
echo "============================================"

module load anaconda3
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate pose_env

mkdir -p logs outputs/checkpoints
nvidia-smi

python scripts/train_hybrid.py \
    --epochs "$EPOCHS" \
    --batch_size 64 \
    --seq_len 243 \
    --pretrained checkpoints/motionbert/pretrain/MB_release.bin \
    --lora --lora_rank 8 \
    $EXTRA_ARGS

echo "============================================"
echo "End: $(date)"
echo "============================================"
