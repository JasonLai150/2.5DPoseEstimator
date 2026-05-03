#!/bin/bash
#SBATCH --job-name=mb_eval
#SBATCH --output=logs/mb_eval_%j.out
#SBATCH --error=logs/mb_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --partition=ice-gpu
#SBATCH --account=coc

# Evaluate pretrained MotionBERT zero-shot on H36M + Fit3D.
WEIGHTS="${1:-checkpoints/motionbert/pose3d/MB_ft_h36m.bin}"
OUT_JSON="${2:-outputs/eval/motionbert_$(basename "${WEIGHTS%.*}")_${SLURM_JOB_ID}.json}"

echo "============================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
echo "Weights: $WEIGHTS"
echo "Output:  $OUT_JSON"
echo "============================================"

module load anaconda3
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate pose_env

mkdir -p logs outputs/eval
nvidia-smi

python scripts/evaluate_motionbert.py \
    --weights "$WEIGHTS" \
    --data_root ./data/processed \
    --fit3d_root ./data/processed/fit3d \
    --seq_len 243 \
    --batch_size 64 \
    --output_json "$OUT_JSON"

echo "============================================"
echo "End: $(date)"
echo "============================================"
