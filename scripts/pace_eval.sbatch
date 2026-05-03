#!/bin/bash
#SBATCH --job-name=pose_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=ice-gpu
#SBATCH --account=coc

# Evaluate a saved checkpoint on H36M and Fit3D test sets.
# Usage:
#   sbatch scripts/pace_eval.sbatch                                           # default checkpoint
#   sbatch scripts/pace_eval.sbatch checkpoints/some_other.pt                 # explicit checkpoint
#   sbatch scripts/pace_eval.sbatch checkpoints/some_other.pt outputs/x.json  # explicit out path

CKPT="${1:-checkpoints/baseline_videopose.pt}"
DEFAULT_OUT="outputs/eval/$(basename "${CKPT%.*}")_${SLURM_JOB_ID}.json"
OUT_JSON="${2:-$DEFAULT_OUT}"

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Checkpoint: $CKPT"
echo "Output:     $OUT_JSON"
echo "============================================"

module load anaconda3
module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate pose_env

mkdir -p logs outputs/eval

nvidia-smi

echo ""
echo "Starting evaluation..."
echo ""

python scripts/evaluate_checkpoint.py \
    --checkpoint "$CKPT" \
    --data_root ./data/processed \
    --fit3d_root ./data/processed/fit3d \
    --batch_size 256 \
    --seq_len 27 \
    --output_json "$OUT_JSON"

echo ""
echo "============================================"
echo "End Time: $(date)"
echo "============================================"
