#!/bin/bash
#SBATCH --job-name=v3_eval
#SBATCH --output=logs/v3_eval_%j.out
#SBATCH --error=logs/v3_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0,atl1-1-02-009-31-0,atl1-1-02-009-32-0,atl1-1-02-009-33-0,atl1-1-02-010-31-0,atl1-1-02-010-32-0,atl1-1-02-010-33-0,atl1-1-02-010-34-0,atl1-1-02-010-35-0,atl1-1-02-010-36-0,atl1-1-02-011-31-0,atl1-1-02-011-32-0
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --partition=ice-gpu
#SBATCH --account=coc

# Evaluate all v3 trained checkpoints + the MotionBERT zero-shot baseline
# for comparison. Produces JSON results in outputs/eval/.

echo "=== Job $SLURM_JOB_ID  Start: $(date) ==="
module load anaconda3
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate pose_env
mkdir -p outputs/eval
nvidia-smi -L

set -e

echo
echo "=== MotionBERT zero-shot baseline (re-eval for direct comparison) ==="
python scripts/evaluate_motionbert.py \
    --weights checkpoints/motionbert/pose3d/MB_ft_h36m.bin \
    --output_json outputs/eval/v3_compare_zeroshot.json

for run in v3_no_h36m_15ep v3_low_lr_15ep v3_biomech2_15ep; do
    echo
    echo "=== $run (LoRA rank 8) ==="
    python scripts/evaluate_motionbert.py \
        --weights outputs/checkpoints/$run/best.pt \
        --lora --lora_rank 8 \
        --output_json outputs/eval/$run.json
done

echo
echo "=== v3_lora_rank2_15ep (LoRA rank 2) ==="
python scripts/evaluate_motionbert.py \
    --weights outputs/checkpoints/v3_lora_rank2_15ep/best.pt \
    --lora --lora_rank 2 \
    --output_json outputs/eval/v3_lora_rank2_15ep.json

echo "=== End: $(date) ==="
