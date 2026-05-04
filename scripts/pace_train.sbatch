#!/bin/bash
#SBATCH --job-name=pose_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0,atl1-1-02-009-31-0,atl1-1-02-009-32-0,atl1-1-02-009-33-0,atl1-1-02-010-31-0,atl1-1-02-010-32-0,atl1-1-02-010-33-0,atl1-1-02-010-34-0,atl1-1-02-010-35-0,atl1-1-02-010-36-0,atl1-1-02-011-31-0,atl1-1-02-011-32-0,atl1-1-01-005-5-0,atl1-1-01-005-7-0
#SBATCH --mem=32G
# Excluded above:
#  - mi210 nodes (atl1-1-01-005-17/19-0): AMD ROCm; our PyTorch is CUDA-only,
#    falls back to CPU and OOM-kills the host (5247701).
#  - v100:2 nodes (atl1-1-02-009-31..33, atl1-1-02-010-31..36, atl1-1-02-011-31..32):
#    16GB V100 variant; OOMs on batch=16,seq_len=243 (5250546).
#  - rtx_6000 nodes (atl1-1-01-005-5/7-0): 24GB advertised, ~22GB usable,
#    OOM-kills on batch=16 (5253980). Other GPUs (V100-32GB, A40, A100, L40S,
#    H100, H200) all have ≥32GB and work fine.
#SBATCH --time=08:00:00
#SBATCH --partition=ice-gpu
#SBATCH --account=coc

# Hybrid training: DSTformer + LoRA + composite loss (lambda_reproj=0 for v1).
# Args: $1 = epochs (default 60), $2..N = extra args
#
# Walltime override examples (long walltimes deprioritize the job in slurm —
# always pick the smallest reasonable ceiling for the run you're starting):
#
#   # full ~60-epoch run (default 8h ceiling)
#   sbatch scripts/pace_train.sbatch 60
#
#   # 1-epoch smoke test (override walltime to 30 min for faster scheduling)
#   sbatch --time=00:30:00 scripts/pace_train.sbatch 1 --smoke
#
#   # 10-epoch debug run (1h ceiling)
#   sbatch --time=01:00:00 scripts/pace_train.sbatch 10
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
# Help PyTorch tolerate fragmentation when the GPU is fully utilized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_hybrid.py \
    --epochs "$EPOCHS" \
    --batch_size 16 \
    --seq_len 243 \
    --pretrained checkpoints/motionbert/pose3d/MB_ft_h36m.bin \
    --lora --lora_rank 8 \
    $EXTRA_ARGS

echo "============================================"
echo "End: $(date)"
echo "============================================"
