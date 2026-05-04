#!/bin/bash
#SBATCH --job-name=coco_vitpose2d
#SBATCH --output=artifacts/logs/coco_vitpose2d_%j.out
#SBATCH --error=artifacts/logs/coco_vitpose2d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=12:00:00
#SBATCH --partition=coe-gpu

set -e

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator
mkdir -p artifacts/logs artifacts/coco_2d

CONDA_SH=/home/hice1/acheng324/scratch/miniconda/etc/profile.d/conda.sh
VITPOSE_ENV=/home/hice1/acheng324/scratch/conda_envs/vitpose

source "$CONDA_SH"

if [ ! -d "$VITPOSE_ENV" ]; then
  echo "Missing ViTPose env: $VITPOSE_ENV"
  echo "Submit setup_vitpose_env.sbatch first."
  exit 2
fi

conda activate "$VITPOSE_ENV"
export PYTHONPATH="$PWD/external/ViTPose_fitness:${PYTHONPATH:-}"

echo "=========================================="
echo "Running ViTPose on COCO train2017"
echo "=========================================="

python -u extract_coco_vitpose_2d.py \
  --coco-root data/coco \
  --annotation-file data/coco/annotations/person_keypoints_train2017.json \
  --vitpose-root external/ViTPose_fitness \
  --pose-config external/ViTPose_fitness/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
  --pose-checkpoint artifacts/checkpoints/vitpose/vitpose_base_coco_256x192.pth \
  --output-dir artifacts/coco_2d/vitpose_train2017

echo ""
echo "=========================================="
echo "COCO ViTPose extraction complete"
echo "=========================================="
