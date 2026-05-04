#!/bin/bash
#SBATCH --job-name=fit3d_vitpose2d
#SBATCH --output=artifacts/logs/fit3d_vitpose2d_%j.out
#SBATCH --error=artifacts/logs/fit3d_vitpose2d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=12:00:00
#SBATCH --partition=coe-gpu

set -e

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator

mkdir -p artifacts/logs artifacts/checkpoints/vitpose artifacts/fit3d_2d

# Preflight:
# - The ViTPose-B checkpoint must be a real .pth, not the 0-byte failed download
#   placeholder. Expected path:
#   artifacts/checkpoints/vitpose/vitpose_base_coco_256x192.pth
# - Run setup_vitpose_env.sbatch once before this job to create the env below.
# - For a smoke test, add "--max-videos 2" to the Python command below.

CONDA_SH=/home/hice1/acheng324/scratch/miniconda/etc/profile.d/conda.sh
VITPOSE_ENV=/home/hice1/acheng324/scratch/conda_envs/vitpose

source "$CONDA_SH"

if [ ! -d "$VITPOSE_ENV" ]; then
  echo "Missing ViTPose env: $VITPOSE_ENV"
  echo "Submit setup_vitpose_env.sbatch first, then rerun this extraction job."
  exit 2
fi

conda activate "$VITPOSE_ENV"

echo "=========================================="
echo "Extracting Fit3D 2D keypoints with ViTPose-B"
echo "BBox mode: full video frame"
echo "=========================================="

export PYTHONPATH="$PWD/external/ViTPose_fitness:${PYTHONPATH:-}"

python -c "import torch, mmcv; from mmpose.apis import init_pose_model, inference_top_down_pose_model; print('torch', torch.__version__); print('mmcv', mmcv.__version__); print('vitpose imports ok')"

python -u extract_fit3d_vitpose_2d.py \
  --fit3d-path data/fit3d_train.tar.gz \
  --vitpose-root external/ViTPose_fitness \
  --pose-config external/ViTPose_fitness/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
  --pose-checkpoint artifacts/checkpoints/vitpose/vitpose_base_coco_256x192.pth \
  --output-dir artifacts/fit3d_2d/vitpose_fullframe \
  --tmp-dir /tmp/fit3d_vitpose_video_extract \
  --device cuda:0

echo ""
echo "=========================================="
echo "Fit3D ViTPose extraction complete"
echo "=========================================="
