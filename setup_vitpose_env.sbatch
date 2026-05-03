#!/bin/bash
#SBATCH --job-name=setup_vitpose
#SBATCH --output=artifacts/logs/setup_vitpose_%j.out
#SBATCH --error=artifacts/logs/setup_vitpose_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=coe-gpu

set -euo pipefail

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator
mkdir -p artifacts/logs

CONDA_SH=/home/hice1/acheng324/scratch/miniconda/etc/profile.d/conda.sh
VITPOSE_ENV=/home/hice1/acheng324/scratch/conda_envs/vitpose

source "$CONDA_SH"

if [ ! -d "$VITPOSE_ENV" ]; then
  conda create -y -p "$VITPOSE_ENV" python=3.9 pip
fi

conda activate "$VITPOSE_ENV"

python -m pip install --upgrade "pip<25" "setuptools<70" wheel
python -m pip install "numpy==1.23.5" cython

python -m pip uninstall -y torch torchvision torchaudio || true
conda install -y -c pytorch -c nvidia \
  "pytorch==1.13.1" \
  "torchvision==0.14.1" \
  "pytorch-cuda=11.7"

python -m pip install \
  "mmcv-full==1.5.0" \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

python -m pip install \
  "timm==0.4.9" \
  einops \
  json_tricks \
  matplotlib \
  munkres \
  "opencv-python-headless<4.9" \
  pillow \
  scipy \
  xtcocotools \
  chumpy

python -m pip install -e external/ViTPose_fitness --no-deps

export PYTHONPATH="$PWD/external/ViTPose_fitness:${PYTHONPATH:-}"

python -c "import torch, mmcv; from mmpose.apis import init_pose_model, inference_top_down_pose_model; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('mmcv', mmcv.__version__); print('vitpose imports ok')"

echo ""
echo "ViTPose environment ready: $VITPOSE_ENV"
