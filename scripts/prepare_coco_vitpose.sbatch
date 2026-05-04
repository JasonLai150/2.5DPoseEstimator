#!/bin/bash
#SBATCH --job-name=prep_coco_vitpose
#SBATCH --output=artifacts/logs/prep_coco_vitpose_%j.out
#SBATCH --error=artifacts/logs/prep_coco_vitpose_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=coe-gpu

set -e

cd /home/hice1/acheng324/scratch/2.5DPoseEstimator
mkdir -p artifacts/logs data/coco

echo "=========================================="
echo "Preparing COCO 2017 layout for ViTPose"
echo "Expected train zip: data/coco/train2017.zip"
echo "Expected annotations zip: data/coco/annotations_trainval2017.zip"
echo "=========================================="

python -u prepare_coco_vitpose.py \
  --coco-root data/coco \
  --train-zip data/coco/train2017.zip \
  --annotations-zip data/coco/annotations_trainval2017.zip

echo ""
echo "=========================================="
echo "COCO ViTPose preparation complete"
echo "=========================================="
