# PACE/ICE Setup Guide

## Quick Start

### 1. Connect to PACE

```bash
# Connect to GT VPN first (GlobalProtect)
ssh <your-gtid>@login-ice.pace.gatech.edu
```

### 2. Initial Setup (One-time)

```bash
# Go to scratch directory (300GB limit vs 30GB home)
cd ~/scratch

# Clone your repo
git clone https://github.com/YOUR_USERNAME/2.5DPoseEstimator.git
cd 2.5DPoseEstimator

# Set up conda symlink (to use scratch for environments)
mkdir -p ~/scratch/.conda
rm -rf ~/.conda  # Remove if exists
ln -s ~/scratch/.conda ~/.conda

# Load anaconda
module load anaconda3

# Create environment
conda create -n pose_env python=3.10 -y
conda activate pose_env

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install wandb numpy scipy einops tqdm pyyaml h5py cdflib
```

### 3. Upload Data

From your local machine:
```bash
# Upload processed data to PACE
scp -r data/processed <your-gtid>@login-ice.pace.gatech.edu:~/scratch/2.5DPoseEstimator/data/
```

Or if you have the raw data on PACE, process it there:
```bash
# On PACE
cd ~/scratch/2.5DPoseEstimator
python scripts/process_videopose_data.py --data_root ./data --output_dir ./data/processed
```

### 4. Submit Job

```bash
# Make sure logs directory exists
mkdir -p logs checkpoints

# Submit the job
sbatch scripts/pace_baseline.sbatch

# Check job status
squeue -u $USER

# Watch output in real-time
tail -f logs/baseline_*.out
```

### 5. Monitor Job

```bash
# Check queue status
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# View output
cat logs/baseline_<JOB_ID>.out
```

## Resource Limits

| Resource | Max per Job |
|----------|-------------|
| GPU Hours | 16 hours |
| CPU Hours | 512 hours |
| Walltime | 8 hours |
| GPUs | Varies by partition |

## Tips

### Storage
- **Home (~/)**: 30GB - configs, small files
- **Scratch (~/scratch)**: 300GB - data, models, environments
- Check quota: `pace-quota`

### Conda on Scratch
```bash
# Set up symlink (one-time)
mkdir -p ~/scratch/.conda
ln -s ~/scratch/.conda ~/.conda
```

### HuggingFace Cache
```bash
export HF_HOME=~/scratch/.cache/huggingface
export TRANSFORMERS_CACHE=~/scratch/.cache/huggingface
```

### Interactive GPU Session
```bash
# Request interactive session (for debugging)
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
```

### Common Modules
```bash
module spider anaconda3  # Find available versions
module load anaconda3    # Load default version
module list              # Show loaded modules
```

## Troubleshooting

### Job won't start
- Check queue: `squeue -u $USER`
- Resources might be busy, wait or reduce request

### Out of memory
- Reduce batch size
- Use gradient checkpointing

### CUDA errors
- Make sure pytorch-cuda version matches cluster CUDA
- Check: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`

### Storage quota exceeded
- Check: `pace-quota`
- Move large files to scratch
- Clean conda caches: `conda clean --all`
