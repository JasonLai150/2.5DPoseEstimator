# Environment and Repo Guide

This repo compares the native VideoPose3D baseline, the ACAE-wrapped model, and mixed/fine-tuned checkpoints on Human3.6M and Fit3D.

## Repository Layout

- `vdp3d/`: VideoPose3D source. Prefer wrappers and scripts over editing core model code.
- `acae_2D_extension/`: ACAE training, validation, and visualization code.
- `data/`: local datasets and 2D keypoint files. This directory is ignored by git.
- `artifacts/checkpoints/`: current checkpoint location for VideoPose3D, ACAE, mixed fine-tuning, and ViTPose weights.
- `artifacts/logs/`: Slurm stdout/stderr logs.
- `artifacts/fit3d_2d/`: extracted Fit3D 2D keypoints.
- `checkpoint/`: legacy VideoPose3D checkpoint location. Existing scripts prefer `artifacts/checkpoints/`.
- `acae_data/`: prepared ACAE data and legacy ACAE outputs.
- `external/ViTPose_fitness/`: vendored ViTPose/MMPose code used for Fit3D 2D extraction.
- `run_paths.py`: shared path constants used by the newer scripts.

## Environments

Work from the repo root:

```sh
cd /home/hice1/acheng324/scratch/2.5DPoseEstimator
```

The ViTPose extraction workflow uses a dedicated Conda env at:

```sh
/home/hice1/acheng324/scratch/conda_envs/vitpose
```

Create it with:

```sh
sbatch setup_vitpose_env.sbatch
```

That job installs PyTorch, MMCV, MMPose/ViTPose dependencies, and `external/ViTPose_fitness` in editable mode. Check `artifacts/logs/setup_vitpose_<jobid>.out` and `.err` for setup output.

For the main VideoPose3D/ACAE scripts, use the Python environment already configured for this project on the cluster. Before running jobs, make sure it has PyTorch, NumPy, SciPy, and Matplotlib available.

## Data and Checkpoints

Human3.6M data should be in `data/`:

- `data/data_3d_h36m.npz`
- `data/data_2d_h36m_gt.npz`
- official detector files such as `data/data_2d_h36m_cpn_ft_h36m_dbb.npz` when evaluating the native baseline with real 2D detections

Fit3D training data is expected at:

```sh
data/fit3d_train.tar.gz
```

Current checkpoint paths are:

- `artifacts/checkpoints/epoch_120.bin`: pretrained or trained VideoPose3D checkpoint used by evaluation scripts.
- `artifacts/checkpoints/mixed_finetuned_model.bin`: mixed/fine-tuned checkpoint.
- `artifacts/checkpoints/acae_aligned_checkpoint.pth`: ACAE checkpoint.
- `artifacts/checkpoints/vitpose/vitpose_base_coco_256x192.pth`: ViTPose-B checkpoint for Fit3D 2D extraction.

Legacy checkpoints may still appear in:

- `checkpoint/`
- `acae_data/checkpoints/`

These checkpoint and log directories are ignored by git.

## Common Jobs

Inspect the Fit3D archive:

```sh
sbatch inspect_fit3d.sbatch
```

Set up the ViTPose environment:

```sh
sbatch setup_vitpose_env.sbatch
```

Extract Fit3D 2D keypoints with ViTPose:

```sh
sbatch extract_fit3d_vitpose_2d.sbatch
```

Train the aligned ACAE:

```sh
sbatch train_acae.sbatch
```

Train VideoPose3D on Human3.6M:

```sh
sbatch train_vdp3d.sbatch
```

Fine-tune on the mixed Fit3D/H36M setup:

```sh
sbatch finetune_mixed.sbatch
```

Run Fit3D and H36M evaluation:

```sh
sbatch eval_fit3d.sbatch
```

Do not wait for long Slurm jobs to finish in an interactive session. Use `squeue`, `sacct`, and the files in `artifacts/logs/` to check progress.

## Direct Evaluation Commands

Native VideoPose3D H36M evaluation should use real 2D inputs and official normalization. For example:

```sh
python -u vdp3d/run.py \
  -d h36m \
  -k cpn_ft_h36m_dbb \
  -arc 3,3,3,3,3 \
  --evaluate epoch_120.bin \
  --checkpoint artifacts/checkpoints \
  --subjects-unlabeled ""
```

Fit3D evaluation:

```sh
python -u eval_fit3d.py --checkpoint artifacts/checkpoints/epoch_120.bin
python -u eval_fit3d.py --checkpoint artifacts/checkpoints/mixed_finetuned_model.bin
```

## Important Correctness Rules

- Never use `seq[..., :2]` from 3D poses as model input except for debugging.
- H36M input must come from official 2D keypoints with the VideoPose3D normalization path.
- Fit3D input must be real 2D, detector output, or real camera projection. Do not guess camera parameters.
- The native "without bridge" baseline must reproduce official VideoPose3D behavior before comparing bridged or fine-tuned runs.
- ACAE is only a representation bridge: encode, run VideoPose3D, then decode. Compute losses in decoded real-skeleton space.
- Report every result with dataset, model variant, checkpoint, metric, and whether the 2D inputs were real detector/official 2D or projected/debug inputs.
