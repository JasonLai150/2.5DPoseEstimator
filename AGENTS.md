# AGENTS.md

## Project Overview

- `acae_2d_extension/`: skeleton-bridging autoencoder (ACAE)
- `vdp3d/`: VideoPose3D model
- `checkpoint/`: pretrained VideoPose3D (H36M)
- `acae_data/`: ACAE checkpoints

Goal:
- Compare VideoPose3D baseline vs ACAE-wrapped vs fine-tuned (Fit3D)
- Evaluate on H36M and Fit3D

---

## Critical Rules

### 1. Input correctness (MOST IMPORTANT)
- Never use `seq[..., :2]` from 3D as 2D input (except debugging).
- Always use proper 2D:
  - H36M → official 2D keypoints + normalization
  - Fit3D → real 2D, detector output, or real camera projection

---

### 2. Native baseline must be correct
- “Without bridge” must replicate official VideoPose3D behavior exactly.
- Target: ~30–40 mm MPJPE on H36M.
- If not achieved → fix pipeline before anything else.

---

### 3. Do not change the problem
- Do NOT feed 3D inputs into VideoPose3D.
- Do NOT add transformations that were not used in training.
- Do NOT modify inputs just to reduce error.

---

### 4. ACAE usage
- ACAE = representation bridge only.
- Encode → VideoPose3D → Decode.
- Compute all losses in decoded (real skeleton) space.
- Do NOT apply biomechanical loss in latent space.

---

### 5. Fit3D rules
- Do NOT guess camera parameters.
- Do NOT train on Fit3D test data.
- Mock projections = debugging only.

---

### 6. Code changes
- OK: inference scripts, naming, organization
- Ask before:
  - modifying VideoPose3D core code
  - changing model architecture
- Prefer wrappers over modifying original model code

---

### 7. Slurm
- Write Slurm scripts for major jobs (match existing format)
- Do not wait for jobs to finish

---

## Debugging Checklist

Before trusting results:
- Check 2D normalization
- Check joint ordering (17 joints)
- Check units (meters vs mm)
- Check camera vs world coordinates
- Check checkpoint compatibility

---

## Output expectations

Always report:
- baseline vs bridged vs fine-tuned
- dataset used (H36M / Fit3D)
- whether inputs are real 2D or projected