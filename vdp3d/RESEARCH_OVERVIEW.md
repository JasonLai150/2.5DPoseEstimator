# Research Overview: 2.5D Pose Estimation for Domain Adaptation

## The Research Problem

**Goal:** Fine-tune a 3D pose estimator for gym/exercise videos using only 2D data, because 3D labeled gym data doesn't exist.

**The gap:**
- 3D labels exist for lab settings (Human3.6M, AMASS)
- 3D labels do NOT exist for gym/exercise domain
- 2D gym video is abundant (YouTube, fitness apps, etc.)

## The Core Insight

You can learn 3D from 2D via **reprojection** if you have a good 3D prior:

```
Pretrained 3D knowledge (from lab data)
         +
2D gym observations (from 2D detector)
         +
Constraints to prevent cheating
         ↓
= 3D predictions for gym domain
```

## Pipeline Overview

### Phase 1: Start with Pretrained Model
- Model has learned "what valid 3D human poses look like"
- Trained on Human3.6M / AMASS with full 3D supervision
- This gives us a strong 3D prior

### Phase 2: Fine-tune on Gym Videos (2D only)

```
Input: 2D keypoints from gym video (via YOLO-Pose)
        ↓
Model predicts 3D pose
        ↓
Project 3D → 2D using camera model
        ↓
Compare projected 2D with input 2D (reprojection loss)
```

### The Problem: Reprojection is Ambiguous

Many different 3D poses project to the same 2D:
- Person with long arms far away = short arms close up
- Model can "cheat" by warping depth arbitrarily

### The Solution: Biomechanical Constraints

- **Bilateral symmetry:** left arm length = right arm length
- **Joint angle limits:** knees don't bend backwards, elbows don't hyperextend
- These prevent anatomically impossible "cheat" solutions

### Why LoRA: Preserve Pretrained Knowledge

- Reprojection loss alone might destroy 3D priors
- LoRA freezes base model, only trains small adapters (~1.5% of params)
- Result: adapts to gym domain WITHOUT forgetting 3D knowledge

## The Training Loop

```python
for batch in dataloader:
    if batch.has_3d_labels:  # Human3.6M (supervised)
        loss_3d = MPJPE(predicted_3d, ground_truth_3d)
    else:  # Gym videos (weakly-supervised, 2D only)
        loss_3d = 0

    # Always: project to 2D and compare
    projected_2d = camera.project(predicted_3d)
    loss_reproj = MSE(projected_2d, input_2d)

    # Always: enforce anatomical validity
    loss_biomech = symmetry_loss + joint_angle_loss

    total_loss = λ1*loss_3d + λ2*loss_reproj + λ3*loss_biomech
```

## Loss Formulation

`Total Loss = λ1*L_3D + λ2*L_reproj + λ3*L_biomech`

| Loss | Purpose | When Applied |
|------|---------|--------------|
| L_3D (MPJPE) | Learn correct 3D scale/structure | Only on labeled data (H36M) |
| L_reproj | Adapt to gym domain | All data (gym + lab) |
| L_biomech | Prevent depth cheating | All data |

### Biomechanical Constraints (L_biomech)

1. **Bilateral Symmetry Loss:** `|length(left_bone) - length(right_bone)|`
   - Enforces: left arm = right arm, left leg = right leg

2. **Anatomical Hinge Loss:** Penalizes angles outside valid range
   - Knee: 0° to 160° (no hyperextension)
   - Elbow: 0° to 160°

## Why Base Model Choice Matters Less Than You Think

The **novel contribution** is the training framework (reprojection + biomech + LoRA), not the base architecture.

| Model | Pros | Cons |
|-------|------|------|
| VideoPose3D | Simple, fast, battle-tested | Conv-based, no pretraining |
| PoseFormer | Transformer, good temporal | No pretraining |
| MotionBERT | AMASS+MPM pretraining | Complex |
| APTPose | Anatomy-aware | Newer, less tested |

**Recommendation:** Start simple (VideoPose3D or PoseFormer), prove the concept works, then try more complex models if needed.

## Datasets

| Dataset | Role | Labels |
|---------|------|--------|
| Human3.6M | Training (supervised) | Full 3D |
| Gym videos | Training (weakly-supervised) | 2D only (from detector) |
| Fit3D | Evaluation | Full 3D (fitness movements) |

## Evaluation Strategy

1. **Baseline:** Pretrained model (no fine-tuning) on Fit3D
2. **Ours:** Fine-tuned with reprojection + biomech on Fit3D
3. **Metrics:** MPJPE, P-MPJPE, BLI (bilateral length inconsistency)

## Key Questions This Research Answers

1. Can we adapt 3D pose models to new domains using only 2D supervision?
2. Do biomechanical constraints prevent degenerate solutions?
3. Does LoRA preserve pretrained knowledge during domain adaptation?
4. How much does this improve over zero-shot transfer?

## Implementation Checklist

- [x] Base model architecture (DSTformer with LoRA support)
- [x] Reprojection loss (camera projection)
- [x] Biomechanical losses (symmetry, hinge)
- [x] Training infrastructure
- [x] Fit3D data preparation
- [ ] Baseline evaluation on Fit3D
- [ ] Gym video data collection/preprocessing
- [ ] Full training with hybrid losses
- [ ] Ablation studies (w/ and w/o each component)
