# Final-Report Evidence Notes

Compiled 2026-05-04. Cite numbers / observations / tables directly from this
document. Every number traces back to a JSON in `outputs/eval/` or a SLURM
log in `logs/`.

---

## 1. Research question

> Can a 3D pose estimator pretrained on lab-captured data (Human3.6M) be
> adapted to gym/exercise video using only weakly-supervised 2D signals,
> when 3D-labeled gym data does not exist?

We test this through a *2.5D hybrid* fine-tuning regime — H36M for 3D
supervision, Fit3D-train (subjects s03–s10) for 2D-only weakly-supervised
training, evaluated on the held-out Fit3D-test (subject s11).

---

## 2. Method

### 2.1 Backbone
**DSTformer** (dual-stream spatio-temporal transformer, MotionBERT
architecture). 42.5M parameters; spatial attention across 17 joints,
temporal attention across 243 frames, 5 dual-stream blocks fused via
learned per-block attention. Input: (B, T=243, J=17, 2D normalized
keypoints + confidence). Output: (B, T, J, 3D meters, root-relative).

### 2.2 Initialization
**MB_ft_h36m.bin** (MotionBERT released checkpoint fine-tuned on H36M-SH,
publicly hosted at `huggingface.co/walterzhu/MotionBERT`). Loads with 0
missing / 0 unexpected / 0 shape mismatches against our DSTformer once
the config matches MotionBERT's actual `embed_dim=512`, `mlp_ratio=2`
defaults and we remap the `ts_attn` ↔ `stream_fusion` key naming.

### 2.3 Adapters: LoRA
**Low-rank adaptation** on `qkv` and `proj` linear layers in every
attention block. Rank 8 → 491,520 trainable params (1.14% of total);
rank 2 → 122,880 trainable (0.29%). All non-LoRA parameters frozen.

### 2.4 Loss
$$
\mathcal{L} = \lambda_{3D}\,L_{3D} + \lambda_{reproj}\,L_{reproj} + \lambda_{biomech}\,L_{biomech}
$$

- **L_3D**: MPJPE on H36M training samples (gated by `has_3d=True`).
- **L_reproj** (v2 implementation): for weakly-supervised samples
  (`~has_3d & has_reproj`), recover absolute camera-space 3D as
  `pred_root_relative + cam_root` (per-frame absolute pelvis position
  saved at preprocess time), project via per-sample camera intrinsics
  to normalized [-1, 1] image space (matching the on-disk 2D format),
  L1 vs the input 2D.
- **L_biomech**: bilateral symmetry loss (left-right bone length
  matching) + anatomical hinge loss (knee/elbow within [0°, 160°]).

### 2.5 Training data
- **H36M**: 1,529 sliding-window samples (T=243, stride=243), 3D-supervised.
- **Fit3D-train (s03–s10)**: 1,181 sliding-window samples, has_3d=False,
  has_reproj=True, paired with `cam_root.npy` and `cam_intrinsics.npy`
  per sequence.
- Total: 2,710 training windows per epoch, batch size 16, mixed-source
  via `ConcatDataset`.

### 2.6 Held-out evaluation
**Fit3D-test (s11)**: 47 sequences, 240 sliding windows after
seq_len=243/stride=243. Reports MPJPE, P-MPJPE (Procrustes-aligned),
BLI (Bilateral Length Inconsistency).

### 2.7 Skeleton mapping
IMAR's `joints3d_25` for Fit3D uses a non-standard joint ordering
(verified by inspection of `imar_tools/util/dataset_util.py`'s
`plot_over_image` limb diagram). Direct index map for limbs and pelvis;
**thorax (H36M[8]) replaced with the shoulder midpoint** of IMAR's
left/right shoulders, **spine (H36M[7]) is the midpoint of pelvis and
the new thorax**. Bone-length sanity post-fix:

| Bone | H36M | Fit3D | Ratio |
|---|---|---|---|
| pelvis → spine | 254 mm | 222 mm | 0.87 |
| spine → thorax | 250 mm | 222 mm | 0.89 |
| thorax → neck | 111 mm | 55 mm | 0.50 |
| humerus | 283 mm | 290 mm | 1.02 |
| forearm | 248 mm | 249 mm | 1.00 |
| femur | 462 mm | 427 mm | 0.93 |
| tibia | 460 mm | 441 mm | 0.96 |

Limbs (arms/legs) match within 7%; torso ~12%. Thorax–neck mismatch is a
known residual issue (shoulder midpoint sits slightly higher than H36M's
joint 8).

### 2.8 Coordinate convention fix
**Critical IMAR convention** (verified against
`imar_tools/util/ghum_util.py`, `smplx_util.py`, and the lab-dataset
notebook): world → camera is `(world − T) @ R^T`, not `world @ R^T + T`.
Wrong convention pushes pixels to e.g. x = -3863 instead of ~395.
Saved `poses_3d.npy` is camera-space root-centered (matching model
output); 2D normalized via `pixel/cx − 1`.

---

## 3. Baselines

### 3.1 VideoPose3D scratch
30 epochs on H36M only, in-repo TCN architecture (16.9M params).
Source: `outputs/eval/baseline_videopose_5245752.json`.

| Test set | MPJPE | P-MPJPE |
|---|---|---|
| H36M (in-domain) | 62.00 mm | 48.06 mm |
| Fit3D s11 (target) | 692.24 mm | 160.92 mm |

### 3.2 MotionBERT MB_ft_h36m zero-shot
DSTformer with official H36M-SH-finetuned weights, no further training.
Source: `outputs/eval/v3_compare_zeroshot.json`.

| Test set | MPJPE | P-MPJPE | BLI |
|---|---|---|---|
| H36M (in-domain) | 520.75 mm | **28.18 mm** | 0.00157 |
| Fit3D s11 (target) | 302.24 mm | **166.05 mm** | 0.00373 |

The 521 mm raw MPJPE on H36M reflects a **scale mismatch** between
MotionBERT's output and our processed H36M targets (MotionBERT's
published P-MPJPE is 37.2 mm; our 28 mm is consistent). After fine-tuning
the model adapts to our scale.

---

## 4. Hybrid training experiments

All runs: DSTformer + LoRA, init MB_ft_h36m, AdamW lr=1e-4 (unless
noted), 15 epochs, batch 16, seq_len 243, mixed H36M+Fit3D-train batches.
Held-out eval on Fit3D s11.

### 4.1 Initial diagnostic runs (not in final ablation table)

**Run 0 — Catastrophic naive fine-tune** (run 5249221, init from MB_release
backbone-only, λ_3d=1, λ_reproj=0, λ_biomech=0.1):
- val P-MPJPE went 0.28 → 0.49 in 14 epochs. Train loss dropped fast
  while validation diverged.
- Diagnosis: H36M overfit + biomech-only regularization at λ=0.1 is
  insufficient. Initing from the *backbone* (no 3D head) means relearning
  the head from a tiny dataset.

**Run v2 — Stronger biomech + reproj** (run 5249527, MB_ft_h36m init,
λ_3d=1, λ_reproj=0.5, λ_biomech=1.0):
- val P-MPJPE held 0.27 → 0.29 across 10 epochs. Hinge loss converged
  by epoch 4, after which biomech became inactive and l3d resumed
  dropping. Best.pt = epoch 1.
- Diagnosis: regularization prevents catastrophic divergence but doesn't
  *improve* below zero-shot. Biomech is a converging regularizer (once
  poses are anatomically valid, it stops pulling).

### 4.2 Final ablation table — Fit3D s11 (out-of-domain target)

All runs init MB_ft_h36m, 15 epochs. Numbers are from
`outputs/eval/v3_*.json`.

| Method | LoRA rank | λ_3d | λ_reproj | λ_biomech | LR | **Fit3D MPJPE** ↓ | **Fit3D P-MPJPE** ↓ | **Fit3D BLI** ↓ |
|---|---|---|---|---|---|---|---|---|
| MotionBERT zero-shot | — | — | — | — | — | 302.24 | **166.05** | 0.00373 |
| Hybrid + low LR | 8 | 1.0 | 0.5 | 1.0 | 1e-5 | 302.25 | 166.06 | 0.00373 |
| Hybrid + biomech ×2 | 8 | 1.0 | 0.5 | 2.0 | 1e-4 | 302.29 | 166.13 | 0.00373 |
| Hybrid + no H36M | 8 | 0 | 0.5 | 1.0 | 1e-4 | 259.17 | 181.48 | 0.00611 |
| Hybrid + rank 2 | 2 | 1.0 | 0.5 | 1.0 | 1e-4 | 260.15 | 185.00 | 0.01213 |
| **Hybrid + no H36M + rank 2** | **2** | **0** | **0.5** | **1.0** | **1e-4** | **245.78** ✓✓ | **175.98** | **0.00528** |

### 4.2b Cross-domain evaluation — H36M test (in-domain, what we trade away)

Same runs evaluated on H36M test set (532 windows, subjects S9 + S11):

| Method | **H36M MPJPE** ↓ | **H36M P-MPJPE** ↓ | **H36M BLI** ↓ |
|---|---|---|---|
| MotionBERT zero-shot | 520.75 | **28.18** | 0.00157 |
| Hybrid + low LR | 520.75 | 28.18 | 0.00157 |
| Hybrid + biomech ×2 | 520.76 | 28.19 | 0.00157 |
| Hybrid + no H36M | 546.84 | 69.47 | 0.00386 |
| Hybrid + rank 2 | **215.16** ✓ | 170.71 ✗ | 0.03359 ✗ |
| Hybrid + no H36M + rank 2 | 550.17 | 65.65 | 0.00318 |

**Cross-domain trade-off observations:**

- **No-free-lunch on the headline run**: gaining 18.6% on Fit3D MPJPE costs
  ~6% on H36M MPJPE (521 → 550) and 2.3× on H36M P-MPJPE (28 → 66 mm).
  Without H36M supervision, the model drifts out of the lab-data
  distribution while adapting to gym domain.
- **The "rank 2 alone" run shows a striking scale-vs-structure trade-off**:
  raw H36M MPJPE drops 60% (521 → 215) — the rank-2 LoRA very
  efficiently learns a *scale correction* — but H36M P-MPJPE
  (Procrustes-aligned, scale-invariant) rises 6× (28 → 171). The 123k
  trainable parameters fixed the scale-mismatch artifact at the cost
  of distorting relative joint geometry. This is the cleanest single
  experimental demonstration of LoRA's "scale gets fixed, structure gets
  broken" failure mode in our setup.
- **Trivial runs preserve H36M** (low_lr, biomech×2): identical H36M
  numbers to zero-shot, confirming their best.pt is effectively the
  unmodified base.

**Headline:** the combined intervention reduces **Fit3D MPJPE by 18.6 %**
(302.24 → 245.78 mm). It also yields the *smallest* P-MPJPE regression
among improving variants (175.98 mm vs 181.48 / 185.00 mm for the
single-intervention runs).

### 4.3 Ablation interpretation

- **λ_3d = 0 (drop H36M supervision)** alone reduces MPJPE 14% (302 →
  259). Without H36M's overfit-driving signal, LoRA adapts purely
  through biomech + reproj on Fit3D-train.
- **rank 2 (low-capacity LoRA)** alone reduces MPJPE 14% (302 → 260),
  matching the no-H36M intervention. Less capacity = less drift from
  the strong starting point.
- **Both stack:** combining gives 19% MPJPE reduction (302 → 246),
  showing the two interventions act on different failure modes (data
  signal vs adapter capacity).
- **Lower LR (1e-5)** has no measurable effect — the saved best.pt
  comes from epoch 1, before LoRA had time to move at this LR.
- **Biomech ×2** also has no measurable effect — same epoch-1 best.pt.
  Biomech contribution converges within ~4 epochs and then stops
  exerting pressure.

### 4.4 The MPJPE-vs-P-MPJPE trade-off

Improving variants reduce raw MPJPE (which captures absolute scale
alignment) while slightly degrading P-MPJPE (which is scale/rotation-
invariant). Interpretation: LoRA fine-tuning primarily adjusts the
model's output **scale and translation** to match Fit3D's distribution,
not its **relative joint structure**. The pretrained backbone's
geometric priors are hard to improve via LoRA — adapters mostly
re-anchor the global frame.

---

## 5. Per-action analysis

Fit3D s11 has 47 unique actions (32,400+ frames split into 240 windows).
**Per-action P-MPJPE** for the headline method
(**Hybrid + no H36M + rank 2**), source
`outputs/eval/v3_no_h36m_rank2.json`:

### 5.1 Hardest 8 actions (P-MPJPE > 220 mm)
| Action | n | MPJPE | P-MPJPE | BLI |
|---|---|---|---|---|
| pushup | 4 | 358.1 | **264.0** | 0.0070 |
| burpees | 5 | 380.6 | 255.2 | 0.0015 |
| diamond_pushup | 3 | 369.9 | 254.5 | 0.0018 |
| warmup_1 | 10 | 364.0 | 254.2 | 0.0020 |
| mule_kick | 4 | 345.4 | 233.3 | 0.0127 |
| man_maker | 16 | 343.7 | 229.5 | 0.0045 |
| warmup_5 | 2 | 288.6 | 223.9 | 0.0126 |
| warmup_19 | 3 | 375.8 | 218.6 | 0.0019 |

**Pattern:** floor-based exercises (pushup, diamond_pushup, burpees,
mule_kick, man_maker) and dynamic warmups dominate. These involve
horizontal body orientations and heavy self-occlusion — pose configurations
poorly represented in H36M's standing/sitting lab data.

### 5.2 Easiest 8 actions (P-MPJPE < 130 mm)
| Action | n | MPJPE | P-MPJPE | BLI |
|---|---|---|---|---|
| dumbbell_overhead_shoulder_press | 3 | 131.7 | **104.9** | 0.0038 |
| warmup_6 | 5 | 145.1 | 121.8 | 0.0003 |
| dumbbell_curl_trifecta | 14 | 142.7 | 122.4 | 0.0007 |
| neutral_overhead_shoulder_press | 4 | 141.1 | 123.7 | 0.0038 |
| dumbbell_hammer_curls | 3 | 141.2 | 124.9 | 0.0003 |
| warmup_9 | 3 | 148.6 | 125.9 | 0.0191 |
| dumbbell_scaptions | 4 | 148.7 | 126.3 | 0.0029 |
| warmup_15 | 4 | 160.7 | 128.8 | 0.0005 |

**Pattern:** standing dumbbell exercises with primarily upper-body
motion and minimal occlusion — pose configurations that closely match
H36M's representational distribution.

### 5.3 What this tells us

The domain gap is **not uniform** — it's strongly correlated with
deviation from H36M's body-orientation distribution. Easiest-case
P-MPJPE (105 mm) is in the same ballpark as MotionBERT's H36M test
P-MPJPE (37 mm × the ~3× scale-mismatch factor); hardest-case
(264 mm) is 2.5× worse. A future "domain adaptation" method should
focus on the floor/inverted regime specifically.

---

## 6. Failure modes catalogued during development

Recorded for the methodology / discussion section.

1. **IMAR W2C convention** — `pts_world @ R^T + T` (intuitive but wrong)
   produces pixel x = -3863 instead of ~395. Correct convention is
   `(pts_world − T) @ R^T`. Verified against IMAR's own visualization
   notebook and rendering code.
2. **Skeleton landmark mismatch** — IMAR's "thorax" is at mid-chest,
   H36M's joint 8 is at base-of-neck. Direct index mapping introduces
   ~13 cm constant per-frame error. Fixed via shoulder-midpoint thorax.
3. **GPU heterogeneity in shared cluster** — cluster offered AMD MI210
   (CUDA-only PyTorch silently falls back to CPU and OOMs the host),
   16 GB V100 variants (insufficient for batch 16, seq_len 243), and
   RTX 6000 (~22 GB usable, also OOMs). Final exclude list spans 15
   nodes; the remaining V100-32GB / A40 / A100 / L40S / H100 / H200
   pool is sufficient.
4. **PyYAML scientific notation** — `lr: 1e-4` parses as a string
   without an explicit decimal point, breaking AdamW's `0.0 ≤ lr` check.
   Fixed by writing `1.0e-4` and casting to float in the trainer.
5. **Reprojection wiring (v1 → v2)** — original reprojection loss was
   applied unconditionally, output pixels while data was normalized,
   and projected root-relative 3D so X/Z exploded. v2 fix: gate on
   `(~has_3d) & has_reproj`, project with per-sample intrinsics in
   normalized space, recover absolute 3D via saved per-frame `cam_root`.
6. **Wrong pretrained init** — initially loaded `MB_release.bin` (the
   masked-pose-modeling backbone, no 3D head), causing val P-MPJPE to
   start at 610 mm because the 3D head was effectively random. Switched
   to `MB_ft_h36m.bin` (pretrained 3D head) → starts at 280 mm.
7. **Mixed-batch loss safety** — original PoseLoss used
   `if batch.get("has_3d", False)` which raises on multi-element bool
   tensors. Fixed to per-sample masking so mixed H36M / Fit3D-2D batches
   work correctly.
8. **PriorityDecayHalfLife = 0** — PACE-ICE's fairshare doesn't decay,
   meaning waiting doesn't restore priority. Practical consequence:
   queue position is governed by current cluster load, not historical
   usage.

---

## 7. Limitations

1. **Single-camera Fit3D processing** — we use only camera ID 50591643
   for all subjects. Multi-camera would give richer 2D supervision.
2. **No confidence-weighted reprojection** — Fit3D's projected 2D has
   uniform unit confidence; an in-the-wild detector would yield
   per-joint confidence that should down-weight occluded joints.
3. **Skeleton mapping is approximate** — bone-length comparison shows
   torso joints 11-13% off H36M conventions. A fully principled fix
   would use the team's skeleton-bridging autoencoder (or fixed
   pelvis-to-neck ratios for spine/thorax).
4. **15 epochs may be too few or too many** — best.pt landed at epoch
   14-15 for our improving runs (still improving at termination), but
   epoch 1 for the failing runs (low-LR / biomech×2 never moved). A
   longer run might extract more from the working configurations.
5. **No knowledge distillation** — adding a frozen MB_ft_h36m teacher
   with an explicit "stay close to teacher" loss might fix the
   P-MPJPE regression while preserving the MPJPE gain. Not implemented.
6. **2D-only weakly-supervised data is from Fit3D** — a "true"
   in-the-wild test would use detector-derived 2D from gym videos
   (Kaggle, YouTube, etc.). We chose Fit3D-train for clean
   experimental scope; the trade-off is a less ambitious "in-the-wild"
   claim.

---

## 8. Reproducibility

- Code: <https://github.com/...> (this repo, branch `main`).
- Pretrained weights mirrored at
  `huggingface.co/walterzhu/MotionBERT`.
- Training: `sbatch scripts/pace_train.sbatch <epochs> <args...>`.
- Eval: `sbatch scripts/pace_eval_v3.sbatch` evaluates all v3
  checkpoints + zero-shot baseline against H36M and Fit3D s11.
- Result JSONs: `outputs/eval/*.json`.
- All experiments on Georgia Tech PACE-ICE cluster, ice-gpu partition,
  L40S GPUs (Run 5253991 was the headline combined run).

---

## 9. Suggested report structure

1. **Introduction** — domain gap problem, gym/exercise vs lab data.
2. **Related work** — VideoPose3D, MotionBERT, APTPose, ACAE; your
   milestone references.
3. **Method** — Sections 2.1-2.7 above. Include the loss equation and
   the LoRA-adapter architecture.
4. **Experimental setup** — Section 2.5-2.6 (data) + 2.7-2.8
   (preprocessing fixes that motivate the methodology section).
5. **Results** — Section 3 (baselines) + Section 4 (ablation table).
6. **Per-action analysis** — Section 5; this is your qualitative
   contribution.
7. **Discussion / Failure modes** — Section 6 + Section 4.4 (the
   MPJPE / P-MPJPE trade-off finding).
8. **Limitations & Future Work** — Section 7.
