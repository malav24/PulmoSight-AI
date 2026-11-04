# ML Architecture — Right for the Right Reason

*Explanation-Guided Deep Learning for Pneumonia Detection and Localization in Chest X-Rays*

> Status: implemented and tested (Stages 1–3). **No model in this repository has been trained.**
> Every number in this document describes the architecture's *shape* or *size*, never its performance.

---

## 1. Research question

Does supervising **where** a classifier draws its evidence (using RSNA radiologist bounding boxes)
change **how well**, **for what reason** and **how robustly** it detects pneumonia-related lung opacity —
compared with an architecturally identical model trained without that supervision?

The study therefore needs (a) two models that differ *only* in their training objective,
(b) conventional baselines, and (c) evaluation of accuracy, localization, explanation faithfulness,
shortcut robustness, external generalisation and calibration.

## 2. Code map

```
ml/
  models/       hybrid.py (proposed), baselines.py (timm), factory.py
  datasets/     preprocessing.py, transforms.py, rsna.py, nih.py, loaders.py
  training/     losses.py, trainer.py, train_*.py (one per model)
  evaluation/   classification, localization, explanation, shortcut, calibration,
                statistics, generalization, evaluate.py (CLI)
  explainability/ gradcam_pp.py, visualization.py
  inference/    predictor.py (API), quality.py, uncertainty.py
  configs/      base.yaml + one YAML per model
  utils/        config, runtime (seed/device/AMP), checkpoints, logging, synthetic (tests only)
```

Training and inference are separate: `ml/inference` never imports the trainer, and it can only load
weights from a trained checkpoint.

## 3. Proposed hybrid model (`ml/models/hybrid.py`)

```
Input 3×384×384 (gray CXR replicated to 3 channels, ImageNet-normalised)
 │
 ├─ EfficientNet-B0 (timm, features_only)
 │     stride-16 map: 112×24×24 ─────────────────────────────┐ (skip)
 │     stride-32 map: 320×12×12                              │
 │                                                           │
 ├─ 1×1 Conv-BN-GELU → d_model=256 (+ optional AvgPool `token_pool`)   ← "proj" (Grad-CAM++ target)
 ├─ Tokenise: 12×12 = 144 tokens × 256
 ├─ + learned 2-D positional embedding (bicubic-interpolated if the grid changes)
 ├─ Transformer encoder × 2 (pre-norm, 8 heads, FFN 512, dropout 0.1)
 ├─ LayerNorm  ⇒  SHARED REPRESENTATION (144 × 256)
 │
 ├─ HEAD 1 — classification
 │     score_i = w₂ᵀ tanh(W₁ t_i)              (per token)
 │     A = softmax(score)  ∈ Δ¹⁴⁴               ← SPATIAL ATTENTION REPRESENTATION
 │     z = Σ_i A_i t_i  →  LN → Dropout → Linear → logit → sigmoid → probability
 │
 └─ HEAD 2 — localization
       tokens → 256×12×12 → bilinear ×2 → + 1×1(stride-16 skip) → Conv-BN-GELU ×2 → 1×1
       → logits 24×24 → bilinear → 384×384 heatmap logits → sigmoid
```

**Spatial reduction.** Self-attention costs O(N²). Using the stride-32 map gives N = 144 tokens at
384 px (vs 576 at stride 16 or 9,216 at stride 4). `token_pool` can reduce further (2 → 36 tokens).
The localization head recovers resolution through the stride-16 skip connection.

**Forward outputs** (all models return a dict):

| key | shape | meaning |
|---|---|---|
| `logits` | (B,) | classification logit |
| `probs` | (B,) | sigmoid(logit) — *uncalibrated* |
| `loc_logits` / `heatmap` | (B,1,H,W) | localization prediction (hybrid only) |
| `attention` | (B,1,12,12) | spatial attention A, sums to 1 per image |
| `attention_map` | (B,1,H,W) | A upsampled + min-max normalised — **display only** |
| `features` | (B,256,12,12) | shared representation |
| `embedding` | (B,256) | pooled feature (used for OOD statistics) |
| `raw_self_attention` | (B,L,N,N) | optional, diagnostics only |

Size: **5.2 M parameters** (EfficientNet-B0 feature backbone: 3.6 M). Measured forward pass: ≈ 80 ms per 384-px image
on a 2-vCPU cloud container (CPU only; GPU timings must be measured with `scripts/benchmark.py`).

### Why attention pooling, not raw Transformer attention?

Raw multi-head self-attention weights describe how tokens mix with each other. They are not a
measure of how much each region contributes to the decision, and earlier work warns against reading
them as explanations. Here the spatial attention A is the *only* route by which spatial evidence
reaches the logit: z = Σ A_i t_i. Guiding A therefore restricts which (contextualised) tokens the
classifier can draw evidence from.

This is still **not** a guarantee of faithfulness. Each token has already mixed with every other token
in the Transformer, so a token "inside the box" can carry information from outside. This is why
the evaluation checks faithfulness separately (deletion/insertion) instead of assuming it.

## 4. Model A, Model B and the ablation

All three hybrid variants use **one** architecture config (`hybrid_common.yaml`). A test
(`test_model_a_and_b_have_identical_architecture`) checks that their parameter names and counts are
identical. They differ only in loss weights:

| Arm | Config | α (localization) | λ (attention) | Checkpoint key |
|---|---|---|---|---|
| Ablation A — classification only | `ablation_cls_only.yaml` | 0 | 0 | `ablation_cls_only` |
| Ablation B — cls + localization = **Model A** | `model_a.yaml` | 1.0 | 0 | `model_a` |
| Ablation C — cls + loc + guidance = **Model B** | `model_b.yaml` | 1.0 | 0.5 | `model_b` |

In ablation A the localization head exists but is never trained. Its heatmaps are meaningless, and the
evaluator and predictor detect this and do not report them.

Every arm uses the same data split, seed, schedule and early-stopping criterion (validation AUC), so
none of them is tuned on its own objective. For a robust claim, repeat each arm with ≥ 3 seeds
(`experiment.seed=…`).

## 5. Objective (`ml/training/losses.py`)

```
L_total = L_cls + α · L_loc + λ · L_attn
```

* **L_cls**: weighted BCE (`pos_weight = n_neg / n_pos` from the training split, ≈ 3.4 for RSNA) or binary focal loss.
* **L_loc**: Dice (positives only — Dice of an empty mask is undefined) + BCE (optionally including the
  empty masks of negatives, teaching "no opacity here").
* **L_attn** — computed only on positive images that carry boxes. The box mask is area-downsampled to
  the 12×12 grid, giving coverage c_i ∈ [0,1]:
  * `outside_mass` (default): Σ_i A_i (1 − c_i), the attention mass outside the annotation. It
    allows attention to concentrate on part of a box, which suits coarse rectangles.
  * `kl`: KL(T ‖ A) with T = c / Σc — pushes attention to spread over the whole box (stricter).
  * `outside_mass+kl`.
  Negatives receive no attention supervision: there is no defined "right region" for "no opacity".

Losses are computed in fp32 even under mixed precision.

## 6. Baselines (`ml/models/baselines.py`)

| key | timm model | Grad-CAM++ layer | params |
|---|---|---|---|
| `resnet50` | resnet50 | `net.layer4` | 23.5 M |
| `densenet121` | densenet121 | `net.features` | 7.0 M |
| `efficientnet_b0` | efficientnet_b0 | `net.conv_head` | 4.0 M |
| `vit_small` | vit_small_patch16_384 | `net.blocks[-1].norm1` (tokens reshaped to 24×24) | 21.8 M |

These baselines are classification-only, and the trainer refuses α or λ > 0 for them. Their head dropout is an explicit
`nn.Dropout` module (`head_drop`) rather than timm's internal functional dropout, so MC-dropout uncertainty works for
every baseline (Stage-3 fix: ResNet50 and EfficientNet-B0 previously produced zero spread). The
EfficientNet-B0 baseline isolates the effect of adding the Transformer and heads to the same backbone.

## 7. Three kinds of maps — kept distinct

| Map | What it is | Available for |
|---|---|---|
| **Localization heatmap** | supervised *prediction* of where the annotated opacity is | Model A, Model B |
| **Spatial attention** | pooling weights the classifier actually uses | all hybrids |
| **Grad-CAM++** | post-hoc gradient attribution of the logit | all models |

They answer different questions and can disagree. `ml/explainability/visualization.MAP_DESCRIPTIONS`
carries the wording used in every panel and API response.

## 8. Training framework (`ml/training/trainer.py`)

* Device auto-detection (CUDA → MPS → CPU). Mixed precision on CUDA (bf16 if supported, otherwise fp16
  with GradScaler).
* AdamW with separate learning rates for backbone and heads; cosine schedule with linear warm-up (or plateau).
* Gradient clipping, optional backbone freezing for the first N epochs.
* Early stopping on `val_auc`. Saves `<key>_best.pth` (best epoch) and `<key>_last.pth` (with optimizer, scheduler,
  scaler and early-stopping state, for `--resume`).
* CSV log per epoch (`outputs/logs/<key>_training_log.csv`) with loss components, val AUC, val Dice
  (positives), validation attention mass inside the boxes, and the val Youden threshold.
* Reproducible seeds (Python/NumPy/torch/DataLoader workers). `experiment.deterministic: true` for
  bit-level determinism (slower).

## 9. Checkpoints (`ml/utils/checkpoints.py`)

Centralised registry (paths relative to `checkpoints/`, overridable with `RFRR_CHECKPOINT_ROOT`):

```
checkpoints/hybrid/{model_a,model_b,ablation_cls_only}_{best,last}.pth
checkpoints/baselines/{resnet50,densenet121,efficientnet_b0,vit_small}_{best,last}.pth
checkpoints/calibration/temperature_scaling.json
checkpoints/calibration/<key>_ood_stats.npz
```

Calibration (temperature, validation threshold) and OOD statistics record the `saved_at`/`epoch` of the checkpoint they
were fitted on; the predictor ignores them (with a warning) for any other checkpoint, so retraining can never silently
reuse stale calibration.

Loading rules: the architecture is rebuilt with `pretrained_backbone=False`, and **all** weights come
from the checkpoint. Loading is strict, the stored `model_key` is verified, and a missing file raises
`CheckpointNotFoundError`. There is no fallback to random or ImageNet weights. Each checkpoint stores its full
config, preprocessing, loss weights, epoch, metrics and validation threshold. Saving is atomic
(write to a temporary file, then rename).

## 10. Inference API (`ml/inference/predictor.py`)

`PneumoniaPredictor(model_key).predict(path_or_array)` returns a JSON-serialisable dict containing:
the probability (raw and temperature-scaled), the threshold (validation Youden J from the evaluation run, else from the
best training epoch), any configuration warnings, the prediction or an
abstention with reasons, MC-dropout uncertainty (std, 5–95 % interval, entropy, mutual information), an OOD
warning (Mahalanobis distance against RSNA training embeddings, threshold at the 99th validation
percentile), quality-check results, the three maps (with optional base64 overlays), and a disclaimer.
Helper functions: `load_model`, `generate_localization_map`, `generate_attention_map`, `generate_gradcam`,
`calculate_uncertainty`, and `to_json_safe`, ready for FastAPI in Stage 2.

## 11. Known limitations of the design

* RSNA boxes are rectangles around opacities, not lesion segmentations. Guidance teaches "inside the
  rectangle", which includes some normal lung tissue.
* The attention guidance applies to positives only. On negatives the model is free to attend anywhere.
* Token mixing in the Transformer means guided attention is necessary but not sufficient for
  "right reason". Faithfulness must be measured (§ evaluation).
* Single-label, single-view (frontal) design. Lateral views are out of scope.
* No lung segmentation model is included, so the "outside-lung" stress test uses a geometric thorax
  ellipse (clearly labelled as such).
