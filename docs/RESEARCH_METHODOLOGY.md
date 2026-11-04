# Research Methodology

**RIGHT FOR THE RIGHT REASON: Explanation-Guided Deep Learning for Pneumonia Detection and Localization in Chest X-Rays**

> This document describes the protocol *as implemented in the code*. No results are reported here. None exist
> until the models are trained and evaluated.

## 1. Problem

Deep networks for chest radiographs can reach high internal accuracy while relying on non-clinical cues
(text markers, laterality tokens, borders, devices, acquisition differences): *shortcut learning*. Such models may
fail on new hospitals or patient groups, and post-hoc heatmaps may then explain *what* the model used without changing it.

## 2. Research gap

Most pneumonia classifiers are optimised and reported on accuracy alone. Explanations are generated only afterwards.
Studies that (a) **use radiologist localisation to supervise where a model attends during training**, (b) compare against
an architecturally identical unguided model, and (c) evaluate faithfulness, shortcut robustness, calibration and
cross-dataset behaviour within the same protocol are scarce.

## 3. Aim and objectives

**Aim:** to develop and evaluate an explanation-guided hybrid CNN–Transformer that detects and localizes
pneumonia-related lung opacity, and to test whether attention guidance makes it "right for the right reason".

**Objectives**
1. Implement a hybrid EfficientNet-B0 + Transformer with classification and localization heads.
2. Train it without (Model A) and with (Model B) attention guidance from RSNA radiologist boxes.
3. Benchmark against ResNet50, DenseNet121, EfficientNet-B0 and ViT-Small.
4. Evaluate classification, localization, explanation quality, shortcut robustness, NIH generalisation and calibration.
5. Provide an interpretable interface with explicit limitations.

## 4. Research questions and hypotheses

| | Question | Hypothesis (to be tested, not assumed) | Primary measure |
|---|---|---|---|
| RQ1 | Does guidance move the model's evidence into annotated regions? | **H1:** Model B > Model A in attention mass-in-box and Grad-CAM++ pointing game | attention mass-in-box (RSNA test positives) |
| RQ2 | Does guidance change classification performance? | **H2:** Model B's AUC is not meaningfully lower than Model A's | ROC-AUC, paired DeLong |
| RQ3 | Is the guided model less sensitive to shortcut perturbations? | **H3:** smaller AUC drop / prediction change for Model B | stress-test ΔAUC, mean \|Δp\| |
| RQ4 | Does guidance reduce the RSNA→NIH performance gap? | **H4:** smaller AUC gap for Model B | AUC difference with bootstrap CI |
| RQ5 | Is the guided model better calibrated? | exploratory | ECE, Brier (before/after temperature scaling) |

A non-inferiority margin for H2 must be fixed **before** test-set evaluation (e.g. ΔAUC > −0.02) and reported.

## 5. Data

### 5.1 RSNA Pneumonia Detection Challenge (training, validation, internal test)
* Images: frontal CXR DICOMs. Labels: `Target=1` ⇔ class "Lung Opacity" with radiologist bounding boxes; the
  classes "Normal" and "No Lung Opacity / Not Normal" are `Target=0`.
* **Interpretation:** pneumonia-related lung-opacity annotations, *not* confirmed clinical diagnoses.
* **Split:** 70 / 15 / 15 train / validation / test, at patient level (`patientId`), stratified on the three-class
  label, seed 42 (`ml/datasets/rsna.py::make_patient_splits`, verified disjoint by assertion and tests). The unlabelled
  official test images are not used.
* Boxes: `(x, y, width, height)` → `(x1, y1, x2, y2)` in original pixels. They are mapped with exactly the same resize geometry
  as the image and rasterised (pixel-centre rule) into a binary mask. During augmentation the mask is transformed
  together with the image (nearest-neighbour), so the two stay pixel-aligned.

### 5.2 NIH ChestX-ray14 (external test only)
* Positive = "Pneumonia" in the NLP-mined `Finding Labels`. Negative = every other image (default) or only "No Finding".
* Official patient-level `test_list.txt`, frontal views (PA/AP). With limited disk space the official ~5 % random
  sample (`nih-chest-xrays/sample`) may be used instead; it has no test list (all images are unseen by the model) and no
  boxes, and its small number of Pneumonia cases widens the external confidence intervals. State which variant was used.
* **RSNA images are a subset of NIH ChestX-ray14.** All NIH images listed in RSNA's published mapping file are
  removed. Without the file the run is flagged as contaminated.
* Never used for training, model selection, threshold choice or calibration. Only NIH's own ~120 expert Pneumonia
  boxes (BBox_List_2017) are used for localization; no boxes are generated.

### 5.3 Preprocessing (identical for training, evaluation and inference)
DICOM rescale slope/intercept → MONOCHROME1 inversion → 0.5–99.5 percentile window → 8-bit → resize to 384×384 (stretch;
letterbox optional) → CLAHE (clip 2.0, 8×8 tiles) → 3 channels, ImageNet normalisation. The training augmentation is
affine only: ±7° rotation, ±8 % scale, ±5 % shift, plus brightness/contrast ±0.15. Horizontal flip is **off** (thoracic anatomy is asymmetric).

## 6. Models

| Model | Architecture | Training objective |
|---|---|---|
| Ablation A | Hybrid (below) | classification only (α = 0, λ = 0) |
| **Model A** (= ablation B) | Hybrid | classification + localization (α = 1, λ = 0) |
| **Model B** (= ablation C) | Hybrid | classification + localization + attention guidance (α = 1, λ = 0.5) |
| ResNet50 / DenseNet121 / EfficientNet-B0 / ViT-Small | timm, single-logit head | classification only |

**Hybrid:** EfficientNet-B0 (stride-16 and stride-32 features) → 1×1 projection to 256 channels → 12×12 = 144 tokens
+ learned 2-D positional embedding → 2 pre-norm Transformer layers (8 heads) →
(1) spatial attention pooling `A = softmax(score(tokens))`, `z = Σ A_i t_i` → logit;
(2) decoder with stride-16 skip connection → heatmap at input resolution. 5.2 M parameters.

## 7. Training objective

`L = L_cls + α·L_loc + λ·L_attn`
* `L_cls`: BCE weighted by `pos_weight = n_neg / n_pos` from the **training** split (or focal loss).
* `L_loc`: soft Dice (positives only) + BCE (all images; empty masks for negatives).
* `L_attn` (positives with boxes only): `Σ_i A_i (1 − c_i)`, where `c_i` is the box coverage of token cell *i*
  (area-downsampled mask). It is the attention mass outside the annotation. An alternative `KL(c/Σc ‖ A)` is available.

Verified by tests: Model A and Model B configs differ only in λ; λ > 0 changes the gradients of the attention head;
Model B's logged `train_loss_attn` is > 0 while Model A's is 0; a localization head trained on synthetic boxes in
opposite corners peaks inside the correct box (tests/test_scientific.py).

## 8. Training protocol

AdamW (head lr 3e-4, backbone lr 1e-4 for hybrids; per-baseline values in configs), cosine schedule with warm-up,
gradient clipping 1.0, mixed precision on CUDA, batch 16, ≤ 30 epochs, early stopping (patience 6) on **validation
ROC-AUC for every model** (the same criterion, so no model is selected on its own objective). ImageNet initialisation
for training. Seeds are fixed. Recommended: ≥ 3 seeds per arm for the A-vs-B comparison.

## 9. Evaluation protocol (`python -m ml.evaluation.evaluate`)

1. Predictions on RSNA validation and test (and NIH when requested) are saved per image.
2. **Temperature scaling** is fitted on RSNA **validation** logits (bounded T ∈ [0.05, 20]; reaching a bound is flagged).
   The **decision threshold** is Youden's J on RSNA **validation**. Both are stored with the identity of the evaluated
   checkpoint and applied unchanged to test and NIH. Inference ignores them if the checkpoint has changed.
3. **Classification:** ROC-AUC, sensitivity, specificity, precision, F1, confusion matrix.
4. **Localization** (hybrids with a trained localization head; RSNA test positives): IoU and Dice of heatmap ≥ 0.5 vs box
   mask, pointing game, mass-in-box.
5. **Attention alignment** (hybrids): attention mass inside the box, KL to the box distribution, attention pointing game.
6. **Explanation faithfulness: deletion / insertion** (documented procedure, `ml/evaluation/explanation.py`):
   * images: the first *N* positive images of the RSNA test split in patientId order (`--explain-n`, default 32;
     64 recommended), identical for every model;
   * saliency: Grad-CAM++ on the positive-class logit, upsampled to 384×384;
   * pixels are ranked by saliency (descending). At *K* + 1 equally spaced fractions (`--explain-steps`, default 20 → 0 %,
     5 %, …, 100 %) the top-ranked pixels are replaced by the **baseline image**: a Gaussian blur of the same
     normalised image with σ = max(2, H/24) px (`baseline="blur"`; `"mean"` and `"zero"` are available);
   * deletion curve = p(positive) as top pixels are blurred; insertion curve = p(positive) as top pixels are restored
     into the blurred image; scores are areas under the curves (trapezoid rule over the fraction).
     Lower deletion AUC and higher insertion AUC indicate more faithful saliency;
   * the procedure is deterministic (tested) and must use the same N, steps and baseline for all models.
7. **Shortcut stress tests** (RSNA test): outside-thorax masking (geometric ellipse proxy: centre (0.5, 0.52), radii
   (0.40, 0.42) of width/height), border masking (outer 10 %), heuristic marker masking (small saturated components in
   the outer 20 %), background Gaussian noise (σ = 0.5 in normalised units) outside the ellipse. Reported: AUC, ΔAUC,
   mean |Δp|, decision-flip rate.
8. **Generalisation:** RSNA test AUC vs NIH AUC. Unpaired bootstrap CI of the difference; NIH CIs resample patients.
9. **Calibration:** ECE (15 equal-width bins and adaptive), Brier score, reliability diagrams, before and after scaling.

## 10. Statistical analysis

* 95 % percentile bootstrap CIs (2000 resamples, seed 42); patient-clustered for NIH.
* **Paired** model comparisons on identical test images: DeLong's test for correlated ROC curves
  (`scripts/compare_models.py`, inner join on image id), Holm correction across comparisons.
* RSNA vs NIH (different images): unpaired bootstrap of the AUC difference. DeLong is not applicable.
* Primary comparison: Model B vs Model A. Others are secondary.

## 11. Threats to validity

* **Construct:** RSNA labels are opacity annotations, not diagnoses. Boxes are coarse rectangles with reader variability.
  "Inside the box" includes normal tissue.
* **Internal:** single split; seed variance (mitigate with ≥ 3 seeds); hyper-parameters not tuned per model
  (tuning must be validation-only and reported).
* **External:** NIH shares the institution with RSNA and uses a different, noisy label definition, so the NIH test measures
  label/prevalence/case-mix shift more than domain shift.
* **Explanation:** token mixing in the Transformer means guided attention is not proof of causal use. Deletion/insertion
  perturbations are out of distribution; plausibility ≠ faithfulness.
* **Stress tests:** thorax ellipse and marker heuristic are approximations; masking itself shifts the input distribution.

## 12. Ethics

Public, de-identified datasets under their licences; no patient interaction. The software is not a medical device and
displays a disclaimer on every page and report. The web application deletes uploads after processing and stores only a
hash and derived 384-px images. Predictions are never presented as diagnoses. The system abstains when uncertain,
when an input looks out of distribution, or when it fails the quality checks. Subgroup performance (sex, age, view) was not
assessed and should be before any further use.

## 13. Reproducibility

Configs in `ml/configs/`, fixed seeds, saved split file, checkpoints with full config and metadata, per-image prediction
CSVs, and evaluation JSONs. `python scripts/model_status.py` lists which artefacts exist.
