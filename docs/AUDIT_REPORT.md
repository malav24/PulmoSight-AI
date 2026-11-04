# Stage-3 Audit Report

Date: 2026-11-03. Scope: every Python module, configuration, script, FastAPI endpoint, React page and document.
**No model was trained in this audit.** Every "trained" checkpoint mentioned below was trained for a few seconds on
synthetic noise images, only to exercise code paths. Its numbers are meaningless and it is not part of the delivery.

## 1. Problems found and fixed

| # | Area | Problem | Severity | Fix | Verified by |
|---|---|---|---|---|---|
| 1 | Uncertainty | ResNet50 and EfficientNet-B0 apply head dropout through timm's *functional* `F.dropout`, which MC-dropout cannot switch on. Their uncertainty was always **0** (reproduced: std = 0.0), so the abstention rule could never fire for them | High | Baselines use an explicit `nn.Dropout` head module (`ml/models/baselines.py`) | `test_mc_dropout_produces_spread_for_all_models` (all 5 architectures) |
| 2 | Calibration | Temperature, threshold and OOD statistics were stored per *model name*. After retraining, inference would silently apply the previous checkpoint's calibration | High (scientific) | Files record the checkpoint's `saved_at`/`epoch`; the predictor ignores mismatches and warns; the API/UI show "stale" | `test_stale_calibration_is_ignored`; UI check "no configuration warning" |
| 3 | Threshold | Inference used the training-time threshold (possibly from a subset of validation batches) while the evaluation reported a different validation threshold | Medium | The evaluation stores its validation Youden threshold with the calibration; inference uses it when it matches the checkpoint; the source is shown | pipeline test asserts `fitted_on == "rsna_val"` and the threshold source |
| 4 | Calibration | A degenerate validation set can drive the fitted temperature to absurd values (seen: T ≈ 30,000 on synthetic data) | Medium | T bounded to [0.05, 20]; reaching a bound is logged, stored and shown as "unreliable" | `test_calibration_metrics_and_temperature`; end-to-end re-evaluation |
| 5 | Dependencies | `torch.amp.GradScaler("cuda")` requires PyTorch ≥ 2.3 but `requirements.txt` allowed 2.1; albumentations 1.x has a different API | Medium | Fallback to `torch.cuda.amp.GradScaler`; requirements pinned to tested ranges (`torch>=2.3`, `albumentations>=2.0,<3`, `timm>=1.0`); backend deps merged into the root file | full suite on Python 3.10 and 3.11 |
| 6 | Configuration | `.env.example` told users to create `.env`, but nothing read it | Medium | Dependency-free `.env` loader in `ml/utils/config.py` (real environment variables take precedence); the API settings load it first | `test_unicode_paths_and_dotenv` |
| 7 | Evaluation | `evaluate.py` used the dataset paths stored in the checkpoint and ignored `RFRR_RSNA_ROOT`/`RFRR_NIH_ROOT` (it breaks if data moves, e.g. Colab → PC) | Medium | Environment overrides re-applied to the checkpoint config | code review; CLI runs with env roots |
| 8 | Windows | `cv2.imread/imwrite` fail on non-ASCII paths (e.g. an accented Windows user name in the temp folder used by uploads) | Medium | Byte-buffer `imread_any_path` / `imwrite_any_path` used for rasters, the cache and uploads | `test_unicode_paths_and_dotenv` |
| 9 | Explainability | No command-line way to produce Grad-CAM++/localization panels for chosen images | Low | `scripts/explain_image.py` (clean error when untrained) | manual runs: hybrid, baseline, missing checkpoint |
| 10 | Tooling | No model status overview or hardware benchmark | Low | `scripts/model_status.py`, `scripts/benchmark.py` | runs in the audit |
| 11 | Docs | Unmeasured claims ("≈ 70 ms on a laptop", "4–6 GB GPU memory", "~10× faster", download sizes stated as exact) | Medium (honesty) | Replaced with measured values (with the hardware named) or removed | grep audit of all docs |
| 12 | Docs | README/TRAINING_GUIDE written for Stage 1 only; no PowerShell sequence; methodology, architecture and audit docs missing | — | Rewritten / created to match the actual commands | commands checked against `--help` of every script |
| 13 | Packaging | `backend/` had no `__init__.py` (it worked only as an implicit namespace package) | Low | added | API tests |

Earlier fixes during Stage 2 (kept): corrupt PNG → 415 instead of 500; duplicate legends; capitalisation; chart colours.

### Checked and found correct (no change needed)
Tensor shapes of all models at 64/128/384 px; positional-embedding interpolation; loss composition; Dice on positives only;
attention loss only on positives with boxes; pos_weight from the training split only; augmentation only on train;
box → resize → mask geometry (stretch and letterbox); mask/box alignment under augmentation; strict checkpoint loading;
registry paths identical for trainer and predictor; Grad-CAM++ target layers (including the ViT token reshape); DeLong
implementation (matches sklearn AUC; identical models → p = 1); API ↔ frontend field names; upload clean-up.

### Investigated, not a bug
The new localization-alignment test first failed at 64 px. Diagnostics showed the heatmap correctly concentrated inside
each box (mean 0.75–0.80 inside vs 0.14–0.16 outside; centre of mass near the box centre). At 64 px the decoder grid is
4×4, so the high plateau touched the image edge and `argmax` landed on an edge pixel. At 128 px the peak was inside both
boxes. The test now uses 128 px and checks the peak, the inside/outside contrast and the centre of mass.

## 2. Scientific validation (items A–L)

| | Check | Result | Evidence |
|---|---|---|---|
| A | RSNA annotations used correctly | ✔ `Target`/boxes consistency enforced (error on mismatch); 3-class stratification; labels described as opacity annotations, not diagnoses | `parse_rsna_labels`, `test_parse_rsna_labels` |
| B | Boxes transformed correctly | ✔ same geometry as the image (stretch/letterbox); clipping; pixel-centre rasterisation | `test_box_transform_matches_image_geometry`, `test_boxes_to_mask_exact_and_union` |
| C | Patient-level split, no leakage | ✔ split on unique patientId with assertion; reproducible; only train is augmented; pos_weight, threshold and temperature never from test | `test_patient_level_splits_are_disjoint` |
| D | Model A vs B comparable | ✔ identical model/data/training/loader/seed configs; only λ differs; identical state-dict keys | `test_model_a_b_differ_only_in_guidance_weight`, `test_model_a_and_b_have_identical_architecture` |
| E | Guided model receives the attention loss | ✔ λ > 0 changes attention-head gradients; logged `train_loss_attn` > 0 for B, 0 for A | `test_attention_guidance_changes_gradients_of_attention_head`, `test_training_log_records_attention_loss_only_for_model_b` |
| F | Localization aligned with masks | ✔ peak inside the correct box for opposite-corner targets; same coordinates used for metrics | `test_localization_head_learns_correct_location`, `test_rsna_dataset_item_mask_matches_boxes` |
| G | NIH evaluated separately, never trained on | ✔ training builds RSNA datasets only; NIH has no augmentation; RSNA overlap excluded via the mapping file or the run is flagged | `test_nih_overlap_exclusion_and_warning` |
| H | No artificial NIH boxes | ✔ only official BBox_List_2017 Pneumonia boxes | `test_nih_dataset_items_use_only_official_boxes` |
| I | Calibration on validation, not test/external | ✔ fitted on RSNA val, bound to the checkpoint, applied unchanged to test/NIH | `test_stale_calibration_is_ignored`, pipeline test |
| J | Documented, consistent deletion/insertion | ✔ fixed images (first N test positives), blur baseline σ = max(2, H/24), K+1 steps, trapezoid AUC; deterministic | `test_deletion_insertion_is_deterministic`, RESEARCH_METHODOLOGY §9 |
| K | Paired statistics | ✔ DeLong on identical image ids (inner join), Holm correction; unpaired bootstrap for RSNA vs NIH; patient-clustered CIs on NIH | `test_delong_matches_sklearn_and_paired_test` |
| L | No fabricated results | ✔ no metrics in code, docs or UI; the UI renders only API values read from files; "Models not trained yet" / "No experiment results available" otherwise | UI scenario 1 checks; doc grep |

## 3. Test results

| Suite | Environment | Result |
|---|---|---|
| ML + backend pytest (91 tests: models, preprocessing, datasets, losses, checkpoints, metrics, explainability, pipeline, scientific, API) | Python 3.11.18, torch 2.15, timm 1.0.30, albumentations 2.0.8, pydicom 3.0.2, NumPy 2.5 | **91 passed, 0 failed** |
| same | Python 3.10.20, torch 2.14.1, NumPy 2.2.6 | **91 passed, 0 failed** |
| Browser smoke test `tests/e2e/ui_smoke.cjs` (Chromium, desktop 1440 px + iPhone-13 viewport, 2 scenarios) | Vite production build + FastAPI | **50 / 50 checks passed** |
| Frontend production build | Node 22, Vite 6.4.5, React Router 7.18.5 | builds without errors; `npm audit`: 0 vulnerabilities |

**Not executed** (not possible in this environment):
* training or inference with a *genuinely* trained checkpoint (none exists; datasets are not on this machine);
* GPU execution, CUDA mixed precision and GPU memory (no GPU available; the CUDA code paths are exercised only by review);
* real RSNA/NIH loading (verified with synthetic files in the official formats);
* the app on native Windows (code reviewed for Windows: path handling, multiprocessing guards, PowerShell commands);
* Python 3.12+ (not targeted).

## 4. Performance measurements (CPU only, 2 vCPU / 8 GB cloud container)

| Measurement | Value |
|---|---|
| Hybrid tokens at 384 px | 144 (12×12) |
| Parameters | hybrid 5.2 M (backbone 3.6 M); ResNet50 23.5 M; DenseNet121 7.0 M; EfficientNet-B0 4.0 M; ViT-Small 21.8 M |
| Hybrid training step, 384 px, fp32, CPU | batch 4: 1.3 s; 8: 2.6 s; 16: 6.4 s (≈ 3 images/s); peak process RAM 2.2 / 3.4 / 5.3 GB |
| Inference per image, hybrid (CPU) | forward 0.08 s; 20 MC-dropout passes 1.6 s; Grad-CAM++ 0.4 s |
| Inference per image, ResNet50 / ViT-Small (CPU) | forward ≈ 0.2 s; MC-dropout ≈ 4.1 s; Grad-CAM++ ≈ 0.7 s |
| Model construction | 0.1–0.3 s (plus checkpoint read and a one-time SHA-256 in the API) |

GPU speed and memory were **not** measured. Run `python scripts/benchmark.py` on your GPU. Configuration advice for
Colab/Kaggle is in TRAINING_GUIDE §6. To make the API faster on CPU, lower `RFRR_MC_DROPOUT_SAMPLES`.

## 5. Model status at delivery

`python scripts/model_status.py --build`, run on the delivered project:

| model | architecture | training script | checkpoint | calibration | ready for inference |
|---|---|---|---|---|---|
| model_a | implemented (5.2 M) | ✔ | missing | missing | no |
| model_b | implemented (5.2 M) | ✔ | missing | missing | no |
| ablation_cls_only | implemented (5.2 M) | ✔ | missing | missing | no |
| resnet50 | implemented (23.5 M) | ✔ | missing | missing | no |
| densenet121 | implemented (7.0 M) | ✔ | missing | missing | no |
| efficientnet_b0 | implemented (4.0 M) | ✔ | missing | missing | no |
| vit_small | implemented (21.8 M) | ✔ | missing | missing | no |

## 6. Remaining limitations (not fixable in code)

* Training, evaluation and every research result are pending your GPU runs.
* No lung-segmentation model: the outside-lung test uses a geometric ellipse.
* The NIH test is same-institution with a different, noisy label. A truly external multi-site dataset would strengthen the study.
* The API keeps one job at a time and runs inference on the same GPU as any training job started from the UI. On a small
  GPU, train from the command line while the API is stopped.
* Subgroup/fairness analysis is not implemented.

## 7. Follow-up: npm audit (after delivery)

`npm install` on Windows reported 6 advisories (5 high, 1 moderate) in pinned frontend dependencies:
react-router (open redirect / XSS; SSR-only constructor injection), postcss (source-map file disclosure, CSS
escaping) and esbuild via vite (dev-server request forgery). None affected the production bundle's logic in this app
(no SSR, no user-controlled redirects), but all were resolved by upgrading: `react-router-dom` 6.28.0 → 7.18.5,
`vite` 5.4.11 → 6.4.5, `@vitejs/plugin-react` 4.3.4 → 4.7.0, `postcss` 8.4.49 → 8.5.32. Result: `npm audit` reports
**0 vulnerabilities**; the build succeeds; the browser smoke test passes 50/50 in three consecutive runs.
React Router 7 renders navigations inside React transitions, which exposed two timing assumptions in the smoke test
(an over-broad link-name match and immediate visibility checks); the test now waits for visibility and matches exactly.
`package.json` declares `"engines": {"node": ">=20.0.0"}`.
