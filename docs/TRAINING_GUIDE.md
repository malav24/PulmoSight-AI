# Training Guide (for beginners)

> Nothing in this repository has been trained. This guide shows how **you** train each model and how to read
> what happens. It does not report results, and it cannot predict how long training will take on your hardware.
> Measure that yourself with `scripts/benchmark.py`.

## 0. Before you start (checklist)

1. Python environment activated and dependencies installed (README sections 10–11).
2. `python scripts\verify_setup.py` shows **cuda_available: true** if you want GPU training.
3. RSNA data in `data\rsna\` (README section 9).
4. Splits and cache created once:
   ```powershell
   python scripts\prepare_rsna.py --cache --workers 4      # add --download to fetch it from Kaggle
   $CACHE = "data.rsna.cache_dir=data/rsna/cache_384_clahe"
   ```
5. **2-minute smoke test** (checks that everything runs, numbers are meaningless):
   ```powershell
   python -m ml.training.train_model_b --epochs 1 --batch-size 4 training.max_train_batches=5 training.max_val_batches=3 $CACHE
   ```
   Then delete the smoke-test checkpoints: `Remove-Item checkpoints\hybrid\model_b_*.pth`.

Training downloads ImageNet starting weights from Hugging Face the first time (internet needed). If that
download fails, training **stops**. It never silently starts from random weights.

## 1. How a training run works

```
config YAML  ──►  RSNA train split (augmented) ──► model ──► loss ──► AdamW update
                  RSNA val split  (no augmentation) ──► val AUC, losses, Dice, attention-in-box
every epoch:  CSV log row  ·  <model>_last.pth (for resume)  ·  <model>_best.pth (if val AUC improved)
early stopping: stop when val AUC has not improved for `early_stopping_patience` epochs
```
All models use the **same** split, seed, schedule and selection rule (validation AUC), so comparisons are fair.

## 2. The models

Each command is run from the project folder with the virtual environment active. Append `$CACHE` if you built the
cache. All commands accept `--epochs N`, `--batch-size B`, `--device cuda|cpu`, `--resume` and dotted overrides
(`key=value`, e.g. `training.lr=0.0002`).

### 2.1 ResNet50 (baseline)
* **What it does:** a 50-layer convolutional network that outputs one score (opacity present or not). No localization head.
  Explanations come from Grad-CAM++.
* **Dataset:** RSNA train/val splits (labels only, boxes not used).
* **Configuration:** `ml/configs/resnet50.yaml` (lr 1e-4, batch 16, max 30 epochs, patience 6).
* **Train:** `python -m ml.training.train_resnet50 $CACHE`
* **Checkpoint:** `checkpoints/baselines/resnet50_best.pth` (and `resnet50_last.pth`)
* **Evaluate:** `python -m ml.evaluation.evaluate --model resnet50 --stress --fit-ood --explain-n 64`

### 2.2 DenseNet121 (baseline)
* **What it does:** densely connected CNN, the backbone used by CheXNet-style chest X-ray classifiers. Score only.
* **Dataset / configuration:** RSNA; `ml/configs/densenet121.yaml`.
* **Train:** `python -m ml.training.train_densenet121 $CACHE`
* **Checkpoint:** `checkpoints/baselines/densenet121_best.pth`
* **Evaluate:** `python -m ml.evaluation.evaluate --model densenet121 --stress --fit-ood --explain-n 64`

### 2.3 EfficientNet-B0 (baseline)
* **What it does:** the same backbone as the hybrid models, *without* the Transformer and heads. It shows what the hybrid design adds.
* **Dataset / configuration:** RSNA; `ml/configs/efficientnet_b0.yaml`.
* **Train:** `python -m ml.training.train_efficientnet $CACHE`
* **Checkpoint:** `checkpoints/baselines/efficientnet_b0_best.pth`
* **Evaluate:** `python -m ml.evaluation.evaluate --model efficientnet_b0 --stress --fit-ood --explain-n 64`

### 2.4 ViT-Small (baseline)
* **What it does:** a pure Vision Transformer (16×16 patches at 384 px, i.e. 576 tokens). Score only.
* **Dataset / configuration:** RSNA; `ml/configs/vit_small.yaml` (lower lr 5e-5, weight decay 0.05, 2 warm-up epochs,
  as Transformers usually need).
* **Train:** `python -m ml.training.train_vit $CACHE`. Uses the most memory per image; start with `--batch-size 8` on 8–12 GB GPUs.
* **Checkpoint:** `checkpoints/baselines/vit_small_best.pth`
* **Evaluate:** `python -m ml.evaluation.evaluate --model vit_small --stress --fit-ood --explain-n 64`

### 2.5 Hybrid Model A (proposed, no guidance)
* **What it does:** EfficientNet-B0 features → 144 tokens → 2-layer Transformer → (1) attention-pooled score and
  (2) localization heatmap. It learns to classify and to draw the radiologist boxes (α = 1), **without** attention guidance (λ = 0).
* **Dataset:** RSNA labels **and** radiologist boxes (rasterised to masks).
* **Configuration:** `ml/configs/model_a.yaml` (inherits `hybrid_common.yaml`).
* **Train:** `python -m ml.training.train_model_a $CACHE`
* **Checkpoint:** `checkpoints/hybrid/model_a_best.pth`
* **Evaluate:** `python -m ml.evaluation.evaluate --model model_a --stress --fit-ood --explain-n 64`

### 2.6 Hybrid Model B (proposed, explanation-guided)
* **What it does:** identical to Model A, plus an extra loss term (λ = 0.5) that penalises classifier attention outside
  the radiologist boxes on positive images.
* **Dataset / configuration:** RSNA labels and boxes; `ml/configs/model_b.yaml`.
* **Train:** `python -m ml.training.train_model_b $CACHE`
* **Checkpoint:** `checkpoints/hybrid/model_b_best.pth`
* **Evaluate:** `python -m ml.evaluation.evaluate --model model_b --stress --fit-ood --explain-n 64`
* Check that `train_loss_attn` in its log is **> 0** (for Model A it is always 0). That is the guidance signal.

### 2.7 Ablation arm (optional, recommended)
`python -m ml.training.train_ablation_cls_only $CACHE`: same hybrid, classification only (α = λ = 0).
Checkpoint `checkpoints/hybrid/ablation_cls_only_best.pth`. Or run the three arms in one go:
`python scripts\run_ablation.py --train $CACHE`.

## 3. Key settings explained

| Setting | Where | Default | Meaning / advice |
|---|---|---|---|
| **GPU vs CPU** | `--device` / `experiment.device` | auto | `auto` picks CUDA → Apple MPS → CPU. CPU training at 384 px is only practical for smoke tests (measured on a 2-vCPU machine: about 3 images/s). |
| **Mixed precision** | `training.amp` | true | On NVIDIA GPUs it uses bfloat16 if supported, else float16 with loss scaling. It saves memory and time. Ignored on CPU. |
| **Batch size** | `--batch-size` / `loader.batch_size` | 16 | Images per step. If you get "CUDA out of memory", halve it. Larger batches need more memory but not necessarily better results. |
| **Epochs** | `--epochs` / `training.epochs` | 30 | Maximum passes over the training set. Early stopping usually ends earlier. |
| **Learning rate** | `training.lr` (heads), `training.backbone_lr` | 3e-4 / 1e-4 (hybrids) | Step size of AdamW. Cosine decay with 1 warm-up epoch. If val loss explodes, lower it ×0.5. |
| **Early stopping** | `training.early_stopping_patience` | 6 | Stop after 6 epochs without a better validation AUC. The best epoch is saved as `_best.pth`. |
| **Workers** | `loader.num_workers` | 4 | Parallel image loading. On Windows, if you see multiprocessing errors, use `loader.num_workers=0`. |
| **Image size** | `data.preprocessing.image_size` | 384 | Changing it requires rebuilding the cache (`prepare_rsna.py --cache data.preprocessing.image_size=512`). |
| **Seed** | `experiment.seed` | 42 | Same seed means the same split and initialisation. Use several seeds for the final comparison (§7). |

Changing hyper-parameters: tune **only** on the validation split, with the same rule for all models, and report every setting you tried.

## 4. Resume training

Training saves `<model>_last.pth` after every epoch (weights, optimizer, scheduler, AMP scaler, early-stopping counter).
If the run is interrupted, use the same command plus `--resume`:
```powershell
python -m ml.training.train_model_b --resume $CACHE
```
To train for longer than planned: `--resume --epochs 40`.

## 5. Reading the training log

`outputs/logs/<model>_training_log.csv`, one row per epoch (open it in Excel, or see the charts on the
**Research Experiments** page):

| Column | Meaning | What you want to see |
|---|---|---|
| `train_loss`, `val_loss` | total loss | both decrease; a widening gap (val rising) means overfitting |
| `train_loss_cls` | classification part | decreases |
| `train_loss_loc` | localization part (hybrids, α>0) | decreases |
| `train_loss_attn` | attention-guidance part | **0 for Model A**, > 0 and decreasing for Model B |
| `val_auc` | validation ROC-AUC (selection criterion) | rises, then plateaus |
| `val_dice_pos` | localization Dice on validation positives | rises (RSNA boxes are coarse, so moderate values are normal) |
| `val_attn_mass_in_box` | share of classifier attention inside boxes | higher for Model B than Model A is the expected effect of guidance |
| `val_threshold_youden` | decision threshold at this epoch | informative only |
| `is_best` | 1 if this epoch was saved as best | |

## 6. Hardware recommendations (Kaggle and Google Colab)

These are starting points, not guarantees. Run `python scripts/benchmark.py` on the actual GPU and keep peak memory
about 15–20 % below the GPU's capacity.

**Google Colab (T4, 16 GB; bfloat16 not supported, so float16 AMP is used automatically)**
```python
# Colab cell - project uploaded to Drive at MyDrive/rfrr
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/rfrr/right-for-the-right-reason
!pip install -q -r requirements.txt          # Colab already has a CUDA PyTorch
!python scripts/verify_setup.py
!python scripts/benchmark.py --models model_b vit_small --batch-sizes 16 32
!python scripts/prepare_rsna.py --cache --workers 2
!python -m ml.training.train_model_b loader.num_workers=2 data.rsna.cache_dir=data/rsna/cache_384_clahe
```
* Colab sessions disconnect. Keep the project (and therefore `checkpoints/`) on Drive and continue with `--resume`.
* Reading many small files from Drive is slow. Copying the cache to `/content` and pointing `data.rsna.cache_dir` there helps.

**Kaggle Notebooks (P100 16 GB or T4 ×2; the code uses one GPU)**
```python
# Kaggle cell - add the competition "rsna-pneumonia-detection-challenge" as notebook input
import os; os.environ["RFRR_RSNA_ROOT"] = "/kaggle/input/rsna-pneumonia-detection-challenge"
%cd /kaggle/working/right-for-the-right-reason      # upload the project as a dataset and copy it here
!pip install -q -r requirements.txt
!python scripts/prepare_rsna.py --cache --workers 2 --cache-dir /kaggle/working/cache_384_clahe
!python -m ml.training.train_model_b loader.num_workers=2 data.rsna.cache_dir=/kaggle/working/cache_384_clahe
```
* `/kaggle/input` is read-only. Splits, cache, checkpoints and outputs are written under `/kaggle/working`. Download
  `checkpoints/` and `outputs/` before the session ends.

Suggested first batch sizes at 384 px (adjust after benchmarking): hybrids and EfficientNet-B0: 16–32; DenseNet121 and
ResNet50: 16; ViT-Small: 8–16.

## 7. Comparing Model A and Model B properly

1. Train both with identical settings (only the config differs: λ = 0 vs 0.5).
2. Evaluate both: `python -m ml.evaluation.evaluate --model model_a --nih --stress --fit-ood --explain-n 64` (same for `model_b`).
3. Paired test on the same test images:
   `python scripts\compare_models.py --models model_b model_a efficientnet_b0` → `outputs/reports/model_comparison_rsna_test.json`
   (DeLong p-values with Holm correction).
4. Ablation table: `python scripts\run_ablation.py --summarize` → `outputs/reports/ablation_summary.json`.
5. Look beyond AUC: localization Dice, attention mass in box, Grad-CAM++ pointing game, deletion/insertion AUC, stress-test AUC
   drops, NIH AUC gap and calibration. The **Model Laboratory** page shows A vs B side by side.
6. **Repeat with at least 3 seeds.** Keep each seed's files separate:
   ```powershell
   $env:RFRR_CHECKPOINT_ROOT="checkpoints_seed1"; $env:RFRR_OUTPUT_ROOT="outputs_seed1"
   python -m ml.training.train_model_a experiment.seed=1 $CACHE
   python -m ml.training.train_model_b experiment.seed=1 $CACHE
   ```
   (The patient split is fixed by the first `prepare_rsna.py` run, so only initialisation and augmentation change.)
   Report mean ± standard deviation across seeds, then clear the variables (`Remove-Item Env:RFRR_CHECKPOINT_ROOT, Env:RFRR_OUTPUT_ROOT`).

## 8. Checking that everything is where it should be

```powershell
python scripts\model_status.py
```
For every model this shows: architecture implemented, training script present, checkpoint found or missing,
calibration found / missing / stale, OOD statistics, evaluation metrics, and **ready for inference**.
