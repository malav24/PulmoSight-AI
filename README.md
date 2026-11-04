# RIGHT FOR THE RIGHT REASON

**Explanation-Guided Deep Learning for Pneumonia Detection and Localization in Chest X-Rays**
An academic research prototype: a machine-learning pipeline, a FastAPI backend and a React web application.

> **Status (read this first)**
> * All code is implemented and tested (91 automated tests on Python 3.10 and 3.11, plus 50 browser checks).
> * **No model has been trained.** The `checkpoints/` folders are empty. No accuracy figures exist yet.
>   They appear only after you train and evaluate the models yourself.
> * This is **not a medical device** and has **no clinical validation**.

---

## Contents
1. [Project introduction](#1-project-introduction)
2. [Research objectives](#2-research-objectives)
3. [Research methodology](#3-research-methodology)
4. [System architecture](#4-system-architecture)
5. [Technology stack](#5-technology-stack)
6. [Folder structure](#6-folder-structure)
7. [Dataset sources](#7-dataset-sources)
8. [Dataset download instructions](#8-dataset-download-instructions)
9. [Dataset folder placement](#9-dataset-folder-placement)
10. [Python environment setup](#10-python-environment-setup)
11. [Dependency installation](#11-dependency-installation)
12. [Frontend installation](#12-frontend-installation)
13. [Dataset preprocessing](#13-dataset-preprocessing)
14. [Model training](#14-model-training)
15. [Model evaluation](#15-model-evaluation)
16. [External validation (NIH)](#16-external-validation-nih)
17. [Explainability](#17-explainability)
18. [Application startup](#18-application-startup)
19. [X-ray analysis instructions](#19-x-ray-analysis-instructions)
20. [Troubleshooting](#20-troubleshooting)
21. [Medical disclaimer](#21-medical-disclaimer)
22. [Known limitations](#22-known-limitations)
- [Appendix A - Complete Windows PowerShell command sequence](#appendix-a--complete-windows-powershell-command-sequence)
- [Appendix B - Documentation index](#appendix-b--documentation-index)

---

## 1. Project introduction

AI models can classify chest X-rays well and still be **right for the wrong reason**: they may react to
hospital text markers, image borders or scanner artefacts instead of the lungs. This project asks a simple
question: *if we show a model, during training, where radiologists saw the abnormality, does it learn to
look there, and does that make it more trustworthy?*

The project contains:
* a **hybrid CNN–Transformer** (EfficientNet-B0 + 2-layer Transformer) with two outputs: a
  pneumonia-related opacity score and a localization heatmap;
* **Model A** (trained to classify and localize) and **Model B** (the same network, additionally guided to put its
  attention inside radiologist boxes), plus four standard baselines;
* a complete evaluation framework (accuracy, localization, explanation faithfulness, shortcut stress tests,
  external NIH data, calibration, statistics);
* a web application for analysing an X-ray and inspecting the explanations.

## 2. Research objectives

1. Build an explainable hybrid model that classifies and localizes pneumonia-related lung opacity.
2. Train it with and without attention guidance from RSNA radiologist boxes (Model B vs Model A).
3. Compare with ResNet50, DenseNet121, EfficientNet-B0 and ViT-Small.
4. Evaluate classification, localization, explanation faithfulness, robustness to shortcuts, calibration and
   generalisation to NIH ChestX-ray14.
5. Present results through an interface clinicians can understand, with clear limitations.

Research questions and hypotheses: [docs/RESEARCH_METHODOLOGY.md](docs/RESEARCH_METHODOLOGY.md).

## 3. Research methodology

* **Data.** RSNA Pneumonia Detection Challenge (training, validation, internal test): split 70/15/15
  **by patient**, stratified on the three RSNA classes. NIH ChestX-ray14 is used **only** for external
  testing, after removing the NIH images that were reused in RSNA.
* **Controlled comparison.** Model A and Model B share architecture, data, seed, schedule and model-selection
  rule. Only the attention-guidance weight λ differs (0 vs 0.5). A third arm (classification only) completes the ablation.
* **Objective.** `L = L_classification + α·L_localization + λ·L_attention`, where `L_attention` is the share of
  the classifier's spatial attention that falls *outside* the radiologist boxes (computed on positive images only).
* **Evaluation.** The threshold and calibration are chosen on validation data only. The metrics have 95 % bootstrap
  confidence intervals, model comparisons use paired DeLong tests with Holm correction, explanations are tested for
  faithfulness (deletion/insertion) and plausibility (pointing game), and the stress tests mask non-lung content.

## 4. System architecture

```
 ┌──────────────── Browser (React, port 5173) ────────────────┐
 │ Overview · X-ray Analysis · Explainability · Model Lab ·   │
 │ Research Experiments · Dataset Management · About          │
 └───────────────▲──────────────────────────┬─────────────────┘
                 │ JSON / PNG layers        │ /api/* (REST)
 ┌───────────────┴──────────────────────────▼─────────────────┐
 │ FastAPI backend (port 8000)  backend/app                   │
 │  uploads (validation) · model manager · analysis/explain   │
 │  reports · dataset status · experiment jobs (subprocess)   │
 └───────────────▲──────────────────────────┬─────────────────┘
                 │ Python imports           │ runs scripts
 ┌───────────────┴──────────────────────────▼─────────────────┐
 │ ml/ (PyTorch)  models · datasets · training · evaluation · │
 │ explainability · inference · configs · utils               │
 └───────▲───────────────────────▲──────────────────▲─────────┘
     data/rsna, data/nih     checkpoints/          outputs/
```
Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (system) and [docs/ML_ARCHITECTURE.md](docs/ML_ARCHITECTURE.md) (models).

## 5. Technology stack

| Layer | Technology |
|---|---|
| Deep learning | Python 3.10/3.11, PyTorch ≥ 2.3, timm ≥ 1.0 (EfficientNet-B0, ResNet50, DenseNet121, ViT-Small) |
| Medical imaging | pydicom, OpenCV (CLAHE), albumentations 2.x |
| Analysis | NumPy, pandas, scikit-learn, SciPy, matplotlib |
| API | FastAPI, Pydantic 2, Uvicorn |
| Web app | React 18, React Router 7, Vite 6, Tailwind CSS 3, Recharts 2, Lucide icons (Node.js ≥ 20) |
| Tests | pytest, FastAPI TestClient, Playwright (optional UI smoke test) |

## 6. Folder structure

```
right-for-the-right-reason/
├── ml/
│   ├── models/          hybrid.py (proposed), baselines.py, factory.py
│   ├── datasets/        preprocessing.py, transforms.py, rsna.py, nih.py, loaders.py
│   ├── training/        trainer.py, losses.py, train_resnet50.py, train_densenet121.py,
│   │                    train_efficientnet.py, train_vit.py, train_model_a.py, train_model_b.py,
│   │                    train_ablation_cls_only.py
│   ├── evaluation/      evaluate.py (CLI), classification, localization, explanation, shortcut,
│   │                    calibration, statistics, generalization
│   ├── explainability/  gradcam_pp.py, visualization.py
│   ├── inference/       predictor.py, quality.py, uncertainty.py
│   ├── configs/         base.yaml + one YAML per model
│   └── utils/           config, runtime, checkpoints, logging, synthetic (tests only)
├── backend/app/         main.py, routers/api.py, services/*  (+ backend/tests)
├── frontend/            React app (src/pages, src/components, src/services/api.js)
├── scripts/             prepare_rsna.py, prepare_nih.py, verify_setup.py, model_status.py,
│                        run_ablation.py, compare_models.py, explain_image.py, benchmark.py
├── data/rsna/  data/nih/                        <- you put the datasets here (not in Git)
├── checkpoints/hybrid/  baselines/  calibration/ <- trained weights appear here (not in Git)
├── outputs/metrics/ predictions/ visualizations/ reports/ logs/ analyses/
├── tests/               ML tests (+ tests/e2e/ui_smoke.cjs browser test)
├── docs/                all documentation
├── requirements.txt     .env.example     pytest.ini
```

## 7. Dataset sources

| Dataset | Use | Source |
|---|---|---|
| RSNA Pneumonia Detection Challenge (2018) | training, validation, internal test | https://www.kaggle.com/c/rsna-pneumonia-detection-challenge |
| NIH ChestX-ray14 | external evaluation only | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| RSNA → NIH image mapping | removes RSNA images from the NIH test set | "Download mapping of RSNA image dataset to original NIH dataset" on https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018 |

## 8. Dataset download instructions

**Easiest: one command for both datasets** (after the one-time Kaggle setup below):
```powershell
python scripts\download_datasets.py
```
This downloads the **full RSNA** training set, plus the official **~5 % random sample of NIH ChestX-ray14** (about 5,600
images) instead of the full NIH set (over 40 GB). It checks free disk space first, extracts everything into
`data\rsna` / `data\nih`, deletes the zips, fetches the RSNA→NIH mapping file and runs both prepare scripts. Existing
copies (in the project, Downloads, Desktop or Documents) are reused instead of downloaded again.
Plan for roughly 9 GB free while RSNA downloads (the zip is deleted afterwards) and about 5 GB for the NIH sample.

**RSNA**: the prepare script finds, unzips or downloads it for you.
```powershell
# One-time Kaggle setup (the `kaggle` package is installed by requirements.txt):
#   1. log in at kaggle.com, open https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules
#      and click "I Understand and Accept"
#   2. sign the command line in to Kaggle (opens a browser):
kaggle auth login
# Download + unzip + split + cache in one command:
python scripts\prepare_rsna.py --download --cache --workers 4
```
Already downloaded it? Skip `--download`. `prepare_rsna.py` looks in `data\rsna` (including sub-folders and an
unextracted Kaggle zip), then in your **Downloads, Desktop and Documents** folders (for example an `archive (7)`
folder). A copy found elsewhere is used where it is, not copied, and its location is saved as `RFRR_RSNA_ROOT` in `.env`,
so training, evaluation and the web app all use it.
**NIH ChestX-ray14, small sample** (default): `python scripts\prepare_nih.py --download-sample`, or `download_datasets.py` above.
**NIH ChestX-ray14, full** (more than 40 GB, only if you have the space): download from the NIH Box link `Data_Entry_2017_v2020.csv`, `test_list.txt`,
`train_val_list.txt`, `BBox_List_2017.csv` and `images_001.tar.gz` … `images_012.tar.gz` into `data\nih` (or anywhere in
Downloads/Desktop/Documents), then
```powershell
python scripts\prepare_nih.py     # locates the folder, extracts the .tar.gz files, downloads the RSNA->NIH mapping file
```
More detail (label meanings, overlap, licences): [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md).

## 9. Dataset folder placement

```
data/rsna/stage_2_train_labels.csv
data/rsna/stage_2_detailed_class_info.csv
data/rsna/stage_2_train_images/<patientId>.dcm
data/nih/Data_Entry_2017_v2020.csv        data/nih/test_list.txt
data/nih/BBox_List_2017.csv               data/nih/pneumonia-challenge-dataset-mappings_2018.json
data/nih/images/<00000001_000>.png        (any sub-folder depth is fine)
```
Datasets elsewhere? Set `RFRR_RSNA_ROOT` / `RFRR_NIH_ROOT` in `.env` (copy `.env.example`).
Check with `python scripts\verify_setup.py`, or open **Dataset Management** in the web app.

## 10. Python environment setup

Install **Python 3.11** (or 3.10) from python.org (tick "Add python.exe to PATH") and **Node.js 20 LTS** from nodejs.org.
```powershell
cd "$HOME\Downloads\AI in Healthcare\right-for-the-right-reason"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt now starts with (.venv)
```
If PowerShell blocks the activation script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then run the activation again.

## 11. Dependency installation

```powershell
python -m pip install --upgrade pip
# PyTorch FIRST - choose ONE line:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # no GPU
pip install -r requirements.txt
python scripts\verify_setup.py        # shows Python, GPU, libraries, datasets, checkpoints
python -m pytest                      # 91 tests on synthetic data, several minutes on CPU
```

## 12. Frontend installation

```powershell
cd frontend
npm install
cd ..
```

## 13. Dataset preprocessing

```powershell
python scripts\prepare_rsna.py --cache --workers 4    # finds/unzips RSNA, patient-level splits + statistics + 384-px cache
                                                      # add --download to fetch it from Kaggle (see section 8)
python scripts\prepare_nih.py                         # NIH external set + statistics (needs NIH data)
```
* Splits: `data/rsna/splits/rsna_patient_splits.csv` (seed 42; each patientId is in exactly one split).
* Cache: `data/rsna/cache_384_clahe/` (pass `data.rsna.cache_dir=data/rsna/cache_384_clahe` when training).
* Statistics: `outputs/reports/rsna_dataset_statistics.json`, `nih_dataset_statistics.json`.

## 14. Model training

Beginner walkthrough for each model: **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**. Commands:

| Model | Command | Checkpoint |
|---|---|---|
| ResNet50 | `python -m ml.training.train_resnet50` | `checkpoints/baselines/resnet50_best.pth` |
| DenseNet121 | `python -m ml.training.train_densenet121` | `checkpoints/baselines/densenet121_best.pth` |
| EfficientNet-B0 | `python -m ml.training.train_efficientnet` | `checkpoints/baselines/efficientnet_b0_best.pth` |
| ViT-Small | `python -m ml.training.train_vit` | `checkpoints/baselines/vit_small_best.pth` |
| Hybrid Model A | `python -m ml.training.train_model_a` | `checkpoints/hybrid/model_a_best.pth` |
| Hybrid Model B | `python -m ml.training.train_model_b` | `checkpoints/hybrid/model_b_best.pth` |
| Ablation (cls only) | `python -m ml.training.train_ablation_cls_only` | `checkpoints/hybrid/ablation_cls_only_best.pth` |

Common options: `--epochs 30 --batch-size 16 --resume`, and dotted overrides such as `data.rsna.cache_dir=data/rsna/cache_384_clahe`
or `loader.num_workers=2`. Measure your GPU first with `python scripts\benchmark.py --models model_b vit_small`.

## 15. Model evaluation

```powershell
python -m ml.evaluation.evaluate --model model_b --stress --fit-ood --explain-n 64
python scripts\compare_models.py --models model_b model_a efficientnet_b0 densenet121 resnet50 vit_small
python scripts\run_ablation.py --summarize
python scripts\model_status.py
```
Outputs: `outputs/metrics/<model>_metrics.json`, per-image CSVs in `outputs/predictions/`, ROC/reliability
plots and panels in `outputs/visualizations/`, temperature + threshold in `checkpoints/calibration/temperature_scaling.json`.
The temperature and threshold are fitted on the **RSNA validation** split and are tied to the evaluated checkpoint. If you
retrain, the old calibration is ignored and you are told to evaluate again.

## 16. External validation (NIH)

```powershell
python -m ml.evaluation.evaluate --model model_b --nih --stress --fit-ood
python scripts\compare_models.py --dataset nih --models model_b model_a
```
NIH is never used for training, threshold selection or calibration. Without the RSNA→NIH mapping file the run
is marked as **contaminated** (images seen in training would be included).

## 17. Explainability

Three different maps are produced and always kept separate:
* **Localization heatmap** (Model A and B): a learned prediction of where the radiologists' boxes would be.
* **Spatial attention** (hybrids): the weights the classifier actually uses to pool image regions.
* **Grad-CAM++** (all models): a post-hoc gradient explanation of the score.

```powershell
python scripts\explain_image.py --model model_b data\rsna\stage_2_train_images\<patientId>.dcm
```
This saves a panel in `outputs/visualizations/explain_model_b_<patientId>.png` with IoU, Dice, pointing game and
share-inside-box when RSNA boxes exist. Use **test-split** images (see TRAINING_GUIDE) for honest examples.
`evaluate.py --explain-n N` additionally measures deletion/insertion faithfulness on N test positives.

## 18. Application startup

Two terminals (both in the project folder, both with `.venv` activated for the first):
```powershell
# Terminal 1 - API
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --port 8000
# Terminal 2 - web app
cd frontend
npm run dev
```
Open **http://localhost:5173** (API docs: http://localhost:8000/docs). The app works without trained models. It
shows "Models not trained yet." and disables analysis.

## 19. X-ray analysis instructions

1. **Model Laboratory** → choose a trained model → **Use for analysis**.
2. **X-ray Analysis** → drag in a frontal (PA/AP) chest X-ray (PNG, JPEG or DICOM, at most 25 MB) → **Analyse X-ray**.
3. Read the result card: outcome wording, calibrated probability (when available), raw score with threshold,
   uncertainty, input-quality and distribution checks. The system **abstains** when it is too uncertain, when the
   image looks unusual, or when it fails the quality checks.
4. Use the tabs **Original / Localization heatmap / Grad-CAM++ / Combined overlay**, zoom, and "Compare with original".
5. **Generate report** opens a printable report (Print → Save as PDF).
6. **Explainability** compares the maps side by side and, if a radiologist box is given (or found for an RSNA file),
   reports IoU, Dice, pointing game and attention alignment.

## 20. Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `verify_setup` shows `cuda_available: false` on an NVIDIA PC | Reinstall PyTorch with the CUDA index URL (section 11); update the NVIDIA driver |
| `CUDA out of memory` | Lower `--batch-size` (e.g. 8), keep AMP on (default), close other GPU programs |
| DataLoader / "spawn" errors on Windows | add `loader.num_workers=0` to the training command |
| Training stops: cannot download ImageNet weights | needs internet (Hugging Face). Or train from scratch with `model.pretrained_backbone=false` and report that |
| `Checkpoint not found` | train the model; the file must be at the path in section 14 |
| "Calibration file was fitted on a different checkpoint" | re-run `python -m ml.evaluation.evaluate --model <key>` |
| Web page says "Cannot reach the analysis server" | Terminal 1 must be running `uvicorn ... --port 8000` |
| `npm` not found | install Node.js 20 LTS, open a new terminal |
| Port already in use | `uvicorn ... --port 8001` and start the UI with `$env:VITE_BACKEND_URL="http://127.0.0.1:8001"; npm run dev` |
| "Cache was built with … but the config requests …" | rebuild the cache with the same image size, or drop `data.rsna.cache_dir` |
| NIH warning "mapping file not found" | download the mapping JSON (section 8) |

## 21. Medical disclaimer

**For research and educational purposes only. This system is not a medical device and must not be used as a
substitute for professional radiological interpretation or clinical diagnosis.** Its outputs estimate the presence
of *pneumonia-related lung opacity as annotated in the RSNA dataset*. They are not diagnoses and have not been
clinically validated.

## 22. Known limitations

* **Not trained yet**: no performance claims can be made until you train and evaluate.
* **NIH sample**: with the default ~5 % NIH sample, only around 1 % of images carry the Pneumonia label, so
  external results rest on few positive cases and have wide confidence intervals; the sample has no official
  test list or bounding boxes. Use the full NIH dataset for final external-validation claims if you can.
* **Labels**: RSNA boxes are coarse rectangles around opacity, not confirmed pneumonia. NIH labels are text-mined
  and noisy, with a different definition.
* **Generalisation**: both datasets come from the NIH Clinical Center. NIH is not a multi-site external validation.
  Only frontal views. Subgroup fairness has not been assessed.
* **Explanations**: every map can be unfaithful. Deletion/insertion uses out-of-distribution perturbations. The
  Transformer mixes tokens, so guided attention does not guarantee "right reason".
* **Stress tests**: the "outside-lung" region is a geometric thorax ellipse (no lung segmentation model). Marker
  detection is heuristic.
* **Uncertainty / OOD / quality checks** are approximate (MC dropout, Mahalanobis distance, simple image heuristics).
* **Compute**: training at 384 px needs a GPU. Measured CPU-only speed is a few images per second.

---

## Appendix A - Complete Windows PowerShell command sequence

Run in order. Lines starting with `#` are comments. Steps J–O take hours on a GPU, so run them when you are ready.

```powershell
# A. Open the project
cd "$HOME\Downloads\AI in Healthcare\right-for-the-right-reason"

# B. Create the virtual environment            C. Activate it
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# D. Python dependencies (pick the CUDA or CPU PyTorch line)
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\verify_setup.py
python -m pytest

# E. Frontend dependencies
cd frontend; npm install; cd ..

# F. Place datasets (see section 8), then check
python scripts\verify_setup.py

# G + H + I. Download RSNA (full) + NIH (small sample), extract, create patient-level splits, statistics, image cache
kaggle auth login                                   # one time; accept the RSNA competition rules on kaggle.com first
python scripts\download_datasets.py
$CACHE = "data.rsna.cache_dir=data/rsna/cache_384_clahe"

# (download_datasets.py already ran prepare_rsna.py and prepare_nih.py - rerun them only if you change data later)

# (optional) measure speed / memory on your GPU to choose the batch size
python scripts\benchmark.py --models model_b resnet50 vit_small --batch-sizes 8 16 32

# J. Baselines
python -m ml.training.train_resnet50 $CACHE
python -m ml.training.train_densenet121 $CACHE
python -m ml.training.train_efficientnet $CACHE
python -m ml.training.train_vit $CACHE

# K. Hybrid Model A          L. Hybrid Model B          (+ ablation arm)
python -m ml.training.train_model_a $CACHE
python -m ml.training.train_model_b $CACHE
python -m ml.training.train_ablation_cls_only $CACHE

# M. Evaluate every model on the RSNA test split (+ stress tests, OOD statistics, Grad-CAM++ on 64 positives)
foreach ($m in "resnet50","densenet121","efficientnet_b0","vit_small","model_a","model_b","ablation_cls_only") {
  python -m ml.evaluation.evaluate --model $m --stress --fit-ood --explain-n 64
}
python scripts\compare_models.py --models model_b model_a efficientnet_b0 densenet121 resnet50 vit_small
python scripts\run_ablation.py --summarize

# N. External NIH evaluation (re-runs RSNA test too and adds NIH + generalisation gap)
foreach ($m in "resnet50","densenet121","efficientnet_b0","vit_small","model_a","model_b") {
  python -m ml.evaluation.evaluate --model $m --nih --stress --fit-ood --explain-n 64
}
python scripts\compare_models.py --dataset nih --models model_b model_a efficientnet_b0 densenet121 resnet50 vit_small

# O. Grad-CAM++ / localization panels for chosen TEST images
$ids = Import-Csv data\rsna\splits\rsna_patient_splits.csv | Where-Object { $_.split -eq "test" -and $_.target -eq "1" } | Select-Object -First 3 -ExpandProperty patientId
foreach ($id in $ids) { python scripts\explain_image.py --model model_b "data\rsna\stage_2_train_images\$id.dcm" }
python scripts\model_status.py

# P. Terminal 1 - start FastAPI
uvicorn backend.app.main:app --port 8000

# Q. Terminal 2 (new window) - start React
cd "$HOME\Downloads\AI in Healthcare\right-for-the-right-reason\frontend"
npm run dev

# R. Open the web application
Start-Process http://localhost:5173
```

## Appendix B - Documentation index

| Document | Content |
|---|---|
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | beginner training guide for every model, Colab/Kaggle settings |
| [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md) | datasets, labels, overlap, preprocessing |
| [docs/RESEARCH_METHODOLOGY.md](docs/RESEARCH_METHODOLOGY.md) | questions, hypotheses, protocol, statistics, threats to validity |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | system architecture and data flow |
| [docs/ML_ARCHITECTURE.md](docs/ML_ARCHITECTURE.md) | network, losses, maps, checkpoints |
| [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md) | every metric and its limitations |
| [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | REST endpoints |
| [docs/FRONTEND_GUIDE.md](docs/FRONTEND_GUIDE.md) | web app pages and structure |
| [docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md) | Stage-3 audit: issues found and fixed, test results, measurements |
