# Dataset Setup

Neither dataset is included, and neither can be committed to Git: both have their own licences, and images are
excluded by `.gitignore`. You download them yourself.

## 1. RSNA Pneumonia Detection Challenge (training + internal test)

**Source:** https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (you must accept the
competition rules on Kaggle; a download of a few GB).

Automatic (recommended):
```bash
kaggle auth login                                         # one time (kaggle >= 2); or put a token in ~/.kaggle/access_token
python scripts/prepare_rsna.py --download --cache         # download + unzip + splits + cache
```
`prepare_rsna.py` (implementation: `ml/datasets/locate.py`) first looks for an existing copy in `data/rsna` (including
sub-folders and an unextracted Kaggle zip), then in Downloads, Desktop and Documents (up to 3 levels deep). A copy found
outside the project is used in place, and its path is written to `.env` as `RFRR_RSNA_ROOT`. A 403 error from Kaggle almost
always means the competition rules have not been accepted on the website yet.

Expected layout (row/image counts below are the published dataset figures; `scripts/prepare_rsna.py` reports the
counts actually found on your disk):

```
data/rsna/
  stage_2_train_labels.csv            30,227 rows (one per box; Target=0 rows have empty box)
  stage_2_detailed_class_info.csv     class per patientId
  stage_2_train_images/<patientId>.dcm   26,684 DICOMs, 1024×1024, 8-bit MONOCHROME2
  stage_2_test_images/                (unlabelled - NOT used)
```

**Label semantics.** `Target = 1` ⇔ class *"Lung Opacity"*: radiologists judged the opacity suspicious
for pneumonia and drew boxes. *"No Lung Opacity / Not Normal"* and *"Normal"* are both `Target = 0`.
These are **pneumonia-related lung-opacity annotations, not confirmed clinical diagnoses** (no clinical
history, labs or follow-up). Treat model outputs in the same way.

**Splits.** The stage-2 test images have no public labels, so the labelled set is split
**70 / 15 / 15 (train / val / test) at patient level**, stratified on the 3-class label, with seed 42:

```bash
python scripts/prepare_rsna.py                    # writes data/rsna/splits/rsna_patient_splits.csv + statistics
python scripts/prepare_rsna.py --cache --workers 8   # recommended: 384-px PNG cache (avoids decoding 1024-px DICOMs every epoch)
# then train with  data.rsna.cache_dir=data/rsna/cache_384_clahe
```

RSNA has one image per `patientId`, so patient-level and image-level splitting coincide. Note that RSNA
assigned new anonymised IDs: if the same real patient appears under two IDs this cannot be detected
from RSNA alone (limitation).

Statistics (counts, prevalence, boxes per image, relative box area, per-split prevalence) are written to
`outputs/reports/rsna_dataset_statistics.json`.

**Preprocessing** (`data.preprocessing`, identical for training, evaluation and inference):
DICOM rescale slope/intercept → MONOCHROME1 inversion if needed → 0.5–99.5 percentile window → uint8 →
resize to 384 (stretch, or letterbox to keep aspect) → CLAHE (clip 2.0, 8×8 tiles) → 3-channel,
ImageNet normalisation. Boxes are mapped with the same geometry and rasterised into masks. Training
augmentation (small affine + brightness/contrast; horizontal flip **off** by default because chest
anatomy is not left–right symmetric) transforms the image and mask together, keeping them pixel-aligned.
A cache built with different preprocessing settings is rejected.

## 2. NIH ChestX-ray14 (external evaluation only)

**Source:** https://nihcc.app.box.com/v/ChestXray-NIHCC (more than 40 GB; the published dataset description lists 112,120
PNGs from 30,805 patients). NIH provides `batch_download_zips.py` on the same page. On Windows, extract with
`Get-ChildItem data\nih\images_*.tar.gz | ForEach-Object { tar -xzf $_.FullName -C data\nih }`.

```
data/nih/
  Data_Entry_2017_v2020.csv   (Data_Entry_2017.csv also accepted)
  test_list.txt, train_val_list.txt
  BBox_List_2017.csv          (optional)
  images_001/images/*.png ... images_012/images/*.png    (any nesting works; files are indexed by name)
  pneumonia-challenge-dataset-mappings_2018.json         (STRONGLY recommended - see below)
```

```bash
python scripts/prepare_nih.py      # selection summary -> outputs/reports/nih_dataset_statistics.json
```

### Small option: the official NIH random sample (default when disk space is limited)

```bash
python scripts/prepare_nih.py --download-sample      # or: python scripts/download_datasets.py
```
Downloads the Kaggle dataset `nih-chest-xrays/sample`, a ~5 % random sample of ChestX-ray14 (about 5,600 images) with
`sample_labels.csv` in the same column format as `Data_Entry_2017`. The code detects it automatically
(`ml/datasets/nih.py::find_entries_csv`). Differences from the full set:
* no `test_list.txt`: all listed images are used (the model never trains on NIH, so all are unseen); this is recorded
  in the run's `selection_notes`;
* no `BBox_List_2017.csv`: NIH localization metrics are not available;
* few Pneumonia-labelled images (prevalence around 1 %), and fewer still after excluding the RSNA overlap, so
  external AUC confidence intervals will be wide. Report this.

### ⚠ The RSNA images are a subset of NIH ChestX-ray14

The RSNA challenge images were drawn from this same NIH database. Evaluating on NIH without removing
them would test the model on images it was trained on. RSNA publishes the mapping on its
challenge page ("Download mapping of RSNA image dataset to original NIH dataset"):

https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json

Save it as `data/nih/pneumonia-challenge-dataset-mappings_2018.json`. With
`exclude_rsna_overlap: true` (the default) every NIH image listed in it is removed. The parser finds
NIH filenames by their format (`00000013_005`), so it does not depend on the JSON's key names. If the file is
missing, a warning is raised and the run's `selection_notes` state that **the NIH evaluation is
contaminated and must not be reported as external.**

### Selection (`data.nih`)

* `subset: official_test` — NIH's patient-level `test_list.txt` (25,596 images). `all` is also allowed, since
  the model never trains on NIH.
* `view_positions: [PA, AP]` — frontal views, as in RSNA.
* `negative_definition`: `all_non_pneumonia` (default: any image without the Pneumonia label, including
  other diseases) or `no_finding` (only "No Finding" images, an easier and more optimistic task).
* `one_image_per_patient` (optional) and patient-clustered bootstrap CIs handle follow-up images.

### Label differences and label noise (document these in the paper)

| | RSNA | NIH ChestX-ray14 |
|---|---|---|
| Positive class | "Lung Opacity" suspicious for pneumonia, **radiologist-annotated** | "Pneumonia" **text-mined (NLP) from reports** |
| Prevalence | ≈ 22–23 % | ≈ 1–2 % in the full set |
| Boxes | ~9.5 k radiologist boxes | ~120 Pneumonia boxes in BBox_List_2017 (only these are used; **none are generated**) |
| Known issues | inter-reader variability; boxes are coarse | NLP label errors; "Pneumonia" overlaps with Infiltration/Consolidation; visual audits found many mislabelled images |

Consequently:
1. A drop in AUC from RSNA to NIH mixes label-definition shift, prevalence and case-mix shift, and image
   shift. It is not purely "domain generalisation".
2. Both datasets come from the NIH Clinical Center. This is **not** a multi-institution external validation.
3. Precision/PPV is not comparable across the two datasets because prevalence differs.
4. NIH is never used for training, threshold selection or calibration. The temperature and threshold are
   fitted on RSNA validation data and applied to NIH unchanged.

## 3. Licences and ethics

Follow each dataset's terms (Kaggle competition rules; NIH CC0 with requested citation of Wang et al.,
CVPR 2017). Both datasets are de-identified. The DICOM loader keeps only technical tags
(size, modality, view position, photometric interpretation); it reads no patient identifiers.
