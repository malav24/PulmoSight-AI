# API Documentation (Stage 2)

FastAPI backend in `backend/app/`. Interactive OpenAPI docs: `http://localhost:8000/docs`.

```bash
# from the project root (ML requirements already installed)
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

The API starts and serves every endpoint **without trained models**. Inference endpoints return
`409 model_not_trained` until a valid checkpoint exists. The API never returns a fabricated prediction.

## Conventions

* Base path: `/api`. JSON responses. CORS origins come from `RFRR_CORS_ORIGINS` (default: the Vite dev and
  preview ports).
* Errors always use this shape:
  ```json
  {"error": {"code": "model_not_trained", "message": "human-readable text", "details": {...}}}
  ```
* Server paths are shown relative to the project (e.g. `checkpoints/hybrid/model_b_best.pth`). Absolute
  locations are never exposed.

| HTTP | `code` examples | Meaning |
|---|---|---|
| 400 | `invalid_analysis_id`, `invalid_file` | malformed identifier or path |
| 404 | `unknown_model`, `analysis_not_found`, `job_not_found`, `file_not_found` | resource does not exist |
| 409 | `model_not_trained`, `job_running` | precondition not met |
| 413 | `file_too_large`, `image_too_large` | upload > `RFRR_MAX_UPLOAD_MB` (25 MB) or > 40 M pixels |
| 415 | `unsupported_file_type`, `content_mismatch`, `corrupt_image`, `invalid_dicom` | not a valid PNG/JPEG/DICOM |
| 422 | `validation_error`, `empty_file`, `image_too_small`, `unreadable_image`, `invalid_boxes`, `confirmation_required`, `invalid_checkpoint` | invalid input |
| 500 | `internal_error` | unexpected; details only in the server log |

## Endpoints

### `GET /api/health`
Server status, ML availability (torch version, device, CUDA), trained / loaded / active models, disclaimer.

### `GET /api/models`
Every model in the study (`model_b`, `model_a`, the four baselines, `ablation_cls_only`), each with:
`name, family, role, architecture, description, training_objective, parameters_millions, maps,
trained, checkpoint{status: not_trained|trained|invalid, path, checkpoint_id, epoch, saved_at, size_mb,
loss_weights, validation_metrics, validation_threshold}, calibration, evaluation (summary of
outputs/metrics/<key>_metrics.json or null), loaded, active`.
`calibration` includes `matches_checkpoint` and `temperature_at_bound`; `evaluation.stale` is true when the metrics
file belongs to an earlier checkpoint. `trained` is true **only** when the checkpoint file exists, is readable and stores the matching model key.
When nothing is trained, `message` is `"Models not trained yet."`.

### `POST /api/models/load`
Body `{"model_key": "model_b", "reload": false}`. Loads the trained checkpoint (strict, via
`ml.utils.checkpoints.load_trained_model`) and makes it the default model for analysis.
Returns 409 if no checkpoint exists, 404 for an unknown key.

### `GET /api/datasets/status[?refresh=true]`
On-disk discovery (cached 60 s): RSNA (folder, annotation/class CSVs, DICOM count, split file and
per-split counts, cache, statistics, `preparation_status`) and NIH (metadata CSV, test list, bbox CSV, RSNA
overlap mapping, PNG count, `ready_for_external_evaluation`, `external_validity_warning`), plus placement
instructions.

### `POST /api/analyze` (multipart)
Fields: `file` (PNG/JPEG/DICOM, required) and `model_key` (optional; default = active → `RFRR_DEFAULT_MODEL`
→ first trained model). The upload is streamed to a private temp file, checked (extension, magic bytes,
structure, size, dimensions), processed, and **always deleted**. Response:

```jsonc
{
  "analysis_id": "32 hex", "created_at": "...",
  "model": {"key","name","architecture","family","checkpoint_id","checkpoint_epoch","localization_trained","attention_guided"},
  "input": {"kind","width","height","bytes","dicom_view","file_name"}, "input_reference": "sha256:<16 hex>",
  "prediction": {"probability","calibrated_probability|null","temperature|null","threshold","threshold_source",
                 "outcome": "suspected|not_detected|abstained","label","abstained","abstention_reasons":[]},
  "configuration_warnings": [],   // e.g. calibration fitted on another checkpoint -> ignored
  "uncertainty": {"mean","std","p05","p95","predictive_entropy","mutual_information","n_samples"},
  "quality": {"passed","errors":[],"warnings":[],"measurements":{},"disclaimer"},
  "ood": {"available", "mahalanobis_distance?", "threshold?", "flag?"},
  "localization": {"available", "peak_probability", "area_fraction_above_0_5", "peak_location", "top_zones"} ,
  "gradcam_top_zones": [{"zone","share"}],
  "image_size": 384,
  "images": {"original": "data:image/png;base64,...", "localization": "...RGBA layer", "attention": "...", "gradcam": "..."},
  "map_descriptions": {...}, "colormaps": {"localization":"viridis","gradcam":"inferno","attention":"cividis"},
  "disclaimer": "..."
}
```
Heatmap layers are transparent PNGs that the UI stacks over `original`. Localization uses its absolute
0–1 scale. Grad-CAM++ and attention are normalised per image.

### `POST /api/explain` (multipart)
Same fields as `/analyze`, plus optional `reference_boxes` = JSON `[[x, y, width, height], ...]` in original-image
pixels. Without boxes, if the file name is `<RSNA patientId>.dcm` and the RSNA label CSV is present, the
radiologist boxes are looked up automatically. Adds the following to the analysis response:
`reference{available, source: provided|rsna_annotations|null, boxes (in image_size pixels)}`,
`metrics{localization{iou,dice,pointing_game_hit,mass_in_box}, gradcam{…, note}, attention_alignment{mass_in_box,
pointing_game, kl_to_box}}`, `influence{gradcam_zones, localization_zones, attention_zones}`, `orientation_note`,
`limitations[]`. Metrics use `ml.evaluation.localization` and `ml.evaluation.explanation`.

### `GET /api/analyses?limit=10` · `GET /api/analyses/{id}` · `DELETE /api/analyses/{id}`
Stored analyses (`outputs/analyses/<id>/`, newest 200 kept). Only the 384-px preprocessed image, the layers and
the model outputs are stored. The original file and its name are **not** stored. Disable storage with
`RFRR_STORE_ANALYSES=false`.

### `GET /api/experiments/results`
Aggregates the real output files: per model `metrics` (`outputs/metrics/<key>_metrics.json`), `training_log`
(CSV), `curves` (ROC points and reliability bins computed from the per-image prediction CSVs),
`ablation_summary`, `comparisons` (DeLong reports), downloadable `files`, recent `jobs`, and the available
experiment `kinds`. `available: false` plus `"No experiment results available."` when nothing has been run.

### `POST /api/experiments/run` → 202
Body `{"kind": "...", "model_key": "model_b", "params": {...}}`. Starts **one** background OS process
(409 if one is already running). Inference is not blocked.

| kind | params | runs |
|---|---|---|
| `evaluate` | `nih`, `stress`, `fit_ood` (bool), `explain_n` | `python -m ml.evaluation.evaluate` |
| `compare_models` | `models` (≥2), `dataset` (`rsna_test`/`nih`) | `scripts/compare_models.py` |
| `ablation_summary` | — | `scripts/run_ablation.py --summarize` |
| `prepare_rsna` | `cache` (bool) | `scripts/prepare_rsna.py` |
| `prepare_nih` | — | `scripts/prepare_nih.py` |
| `train` | `confirm: true` (required), `epochs`, `resume` | `python -m ml.training.train_<model>` |

When a job finishes, cached models are dropped so new calibration and OOD files take effect.

### `GET /api/experiments/jobs` · `GET /api/experiments/jobs/{job_id}`
Status (`running|succeeded|failed`), command, return code, last log lines.

### `GET /api/experiments/files/{folder}/{name}`
Downloads a `.csv`/`.json` from `outputs/{predictions,reports,logs,metrics}` only (whitelisted).

### `GET /api/research/report[?analysis_id=…][&format=html|json]`
With `analysis_id`: a printable, self-contained HTML report (analysis ID, timestamp, model architecture, checkpoint
identifier, input hash, embedded preprocessed image and localization / Grad-CAM++ overlays, probabilities,
uncertainty, quality and OOD checks, available evaluation figures, limitations, disclaimer). Use the browser's
**Print → Save as PDF**. It contains no patient details and no diagnostic recommendation.
Without `analysis_id`: a research status report (models, training status, metrics read from files).

## Integrated Stage-1 modules

| Backend service | Uses |
|---|---|
| `services/model_manager.py` | `ml.utils.checkpoints` (registry, `read_checkpoint`, `load_calibration`), `ml.inference.predictor.PneumoniaPredictor` |
| `services/analysis.py` | `PneumoniaPredictor.predict/load_input/forward`, `ml.datasets.preprocessing` (box transforms, masks), `ml.evaluation.localization`, `ml.evaluation.explanation.attention_alignment`, `ml.explainability.visualization.MAP_DESCRIPTIONS` |
| `services/experiments.py` | `ml.evaluation.calibration.reliability_curve`; runs `ml.evaluation.evaluate`, `ml.training.train_*`, `scripts/*` |
| `services/datasets.py` | `ml.utils.config` (paths, env overrides) |
| `services/uploads.py`, `imaging.py`, `reports.py` | API-specific (validation, display layers, HTML reports) |

## Concurrency and safety

* Inference runs in FastAPI's thread pool, with one lock per model (Grad-CAM++ hooks are not re-entrant).
  Experiments run as separate processes.
* Upload checks: extension whitelist, PNG/JPEG/DICOM magic bytes, PIL `verify()` / DICOM header parse, 25 MB
  size, 40 M-pixel and 64-px minimum dimensions. The temp file is removed in a `finally` block.
* No automatic training. `train` requires `confirm: true`.

## Tests

`python -m pytest backend/tests` covers 15 cases: health, model listing without checkpoints, missing-checkpoint
409s, five invalid-upload types, size and pixel limits, dataset status, empty experiments, reports,
a full flow with a briefly trained **synthetic** checkpoint (analyze, stored analysis, report, explain with
RSNA and provided boxes), and a background evaluation job followed by result loading and downloads.
