# Frontend Guide (Stage 2)

React 18 + React Router 7 + Vite 6 + Tailwind CSS 3 + Recharts 2 + Lucide icons, in `frontend/`. Requires Node.js 20 or newer.

## Run

```bash
# terminal 1 - backend (project root)
uvicorn backend.app.main:app --reload --port 8000
# terminal 2 - frontend
cd frontend
npm install
npm run dev            # http://localhost:5173  (proxies /api -> http://127.0.0.1:8000)
npm run build          # production bundle in dist/
npm run preview        # serve dist/ on http://localhost:4173 (also proxies /api)
```

Configuration (`frontend/.env`, see `.env.example`): `VITE_API_BASE_URL` (default `/api`) and
`VITE_BACKEND_URL` (proxy target). When deploying `dist/` behind another web server, route `/api` to the
backend, or set `VITE_API_BASE_URL` to the backend URL and add the site to `RFRR_CORS_ORIGINS`.
Routing uses `HashRouter`, so static hosting needs no rewrite rules.

## Structure

```
src/
  services/api.js        the ONLY place that knows backend URLs (fetch + XHR upload with progress, ApiError)
  context/AppContext.jsx health polling (30 s), model list, selected model (localStorage, try/catch),
                         last file / analysis shared between Analysis and Explainability
  components/
    Layout.jsx           sidebar (drawer on mobile), top bar (server status, model picker), disclaimer banner
    ui.jsx               Card, Badge, Button, Tabs, Collapsible, EmptyState, ErrorState, Spinner, Notice, ...
    Dropzone.jsx         drag & drop / browse, client-side validation, preview + dimensions
    ImageViewer.jsx      stacked image + heatmap layers, zoom (buttons / ctrl+wheel), pan, compare slider,
                         per-layer opacity, colour legends, SVG reference boxes
    common.jsx           NoTrainedModelNotice, ProbabilityMeter, ZoneBars
  lib/format.js          number formatting, uncertainty levels, chart colours, map descriptions
  pages/                 the seven sections
```

## Pages

| Route | Page | Backend calls | Without trained models |
|---|---|---|---|
| `#/` | **Overview**: summary, model status, datasets, system, experiment performance, recent analyses, quick action | health, models, datasets/status, analyses, experiments/results | "Models not trained yet.", "No experiment results available" |
| `#/analysis` | **X-ray Analysis**: upload panel; output (label, calibrated + raw probability with threshold), uncertainty, quality/OOD, model, localization; tabs Original / Localization / Grad-CAM++ / Combined; zoom & compare; **Generate report** | analyze, research/report | upload works, Analyse disabled, step-by-step training instructions |
| `#/explainability` | **Explainability**: Original \| Predicted localization \| Grad-CAM++ side by side with reference boxes; IoU, Dice, pointing game, share inside box, attention alignment; "What influenced this model's prediction?"; limitations; researcher attention view | explain | same notice; nothing generated |
| `#/models` | **Model Laboratory**: cards for ResNet50, DenseNet121, EfficientNet-B0, ViT-Small, Model A, Model B (+ ablation), each with architecture, description, status, checkpoint, dataset, validation and test metrics; "Use for analysis"; Model A vs B table + chart + DeLong | models, models/load, experiments/results | every card "Not trained", buttons disabled, comparison explains why it is unavailable |
| `#/experiments` | **Research Experiments**: experiment runner with live log; panels A Classification (table + ROC), B Localization, C Explanation quality, D Shortcut stress tests, E Cross-dataset generalisation, F Calibration (table + reliability diagram), G Statistics (bootstrap / DeLong CIs, paired tests); training history; CSV/JSON downloads | experiments/results, experiments/run, experiments/jobs/{id}, experiments/files | each panel: "No experiment results available" |
| `#/datasets` | **Dataset Management**: RSNA and NIH checklists with actual counts, split table, overlap warning, placement instructions, re-scan | datasets/status | shows what is missing and how to add it |
| `#/about` | **About Research**: problem, gap, aim, objectives, questions, hypotheses, methodology, contributions, dataset limitations, ethics | — | — |

## Design rules applied

* White cards on a light grey canvas, navy text, blue (`#2f6fd6`) and teal (`#0e8f86`) accents, rounded cards,
  soft shadows, Inter font with system fallback, no dark backgrounds or decorative animation.
* Plain-language primary text. Researcher detail sits inside "Technical details" / "For researchers" collapsibles.
* The disclaimer is permanently visible under the top bar on every page. Results say "Model output only - not a diagnosis";
  the words "diagnosis" and "confirmed" are never used for a result.
* Heatmaps use perceptually uniform colour maps (viridis = localization, inferno = Grad-CAM++, cividis = attention).
  They have a smooth transparency ramp and no contours, so they don't look like disease boundaries.
  Legends state each scale.
* Localization predictions and Grad-CAM++ are always labelled and shown separately (Combined view layers both,
  each with its own legend and opacity).
* Every data view has loading, empty and error states. If the API is unreachable, the page says so with the
  start command. Errors show the backend's message and code.

## Verified (Stage 2 integration run)

A headless-Chromium run against the real backend (`tests/e2e/ui_smoke.cjs`, re-run in the Stage-3 audit) made 50 checks,
all passing:

* every page at 1440 px desktop and iPhone-13 width with no horizontal overflow and no console errors (two scenarios);
* no-checkpoint scenario: "Models not trained yet", Analyse disabled, client-side rejection of a `.txt` upload,
  image preview with dimensions, Model Lab buttons disabled, "No experiment results available";
* API-unreachable scenario: error state shown; server-side rejection of a text file renamed `.png`; loading state;
  model selection from the top bar; no calibration warning after evaluating the checkpoint;
* with checkpoints trained for a few seconds on **synthetic noise** (software test only, meaningless numbers):
  model selection, analysis with all tabs, combined two-layer overlay, zoom and compare, report opening (and printing to
  PDF), explanation of an RSNA-format DICOM with automatic reference boxes and metrics, experiment charts, recent
  analyses, mobile analysis and menu.

No mock data ships in the frontend. Every number on screen comes from the API.
