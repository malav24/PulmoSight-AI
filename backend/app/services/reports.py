"""Printable research reports (self-contained HTML; use the browser's Print → Save as PDF).

Contains no patient details and no diagnostic recommendations.
"""
from __future__ import annotations

import datetime as dt
import html
from typing import Any, Dict, List, Optional

from ..catalog import MODEL_CATALOG
from ..settings import DISCLAIMER
from .analysis import EXPLANATION_LIMITATIONS
from .model_manager import manager

CSS = """
*{box-sizing:border-box}body{font-family:Inter,Segoe UI,Arial,sans-serif;color:#0f2744;background:#f4f6f9;margin:0}
.page{max-width:900px;margin:24px auto;background:#fff;border-radius:14px;padding:36px 44px;box-shadow:0 2px 12px rgba(15,39,68,.08)}
h1{font-size:22px;margin:0;letter-spacing:.02em}h2{font-size:15px;margin:28px 0 10px;color:#0f2744;border-bottom:1px solid #e3e8ef;padding-bottom:6px}
.sub{color:#4a6380;margin:4px 0 0;font-size:13px}.disc{background:#fff7e6;border:1px solid #f3d9a4;color:#7a4b00;border-radius:10px;padding:10px 14px;font-size:12.5px;margin:18px 0}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid #eef1f5;vertical-align:top}th{color:#4a6380;font-weight:600;width:38%}
.imgs{display:flex;gap:12px;flex-wrap:wrap}.imgs figure{margin:0;flex:1;min-width:200px}.imgs .stack{position:relative}.imgs img{width:100%;border-radius:8px;display:block}
.imgs .stack img+img{position:absolute;inset:0}figcaption{font-size:11.5px;color:#4a6380;margin-top:4px}
ul{font-size:12.5px;color:#28405e;padding-left:18px}li{margin:3px 0}.muted{color:#6b7f96;font-size:12px}
.btn{position:fixed;top:16px;right:16px;background:#0e7490;color:#fff;border:0;border-radius:8px;padding:9px 16px;font-size:13px;cursor:pointer}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;max-width:none;padding:0}.btn{display:none}}
"""


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return html.escape(str(v))


def _rows(pairs: List[tuple]) -> str:
    return "<table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in pairs) + "</table>"


def _shell(title: str, body: str) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style>"
            f"</head><body><button class='btn' onclick='window.print()'>Print / Save as PDF</button>"
            f"<div class='page'>{body}</div></body></html>")


def _evaluation_rows(key: str) -> str:
    d = manager.describe(key)
    ck = d["checkpoint"]
    val = ck.get("validation_metrics", {})
    ev = d.get("evaluation")
    pairs = [("Validation AUC (RSNA val, best epoch)", _fmt(val.get("val_auc"))),
             ("Validation Dice on positives", _fmt(val.get("val_dice_pos"))),
             ("Validation attention mass in box", _fmt(val.get("val_attn_mass_in_box")))]
    if ev:
        pairs += [("RSNA test AUC", _fmt(ev.get("rsna_test_auc"))),
                  ("RSNA test sensitivity / specificity", f"{_fmt(ev.get('sensitivity'))} / {_fmt(ev.get('specificity'))}"),
                  ("Expected calibration error (temperature-scaled)", _fmt(ev.get("ece_temperature_scaled"))),
                  ("NIH external AUC", _fmt(ev.get("nih_auc")))]
    else:
        pairs.append(("Test-set evaluation", "Not run yet for this checkpoint."))
    return _rows(pairs)


def analysis_report(rec: Dict[str, Any]) -> str:
    p, u, q, o, m = rec["prediction"], rec.get("uncertainty") or {}, rec["quality"], rec["ood"], rec["model"]
    imgs = rec.get("images", {})

    def fig(layer: Optional[str], caption: str) -> str:
        if "original" not in imgs or (layer and layer not in imgs):
            return ""
        top = f"<img src='{imgs[layer]}' alt='{layer} layer'>" if layer else ""
        return (f"<figure><div class='stack'><img src='{imgs['original']}' alt='preprocessed X-ray'>{top}</div>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>")

    body = [
        "<h1>RIGHT FOR THE RIGHT REASON</h1>",
        "<p class='sub'>Explanation-Guided Pneumonia Detection and Localization — research analysis report</p>",
        f"<div class='disc'><b>Disclaimer.</b> {html.escape(DISCLAIMER)} The model output below is not a diagnosis.</div>",
        "<h2>Analysis</h2>",
        _rows([("Analysis ID", _fmt(rec["analysis_id"])), ("Created", _fmt(rec["created_at"])),
               ("Report generated", dt.datetime.now().isoformat(timespec="seconds")),
               ("Input reference", f"{_fmt(rec['input_reference'])} <span class='muted'>(content hash; the original "
                                   f"file is not stored)</span>"),
               ("Input", f"{_fmt(rec['input']['kind'])}, {rec['input']['width']}×{rec['input']['height']} px")]),
        "<h2>Model</h2>",
        _rows([("Model", _fmt(m["name"])), ("Architecture", _fmt(m["architecture"])),
               ("Checkpoint identifier", _fmt(m.get("checkpoint_id"))),
               ("Localization head trained", _fmt(m.get("localization_trained"))),
               ("Attention guidance used in training", _fmt(m.get("attention_guided")))]),
        "<h2>Model output</h2>",
        _rows([("Result", f"<b>{_fmt(p['label'])}</b>"),
               ("Model probability (uncalibrated)", _fmt(p["probability"])),
               ("Calibrated probability", _fmt(p.get("calibrated_probability")) if p.get("calibrated_probability") is not None
                else "Not available (calibration not fitted)"),
               ("Decision threshold", f"{_fmt(p['threshold'])} <span class='muted'>({_fmt(p['threshold_source'])})</span>"),
               ("Abstention reasons", _fmt("; ".join(p["abstention_reasons"])) if p["abstention_reasons"] else "—")]),
        "<h2>Uncertainty, input quality and distribution check</h2>",
        _rows([("MC-dropout mean / std", f"{_fmt(u.get('mean'))} / {_fmt(u.get('std'))}"),
               ("5–95 % interval", f"{_fmt(u.get('p05'))} – {_fmt(u.get('p95'))}"),
               ("Input quality checks", ("passed" if q["passed"] else "failed: " + _fmt("; ".join(q["errors"])))
                + (f"<br><span class='muted'>warnings: {_fmt('; '.join(q['warnings']))}</span>" if q["warnings"] else "")),
               ("Out-of-distribution check", ("not available (statistics not fitted)" if not o.get("available") else
                                              ("flagged as unusual" if o.get("flag") else "within training range")))]),
        "<h2>Visualizations</h2><div class='imgs'>",
        fig(None, "Preprocessed X-ray (384 px)"),
        fig("localization", "Predicted localization (model output, viridis)") if "localization" in imgs else "",
        fig("gradcam", "Grad-CAM++ (post-hoc explanation, inferno)"),
        "</div><p class='muted'>Heatmaps are graded model outputs, not disease boundaries.</p>",
        "<h2>Available evaluation information</h2>", _evaluation_rows(rec["model_key"]),
        "<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in EXPLANATION_LIMITATIONS + [
            "Trained on RSNA 'Lung Opacity' annotations, which are pneumonia-related findings, not confirmed diagnoses.",
            "Performance on other hospitals, scanners, patient groups or lateral views is unknown.",
            "Input quality checks are simple heuristics, not a validated image-quality assessment."]) + "</ul>",
    ]
    return _shell(f"Analysis report {rec['analysis_id'][:8]}", "".join(body))


def research_report() -> str:
    models = manager.list_models()
    rows = []
    for d in models:
        ck = d["checkpoint"]
        ev = d.get("evaluation") or {}
        rows.append(f"<tr><td>{html.escape(d['name'])}</td><td>{'trained' if d['trained'] else 'not trained'}</td>"
                    f"<td>{_fmt(ck.get('validation_metrics', {}).get('val_auc'))}</td>"
                    f"<td>{_fmt(ev.get('rsna_test_auc'))}</td><td>{_fmt(ev.get('localization_dice'))}</td>"
                    f"<td>{_fmt(ev.get('nih_auc'))}</td></tr>")
    any_trained = any(d["trained"] for d in models)
    body = ("<h1>RIGHT FOR THE RIGHT REASON</h1><p class='sub'>Research status report — "
            f"{dt.datetime.now().isoformat(timespec='seconds')}</p>"
            f"<div class='disc'>{html.escape(DISCLAIMER)}</div><h2>Models</h2>"
            + ("" if any_trained else "<p><b>Models not trained yet.</b> No performance figures exist.</p>")
            + "<table><tr><th>Model</th><th>Status</th><th>Val AUC</th><th>RSNA test AUC</th><th>Loc. Dice</th>"
              "<th>NIH AUC</th></tr>" + "".join(rows) + "</table>"
            "<p class='muted'>All values are read from checkpoints and evaluation output files; '—' means not "
            "available.</p>")
    return _shell("Research status report", body)
