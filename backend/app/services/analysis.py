"""Analysis and explanation workflows built on the Stage-1 PneumoniaPredictor."""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ml.datasets.preprocessing import boxes_to_mask, transform_boxes, xywh_to_xyxy
from ml.evaluation.explanation import attention_alignment
from ml.evaluation.localization import dice, iou, mass_in_box, pointing_game_hit
from ml.explainability.visualization import MAP_DESCRIPTIONS
from ml.utils.config import load_config, resolve_path

from ..catalog import MODEL_CATALOG
from ..errors import APIError
from ..settings import DISCLAIMER
from . import analyses
from .imaging import COLORMAPS, heatmap_layer, peak_location, png_b64, region_summary
from .model_manager import manager

log = logging.getLogger("rfrr.api.analysis")

OUTCOME_TEXT = {
    "suspected": "Pneumonia-related opacity pattern suspected",
    "not_detected": "No pneumonia-related opacity pattern detected",
    "abstained": "No result - the system abstained (see reasons)",
}

EXPLANATION_LIMITATIONS = [
    "These maps are computational outputs of the model. They are not proof of clinical reasoning and do not "
    "mark confirmed disease boundaries.",
    "The localization map is a learned prediction of where RSNA radiologists drew boxes around lung opacity.",
    "Grad-CAM++ shows which image regions increased the model's score; it can highlight regions that are not "
    "medically relevant, and different explanation methods can disagree.",
    "Agreement with a reference box (IoU, Dice, pointing game) shows plausibility, not that the explanation "
    "faithfully reflects the model's computation.",
    "RSNA reference boxes are coarse rectangles drawn by radiologists and vary between readers.",
]


def _model_block(key: str, pred) -> Dict[str, Any]:
    info = manager.checkpoint_info(key)
    cat = MODEL_CATALOG[key]
    return {"key": key, "name": cat["name"], "architecture": cat["architecture"], "family": cat["family"],
            "checkpoint_id": info.get("checkpoint_id"), "checkpoint_epoch": info.get("epoch"),
            "localization_trained": pred.localization_trained, "attention_guided": pred.attention_guided}


def _outcome(result: Dict[str, Any]) -> str:
    if result["abstention"]["abstain"]:
        return "abstained"
    return "suspected" if result["prediction"] == 1 else "not_detected"


def _layers(result: Dict[str, Any]) -> Dict[str, str]:
    images = {"original": "data:image/png;base64," + result["preprocessed_png_base64"]}
    for name, entry in result["maps"].items():
        # localization is a per-pixel probability: show it on an absolute scale (not re-normalised)
        images[name] = png_b64(heatmap_layer(entry["array"], name, normalize=name != "localization"))
    return images


def _localization_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    loc = result["maps"].get("localization")
    if loc is None:
        return {"available": False,
                "reason": ("This model has no trained localization head; only Grad-CAM++ is available."
                           if not result["localization_trained"] else "Localization map not produced.")}
    hm = loc["array"]
    return {"available": True, "peak_probability": float(hm.max()),
            "area_fraction_above_0_5": float((hm >= 0.5).mean()), "peak_location": peak_location(hm),
            "top_zones": region_summary(hm)[:3]}


def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items() if k != "array"}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def run_analysis(upload: Dict[str, Any], model_key: Optional[str]) -> Dict[str, Any]:
    key = manager.resolve_key(model_key)
    pred = manager.get_predictor(key)
    with manager.lock_for(key):
        try:
            result = pred.predict(upload["path"], explain=True, gradcam=True, uncertainty=True, return_images=True)
        except (ValueError, OSError) as exc:
            log.warning("Image could not be processed: %s", exc)
            raise APIError(422, "unreadable_image", "The image could not be decoded or preprocessed.") from exc
    outcome = _outcome(result)
    analysis_id = uuid.uuid4().hex
    record = {
        "analysis_id": analysis_id,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model_key": key, "model_name": MODEL_CATALOG[key]["name"],
        "model": _model_block(key, pred),
        "input_reference": upload["reference"],
        "input": {"kind": upload["kind"], "width": upload["width"], "height": upload["height"],
                  "bytes": upload["bytes"], "dicom_view": result["input_meta"].get("view_position")},
        "prediction": {
            "probability": result["probability"], "calibrated_probability": result["calibrated_probability"],
            "temperature": result["temperature"], "threshold": result["threshold"],
            "threshold_source": result["threshold_source"], "outcome": outcome, "label": OUTCOME_TEXT[outcome],
            "abstained": outcome == "abstained", "abstention_reasons": result["abstention"]["reasons"],
        },
        "configuration_warnings": result.get("configuration_warnings", []),
        "uncertainty": result["uncertainty"],
        "quality": result["quality"],
        "ood": result["ood"],
        "localization": _localization_summary(result),
        "gradcam_top_zones": region_summary(result["maps"]["gradcam"]["array"])[:3],
        "image_size": result["preprocessing_geometry"]["out_size"],
        "map_descriptions": {k: MAP_DESCRIPTIONS[k] for k in result["maps"]},
        "colormaps": {k: COLORMAPS[k][0] for k in result["maps"]},
        "disclaimer": DISCLAIMER,
    }
    record = _json_safe(record)
    images = _layers(result)
    analyses.save(record, images)
    return {**record, "images": images, "input": {**record["input"], "file_name": upload["display_name"]},
            "_maps": result["maps"], "_geometry": result["preprocessing_geometry"], "_pred": pred}


# ------------------------------------------------------------------ explanation
@lru_cache(maxsize=1)
def _rsna_boxes_by_id(labels_path: str, mtime: float) -> Dict[str, List[List[float]]]:
    import pandas as pd
    df = pd.read_csv(labels_path)
    df = df.dropna(subset=["x", "y", "width", "height"])
    return {pid: g[["x", "y", "width", "height"]].values.tolist() for pid, g in df.groupby("patientId")}


def lookup_rsna_reference(display_name: str) -> Optional[List[List[float]]]:
    cfg = load_config("base.yaml")["data"]["rsna"]
    p = resolve_path(cfg["root"]) / cfg["labels_csv"]
    if not p.exists():
        return None
    stem = Path(display_name).stem
    return _rsna_boxes_by_id(str(p), p.stat().st_mtime).get(stem)


def parse_boxes(raw: Optional[str]) -> Optional[List[List[float]]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        boxes = [[float(v) for v in (b if isinstance(b, (list, tuple)) else (b["x"], b["y"], b["width"], b["height"]))]
                 for b in data]
    except (ValueError, KeyError, TypeError) as exc:
        raise APIError(422, "invalid_boxes", "reference_boxes must be a JSON list of [x, y, width, height] in "
                       "original image pixels.") from exc
    if not 0 < len(boxes) <= 16 or any(len(b) != 4 or b[2] <= 0 or b[3] <= 0 for b in boxes):
        raise APIError(422, "invalid_boxes", "Provide 1-16 boxes as [x, y, width, height] with positive size.")
    return boxes


def _map_metrics(hm: np.ndarray, gt: np.ndarray, threshold_ok: bool) -> Dict[str, Any]:
    out = {"pointing_game_hit": pointing_game_hit(hm, gt), "mass_in_box": mass_in_box(hm, gt)}
    if threshold_ok:
        pm = (hm >= 0.5).astype(np.float32)
        out.update(iou=iou(pm, gt), dice=dice(pm, gt), threshold=0.5)
    return out


def run_explanation(upload: Dict[str, Any], model_key: Optional[str], boxes_raw: Optional[str]) -> Dict[str, Any]:
    res = run_analysis(upload, model_key)
    maps, geom, pred = res.pop("_maps"), res.pop("_geometry"), res.pop("_pred")
    boxes_xywh = parse_boxes(boxes_raw)
    source = "provided" if boxes_xywh else None
    if boxes_xywh is None:
        boxes_xywh = lookup_rsna_reference(upload["display_name"])
        source = "rsna_annotations" if boxes_xywh else None
    reference: Dict[str, Any] = {"available": bool(boxes_xywh), "source": source, "boxes": []}
    metrics: Dict[str, Any] = {}
    if boxes_xywh:
        from ml.datasets.preprocessing import ResizeGeometry
        g = ResizeGeometry(**geom)
        boxes = transform_boxes(xywh_to_xyxy(np.asarray(boxes_xywh)), g)
        size = g.out_size
        gt = boxes_to_mask(boxes, size, size)
        reference["boxes"] = boxes.round(1).tolist()
        if "localization" in maps:
            metrics["localization"] = _map_metrics(maps["localization"]["array"], gt, True)
        metrics["gradcam"] = _map_metrics(maps["gradcam"]["array"], gt, True)
        metrics["gradcam"]["note"] = "IoU/Dice threshold 0.5 applied to a min-max normalised saliency map (arbitrary)."
        if getattr(pred.model, "has_localization", False):
            with manager.lock_for(res["model_key"]):
                x, *_ = pred.load_input(upload["path"])
                with torch.no_grad():
                    att = pred.forward(x)["attention"].float().cpu()
            al = attention_alignment(att, torch.from_numpy(gt)[None, None])
            metrics["attention_alignment"] = {"mass_in_box": al.get("attention_mass_in_box"),
                                              "pointing_game": al.get("attention_pointing_game"),
                                              "kl_to_box": al.get("attention_kl_to_box")}
    influence = {"gradcam_zones": region_summary(maps["gradcam"]["array"])}
    if "localization" in maps:
        influence["localization_zones"] = region_summary(maps["localization"]["array"])
    if "attention" in maps:
        influence["attention_zones"] = region_summary(maps["attention"]["array"])
    return _json_safe({**res, "reference": reference, "metrics": metrics, "influence": influence,
                       "orientation_note": ("Zones are in image coordinates. On a standard frontal (PA) chest "
                                            "X-ray, the image-left side usually shows the patient's right lung."),
                       "limitations": EXPLANATION_LIMITATIONS})


def strip_private(res: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in res.items() if not k.startswith("_")}
