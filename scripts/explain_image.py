"""Explain one or more X-rays with a TRAINED model: saves Original | Localization | Attention | Grad-CAM++ panels.

    python scripts/explain_image.py --model model_b path/to/image.dcm [more images ...]
    python scripts/explain_image.py --model resnet50 xray.png --boxes "264,152,213,379"

For an RSNA DICOM named <patientId>.dcm the radiologist boxes are read from data/rsna automatically.
Outputs: outputs/visualizations/explain_<model>_<image>.png and outputs/reports/explain_<model>_<image>.json
Requires a trained checkpoint (no fallback to random or ImageNet weights).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.datasets.preprocessing import boxes_to_mask, transform_boxes, xywh_to_xyxy  # noqa: E402
from ml.evaluation.localization import dice, iou, mass_in_box, pointing_game_hit  # noqa: E402
from ml.explainability.visualization import comparison_panel  # noqa: E402
from ml.inference.predictor import PneumoniaPredictor, to_json_safe  # noqa: E402
from ml.utils.config import load_config, output_root, resolve_path  # noqa: E402


def rsna_boxes(image: Path):
    r = load_config("base.yaml")["data"]["rsna"]
    labels = resolve_path(r["root"]) / r["labels_csv"]
    if not labels.exists():
        return None
    df = pd.read_csv(labels).dropna(subset=["x"])
    rows = df[df.patientId == image.stem]
    return rows[["x", "y", "width", "height"]].to_numpy() if len(rows) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="model_b")
    ap.add_argument("--boxes", help='reference boxes "x,y,w,h;x,y,w,h" in original pixels')
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    from ml.utils.checkpoints import CheckpointNotFoundError
    try:
        pred = PneumoniaPredictor(a.model, device=a.device, mc_samples=10)
    except CheckpointNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")
    for w in pred.warnings:
        print("WARNING:", w)
    vis, rep = output_root() / "visualizations", output_root() / "reports"
    vis.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    for img_path in map(Path, a.images):
        res = pred.predict(img_path, explain=True, gradcam=True, uncertainty=True)
        geom = pred.last_geometry
        boxes_xywh = (np.array([[float(v) for v in b.split(",")] for b in a.boxes.split(";")]) if a.boxes
                      else rsna_boxes(img_path))
        boxes = transform_boxes(xywh_to_xyxy(boxes_xywh), geom) if boxes_xywh is not None else None
        x, img_u8, *_ = pred.load_input(img_path)
        maps = {"Grad-CAM++": res["maps"]["gradcam"]["array"]}
        if "localization" in res["maps"]:
            maps = {"Localization": res["maps"]["localization"]["array"], **maps}
        if "attention" in res["maps"]:
            maps["Attention"] = res["maps"]["attention"]["array"]
        metrics = {}
        if boxes is not None and len(boxes):
            gt = boxes_to_mask(boxes, geom.out_size, geom.out_size)
            for name, m in maps.items():
                metrics[name] = {"pointing_game_hit": pointing_game_hit(m, gt), "mass_in_box": mass_in_box(m, gt),
                                 "iou@0.5": iou((m >= 0.5).astype(np.float32), gt),
                                 "dice@0.5": dice((m >= 0.5).astype(np.float32), gt)}
        stem = f"explain_{a.model}_{img_path.stem}"
        comparison_panel(img_u8 / 255.0, maps, boxes, title=f"{a.model} | {res['prediction_label']}",
                         probability=res["probability"], path=str(vis / f"{stem}.png"))
        out = to_json_safe({k: v for k, v in res.items() if k != "maps"})
        out["reference_box_metrics"] = metrics
        (rep / f"{stem}.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"{img_path.name}: p={res['probability']:.3f} ({res['prediction_label']}) -> {vis / (stem + '.png')}")
        if metrics:
            print("   ", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                         for k, v in metrics.items()})


if __name__ == "__main__":
    main()
