"""Model status report: what is implemented, trained, calibrated and ready for inference.

    python scripts/model_status.py            # prints a table, writes outputs/reports/model_status.{md,json}
    python scripts/model_status.py --build    # additionally instantiates every architecture (slower)

It reads files only - it never trains, and it reports a checkpoint as usable only if it loads strictly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.utils.checkpoints import (CHECKPOINT_REGISTRY, calibration_matches, checkpoint_path,  # noqa: E402
                                  load_calibration, read_checkpoint)
from ml.utils.config import PROJECT_ROOT, checkpoint_root, load_config, output_root  # noqa: E402

CONFIGS = {"model_a": "model_a.yaml", "model_b": "model_b.yaml", "ablation_cls_only": "ablation_cls_only.yaml",
           "resnet50": "resnet50.yaml", "densenet121": "densenet121.yaml",
           "efficientnet_b0": "efficientnet_b0.yaml", "vit_small": "vit_small.yaml"}
SCRIPTS = {"model_a": "train_model_a.py", "model_b": "train_model_b.py",
           "ablation_cls_only": "train_ablation_cls_only.py", "resnet50": "train_resnet50.py",
           "densenet121": "train_densenet121.py", "efficientnet_b0": "train_efficientnet.py",
           "vit_small": "train_vit.py"}


def status(key: str, build: bool) -> dict:
    cfg = load_config(CONFIGS[key])
    row = {"model": key, "config": f"ml/configs/{CONFIGS[key]}",
           "training_script": f"ml/training/{SCRIPTS[key]}",
           "training_script_exists": (PROJECT_ROOT / "ml" / "training" / SCRIPTS[key]).exists()}
    if build:
        from ml.models.factory import build_model, count_parameters
        try:
            m = build_model(dict(cfg["model"], pretrained_backbone=False), cfg["data"]["preprocessing"]["image_size"])
            row["architecture"] = f"implemented ({count_parameters(m)['total'] / 1e6:.1f} M params)"
        except Exception as exc:  # noqa: BLE001
            row["architecture"] = f"BUILD FAILED: {exc}"
    else:
        row["architecture"] = "implemented (use --build to instantiate)"
    p = checkpoint_path(key)
    try:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = str(p)
    row["checkpoint_path"] = rel
    meta = None
    if not p.exists():
        row["checkpoint"] = "missing"
    else:
        try:
            state = read_checkpoint(p)
            ok = state.get("model_key") == key
            row["checkpoint"] = f"found (epoch {state.get('epoch')}, saved {state.get('saved_at')})" if ok \
                else f"INVALID (stores {state.get('model_key')})"
            meta = state if ok else None
        except Exception as exc:  # noqa: BLE001
            row["checkpoint"] = f"INVALID ({type(exc).__name__})"
    cal = load_calibration(key)
    row["calibration"] = ("missing" if not cal else
                          f"found (T={cal.get('temperature'):.3f})" if meta and calibration_matches(cal, meta)
                          else "stale (fitted on another checkpoint)")
    ood = checkpoint_root() / "calibration" / f"{key}_ood_stats.npz"
    row["ood_statistics"] = "found" if ood.exists() else "missing"
    row["evaluation_metrics"] = "found" if (output_root() / "metrics" / f"{key}_metrics.json").exists() else "missing"
    row["ready_for_inference"] = meta is not None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    rows = [status(k, a.build) for k in CHECKPOINT_REGISTRY]
    cols = ["model", "architecture", "training_script_exists", "checkpoint", "calibration", "ood_statistics",
            "evaluation_metrics", "ready_for_inference"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    out = output_root() / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_status.md").write_text("# Model status\n\n" + "\n".join(md) + "\n\nCheckpoint locations:\n\n"
                                         + "\n".join(f"- {r['model']}: `{r['checkpoint_path']}`" for r in rows) + "\n",
                                         encoding="utf-8")
    (out / "model_status.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['model']:18s} {'READY' if r['ready_for_inference'] else 'not ready':10s} checkpoint: {r['checkpoint']:<45s}"
              f" calibration: {r['calibration']}")
    print(f"\nWritten: {out / 'model_status.md'}")


if __name__ == "__main__":
    main()
