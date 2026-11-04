"""Run the three-arm ablation with identical architecture, data, seed and schedule.

    A. classification only                      -> ablation_cls_only.yaml
    B. classification + localization            -> model_a.yaml  (Model A)
    C. classification + localization + guidance -> model_b.yaml  (Model B)

    python scripts/run_ablation.py --train          # trains all three (long!)
    python scripts/run_ablation.py --evaluate       # evaluates existing checkpoints
    python scripts/run_ablation.py --summarize      # table from outputs/metrics/*.json
Extra dotted overrides are applied to all arms, e.g.  training.epochs=20
For seed variability, repeat with experiment.seed=<n> and different checkpoint dirs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.utils.config import output_root, load_config, parse_cli_overrides, resolve_path  # noqa: E402
from ml.utils.logging_utils import save_json  # noqa: E402

ARMS = [("A_cls_only", "ablation_cls_only.yaml", "ablation_cls_only"),
        ("B_cls_loc", "model_a.yaml", "model_a"),
        ("C_cls_loc_attn", "model_b.yaml", "model_b")]


def summarize():
    rows = {}
    for arm, _, key in ARMS:
        p = output_root() / f"metrics/{key}_metrics.json"
        if not p.exists():
            rows[arm] = "not evaluated"
            continue
        m = json.loads(p.read_text())
        t = m.get("rsna_test", {})
        c = t.get("classification", {})
        rows[arm] = {
            "auc": c.get("roc_auc"), "auc_ci": c.get("roc_auc_bootstrap_ci"),
            "sensitivity": c.get("sensitivity"), "specificity": c.get("specificity"),
            "loc_dice": t.get("localization_heatmap", {}).get("dice"),
            "attn_mass_in_box": t.get("attention_alignment", {}).get("attention_mass_in_box"),
            "gradcam_pointing": t.get("explanations", {}).get("gradcam_localization", {}).get("pointing_game"),
            "deletion_auc": t.get("explanations", {}).get("gradcam_faithfulness", {}).get("deletion_auc"),
            "ece_scaled": t.get("calibration_temperature_scaled", {}).get("ece"),
            "nih_auc": m.get("nih_external", {}).get("classification", {}).get("roc_auc"),
        }
    out = save_json(rows, output_root() / f"reports/ablation_summary.json")
    print(json.dumps(rows, indent=2, default=str))
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--nih", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    ov = parse_cli_overrides(a.overrides)
    if a.train:
        from ml.training.trainer import run_training
        for arm, cfg_name, _ in ARMS:
            print(f"==== training arm {arm} ({cfg_name}) ====")
            run_training(load_config(cfg_name, ov))
    if a.evaluate:
        from ml.evaluation.evaluate import evaluate
        for arm, _, key in ARMS:
            print(f"==== evaluating arm {arm} ({key}) ====")
            evaluate(key, use_nih=a.nih, stress=True)
    if a.summarize or not (a.train or a.evaluate):
        summarize()


if __name__ == "__main__":
    main()
