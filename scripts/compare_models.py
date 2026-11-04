"""Paired comparison of models on the same RSNA test images (DeLong + Holm correction).

    python scripts/compare_models.py --models model_b model_a resnet50 densenet121 efficientnet_b0 vit_small
Requires outputs/predictions/<model>_rsna_test.csv from ml.evaluation.evaluate.
The first model is the reference; each other model is compared against it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from ml.evaluation.statistics import delong_paired_test, holm_correction  # noqa: E402
from ml.utils.config import output_root, resolve_path  # noqa: E402
from ml.utils.logging_utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default="rsna_test", help="rsna_test | nih")
    a = ap.parse_args()
    dfs = {}
    for m in a.models:
        p = output_root() / f"predictions/{m}_{a.dataset}.csv"
        if not p.exists():
            sys.exit(f"Missing predictions for {m}: {p} (run ml.evaluation.evaluate first)")
        dfs[m] = pd.read_csv(p)[["id", "label", "prob"]].rename(columns={"prob": m})
    merged = None
    for m, df in dfs.items():
        merged = df if merged is None else merged.merge(df.drop(columns="label"), on="id", how="inner")
    ref = a.models[0]
    results, pvals = {}, {}
    for m in a.models[1:]:
        r = delong_paired_test(merged.label.to_numpy(), merged[ref].to_numpy(), merged[m].to_numpy())
        results[f"{ref}_vs_{m}"] = r
        pvals[f"{ref}_vs_{m}"] = r["p_value"]
    for k, adj in holm_correction(pvals).items():
        results[k]["p_value_holm"] = adj
    results["_n_paired_images"] = int(len(merged))
    out = save_json(results, output_root() / f"reports/model_comparison_{a.dataset}.json")
    print(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
