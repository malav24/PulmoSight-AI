"""Prepare and describe the NIH ChestX-ray14 external evaluation set (no training data is produced).

    python scripts/prepare_nih.py
Writes outputs/reports/nih_external_set.csv and nih_dataset_statistics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.datasets.nih import (compute_nih_statistics, index_nih_images, load_nih_pneumonia_bboxes,  # noqa: E402
                             parse_nih_entries, select_nih_external_set)
from ml.utils.config import output_root, load_config, parse_cli_overrides, resolve_path  # noqa: E402
from ml.utils.logging_utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. data.preprocessing.image_size=512")
    ap.add_argument("--download-sample", action="store_true",
                    help="if NIH is not found, download the official ~5 %% random sample from Kaggle (full set > 40 GB)")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_cli_overrides(a.overrides))
    ncfg = cfg["data"]["nih"]
    ap_root = resolve_path(ncfg["root"])
    from ml.datasets.locate import ensure_nih
    try:
        root = ensure_nih(ap_root, search=True, fetch_mapping=True, download_sample=a.download_sample)
    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")
    if root is None:
        sys.exit(f"NIH ChestX-ray14 not found (looked in {ap_root}, its sub-folders, Downloads, Desktop, Documents). "
                 "Get the small official sample automatically:  python scripts/prepare_nih.py --download-sample\n"
                 "  (full dataset, > 40 GB: https://nihcc.app.box.com/v/ChestXray-NIHCC - see docs/DATASET_SETUP.md)")
    ncfg["root"] = str(root)
    from ml.datasets.nih import find_entries_csv
    entries = find_entries_csv(root, ncfg.get("entries_csv"))
    if entries is None:
        sys.exit(f"NIH entries CSV not found under {root}. See docs/DATASET_SETUP.md")
    full = parse_nih_entries(entries)
    sel = select_nih_external_set(full, ncfg, root, cfg["experiment"]["seed"])
    index = index_nih_images(root)
    missing = [n for n in sel.image_index if n not in index]
    stats = {"full_dataset": compute_nih_statistics(full), "external_set": compute_nih_statistics(sel),
             "missing_image_files": len(missing)}
    bbox_csv = root / ncfg["bbox_csv"]
    if bbox_csv.exists():
        bb = load_nih_pneumonia_bboxes(bbox_csv)
        stats["official_pneumonia_boxes"] = {"n_boxes": len(bb), "n_images": int(bb.image_index.nunique()),
                                             "in_external_set": int(bb.image_index.isin(sel.image_index).sum())}
    (output_root() / "reports").mkdir(parents=True, exist_ok=True)
    sel.drop(columns=["labels"]).to_csv(output_root() / f"reports/nih_external_set.csv", index=False)
    save_json(stats, output_root() / f"reports/nih_dataset_statistics.json")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
