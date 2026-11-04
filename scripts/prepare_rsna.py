"""Parse RSNA labels, create patient-level splits, compute statistics, optionally build a PNG cache.

    python scripts/prepare_rsna.py                      # splits + stats
    python scripts/prepare_rsna.py --cache --workers 8  # + preprocessed 384px cache (much faster training)
Then set `data.rsna.cache_dir` in the config (or pass data.rsna.cache_dir=... on the command line).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ml.datasets.preprocessing import Preprocessor, imwrite_any_path, load_dicom, read_dicom_header  # noqa: E402
from ml.datasets.rsna import compute_rsna_statistics, load_or_create_splits  # noqa: E402
from ml.utils.config import output_root, load_config, parse_cli_overrides, resolve_path  # noqa: E402
from ml.utils.logging_utils import save_json  # noqa: E402


def _cache_one(args):
    dcm, out_png, pre_cfg = args
    raw, meta = load_dicom(dcm)
    img, _ = Preprocessor(pre_cfg)(raw)
    imwrite_any_path(out_png, img)
    return meta["rows"], meta["cols"], meta.get("view_position", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. data.preprocessing.image_size=512")
    ap.add_argument("--cache", action="store_true", help="write preprocessed PNG cache")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--headers", action="store_true", help="read DICOM headers to record size/view (no cache)")
    ap.add_argument("--download", action="store_true",
                    help="download RSNA from Kaggle if it is not found (needs a Kaggle login: kaggle auth login, and accepted competition rules)")
    ap.add_argument("--no-search", action="store_true",
                    help="do not search Downloads/Desktop/Documents for an existing copy")
    a = ap.parse_args()

    cfg = load_config(a.config, parse_cli_overrides(a.overrides))
    rcfg = cfg["data"]["rsna"]
    from ml.datasets.locate import ensure_rsna
    try:
        root = ensure_rsna(resolve_path(rcfg["root"]), download=a.download, search=not a.no_search)
    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")
    if root is None:
        sys.exit("RSNA dataset not found (looked in the configured folder, its sub-folders, Downloads, Desktop and "
                 "Documents; no Kaggle zip found either).\n"
                 "  Download it automatically:  python scripts\\prepare_rsna.py --download --cache\n"
                 "  (first time: accept the rules at https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules "
                 "and place kaggle.json in ~/.kaggle/ - see docs/DATASET_SETUP.md)")
    rcfg["root"] = str(root)
    splits, boxes = load_or_create_splits(rcfg, cfg["data"]["split"], cfg["experiment"]["seed"])
    img_dir = root / rcfg["image_dir"]
    missing = [p for p in splits.patientId if not (img_dir / f"{p}.dcm").exists()]
    print(f"patients: {len(splits)} | missing DICOMs: {len(missing)}")
    if missing:
        print("  e.g.", missing[:5])

    pre_cfg = cfg["data"]["preprocessing"]
    if a.cache:
        cache_dir = resolve_path(a.cache_dir or f"data/rsna/cache_{pre_cfg['image_size']}"
                                 f"{'_clahe' if pre_cfg.get('clahe') else ''}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(img_dir / f"{p}.dcm", cache_dir / f"{p}.png", pre_cfg) for p in splits.patientId if p not in missing]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(tqdm(ex.map(_cache_one, jobs, chunksize=32), total=len(jobs), desc="caching"))
        dims = pd.DataFrame(res, columns=["orig_h", "orig_w", "view_position"])
        dims["patientId"] = [j[0].stem for j in jobs]
        splits = splits.drop(columns=[c for c in ("orig_h", "orig_w", "view_position") if c in splits]).merge(
            dims, on="patientId", how="left")
        with open(cache_dir / "cache_meta.json", "w") as fh:
            json.dump({"preprocessing": Preprocessor(pre_cfg).signature(), "n_images": len(jobs)}, fh, indent=2)
        print(f"cache written to {cache_dir}; set data.rsna.cache_dir to use it")
    elif a.headers:
        hdr = [read_dicom_header(img_dir / f"{p}.dcm") for p in tqdm(splits.patientId, desc="headers")]
        splits["orig_h"] = [h["rows"] for h in hdr]
        splits["orig_w"] = [h["cols"] for h in hdr]
        splits["view_position"] = [h["view_position"] for h in hdr]
    splits.to_csv(resolve_path(rcfg["splits_csv"]), index=False)

    stats = compute_rsna_statistics(splits, boxes)
    if "view_position" in splits:
        stats["view_positions"] = splits.view_position.value_counts().to_dict()
    out = save_json(stats, output_root() / f"reports/rsna_dataset_statistics.json")
    print(json.dumps(stats, indent=2, default=str))
    print(f"statistics saved to {out}")


if __name__ == "__main__":
    main()
