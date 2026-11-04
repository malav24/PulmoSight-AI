"""One command to get both datasets ready.

    python scripts/download_datasets.py            # RSNA (full) + NIH random sample, then prepare both
    python scripts/download_datasets.py --skip-nih # only RSNA

What it does
  1. RSNA Pneumonia Detection Challenge (full labelled training set, a few GB):
     uses an existing copy if one is found (data/rsna, its sub-folders, Downloads/Desktop/Documents),
     otherwise downloads it from Kaggle, extracts it and deletes the zip.
  2. NIH ChestX-ray14: the FULL dataset is > 40 GB, so by default only the official ~5 % random sample
     (Kaggle dataset "nih-chest-xrays/sample", ~5,600 images) is downloaded. If you already have the full
     dataset somewhere, it is found and used instead.
  3. Downloads RSNA's public RSNA->NIH mapping file (to exclude overlapping images from the NIH test).
  4. Runs scripts/prepare_rsna.py --cache and scripts/prepare_nih.py.

Requirements (one time): a Kaggle account, the RSNA competition rules accepted at
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules, and  `kaggle auth login`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.locate import (NIH_SAMPLE_NEED_GB, RSNA_NEED_GB, ensure_nih, ensure_rsna,  # noqa: E402
                                free_gb, kaggle_api)
from ml.utils.config import load_config, resolve_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-rsna", action="store_true")
    ap.add_argument("--skip-nih", action="store_true")
    ap.add_argument("--no-prepare", action="store_true", help="only download, do not run the prepare scripts")
    ap.add_argument("--no-cache", action="store_true", help="prepare RSNA without the 384-px image cache")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config("base.yaml")
    rsna_root = resolve_path(cfg["data"]["rsna"]["root"])
    nih_root = resolve_path(cfg["data"]["nih"]["root"])
    print(f"Free disk space: {free_gb(rsna_root):.1f} GB  (RSNA needs ~{RSNA_NEED_GB:.0f} GB while downloading, "
          f"NIH sample ~{NIH_SAMPLE_NEED_GB:.0f} GB)\n")
    problems = []

    # 1. reuse existing copies (project folder, sub-folders, zips, Downloads/Desktop/Documents)
    rsna_found = nih_found = None
    if not a.skip_rsna:
        print("=== RSNA Pneumonia Detection Challenge: looking for an existing copy ===")
        rsna_found = ensure_rsna(rsna_root, download=False, search=True)
    if not a.skip_nih:
        print("=== NIH ChestX-ray14: looking for an existing copy ===")
        nih_found = ensure_nih(nih_root, search=True, fetch_mapping=True, download_sample=False)

    # 2. one Kaggle login check, only if something must be downloaded
    need_rsna = not a.skip_rsna and rsna_found is None
    need_nih = not a.skip_nih and nih_found is None
    if need_rsna or need_nih:
        print("\n=== Kaggle sign-in check ===")
        try:
            kaggle_api(interactive=True)
            print("Signed in to Kaggle.\n")
        except RuntimeError as exc:
            print(f"ERROR: {exc}\n")
            problems += (["RSNA"] if need_rsna else []) + (["NIH"] if need_nih else [])
            need_rsna = need_nih = False

    # 3. downloads
    if need_rsna:
        print("=== Downloading RSNA (full labelled training set) ===")
        try:
            rsna_found = ensure_rsna(rsna_root, download=True, search=False)
        except RuntimeError as exc:
            print(f"ERROR: {exc}\n")
        if rsna_found is None and "RSNA" not in problems:
            problems.append("RSNA")
    if need_nih:
        print("=== Downloading NIH ChestX-ray14 random sample (~5 %; the full set is > 40 GB) ===")
        try:
            nih_found = ensure_nih(nih_root, search=False, fetch_mapping=True, download_sample=True)
        except RuntimeError as exc:
            print(f"ERROR: {exc}\n")
        if nih_found is None and "NIH" not in problems:
            problems.append("NIH")
    if rsna_found:
        print(f"RSNA ready at {rsna_found}")
    if nih_found:
        print(f"NIH ready at {nih_found}")
    print()

    if not a.no_prepare:
        py = sys.executable
        if not a.skip_rsna and "RSNA" not in problems:
            print("=== Preparing RSNA (patient-level splits, statistics, image cache) ===")
            cmd = [py, "scripts/prepare_rsna.py", "--workers", str(a.workers), "--no-search"]
            if not a.no_cache:
                cmd.append("--cache")
            if subprocess.call(cmd, cwd=ROOT) != 0:
                problems.append("prepare_rsna")
        if not a.skip_nih and "NIH" not in problems:
            print("=== Preparing NIH external set ===")
            if subprocess.call([py, "scripts/prepare_nih.py"], cwd=ROOT) != 0:
                problems.append("prepare_nih")

    print("\n" + ("Done. Next: python scripts\\verify_setup.py, then train (docs/TRAINING_GUIDE.md)." if not problems
                  else f"Finished with problems in: {', '.join(problems)} - see the messages above."))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
