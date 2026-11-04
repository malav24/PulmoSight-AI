"""Actual on-disk dataset status (no invented statistics). Results cached briefly - counting
~112k NIH PNGs on every request would be slow."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ml.utils.config import PROJECT_ROOT, load_config, output_root, resolve_path

_CACHE: Dict[str, Any] = {"t": 0.0, "value": None}
TTL_SECONDS = 60


def _display(p: Path) -> str:
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "custom location (set by environment variable)"


def _count(directory: Path, pattern: str, recursive: bool = False) -> int:
    if not directory.exists():
        return 0
    it = directory.rglob(pattern) if recursive else directory.glob(pattern)
    return sum(1 for _ in it)


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except json.JSONDecodeError:
        return None


def rsna_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    r = cfg["data"]["rsna"]
    root = resolve_path(r["root"])
    labels = root / r["labels_csv"]
    splits_csv = resolve_path(r["splits_csv"])
    n_dcm = _count(root / r["image_dir"], "*.dcm")
    split_counts = None
    if splits_csv.exists():
        try:
            s = pd.read_csv(splits_csv)
            split_counts = {k: {"images": int(len(g)), "positives": int(g["target"].sum())}
                            for k, g in s.groupby("split")}
        except Exception:  # noqa: BLE001
            split_counts = "unreadable"
    cache = resolve_path(r["cache_dir"]) if r.get("cache_dir") else None
    status = {
        "name": "RSNA Pneumonia Detection Challenge",
        "role": "training, validation and internal test",
        "directory": _display(root),
        "directory_exists": root.exists(),
        "annotation_csv": labels.exists(),
        "class_info_csv": (root / r["class_info_csv"]).exists(),
        "image_count": n_dcm,
        "images_available": n_dcm > 0,
        "split_file": splits_csv.exists(),
        "split_counts": split_counts,
        "cache_configured": bool(cache),
        "cache_ready": bool(cache and (cache / "cache_meta.json").exists()),
        "statistics": _read_json(output_root() / "reports" / "rsna_dataset_statistics.json"),
    }
    status["ready_for_training"] = bool(status["annotation_csv"] and status["images_available"])
    status["preparation_status"] = ("prepared" if status["split_file"] else
                                    "ready to prepare" if status["ready_for_training"] else "not downloaded")
    return status


def nih_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    n = cfg["data"]["nih"]
    root = resolve_path(n["root"])
    from ml.datasets.nih import find_entries_csv, is_nih_sample
    found_entries = find_entries_csv(root, n["entries_csv"])
    entries = found_entries if found_entries is not None else root / n["entries_csv"]
    n_png = _count(root, "*.png", recursive=True)
    status = {
        "name": "NIH ChestX-ray14",
        "role": "external evaluation only (never used for training)",
        "directory": _display(root),
        "directory_exists": root.exists(),
        "metadata_csv": entries.exists(),
        "variant": ("random sample (~5 %)" if is_nih_sample(found_entries) else
                    "full dataset" if found_entries is not None else None),
        "test_list": (root / n["test_list"]).exists(),
        "bbox_csv": (root / n["bbox_csv"]).exists(),
        "rsna_overlap_mapping": resolve_path(n["rsna_mapping_json"]).exists()
                                or (root / "pneumonia-challenge-dataset-mappings_2018.json").exists(),
        "image_count": n_png,
        "images_available": n_png > 0,
        "statistics": _read_json(output_root() / "reports" / "nih_dataset_statistics.json"),
    }
    status["ready_for_external_evaluation"] = bool(status["metadata_csv"] and status["images_available"])
    status["external_validity_warning"] = (None if status["rsna_overlap_mapping"] else
                                           "RSNA overlap mapping file missing: NIH images re-used in RSNA cannot be "
                                           "excluded, so NIH results would not be a valid external evaluation.")
    return status


def dataset_status(force: bool = False) -> Dict[str, Any]:
    now = time.time()
    if not force and _CACHE["value"] is not None and now - _CACHE["t"] < TTL_SECONDS:
        return _CACHE["value"]
    cfg = load_config("base.yaml")
    value = {"rsna": rsna_status(cfg), "nih": nih_status(cfg), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "instructions": INSTRUCTIONS}
    _CACHE.update(t=now, value=value)
    return value


INSTRUCTIONS = {
    "rsna": [
        "Accept the competition rules at kaggle.com/c/rsna-pneumonia-detection-challenge/rules, then sign in once "
        "on the command line:  kaggle auth login",
        "Run:  python scripts/download_datasets.py  (downloads, extracts and prepares RSNA and the NIH sample), or "
        "place an existing copy in data/rsna/ (any sub-folder or the Kaggle zip works; Downloads/Desktop/Documents "
        "are searched too) - details below.",
        "Download and unzip into the project folder data/rsna/ so that stage_2_train_labels.csv, "
        "stage_2_detailed_class_info.csv and the stage_2_train_images/ folder sit directly inside it.",
        "Run: python scripts/prepare_rsna.py --cache   (creates patient-level splits and a fast image cache).",
    ],
    "nih": [
        "Small option (recommended if disk space is limited): python scripts/download_datasets.py --skip-rsna "
        "downloads the official ~5 % NIH random sample from Kaggle into data/nih/.",
        "Full dataset (> 40 GB): download from nihcc.app.box.com/v/ChestXray-NIHCC into data/nih/ "
        "(Data_Entry_2017_v2020.csv, test_list.txt, BBox_List_2017.csv and images_001 … images_012).",
        "Also download the RSNA → NIH mapping file (pneumonia-challenge-dataset-mappings_2018.json, linked on the RSNA "
        "challenge page) into data/nih/ so overlapping images are excluded.",
        "Run: python scripts/prepare_nih.py",
    ],
}
