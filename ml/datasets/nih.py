"""NIH ChestX-ray14 - EXTERNAL EVALUATION ONLY (never used for training or model selection).

Source: https://nihcc.app.box.com/v/ChestXray-NIHCC
Expected files under `data/nih/`:
    Data_Entry_2017_v2020.csv   (or Data_Entry_2017.csv, or sample_labels.csv for the 5 % NIH random sample)
    test_list.txt / train_val_list.txt  (official patient-level split)
    BBox_List_2017.csv          (optional, ~1k expert boxes for 8 findings, ~120 Pneumonia)
    images_001/images/*.png ... images_012/images/*.png  (any nesting under root works)
    pneumonia-challenge-dataset-mappings_2018.json  (optional but strongly recommended, from RSNA)

Important caveats (see docs/DATASET_SETUP.md):
  * Labels were text-mined from radiology reports (NLP); the Pneumonia label is noisy and is a
    different definition from RSNA's radiologist "Lung Opacity" annotation.
  * The RSNA challenge images were themselves drawn from this NIH dataset, so without removing
    the overlap the "external" set contains images the model was trained on. Use the RSNA mapping
    JSON with `exclude_rsna_overlap: true`.
  * Both datasets come from the NIH Clinical Center: this is a label-definition / distribution
    shift test, not a multi-institution external validation.
No bounding boxes are fabricated: only the official BBox_List_2017 pneumonia boxes are used.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.utils.config import resolve_path
from .preprocessing import Preprocessor, boxes_to_mask, load_raster, transform_boxes
from .transforms import pad_boxes

NIH_ID_RE = re.compile(r"^\d{8}_\d{3}(\.png)?$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


ENTRIES_CANDIDATES = ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv", "sample_labels.csv")


def find_entries_csv(root: str | Path, configured: Optional[str] = None) -> Optional[Path]:
    """The NIH metadata CSV: the configured name, the full-dataset names, or the random sample's sample_labels.csv."""
    root = Path(root)
    for name in ([configured] if configured else []) + list(ENTRIES_CANDIDATES):
        if name and (root / name).is_file():
            return root / name
    return None


def is_nih_sample(entries_csv: Optional[Path]) -> bool:
    return entries_csv is not None and Path(entries_csv).name == "sample_labels.csv"


def parse_nih_entries(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename = {"Image Index": "image_index", "Finding Labels": "finding_labels", "Patient ID": "patient_id",
              "Follow-up #": "follow_up", "Patient Age": "age", "Patient Gender": "sex",
              "View Position": "view_position"}
    df = df.rename(columns=rename)
    for col in ("image_index", "finding_labels", "patient_id"):
        if col not in df:
            raise ValueError(f"{csv_path} missing column for {col}")
    w_col = next((c for c in df.columns if c.startswith("OriginalImage[Width") or c == "OriginalImageWidth"), None)
    h_col = next((c for c in df.columns if c.startswith("Height]") or c == "OriginalImageHeight"), None)
    if w_col and h_col:
        df = df.rename(columns={w_col: "orig_w", h_col: "orig_h"})
    labels = df["finding_labels"].fillna("").str.split("|")
    df["labels"] = labels
    df["pneumonia"] = labels.apply(lambda ls: int("Pneumonia" in ls))
    df["no_finding"] = labels.apply(lambda ls: int(ls == ["No Finding"]))
    return df


def load_rsna_overlap_ids(mapping_json: str | Path) -> Set[str]:
    """Return NIH image filenames (e.g. '00000013_005.png') that appear in the RSNA challenge.

    The key names inside the RSNA mapping file are detected from the value format
    (NIH ids look like 00000013_005, RSNA ids are UUIDs) rather than hard-coded.
    """
    with open(mapping_json, encoding="utf-8") as fh:
        data = json.load(fh)
    records = data if isinstance(data, list) else data.get("data", data.get("images", []))
    if isinstance(records, dict):
        records = [{"k": k, "v": v} for k, v in records.items()]
    nih_ids: Set[str] = set()
    for rec in records:
        values = rec.values() if isinstance(rec, dict) else [rec]
        for v in values:
            if isinstance(v, str) and NIH_ID_RE.match(v.strip()):
                v = v.strip()
                nih_ids.add(v if v.endswith(".png") else v + ".png")
    if not nih_ids:
        raise ValueError(f"No NIH image ids found in {mapping_json}; check the file format.")
    return nih_ids


def index_nih_images(root: str | Path) -> Dict[str, Path]:
    root = Path(root)
    return {p.name: p for p in root.rglob("*.png")}


def load_nih_pneumonia_bboxes(bbox_csv: str | Path) -> pd.DataFrame:
    """Official NIH boxes (x, y, w, h in original 1024px coords) restricted to Pneumonia."""
    raw = pd.read_csv(bbox_csv)
    cols = list(raw.columns)
    df = pd.DataFrame({"image_index": raw[cols[0]], "finding": raw[cols[1]],
                       "x": raw[cols[2]], "y": raw[cols[3]], "w": raw[cols[4]], "h": raw[cols[5]]})
    df = df[df.finding == "Pneumonia"].copy()
    df["x1"], df["y1"] = df.x, df.y
    df["x2"], df["y2"] = df.x + df.w, df.y + df.h
    return df[["image_index", "x1", "y1", "x2", "y2"]].reset_index(drop=True)


def select_nih_external_set(df: pd.DataFrame, cfg: Dict[str, Any], root: Path, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    notes: List[str] = []
    if cfg.get("subset", "official_test") == "official_test":
        test_list = root / cfg.get("test_list", "test_list.txt")
        if test_list.exists():
            names = set(test_list.read_text().split())
            out = out[out.image_index.isin(names)]
        else:
            # e.g. the NIH random sample, which ships without test_list.txt. NIH is never used for training,
            # so every image is unseen by the model; only the RSNA-overlap exclusion below matters.
            msg = (f"{test_list.name} not found - using all {len(out)} NIH images listed (none are used for training)")
            warnings.warn(msg)
            notes.append(msg)
    views = cfg.get("view_positions")
    if views and "view_position" in out:
        out = out[out.view_position.isin(views)]
    if cfg.get("negative_definition", "all_non_pneumonia") == "no_finding":
        out = out[(out.pneumonia == 1) | (out.no_finding == 1)]

    if cfg.get("exclude_rsna_overlap", True):
        mapping = resolve_path(cfg.get("rsna_mapping_json")) if cfg.get("rsna_mapping_json") else None
        if mapping is None or not mapping.exists():  # also accept the file stored next to the NIH data
            alt = root / "pneumonia-challenge-dataset-mappings_2018.json"
            mapping = alt if alt.exists() else mapping
        if mapping is not None and mapping.exists():
            overlap = load_rsna_overlap_ids(mapping)
            before = len(out)
            out = out[~out.image_index.isin(overlap)]
            notes.append(f"excluded {before - len(out)} NIH images that also appear in RSNA")
        else:
            msg = ("RSNA->NIH mapping file not found; NIH images re-used in the RSNA challenge are NOT excluded. "
                   "The NIH evaluation is therefore contaminated and must not be reported as external.")
            warnings.warn(msg)
            notes.append(msg)
    if cfg.get("one_image_per_patient", False):
        out = out.sample(frac=1.0, random_state=seed).drop_duplicates("patient_id")
    if cfg.get("max_samples"):
        out = out.sample(n=min(int(cfg["max_samples"]), len(out)), random_state=seed)
    out = out.reset_index(drop=True)
    out.attrs["selection_notes"] = notes
    return out


def compute_nih_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_images": int(len(df)),
        "n_patients": int(df.patient_id.nunique()),
        "n_pneumonia": int(df.pneumonia.sum()),
        "prevalence": float(df.pneumonia.mean()) if len(df) else 0.0,
        "n_no_finding": int(df.no_finding.sum()),
        "pneumonia_only_label": int(df.finding_labels.eq("Pneumonia").sum()),
        "view_positions": df.view_position.value_counts().to_dict() if "view_position" in df else {},
        "images_per_patient_mean": float(df.groupby("patient_id").size().mean()) if len(df) else 0.0,
        "selection_notes": df.attrs.get("selection_notes", []),
    }


class NIHChestXrayDataset(Dataset):
    """External evaluation dataset. No augmentation; boxes only where officially provided."""

    def __init__(self, root: str | Path, records: pd.DataFrame, preprocessing: Dict[str, Any],
                 bboxes: Optional[pd.DataFrame] = None, image_index: Optional[Dict[str, Path]] = None,
                 max_boxes: int = 8):
        self.root = Path(root)
        self.records = records.reset_index(drop=True)
        self.pre = Preprocessor(preprocessing)
        self.max_boxes = max_boxes
        self.paths = image_index if image_index is not None else index_nih_images(self.root)
        missing = [n for n in self.records.image_index if n not in self.paths]
        if missing:
            raise FileNotFoundError(f"{len(missing)} NIH images listed but not found under {self.root}, "
                                    f"e.g. {missing[:3]}")
        self.box_lookup: Dict[str, np.ndarray] = {}
        if bboxes is not None:
            self.box_lookup = {n: g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
                               for n, g in bboxes.groupby("image_index")}

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> np.ndarray:
        return self.records["pneumonia"].to_numpy().astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records.iloc[idx]
        name = row["image_index"]
        raw, _ = load_raster(self.paths[name])
        img, geom = self.pre(raw)
        official = self.box_lookup.get(name)
        boxes = transform_boxes(official, geom) if official is not None else np.zeros((0, 4), np.float32)
        mask = boxes_to_mask(boxes, img.shape[0], img.shape[1])
        padded, n = pad_boxes(boxes, self.max_boxes)
        return {
            "image": self.pre.to_tensor(img),
            "label": torch.tensor(float(row["pneumonia"]), dtype=torch.float32),
            "mask": torch.from_numpy(mask[None]),
            "boxes": torch.from_numpy(padded),
            "n_boxes": torch.tensor(n, dtype=torch.long),
            "has_boxes": torch.tensor(bool(n > 0)),
            "id": name,
            "patient_id": str(row["patient_id"]),
            "dataset": "nih",
        }
