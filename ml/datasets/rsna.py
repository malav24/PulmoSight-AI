"""RSNA Pneumonia Detection Challenge (2018) dataset.

Source: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
Expected files under `data/rsna/`:
    stage_2_train_labels.csv          patientId,x,y,width,height,Target  (one row per box; Target=0 rows have no box)
    stage_2_detailed_class_info.csv   patientId,class   ("Lung Opacity" | "No Lung Opacity / Not Normal" | "Normal")
    stage_2_train_images/<patientId>.dcm

Label semantics: Target=1 means radiologists marked opacity suspicious for pneumonia
("Lung Opacity"). These are pneumonia-related lung-opacity annotations, NOT confirmed clinical
diagnoses. The public stage-2 test images have no labels, so the labelled training set is split
into train / val / test here at patient level.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.utils.config import resolve_path
from .preprocessing import (Preprocessor, ResizeGeometry, boxes_to_mask, compute_resize_geometry, imread_any_path,
                            load_dicom, transform_boxes, xywh_to_xyxy)
from .transforms import apply_augmentation, build_train_augmentation, pad_boxes

RSNA_CLASSES = ["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"]


# --------------------------------------------------------------------------- CSV parsing
def parse_rsna_labels(labels_csv: str | Path, class_info_csv: Optional[str | Path] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (patients, boxes).

    patients: one row per patientId with columns [patientId, target, n_boxes, class]
    boxes:    one row per box with columns [patientId, x1, y1, x2, y2] in original pixel coords
    """
    labels = pd.read_csv(labels_csv)
    required = {"patientId", "x", "y", "width", "height", "Target"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{labels_csv} is missing columns {missing}")

    has_box = labels[["x", "y", "width", "height"]].notna().all(axis=1)
    box_rows = labels[has_box].copy()
    xyxy = xywh_to_xyxy(box_rows[["x", "y", "width", "height"]].to_numpy())
    boxes = pd.DataFrame({"patientId": box_rows["patientId"].to_numpy(),
                          "x1": xyxy[:, 0], "y1": xyxy[:, 1], "x2": xyxy[:, 2], "y2": xyxy[:, 3]})
    boxes = boxes.drop_duplicates().reset_index(drop=True)

    patients = labels.groupby("patientId").agg(target=("Target", "max")).reset_index()
    counts = boxes.groupby("patientId").size().rename("n_boxes")
    patients = patients.merge(counts, on="patientId", how="left")
    patients["n_boxes"] = patients["n_boxes"].fillna(0).astype(int)
    patients["target"] = patients["target"].astype(int)

    inconsistent = patients[(patients.target == 1) != (patients.n_boxes > 0)]
    if len(inconsistent):
        raise ValueError(f"{len(inconsistent)} patients have Target/box inconsistency, e.g. "
                         f"{inconsistent.patientId.head(3).tolist()}")

    if class_info_csv is not None and Path(class_info_csv).exists():
        cls = pd.read_csv(class_info_csv).drop_duplicates("patientId")
        patients = patients.merge(cls[["patientId", "class"]], on="patientId", how="left")
    else:
        patients["class"] = np.where(patients.target == 1, "Lung Opacity", "Unknown")
    return patients, boxes


def make_patient_splits(patients: pd.DataFrame, ratios: Dict[str, float], seed: int = 42,
                        stratify_on: str = "class") -> pd.DataFrame:
    """Assign each patientId to exactly one of train/val/test, stratified.

    RSNA has one image per patientId, so patient-level == image-level here; grouping is still
    done on patientId so the guarantee holds if multiple images per id ever appear.
    """
    from sklearn.model_selection import train_test_split

    ratios = {k: float(v) for k, v in ratios.items()}
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {total}")
    per_patient = patients.drop_duplicates("patientId").reset_index(drop=True)
    strat_col = "class" if stratify_on == "class" and "class" in per_patient else "target"
    strat = per_patient[strat_col].astype(str)
    if strat.value_counts().min() < 3:  # tiny/synthetic data cannot be stratified
        strat = None

    test_frac = ratios["test"]
    val_frac = ratios["val"] / (1.0 - test_frac)
    trval, test = train_test_split(per_patient, test_size=test_frac, random_state=seed, stratify=strat)
    strat_trval = trval[strat_col].astype(str) if strat is not None else None
    train, val = train_test_split(trval, test_size=val_frac, random_state=seed, stratify=strat_trval)

    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    assert out.patientId.is_unique, "patient leakage across splits"
    return out.sort_values("patientId").reset_index(drop=True)


def compute_rsna_statistics(patients: pd.DataFrame, boxes: pd.DataFrame, image_size: Tuple[int, int] = (1024, 1024)
                            ) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "n_images": int(len(patients)),
        "n_positive": int(patients.target.sum()),
        "n_negative": int((patients.target == 0).sum()),
        "prevalence": float(patients.target.mean()) if len(patients) else 0.0,
        "class_counts": patients["class"].value_counts().to_dict() if "class" in patients else {},
        "n_boxes": int(len(boxes)),
        "boxes_per_positive_image": patients.loc[patients.target == 1, "n_boxes"].value_counts().sort_index().to_dict(),
    }
    if len(boxes):
        h, w = image_size
        area = ((boxes.x2 - boxes.x1) * (boxes.y2 - boxes.y1)) / float(h * w)
        stats["box_relative_area"] = {"mean": float(area.mean()), "median": float(area.median()),
                                      "min": float(area.min()), "max": float(area.max())}
    if "split" in patients:
        stats["splits"] = {s: {"n": int(len(g)), "n_positive": int(g.target.sum()),
                               "prevalence": float(g.target.mean())}
                           for s, g in patients.groupby("split")}
    return stats


def load_or_create_splits(rsna_cfg: Dict[str, Any], split_cfg: Dict[str, Any], seed: int,
                          create_if_missing: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = resolve_path(rsna_cfg["root"])
    patients, boxes = parse_rsna_labels(root / rsna_cfg["labels_csv"],
                                        root / rsna_cfg.get("class_info_csv", "stage_2_detailed_class_info.csv"))
    splits_csv = resolve_path(rsna_cfg["splits_csv"])
    if splits_csv.exists():
        splits = pd.read_csv(splits_csv)
        if set(splits.patientId) != set(patients.patientId):
            raise ValueError(f"{splits_csv} does not match the label CSV patient set; delete it and re-run "
                             "scripts/prepare_rsna.py")
    elif create_if_missing:
        splits = make_patient_splits(patients, {k: split_cfg[k] for k in ("train", "val", "test")}, seed,
                                     split_cfg.get("stratify_on", "class"))
        splits_csv.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(splits_csv, index=False)
    else:
        raise FileNotFoundError(f"Split file not found: {splits_csv}")
    return splits, boxes


# --------------------------------------------------------------------------- dataset
class RSNAPneumoniaDataset(Dataset):
    """Returns dict(image, label, mask, boxes, n_boxes, has_boxes, id, dataset)."""

    def __init__(self, root: str | Path, records: pd.DataFrame, boxes: pd.DataFrame, preprocessing: Dict[str, Any],
                 augmentation: Optional[Dict[str, Any]] = None, image_dir: str = "stage_2_train_images",
                 cache_dir: Optional[str | Path] = None, max_boxes: int = 8):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.records = records.reset_index(drop=True)
        self.pre = Preprocessor(preprocessing)
        self.aug = build_train_augmentation(augmentation) if augmentation else None
        self.max_boxes = int(max_boxes)
        self.box_lookup: Dict[str, np.ndarray] = {
            pid: g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32) for pid, g in boxes.groupby("patientId")}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self._validate_cache()

    def _validate_cache(self) -> None:
        meta_path = self.cache_dir / "cache_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Cache metadata missing: {meta_path}. Build it with scripts/prepare_rsna.py --cache")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("preprocessing") != self.pre.signature():
            raise ValueError(f"Cache {self.cache_dir} was built with {meta.get('preprocessing')} but the config "
                             f"requests {self.pre.signature()}. Rebuild the cache or set data.rsna.cache_dir: null")
        if not {"orig_h", "orig_w"}.issubset(self.records.columns):
            raise ValueError("Cached loading needs orig_h/orig_w columns in the split CSV "
                             "(scripts/prepare_rsna.py --cache adds them).")

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> np.ndarray:
        return self.records["target"].to_numpy().astype(np.float32)

    def _load(self, pid: str, row) -> Tuple[np.ndarray, ResizeGeometry]:
        if self.cache_dir is not None:
            img = imread_any_path(self.cache_dir / f"{pid}.png", cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Missing cached image for {pid} in {self.cache_dir}")
            geom = compute_resize_geometry(int(row["orig_h"]), int(row["orig_w"]), self.pre.size, self.pre.mode)
            return img, geom
        raw, _ = load_dicom(self.image_dir / f"{pid}.dcm")
        return self.pre(raw)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records.iloc[idx]
        pid = str(row["patientId"])
        img, geom = self._load(pid, row)
        boxes = transform_boxes(self.box_lookup.get(pid, np.zeros((0, 4), np.float32)), geom)
        mask = boxes_to_mask(boxes, img.shape[0], img.shape[1])
        img, mask, boxes = apply_augmentation(self.aug, img, mask, boxes)
        padded, n = pad_boxes(boxes, self.max_boxes)
        label = float(row["target"])
        return {
            "image": self.pre.to_tensor(img),
            "label": torch.tensor(label, dtype=torch.float32),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "boxes": torch.from_numpy(padded),
            "n_boxes": torch.tensor(n, dtype=torch.long),
            # Positive images always carry radiologist boxes in RSNA; augmentation may crop a box
            # away, in which case the mask (transformed with the image) is still the target.
            "has_boxes": torch.tensor(bool(label > 0 and mask.sum() > 0)),
            "id": pid,
            "dataset": "rsna",
        }
