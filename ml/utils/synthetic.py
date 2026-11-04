"""SYNTHETIC data for software tests and smoke runs ONLY.

Creates RSNA-format and NIH-format folders with random-noise "images" and random boxes. Nothing
produced from this data (losses, metrics, checkpoints) has any scientific or clinical meaning.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd


def write_synthetic_dicom(path: Path, pixels: np.ndarray, view: str = "PA", photometric: str = "MONOCHROME2",
                          slope: float = 1.0, intercept: float = 0.0) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CR"
    ds.ViewPosition = view
    ds.BodyPartExamined = "CHEST"
    ds.PatientID = "SYNTHETIC"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)   # pydicom >= 3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)  # pydicom 2.x


def _fake_cxr(rng: np.random.Generator, size: int, boxes) -> np.ndarray:
    img = rng.normal(1500, 200, (size, size)).clip(0, 4095)
    yy, xx = np.mgrid[0:size, 0:size]
    for cx in (0.3, 0.7):  # two darker "lung" ellipses
        lung = ((xx / size - cx) / 0.18) ** 2 + ((yy / size - 0.5) / 0.32) ** 2 <= 1
        img[lung] -= 600
    for x1, y1, x2, y2 in boxes:  # brighter "opacity"
        img[int(y1):int(y2), int(x1):int(x2)] += 500
    return img.clip(0, 4095)


def make_synthetic_rsna(root: Path, n: int = 24, size: int = 256, pos_frac: float = 0.4, seed: int = 0
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    root = Path(root)
    img_dir = root / "stage_2_train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows, cls_rows = [], []
    for i in range(n):
        pid = str(uuid.UUID(int=int(rng.integers(0, 2**63)) + i))
        positive = i < int(round(n * pos_frac))
        boxes = []
        if positive:
            for _ in range(int(rng.integers(1, 3))):
                w, h = rng.integers(size // 8, size // 4, 2)
                x, y = rng.integers(size // 8, size - w - size // 8), rng.integers(size // 8, size - h - size // 8)
                boxes.append((float(x), float(y), float(x + w), float(y + h)))
                rows.append({"patientId": pid, "x": float(x), "y": float(y), "width": float(w), "height": float(h),
                             "Target": 1})
            cls_rows.append({"patientId": pid, "class": "Lung Opacity"})
        else:
            rows.append({"patientId": pid, "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan, "Target": 0})
            cls_rows.append({"patientId": pid, "class": "Normal" if i % 2 else "No Lung Opacity / Not Normal"})
        write_synthetic_dicom(img_dir / f"{pid}.dcm", _fake_cxr(rng, size, boxes))
    labels = pd.DataFrame(rows)
    labels.to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(root / "stage_2_detailed_class_info.csv", index=False)
    return labels, pd.DataFrame(cls_rows)


def make_synthetic_nih(root: Path, n: int = 16, size: int = 256, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    root = Path(root)
    img_dir = root / "images_001" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows, bbox_rows = [], []
    for i in range(n):
        name = f"{i // 2 + 1:08d}_{i % 2:03d}.png"
        pneu = i % 3 == 0
        labels = "Pneumonia" if pneu else ("No Finding" if i % 3 == 1 else "Effusion")
        img = _fake_cxr(rng, size, [])
        cv2.imwrite(str(img_dir / name), (img / 4095 * 255).astype(np.uint8))
        rows.append({"Image Index": name, "Finding Labels": labels, "Follow-up #": i % 2, "Patient ID": i // 2 + 1,
                     "Patient Age": 50, "Patient Gender": "M", "View Position": "PA",
                     "OriginalImage[Width": size, "Height]": size, "OriginalImagePixelSpacing[x": 0.14, "y]": 0.14})
        if pneu and i == 0:
            bbox_rows.append({"Image Index": name, "Finding Label": "Pneumonia", "Bbox [x": 40.0, "y": 50.0,
                              "w": 60.0, "h": 70.0, "h]": None})
    df = pd.DataFrame(rows)
    df.to_csv(root / "Data_Entry_2017_v2020.csv", index=False)
    (root / "test_list.txt").write_text("\n".join(df["Image Index"]))
    pd.DataFrame(bbox_rows).iloc[:, :6].to_csv(root / "BBox_List_2017.csv", index=False)
    return df
