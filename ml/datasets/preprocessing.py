"""Image I/O and deterministic preprocessing shared by training, evaluation and inference.

Pipeline (per image):
    raw pixels (DICOM / PNG / JPEG)
      -> float32, rescale slope/intercept, MONOCHROME1 inversion
      -> percentile windowing to uint8
      -> optional CLAHE
      -> resize to image_size (stretch or letterbox) with recorded geometry
Bounding boxes are mapped with exactly the same geometry (`transform_boxes`).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

DICOM_SUFFIXES = {".dcm", ".dicom"}


# --------------------------------------------------------------------------- loading
def imread_any_path(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED):
    """cv2.imread that also works with non-ASCII Windows paths (reads bytes, then decodes)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite_any_path(path: str | Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def load_dicom(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a DICOM as float32 (H, W). Returns only technical metadata (no PHI)."""
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # multi-frame or RGB - take first frame / convert to gray
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_RGB2GRAY)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr
    meta = {
        "source": "dicom",
        "rows": int(arr.shape[0]),
        "cols": int(arr.shape[1]),
        "modality": str(getattr(ds, "Modality", "")),
        "view_position": str(getattr(ds, "ViewPosition", "")),
        "body_part": str(getattr(ds, "BodyPartExamined", "")),
        "photometric": photometric,
        "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
    }
    return arr, meta


def read_dicom_header(path: str | Path) -> Dict[str, Any]:
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    return {"rows": int(ds.Rows), "cols": int(ds.Columns), "view_position": str(getattr(ds, "ViewPosition", ""))}


def load_raster(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load PNG/JPEG (8 or 16 bit, gray/RGB/RGBA) as float32 grayscale."""
    img = imread_any_path(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    color_spread = 0.0
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        color_spread = float(np.abs(img.astype(np.float32) - img.mean(axis=2, keepdims=True)).mean())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    meta = {"source": "raster", "rows": int(img.shape[0]), "cols": int(img.shape[1]),
            "dtype": str(img.dtype), "color_spread": color_spread}
    return img.astype(np.float32), meta


def load_image_any(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in DICOM_SUFFIXES:
        return load_dicom(path)
    return load_raster(path)


# --------------------------------------------------------------------------- intensity
def window_to_uint8(arr: np.ndarray, percentiles: Sequence[float] = (0.5, 99.5)) -> np.ndarray:
    """Robust percentile windowing to [0, 255] uint8."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, percentiles) if percentiles else (arr.min(), arr.max())
    if hi - lo < 1e-6:
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    return (out * 255.0 + 0.5).astype(np.uint8)


def apply_clahe(img_u8: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    return clahe.apply(img_u8)


# --------------------------------------------------------------------------- geometry
@dataclass
class ResizeGeometry:
    orig_h: int
    orig_w: int
    out_size: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_resize_geometry(orig_h: int, orig_w: int, size: int, mode: str = "stretch") -> ResizeGeometry:
    if mode == "stretch":
        return ResizeGeometry(orig_h, orig_w, size, size / orig_w, size / orig_h, 0, 0, mode)
    if mode == "letterbox":
        s = size / max(orig_h, orig_w)
        new_w, new_h = int(round(orig_w * s)), int(round(orig_h * s))
        return ResizeGeometry(orig_h, orig_w, size, s, s, (size - new_w) // 2, (size - new_h) // 2, mode)
    raise ValueError(f"Unknown resize mode {mode!r}")


def resize_image(img: np.ndarray, geom: ResizeGeometry) -> np.ndarray:
    size = geom.out_size
    if geom.mode == "stretch":
        interp = cv2.INTER_AREA if (geom.orig_h > size or geom.orig_w > size) else cv2.INTER_LINEAR
        return cv2.resize(img, (size, size), interpolation=interp)
    new_w = int(round(geom.orig_w * geom.scale_x))
    new_h = int(round(geom.orig_h * geom.scale_y))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=img.dtype)
    canvas[geom.pad_y:geom.pad_y + new_h, geom.pad_x:geom.pad_x + new_w] = resized
    return canvas


def transform_boxes(boxes_xyxy: np.ndarray, geom: ResizeGeometry) -> np.ndarray:
    """Map boxes (N,4) [x1,y1,x2,y2] in original pixels to resized pixels, clipped."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if len(boxes) == 0:
        return boxes
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * geom.scale_x + geom.pad_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * geom.scale_y + geom.pad_y
    boxes = np.clip(boxes, 0, geom.out_size)
    keep = (boxes[:, 2] - boxes[:, 0] > 0.5) & (boxes[:, 3] - boxes[:, 1] > 0.5)
    return boxes[keep]


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4).copy()
    b[:, 2] = b[:, 0] + b[:, 2]
    b[:, 3] = b[:, 1] + b[:, 3]
    return b


def boxes_to_mask(boxes_xyxy: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterise boxes into a binary (H, W) float32 mask (union of boxes).

    Pixel (r, c) is inside a box when its centre (c+0.5, r+0.5) lies in [x1, x2) x [y1, y2).
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4):
        c1 = int(np.clip(np.ceil(x1 - 0.5), 0, width))
        c2 = int(np.clip(np.ceil(x2 - 0.5), 0, width))
        r1 = int(np.clip(np.ceil(y1 - 0.5), 0, height))
        r2 = int(np.clip(np.ceil(y2 - 0.5), 0, height))
        if c2 > c1 and r2 > r1:
            mask[r1:r2, c1:c2] = 1.0
    return mask


# --------------------------------------------------------------------------- full pipeline
class Preprocessor:
    """Deterministic preprocessing configured from `data.preprocessing`."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.size = int(cfg.get("image_size", 384))
        self.mode = cfg.get("resize_mode", "stretch")
        self.percentiles = cfg.get("intensity_percentiles", (0.5, 99.5))
        self.clahe = bool(cfg.get("clahe", True))
        self.clip = float(cfg.get("clahe_clip_limit", 2.0))
        self.tile = int(cfg.get("clahe_tile_grid", 8))
        self.mean = np.asarray(cfg.get("normalize_mean", (0.485, 0.456, 0.406)), dtype=np.float32)
        self.std = np.asarray(cfg.get("normalize_std", (0.229, 0.224, 0.225)), dtype=np.float32)

    def signature(self) -> Dict[str, Any]:
        """Parameters that change cached pixels (used to validate caches)."""
        return {"image_size": self.size, "resize_mode": self.mode, "intensity_percentiles": list(self.percentiles),
                "clahe": self.clahe, "clahe_clip_limit": self.clip, "clahe_tile_grid": self.tile}

    def __call__(self, raw: np.ndarray) -> Tuple[np.ndarray, ResizeGeometry]:
        img = window_to_uint8(raw, self.percentiles)
        geom = compute_resize_geometry(img.shape[0], img.shape[1], self.size, self.mode)
        img = resize_image(img, geom)
        # CLAHE after resizing so the tile grid covers the same anatomy regardless of source size
        if self.clahe:
            img = apply_clahe(img, self.clip, self.tile)
        return img, geom

    def to_tensor(self, img_u8: np.ndarray) -> torch.Tensor:
        x = img_u8.astype(np.float32) / 255.0
        x = np.repeat(x[None], 3, axis=0)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(x.astype(np.float32))

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Inverse of to_tensor for display: returns (H, W) float in [0, 1]."""
        x = tensor.detach().cpu().float().numpy()
        x = x * self.std[:, None, None] + self.mean[:, None, None]
        return np.clip(x.mean(axis=0), 0, 1)
