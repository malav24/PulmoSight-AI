"""Basic input validation and image-quality heuristics.

These checks catch obviously unsuitable inputs (wrong file, colour photo, blank, tiny, blurry,
non-CXR DICOM tags). They are simple heuristics with fixed thresholds and are NOT a validated
medical image-quality assessment; passing them does not mean the image is diagnostic quality.
"""
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

ALLOWED_SUFFIXES = {".dcm", ".dicom", ".png", ".jpg", ".jpeg"}
MIN_SIDE = 256


def check_image_quality(raw: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    """raw: float32 (H, W) as loaded. Returns {'passed', 'errors', 'warnings', 'measurements'}."""
    errors: List[str] = []
    warnings: List[str] = []
    h, w = raw.shape[:2]
    m: Dict[str, Any] = {"height": int(h), "width": int(w)}

    if min(h, w) < MIN_SIDE:
        errors.append(f"Image too small ({w}x{h}); at least {MIN_SIDE}px per side expected.")
    aspect = w / max(h, 1)
    m["aspect_ratio"] = round(float(aspect), 3)
    if not 0.6 <= aspect <= 1.6:
        warnings.append(f"Unusual aspect ratio {aspect:.2f} for a chest radiograph.")

    if meta.get("color_spread", 0) > 8:
        errors.append("Image appears to be colour, not a grayscale radiograph.")

    lo, hi = np.percentile(raw, [1, 99])
    dyn = float(hi - lo)
    m["dynamic_range_p1_p99"] = dyn
    if dyn < 1e-6:
        errors.append("Image is blank or constant.")
    u8 = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean, std = float(u8.mean()), float(u8.std())
    m["mean_intensity"], m["std_intensity"] = round(mean, 1), round(std, 1)
    if std < 15:
        warnings.append("Very low contrast.")
    if mean < 20 or mean > 235:
        warnings.append("Image is extremely dark or bright (possible exposure/windowing problem).")
    small = cv2.resize(u8, (512, int(512 * h / max(w, 1))), interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
    m["laplacian_variance"] = round(lap_var, 1)
    if lap_var < 20:
        warnings.append("Image may be blurred or heavily downsampled.")

    if meta.get("source") == "dicom":
        modality = meta.get("modality", "")
        if modality and modality not in {"CR", "DX", "DR"}:
            warnings.append(f"DICOM modality {modality!r} is not a radiograph (CR/DX expected).")
        view = meta.get("view_position", "")
        if view and view not in {"PA", "AP"}:
            warnings.append(f"View position {view!r}: models were trained on frontal (PA/AP) views only.")
        body = meta.get("body_part", "").upper()
        if body and "CHEST" not in body and body not in {"THORAX", "LUNG"}:
            warnings.append(f"BodyPartExamined={body!r} - may not be a chest image.")

    return {"passed": not errors, "errors": errors, "warnings": warnings, "measurements": m,
            "disclaimer": "Heuristic checks only; not a validated image-quality assessment."}
