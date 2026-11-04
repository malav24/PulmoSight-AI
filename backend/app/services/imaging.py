"""Turn model maps into display layers for the web UI.

Heatmaps are rendered as semi-transparent RGBA layers with a smooth alpha ramp (no contours or hard
edges), so they read as graded model output, not as disease boundaries. Perceptually uniform colour
maps are used: viridis for localization, inferno for Grad-CAM++, cividis for attention.
"""
from __future__ import annotations

import base64
from typing import Dict, List, Optional

import cv2
import numpy as np

COLORMAPS = {"localization": ("viridis", cv2.COLORMAP_VIRIDIS),
             "gradcam": ("inferno", cv2.COLORMAP_INFERNO),
             "attention": ("cividis", getattr(cv2, "COLORMAP_CIVIDIS", cv2.COLORMAP_VIRIDIS))}


def png_b64(img: np.ndarray) -> str:
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    elif img.ndim == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def heatmap_layer(hm: np.ndarray, kind: str, normalize: bool = True, max_alpha: float = 0.75) -> np.ndarray:
    """RGBA layer: colour from the colormap, alpha rising smoothly with intensity."""
    hm = np.asarray(hm, dtype=np.float32)
    if normalize:
        lo, hi = float(hm.min()), float(hm.max())
        hm = (hm - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(hm)
    hm = np.clip(hm, 0, 1)
    hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=max(1.0, hm.shape[0] / 192))
    color = cv2.cvtColor(cv2.applyColorMap((hm * 255).astype(np.uint8), COLORMAPS[kind][1]), cv2.COLOR_BGR2RGB)
    alpha = (np.clip((hm - 0.1) / 0.9, 0, 1) ** 1.2 * max_alpha * 255).astype(np.uint8)
    return np.dstack([color, alpha])


def region_summary(hm: np.ndarray, rows: int = 3, cols: int = 2) -> List[Dict]:
    """Share of map mass in a 3×2 grid of image zones (image coordinates)."""
    hm = np.clip(np.asarray(hm, dtype=np.float64), 0, None)
    total = hm.sum()
    h, w = hm.shape
    names_r = ["upper", "middle", "lower"]
    names_c = ["image-left", "image-right"]
    out = []
    for r in range(rows):
        for c in range(cols):
            part = hm[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols].sum()
            out.append({"zone": f"{names_r[r]} {names_c[c]}", "share": float(part / total) if total > 0 else 0.0})
    return sorted(out, key=lambda z: -z["share"])


def peak_location(hm: np.ndarray) -> Dict[str, float]:
    r, c = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return {"x": float(c / hm.shape[1]), "y": float(r / hm.shape[0])}
