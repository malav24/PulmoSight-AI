"""Overlays and comparison panels for heatmaps, attention, Grad-CAM++ and radiologist boxes."""
from __future__ import annotations

import base64
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

# Legend text used everywhere so the three map types are never conflated.
MAP_DESCRIPTIONS = {
    "localization": "Localization heatmap - supervised prediction of opacity location (hybrid models only)",
    "attention": "Spatial attention - pooling weights the hybrid classifier uses; not a validated explanation",
    "gradcam": "Grad-CAM++ - post-hoc gradient explanation of the classification score",
}


def normalize_heatmap(hm: np.ndarray, mode: str = "minmax", eps: float = 1e-8) -> np.ndarray:
    hm = np.asarray(hm, dtype=np.float32)
    if mode == "none":
        return np.clip(hm, 0, 1)
    if mode == "percentile":
        lo, hi = np.percentile(hm, [1, 99])
    else:
        lo, hi = float(hm.min()), float(hm.max())
    if hi - lo < eps:
        return np.zeros_like(hm)
    return np.clip((hm - lo) / (hi - lo), 0, 1)


def to_uint8_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET,
                    normalize: str = "minmax") -> np.ndarray:
    """image: (H,W) gray [0,1] or uint8; heatmap (h,w). Returns RGB uint8."""
    gray = to_uint8_gray(image)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    hm = normalize_heatmap(heatmap, normalize)
    if hm.shape != gray.shape:
        hm = cv2.resize(hm, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = cv2.cvtColor(cv2.applyColorMap((hm * 255).astype(np.uint8), colormap), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1 - alpha, color, alpha, 0)


def draw_boxes(rgb: np.ndarray, boxes: Optional[np.ndarray], color=(0, 255, 0), thickness: Optional[int] = None,
               label: Optional[str] = "GT") -> np.ndarray:
    """Draw radiologist boxes; line width and font scale with image size."""
    out = rgb.copy()
    if boxes is None:
        return out
    side = min(out.shape[:2])
    thickness = thickness or max(1, int(round(side / 256)))
    font = side / 900.0
    for b in np.asarray(boxes).reshape(-1, 4):
        if (b < 0).all():
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if label and side >= 128:
            cv2.putText(out, label, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, font, color, 1, cv2.LINE_AA)
    return out


def comparison_panel(image: np.ndarray, maps: Dict[str, np.ndarray], boxes: Optional[np.ndarray] = None,
                     title: Optional[str] = None, path: Optional[str] = None, probability: Optional[float] = None):
    """Original | each map overlay (with radiologist boxes). Returns the matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 1 + len(maps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    axes = np.atleast_1d(axes)
    base = cv2.cvtColor(to_uint8_gray(image), cv2.COLOR_GRAY2RGB)
    axes[0].imshow(draw_boxes(base, boxes))
    axes[0].set_title("Original" + (" + boxes" if boxes is not None else ""), fontsize=9)
    for ax, (name, hm) in zip(axes[1:], maps.items()):
        ax.imshow(draw_boxes(overlay_heatmap(image, hm), boxes))
        ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.axis("off")
    sup = title or ""
    if probability is not None:
        sup += f"  p(pneumonia-related opacity)={probability:.3f}"
    if sup:
        fig.suptitle(sup, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def encode_png_base64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
