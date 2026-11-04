"""Localization metrics comparing a heatmap / saliency map with RSNA box masks.

Limitations
-----------
* RSNA boxes are coarse rectangles around opacities; a perfectly anatomical prediction can
  still have modest IoU/Dice against a rectangle. Treat absolute values with care and compare
  models under identical settings.
* IoU/Dice need a binarisation threshold for the heatmap; results depend on it (report it).
  Saliency maps (Grad-CAM++, attention) are not calibrated probabilities, so a fixed threshold is
  arbitrary for them; prefer threshold-free measures (pointing game, mass-in-box) for those.
* The pointing game only checks the single maximum - it ignores spread and multiple lesions,
  and a large box makes it easy. It is defined only for images with boxes.
* Only images that carry annotations (RSNA positives; a handful of NIH images) are evaluated.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def binarize(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(heatmap) >= threshold).astype(np.float32)


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    p, g = pred_mask > 0.5, gt_mask > 0.5
    union = np.logical_or(p, g).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(p, g).sum() / union)


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    p, g = pred_mask > 0.5, gt_mask > 0.5
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")
    return float(2 * np.logical_and(p, g).sum() / denom)


def pointing_game_hit(saliency: np.ndarray, gt_mask: np.ndarray, tolerance: int = 0) -> Optional[bool]:
    """True if the saliency maximum falls inside the GT mask (dilated by `tolerance` px)."""
    if gt_mask.sum() == 0:
        return None
    sal = np.asarray(saliency, dtype=np.float32)
    r, c = np.unravel_index(int(np.argmax(sal)), sal.shape)
    if tolerance > 0:
        import cv2
        k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
        gt_mask = cv2.dilate(gt_mask.astype(np.uint8), k)
    return bool(gt_mask[r, c] > 0)


def mass_in_box(saliency: np.ndarray, gt_mask: np.ndarray) -> float:
    """Fraction of (non-negative) saliency energy inside the annotation (energy-based pointing game)."""
    s = np.clip(np.asarray(saliency, dtype=np.float64), 0, None)
    total = s.sum()
    if total <= 0 or gt_mask.sum() == 0:
        return float("nan")
    return float((s * (gt_mask > 0.5)).sum() / total)


def heatmap_to_boxes(heatmap: np.ndarray, threshold: float = 0.5, min_area: int = 16) -> np.ndarray:
    """Connected components of the thresholded heatmap -> (N,4) xyxy boxes (for reporting)."""
    import cv2

    binm = (np.asarray(heatmap) >= threshold).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    boxes = [[x, y, x + w, y + h] for (x, y, w, h, a) in stats[1:] if a >= min_area]
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def localization_metrics(heatmaps: List[np.ndarray], gt_masks: List[np.ndarray], threshold: float = 0.5,
                         tolerance: int = 0) -> Dict[str, float]:
    """Aggregate over images that have annotations."""
    ious, dices, hits, mass = [], [], [], []
    for hm, gt in zip(heatmaps, gt_masks):
        if gt.sum() == 0:
            continue
        pm = binarize(hm, threshold)
        ious.append(iou(pm, gt))
        dices.append(dice(pm, gt))
        hits.append(pointing_game_hit(hm, gt, tolerance))
        mass.append(mass_in_box(hm, gt))

    def m(x):
        x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(x)) if x else float("nan")

    return {"n_annotated": len(ious), "threshold": threshold, "iou": m(ious), "dice": m(dices),
            "pointing_game": m([float(h) for h in hits if h is not None]), "mass_in_box": m(mass)}
