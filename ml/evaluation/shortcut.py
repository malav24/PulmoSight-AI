"""Shortcut-learning stress tests.

Each perturbation removes or alters image content that should NOT carry pneumonia evidence;
a model that is "right for the right reason" should change its predictions little, while a model
relying on shortcuts (laterality markers, text, borders, devices, background) may change a lot.

IMPORTANT LIMITATIONS
* No lung-segmentation model ships with this project. `approximate_thorax_mask` is a fixed
  geometric ellipse (a crude proxy), NOT a lung segmentation. If you have real lung masks
  (e.g. from a validated segmentation model), pass them to `outside_lung_masking` instead and
  report which you used.
* `detect_marker_candidates` finds small saturated blobs near the image border (typical of burned-in
  'L'/'R' markers and text). It is a heuristic with false positives/negatives - "where identifiable".
* Masking itself changes the image distribution; a prediction change does not prove shortcut use.
  Compare models under identical perturbations and look at relative differences.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import cv2
import numpy as np
import torch

from .classification import safe_auc


def approximate_thorax_mask(h: int, w: int, cx: float = 0.5, cy: float = 0.52, rx: float = 0.40, ry: float = 0.42
                            ) -> np.ndarray:
    """Geometric ellipse covering the typical thorax position in a PA/AP CXR (proxy only)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return ((((xx + 0.5) / w - cx) / rx) ** 2 + (((yy + 0.5) / h - cy) / ry) ** 2 <= 1.0).astype(np.float32)


def border_mask(h: int, w: int, frac: float = 0.1) -> np.ndarray:
    """1 on the outer `frac` border, 0 inside."""
    m = np.ones((h, w), np.float32)
    bh, bw = int(round(h * frac)), int(round(w * frac))
    m[bh:h - bh, bw:w - bw] = 0
    return m


def detect_marker_candidates(gray_u8: np.ndarray, border_frac: float = 0.2, intensity_pct: float = 99.5,
                             max_area_frac: float = 0.005) -> np.ndarray:
    """Heuristic mask of small, very bright components in the border region."""
    h, w = gray_u8.shape
    thr = np.percentile(gray_u8, intensity_pct)
    cand = ((gray_u8 >= thr) & (border_mask(h, w, border_frac) > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros((h, w), np.float32)
    for i in range(1, n):
        if 4 <= stats[i, cv2.CC_STAT_AREA] <= max_area_frac * h * w:
            out[lab == i] = 1.0
    if out.any():
        out = cv2.dilate(out, np.ones((5, 5), np.uint8))
    return out


def _fill(images: torch.Tensor, region: torch.Tensor, fill: str) -> torch.Tensor:
    if fill == "mean":
        val = images.mean(dim=(2, 3), keepdim=True)
    else:  # "zero" in normalised space ~ ImageNet-mean gray
        val = torch.zeros_like(images[:, :, :1, :1])
    return images * (1 - region) + val * region


def outside_lung_masking(images: torch.Tensor, lung_masks: Optional[torch.Tensor] = None, fill: str = "mean"
                         ) -> torch.Tensor:
    B, _, H, W = images.shape
    if lung_masks is None:
        lung_masks = torch.from_numpy(approximate_thorax_mask(H, W))[None, None].expand(B, 1, H, W)
    return _fill(images, 1 - lung_masks.to(images), fill)


def border_masking(images: torch.Tensor, frac: float = 0.1, fill: str = "mean") -> torch.Tensor:
    B, _, H, W = images.shape
    region = torch.from_numpy(border_mask(H, W, frac))[None, None].to(images)
    return _fill(images, region, fill)


def marker_masking(images: torch.Tensor, gray_u8: Optional[np.ndarray] = None, fill: str = "mean"):
    """Mask heuristic marker candidates. Returns (images, fraction_of_images_with_candidates)."""
    B, _, H, W = images.shape
    if gray_u8 is None:
        x = images.mean(1).cpu().numpy()
        gray_u8 = np.stack([((xi - xi.min()) / (xi.max() - xi.min() + 1e-8) * 255).astype(np.uint8) for xi in x])
    regions = np.stack([detect_marker_candidates(g) for g in gray_u8])[:, None]
    found = float((regions.reshape(B, -1).sum(1) > 0).mean())
    return _fill(images, torch.from_numpy(regions).to(images), fill), found


def background_perturbation(images: torch.Tensor, lung_masks: Optional[torch.Tensor] = None, noise_std: float = 0.5,
                            seed: int = 0) -> torch.Tensor:
    """Add Gaussian noise only outside the (approximate) thorax."""
    B, _, H, W = images.shape
    if lung_masks is None:
        lung_masks = torch.from_numpy(approximate_thorax_mask(H, W))[None, None].expand(B, 1, H, W)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(images.shape, generator=g).to(images) * noise_std
    return images + noise * (1 - lung_masks.to(images))


STRESS_TESTS = ("outside_lung_masking", "border_masking", "marker_masking", "background_perturbation")


@torch.no_grad()
def run_stress_tests(predict_fn: Callable[[torch.Tensor], torch.Tensor], loader, device, threshold: float = 0.5,
                     max_batches: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Compare predictions on original vs perturbed images over a loader."""
    base_p, labels = [], []
    pert_p = {k: [] for k in STRESS_TESTS}
    marker_found = []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["image"].to(device)
        labels.append(batch["label"].numpy())
        base_p.append(predict_fn(x).float().cpu().numpy())
        pert_p["outside_lung_masking"].append(predict_fn(outside_lung_masking(x)).float().cpu().numpy())
        pert_p["border_masking"].append(predict_fn(border_masking(x)).float().cpu().numpy())
        xm, found = marker_masking(x)
        marker_found.append(found)
        pert_p["marker_masking"].append(predict_fn(xm).float().cpu().numpy())
        pert_p["background_perturbation"].append(predict_fn(background_perturbation(x, seed=i)).float().cpu().numpy())
    y = np.concatenate(labels)
    p0 = np.concatenate(base_p)
    report = {"original": {"auc": safe_auc(y, p0), "n": int(len(y))}}
    for k, v in pert_p.items():
        p = np.concatenate(v)
        report[k] = {"auc": safe_auc(y, p), "auc_drop": safe_auc(y, p0) - safe_auc(y, p),
                     "mean_abs_prob_change": float(np.mean(np.abs(p - p0))),
                     "decision_flip_rate": float(np.mean((p >= threshold) != (p0 >= threshold)))}
    report["marker_masking"]["fraction_images_with_marker_candidates"] = float(np.mean(marker_found)) if marker_found else 0.0
    report["_notes"] = ("outside_lung/background tests use a geometric thorax ellipse, not a lung segmentation; "
                        "marker detection is heuristic.")
    return report
