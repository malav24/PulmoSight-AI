"""Faithfulness and plausibility of saliency maps.

* Deletion AUC  (lower is better): progressively replace the most-salient pixels with a baseline
  and track p(positive). A faithful map makes the score drop fast.
* Insertion AUC (higher is better): start from the baseline image and progressively restore the
  most-salient pixels.
* Attention alignment: saliency mass inside the radiologist box and pointing game (plausibility).

Limitations
-----------
* Deletion/insertion create out-of-distribution images (blurred/mean patches); score changes may
  reflect the distribution shift as much as the evidence removed. Report the baseline used.
* Results depend on step count, baseline type and map resolution; keep them fixed across models.
* Plausibility (agreement with boxes) is not faithfulness: a map can match the box yet not drive
  the prediction, and vice versa.
* Only meaningful for images the model scores as positive (deleting evidence from a negative is ill-defined);
  by default only RSNA positives are evaluated.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch

from .localization import mass_in_box, pointing_game_hit


def _baseline(images: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros_like(images)
    if kind == "mean":
        return images.mean(dim=(2, 3), keepdim=True).expand_as(images).clone()
    if kind == "blur":
        arr = images.detach().cpu().numpy()
        out = np.stack([np.stack([cv2.GaussianBlur(ch, (0, 0), sigmaX=max(2.0, ch.shape[0] / 24)) for ch in im])
                        for im in arr])
        return torch.from_numpy(out).to(images.device, images.dtype)
    raise ValueError(f"unknown baseline {kind}")


@torch.no_grad()
def deletion_insertion_curves(predict_fn: Callable[[torch.Tensor], torch.Tensor], images: torch.Tensor,
                              saliency: np.ndarray, steps: int = 20, baseline: str = "blur"
                              ) -> Dict[str, np.ndarray]:
    """predict_fn maps (B,3,H,W) -> probabilities (B,). saliency: (B,H,W). Returns curves (B, steps+1)."""
    B, _, H, W = images.shape
    base = _baseline(images, baseline)
    order = np.argsort(-saliency.reshape(B, -1), axis=1)  # most salient first
    n_pix = H * W
    fracs = np.linspace(0, 1, steps + 1)
    del_scores = np.zeros((B, steps + 1), np.float32)
    ins_scores = np.zeros((B, steps + 1), np.float32)
    for s, f in enumerate(fracs):
        k = int(round(f * n_pix))
        m = np.zeros((B, n_pix), np.float32)
        if k > 0:
            np.put_along_axis(m, order[:, :k], 1.0, axis=1)
        mask = torch.from_numpy(m.reshape(B, 1, H, W)).to(images.device, images.dtype)
        deleted = images * (1 - mask) + base * mask
        inserted = base * (1 - mask) + images * mask
        del_scores[:, s] = predict_fn(deleted).float().cpu().numpy()
        ins_scores[:, s] = predict_fn(inserted).float().cpu().numpy()
    return {"fractions": fracs, "deletion": del_scores, "insertion": ins_scores}


def curve_auc(fracs: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.trapezoid(scores, fracs, axis=-1) if hasattr(np, "trapezoid") else np.trapz(scores, fracs, axis=-1)


def explanation_metrics(predict_fn, images: torch.Tensor, saliency: np.ndarray, gt_masks: Optional[np.ndarray] = None,
                        steps: int = 20, baseline: str = "blur") -> Dict[str, float]:
    curves = deletion_insertion_curves(predict_fn, images, saliency, steps, baseline)
    del_auc = curve_auc(curves["fractions"], curves["deletion"])
    ins_auc = curve_auc(curves["fractions"], curves["insertion"])
    out = {"deletion_auc": float(np.mean(del_auc)), "insertion_auc": float(np.mean(ins_auc)),
           "n": int(images.shape[0]), "steps": steps, "baseline": baseline}
    if gt_masks is not None:
        mass = [mass_in_box(s, g) for s, g in zip(saliency, gt_masks) if g.sum() > 0]
        hits = [pointing_game_hit(s, g) for s, g in zip(saliency, gt_masks) if g.sum() > 0]
        out["mass_in_box"] = float(np.nanmean(mass)) if mass else float("nan")
        out["pointing_game"] = float(np.mean(hits)) if hits else float("nan")
    return out


def attention_alignment(attention: torch.Tensor, masks: torch.Tensor) -> Dict[str, float]:
    """Alignment of the hybrid model's normalised attention (B,1,h,w) with box masks (B,1,H,W)."""
    from ml.training.losses import attention_kl, attention_outside_mass, mask_to_grid

    cov = mask_to_grid(masks, attention.shape[-2:])
    keep = masks.flatten(1).sum(1) > 0
    if not keep.any():
        return {"n": 0}
    a, c = attention[keep], cov[keep]
    outside = attention_outside_mass(a, c)
    up = torch.nn.functional.interpolate(a.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)
    hits = [pointing_game_hit(u[0].cpu().numpy(), m[0].cpu().numpy()) for u, m in zip(up, masks[keep])]
    return {"n": int(keep.sum()), "attention_mass_in_box": float((1 - outside).mean()),
            "attention_kl_to_box": float(attention_kl(a, c).mean()), "attention_pointing_game": float(np.mean(hits))}
