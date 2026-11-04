"""Explanation-guided objective.

    L_total = L_classification + alpha * L_localization + lambda * L_attention

* L_classification : weighted BCE (pos_weight) or binary focal loss.
* L_localization   : Dice and/or BCE between the heatmap logits and the rasterised RSNA box mask.
                     BCE may include negatives (empty mask); Dice is computed on positives only
                     (Dice of an empty target is undefined/unstable).
* L_attention      : aligns the normalised spatial attention A (sums to 1 over the token grid)
                     with the annotation mask downsampled to that grid (area coverage c in [0,1]):
                       outside_mass : sum_i A_i * (1 - c_i)   -> attention mass outside the boxes.
                                      Does not force attention to spread over the whole box,
                                      which suits coarse rectangular annotations.
                       kl           : KL(T || A) with T = c / sum(c)  -> match the box distribution.
                     Computed only on positive images that carry boxes; negatives have no
                     defined "right reason" region and receive no attention supervision.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
                           ) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Mean over images of 1 - soft Dice. logits/targets: (B,1,H,W)."""
    probs = torch.sigmoid(logits).flatten(1)
    t = targets.flatten(1)
    inter = (probs * t).sum(1)
    denom = probs.sum(1) + t.sum(1)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def mask_to_grid(mask: torch.Tensor, size) -> torch.Tensor:
    """Area-downsample a (B,1,H,W) binary mask to per-cell coverage in [0,1]."""
    return F.adaptive_avg_pool2d(mask.float(), size)


def attention_outside_mass(attention: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    """Per-image attention mass falling outside the annotation. attention sums to 1. -> (B,)"""
    return (attention.float() * (1.0 - coverage)).flatten(1).sum(1)


def attention_kl(attention: torch.Tensor, coverage: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image KL(T || A) with T the normalised coverage. -> (B,)"""
    a = attention.float().flatten(1).clamp_min(eps)
    t = coverage.flatten(1)
    t = t / t.sum(1, keepdim=True).clamp_min(eps)
    return (t * (torch.log(t.clamp_min(eps)) - torch.log(a))).sum(1)


class ExplanationGuidedLoss(nn.Module):
    def __init__(self, cfg: Dict[str, Any], pos_weight: Optional[float] = None):
        super().__init__()
        self.cls_type = cfg.get("classification", "weighted_bce")
        self.focal_alpha = float(cfg.get("focal_alpha", 0.25))
        self.focal_gamma = float(cfg.get("focal_gamma", 2.0))
        self.alpha = float(cfg.get("alpha", 0.0))
        self.lam = float(cfg.get("lambda", 0.0))
        self.loc_type = cfg.get("localization", "dice_bce")
        self.loc_on_negatives = bool(cfg.get("loc_on_negatives", True))
        self.attn_type = cfg.get("attention", "outside_mass")
        self.kl_weight = float(cfg.get("attention_kl_weight", 1.0))
        pw = torch.tensor(float(pos_weight)) if (pos_weight is not None and self.cls_type == "weighted_bce") else None
        self.register_buffer("pos_weight", pw if pw is not None else torch.tensor(1.0))
        self.use_pos_weight = pw is not None
        valid_attn = {"outside_mass", "kl", "outside_mass+kl"}
        if self.attn_type not in valid_attn:
            raise ValueError(f"loss.attention must be one of {valid_attn}")
        if self.loc_type not in {"dice", "bce", "dice_bce"}:
            raise ValueError("loss.localization must be dice | bce | dice_bce")

    # ---------------------------------------------------------------- components
    def classification(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits, labels = logits.float(), labels.float()
        if self.cls_type == "focal":
            return focal_loss_with_logits(logits, labels, self.focal_alpha, self.focal_gamma)
        pw = self.pos_weight if self.use_pos_weight else None
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)

    def localization(self, loc_logits: torch.Tensor, masks: torch.Tensor, has_boxes: torch.Tensor) -> torch.Tensor:
        loc_logits, masks = loc_logits.float(), masks.float()
        total = loc_logits.new_zeros(())
        pos = has_boxes.bool()
        if "bce" in self.loc_type:
            sel = torch.ones_like(pos) if self.loc_on_negatives else pos
            if sel.any():
                total = total + F.binary_cross_entropy_with_logits(loc_logits[sel], masks[sel])
        if "dice" in self.loc_type and pos.any():
            total = total + soft_dice_loss(loc_logits[pos], masks[pos])
        return total

    def attention(self, attention: torch.Tensor, masks: torch.Tensor, has_boxes: torch.Tensor) -> torch.Tensor:
        pos = has_boxes.bool()
        if not pos.any():
            return attention.new_zeros((), dtype=torch.float32) + 0.0 * attention.sum()
        a = attention[pos]
        cov = mask_to_grid(masks[pos], a.shape[-2:])
        loss = a.new_zeros(pos.sum().item(), dtype=torch.float32)
        if "outside_mass" in self.attn_type:
            loss = loss + attention_outside_mass(a, cov)
        if "kl" in self.attn_type:
            loss = loss + self.kl_weight * attention_kl(a, cov)
        return loss.mean()

    # ---------------------------------------------------------------- total
    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = batch["label"]
        l_cls = self.classification(outputs["logits"], labels)
        zero = l_cls.new_zeros(())
        l_loc, l_att = zero, zero
        if self.alpha > 0:
            if "loc_logits" not in outputs:
                raise ValueError("alpha > 0 but the model has no localization head")
            l_loc = self.localization(outputs["loc_logits"], batch["mask"], batch["has_boxes"])
        if self.lam > 0:
            if "attention" not in outputs:
                raise ValueError("lambda > 0 but the model exposes no spatial attention")
            l_att = self.attention(outputs["attention"], batch["mask"], batch["has_boxes"])
        total = l_cls + self.alpha * l_loc + self.lam * l_att
        return {"loss": total, "loss_cls": l_cls.detach(), "loss_loc": l_loc.detach(), "loss_attn": l_att.detach()}
