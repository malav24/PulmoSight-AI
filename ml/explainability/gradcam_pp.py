"""Grad-CAM++ (Chattopadhay et al., 2018) for any model in this project.

Grad-CAM++ is a *post-hoc gradient-based explanation* of the classification logit. It is distinct
from (a) the hybrid model's spatial attention pooling weights (an internal quantity that the
classifier actually uses) and (b) the localization heatmap (a supervised prediction of where the
annotated opacity is). The three can disagree; none is guaranteed to be a faithful explanation,
which is why deletion/insertion tests are provided in ml.evaluation.explanation.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_module(model: nn.Module, dotted: str) -> nn.Module:
    """Resolve 'net.blocks.-1.norm1' style paths (negative indices allowed)."""
    mod = model
    for part in dotted.split("."):
        if part.lstrip("-").isdigit():
            mod = mod[int(part)]
        else:
            mod = getattr(mod, part)
    return mod


class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None,
                 reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.model = model
        target_layer = target_layer or getattr(model, "gradcam_target")
        self.layer = get_module(model, target_layer)
        if reshape_transform is None and hasattr(model, "gradcam_reshape"):
            reshape_transform = model.gradcam_reshape()
        self.reshape = reshape_transform
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

    def _fwd_hook(self, module, inp, out):
        out = out[0] if isinstance(out, (tuple, list)) else out
        self._acts = out
        out.register_hook(lambda g: setattr(self, "_grads", g))

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Return (B, H, W) maps in [0, 1] for the positive-class logit."""
        handle = self.layer.register_forward_hook(self._fwd_hook)
        was_training = self.model.training
        self.model.eval()
        try:
            x = x.clone().requires_grad_(True)
            with torch.enable_grad():
                out = self.model(x)
                logits = out["logits"] if isinstance(out, dict) else out
                self.model.zero_grad(set_to_none=True)
                logits.sum().backward()
            acts, grads = self._acts, self._grads
            if acts is None or grads is None:
                raise RuntimeError("Grad-CAM++ hooks captured nothing; check the target layer")
            if self.reshape is not None:
                acts, grads = self.reshape(acts), self.reshape(grads)
            acts, grads = acts.detach().float(), grads.detach().float()
            g2, g3 = grads.pow(2), grads.pow(3)
            sum_a = acts.sum(dim=(2, 3), keepdim=True)
            denom = 2 * g2 + sum_a * g3
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha = g2 / denom
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return normalize_maps(cam[:, 0].cpu().numpy())
        finally:
            handle.remove()
            self.model.train(was_training)
            self._acts = self._grads = None


def normalize_maps(maps: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-map min-max normalisation to [0,1]; constant maps become zeros."""
    maps = np.asarray(maps, dtype=np.float32)
    out = np.zeros_like(maps)
    for i in range(maps.shape[0]):
        lo, hi = maps[i].min(), maps[i].max()
        if hi - lo > eps:
            out[i] = (maps[i] - lo) / (hi - lo)
    return out
