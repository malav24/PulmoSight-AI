"""Single entry point for building any model in the study."""
from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .baselines import BASELINE_SPECS, BaselineClassifier
from .hybrid import HybridEfficientNetTransformer

MODEL_NAMES = ["hybrid", *BASELINE_SPECS.keys()]


def build_model(model_cfg: Dict[str, Any], image_size: int = 384, pretrained_backbone: bool | None = None) -> nn.Module:
    """Build from the `model:` config block.

    `pretrained_backbone` overrides the config (inference passes False: all weights must come from
    a trained checkpoint). If ImageNet weights are requested but cannot be downloaded, timm raises -
    there is no silent fallback to random initialisation.
    """
    cfg = dict(model_cfg)
    name = cfg.pop("name", "hybrid")
    pretrained = cfg.pop("pretrained_backbone", True) if pretrained_backbone is None else pretrained_backbone
    cfg.pop("pretrained_backbone", None)
    if name == "hybrid":
        return HybridEfficientNetTransformer(pretrained_backbone=pretrained, image_size=image_size, **cfg)
    if name in BASELINE_SPECS:
        return BaselineClassifier(name, pretrained_backbone=pretrained, image_size=image_size, **cfg)
    raise KeyError(f"Unknown model name {name!r}; options {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
