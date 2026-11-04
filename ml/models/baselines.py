"""Classification-only baselines built with timm, returning the same output dict as the hybrid
(localization keys are absent; use Grad-CAM++ for their explanations)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

BASELINE_SPECS: Dict[str, Dict[str, Any]] = {
    "resnet50": {"timm_name": "resnet50", "gradcam_target": "net.layer4"},
    "densenet121": {"timm_name": "densenet121", "gradcam_target": "net.features"},
    "efficientnet_b0": {"timm_name": "efficientnet_b0", "gradcam_target": "net.conv_head"},
    "vit_small": {"timm_name": "vit_small_patch16_384", "gradcam_target": "net.blocks.-1.norm1"},
}


class BaselineClassifier(nn.Module):
    family = "baseline"
    has_localization = False

    def __init__(self, name: str, pretrained_backbone: bool = True, image_size: int = 384, head_dropout: float = 0.2,
                 **unused: Any):
        super().__init__()
        import timm

        if name not in BASELINE_SPECS:
            raise KeyError(f"Unknown baseline {name!r}; options {sorted(BASELINE_SPECS)}")
        spec = BASELINE_SPECS[name]
        self.name = name
        # timm's own head dropout is a *functional* call in ResNet/EfficientNet, which MC-dropout cannot
        # re-enable at inference. Use an explicit nn.Dropout module instead (identical in training).
        kwargs: Dict[str, Any] = {"num_classes": 1, "drop_rate": 0.0}
        if name == "vit_small":
            kwargs["img_size"] = image_size
        self.net = timm.create_model(spec["timm_name"], pretrained=pretrained_backbone, **kwargs)
        self.head_drop = nn.Dropout(head_dropout)
        self.gradcam_target = spec["gradcam_target"]
        self.is_vit = name == "vit_small"

    def backbone_parameters(self) -> List[nn.Parameter]:
        head = self.net.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
        return [p for p in self.net.parameters() if id(p) not in head_ids]

    def gradcam_reshape(self) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """ViT tokens (B, 1+N, C) -> (B, C, h, w); None for CNNs."""
        if not self.is_vit:
            return None
        n_prefix = getattr(self.net, "num_prefix_tokens", 1)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            t = t[:, n_prefix:, :]
            side = int(round(t.shape[1] ** 0.5))
            return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], side, side)

        return reshape

    def forward(self, x: torch.Tensor, **unused: Any) -> Dict[str, torch.Tensor]:
        feats = self.net.forward_features(x)
        embedding = self.net.forward_head(feats, pre_logits=True)
        logits = self.net.get_classifier()(self.head_drop(embedding)).squeeze(-1)
        return {"logits": logits, "probs": torch.sigmoid(logits), "embedding": embedding}
