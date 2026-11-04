"""Hybrid EfficientNet-B0 + Transformer with classification and localization heads.

    image (B,3,H,W)
      -> EfficientNet-B0 features: stride-16 (112ch) and stride-32 (320ch)
      -> 1x1 projection of stride-32 map to d_model  [+ optional avg-pool `token_pool`]
         (spatial reduction: 384px -> 12x12 = 144 tokens, keeps self-attention cheap)
      -> tokenisation + learned 2-D positional embedding (interpolated if the grid changes)
      -> 2-layer pre-norm Transformer encoder  => shared token representation
      Head 1 (classification): spatial attention pooling over tokens -> MLP -> logit
             The softmax pooling weights form the *spatial attention representation* A (sums to 1).
             A is part of the classification computation: the logit depends only on A-weighted
             features, so guiding A changes which regions the classifier can draw evidence from.
      Head 2 (localization): tokens reshaped to a grid, upsampled, fused with stride-16 skip
             features, conv decoder -> per-pixel heatmap logits (upsampled to input size).

Raw multi-head self-attention weights are optionally returned for diagnostics only; they are
NOT treated as explanations (they describe token mixing, not evidence for the prediction).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer that can return head-averaged attention weights."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(dim_feedforward, d_model))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=True)
        x = x + self.drop1(a)
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x, w


def _conv_bn_act(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class HybridEfficientNetTransformer(nn.Module):
    family = "hybrid"
    has_localization = True
    gradcam_target = "proj"  # stride-32 projected CNN feature map, before the Transformer

    def __init__(self, backbone: str = "efficientnet_b0", pretrained_backbone: bool = True, image_size: int = 384,
                 d_model: int = 256, num_layers: int = 2, num_heads: int = 8, dim_feedforward: int = 512,
                 dropout: float = 0.1, token_pool: int = 1, attn_pool_hidden: int = 128, head_dropout: float = 0.2,
                 use_skip: bool = True, **unused: Any):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained_backbone, features_only=True,
                                          out_indices=(3, 4))
        c16, c32 = self.backbone.feature_info.channels()
        self.d_model = d_model
        self.use_skip = use_skip

        self.proj = _conv_bn_act(c32, d_model, k=1)
        self.token_pool = nn.AvgPool2d(token_pool) if token_pool > 1 else nn.Identity()
        grid = max(1, image_size // 32 // max(1, token_pool))
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, grid, grid))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, dim_feedforward, dropout)
                                      for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        # Head 1: attention pooling + classifier
        self.attn_score = nn.Sequential(nn.Linear(d_model, attn_pool_hidden), nn.Tanh(),
                                        nn.Linear(attn_pool_hidden, 1))
        self.cls_head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(head_dropout), nn.Linear(d_model, 1))

        # Head 2: localization decoder
        self.skip = nn.Conv2d(c16, d_model, 1) if use_skip else None
        self.loc_decoder = nn.Sequential(_conv_bn_act(d_model, d_model // 2), nn.Dropout2d(head_dropout / 2),
                                         _conv_bn_act(d_model // 2, d_model // 4), nn.Conv2d(d_model // 4, 1, 1))

    def backbone_parameters(self) -> List[nn.Parameter]:
        return list(self.backbone.parameters())

    def forward(self, x: torch.Tensor, return_raw_attention: bool = False) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        f16, f32 = self.backbone(x)
        z = self.token_pool(self.proj(f32))
        _, C, h, w = z.shape
        pos = self.pos_embed
        if pos.shape[-2:] != (h, w):
            pos = F.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        tokens = (z + pos).flatten(2).transpose(1, 2)  # (B, N, C)

        raw_attn = []
        for layer in self.encoder:
            tokens, wts = layer(tokens, need_weights=return_raw_attention)
            if return_raw_attention:
                raw_attn.append(wts)
        tokens = self.norm(tokens)

        scores = self.attn_score(tokens).squeeze(-1)                # (B, N)
        attention = torch.softmax(scores.float(), dim=1).to(tokens.dtype)
        pooled = torch.bmm(attention.unsqueeze(1), tokens).squeeze(1)  # (B, C)
        logits = self.cls_head(pooled).squeeze(-1)

        shared = tokens.transpose(1, 2).reshape(B, C, h, w)
        target_hw = f16.shape[-2:] if self.use_skip else (h, w)
        dec = F.interpolate(shared, size=target_hw, mode="bilinear", align_corners=False)
        if self.skip is not None:
            dec = dec + self.skip(f16)
        loc_lr = self.loc_decoder(dec)
        loc_logits = F.interpolate(loc_lr, size=(H, W), mode="bilinear", align_corners=False)

        attn_grid = attention.view(B, 1, h, w)
        attn_up = F.interpolate(attn_grid.float(), size=(H, W), mode="bilinear", align_corners=False)
        a_min = attn_up.amin(dim=(2, 3), keepdim=True)
        a_max = attn_up.amax(dim=(2, 3), keepdim=True)
        out = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "loc_logits": loc_logits,
            "heatmap": torch.sigmoid(loc_logits),
            "attention": attn_grid,                                   # (B,1,h,w), sums to 1 per image
            "attention_map": (attn_up - a_min) / (a_max - a_min + 1e-8),  # display only, [0,1]
            "features": shared,
            "embedding": pooled,                                      # used for OOD feature statistics
        }
        if return_raw_attention:
            out["raw_self_attention"] = torch.stack(raw_attn, dim=1)  # (B, L, N, N) diagnostics only
        return out
