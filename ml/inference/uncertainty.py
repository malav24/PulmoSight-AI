"""Prediction uncertainty (MC dropout), abstention, and feature-space OOD warning.

MC dropout gives an approximate, model-specific spread of predictions; it is not a calibrated
Bayesian posterior and can be over-confident on far-OOD inputs. The Mahalanobis OOD score flags
inputs whose embedding is unusual relative to RSNA training images; it cannot certify that an
input is in-distribution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode everywhere except Dropout layers (BatchNorm stays frozen)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) or m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 20, temperature: float = 1.0
                       ) -> Dict[str, np.ndarray]:
    was_training = model.training
    enable_mc_dropout(model)
    try:
        logits = torch.stack([model(x)["logits"].float() for _ in range(n_samples)])  # (S, B)
    finally:
        model.train(was_training)
        if not was_training:
            model.eval()
    probs = torch.sigmoid(logits / temperature)
    mean = probs.mean(0)
    eps = 1e-7
    entropy = -(mean * torch.log(mean + eps) + (1 - mean) * torch.log(1 - mean + eps))
    exp_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).mean(0)
    return {"mean": mean.cpu().numpy(), "std": probs.std(0).cpu().numpy(),
            "p05": probs.quantile(0.05, dim=0).cpu().numpy(), "p95": probs.quantile(0.95, dim=0).cpu().numpy(),
            "predictive_entropy": entropy.cpu().numpy(),
            "mutual_information": (entropy - exp_entropy).cpu().numpy(), "n_samples": n_samples}


def abstention_decision(prob: float, mc_std: Optional[float], threshold: float, max_std: float = 0.15,
                        margin: float = 0.0, quality_passed: bool = True, ood_flag: bool = False,
                        abstain_on_ood: bool = True) -> Dict[str, Any]:
    reasons = []
    if not quality_passed:
        reasons.append("input failed basic quality checks")
    if mc_std is not None and mc_std > max_std:
        reasons.append(f"MC-dropout std {mc_std:.3f} exceeds {max_std}")
    if margin > 0 and abs(prob - threshold) < margin:
        reasons.append(f"probability within {margin} of the decision threshold")
    if ood_flag and abstain_on_ood:
        reasons.append("input looks out-of-distribution relative to training data")
    return {"abstain": bool(reasons), "reasons": reasons}


class MahalanobisOOD:
    """Gaussian fit to training embeddings; distance threshold from a validation percentile."""

    def __init__(self, mean: np.ndarray, precision: np.ndarray, threshold: float, percentile: float,
                 checkpoint_saved_at: Optional[str] = None):
        self.mean, self.precision, self.threshold, self.percentile = mean, precision, threshold, percentile
        self.checkpoint_saved_at = checkpoint_saved_at

    @staticmethod
    def fit(train_emb: np.ndarray, val_emb: np.ndarray, percentile: float = 99.0, shrinkage: float = 1e-3
            ) -> "MahalanobisOOD":
        mean = train_emb.mean(0)
        cov = np.cov(train_emb - mean, rowvar=False)
        cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        precision = np.linalg.inv(cov)
        tmp = MahalanobisOOD(mean, precision, np.inf, percentile)
        return MahalanobisOOD(mean, precision, float(np.percentile(tmp.score(val_emb), percentile)), percentile)

    def score(self, emb: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(emb) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.precision, d))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, precision=self.precision, threshold=self.threshold, percentile=self.percentile,
                 checkpoint_saved_at=np.array(self.checkpoint_saved_at or ""))

    @staticmethod
    def load(path: str | Path) -> Optional["MahalanobisOOD"]:
        if not Path(path).exists():
            return None
        d = np.load(path)
        saved = str(d["checkpoint_saved_at"]) if "checkpoint_saved_at" in d.files else None
        return MahalanobisOOD(d["mean"], d["precision"], float(d["threshold"]), float(d["percentile"]), saved or None)
