"""Probability calibration: temperature scaling, ECE, Brier score, reliability diagram.

Limitations
-----------
* Temperature must be fitted on held-out VALIDATION logits and then frozen; fitting on test is leakage.
* A single temperature corrects global over/under-confidence only; it cannot fix miscalibration
  that differs across sub-groups or datasets (e.g. NIH, whose prevalence and label definition differ).
* ECE depends on the number/type of bins and is biased for small samples; report bins and n.
* Brier score mixes calibration and discrimination and is dominated by the majority class at low prevalence.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


class TemperatureScaler:
    """Single-parameter temperature scaling. The fitted T is bounded to [T_MIN, T_MAX]; hitting a bound
    (e.g. a near-random model or a validation set with almost no positives) sets `at_bound` - such a
    calibration is unreliable and is reported as such."""

    T_MIN, T_MAX = 0.05, 20.0

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)
        self.at_bound = False

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> "TemperatureScaler":
        z = torch.as_tensor(np.asarray(logits, dtype=np.float64))
        y = torch.as_tensor(np.asarray(labels, dtype=np.float64))
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        t = float(log_t.exp().item())
        if not np.isfinite(t):
            t = self.T_MAX
        self.at_bound = not (self.T_MIN < t < self.T_MAX)
        self.temperature = float(np.clip(t, self.T_MIN, self.T_MAX))
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.transform_logits(logits)))


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15, strategy: str = "uniform"
                               ) -> float:
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    if len(labels) == 0:
        return float("nan")
    if strategy == "quantile":
        edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(labels[sel].mean() - probs[sel].mean())
    return float(ece)


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    return float(np.mean((probs - labels) ** 2)) if len(labels) else float("nan")


def reliability_curve(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict[str, list]:
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    conf, acc, count = [], [], []
    for b in range(n_bins):
        sel = idx == b
        count.append(int(sel.sum()))
        conf.append(float(probs[sel].mean()) if sel.any() else float("nan"))
        acc.append(float(labels[sel].mean()) if sel.any() else float("nan"))
    return {"bin_edges": edges.tolist(), "mean_confidence": conf, "observed_frequency": acc, "count": count}


def plot_reliability_diagram(curves: Dict[str, tuple], path: str, n_bins: int = 10, title: str = "Reliability") -> str:
    """curves: {name: (labels, probs)}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    for name, (y, p) in curves.items():
        rc = reliability_curve(y, p, n_bins)
        ax.plot(rc["mean_confidence"], rc["observed_frequency"], "o-",
                label=f"{name} (ECE {expected_calibration_error(y, p):.3f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def calibration_report(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    return {"ece": expected_calibration_error(labels, probs, n_bins), "ece_bins": n_bins,
            "adaptive_ece": expected_calibration_error(labels, probs, n_bins, "quantile"),
            "brier": brier_score(labels, probs), "n": int(len(labels))}
