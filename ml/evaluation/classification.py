"""Image-level classification metrics.

Limitations
-----------
* ROC-AUC is threshold-free and prevalence-insensitive, but says nothing about calibration or
  about performance at the operating point actually used.
* Sensitivity/specificity/precision/F1 depend on the threshold. Choose it on the VALIDATION set
  (e.g. Youden's J, or a target sensitivity) and apply it unchanged to test/external data;
  choosing it on the test set is optimistic.
* Precision (PPV) depends on prevalence (RSNA ~22-23%, NIH Pneumonia ~1-2%), so it is not
  comparable across datasets.
* Labels themselves are imperfect (radiologist opacity boxes for RSNA, NLP-mined report labels
  for NIH); metrics measure agreement with these reference labels, not clinical truth.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def threshold_for_sensitivity(labels: np.ndarray, probs: np.ndarray, target: float = 0.9) -> float:
    """Highest threshold achieving at least `target` sensitivity (choose on validation data)."""
    pos = np.sort(np.asarray(probs)[np.asarray(labels) == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def confusion_counts(labels: np.ndarray, preds: np.ndarray) -> Dict[str, int]:
    labels, preds = np.asarray(labels).astype(int), np.asarray(preds).astype(int)
    return {"tp": int(((labels == 1) & (preds == 1)).sum()), "fp": int(((labels == 0) & (preds == 1)).sum()),
            "tn": int(((labels == 0) & (preds == 0)).sum()), "fn": int(((labels == 1) & (preds == 0)).sum())}


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    c = confusion_counts(labels, preds)
    sens = _div(c["tp"], c["tp"] + c["fn"])
    spec = _div(c["tn"], c["tn"] + c["fp"])
    prec = _div(c["tp"], c["tp"] + c["fp"])
    f1 = _div(2 * prec * sens, prec + sens) if not (np.isnan(prec) or np.isnan(sens)) else float("nan")
    return {
        "n": int(len(labels)), "n_positive": int(labels.sum()), "prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "threshold": float(threshold), "roc_auc": safe_auc(labels, probs),
        "sensitivity": sens, "specificity": spec, "precision": prec, "recall": sens,
        "npv": _div(c["tn"], c["tn"] + c["fn"]), "f1": f1,
        "accuracy": _div(c["tp"] + c["tn"], len(labels)),
        "confusion_matrix": {"labels": ["negative", "positive"],
                             "matrix": [[c["tn"], c["fp"]], [c["fn"], c["tp"]]], **c},
    }


def plot_roc(curves: Dict[str, tuple], path: str, title: str = "ROC") -> str:
    """curves: {name: (labels, probs)}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, p) in curves.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {safe_auc(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
