"""Uncertainty of metrics: bootstrap confidence intervals and DeLong's test for paired AUCs.

Limitations
-----------
* Percentile bootstrap CIs can under-cover for small samples or metrics near 0/1.
* When a patient contributes several images (NIH follow-ups), resample PATIENTS (`groups=`),
  otherwise CIs are too narrow.
* DeLong's test is valid for comparing two models' AUCs on the SAME images (paired). For AUCs
  on different datasets (RSNA vs NIH) use the unpaired bootstrap difference instead.
* Multiple comparisons (6 models x several datasets) inflate false positives; correct (e.g. Holm)
  or pre-register the primary comparison (Model A vs Model B).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(metric_fn: Callable[..., float], *arrays: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42, groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Percentile bootstrap CI of metric_fn(*arrays). With `groups`, resamples whole groups (clusters)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays))
    if groups is not None:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.where(inv == g)[0] for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[i] for i in pick])
        try:
            v = metric_fn(*[a[idx] for a in arrays])
        except ValueError:
            continue
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot_valid": len(vals),
            "level": 1 - alpha}


def bootstrap_auc_difference(y1: np.ndarray, p1: np.ndarray, y2: np.ndarray, p2: np.ndarray, n_boot: int = 2000,
                             alpha: float = 0.05, seed: int = 42) -> Dict[str, float]:
    """Unpaired bootstrap of AUC(dataset1) - AUC(dataset2) (e.g. RSNA internal vs NIH external)."""
    from .classification import safe_auc

    rng = np.random.default_rng(seed)
    y1, p1, y2, p2 = map(np.asarray, (y1, p1, y2, p2))
    point = safe_auc(y1, p1) - safe_auc(y2, p2)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y1), len(y1))
        j = rng.integers(0, len(y2), len(y2))
        d = safe_auc(y1[i], p1[i]) - safe_auc(y2[j], p2[j])
        if not np.isnan(d):
            diffs.append(d)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if diffs else (np.nan, np.nan)
    return {"auc_difference": float(point), "ci_low": float(lo), "ci_high": float(hi), "level": 1 - alpha}


# --------------------------------------------------------------------------- DeLong (Sun & Xu, 2014 fast algorithm)
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """preds_sorted: (k, n) with the m positives first. Returns (aucs, covariance)."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def _prep(labels: np.ndarray, *preds: np.ndarray):
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("DeLong requires both classes")
    order = np.argsort(-labels, kind="mergesort")  # positives first
    m = int(labels.sum())
    return np.vstack([np.asarray(p, float)[order] for p in preds]), m


def delong_auc_ci(labels: np.ndarray, probs: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    ps, m = _prep(labels, probs)
    aucs, cov = _fast_delong(ps, m)
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"auc": float(aucs[0]), "se": se, "ci_low": float(max(0, aucs[0] - z * se)),
            "ci_high": float(min(1, aucs[0] + z * se))}


def delong_paired_test(labels: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray) -> Dict[str, float]:
    """Two-sided DeLong test of AUC_a == AUC_b on the same images."""
    ps, m = _prep(labels, probs_a, probs_b)
    aucs, cov = _fast_delong(ps, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": diff, "z": float("nan"),
                "p_value": 1.0 if diff == 0 else float("nan")}
    z = diff / np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": diff, "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}


def holm_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted
