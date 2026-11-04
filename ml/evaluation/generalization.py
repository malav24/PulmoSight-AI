"""Internal (RSNA test) vs external (NIH) generalisation summary.

Limitations: NIH 'Pneumonia' is an NLP-mined report label with a different definition and far
lower prevalence than RSNA 'Lung Opacity'; an AUC drop mixes label shift, prevalence/case-mix shift
and image-domain shift. Both datasets come from the NIH Clinical Center, so this is not a
multi-site external validation. Without excluding the RSNA/NIH image overlap the comparison is invalid.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .classification import safe_auc
from .statistics import bootstrap_auc_difference, bootstrap_ci


def generalization_report(y_int: np.ndarray, p_int: np.ndarray, y_ext: np.ndarray, p_ext: np.ndarray,
                          ext_groups: Optional[np.ndarray] = None, n_boot: int = 2000, seed: int = 42) -> Dict:
    return {
        "internal_auc": bootstrap_ci(safe_auc, y_int, p_int, n_boot=n_boot, seed=seed),
        "external_auc": bootstrap_ci(safe_auc, y_ext, p_ext, n_boot=n_boot, seed=seed, groups=ext_groups),
        "auc_difference_internal_minus_external": bootstrap_auc_difference(y_int, p_int, y_ext, p_ext, n_boot, seed=seed),
        "internal_prevalence": float(np.mean(y_int)),
        "external_prevalence": float(np.mean(y_ext)),
        "notes": "Difference CI from an unpaired bootstrap; external CI is patient-clustered when groups are given.",
    }
