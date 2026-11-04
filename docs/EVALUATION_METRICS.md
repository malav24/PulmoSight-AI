# Evaluation Metrics and Their Limitations

| Group | Metric | Module | Limitation |
|---|---|---|---|
| Classification | ROC-AUC | `classification.py` | Threshold-free; ignores calibration and the operating point |
| | Sensitivity, specificity, precision, recall, F1, confusion matrix | `classification.py` | Depend on the threshold (choose it on validation); precision depends on prevalence |
| Localization | IoU, Dice (heatmap ≥ 0.5 vs box mask) | `localization.py` | Boxes are coarse rectangles; results depend on the threshold; positives only |
| | Pointing game (max inside box) | `localization.py` | Ignores spread and multiple lesions; large boxes make it easy |
| | Mass-in-box (energy pointing game) | `localization.py` | Rewards concentration; box size affects chance level |
| Explanation | Deletion AUC (↓), insertion AUC (↑) | `explanation.py` | Perturbed images are out-of-distribution; depends on the baseline (blur/mean) and step count |
| | Attention alignment (mass in box, KL, pointing) | `explanation.py` | Plausibility ≠ faithfulness |
| Shortcut stress | Outside-thorax masking, border masking, marker masking, background noise: AUC drop, mean abs. Δp, decision flip rate | `shortcut.py` | Thorax region is a **geometric ellipse, not a lung segmentation**; marker detection is heuristic; masking itself shifts the distribution |
| Generalisation | RSNA-test AUC, NIH AUC, difference with bootstrap CI | `generalization.py` | Mixes label, prevalence and image shift; same institution |
| Calibration | Temperature scaling, ECE (15 equal-width bins + adaptive), Brier, reliability diagram | `calibration.py` | One global temperature; ECE is biased for small n and depends on binning |
| Statistics | Percentile bootstrap CI (optionally patient-clustered); DeLong paired test & CI; Holm | `statistics.py` | Bootstrap can under-cover at small n; DeLong applies only to paired AUCs on the same images |

Plausibility (do the maps match the boxes?) and faithfulness (do the maps reflect what drives the
prediction?) are separate questions. "Right for the right reason" needs evidence for both, plus
robustness under the stress tests.

Exact procedures and parameters (images used, perturbation baseline, step count, stress-test geometry, bootstrap
settings) are specified in [RESEARCH_METHODOLOGY.md §9–10](RESEARCH_METHODOLOGY.md). Calibration and the decision
threshold are always fitted on the RSNA **validation** split and bound to the evaluated checkpoint; the temperature is
limited to [0.05, 20] and a fit that reaches a limit is flagged as unreliable.
