import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from ml.evaluation.calibration import (TemperatureScaler, brier_score, calibration_report,
                                       expected_calibration_error, plot_reliability_diagram, reliability_curve)
from ml.evaluation.classification import classification_metrics, safe_auc, threshold_for_sensitivity
from ml.evaluation.explanation import attention_alignment, curve_auc, deletion_insertion_curves, explanation_metrics
from ml.evaluation.generalization import generalization_report
from ml.evaluation.localization import (dice, heatmap_to_boxes, iou, localization_metrics, mass_in_box,
                                        pointing_game_hit)
from ml.evaluation.shortcut import (approximate_thorax_mask, border_masking, detect_marker_candidates,
                                    marker_masking, outside_lung_masking, run_stress_tests)
from ml.evaluation.statistics import (bootstrap_auc_difference, bootstrap_ci, delong_auc_ci, delong_paired_test,
                                      holm_correction)

rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 300)
P = np.clip(Y * 0.3 + rng.random(300) * 0.7, 0, 1)


def test_classification_metrics_by_hand():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.2, 0.6, 0.8, 0.1])
    m = classification_metrics(y, p, 0.5)
    assert (m["confusion_matrix"]["tp"], m["confusion_matrix"]["fn"], m["confusion_matrix"]["fp"],
            m["confusion_matrix"]["tn"]) == (2, 1, 1, 2)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert np.isnan(safe_auc(np.ones(3), np.ones(3)))
    t = threshold_for_sensitivity(Y, P, 0.9)
    assert ((P >= t)[Y == 1]).mean() >= 0.9


def test_calibration_metrics_and_temperature():
    assert expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0)
    assert brier_score(np.array([1, 0]), np.array([1.0, 0.0])) == 0
    z = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-z))).astype(float)
    ts = TemperatureScaler().fit(z * 3.0, y)  # over-confident logits by factor 3
    assert ts.temperature == pytest.approx(3.0, rel=0.15)
    before = expected_calibration_error(y, 1 / (1 + np.exp(-3 * z)))
    after = expected_calibration_error(y, ts.predict_proba(z * 3.0))
    assert after < before
    assert not ts.at_bound
    degenerate = TemperatureScaler().fit(np.array([5.0, -5.0, 5.0, -5.0]), np.array([0, 1, 1, 0]))
    assert degenerate.at_bound and degenerate.temperature <= TemperatureScaler.T_MAX
    rc = reliability_curve(y, ts.predict_proba(z * 3), 10)
    assert sum(rc["count"]) == 5000
    assert calibration_report(y, ts.predict_proba(z * 3))["n"] == 5000


def test_reliability_plot(tmp_path):
    out = plot_reliability_diagram({"m": (Y, P)}, str(tmp_path / "r.png"))
    assert (tmp_path / "r.png").exists() and out


def test_delong_matches_sklearn_and_paired_test():
    ci = delong_auc_ci(Y, P)
    assert ci["auc"] == pytest.approx(roc_auc_score(Y, P))
    assert ci["ci_low"] < ci["auc"] < ci["ci_high"]
    same = delong_paired_test(Y, P, P)
    assert same["difference"] == 0 and same["p_value"] == 1.0
    better = np.clip(Y * 0.6 + rng.random(300) * 0.4, 0, 1)
    r = delong_paired_test(Y, better, P)
    assert r["auc_a"] > r["auc_b"] and r["p_value"] < 0.05


def test_bootstrap_ci_and_holm():
    r = bootstrap_ci(safe_auc, Y, P, n_boot=300)
    assert r["ci_low"] <= r["estimate"] <= r["ci_high"]
    groups = np.repeat(np.arange(150), 2)
    rg = bootstrap_ci(safe_auc, Y, P, n_boot=300, groups=groups)
    assert rg["n_boot_valid"] > 0
    d = bootstrap_auc_difference(Y, P, Y, P, n_boot=200)
    assert d["auc_difference"] == 0 and d["ci_low"] <= 0 <= d["ci_high"]
    adj = holm_correction({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    g = generalization_report(Y, P, Y, P, n_boot=100)
    assert "auc_difference_internal_minus_external" in g


def test_localization_metrics():
    gt = np.zeros((10, 10)); gt[2:6, 2:6] = 1
    pred = np.zeros((10, 10)); pred[2:6, 4:8] = 1
    assert iou(pred, gt) == pytest.approx(8 / 24) and dice(pred, gt) == pytest.approx(16 / 32)
    sal = np.zeros((10, 10)); sal[3, 3] = 1
    assert pointing_game_hit(sal, gt) is True
    sal2 = np.zeros((10, 10)); sal2[8, 8] = 1
    assert pointing_game_hit(sal2, gt) is False and pointing_game_hit(sal2, gt, tolerance=3) is True
    assert pointing_game_hit(sal, np.zeros((10, 10))) is None
    assert mass_in_box(gt, gt) == pytest.approx(1.0)
    assert len(heatmap_to_boxes(pred, 0.5, 1)) == 1
    agg = localization_metrics([pred, pred], [gt, np.zeros((10, 10))])
    assert agg["n_annotated"] == 1


def _toy_predict(x):
    # "model" that reacts only to the top-left quadrant brightness
    return torch.sigmoid(x[:, :, :8, :8].mean(dim=(1, 2, 3)) * 5)


def test_deletion_insertion_rewards_faithful_saliency():
    x = torch.zeros(2, 3, 16, 16); x[:, :, :8, :8] = 2.0
    good = np.zeros((2, 16, 16), np.float32); good[:, :8, :8] = 1
    bad = np.zeros((2, 16, 16), np.float32); bad[:, 8:, 8:] = 1
    cg = deletion_insertion_curves(_toy_predict, x, good, steps=8, baseline="zero")
    cb = deletion_insertion_curves(_toy_predict, x, bad, steps=8, baseline="zero")
    assert curve_auc(cg["fractions"], cg["deletion"]).mean() < curve_auc(cb["fractions"], cb["deletion"]).mean()
    assert curve_auc(cg["fractions"], cg["insertion"]).mean() > curve_auc(cb["fractions"], cb["insertion"]).mean()
    em = explanation_metrics(_toy_predict, x, good, gt_masks=good, steps=4, baseline="blur")
    assert em["mass_in_box"] == pytest.approx(1.0) and em["pointing_game"] == 1.0


def test_attention_alignment_metric():
    masks = torch.zeros(2, 1, 32, 32); masks[0, :, :16, :16] = 1
    att = torch.tensor([[1.0, 0, 0, 0], [0.25] * 4]).view(2, 1, 2, 2)
    r = attention_alignment(att, masks)
    assert r["n"] == 1 and r["attention_mass_in_box"] == pytest.approx(1.0)


def test_shortcut_perturbations_and_stress_runner():
    x = torch.randn(4, 3, 64, 64)
    assert outside_lung_masking(x).shape == x.shape
    bm = border_masking(x, 0.1)
    assert torch.equal(bm[:, :, 10:54, 10:54], x[:, :, 10:54, 10:54])
    g = np.zeros((64, 64), np.uint8); g[2:6, 2:6] = 255; g[20:40, 20:40] = 100
    assert detect_marker_candidates(g)[3, 3] == 1 and detect_marker_candidates(g)[30, 30] == 0
    assert 0 < approximate_thorax_mask(64, 64).mean() < 1
    _, frac = marker_masking(x)
    assert 0 <= frac <= 1
    loader = [{"image": x, "label": torch.tensor([0., 1., 0., 1.])}]
    rep = run_stress_tests(lambda t: torch.sigmoid(t.mean(dim=(1, 2, 3))), loader, "cpu")
    assert set(rep) >= {"original", "outside_lung_masking", "border_masking", "marker_masking",
                        "background_perturbation"}
