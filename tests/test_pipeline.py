"""End-to-end SMOKE tests on synthetic data: train -> resume -> evaluate -> inference.

These verify the software runs; the resulting numbers are meaningless (random noise images).
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from ml.utils.checkpoints import checkpoint_path, discover_checkpoints
from ml.utils.config import output_root


@pytest.mark.parametrize("cfg_name,key", [("model_b.yaml", "model_b"), ("resnet50.yaml", "resnet50")])
def test_train_resume_and_log(make_cfg, cfg_name, key):
    from ml.training.trainer import run_training
    cfg = make_cfg(cfg_name)
    res = run_training(cfg)
    assert checkpoint_path(key, "best").exists() and checkpoint_path(key, "last").exists()
    log = pd.read_csv(output_root() / "logs" / f"{key}_training_log.csv")
    assert list(log.epoch) == [0, 1]
    for col in ("train_loss", "val_loss", "val_auc", "train_loss_cls"):
        assert col in log.columns
    if key == "model_b":
        assert (log.train_loss_attn > 0).all() and (log.train_loss_loc > 0).all()
        assert "val_attn_mass_in_box" in log.columns
    # resume: continue to epoch 3 from the last checkpoint
    cfg2 = make_cfg(cfg_name, training={"epochs": 3, "resume": True})
    run_training(cfg2)
    log2 = pd.read_csv(output_root() / "logs" / f"{key}_training_log.csv")
    assert list(log2.epoch) == [0, 1, 2]
    state = torch.load(checkpoint_path(key, "last"), weights_only=False)
    assert state["epoch"] == 2 and "optimizer_state" in state


def test_baseline_rejects_guidance(make_cfg):
    from ml.training.trainer import run_training
    with pytest.raises(ValueError, match="classification-only"):
        run_training(make_cfg("resnet50.yaml", loss={"alpha": 1.0}))


def test_evaluate_and_predict_end_to_end(make_cfg, nih_root, rsna_root):
    from ml.evaluation.evaluate import evaluate
    from ml.inference.predictor import PneumoniaPredictor, to_json_safe
    from ml.training.trainer import run_training

    run_training(make_cfg("model_b.yaml", training={"epochs": 1}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evaluate("model_b", use_nih=True, explain_n=3, explain_steps=4, stress=True, fit_ood=True, n_boot=50,
                       device="cpu")
    assert "rsna_test" in res and "nih_external" in res and "generalization" in res
    assert "localization_heatmap" in res["rsna_test"] and "attention_alignment" in res["rsna_test"]
    assert "gradcam_faithfulness" in res["rsna_test"]["explanations"]
    assert "outside_lung_masking" in res["rsna_test"]["stress_tests"]
    assert (output_root() / "metrics" / "model_b_metrics.json").exists()
    assert (output_root() / "predictions" / "model_b_rsna_test.csv").exists()
    assert discover_checkpoints()["_calibration"]["exists"]
    from ml.utils.checkpoints import load_calibration
    cal = load_calibration("model_b")
    assert cal["fitted_on"] == "rsna_val" and 0 < cal["threshold"] < 1 and cal["checkpoint_saved_at"]
    assert res["threshold"]["source"].startswith("Youden J on RSNA validation")

    pred = PneumoniaPredictor("model_b", device="cpu", mc_samples=4)
    dcm = next((rsna_root / "stage_2_train_images").glob("*.dcm"))
    out = pred.predict(dcm, return_images=True)
    assert 0 <= out["probability"] <= 1 and out["temperature"] is not None
    assert set(out["maps"]) == {"localization", "attention", "gradcam"}
    assert out["uncertainty"]["n_samples"] == 4 and out["ood"]["available"]
    assert "disclaimer" in out and "overlay_png_base64" in out["maps"]["gradcam"]
    json.dumps(to_json_safe(out))  # API-serialisable
    with pytest.raises(ValueError):
        pred.predict(rsna_root / "stage_2_train_labels.csv")


def test_predictor_requires_trained_checkpoint():
    from ml.inference.predictor import PneumoniaPredictor
    from ml.utils.checkpoints import CheckpointNotFoundError
    with pytest.raises(CheckpointNotFoundError):
        PneumoniaPredictor("model_a", device="cpu")


def test_quality_checks_flag_bad_inputs():
    from ml.inference.quality import check_image_quality
    ok = check_image_quality(np.random.default_rng(0).normal(1000, 200, (512, 512)).astype(np.float32),
                             {"source": "dicom", "modality": "CR", "view_position": "LL"})
    assert ok["passed"] and any("View position" in w for w in ok["warnings"])
    bad = check_image_quality(np.zeros((100, 100), np.float32), {"source": "raster", "color_spread": 20})
    assert not bad["passed"] and len(bad["errors"]) >= 2


def test_mc_dropout_and_abstention():
    from ml.inference.uncertainty import MahalanobisOOD, abstention_decision, mc_dropout_predict
    from ml.models.factory import build_model
    from ml.utils.config import load_config
    m = build_model(dict(load_config("hybrid_common.yaml")["model"], pretrained_backbone=False), 64).eval()
    r = mc_dropout_predict(m, torch.randn(2, 3, 64, 64), n_samples=5)
    assert r["std"].shape == (2,) and (r["std"] > 0).all() and not m.training
    d = abstention_decision(0.5, 0.3, 0.5, max_std=0.15)
    assert d["abstain"] and d["reasons"]
    assert not abstention_decision(0.9, 0.01, 0.5)["abstain"]
    emb = np.random.default_rng(0).normal(size=(200, 8))
    ood = MahalanobisOOD.fit(emb[:150], emb[150:])
    assert ood.score(emb[:1] + 50)[0] > ood.threshold
