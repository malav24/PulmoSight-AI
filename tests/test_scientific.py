"""Scientific-validity checks (Stage-3 audit). All inputs are synthetic; these verify mechanics, not performance."""
import os

import numpy as np
import pytest
import torch

from ml.models.factory import build_model
from ml.training.losses import ExplanationGuidedLoss
from ml.utils.config import load_config


def _hybrid(size=64, seed=0):
    torch.manual_seed(seed)
    return build_model(dict(load_config("hybrid_common.yaml")["model"], pretrained_backbone=False), size)


# D. Model A and Model B are comparable -------------------------------------------------------------
def test_model_a_b_differ_only_in_guidance_weight():
    a, b = load_config("model_a.yaml"), load_config("model_b.yaml")
    assert a["model"] == b["model"] and a["data"] == b["data"] and a["training"] == b["training"]
    assert a["loader"] == b["loader"] and a["experiment"]["seed"] == b["experiment"]["seed"]
    diff = {k for k in a["loss"] if a["loss"][k] != b["loss"].get(k)}
    assert diff == {"lambda"}, diff
    assert a["loss"]["lambda"] == 0 and b["loss"]["lambda"] > 0 and a["loss"]["alpha"] == b["loss"]["alpha"] > 0
    c = load_config("ablation_cls_only.yaml")
    assert c["model"] == a["model"] and c["loss"]["alpha"] == 0 and c["loss"]["lambda"] == 0


# E. The guided model really receives the attention loss -------------------------------------------
def test_attention_guidance_changes_gradients_of_attention_head():
    x = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, :32, :32] = 1
    batch = {"label": torch.ones(2), "mask": mask, "has_boxes": torch.tensor([True, True])}
    grads = {}
    for lam in (0.0, 1.0):
        m = _hybrid()
        m.eval()  # deterministic (no dropout) so only the loss differs
        out = m(x)
        losses = ExplanationGuidedLoss({"alpha": 1.0, "lambda": lam})(out, batch)
        losses["loss"].backward()
        grads[lam] = m.attn_score[2].weight.grad.clone()
        if lam > 0:
            assert losses["loss_attn"] > 0
    assert not torch.allclose(grads[0.0], grads[1.0]), "attention loss did not reach the attention-pooling head"


def test_training_log_records_attention_loss_only_for_model_b(make_cfg):
    import pandas as pd
    from ml.training.trainer import run_training
    from ml.utils.config import output_root
    for name in ("model_a.yaml", "model_b.yaml"):
        run_training(make_cfg(name, training={"epochs": 1}))
    la = pd.read_csv(output_root() / "logs" / "model_a_training_log.csv")
    lb = pd.read_csv(output_root() / "logs" / "model_b_training_log.csv")
    # Model A computes no attention loss; Model B's is positive and part of its total loss.
    assert (la.train_loss_attn == 0).all() and (lb.train_loss_attn > 0).all()


# F. Localization output is spatially aligned with the annotation masks ----------------------------
def test_localization_head_learns_correct_location():
    """Overfit two images whose boxes are in opposite corners. The heatmap must be high inside each image's
    own box, low outside, peak inside it, and its centre of mass must sit near the box centre.
    A transposed, flipped or shifted head/mask would fail. 128 px is used because at 64 px the decoder grid
    is only 4x4, so a predicted region reaches the image edge and a single-pixel argmax is not informative."""
    torch.manual_seed(0)
    size = 128
    m = _hybrid(size)
    boxes = [(8, 48, 12, 52), (80, 120, 72, 116)]  # (r1, r2, c1, c2): top-left, bottom-right
    x = torch.randn(2, 3, size, size) * 0.1
    mask = torch.zeros(2, 1, size, size)
    for i, (r1, r2, c1, c2) in enumerate(boxes):
        x[i, :, r1:r2, c1:c2] += 2.0
        mask[i, :, r1:r2, c1:c2] = 1
    batch = {"label": torch.ones(2), "mask": mask, "has_boxes": torch.tensor([True, True])}
    loss_fn = ExplanationGuidedLoss({"alpha": 1.0, "lambda": 0.0})
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    m.train()
    for _ in range(80):
        opt.zero_grad()
        loss_fn(m(x), batch)["loss"].backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        hm = m(x)["heatmap"][:, 0].numpy()
    yy, xx = np.mgrid[:size, :size]
    for i, (r1, r2, c1, c2) in enumerate(boxes):
        h, gt = hm[i], mask[i, 0].numpy() > 0
        r, c = np.unravel_index(h.argmax(), h.shape)
        assert gt[r, c], f"image {i}: peak at {(r, c)} is outside the annotated box"
        assert h[gt].mean() > h[~gt].mean() + 0.5
        w = h / h.sum()
        cy, cx = (w * yy).sum(), (w * xx).sum()
        assert abs(cy - (r1 + r2) / 2) < 12 and abs(cx - (c1 + c2) / 2) < 12, (cy, cx)


# Uncertainty: MC dropout must be stochastic for every architecture --------------------------------
@pytest.mark.parametrize("name", ["hybrid", "resnet50", "densenet121", "efficientnet_b0", "vit_small"])
def test_mc_dropout_produces_spread_for_all_models(name):
    from ml.inference.uncertainty import mc_dropout_predict
    torch.manual_seed(0)
    cfg = dict(load_config("hybrid_common.yaml")["model"]) if name == "hybrid" else {"name": name, "head_dropout": 0.2}
    cfg["pretrained_backbone"] = False
    m = build_model(cfg, 64).eval()
    r = mc_dropout_predict(m, torch.randn(2, 3, 64, 64), n_samples=6)
    assert (r["std"] > 0).all(), f"{name}: MC dropout had no effect"
    assert not m.training


# I. Calibration is fitted on validation and never reused for a different checkpoint ---------------
def test_stale_calibration_is_ignored(make_cfg):
    from ml.inference.predictor import PneumoniaPredictor
    from ml.training.trainer import run_training
    from ml.utils.checkpoints import load_trained_model, save_calibration, checkpoint_identity
    run_training(make_cfg("model_b.yaml", training={"epochs": 1}))
    _, meta = load_trained_model("model_b")
    save_calibration("model_b", {"temperature": 2.0, "threshold": 0.3, "fitted_on": "rsna_val", **checkpoint_identity(meta)})
    p = PneumoniaPredictor("model_b", device="cpu")
    assert p.temperature == 2.0 and p.threshold == 0.3 and not p.warnings
    save_calibration("model_b", {"temperature": 2.0, "threshold": 0.3, "fitted_on": "rsna_val",
                                 "checkpoint_saved_at": "2000-01-01T00:00:00", "checkpoint_epoch": 0})
    p = PneumoniaPredictor("model_b", device="cpu")
    assert p.temperature is None and p.threshold == meta["val_threshold"] and p.warnings


# J. Deletion/insertion procedure is deterministic and documented ---------------------------------
def test_deletion_insertion_is_deterministic():
    from ml.evaluation.explanation import deletion_insertion_curves
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    sal = np.random.default_rng(0).random((2, 32, 32)).astype(np.float32)
    f = lambda t: torch.sigmoid(t.mean(dim=(1, 2, 3)))  # noqa: E731
    a = deletion_insertion_curves(f, x, sal, steps=10, baseline="blur")
    b = deletion_insertion_curves(f, x, sal, steps=10, baseline="blur")
    np.testing.assert_array_equal(a["deletion"], b["deletion"])
    assert a["deletion"].shape == (2, 11) and a["fractions"][0] == 0 and a["fractions"][-1] == 1
    np.testing.assert_allclose(a["deletion"][:, 0], f(x).numpy(), rtol=1e-6)    # 0% removed = original
    np.testing.assert_allclose(a["insertion"][:, -1], f(x).numpy(), rtol=1e-6)  # 100% restored = original


# Portability fixes --------------------------------------------------------------------------------
def test_unicode_paths_and_dotenv(tmp_path, monkeypatch):
    import cv2
    from ml.datasets.preprocessing import imwrite_any_path, load_raster
    from ml.utils.config import load_dotenv
    d = tmp_path / "Bilder_Ünïcødé"
    d.mkdir()
    img = (np.random.default_rng(0).random((40, 50)) * 255).astype(np.uint8)
    assert imwrite_any_path(d / "x.png", img)
    arr, _ = load_raster(d / "x.png")
    np.testing.assert_array_equal(arr.astype(np.uint8), img)
    env = tmp_path / ".env"
    env.write_text("RFRR_TEST_A=hello   # comment\n# RFRR_TEST_B=no\nRFRR_TEST_C='quoted'\n")
    monkeypatch.delenv("RFRR_TEST_A", raising=False)
    monkeypatch.setenv("RFRR_TEST_C", "from-env")
    load_dotenv(env)
    assert os.environ["RFRR_TEST_A"] == "hello" and "RFRR_TEST_B" not in os.environ
    assert os.environ["RFRR_TEST_C"] == "from-env"  # real environment wins
    monkeypatch.delenv("RFRR_TEST_A")


def test_training_and_inference_use_the_same_checkpoint_locations(make_cfg):
    from ml.utils.checkpoints import CHECKPOINT_REGISTRY, checkpoint_path
    from ml.utils.config import checkpoint_root
    names = {"model_a": "model_a.yaml", "model_b": "model_b.yaml", "ablation_cls_only": "ablation_cls_only.yaml",
             "resnet50": "resnet50.yaml", "densenet121": "densenet121.yaml",
             "efficientnet_b0": "efficientnet_b0.yaml", "vit_small": "vit_small.yaml"}
    assert set(names) == set(CHECKPOINT_REGISTRY)
    for key, cfg_name in names.items():
        cfg = load_config(cfg_name)
        assert cfg["checkpoint"]["model_key"] == key
        trainer_path = checkpoint_root() / cfg["checkpoint"]["subdir"] / f"{key}_best.pth"
        assert trainer_path == checkpoint_path(key, "best")
