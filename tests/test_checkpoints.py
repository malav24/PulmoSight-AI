import pytest
import torch

from ml.models.factory import build_model
from ml.utils.checkpoints import (CHECKPOINT_REGISTRY, CheckpointMismatchError, CheckpointNotFoundError,
                                  checkpoint_path, discover_checkpoints, load_calibration, load_trained_model,
                                  read_checkpoint, save_calibration, save_checkpoint)
from ml.utils.config import load_config


def _cfg():
    cfg = load_config("model_b.yaml")
    cfg["model"]["pretrained_backbone"] = False
    cfg["data"]["preprocessing"]["image_size"] = 64
    return cfg


def test_registry_has_all_expected_locations(isolated_roots):
    for key in ["model_a", "model_b", "resnet50", "densenet121", "efficientnet_b0", "vit_small"]:
        assert key in CHECKPOINT_REGISTRY
    assert checkpoint_path("model_b").as_posix().endswith("checkpoints/hybrid/model_b_best.pth")
    assert checkpoint_path("vit_small").as_posix().endswith("checkpoints/baselines/vit_small_best.pth")
    rep = discover_checkpoints()
    assert rep["model_b"]["best"]["exists"] is False


def test_missing_checkpoint_raises_clear_error():
    with pytest.raises(CheckpointNotFoundError, match="Train it first"):
        load_trained_model("model_b")
    with pytest.raises(KeyError):
        checkpoint_path("not_a_model")


def test_save_and_load_roundtrip(tmp_path):
    cfg = _cfg()
    model = build_model(cfg["model"], 64)
    p = save_checkpoint(checkpoint_path("model_b"), model, model_key="model_b", config=cfg, epoch=3,
                        metrics={"val_auc": 0.5}, extra={"val_threshold": 0.4})
    loaded, meta = load_trained_model("model_b")
    for (k, v), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)
    assert meta["epoch"] == 3 and meta["val_threshold"] == 0.4 and not loaded.training
    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x)["logits"], loaded(x)["logits"])
    assert p.exists() and not p.with_suffix(".pth.tmp").exists()


def test_wrong_model_key_and_untrained_files_rejected(tmp_path):
    cfg = _cfg()
    model = build_model(cfg["model"], 64)
    p = save_checkpoint(tmp_path / "x.pth", model, model_key="model_a", config=cfg, epoch=0, metrics={})
    with pytest.raises(CheckpointMismatchError):
        load_trained_model("model_b", p)
    torch.save(model.state_dict(), tmp_path / "raw.pth")  # e.g. a bare ImageNet/state_dict file
    with pytest.raises(CheckpointMismatchError):
        read_checkpoint(tmp_path / "raw.pth")


def test_calibration_store():
    assert load_calibration("model_b") is None
    save_calibration("model_b", {"temperature": 1.7})
    assert load_calibration("model_b")["temperature"] == 1.7
