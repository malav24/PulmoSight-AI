"""Shared fixtures. All data here is SYNTHETIC (random noise + random boxes) for software testing only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.utils.config import load_config  # noqa: E402
from ml.utils.synthetic import make_synthetic_nih, make_synthetic_rsna  # noqa: E402

torch.set_num_threads(2)
TINY = 64  # image size used in tests (keeps CPU tests fast)


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path, monkeypatch):
    """Never write checkpoints/outputs into the real project folders during tests."""
    monkeypatch.setenv("RFRR_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("RFRR_OUTPUT_ROOT", str(tmp_path / "outputs"))
    for var in ("RFRR_RSNA_ROOT", "RFRR_NIH_ROOT", "RFRR_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture(scope="session")
def rsna_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rsna")
    make_synthetic_rsna(root, n=30, size=128, pos_frac=0.4, seed=0)
    return root


@pytest.fixture(scope="session")
def nih_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("nih")
    make_synthetic_nih(root, n=12, size=128)
    return root


def tiny_config(name: str, rsna_root: Path, nih_root: Path | None = None, tmp: Path | None = None, **extra):
    tmp = tmp or rsna_root
    overrides = {
        "experiment": {"seed": 0, "device": "cpu"},
        "data": {
            "rsna": {"root": str(rsna_root), "splits_csv": str(tmp / "splits.csv"), "cache_dir": None},
            "nih": {"root": str(nih_root or rsna_root), "rsna_mapping_json": str((nih_root or rsna_root) / "map.json")},
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
            "preprocessing": {"image_size": TINY},
        },
        "loader": {"batch_size": 4, "eval_batch_size": 8, "num_workers": 0, "pin_memory": False},
        "model": {"pretrained_backbone": False},
        "training": {"epochs": 2, "amp": False, "warmup_epochs": 0, "early_stopping_patience": 0},
    }
    if name not in ("model_a.yaml", "model_b.yaml", "ablation_cls_only.yaml"):
        pass
    from ml.utils.config import deep_merge
    return load_config(name, deep_merge(overrides, extra))


@pytest.fixture
def make_cfg(rsna_root, nih_root, tmp_path):
    def _make(name="model_b.yaml", **extra):
        return tiny_config(name, rsna_root, nih_root, tmp_path, **extra)
    return _make
