"""Centralised checkpoint locations, saving, discovery and strict loading.

Rules enforced here:
  * Inference never uses randomly initialised or ImageNet-only weights.
  * A missing checkpoint raises CheckpointNotFoundError with the expected path.
  * Loading is strict (all keys must match) and the stored model name is verified.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import checkpoint_root, resolve_path

# Intended locations, relative to checkpoint_root() (default <project>/checkpoints,
# override with env RFRR_CHECKPOINT_ROOT). These files do NOT exist until you train the models.
CHECKPOINT_REGISTRY: Dict[str, Dict[str, str]] = {
    "model_a": {
        "best": "hybrid/model_a_best.pth",
        "last": "hybrid/model_a_last.pth",
        "family": "hybrid",
    },
    "model_b": {
        "best": "hybrid/model_b_best.pth",
        "last": "hybrid/model_b_last.pth",
        "family": "hybrid",
    },
    "ablation_cls_only": {
        "best": "hybrid/ablation_cls_only_best.pth",
        "last": "hybrid/ablation_cls_only_last.pth",
        "family": "hybrid",
    },
    "resnet50": {
        "best": "baselines/resnet50_best.pth",
        "last": "baselines/resnet50_last.pth",
        "family": "baseline",
    },
    "densenet121": {
        "best": "baselines/densenet121_best.pth",
        "last": "baselines/densenet121_last.pth",
        "family": "baseline",
    },
    "efficientnet_b0": {
        "best": "baselines/efficientnet_b0_best.pth",
        "last": "baselines/efficientnet_b0_last.pth",
        "family": "baseline",
    },
    "vit_small": {
        "best": "baselines/vit_small_best.pth",
        "last": "baselines/vit_small_last.pth",
        "family": "baseline",
    },
}

CALIBRATION_FILE = "calibration/temperature_scaling.json"


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a trained checkpoint is required but absent."""


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint does not match the requested model."""


def checkpoint_path(model_key: str, kind: str = "best") -> Path:
    if model_key not in CHECKPOINT_REGISTRY:
        raise KeyError(f"Unknown model key {model_key!r}. Known: {sorted(CHECKPOINT_REGISTRY)}")
    return checkpoint_root() / CHECKPOINT_REGISTRY[model_key][kind]


def discover_checkpoints() -> Dict[str, Dict[str, Any]]:
    """Report which registered checkpoints exist on disk (no loading)."""
    report: Dict[str, Dict[str, Any]] = {}
    for key, entry in CHECKPOINT_REGISTRY.items():
        report[key] = {}
        for kind in ("best", "last"):
            p = checkpoint_root() / entry[kind]
            report[key][kind] = {"path": str(p), "exists": p.exists(),
                                 "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None}
    cal = checkpoint_root() / CALIBRATION_FILE
    report["_calibration"] = {"path": str(cal), "exists": cal.exists()}
    return report


def save_checkpoint(path: str | Path, model: torch.nn.Module, *, model_key: str, config: Dict[str, Any],
                    epoch: int, metrics: Dict[str, Any], optimizer=None, scheduler=None, scaler=None,
                    early_stopping_state: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "model_key": model_key,
        "model_config": config.get("model", {}),
        "image_size": config.get("data", {}).get("preprocessing", {}).get("image_size"),
        "preprocessing": config.get("data", {}).get("preprocessing", {}),
        "loss_config": config.get("loss", {}),
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
        "model_state": model.state_dict(),
        "trained": True,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    if early_stopping_state is not None:
        state["early_stopping_state"] = early_stopping_state
    if extra:
        state.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic on the same filesystem
    return path


def read_checkpoint(path: str | Path, map_location="cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise CheckpointNotFoundError(
            f"Checkpoint not found: {path}\n"
            "No trained weights are available for this model. Train it first "
            "(see docs/TRAINING_GUIDE.md); inference will not fall back to random or ImageNet weights.")
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13
        state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict) or "model_state" not in state:
        raise CheckpointMismatchError(f"{path} is not a checkpoint produced by this project (no 'model_state').")
    if not state.get("trained", False):
        raise CheckpointMismatchError(f"{path} is not marked as a trained checkpoint.")
    return state


def load_trained_model(model_key: str, path: Optional[str | Path] = None, device: str | torch.device = "cpu",
                       kind: str = "best"):
    """Build the architecture stored in the checkpoint and load weights strictly.

    Backbones are constructed with pretrained=False; *all* weights come from the checkpoint.
    Returns (model in eval mode, checkpoint metadata dict without tensors).
    """
    from ml.models.factory import build_model  # local import to avoid cycles

    path = Path(path) if path is not None else checkpoint_path(model_key, kind)
    state = read_checkpoint(path)
    stored_key = state.get("model_key")
    if stored_key != model_key:
        raise CheckpointMismatchError(f"Checkpoint {path} stores model {stored_key!r}, expected {model_key!r}.")
    model_cfg = dict(state["model_config"])
    model = build_model(model_cfg, image_size=state.get("image_size") or 384, pretrained_backbone=False)
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    if missing or unexpected:
        raise CheckpointMismatchError(
            f"State dict mismatch for {path}: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    model.to(device).eval()
    meta = {k: v for k, v in state.items() if not k.endswith("_state")}
    meta["checkpoint_path"] = str(path)
    return model, meta


def checkpoint_identity(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Fields that identify one specific trained checkpoint (used to detect stale calibration/OOD files)."""
    return {"checkpoint_saved_at": meta.get("saved_at"), "checkpoint_epoch": meta.get("epoch")}


def calibration_matches(cal: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> bool:
    return bool(cal) and cal.get("checkpoint_saved_at") == meta.get("saved_at") \
        and cal.get("checkpoint_epoch") == meta.get("epoch")


def load_calibration(model_key: str) -> Optional[Dict[str, Any]]:
    p = checkpoint_root() / CALIBRATION_FILE
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get(model_key)


def save_calibration(model_key: str, payload: Dict[str, Any]) -> Path:
    p = checkpoint_root() / CALIBRATION_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    data[model_key] = payload
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return p
