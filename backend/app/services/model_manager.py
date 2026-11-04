"""Discovery, metadata and thread-safe loading of TRAINED checkpoints (via Stage-1 ml.utils.checkpoints)."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.utils.checkpoints import (CHECKPOINT_REGISTRY, CheckpointMismatchError, CheckpointNotFoundError,
                                  calibration_matches, checkpoint_path, load_calibration, read_checkpoint)
from ml.utils.config import PROJECT_ROOT, output_root

from ..catalog import MODEL_CATALOG
from ..errors import APIError, model_not_trained
from ..settings import get_settings

log = logging.getLogger("rfrr.api.models")

VALIDATION_KEYS = ("val_auc", "val_loss", "val_loss_cls", "val_dice_pos", "val_attn_mass_in_box",
                   "val_threshold_youden", "train_loss")


def display_path(p: Path) -> str:
    """Project-relative path for users; never expose absolute server paths."""
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"<checkpoint root>/{Path(p).parent.name}/{Path(p).name}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelManager:
    def __init__(self) -> None:
        self._predictors: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        self._meta_cache: Dict[str, tuple] = {}
        self.active_key: Optional[str] = None

    # ------------------------------------------------------------ metadata
    def checkpoint_info(self, key: str) -> Dict[str, Any]:
        path = checkpoint_path(key, "best")
        info: Dict[str, Any] = {"exists": path.exists(), "path": display_path(path), "status": "not_trained"}
        if not path.exists():
            return info
        stat = path.stat()
        cached = self._meta_cache.get(key)
        if cached and cached[0] == (stat.st_mtime, stat.st_size):
            return cached[1]
        try:
            state = read_checkpoint(path)
            if state.get("model_key") != key:
                raise CheckpointMismatchError(f"stores {state.get('model_key')!r}")
            metrics = state.get("metrics", {}) or {}
            loss = state.get("loss_config", {}) or {}
            sha = _sha256(path)
            info.update({
                "status": "trained",
                "epoch": state.get("epoch"),
                "saved_at": state.get("saved_at"),
                "size_mb": round(stat.st_size / 1e6, 1),
                "checkpoint_id": f"{key}@epoch{state.get('epoch')}#{sha[:12]}",
                "sha256": sha,
                "image_size": state.get("image_size"),
                "loss_weights": {"alpha": loss.get("alpha"), "lambda": loss.get("lambda"),
                                 "attention_loss": loss.get("attention")},
                "validation_metrics": {k: metrics[k] for k in VALIDATION_KEYS if k in metrics
                                       and metrics[k] not in (None, "")},
                "validation_threshold": state.get("val_threshold"),
                "monitor": state.get("monitor"),
                "_identity": {"saved_at": state.get("saved_at"), "epoch": state.get("epoch")},
                "dataset": "RSNA Pneumonia Detection (patient-level validation split)",
            })
        except (CheckpointMismatchError, Exception) as exc:  # noqa: BLE001
            log.warning("Checkpoint %s unreadable: %s", key, exc)
            info.update({"status": "invalid", "error": "Checkpoint file exists but could not be read or does not "
                                                       "match this model."})
        self._meta_cache[key] = ((stat.st_mtime, stat.st_size), info)
        return info

    def evaluation_summary(self, key: str) -> Optional[Dict[str, Any]]:
        p = output_root() / "metrics" / f"{key}_metrics.json"
        if not p.exists():
            return None
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
        t = m.get("rsna_test", {})
        c = t.get("classification", {})
        return {"rsna_test_auc": c.get("roc_auc"), "rsna_test_auc_ci": c.get("roc_auc_bootstrap_ci"),
                "sensitivity": c.get("sensitivity"), "specificity": c.get("specificity"), "f1": c.get("f1"),
                "threshold": m.get("threshold", {}).get("value"),
                "localization_dice": t.get("localization_heatmap", {}).get("dice"),
                "localization_iou": t.get("localization_heatmap", {}).get("iou"),
                "attention_mass_in_box": t.get("attention_alignment", {}).get("attention_mass_in_box"),
                "ece_temperature_scaled": t.get("calibration_temperature_scaled", {}).get("ece"),
                "nih_auc": m.get("nih_external", {}).get("classification", {}).get("roc_auc"),
                "evaluated_checkpoint_epoch": m.get("epoch"),
                "evaluated_checkpoint_saved_at": m.get("checkpoint_saved_at")}

    def describe(self, key: str) -> Dict[str, Any]:
        cat = MODEL_CATALOG[key]
        ck = dict(self.checkpoint_info(key))
        identity = ck.pop("_identity", None)
        cal = load_calibration(key)
        calibration = None
        if cal:
            calibration = {**cal, "matches_checkpoint": bool(identity) and calibration_matches(cal, identity)}
        evaluation = self.evaluation_summary(key)
        if evaluation is not None and identity and (
                evaluation.get("evaluated_checkpoint_saved_at") != identity.get("saved_at")
                or evaluation.get("evaluated_checkpoint_epoch") != identity.get("epoch")):
            evaluation["stale"] = True  # metrics file belongs to an earlier checkpoint
        return {"key": key, **cat, "trained": ck["status"] == "trained", "checkpoint": ck,
                "calibration": calibration, "evaluation": evaluation,
                "loaded": key in self._predictors, "active": key == self.active_key}

    def list_models(self) -> List[Dict[str, Any]]:
        return [self.describe(k) for k in MODEL_CATALOG if k in CHECKPOINT_REGISTRY]

    def trained_keys(self) -> List[str]:
        return [k for k in MODEL_CATALOG if self.checkpoint_info(k)["status"] == "trained"]

    # ------------------------------------------------------------ loading
    def _validate_key(self, key: str) -> None:
        if key not in MODEL_CATALOG:
            raise APIError(404, "unknown_model", f"Unknown model '{key}'.", {"known": list(MODEL_CATALOG)})

    def get_predictor(self, key: str, reload: bool = False):
        self._validate_key(key)
        with self._registry_lock:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if reload:
                self._predictors.pop(key, None)
            if key not in self._predictors:
                from ml.inference.predictor import PneumoniaPredictor
                s = get_settings()
                try:
                    self._predictors[key] = PneumoniaPredictor(key, device=s.device, mc_samples=s.mc_samples,
                                                               abstain_std=s.abstain_std)
                except CheckpointNotFoundError:
                    raise model_not_trained(key, display_path(checkpoint_path(key)))
                except CheckpointMismatchError as exc:
                    raise APIError(422, "invalid_checkpoint", f"Checkpoint for '{key}' is invalid: {exc}")
                log.info("Loaded model %s", key)
        return self._predictors[key]

    def lock_for(self, key: str) -> threading.Lock:
        with self._registry_lock:
            return self._locks.setdefault(key, threading.Lock())

    def load(self, key: str, reload: bool = False) -> Dict[str, Any]:
        self.get_predictor(key, reload=reload)
        self.active_key = key
        return self.describe(key)

    def resolve_key(self, requested: Optional[str]) -> str:
        if requested:
            self._validate_key(requested)
            return requested
        if self.active_key:
            return self.active_key
        trained = self.trained_keys()
        default = get_settings().default_model
        if default in trained:
            return default
        if trained:
            return trained[0]
        raise model_not_trained(default, display_path(checkpoint_path(default)))

    def invalidate(self) -> None:
        """Drop cached predictors/metadata (e.g. after an evaluation job refreshed calibration files)."""
        with self._registry_lock:
            self._predictors.clear()
            self._meta_cache.clear()


manager = ModelManager()
