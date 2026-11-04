"""Generic trainer shared by all six models (hybrids and baselines)."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.utils.checkpoints import read_checkpoint, save_checkpoint
from ml.utils.config import checkpoint_root, load_config, output_root, parse_cli_overrides
from ml.utils.logging_utils import CSVLogger, get_logger, save_json
from ml.utils.runtime import amp_settings, environment_report, get_device, set_seed
from .losses import ExplanationGuidedLoss

log = get_logger("rfrr.train")


# --------------------------------------------------------------------------- helpers
class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def improved(self, value: float) -> bool:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        if self.best is None:
            return True
        return value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Return True if this value is a new best."""
        if self.improved(value):
            self.best, self.bad_epochs = value, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience

    def state_dict(self) -> Dict[str, Any]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, s: Dict[str, Any]) -> None:
        self.best, self.bad_epochs = s.get("best"), s.get("bad_epochs", 0)


def build_optimizer(model: nn.Module, tcfg: Dict[str, Any]) -> torch.optim.Optimizer:
    backbone = {id(p) for p in getattr(model, "backbone_parameters", lambda: [])()}
    bb_params = [p for p in model.parameters() if id(p) in backbone]
    other = [p for p in model.parameters() if id(p) not in backbone]
    groups = [{"params": other, "lr": float(tcfg["lr"]), "name": "head"}]
    if bb_params:
        groups.append({"params": bb_params, "lr": float(tcfg.get("backbone_lr", tcfg["lr"])), "name": "backbone"})
    return torch.optim.AdamW(groups, weight_decay=float(tcfg.get("weight_decay", 1e-4)))


def build_scheduler(optimizer, tcfg: Dict[str, Any], steps_per_epoch: int):
    kind = tcfg.get("scheduler", "cosine")
    if kind == "cosine":
        total = max(1, int(tcfg["epochs"]) * steps_per_epoch)
        warm = int(tcfg.get("warmup_epochs", 0)) * steps_per_epoch
        min_ratio = float(tcfg.get("min_lr", 1e-6)) / float(tcfg["lr"])

        def fn(step: int) -> float:
            if warm and step < warm:
                return (step + 1) / warm
            prog = min(1.0, (step - warm) / max(1, total - warm))
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn), "step"
    if kind == "plateau":
        mode = tcfg.get("monitor_mode", "max")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=2), "epoch"
    return None, None


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    j = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[j], 1e-6, 1 - 1e-6))


def _make_grad_scaler(enabled: bool):
    """torch.amp.GradScaler('cuda') exists from torch 2.3; fall back to torch.cuda.amp for older versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, model: nn.Module, cfg: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader,
                 device: Optional[torch.device] = None, pos_weight: Optional[float] = None):
        self.cfg = cfg
        self.tcfg = cfg["training"]
        self.model_key = cfg["checkpoint"]["model_key"]
        self.device = device or get_device(cfg["experiment"].get("device", "auto"))
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.criterion = ExplanationGuidedLoss(cfg["loss"], pos_weight=pos_weight).to(self.device)
        self.optimizer = build_optimizer(self.model, self.tcfg)
        steps = len(train_loader) if not self.tcfg.get("max_train_batches") else min(
            len(train_loader), int(self.tcfg["max_train_batches"]))
        self.scheduler, self.sched_mode = build_scheduler(self.optimizer, self.tcfg, max(1, steps))
        amp = amp_settings(self.device, bool(self.tcfg.get("amp", True)))
        self.amp_enabled, self.amp_dtype = amp["enabled"], amp["dtype"]
        self.scaler = _make_grad_scaler(amp["use_scaler"]) if self.device.type == "cuda" else None
        self.early = EarlyStopping(int(self.tcfg.get("early_stopping_patience", 0)), self.tcfg.get("monitor_mode", "max"))
        ck_dir = checkpoint_root() / cfg["checkpoint"].get("subdir", "hybrid")
        self.best_path = ck_dir / f"{self.model_key}_best.pth"
        self.last_path = ck_dir / f"{self.model_key}_last.pth"
        self.log_path = output_root() / "logs" / f"{self.model_key}_training_log.csv"
        self.start_epoch = 0
        resume = bool(self.tcfg.get("resume", False))
        if resume:
            self._resume()
        self.csv = CSVLogger(self.log_path, resume=resume and self.start_epoch > 0)

    # ---------------------------------------------------------------- resume
    def _resume(self) -> None:
        if not self.last_path.exists():
            log.warning("resume requested but %s does not exist - starting from scratch", self.last_path)
            return
        state = read_checkpoint(self.last_path, map_location=self.device)
        if state.get("model_key") != self.model_key:
            raise RuntimeError(f"Resume checkpoint is for {state.get('model_key')}, not {self.model_key}")
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler is not None and "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if self.scaler is not None and "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        if "early_stopping_state" in state:
            self.early.load_state_dict(state["early_stopping_state"])
        self.start_epoch = int(state["epoch"]) + 1
        log.info("Resumed %s from epoch %d (best %s)", self.model_key, state["epoch"], self.early.best)

    # ---------------------------------------------------------------- epochs
    def _set_backbone_trainable(self, trainable: bool) -> None:
        for p in getattr(self.model, "backbone_parameters", lambda: [])():
            p.requires_grad_(trainable)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        freeze = epoch < int(self.tcfg.get("freeze_backbone_epochs", 0))
        self._set_backbone_trainable(not freeze)
        sums: Dict[str, float] = {}
        n = 0
        max_b = self.tcfg.get("max_train_batches")
        for i, batch in enumerate(self.train_loader):
            if max_b and i >= int(max_b):
                break
            batch = move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                out = self.model(batch["image"])
            losses = self.criterion(out, batch)  # loss maths in fp32
            loss = losses["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch} batch {i}: {losses}")
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            if self.tcfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.tcfg["grad_clip"]))
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None and self.sched_mode == "step":
                self.scheduler.step()
            bs = batch["image"].shape[0]
            n += bs
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * bs
        return {f"train_{k}": v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        from ml.evaluation.classification import safe_auc

        self.model.eval()
        sums: Dict[str, float] = {}
        n = 0
        probs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        dices: List[float] = []
        inside: List[float] = []
        max_b = self.tcfg.get("max_val_batches")
        for i, batch in enumerate(self.val_loader):
            if max_b and i >= int(max_b):
                break
            batch = move_batch(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                out = self.model(batch["image"])
            losses = self.criterion(out, batch)
            bs = batch["image"].shape[0]
            n += bs
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v) * bs
            probs.append(out["probs"].float().cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            pos = batch["has_boxes"].bool()
            if pos.any() and "heatmap" in out:
                hm = (out["heatmap"][pos].float() > 0.5).float()
                m = batch["mask"][pos].float()
                inter = (hm * m).flatten(1).sum(1)
                dices += ((2 * inter + 1e-6) / (hm.flatten(1).sum(1) + m.flatten(1).sum(1) + 1e-6)).cpu().tolist()
            if pos.any() and "attention" in out:
                from .losses import mask_to_grid
                a = out["attention"][pos].float()
                cov = mask_to_grid(batch["mask"][pos], a.shape[-2:])
                inside += (a * (cov > 0).float()).flatten(1).sum(1).cpu().tolist()
        p = np.concatenate(probs) if probs else np.zeros(0)
        y = np.concatenate(labels) if labels else np.zeros(0)
        metrics = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
        metrics["val_auc"] = safe_auc(y, p)
        metrics["val_threshold_youden"] = youden_threshold(y, p) if len(p) else 0.5
        if dices:
            metrics["val_dice_pos"] = float(np.mean(dices))
        if inside:
            metrics["val_attn_mass_in_box"] = float(np.mean(inside))
        return metrics

    # ---------------------------------------------------------------- loop
    def fit(self) -> Dict[str, Any]:
        epochs = int(self.tcfg["epochs"])
        monitor = self.tcfg.get("monitor", "val_auc")
        history: Dict[str, Any] = {}
        log.info("Training %s on %s (AMP=%s, dtype=%s) epochs %d..%d", self.model_key, self.device,
                 self.amp_enabled, self.amp_dtype, self.start_epoch, epochs - 1)
        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            row: Dict[str, Any] = {"epoch": epoch}
            row.update(self.train_one_epoch(epoch))
            row.update(self.validate())
            row["lr_head"] = self.optimizer.param_groups[0]["lr"]
            row["epoch_seconds"] = round(time.time() - t0, 1)
            value = row.get(monitor)
            if self.scheduler is not None and self.sched_mode == "epoch" and value is not None:
                self.scheduler.step(value)
            is_best = self.early.step(value)
            row["is_best"] = int(is_best)
            self.csv.log(row)
            meta = {"val_threshold": row.get("val_threshold_youden", 0.5), "monitor": monitor,
                    "monitor_value": value}
            common = dict(model_key=self.model_key, config=self.cfg, epoch=epoch, metrics=row, extra=meta)
            if is_best:
                save_checkpoint(self.best_path, self.model, **common)
            save_checkpoint(self.last_path, self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                            scaler=self.scaler, early_stopping_state=self.early.state_dict(), **common)
            log.info("epoch %d | loss %.4f | val_loss %.4f | %s %.4f%s", epoch, row.get("train_loss", float("nan")),
                     row.get("val_loss", float("nan")), monitor, value if value is not None else float("nan"),
                     " *best*" if is_best else "")
            history = row
            if self.early.should_stop:
                log.info("Early stopping at epoch %d (no %s improvement for %d epochs)", epoch, monitor,
                         self.early.patience)
                break
        return {"best": self.early.best, "best_path": str(self.best_path), "last_path": str(self.last_path),
                "log": str(self.log_path), "last_epoch_metrics": history}


# --------------------------------------------------------------------------- entry point
def run_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from ml.datasets.loaders import build_rsna_datasets, make_loader
    from ml.models.factory import build_model, count_parameters

    set_seed(int(cfg["experiment"]["seed"]), bool(cfg["experiment"].get("deterministic", False)))
    device = get_device(cfg["experiment"].get("device", "auto"))
    log.info("Environment: %s", {k: v for k, v in environment_report().items() if k in ("python", "torch", "cuda_available", "gpus")})
    if cfg["model"]["name"] != "hybrid" and (cfg["loss"].get("alpha", 0) or cfg["loss"].get("lambda", 0)):
        raise ValueError("Baselines are classification-only: set loss.alpha=0 and loss.lambda=0")

    ds = build_rsna_datasets(cfg, splits=("train", "val"))
    train_loader = make_loader(ds["train"], cfg, train=True)
    val_loader = make_loader(ds["val"], cfg, train=False)
    pw_cfg = cfg["loss"].get("pos_weight", "auto")
    labels = ds["train"].labels
    pos_weight = None
    if cfg["loss"].get("classification") == "weighted_bce":
        pos_weight = float((len(labels) - labels.sum()) / max(labels.sum(), 1)) if pw_cfg == "auto" else float(pw_cfg)
    log.info("Train %d (pos %d) | val %d | pos_weight %s", len(labels), int(labels.sum()), len(ds["val"]), pos_weight)

    image_size = int(cfg["data"]["preprocessing"]["image_size"])
    model = build_model(cfg["model"], image_size=image_size)
    log.info("Model %s params: %s", cfg["checkpoint"]["model_key"], count_parameters(model))
    trainer = Trainer(model, cfg, train_loader, val_loader, device=device, pos_weight=pos_weight)
    result = trainer.fit()
    save_json(result, output_root() / "reports" / f"{cfg['checkpoint']['model_key']}_training_summary.json")
    return result


def main(default_config: str, argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=f"Train using {default_config}")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--resume", action="store_true", help="resume from <model>_last.pth")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. loss.lambda=1.0")
    args = parser.parse_args(argv)
    overrides = parse_cli_overrides(args.overrides)
    cfg = load_config(args.config, overrides)
    if args.resume:
        cfg["training"]["resume"] = True
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["loader"]["batch_size"] = args.batch_size
    if args.device:
        cfg["experiment"]["device"] = args.device
    return run_training(cfg)
