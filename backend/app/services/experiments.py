"""Experiment results (read from real output files only) and background experiment jobs.

Jobs run as separate OS processes (python -m ...), so a long evaluation or training run never blocks
API inference requests. Only one job runs at a time. Nothing is started automatically.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.utils.config import PROJECT_ROOT, output_root

from ..catalog import MODEL_CATALOG, TRAIN_MODULES
from ..errors import APIError

log = logging.getLogger("rfrr.api.experiments")
DOWNLOAD_DIRS = ("predictions", "reports", "logs", "metrics")
DOWNLOAD_SUFFIXES = (".csv", ".json")


# ------------------------------------------------------------------ results
def _read_json(p: Path) -> Optional[Any]:
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def _clean(o: Any) -> Any:
    """NaN/inf are not valid JSON - convert to None."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def _roc_points(df: pd.DataFrame, max_points: int = 120) -> Optional[List[Dict[str, float]]]:
    from sklearn.metrics import roc_curve

    if df["label"].nunique() < 2:
        return None
    fpr, tpr, _ = roc_curve(df["label"], df["prob"])
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]


def _reliability(df: pd.DataFrame, col: str) -> List[Dict[str, Any]]:
    from ml.evaluation.calibration import reliability_curve

    rc = reliability_curve(df["label"].to_numpy(), df[col].to_numpy(), 10)
    return [{"bin": i, "confidence": c, "observed": o, "count": n}
            for i, (c, o, n) in enumerate(zip(rc["mean_confidence"], rc["observed_frequency"], rc["count"])) if n]


def training_log(key: str) -> Optional[List[Dict[str, Any]]]:
    p = output_root() / "logs" / f"{key}_training_log.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None
    cols = [c for c in ("epoch", "train_loss", "val_loss", "val_auc", "train_loss_cls", "train_loss_loc",
                        "train_loss_attn", "val_dice_pos", "val_attn_mass_in_box") if c in df]
    return _clean(df[cols].replace({np.nan: None}).to_dict("records"))


def experiment_results() -> Dict[str, Any]:
    root = output_root()
    models: Dict[str, Any] = {}
    for key in MODEL_CATALOG:
        metrics = _read_json(root / "metrics" / f"{key}_metrics.json")
        log_rows = training_log(key)
        if metrics is None and log_rows is None:
            continue
        entry: Dict[str, Any] = {"key": key, "name": MODEL_CATALOG[key]["name"], "metrics": metrics,
                                 "training_log": log_rows}
        for ds, fname in (("rsna_test", f"{key}_rsna_test.csv"), ("nih", f"{key}_nih.csv")):
            p = root / "predictions" / fname
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    entry.setdefault("curves", {})[ds] = {
                        "roc": _roc_points(df),
                        "reliability_raw": _reliability(df, "prob"),
                        "reliability_calibrated": _reliability(df, "prob_calibrated") if "prob_calibrated" in df else None,
                    }
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not read %s: %s", p, exc)
        models[key] = entry
    comparisons = {p.stem.replace("model_comparison_", ""): _read_json(p)
                   for p in sorted((root / "reports").glob("model_comparison_*.json"))} if (root / "reports").exists() else {}
    files = list_files()
    return _clean({"available": bool(models), "models": models,
                   "ablation_summary": _read_json(root / "reports" / "ablation_summary.json"),
                   "comparisons": comparisons, "files": files,
                   "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


def list_files() -> List[Dict[str, Any]]:
    out = []
    for d in DOWNLOAD_DIRS:
        base = output_root() / d
        if not base.exists():
            continue
        for p in sorted(base.glob("*")):
            if p.is_file() and p.suffix in DOWNLOAD_SUFFIXES:
                out.append({"name": f"{d}/{p.name}", "size_kb": round(p.stat().st_size / 1024, 1)})
    return out


def resolve_download(name: str) -> Path:
    parts = name.split("/")
    if len(parts) != 2 or parts[0] not in DOWNLOAD_DIRS or ".." in name or not parts[1].endswith(DOWNLOAD_SUFFIXES):
        raise APIError(400, "invalid_file", "Only CSV/JSON result files can be downloaded.")
    p = output_root() / parts[0] / parts[1]
    if not p.is_file():
        raise APIError(404, "file_not_found", "Result file not found.")
    return p


# ------------------------------------------------------------------ jobs
EXPERIMENT_KINDS = {
    "evaluate": "Evaluate a trained model (RSNA test; optional NIH, stress tests, OOD statistics)",
    "compare_models": "Paired DeLong comparison of evaluated models",
    "ablation_summary": "Summarise the three-arm ablation from evaluated metrics",
    "prepare_rsna": "Create RSNA patient-level splits and dataset statistics",
    "prepare_nih": "Prepare NIH external-evaluation subset and statistics",
    "train": "Train a model (long-running; requires explicit confirmation)",
}


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.on_finish = None  # callback (e.g. invalidate model cache)

    def _command(self, kind: str, p: Dict[str, Any]) -> List[str]:
        py = sys.executable
        model = p.get("model_key")
        if kind in ("evaluate", "train") and model not in MODEL_CATALOG:
            raise APIError(422, "invalid_parameters", "A valid model_key is required for this experiment.")
        if kind == "evaluate":
            cmd = [py, "-m", "ml.evaluation.evaluate", "--model", model,
                   "--explain-n", str(int(p.get("explain_n", 32)))]
            for flag in ("nih", "stress", "fit_ood"):
                if p.get(flag):
                    cmd.append("--" + flag.replace("_", "-"))
            return cmd
        if kind == "compare_models":
            models = [m for m in p.get("models", ["model_b", "model_a"]) if m in MODEL_CATALOG]
            if len(models) < 2:
                raise APIError(422, "invalid_parameters", "Select at least two evaluated models.")
            ds = p.get("dataset", "rsna_test")
            if ds not in ("rsna_test", "nih"):
                raise APIError(422, "invalid_parameters", "dataset must be rsna_test or nih.")
            return [py, "scripts/compare_models.py", "--dataset", ds, "--models", *models]
        if kind == "ablation_summary":
            return [py, "scripts/run_ablation.py", "--summarize"]
        if kind == "prepare_rsna":
            return ([py, "scripts/prepare_rsna.py"] + (["--cache"] if p.get("cache") else [])
                    + (["--download"] if p.get("download") else []))
        if kind == "prepare_nih":
            return [py, "scripts/prepare_nih.py"]
        if kind == "train":
            if not p.get("confirm"):
                raise APIError(422, "confirmation_required",
                               "Training is long-running and GPU-intensive; resend with confirm=true to start it.")
            cmd = [py, "-m", TRAIN_MODULES[model]]
            if p.get("epochs"):
                cmd += ["--epochs", str(int(p["epochs"]))]
            if p.get("resume"):
                cmd.append("--resume")
            return cmd
        raise APIError(422, "unknown_experiment", f"Unknown experiment kind '{kind}'.",
                       {"kinds": EXPERIMENT_KINDS})

    def start(self, kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
        cmd = self._command(kind, params or {})
        with self._lock:
            if any(j["status"] in ("queued", "running") for j in self.jobs.values()):
                raise APIError(409, "job_running", "Another experiment is still running. Wait for it to finish.")
            job_id = uuid.uuid4().hex
            log_dir = output_root() / "logs" / "jobs"
            log_dir.mkdir(parents=True, exist_ok=True)
            job = {"job_id": job_id, "kind": kind, "params": params, "status": "running",
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished_at": None, "return_code": None,
                   "log_file": str(log_dir / f"{job_id}.log"),
                   "command": " ".join(c if c != sys.executable else "python" for c in cmd)}
            self.jobs[job_id] = job
        fh = open(job["log_file"], "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._wait, args=(job_id, proc, fh), daemon=True).start()
        log.info("Started job %s: %s", job_id, job["command"])
        return self.public(job)

    def _wait(self, job_id: str, proc: subprocess.Popen, fh) -> None:
        rc = proc.wait()
        fh.close()
        job = self.jobs[job_id]
        job.update(status="succeeded" if rc == 0 else "failed", return_code=rc,
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        log.info("Job %s finished with %s", job_id, rc)
        if self.on_finish:
            try:
                self.on_finish(job)
            except Exception:  # noqa: BLE001
                log.exception("on_finish callback failed")

    def public(self, job: Dict[str, Any], tail: int = 40) -> Dict[str, Any]:
        out = {k: v for k, v in job.items() if k != "log_file"}
        try:
            lines = Path(job["log_file"]).read_text(encoding="utf-8", errors="replace").splitlines()
            out["log_tail"] = lines[-tail:]
        except OSError:
            out["log_tail"] = []
        return out

    def get(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self.jobs:
            raise APIError(404, "job_not_found", "Unknown job id.")
        return self.public(self.jobs[job_id])

    def list(self) -> List[Dict[str, Any]]:
        return [self.public(j, tail=5) for j in sorted(self.jobs.values(), key=lambda j: j["started_at"], reverse=True)]


jobs = JobManager()
