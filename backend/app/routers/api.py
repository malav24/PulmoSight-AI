"""REST endpoints. All are prefixed with /api (see main.py)."""
from __future__ import annotations

import logging
import platform
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ..catalog import PRIMARY_MODELS
from ..errors import APIError
from ..services import analyses, datasets, experiments, reports
from ..services.analysis import run_analysis, run_explanation, strip_private
from ..services.model_manager import manager
from ..services.uploads import saved_upload
from ..settings import DISCLAIMER, get_settings

log = logging.getLogger("rfrr.api")
router = APIRouter()
STARTED = time.time()


# ------------------------------------------------------------------ schemas
class LoadModelRequest(BaseModel):
    model_key: str = Field(..., examples=["model_b"])
    reload: bool = False


class ExperimentRunRequest(BaseModel):
    kind: str = Field(..., examples=["evaluate"])
    model_key: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ system
@router.get("/health", tags=["system"])
def health() -> Dict[str, Any]:
    info: Dict[str, Any] = {"status": "ok", "app": get_settings().app_name, "version": get_settings().version,
                            "uptime_seconds": round(time.time() - STARTED), "python": platform.python_version(),
                            "disclaimer": DISCLAIMER}
    try:
        import torch
        from ml.utils.runtime import get_device
        info["ml"] = {"available": True, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                      "device": str(get_device(get_settings().device))}
    except Exception as exc:  # noqa: BLE001
        info["ml"] = {"available": False, "error": str(exc)}
        info["status"] = "degraded"
    trained = manager.trained_keys()
    info["models"] = {"trained": trained, "trained_count": len(trained), "active": manager.active_key,
                      "loaded": sorted(manager._predictors)}
    return info


# ------------------------------------------------------------------ models
@router.get("/models", tags=["models"])
def list_models() -> Dict[str, Any]:
    models = manager.list_models()
    return {"models": models, "primary": PRIMARY_MODELS, "active": manager.active_key,
            "any_trained": any(m["trained"] for m in models),
            "message": None if any(m["trained"] for m in models) else "Models not trained yet."}


@router.post("/models/load", tags=["models"])
async def load_model(req: LoadModelRequest) -> Dict[str, Any]:
    model = await run_in_threadpool(manager.load, req.model_key, req.reload)
    return {"loaded": True, "model": model}


# ------------------------------------------------------------------ datasets
@router.get("/datasets/status", tags=["datasets"])
async def dataset_status(refresh: bool = False) -> Dict[str, Any]:
    return await run_in_threadpool(datasets.dataset_status, refresh)


# ------------------------------------------------------------------ analysis
@router.post("/analyze", tags=["analysis"])
async def analyze(file: UploadFile = File(...), model_key: Optional[str] = Form(None)) -> Dict[str, Any]:
    async with saved_upload(file, get_settings()) as up:
        res = await run_in_threadpool(run_analysis, up, model_key)
    return strip_private(res)


@router.post("/explain", tags=["analysis"])
async def explain(file: UploadFile = File(...), model_key: Optional[str] = Form(None),
                  reference_boxes: Optional[str] = Form(None, description="JSON [[x, y, w, h], ...] in original pixels")
                  ) -> Dict[str, Any]:
    async with saved_upload(file, get_settings()) as up:
        res = await run_in_threadpool(run_explanation, up, model_key, reference_boxes)
    return strip_private(res)


@router.get("/analyses", tags=["analysis"])
def recent_analyses(limit: int = Query(10, ge=1, le=100)) -> Dict[str, Any]:
    return {"analyses": analyses.recent(limit), "stored": get_settings().store_analyses}


@router.get("/analyses/{analysis_id}", tags=["analysis"])
def get_analysis(analysis_id: str) -> Dict[str, Any]:
    return analyses.load(analysis_id, with_images=True)


@router.delete("/analyses/{analysis_id}", tags=["analysis"], status_code=204)
def delete_analysis(analysis_id: str) -> None:
    analyses.delete(analysis_id)


# ------------------------------------------------------------------ experiments
@router.get("/experiments/results", tags=["experiments"])
async def experiment_results() -> Dict[str, Any]:
    res = await run_in_threadpool(experiments.experiment_results)
    if not res["available"]:
        res["message"] = "No experiment results available."
    res["jobs"] = experiments.jobs.list()
    res["kinds"] = experiments.EXPERIMENT_KINDS
    return res


@router.post("/experiments/run", tags=["experiments"], status_code=202)
def run_experiment(req: ExperimentRunRequest) -> Dict[str, Any]:
    params = dict(req.params)
    if req.model_key:
        params["model_key"] = req.model_key
    return experiments.jobs.start(req.kind, params)


@router.get("/experiments/jobs", tags=["experiments"])
def list_jobs() -> Dict[str, Any]:
    return {"jobs": experiments.jobs.list()}


@router.get("/experiments/jobs/{job_id}", tags=["experiments"])
def get_job(job_id: str) -> Dict[str, Any]:
    return experiments.jobs.get(job_id)


@router.get("/experiments/files/{folder}/{name}", tags=["experiments"])
def download_file(folder: str, name: str) -> FileResponse:
    p = experiments.resolve_download(f"{folder}/{name}")
    return FileResponse(p, filename=name, media_type="text/csv" if name.endswith(".csv") else "application/json")


# ------------------------------------------------------------------ reports
@router.get("/research/report", tags=["reports"], response_class=HTMLResponse)
def research_report(analysis_id: Optional[str] = None, format: str = Query("html", pattern="^(html|json)$")):
    if analysis_id:
        rec = analyses.load(analysis_id, with_images=True)
        if format == "json":
            return JSONResponse({k: v for k, v in rec.items() if k != "images"})
        return HTMLResponse(reports.analysis_report(rec))
    if format == "json":
        return JSONResponse({"models": manager.list_models()})
    return HTMLResponse(reports.research_report())
