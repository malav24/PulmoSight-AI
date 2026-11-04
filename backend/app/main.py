"""FastAPI application.

Run from the project root:
    uvicorn backend.app.main:app --reload --port 8000
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # make the Stage-1 `ml` package importable
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from .routers.api import router  # noqa: E402
from .services.experiments import jobs  # noqa: E402
from .services.model_manager import manager  # noqa: E402
from .settings import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("rfrr.api")

settings = get_settings()
app = FastAPI(title="Right for the Right Reason API", version=settings.version,
              description="Research API for explanation-guided pneumonia detection. Not a medical device.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
app.include_router(router, prefix="/api")

# After an evaluation/training job, drop cached models so new calibration/OOD files are used.
jobs.on_finish = lambda job: manager.invalidate()


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details}})


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return _error(exc.status_code, exc.detail["code"], exc.detail["message"], exc.detail.get("details"))
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    return _error(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e.get("loc", [])), "message": e.get("msg")} for e in exc.errors()]
    return _error(422, "validation_error", "The request is missing fields or has invalid values.", details)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal_error", "An unexpected error occurred while processing the request. "
                                         "Details were written to the server log.")


@app.middleware("http")
async def access_log(request: Request, call_next):
    import time
    t0 = time.time()
    response = await call_next(request)
    if request.url.path.startswith("/api"):
        log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code,
                 (time.time() - t0) * 1000)
    return response


@app.get("/", include_in_schema=False)
def root():
    return {"app": settings.app_name, "docs": "/docs", "api": "/api/health"}
