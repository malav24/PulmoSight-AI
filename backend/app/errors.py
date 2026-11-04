"""Structured API errors: {"error": {"code", "message", "details"}}."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException


class APIError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str, details: Optional[Any] = None):
        super().__init__(status_code=status_code, detail={"code": code, "message": message, "details": details})
        self.code, self.message, self.details = code, message, details


def model_not_trained(model_key: str, relative_path: str) -> APIError:
    return APIError(
        409, "model_not_trained",
        f"No trained checkpoint is available for '{model_key}'. Train the model (see docs/TRAINING_GUIDE.md) or "
        f"place a checkpoint at {relative_path}, then load it from the Model Laboratory.",
        {"model_key": model_key, "expected_checkpoint": relative_path})
