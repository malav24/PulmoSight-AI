"""Runtime settings, read from environment variables (see ../.env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import ml.utils.config  # noqa: F401  (loads the project .env before settings are read)
from functools import lru_cache
from typing import List


def _env_list(name: str, default: str) -> List[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = "RIGHT FOR THE RIGHT REASON"
    version: str = "0.2.0-stage2"
    max_upload_mb: float = float(os.environ.get("RFRR_MAX_UPLOAD_MB", "25"))
    max_image_pixels: int = int(os.environ.get("RFRR_MAX_IMAGE_PIXELS", str(40_000_000)))
    min_image_side: int = 64
    allowed_extensions: tuple = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")
    cors_origins: List[str] = field(default_factory=lambda: _env_list(
        "RFRR_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"))
    default_model: str = os.environ.get("RFRR_DEFAULT_MODEL", "model_b")
    device: str = os.environ.get("RFRR_DEVICE", "auto")
    mc_samples: int = int(os.environ.get("RFRR_MC_DROPOUT_SAMPLES", "20"))
    abstain_std: float = float(os.environ.get("RFRR_ABSTAIN_UNCERTAINTY", "0.15"))
    store_analyses: bool = os.environ.get("RFRR_STORE_ANALYSES", "true").lower() in ("1", "true", "yes")
    max_stored_analyses: int = int(os.environ.get("RFRR_MAX_STORED_ANALYSES", "200"))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()


DISCLAIMER = ("For research and educational purposes only. This system is not a medical device and must not be "
              "used as a substitute for professional radiological interpretation or clinical diagnosis.")
