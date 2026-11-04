"""Persist analysis results (for 'recent analyses' and reports).

Privacy: only the 384-px preprocessed image, map layers and the model outputs are stored, under
outputs/analyses/<id>/. The original upload is never kept; the input is referenced by a SHA-256 prefix.
The uploaded file name is returned to the browser but NOT persisted (it may contain identifiers).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.utils.config import output_root

from ..errors import APIError
from ..settings import get_settings

ID_RE = re.compile(r"^[0-9a-f]{32}$")


def analyses_dir() -> Path:
    return output_root() / "analyses"


def _check_id(analysis_id: str) -> None:
    if not ID_RE.match(analysis_id or ""):
        raise APIError(400, "invalid_analysis_id", "Analysis id must be 32 hexadecimal characters.")


def save(record: Dict[str, Any], images: Dict[str, str]) -> None:
    if not get_settings().store_analyses:
        return
    d = analyses_dir() / record["analysis_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "images.json").write_text(json.dumps(images))
    (d / "record.json").write_text(json.dumps(record, default=str))
    _prune()


def _prune() -> None:
    limit = get_settings().max_stored_analyses
    items = sorted((p for p in analyses_dir().glob("*/record.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in items[limit:]:
        shutil.rmtree(old.parent, ignore_errors=True)


def load(analysis_id: str, with_images: bool = False) -> Dict[str, Any]:
    _check_id(analysis_id)
    d = analyses_dir() / analysis_id
    if not (d / "record.json").exists():
        raise APIError(404, "analysis_not_found", "No stored analysis with this id.")
    rec = json.loads((d / "record.json").read_text())
    if with_images and (d / "images.json").exists():
        rec["images"] = json.loads((d / "images.json").read_text())
    return rec


def recent(limit: int = 10) -> List[Dict[str, Any]]:
    out = []
    if not analyses_dir().exists():
        return out
    files = sorted(analyses_dir().glob("*/record.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for f in files:
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        out.append({k: r.get(k) for k in ("analysis_id", "created_at", "model_key", "model_name", "input_reference")}
                   | {"probability": r.get("prediction", {}).get("probability"),
                      "outcome": r.get("prediction", {}).get("outcome"),
                      "abstained": r.get("prediction", {}).get("abstained")})
    return out


def delete(analysis_id: str) -> None:
    _check_id(analysis_id)
    d = analyses_dir() / analysis_id
    if not d.exists():
        raise APIError(404, "analysis_not_found", "No stored analysis with this id.")
    shutil.rmtree(d)
