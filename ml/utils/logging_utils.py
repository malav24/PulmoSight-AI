"""Lightweight CSV / JSON experiment logging."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List


def get_logger(name: str = "rfrr", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class CSVLogger:
    """Append rows to a CSV; the header grows if new keys appear (file is rewritten)."""

    def __init__(self, path: str | Path, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []
        if resume and self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                self.rows = list(csv.DictReader(fh))
        elif self.path.exists():
            self.path.unlink()

    def log(self, row: Dict[str, Any]) -> None:
        self.rows.append({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        keys: List[str] = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:
            pass
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return path
