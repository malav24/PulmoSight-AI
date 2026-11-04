"""YAML configuration loading with `_base_` inheritance and path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "ml" / "configs"


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, '#' comments. Real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config. A `_base_` key names a parent file (relative to this file)."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        candidate = CONFIG_DIR / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base_name = cfg.pop("_base_", None)
    if base_name:
        base_cfg = load_config(path.parent / base_name)
        cfg = deep_merge(base_cfg, cfg)
    cfg = apply_env_overrides(cfg)          # environment first ...
    if overrides:
        cfg = deep_merge(cfg, overrides)    # ... explicit overrides win
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    return cfg


def checkpoint_root() -> Path:
    """Root holding hybrid/, baselines/, calibration/ (env RFRR_CHECKPOINT_ROOT overrides)."""
    return resolve_path(os.environ.get("RFRR_CHECKPOINT_ROOT", "checkpoints"))


def output_root() -> Path:
    """Root holding metrics/, predictions/, visualizations/, reports/, logs/ (env RFRR_OUTPUT_ROOT)."""
    return resolve_path(os.environ.get("RFRR_OUTPUT_ROOT", "outputs"))


def apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Environment variables RFRR_RSNA_ROOT / RFRR_NIH_ROOT / RFRR_DEVICE override the config."""
    data = cfg.setdefault("data", {})
    if os.environ.get("RFRR_RSNA_ROOT") and "rsna" in data:
        data["rsna"]["root"] = os.environ["RFRR_RSNA_ROOT"]
    if os.environ.get("RFRR_NIH_ROOT") and "nih" in data:
        data["nih"]["root"] = os.environ["RFRR_NIH_ROOT"]
    if os.environ.get("RFRR_DEVICE"):
        cfg.setdefault("experiment", {})["device"] = os.environ["RFRR_DEVICE"]
    return cfg


def resolve_path(p: Optional[str | os.PathLike]) -> Optional[Path]:
    """Resolve a project-relative path to an absolute Path (None passes through)."""
    if p is None:
        return None
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def parse_cli_overrides(pairs) -> Dict[str, Any]:
    """Turn ['training.epochs=2', 'loader.batch_size=4'] into a nested dict (values YAML-parsed)."""
    result: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result
