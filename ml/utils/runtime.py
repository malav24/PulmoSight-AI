"""Seeds, device selection and environment reporting."""
from __future__ import annotations

import os
import platform
import random
import shutil
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            pass
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn giving each worker a reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preference: str = "auto") -> torch.device:
    preference = (preference or "auto").lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(preference)


def amp_settings(device: torch.device, enabled: bool) -> Dict[str, Any]:
    """Return autocast dtype / scaler usage. AMP is only enabled on CUDA."""
    if not enabled or device.type != "cuda":
        return {"enabled": False, "dtype": torch.float32, "use_scaler": False}
    if torch.cuda.is_bf16_supported():
        return {"enabled": True, "dtype": torch.bfloat16, "use_scaler": False}
    return {"enabled": True, "dtype": torch.float16, "use_scaler": True}


def environment_report(path: str = ".") -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(),
    }
    for mod in ["torchvision", "timm", "pydicom", "albumentations", "cv2", "sklearn", "pandas", "numpy", "scipy"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = "MISSING"
    try:
        total, used, free = shutil.disk_usage(path)
        info["disk_free_gb"] = round(free / 1e9, 1)
    except Exception:  # pragma: no cover
        info["disk_free_gb"] = None
    return info
