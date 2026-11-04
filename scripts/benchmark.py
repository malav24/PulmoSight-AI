"""Measure training-step speed / memory and inference latency on YOUR hardware (random weights, random data).

    python scripts/benchmark.py                                   # hybrid, batch 4/8/16 at 384 px
    python scripts/benchmark.py --models model_b resnet50 vit_small --batch-sizes 8 16 32
    python scripts/benchmark.py --inference                       # also time full predictor-style inference

Numbers are hardware measurements only - they say nothing about model accuracy. Use them to choose
`loader.batch_size`: pick the largest batch whose peak memory stays comfortably (~15-20 %) below your GPU's.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ml.models.factory import build_model, count_parameters  # noqa: E402
from ml.training.losses import ExplanationGuidedLoss  # noqa: E402
from ml.utils.config import load_config, output_root  # noqa: E402
from ml.utils.runtime import amp_settings, environment_report, get_device  # noqa: E402

CFG = {"model_a": "model_a.yaml", "model_b": "model_b.yaml", "resnet50": "resnet50.yaml",
       "densenet121": "densenet121.yaml", "efficientnet_b0": "efficientnet_b0.yaml", "vit_small": "vit_small.yaml"}


def rss_mb() -> float:
    """Peak resident memory of this process in MB (Windows, Linux and macOS)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        try:
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / 1024 / 1024
        except OSError:
            pass
        return float("nan")
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 1024 / 1024


def train_step_benchmark(key, batch, size, device, amp, steps):
    cfg = load_config(CFG[key])
    model = build_model(dict(cfg["model"], pretrained_backbone=False), size).to(device).train()
    loss_fn = ExplanationGuidedLoss(cfg["loss"], pos_weight=3.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    a = amp_settings(device, amp)
    x = torch.randn(batch, 3, size, size, device=device)
    mask = torch.zeros(batch, 1, size, size, device=device)
    mask[:, :, size // 4: size // 2, size // 4: size // 2] = 1
    b = {"label": torch.ones(batch, device=device), "mask": mask, "has_boxes": torch.ones(batch, dtype=torch.bool, device=device)}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps + 1):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=a["dtype"], enabled=a["enabled"]):
            out = model(x)
        loss_fn(out, b)["loss"].backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i:  # first step is warm-up
            times.append(time.perf_counter() - t0)
    res = {"model": key, "batch": batch, "image_size": size, "device": str(device), "amp": a["enabled"],
           "params_m": round(count_parameters(model)["total"] / 1e6, 1),
           "sec_per_step": round(sum(times) / len(times), 3), "images_per_sec": round(batch * len(times) / sum(times), 1)}
    if device.type == "cuda":
        res["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if key in ("model_a", "model_b"):
        with torch.no_grad():
            res["transformer_tokens"] = int(model(x[:1])["attention"].flatten().numel())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["model_b"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--inference", action="store_true")
    a = ap.parse_args()
    device = get_device(a.device)
    env = {k: v for k, v in environment_report().items() if k in ("python", "torch", "cuda_available", "gpus", "cpu_count")}
    print("environment:", env)
    rows = []
    for key in a.models:
        for bs in a.batch_sizes:
            try:
                r = train_step_benchmark(key, bs, a.image_size, device, not a.no_amp, a.steps)
            except RuntimeError as exc:  # e.g. CUDA out of memory
                r = {"model": key, "batch": bs, "error": str(exc).splitlines()[0]}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            r["process_peak_ram_mb"] = round(rss_mb())
            print(r)
            rows.append(r)
    if a.inference:
        from ml.inference.uncertainty import mc_dropout_predict
        from ml.explainability.gradcam_pp import GradCAMPlusPlus
        for key in a.models:
            cfg = load_config(CFG[key])
            t0 = time.perf_counter()
            m = build_model(dict(cfg["model"], pretrained_backbone=False), a.image_size).to(device).eval()
            build_s = time.perf_counter() - t0
            x = torch.randn(1, 3, a.image_size, a.image_size, device=device)
            with torch.no_grad():
                m(x)
                t0 = time.perf_counter(); m(x); fwd = time.perf_counter() - t0
            t0 = time.perf_counter(); mc_dropout_predict(m, x, 20); mc = time.perf_counter() - t0
            t0 = time.perf_counter(); GradCAMPlusPlus(m)(x); gc = time.perf_counter() - t0
            r = {"model": key, "inference": True, "build_s": round(build_s, 2), "forward_s": round(fwd, 3),
                 "mc_dropout_20_s": round(mc, 2), "gradcam_pp_s": round(gc, 2)}
            print(r)
            rows.append(r)
    out = output_root() / "reports" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"environment": env, "results": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
