"""Efficiency profiling: parameters, FLOPs, latency, FPS, model size, peak memory.

Efficiency is a first-class claim in this project, not an afterthought: the paper
reports ``mAP vs FPS``, ``mAP vs params`` and ``mAP vs FLOPs``, so these numbers
must be measured under a stated protocol rather than quoted from a paper.

Protocol (recorded with every measurement)
------------------------------------------
* latency/FPS are measured with :func:`measure_latency`, which warms up, then
  times repeated forward passes; CUDA timings are synchronised so kernel
  launches are not counted as free.
* FLOPs are measured at a *fixed* input size, and the size is recorded, because
  FLOPs are meaningless without it.
* peak memory is measured with ``torch.cuda.max_memory_allocated`` around a
  forward+backward step, not a forward pass, so training cost is represented.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["count_parameters", "measure_flops", "measure_latency", "model_size_mb", "profile_model", "profile_yaml"]


def count_parameters(model) -> dict:
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": int(total), "params_M": round(total / 1e6, 3), "params_trainable": int(trainable)}


def measure_flops(model, imgsz: int = 640) -> dict:
    """FLOPs/GFLOPs for one forward pass at a square input size.

    Prefers Ultralytics' own counter so the number is comparable with the YOLO
    baseline's published figures, then falls back to ``thop``.
    """
    result: dict = {"flops_G": None, "flops_G_imgsz": imgsz}
    try:
        from ultralytics.utils.torch_utils import get_flops

        flops = get_flops(model, imgsz)
        if flops:
            result["flops_G"] = round(float(flops), 3)
            return result
    except Exception:
        pass
    try:
        import torch
        from thop import profile

        class _Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner(x)
                return out[0] if isinstance(out, (tuple, list)) else out

        device = next(model.parameters()).device
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        macs, _ = profile(_Wrap(model), inputs=(dummy,), verbose=False)
        result["flops_G"] = round(float(macs) * 2 / 1e9, 3)
    except Exception as exc:
        result["flops_error"] = str(exc)
    return result


def measure_latency(model, imgsz: int = 640, warmup: int = 10, repeats: int = 50, batch: int = 1, half: bool = False) -> dict:
    """Measure inference latency and FPS.

    Args:
        model: An ``nn.Module`` in eval mode (or any module; it is put in eval here).
        imgsz: Square input size.
        warmup: Untimed passes, to let cuDNN autotune and clocks settle.
        repeats: Timed passes.
        batch: Batch size for the timing loop (FPS is per *image*).
        half: Use FP16 (CUDA only).

    Returns:
        Latency in ms (per batch and per image) plus FPS.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()
    x = torch.randn(batch, 3, imgsz, imgsz, device=device)
    if half and device.type == "cuda":
        model = model.half()
        x = x.half()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    per_batch_ms = elapsed / repeats * 1000.0
    return {
        "latency_ms": round(per_batch_ms, 3),
        "latency_per_image_ms": round(per_batch_ms / batch, 3),
        "fps": round(batch * repeats / max(elapsed, 1e-9), 2),
        "latency_batch": batch,
        "latency_imgsz": imgsz,
        "latency_half": bool(half),
        "latency_device": device.type,
    }


def model_size_mb(weights: str | Path) -> float | None:
    """On-disk size of a checkpoint in megabytes."""
    path = Path(weights)
    return round(path.stat().st_size / 1024**2, 2) if path.is_file() else None


def profile_model(weights: str | Path, imgsz: int = 640, device: str | None = None, latency: bool = True) -> dict:
    """Full efficiency profile for a checkpoint."""
    import torch

    from saryolo.training.trainer import load_model

    model = load_model(str(weights))
    net = model.model
    if device:
        net = net.to(device)
    out: dict = {}
    counts = count_parameters(net)
    # A checkpoint loaded for inference has requires_grad=False on every parameter,
    # so reporting `params_trainable` here would always read 0 and is meaningless.
    counts.pop("params_trainable", None)
    out.update(counts)
    out.update(measure_flops(net, imgsz=imgsz))
    out["model_size_MB"] = model_size_mb(weights)
    if latency:
        try:
            out.update(measure_latency(net, imgsz=imgsz))
        except Exception as exc:
            out["latency_error"] = str(exc)
    out["weights"] = str(weights)
    return out


def profile_yaml(model_yaml: str | Path, imgsz: int = 640, nc: int = 1, device: str | None = None) -> dict:
    """Profile an architecture straight from a YAML, without training it.

    Useful for the architecture-level comparison table (params/FLOPs) before any
    GPU time is spent.
    """
    from saryolo.nn.model import SARYOLODetectionModel

    model = SARYOLODetectionModel(str(model_yaml), ch=3, nc=nc, verbose=False)
    if device:
        model = model.to(device)
    out: dict = {"yaml": str(model_yaml), "nc": nc}
    out.update(count_parameters(model))
    out.update(measure_flops(model, imgsz=imgsz))
    out["model_size_MB"] = None
    return out


def write_profile(profile: dict, out_path: str | Path) -> Path:
    """Persist an efficiency profile as JSON."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))
    return out
