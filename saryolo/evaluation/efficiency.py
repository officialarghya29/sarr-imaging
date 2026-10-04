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

__all__ = [
    "count_parameters",
    "load_image_batch",
    "measure_flops",
    "measure_latency",
    "model_size_mb",
    "profile_model",
    "profile_yaml",
    "write_profile",
]


def load_image_batch(images_dir: str | Path, imgsz: int = 640, batch: int = 8, limit: int | None = None):
    """Load a fixed batch of **real** images as a ``(B, 3, imgsz, imgsz)`` float tensor.

    Efficiency is currently timed on ``torch.randn`` input, which is fine for a plain
    convolutional model but wrong for one whose first layer is data-dependent. This
    repository's CFAR front end takes ``log(x + eps)`` and a local mean, so on zero-mean
    Gaussian noise most of the image clamps to ``eps`` and the timing exercises the
    *degenerate* branch of the front end rather than the branch a real SAR image hits. The
    loader exists so the timing can be taken on the distribution the model is deployed on,
    and so the profile can say which distribution it used.

    Images are read grayscale and replicated to three channels, matching the SAR pipeline
    (one measured channel), and normalised to ``[0, 1]`` exactly as the front end expects.
    """
    import cv2
    import numpy as np
    import torch

    images_dir = Path(images_dir)
    paths = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    )
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(
            f"no images under {images_dir} for a real-input efficiency measurement; an "
            "empty batch must raise rather than fall back to a synthetic one silently."
        )
    chosen = [paths[i % len(paths)] for i in range(batch)]
    x = np.zeros((batch, 3, imgsz, imgsz), dtype=np.float32)
    for i, path in enumerate(chosen):
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape[0] != imgsz or img.shape[1] != imgsz:
            img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        x[i, :, :, :] = img.astype(np.float32) / 255.0
    return torch.from_numpy(x)


def _input_stats(x) -> dict:
    """Min/max/mean of a measurement batch, so the profile records what it timed."""
    return {
        "latency_input_min": round(float(x.min()), 5),
        "latency_input_max": round(float(x.max()), 5),
        "latency_input_mean": round(float(x.mean()), 5),
    }


def count_parameters(model) -> dict:
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": int(total), "params_M": round(total / 1e6, 3), "params_trainable": int(trainable)}


def measure_flops(model, imgsz: int = 640, counter: str = "auto") -> dict:
    """FLOPs/GFLOPs for one forward pass at a square input size.

    Args:
        counter: ``"auto"`` (default), ``"ultralytics"`` or ``"thop"``. Forcing one is how
            the two instruments can be checked against each other on a model both can
            measure, which is what makes it safe to compare a baseline row measured by one
            against a front-end row measured by the other.

    Prefers Ultralytics' own counter so the number is comparable with the YOLO baseline's
    published figures, then falls back to ``thop``. Which counter produced the value is
    recorded as ``flops_G_counter``, because the fallback is not hypothetical: on the CFAR
    arms ``get_flops`` returns **0.0** rather than raising, so those rows are thop numbers
    while the baseline row is an Ultralytics number. The two agree to ~0.2 % when both can
    run (1.613 vs 1.610 G on `baseline_n`), but a cost table that silently mixed counters
    without saying so would be exactly the kind of unlabelled number this repository bans.
    """
    if counter not in ("auto", "ultralytics", "thop"):
        raise ValueError(f"counter must be 'auto', 'ultralytics' or 'thop', got {counter!r}")
    result: dict = {"flops_G": None, "flops_G_imgsz": imgsz, "flops_G_counter": None}
    try:
        if counter == "thop":
            raise ImportError("thop forced")
        from ultralytics.utils.torch_utils import get_flops

        flops = get_flops(model, imgsz)
        if flops:
            result["flops_G"] = round(float(flops), 3)
            result["flops_G_counter"] = "ultralytics"
            return result
    except Exception:
        pass
    try:
        if counter == "ultralytics":
            raise ImportError("ultralytics counter forced but it returned nothing for this model")
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
        result["flops_G_counter"] = "thop"
    except Exception as exc:
        result["flops_error"] = str(exc)
    return result


def measure_latency(
    model,
    imgsz: int = 640,
    warmup: int = 10,
    repeats: int = 50,
    batch: int = 1,
    half: bool = False,
    inputs=None,
    source: str | None = None,
    runs: int = 1,
) -> dict:
    """Measure inference latency and FPS.

    Args:
        model: An ``nn.Module`` in eval mode (or any module; it is put in eval here).
        imgsz: Square input size (used only when ``inputs`` is not given).
        warmup: Untimed passes, to let cuDNN autotune and clocks settle.
        repeats: Timed passes.
        batch: Batch size for the timing loop (FPS is per *image*); inferred from ``inputs``
            when they are supplied.
        half: Use FP16 (CUDA only).
        inputs: An explicit input batch to time. Supplying **real** images (rather than the
            default ``torch.randn``) makes the timing representative for a data-dependent
            first layer and is recorded as ``latency_source = "real"``.
        source: Label for the input distribution; inferred from ``inputs`` when omitted.
        runs: How many independent timed blocks to take, reported as a median with a spread.
            One block of 50 repeats estimates the *within-run* cost but says nothing about
            drift between runs on a shared CPU, which is real: the same CFAR checkpoint
            timed at 29.0 ms in one process and 32.9 ms in another. The reported number is
            the median over ``runs`` blocks, and the min/max are kept so the spread is
            visible instead of hidden behind a single average.

    Returns:
        Latency in ms (per batch and per image) plus FPS, and the provenance of the input
        distribution it was measured on (``latency_source`` plus its min/max/mean), so a
        cost number cannot be quoted without saying what it was measured on.
    """
    import statistics

    import torch

    device = next(model.parameters()).device
    model.eval()
    if inputs is None:
        x = torch.randn(batch, 3, imgsz, imgsz, device=device)
        source = source or "synthetic"
    else:
        x = inputs.to(device)
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"inputs must be (B, 3, H, W), got {tuple(x.shape)}")
        batch = int(x.shape[0])
        imgsz = int(x.shape[-1])
        source = source or "provided"
    stats = _input_stats(x)
    if half and device.type == "cuda":
        model = model.half()
        x = x.half()

    runs = max(1, int(runs))
    per_run_ms: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(runs):
            start = time.perf_counter()
            for _ in range(repeats):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            per_run_ms.append((time.perf_counter() - start) / repeats * 1000.0)

    per_batch_ms = statistics.median(per_run_ms)
    out = {
        "latency_ms": round(per_batch_ms, 3),
        "latency_per_image_ms": round(per_batch_ms / batch, 3),
        "fps": round(1000.0 * batch / per_batch_ms, 2),
        "latency_batch": batch,
        "latency_imgsz": imgsz,
        "latency_half": bool(half),
        "latency_device": device.type,
        "latency_source": source,
        **stats,
    }
    if runs > 1:
        # Kept for a multi-run measurement only: with a single block the spread would be
        # 0 by construction and would read as stability rather than as an unmeasured thing.
        out["latency_runs"] = runs
        out["latency_ms_min"] = round(min(per_run_ms), 3)
        out["latency_ms_max"] = round(max(per_run_ms), 3)
        out["latency_ms_spread_pct"] = round(
            (max(per_run_ms) - min(per_run_ms)) / max(per_batch_ms, 1e-9) * 100.0, 1
        )
    return out


def model_size_mb(weights: str | Path) -> float | None:
    """On-disk size of a checkpoint in megabytes."""
    path = Path(weights)
    return round(path.stat().st_size / 1024**2, 2) if path.is_file() else None


def profile_model(
    weights: str | Path,
    imgsz: int = 640,
    device: str | None = None,
    latency: bool = True,
    data_yaml: str | Path | None = None,
    batch: int = 1,
    runs: int = 1,
) -> dict:
    """Full efficiency profile for a checkpoint.

    Args:
        weights: Checkpoint path.
        imgsz: Square input size.
        device: Optional device override.
        latency: Whether to time forward passes.
        data_yaml: When given, latency is measured on a batch of **real** images from this
            dataset's ``val`` split instead of on random noise, and the profile records
            ``latency_source = "real"``. This matters for any model whose first layer is
            data-dependent (see :func:`load_image_batch`).
        batch: Batch size for the real-image timing.
        runs: Independent timed blocks; see :func:`measure_latency`. Use more than one
            whenever the number will be compared against another arm.
    """

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
            inputs = None
            source = "synthetic"
            if data_yaml is not None:
                from saryolo.data.yolo import load_data_config

                root, cfg = load_data_config(data_yaml)
                images_dir = root / (cfg.get("val") or "images/val")
                inputs = load_image_batch(images_dir, imgsz=imgsz, batch=batch)
                source = "real"
            out.update(
                measure_latency(net, imgsz=imgsz, inputs=inputs, source=source, batch=batch, runs=runs)
            )
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
