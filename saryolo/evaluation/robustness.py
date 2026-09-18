"""Robustness evaluation under controlled SAR-relevant degradations.

Motivation
----------
A detector that wins on clean scenes but collapses under heavy speckle is not
useful for SAR. This module applies *controlled, documented* degradations to a
copy of the validation split, re-evaluates, and reports the degradation for both
the baseline and SAR-YOLO.

Design choices that keep the comparison fair
--------------------------------------------
* Corruptions are deterministic given a seed, so baseline and proposed model face
  byte-identical inputs.
* Each corruption has a single scalar severity parameter, and the applied
  parameter is recorded, so the x-axis of the robustness figure is meaningful.
* Only the *images* are degraded; ground truth is untouched, because a physical
  degradation changes the sensor signal, not where the targets are.
* Speckle is applied multiplicatively (the physically correct model for SAR),
  not as additive Gaussian noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Corruption", "CORRUPTIONS", "apply_corruption", "build_corrupted_split", "robustness_sweep"]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class Corruption:
    """One named, parameterised degradation."""

    name: str
    param_name: str
    severities: tuple[float, ...]
    description: str
    physical: bool = True


CORRUPTIONS: dict[str, Corruption] = {
    "speckle": Corruption(
        "speckle", "looks", (16.0, 8.0, 4.0, 2.0, 1.0),
        "Multiplicative Gamma speckle; lower 'looks' = more severe.", physical=True,
    ),
    "low_contrast": Corruption(
        "low_contrast", "gamma", (1.0, 1.6, 2.2, 3.0, 4.0),
        "Power-law contrast compression toward mid-grey.", physical=True,
    ),
    "blur": Corruption(
        "blur", "sigma", (0.0, 0.8, 1.6, 2.4, 3.2),
        "Gaussian blur, as from reduced resolution or defocus.", physical=True,
    ),
    "low_resolution": Corruption(
        "low_resolution", "scale", (1.0, 0.75, 0.5, 0.35, 0.25),
        "Downsample then upsample, degrading spatial resolution.", physical=True,
    ),
    "clutter": Corruption(
        "clutter", "rate", (0.0, 4.0, 8.0, 16.0, 32.0),
        "Inject bright non-target blobs as hard background clutter.", physical=True,
    ),
    "low_snr": Corruption(
        "low_snr", "noise", (0.0, 0.10, 0.20, 0.35, 0.50),
        "Additive noise at a fraction of the dynamic range.", physical=False,
    ),
}


def apply_corruption(img: np.ndarray, name: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Apply one corruption to a grayscale/colour image (float in [0,1] or uint8)."""
    import cv2

    arr = img.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    if name == "speckle":
        looks = max(float(severity), 0.5)
        arr = arr * rng.gamma(shape=looks, scale=1.0 / looks, size=arr.shape).astype(np.float32)
    elif name == "low_contrast":
        g = max(float(severity), 1e-6)
        arr = np.power(np.clip(arr, 0, 1), g)
        arr = 0.15 + 0.7 * arr  # compress into a narrow band: low contrast
    elif name == "blur":
        sigma = float(severity)
        if sigma > 0:
            arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)
    elif name == "low_resolution":
        scale = float(severity)
        if scale < 1.0:
            h, w = arr.shape[:2]
            small = cv2.resize(arr, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)
            arr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif name == "clutter":
        rate = float(severity)
        h, w = arr.shape[:2]
        for _ in range(int(rng.poisson(rate))):
            bw = int(rng.integers(max(h // 40, 3), max(h // 12, 4)))
            bh = int(rng.integers(max(h // 40, 3), max(h // 12, 4)))
            x = int(rng.integers(0, max(w - bw, 1)))
            y = int(rng.integers(0, max(h - bh, 1)))
            arr[y:y + bh, x:x + bw] += float(rng.uniform(0.25, 0.6))
    elif name == "low_snr":
        sd = float(severity)
        if sd > 0:
            arr = arr + rng.normal(0, sd, size=arr.shape).astype(np.float32)
    elif name == "identity":
        pass
    else:
        raise KeyError(f"Unknown corruption {name!r}. Known: {sorted(CORRUPTIONS)}")
    return _to_uint8(np.clip(arr, 0.0, 1.0))


def build_corrupted_split(
    images_dir: str | Path,
    out_root: str | Path,
    name: str,
    severity: float,
    seed: int = 0,
    labels_dir: str | Path | None = None,
    limit: int | None = None,
) -> Path:
    """Write a corrupted copy of an image split, with labels symlinked in place.

    Ground truth is deliberately *not* modified: degradation affects the sensor
    signal, not target locations.

    Returns:
        Path to the corrupted images directory (labels sit alongside).
    """
    import cv2
    import os

    images_dir, out_root = Path(images_dir), Path(out_root)
    dest = out_root / name / str(severity) / "images" / "val"
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    if limit:
        images = images[:limit]
    for path in images:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out = apply_corruption(img, name, severity, rng)
        cv2.imwrite(str(dest / path.name), out)

    labels_src = Path(labels_dir) if labels_dir else Path(str(images_dir).replace("images", "labels", 1))
    labels_dest = out_root / name / str(severity) / "labels" / "val"
    labels_dest.mkdir(parents=True, exist_ok=True)
    if labels_src.exists():
        for label in sorted(labels_src.glob("*.txt")):
            target = labels_dest / label.name
            if not target.exists():
                try:
                    os.symlink(label.resolve(), target)
                except OSError:
                    target.write_text(label.read_text())
    return dest


def robustness_sweep(
    weights: str | Path,
    data_yaml: str | Path,
    out_dir: str | Path = "results/robustness",
    imgsz: int = 640,
    corruptions: tuple[str, ...] = ("speckle", "low_contrast", "blur", "low_resolution", "clutter"),
    seed: int = 0,
    limit: int | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate a checkpoint under the full corruption sweep.

    Returns:
        ``{corruption: {severity: {mAP50: ..., mAP50_95: ...}}}`` plus a
        ``clean`` baseline entry. Only measured values appear; a corruption that
        fails is recorded with an ``error`` key rather than being dropped.
    """
    from saryolo.data.yolo import load_data_config

    from .metrics import evaluate_detections

    data_yaml, out_dir = Path(data_yaml), Path(out_dir)
    root, cfg = load_data_config(data_yaml)
    images_dir = root / (cfg.get("val") or "images/val")

    results: dict = {"clean": {"mAP50": None, "mAP50_95": None}, "corruptions": {}}

    clean = evaluate_detections(weights, data_yaml, imgsz=imgsz, device=device, out_dir=out_dir / "preds_clean")
    results["clean"] = {"mAP50": clean["mAP50"], "mAP50_95": clean["mAP50_95"]}

    for name in corruptions:
        spec = CORRUPTIONS[name]
        per_severity: dict = {}
        for severity in spec.severities:
            try:
                build_corrupted_split(images_dir, out_dir / "data", name, severity, seed=seed, limit=limit)
                corrupted_yaml = _write_corrupted_yaml(cfg, out_dir / "data" / name / str(severity), name, severity)
                metrics = evaluate_detections(
                    weights, corrupted_yaml, imgsz=imgsz, device=device,
                    out_dir=out_dir / "preds" / f"{name}_{severity}",
                )
                per_severity[str(severity)] = {"mAP50": metrics["mAP50"], "mAP50_95": metrics["mAP50_95"]}
            except Exception as exc:
                per_severity[str(severity)] = {"error": str(exc)}
        results["corruptions"][name] = per_severity

    (out_dir / "robustness.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def _write_corrupted_yaml(base_cfg: dict, root: Path, name: str, severity: float) -> Path:
    """Write a data.yaml pointing at a corrupted copy of the val split."""
    cfg = dict(base_cfg)
    cfg["path"] = str(root.resolve())
    cfg["train"] = "images/val"  # only used for evaluation here
    cfg["val"] = "images/val"
    out = root / f"data_{name}_{severity}.yaml"
    out.write_text(json.dumps(cfg, indent=2))
    return out
