"""Gain-collapse diagnostic for the ratio-space CFAR front end.

Why this exists
---------------
The architecture proposal (`docs/architecture_proposals.md` §2) rests on one
mechanism claim: the learned gain ``g`` is a **per-pixel** decision about how much
of the radar statistic to apply. A module whose gain collapsed to a single scalar
would still produce a plausible output, still train, and still report a mAP -- and
would make the claim empty while every other test passed. The proposal named this
failure mode explicitly and named the diagnostic that rules it out, so the
diagnostic is built here rather than argued in prose.

What is measured
----------------
For each held-out image the front end's gain map ``g`` (shape ``(1, H, W)``) is
read directly through :meth:`RatioSpaceCFARFrontEnd.gain_map`. From those maps:

* ``mean_abs_gain`` -- how hard the front end is modulating on average. Near zero
  means the representation was learned to be a near-identity.
* ``within_image_std_mean`` -- the average spread of ``g`` *inside* one image. If
  this is the only non-zero term, the gain is a spatially varying decision.
* ``between_image_std`` -- the spread of the per-image means. If this dominates,
  the gain is a per-scene scalar rather than a per-pixel decision.
* ``active_fraction`` -- fraction of pixels whose ``|g|`` exceeds ``active``. This
  is the direct answer to "is the per-pixel path being used".
* ``collapsed`` -- ``True`` when the pooled standard deviation is at the floor,
  i.e. the gain is effectively constant and the statistic is being applied
  uniformly.

Synthetic acquisition shifts
----------------------------
The same maps are collected after a *labelled synthetic* acquisition shift
(radiometric gain, contrast compression, speckle) so the gain's response to a
change of acquisition can be read off directly. These shifts reuse the tested
:func:`saryolo.evaluation.robustness.apply_corruption` functions; the shift is
synthetic and is named as such in the output, so it cannot be mistaken for a
second real sensor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "find_front_end",
    "gain_maps",
    "summarise_gain",
    "diagnose_gain",
    "DEFAULT_SHIFTS",
    "COLLAPSE_STD_FLOOR",
]

#: A pooled gain standard deviation at or below this is treated as constant.
COLLAPSE_STD_FLOOR = 1e-4

#: Synthetic acquisition shifts applied in the pilot: ``(name, severity)`` where the
#: name is a key of :data:`saryolo.evaluation.robustness.CORRUPTIONS`. ``identity``
#: is the unshifted reference. Deliberately small and named, so the output is a
#: measurement of the front end, not a benchmark of a second sensor.
DEFAULT_SHIFTS: tuple[tuple[str, float], ...] = (
    ("identity", 0.0),
    ("brightness", 0.5),
    ("brightness", 0.2),
    ("low_contrast", 3.0),
    ("speckle", 2.0),
)


def find_front_end(model):
    """Return the first :class:`RatioSpaceCFARFrontEnd` in a loaded model, or ``None``.

    Accepts either an Ultralytics facade (``model.model``) or the bare ``nn.Module``.
    """
    from saryolo.nn.modules import RatioSpaceCFARFrontEnd

    net = getattr(model, "model", model)
    for module in net.modules():
        if isinstance(module, RatioSpaceCFARFrontEnd):
            return module
    return None


def _read_gray(paths: list[Path], imgsz: int) -> np.ndarray:
    """Load images as a ``(N, 1, imgsz, imgsz)`` float array in ``[0, 1]``.

    Grayscale on purpose: a SAR magnitude product is one measured channel, and the
    statistic is computed on the channel mean, so replicating a single channel is a
    faithful front-end input.
    """
    import cv2

    out = np.zeros((len(paths), 1, imgsz, imgsz), dtype=np.float32)
    for i, path in enumerate(paths):
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape[0] != imgsz or img.shape[1] != imgsz:
            img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        out[i, 0] = img.astype(np.float32) / 255.0
    return out


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _apply_shift(images: np.ndarray, name: str, severity: float, seed: int) -> np.ndarray:
    """Apply a named synthetic acquisition shift to a ``(N, 1, H, W)`` batch."""
    from .robustness import apply_corruption

    out = np.zeros_like(images)
    for i in range(images.shape[0]):
        gray = _to_uint8(images[i, 0])
        shifted = apply_corruption(gray, name, severity, np.random.default_rng(seed + i))
        out[i, 0] = shifted.astype(np.float32) / 255.0
    return out


def gain_maps(front, images: np.ndarray) -> np.ndarray:
    """Collect the per-pixel gain for each image as a ``(N, H*W)`` array.

    The front end is run in eval mode with gradients off: the question is what the
    trained representation *does*, not what it could be trained to do.
    """
    import torch

    front.eval()
    with torch.no_grad():
        x = torch.from_numpy(images)
        gains = front.gain_map(x)
    return gains.reshape(gains.shape[0], -1).cpu().numpy()


def summarise_gain(gains: np.ndarray, active: float = 0.05) -> dict:
    """Turn ``(N, P)`` gain maps into the collapse report.

    ``active`` is the ``|g|`` above which a pixel counts as an active decision;
    ``0.05`` is one twentieth of the module's full ``[-1, 1]`` range, so a pixel
    below it is being left essentially untouched.
    """
    gains = np.asarray(gains, dtype=np.float64)
    per_image_std = gains.std(axis=1)
    per_image_mean = gains.mean(axis=1)
    pooled_std = float(gains.std())
    between_image_std = float(per_image_mean.std())
    within_image_std = float(per_image_std.mean())
    report = {
        "images": int(gains.shape[0]),
        "pixels_per_image": int(gains.shape[1]),
        "mean_gain": round(float(gains.mean()), 6),
        "mean_abs_gain": round(float(np.abs(gains).mean()), 6),
        "pooled_std": round(pooled_std, 6),
        "within_image_std_mean": round(within_image_std, 6),
        "between_image_std": round(between_image_std, 6),
        "min_gain": round(float(gains.min()), 6),
        "max_gain": round(float(gains.max()), 6),
        "active_threshold": float(active),
        "active_fraction": round(float((np.abs(gains) > active).mean()), 6),
        "collapsed": bool(pooled_std <= COLLAPSE_STD_FLOOR),
    }
    # A gain that varies only between scenes is not the per-pixel decision the
    # design claims, even though its pooled std is large. It is the case where each
    # image has a constant gain but different images disagree: no pixel inside a
    # scene differs from its neighbours, so the per-pixel path is unused.
    report["scene_scalar_dominated"] = bool(
        within_image_std <= COLLAPSE_STD_FLOOR and between_image_std > COLLAPSE_STD_FLOOR
    )
    return report


def diagnose_gain(
    weights: str | Path,
    data_yaml: str | Path,
    out_dir: str | Path = "results/gain",
    imgsz: int = 320,
    limit: int | None = None,
    shifts: tuple[tuple[str, float], ...] = DEFAULT_SHIFTS,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    """Measure the trained front end's gain on the held-out split, per shift.

    Returns a JSON-serialisable dict with a ``clean`` report, one report per named
    synthetic shift, and the provenance needed to reproduce it. Writes
    ``<out_dir>/gain.json``.
    """
    from saryolo.data.yolo import load_data_config
    from saryolo.training.trainer import load_model

    data_yaml, out_dir = Path(data_yaml), Path(out_dir)
    root, cfg = load_data_config(data_yaml)
    images_dir = root / (cfg.get("val") or "images/val")
    paths = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    )
    if limit:
        paths = paths[:limit]

    model = load_model(str(weights))
    if device:
        getattr(model, "model", model).to(device)
    front = find_front_end(model)
    if front is None:
        raise ValueError(
            f"{weights} carries no RatioSpaceCFARFrontEnd; the gain diagnostic does not apply "
            "to a stock detector (and reporting a constant zero gain for one would be a lie)."
        )

    clean_images = _read_gray(paths, imgsz)
    result: dict = {
        "weights": str(weights),
        "data": str(data_yaml),
        "imgsz": imgsz,
        "limit": limit,
        "mode": front.mode,
        "scales": list(front.scales),
        "synthetic": True,
        "note": (
            "Gain statistics of the trained front end. The shift rows are labelled synthetic "
            "acquisition shifts applied to real HRSID chips; they are not a second sensor."
        ),
        "clean": summarise_gain(gain_maps(front, clean_images)),
        "shifts": {},
    }
    for name, severity in shifts:
        if name == "identity":
            continue
        shifted = _apply_shift(clean_images, name, severity, seed)
        result["shifts"][f"{name}_{severity}"] = summarise_gain(gain_maps(front, shifted))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gain.json").write_text(json.dumps(result, indent=2))
    return result
