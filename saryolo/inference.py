"""Single-image inference for SARVO: load once, validate, predict, return one stable format.

Why this module exists
----------------------
The repository could already *evaluate* a checkpoint over a dataset
(:func:`saryolo.evaluation.metrics.evaluate_detections`), but it had no clean single-image
entry point -- so every consumer that wanted "run the model on this picture" would have had to
rebuild the model wiring for itself, and any demo would then be running a *different* graph
from the one the results were measured on. This module is that entry point: it loads the
checkpoint through the same facade the evaluation and training paths use, decodes and validates
the input exactly as the SAR pipeline expects (one measured channel replicated to three), runs
one forward pass, and returns detections in one documented shape.

Two rules shape it:

* **No fabricated output.** If the weights are missing or unreadable, the loader raises a
  :class:`CheckpointError` naming the path; it never falls back to a randomly-initialised model,
  because predictions from untrained weights look exactly like real predictions.
* **Errors are actionable.** Bad uploads raise :class:`InvalidImageError` with a message that
  says what to change, so a UI can show it verbatim without inventing guidance.

The returned :class:`Detection` objects are the *same* type the evaluator uses
(:class:`saryolo.evaluation.metrics.Detection`), so a box shown in a demo is the box the metric
code would score -- one representation, not two.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from saryolo.evaluation.metrics import Detection

__all__ = [
    "ALLOWED_SUFFIXES",
    "DEFAULT_WEIGHT_CANDIDATES",
    "MAX_SIDE",
    "MAX_UPLOAD_BYTES",
    "MIN_SIDE",
    "CheckpointError",
    "InferenceError",
    "InferenceResult",
    "InvalidImageError",
    "ValidImage",
    "annotate_image",
    "class_names",
    "describe_model",
    "load_detector",
    "predict_image",
    "resolve_weights",
    "validate_and_decode",
]

#: Accepted upload extensions. Anything else is refused before a decode is attempted.
ALLOWED_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
#: Refuse uploads larger than this before decoding, so a hostile file cannot exhaust memory.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
#: Images below this side are smaller than the model's stride and cannot carry a detection.
MIN_SIDE = 32
#: Refuse absurd dimensions: a 6000 px square already costs ~108 MB as float32 three-channel.
MAX_SIDE = 6000
#: Where a deployment looks for a checkpoint when nothing is configured explicitly, in order.
#: ``weights/`` is the committed-weights location the deployment guide describes; the ``results/``
#: path is where training writes, so a local run works with no configuration at all.
DEFAULT_WEIGHT_CANDIDATES = (
    "weights/sarvo_ssac001.pt",
    "results/runs/SSAC-001/weights/best.pt",
)


class InvalidImageError(ValueError):
    """An upload is not a usable image; the message says what to change."""


class CheckpointError(RuntimeError):
    """The weights file is missing or cannot be loaded as a detector."""


class InferenceError(RuntimeError):
    """A forward pass failed; the message is phrased so a UI can show it verbatim."""


@dataclass(frozen=True)
class ValidImage:
    """A validated upload: a uint8 BGR array (grayscale replicated) plus its true size."""

    array: object  # numpy.ndarray, kept untyped so this module imports without numpy
    height: int
    width: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class InferenceResult:
    """One image's detections plus the provenance needed to read them.

    The fields are deliberately flat and JSON-serialisable: a UI, a CLI, and a test all read the
    same object, so what a reader sees is what was measured.
    """

    detections: tuple[Detection, ...]
    latency_ms: float
    image_height: int
    image_width: int
    imgsz: int
    conf: float
    iou: float
    device: str
    weights: str

    @property
    def n_detections(self) -> int:
        return len(self.detections)

    def as_rows(self) -> list[dict]:
        """Detections as plain dicts, in the format a table or the CLI prints."""
        return [
            {
                "class": d.cls,
                "xyxy": [round(v, 2) for v in d.xyxy],
                "confidence": round(d.score, 4),
                "area_px": round(d.area, 1),
            }
            for d in self.detections
        ]

    def as_dict(self) -> dict:
        return {
            "n_detections": self.n_detections,
            "detections": self.as_rows(),
            "latency_ms": round(self.latency_ms, 2),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "weights": self.weights,
        }


def validate_and_decode(data: bytes, filename: str = "upload.png") -> ValidImage:
    """Validate upload bytes and decode to a three-channel BGR array, or raise.

    The checks are ordered cheapest-first (suffix, size, decode, dimensions) so a clearly bad
    upload is refused before any pixel work, and every failure names the fix rather than the
    symptom. A grayscale SAR chip is replicated to the three channels the backbone takes, which
    is exactly what :func:`saryolo.evaluation.efficiency.load_image_batch` does for timing -- so
    a demo input is preprocessed the same way a measured input was.
    """
    import cv2
    import numpy as np

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise InvalidImageError(
            f"{suffix or 'no extension'!r} is not a supported image type; "
            f"use one of {', '.join(ALLOWED_SUFFIXES)}"
        )
    if not data:
        raise InvalidImageError("the uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError(
            f"the file is {len(data) / 1e6:.1f} MB, over the {MAX_UPLOAD_BYTES / 1e6:.0f} MB "
            "limit; downscale or re-encode it"
        )
    gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise InvalidImageError("the file could not be decoded as an image; it may be corrupt")
    height, width = gray.shape[:2]
    if min(height, width) < MIN_SIDE:
        raise InvalidImageError(
            f"the image is {width}x{height} px, smaller than the {MIN_SIDE} px minimum the "
            "model's stride requires"
        )
    if max(height, width) > MAX_SIDE:
        raise InvalidImageError(
            f"the image is {width}x{height} px, larger than the {MAX_SIDE} px limit; downscale it"
        )
    bgr = np.repeat(gray[:, :, None], 3, axis=2)
    return ValidImage(array=bgr, height=height, width=width)


def resolve_weights(explicit: str | Path | None = None) -> str | None:
    """Resolve which checkpoint to serve, or ``None`` when nothing is configured.

    Order: an explicit argument, then ``SARVO_WEIGHTS``, then :data:`DEFAULT_WEIGHT_CANDIDATES`.
    A *configured* path is returned even when the file does not exist, so the resulting error
    names the path the caller actually asked for -- while ``None`` means "nothing was configured
    at all", which is a different message (and a different fix) from "configured but missing".

    This exists because a hosted deployment cannot rely on a training artefact being present: the
    environment variable is the documented way to point an instance at its weights, and the
    default candidates keep a local checkout working with no configuration.
    """
    for value in (explicit, os.environ.get("SARVO_WEIGHTS")):
        if value:
            return str(value)
    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        if Path(candidate).is_file():
            return str(candidate)
    return None


def load_detector(weights: str | Path, device: str = "cpu"):
    """Load a SARVO detector from a checkpoint, raising rather than guessing.

    Uses the repository's own facade resolution (``load_model``) so the graph built here is the
    graph the checkpoint was trained as -- a custom-module arm must not silently fall back to a
    stock ``YOLO``. Loading is expensive and is therefore the caller's job to cache; the Streamlit
    app caches it, and the CLI loads it once per invocation.
    """
    path = Path(weights)
    if not path.is_file():
        raise CheckpointError(
            f"checkpoint not found at {path}; train an arm or point --weights at an existing "
            "checkpoint (this module never substitutes an untrained model)"
        )
    try:
        from saryolo.training.trainer import load_model

        model = load_model(str(path))
    except Exception as exc:  # noqa: BLE001 - re-raised with context, never swallowed
        raise CheckpointError(f"could not load the checkpoint at {path}: {exc}") from exc
    net = getattr(model, "model", model)
    net.to(device)
    net.eval()
    return model


def class_names(model) -> dict[int, str]:
    """Class-id to name mapping from the detector, defaulting to the single SAR ship class."""
    names = getattr(getattr(model, "model", model), "names", None) or getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {0: "ship"}


def describe_model(model, weights: str | Path) -> dict:
    """Parameters, checkpoint size and device for the model-information panel."""
    net = getattr(model, "model", model)
    params = sum(p.numel() for p in net.parameters())
    path = Path(weights)
    return {
        "weights": str(path),
        "params": int(params),
        "params_M": round(params / 1e6, 3),
        "checkpoint_MB": round(path.stat().st_size / 1024**2, 2) if path.is_file() else None,
        "device": str(next(net.parameters()).device),
        "classes": class_names(model),
    }


def annotate_image(image_bgr, detections, names: dict[int, str] | None = None):
    """Draw the detections onto a copy of a uint8 BGR image and return it.

    Kept in this module (rather than in the Streamlit file) so the rendering that a *demo* shows
    is the same code a *test* can call: an interface that drew its own boxes could render a box
    the model never produced, and nothing would catch it. Pure and side-effect free -- the input
    array is not modified -- so it is safe to call on a cached image more than once.
    """
    import cv2

    names = names or {0: "ship"}
    canvas = image_bgr.copy()
    thickness = max(1, round(min(canvas.shape[:2]) / 400))
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        label = f"{names.get(det.cls, det.cls)} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(
            canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            cv2.LINE_AA,
        )
    return canvas


def predict_image(
    model,
    image,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "cpu",
    weights: str = "",
) -> InferenceResult:
    """Run one forward pass and return detections in the stable :class:`InferenceResult` format.

    ``image`` may be a path or a uint8 BGR array (as :func:`validate_and_decode` returns).
    ``conf`` defaults to ``0.25`` -- a *deployment* threshold, deliberately different from the
    evaluator's ``0.001``, which exists to trace the full AP curve. Both are stated in the
    result, so a demo number is never confused with a metric number.

    An image with no detections is a valid outcome (``n_detections == 0``), not an error; only a
    genuinely failed pass raises, and it is wrapped so the caller can show it.
    """
    if isinstance(image, Path):
        image = str(image)
    start = time.perf_counter()
    try:
        results = model.predict(
            source=image, imgsz=imgsz, conf=conf, iou=iou, device=device,
            save=False, verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, actionable error
        raise InferenceError(f"inference failed: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000.0
    result = results[0]
    boxes = getattr(result, "boxes", None)
    detections: list[Detection] = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, cls in zip(xyxy, scores, classes, strict=True):
            values = (float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score))
            # A non-finite coordinate or score is not a detection: it cannot be drawn or scored,
            # and a UI would render it as a stray mark or a ``nan`` label. Dropping it silently
            # would hide a genuine numerical fault in the forward pass, so it is refused instead.
            if not all(math.isfinite(v) for v in values):
                raise InferenceError(
                    f"the model returned a non-finite detection {values!r}; refusing to report "
                    "it as a result"
                )
            detections.append(
                Detection(
                    image="",
                    cls=int(cls),
                    xyxy=values[:4],
                    score=values[4],
                )
            )
    # Ultralytics reports the original image size on the result; fall back to the array shape.
    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape is not None and len(orig_shape) == 2:
        height, width = int(orig_shape[0]), int(orig_shape[1])
    else:
        shape = getattr(image, "shape", ())
        height = int(shape[0]) if len(shape) >= 2 else 0
        width = int(shape[1]) if len(shape) >= 2 else 0
    return InferenceResult(
        detections=tuple(detections),
        latency_ms=latency_ms,
        image_height=height,
        image_width=width,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        weights=weights,
    )
