"""Failure analysis: an explicit, countable error taxonomy.

The project brief requires failure cases to be categorised rather than hidden.
Each prediction/ground-truth pair is assigned exactly one outcome, so the
categories sum to a total and a table of counts is directly checkable.

Outcomes
--------
``true_positive``     matched at IoU >= threshold with the correct class
``classification_error`` matched location, wrong class
``localization_error`` correct class present but IoU in [0.1, threshold)
``false_positive``    no ground truth overlaps
``false_negative``    ground truth with no detection of its class
``small_object_miss`` false negative specifically on a COCO-small target
``clutter_false_positive`` false positive on a target-sized region of a scene
    whose mean local contrast places it in the low-contrast tail (a proxy for
    "the model fired on ambiguous clutter")
``clutter_confusion`` false positive overlapping a *large* ground-truth box
    without matching it, i.e. duplicated/partial firings on one object
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

__all__ = ["FailureCase", "analyse_failures", "FailureSummary"]


class FailureCase:
    """One categorised error, kept with its coordinates for later visualization."""

    __slots__ = ("image", "kind", "cls", "iou", "score", "xyxy")

    def __init__(self, image: str, kind: str, cls: int, iou: float, score: float, xyxy=(0.0, 0.0, 0.0, 0.0)):
        self.image = image
        self.kind = kind
        self.cls = cls
        self.iou = float(iou)
        self.score = float(score)
        self.xyxy = tuple(float(v) for v in xyxy)

    def to_dict(self) -> dict:
        return {"image": self.image, "kind": self.kind, "cls": self.cls,
                "iou": self.iou, "score": self.score, "xyxy": list(self.xyxy)}

    def __repr__(self) -> str:
        return f"FailureCase({self.image}, {self.kind}, cls={self.cls}, iou={self.iou:.3f})"


class FailureSummary:
    """Counts per outcome plus concrete examples of each."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.examples: dict[str, list[FailureCase]] = {}
        self.n_gt = 0
        self.n_det = 0

    def add(self, case: FailureCase, keep: int = 5) -> None:
        self.counts[case.kind] += 1
        bucket = self.examples.setdefault(case.kind, [])
        if len(bucket) < keep:
            bucket.append(case)

    def summary(self) -> str:
        lines = [f"Failure analysis: {self.n_gt} GT, {self.n_det} detections"]
        for kind, count in self.counts.most_common():
            lines.append(f"  {kind}: {count}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_gt": self.n_gt,
            "n_det": self.n_det,
            "counts": dict(self.counts),
            "examples": {k: [c.to_dict() for c in v] for k, v in self.examples.items()},
        }


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def analyse_failures(
    preds: list,
    gts: list,
    iou_threshold: float = 0.5,
    low_iou_threshold: float = 0.1,
    small_area: float = 32.0**2,
    contrast_by_image: dict[str, float] | None = None,
    contrast_quantile: float = 0.25,
) -> FailureSummary:
    """Categorise every prediction and ground-truth box into one outcome.

    Args:
        preds: Predictions as :class:`saryolo.evaluation.metrics.Detection`.
        gts: Ground truth as :class:`saryolo.evaluation.metrics.Detection`.
        iou_threshold: IoU above which a match counts as localised.
        low_iou_threshold: IoU above which a wrong-class or partial overlap is
            treated as a localization/classification error rather than a pure
            false positive.
        small_area: COCO small-object area threshold (pixels^2).
        contrast_by_image: Optional per-image local-contrast value, used to flag
            clutter false positives in the low-contrast tail.
        contrast_quantile: Quantile below which an image is "low contrast".

    Returns:
        A :class:`FailureSummary`.
    """
    summary = FailureSummary()
    summary.n_gt = len(gts)
    summary.n_det = len(preds)

    by_image_pred: dict[str, list] = {}
    by_image_gt: dict[str, list] = {}
    for d in preds:
        by_image_pred.setdefault(d.image, []).append(d)
    for g in gts:
        by_image_gt.setdefault(g.image, []).append(g)

    threshold = None
    if contrast_by_image and len(contrast_by_image) > 3:
        threshold = float(np.quantile(list(contrast_by_image.values()), contrast_quantile))

    for image in set(by_image_pred) | set(by_image_gt):
        image_preds = sorted(by_image_pred.get(image, []), key=lambda d: -d.score)
        image_gts = list(by_image_gt.get(image, []))
        matched_gt = [False] * len(image_gts)

        for det in image_preds:
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(image_gts):
                if matched_gt[j]:
                    continue
                iou = _iou(det.xyxy, gt.xyxy)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_threshold and image_gts[best_j].cls == det.cls:
                matched_gt[best_j] = True
                summary.add(FailureCase(image, "true_positive", det.cls, best_iou, det.score, det.xyxy))
            elif best_j >= 0 and best_iou >= iou_threshold:
                matched_gt[best_j] = True
                summary.add(FailureCase(image, "classification_error", det.cls, best_iou, det.score, det.xyxy))
            elif best_j >= 0 and best_iou >= low_iou_threshold:
                kind = "clutter_confusion" if image_gts[best_j].area > 4 * small_area else "localization_error"
                summary.add(FailureCase(image, kind, det.cls, best_iou, det.score, det.xyxy))
            else:
                low_contrast = threshold is not None and contrast_by_image.get(image, 1.0) <= threshold
                kind = "clutter_false_positive" if low_contrast else "false_positive"
                summary.add(FailureCase(image, kind, det.cls, best_iou, det.score, det.xyxy))

        for j, gt in enumerate(image_gts):
            if matched_gt[j]:
                continue
            kind = "small_object_miss" if gt.area < small_area else "false_negative"
            summary.add(FailureCase(image, kind, gt.cls, 0.0, 0.0, gt.xyxy))

    return summary


def image_contrast_map(data_yaml: str | Path, limit: int | None = None) -> dict[str, float]:
    """Per-image local contrast for the validation split (clutter proxy)."""
    import cv2

    from saryolo.data.yolo import load_data_config

    root, cfg = load_data_config(data_yaml)
    images_dir = root / (cfg.get("val") or "images/val")

    out: dict[str, float] = {}
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    for path in images[:limit] if limit else images:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        arr = img.astype(np.float32) / 255.0
        out[path.stem] = float(((arr - cv2.blur(arr, (5, 5))) ** 2).mean() ** 0.5)
    return out


def write_failure_report(summary: FailureSummary, out_path: str | Path) -> Path:
    """Persist the failure taxonomy to JSON."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary.to_dict(), indent=2))
    return out
