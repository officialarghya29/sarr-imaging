"""COCO-style detection metrics, including scale-wise AP.

Why a dedicated implementation
------------------------------
The paper's central claim is that the SAR modules help most on *small, weak*
targets. A single mAP can hide exactly the effect being claimed, so
``AP_small`` / ``AP_medium`` / ``AP_large`` are reported separately.

Ultralytics' validator does not expose scale AP in a stable way, and its
``save_json`` path relies on an internal image-id mapping that is easy to get
subtly wrong. This module therefore avoids that mapping entirely:

* ground truth is read straight from the YOLO label files (unambiguous);
* predictions are written with ``save_txt=True``, so every detection is keyed by
  image *file name* — no index mapping to trust;
* AP is computed here with the COCO protocol (101-point interpolation, greedy
  matching, and the "ignored ground truth" rule for area ranges).

Documented deviations from pycocotools
--------------------------------------
* ``maxDet`` is applied per image per class (COCO) but there is no cross-image
  score truncation beyond that.
* AP is averaged over the categories that actually have ground truth in the
  given area range; COCO averages over all categories, so a dataset with an
  empty class would report slightly differently. Both the count and the
  categories used are returned so the difference is visible and reportable.
These are stated here rather than silently absorbed, because a reported metric
must mean exactly what the paper says it means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Detection",
    "load_yolo_ground_truth",
    "load_yolo_predictions",
    "predict_to_labels",
    "compute_ap",
    "evaluate_detections",
    "AREA_RANGES",
]

#: COCO area ranges in pixels^2 (32^2 and 96^2 thresholds).
AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}
IOU_THRESHOLDS = tuple(np.arange(0.5, 1.0, 0.05).round(2))


@dataclass
class Detection:
    """One box: absolute pixel coordinates in the original image."""

    image: str
    cls: int
    xyxy: tuple[float, float, float, float]
    score: float = 1.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    return img.shape[1], img.shape[0]


def _resolve_split_dirs(data_yaml: str | Path) -> tuple[Path, Path, list[str]]:
    """Return ``(images_dir, labels_dir, class_names)`` for the validation split.

    Path resolution is delegated to :func:`saryolo.data.yolo.load_data_config` so
    that training and evaluation can never disagree about where a split lives.
    """
    from saryolo.data.yolo import load_data_config

    root, cfg = load_data_config(data_yaml)
    split = cfg.get("val") or cfg.get("val_images") or "images/val"
    images_dir = Path(split) if Path(split).is_absolute() else root / split
    labels_dir = Path(str(images_dir).replace("images", "labels", 1))
    names = cfg.get("names") or [f"class_{i}" for i in range(int(cfg.get("nc", 1)))]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return images_dir, labels_dir, list(names)


def load_yolo_ground_truth(data_yaml: str | Path) -> list[Detection]:
    """Read all validation-split boxes from YOLO label files in absolute pixels."""
    images_dir, labels_dir, _ = _resolve_split_dirs(data_yaml)
    gts: list[Detection] = []
    for label in sorted(labels_dir.glob("*.txt")):
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        size = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate = images_dir / f"{label.stem}{ext}"
            if candidate.exists():
                size = _image_size(candidate)
                break
        if size is None:
            continue
        w, h = size
        for row in rows:
            if len(row) < 5:
                continue
            cid = int(float(row[0]))
            cx, cy, bw, bh = (float(v) for v in row[1:5])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            gts.append(Detection(label.stem, cid, (x1, y1, x2, y2), score=1.0))
    return gts


def predict_to_labels(
    weights: str | Path,
    images_dir: str | Path,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.7,
    out_dir: str | Path = "results/predictions",
    device: str | None = None,
    batch: int = 16,
) -> Path:
    """Run inference and write per-image YOLO label files with confidences.

    ``conf`` defaults to 0.001 so the AP curve is computed over the full
    precision/recall range rather than truncated at a deployment threshold.

    ``out_dir`` is resolved to an absolute path before being handed to ultralytics. This is
    not tidiness: ultralytics re-roots a *relative* ``project`` under its own runs directory,
    so ``out_dir="results/eval"`` had inference written to ``runs/detect/results/eval`` while
    this function then looked in ``results/eval`` and raised ``FileNotFoundError`` -- meaning
    every caller that passed a relative path failed, including this function's own default.
    With an absolute path the two agree and the returned directory is the one written.
    """
    from saryolo.training.trainer import load_model

    out_dir = Path(out_dir).resolve()
    model = load_model(str(weights))
    kwargs = dict(
        source=str(images_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save_txt=True,
        save_conf=True,
        save=False,
        verbose=False,
        project=str(out_dir),
        name=".",
        exist_ok=True,
    )
    if device:
        kwargs["device"] = device
    model.predict(**kwargs)
    labels = out_dir / "labels"
    if not labels.exists():
        # Ultralytics sometimes nests under the run name.
        candidates = sorted(out_dir.rglob("labels"))
        if not candidates:
            raise FileNotFoundError(
                f"No prediction labels were written under {out_dir}. Note that ultralytics "
                "re-roots a relative `project` under runs/detect/, so a caller passing a "
                "relative out_dir would find its output elsewhere."
            )
        labels = candidates[0]
    return labels


def load_yolo_predictions(labels_dir: str | Path, images_dir: str | Path) -> list[Detection]:
    """Read ultralytics' ``save_txt`` output (``cls cx cy w h conf``, normalised)."""
    labels_dir, images_dir = Path(labels_dir), Path(images_dir)
    dets: list[Detection] = []
    size_cache: dict[str, tuple[int, int]] = {}
    for label in sorted(labels_dir.glob("*.txt")):
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        stem = label.stem
        if stem not in size_cache:
            found = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    found = _image_size(candidate)
                    break
            if found is None:
                continue
            size_cache[stem] = found
        w, h = size_cache[stem]
        for row in rows:
            if len(row) < 6:
                continue
            cid = int(float(row[0]))
            cx, cy, bw, bh = (float(v) for v in row[1:5])
            score = float(row[5])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            dets.append(Detection(stem, cid, (x1, y1, x2, y2), score=score))
    return dets


def _iou_matrix(dets: list[Detection], gts: list[Detection]) -> np.ndarray:
    """IoU between every detection and every ground-truth box in one image/class."""
    if not dets or not gts:
        return np.zeros((len(dets), len(gts)), dtype=np.float64)
    d = np.asarray([x.xyxy for x in dets], dtype=np.float64)
    g = np.asarray([x.xyxy for x in gts], dtype=np.float64)
    ix1 = np.maximum(d[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(d[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(d[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(d[:, None, 3], g[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    da = np.clip(d[:, 2] - d[:, 0], 0, None) * np.clip(d[:, 3] - d[:, 1], 0, None)
    ga = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    union = da[:, None] + ga[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _match_image(
    dets: list[Detection],
    gts: list[Detection],
    iou_thr: float,
    area_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy per-image matching for one class.

    Returns ``(tp, fp)`` boolean arrays aligned with ``dets``, implementing the
    COCO "ignored ground truth" rule: a detection that matches ground truth
    outside the requested area range is neither a true nor a false positive.
    """
    n = len(dets)
    tp = np.zeros(n, dtype=bool)
    fp = np.zeros(n, dtype=bool)
    if n == 0:
        return tp, fp
    if not gts:
        fp[:] = True
        return tp, fp

    order = np.argsort([-d.score for d in dets], kind="stable")
    ious = _iou_matrix(dets, gts)
    lo, hi = area_range
    gt_area = np.asarray([g.area for g in gts], dtype=np.float64)
    in_range = (gt_area >= lo) & (gt_area < hi)
    already_matched = np.zeros(len(gts), dtype=bool)

    for idx in order:
        row = ious[idx]
        # Initialised below the threshold, not at it: a detection whose IoU is *exactly*
        # the threshold must match (the >= convention), and seeding best_iou with iou_thr
        # makes the `> best_iou` comparison silently reject those exact hits.
        best_j, best_iou = -1, -1.0
        ignored_j, ignored_iou = -1, -1.0
        for j in range(len(gts)):
            if already_matched[j] or row[j] < iou_thr:
                continue
            if in_range[j]:
                if row[j] > best_iou:
                    best_j, best_iou = j, row[j]
            elif row[j] > ignored_iou:
                ignored_j, ignored_iou = j, row[j]
        if best_j >= 0:
            already_matched[best_j] = True
            tp[idx] = True
        elif ignored_j >= 0:
            # Matches ground truth outside the area range: the COCO protocol
            # ignores this detection rather than penalising it as a false positive.
            continue
        else:
            fp[idx] = True
    return tp, fp


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP, as used by COCO."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.searchsorted(mrec, np.linspace(0, 1, 101))
    idx = np.clip(idx, 0, mpre.size - 1)
    return float(mpre[idx].mean())


def compute_ap(
    dets: list[Detection],
    gts: list[Detection],
    nc: int,
    area_range: tuple[float, float] = AREA_RANGES["all"],
    max_det: int = 100,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
) -> dict:
    """COCO-style AP for one area range, averaged over IoU thresholds."""
    by_image: dict[str, dict[str, list[Detection]]] = {}
    for det in dets:
        by_image.setdefault(det.image, {}).setdefault("det", []).append(det)
    for gt in gts:
        by_image.setdefault(gt.image, {}).setdefault("gt", []).append(gt)

    ap_per_iou: list[float] = []
    per_class: dict[int, list[float]] = {}
    categories_used: set[int] = set()

    for iou_thr in iou_thresholds:
        class_scores: dict[int, list[tuple[float, int]]] = {c: [] for c in range(nc)}
        n_pos: dict[int, float] = {c: 0.0 for c in range(nc)}
        for payload in by_image.values():
            image_dets = payload.get("det", [])
            image_gts = payload.get("gt", [])
            for c in range(nc):
                c_dets = sorted([d for d in image_dets if d.cls == c], key=lambda d: -d.score)[:max_det]
                c_gts = [g for g in image_gts if g.cls == c]
                lo, hi = area_range
                n_pos[c] += sum(1 for g in c_gts if lo <= g.area < hi)
                tp, fp = _match_image(c_dets, c_gts, iou_thr, area_range)
                for det, is_tp, is_fp in zip(c_dets, tp, fp, strict=True):
                    if is_tp or is_fp:
                        class_scores[c].append((det.score, int(is_tp)))
        iou_aps = []
        for c in range(nc):
            if n_pos[c] <= 0:
                continue
            entries = sorted(class_scores[c], key=lambda t: -t[0])
            if not entries:
                # Ground truth exists for this class but nothing was predicted for it:
                # that is an AP of 0, not an unmeasurable metric. Returning None here
                # would make a detector that predicts nothing look like it was never
                # evaluated.
                ap = 0.0
            else:
                tps = np.cumsum([e[1] for e in entries]).astype(np.float64)
                fps = np.cumsum([1 - e[1] for e in entries]).astype(np.float64)
                recall = tps / n_pos[c]
                precision = tps / np.maximum(tps + fps, 1e-9)
                ap = _average_precision(recall, precision)
            per_class.setdefault(c, []).append(ap)
            categories_used.add(c)
            iou_aps.append(ap)
        if iou_aps:
            ap_per_iou.append(float(np.mean(iou_aps)))

    return {
        "AP": float(np.mean(ap_per_iou)) if ap_per_iou else None,
        "AP50": float(ap_per_iou[0]) if iou_thresholds and abs(iou_thresholds[0] - 0.5) < 1e-9 and ap_per_iou else None,
        "per_class_AP": {int(c): float(np.mean(v)) for c, v in per_class.items()},
        "categories_with_gt": sorted(categories_used),
        "n_positives": float(sum(n_pos.values())),
    }


def evaluate_detections(
    weights: str | Path,
    data_yaml: str | Path,
    imgsz: int = 640,
    conf: float = 0.001,
    device: str | None = None,
    out_dir: str | Path = "results/predictions",
) -> dict:
    """Full evaluation: overall and scale-wise AP for a checkpoint.

    Returns:
        Dict with ``mAP50``, ``mAP50_95`` and ``AP_small``/``AP_medium``/
        ``AP_large`` (each possibly ``None`` if the dataset has no ground truth
        in that range — ``None`` is never replaced with 0.0, so an impossible
        measurement is never presented as a real one).
    """
    images_dir, _, class_names = _resolve_split_dirs(data_yaml)
    labels_dir = predict_to_labels(weights, images_dir, imgsz=imgsz, conf=conf, out_dir=out_dir, device=device)
    dets = load_yolo_predictions(labels_dir, images_dir)
    gts = load_yolo_ground_truth(data_yaml)
    nc = len(class_names)

    overall = compute_ap(dets, gts, nc, AREA_RANGES["all"])
    result = {
        "mAP50": overall["AP50"],
        "mAP50_95": overall["AP"],
        "per_class_AP50_95": overall["per_class_AP"],
        "num_gt": len(gts),
        "num_det": len(dets),
        "classes": class_names,
    }
    for label in ("small", "medium", "large"):
        stats = compute_ap(dets, gts, nc, AREA_RANGES[label])
        n_pos = stats["n_positives"]
        key = f"AP_{label}"
        # A range with no ground truth is not a 0.0 score; it is not measurable.
        result[key] = stats["AP"] if n_pos > 0 else None
        result[f"n_gt_{label}"] = int(n_pos)
    return result


def write_metrics(metrics: dict, out_path: str | Path) -> Path:
    """Persist a metrics dict as JSON."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, default=float))
    return out
