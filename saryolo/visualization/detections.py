"""Qualitative detection visualization: ground truth vs baseline vs SAR-YOLO."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["draw_detections", "annotate_gt", "comparison_panel", "select_examples"]

_GT_COLOUR = (0, 210, 0)
_PRED_COLOUR = (255, 90, 0)


def _read_gray(path: Path) -> np.ndarray | None:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def annotate_gt(image: np.ndarray, image_stem: str, labels_dir: str | Path) -> np.ndarray:
    """Draw ground-truth boxes on a copy of ``image``."""
    import cv2

    out = image.copy()
    label = Path(labels_dir) / f"{image_stem}.txt"
    if not label.exists():
        return out
    h, w = out.shape[:2]
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), _GT_COLOUR, 2)
    return out


def draw_detections(image: np.ndarray, boxes: list[dict], conf_threshold: float = 0.25) -> np.ndarray:
    """Draw predicted boxes (``{"xyxy", "conf", "cls"[, "name"]}``) on a copy."""
    import cv2

    out = image.copy()
    for box in boxes:
        if float(box.get("conf", 1.0)) < conf_threshold:
            continue
        x1, y1, x2, y2 = (int(v) for v in box["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), _PRED_COLOUR, 2)
        label = f"{box.get('name', box.get('cls', ''))} {float(box.get('conf', 0.0)):.2f}".strip()
        cv2.putText(out, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _PRED_COLOUR, 1, cv2.LINE_AA)
    return out


def _predict_boxes(model, image_path: Path, imgsz: int = 640, conf: float = 0.25, device: str | None = None) -> list[dict]:
    """Run one-image inference and return boxes in absolute pixel coordinates."""
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False,
                            device=device) if device else model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    boxes: list[dict] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls.item())
            boxes.append({
                "xyxy": tuple(float(v) for v in box.xyxy[0].tolist()),
                "conf": float(box.conf.item()),
                "cls": cls,
                "name": names.get(cls, str(cls)),
            })
    return boxes


def comparison_panel(
    model,
    image_paths: list[str | Path],
    labels_dir: str | Path,
    out_path: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | None = None,
    titles: tuple[str, str, str] = ("SAR image + GT", "predictions", "prediction overlay"),
    dpi: int = 200,
) -> Path | None:
    """Render a ``GT | predictions | overlay`` grid for several images.

    Overlay means: ground truth in green *and* predictions in orange on the same
    axes, which is what makes false negatives and false positives legible at a
    glance instead of requiring the reader to compare two panels.
    """
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for path in image_paths:
        path = Path(path)
        image = _read_gray(path)
        if image is None:
            continue
        gt = annotate_gt(image, path.stem, labels_dir)
        boxes = _predict_boxes(model, path, imgsz=imgsz, conf=conf, device=device)
        pred = draw_detections(image, boxes)
        overlay = draw_detections(annotate_gt(image, path.stem, labels_dir), boxes)
        rows.append((gt, pred, overlay))
    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.6 * len(rows)), squeeze=False)
    for r, panels in enumerate(rows):
        for c, panel in enumerate(panels):
            axes[r][c].imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(titles[c])
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def select_examples(data_yaml: str | Path, limit: int = 4, mode: str = "dense") -> list[Path]:
    """Pick illustrative images: densest, sparsest, or smallest-object scenes."""
    from saryolo.data.yolo import load_data_config

    root, cfg = load_data_config(data_yaml)
    images_dir = root / (cfg.get("val") or "images/val")
    labels_dir = Path(str(images_dir).replace("images", "labels", 1))

    scored = []
    for label in sorted(labels_dir.glob("*.txt")):
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        areas = []
        for row in rows:
            if len(row) >= 5:
                areas.append(float(row[3]) * float(row[4]))
        image = next((images_dir / f"{label.stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                      if (images_dir / f"{label.stem}{ext}").exists()), None)
        if image is None:
            continue
        if mode == "dense":
            key = -len(rows)
        elif mode == "sparse":
            key = len(rows)
        elif mode == "small":
            key = float(np.mean(areas)) if areas else 1.0
        else:
            raise ValueError(f"Unknown mode {mode!r}; expected 'dense', 'sparse' or 'small'")
        scored.append((key, image))
    scored.sort(key=lambda t: t[0])
    return [p for _, p in scored[:limit]]
