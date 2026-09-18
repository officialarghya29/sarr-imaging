"""Dataset statistics and profiling.

The project brief is explicit that the architecture must be *evidence-driven*:
before adding a module, characterise the data. This module produces exactly the
evidence a module decision needs:

* class distribution, including imbalance
* object-count-per-image distribution (density / crowding)
* box width/height/aspect-ratio distributions
* object-size histogram split by the COCO small/medium/large thresholds, which is
  what justifies (or kills) the P2 detection head
* per-image first-order statistics used as SAR difficulty proxies: mean
  intensity, local-contrast (std) and an SNR-style target/background ratio
* object density heat map, so the "dense objects / clutter" claim is measured

Writes JSON plus figures to ``dataset_statistics/``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["DatasetStatistics", "profile_dataset", "COCO_SIZE_BINS"]

#: COCO object-size convention, in **area** (pixels^2) of the original image.
#: The thresholds are 32^2 and 96^2; binning raw linear sizes against 32/96 would
#: misclassify small objects as large and would wrongly justify the P2 head.
COCO_SIZE_BINS = {"small": (0.0, 32.0**2), "medium": (32.0**2, 96.0**2), "large": (96.0**2, float("inf"))}


@dataclass
class DatasetStatistics:
    """Computed statistics for one dataset split."""

    name: str = ""
    split: str = ""
    num_images: int = 0
    num_boxes: int = 0
    class_names: list[str] = field(default_factory=list)
    boxes_per_class: dict[str, int] = field(default_factory=dict)
    images_per_class: dict[str, int] = field(default_factory=dict)
    boxes_per_image: dict[str, float] = field(default_factory=dict)
    image_sizes: dict[str, int] = field(default_factory=dict)
    box_width_px: dict[str, float] = field(default_factory=dict)
    box_height_px: dict[str, float] = field(default_factory=dict)
    aspect_ratio: dict[str, float] = field(default_factory=dict)
    size_distribution: dict[str, int] = field(default_factory=dict)
    intensity: dict[str, float] = field(default_factory=dict)
    local_contrast: dict[str, float] = field(default_factory=dict)
    target_background_ratio: dict[str, float] = field(default_factory=dict)
    imbalance_ratio: float = 0.0
    histogram_data: dict[str, list[float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Dataset statistics: {self.name} [{self.split}]",
            f"  images: {self.num_images}   boxes: {self.num_boxes}",
            f"  classes: {self.class_names}",
            f"  boxes/class: {self.boxes_per_class}",
            f"  imbalance ratio (max/min class): {self.imbalance_ratio:.2f}",
            f"  boxes/image: mean {self.boxes_per_image.get('mean', 0):.2f} "
            f"max {self.boxes_per_image.get('max', 0):.0f}",
            f"  size bins: {self.size_distribution}",
            f"  mean local contrast: {self.local_contrast.get('mean', 0):.4f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def _stats(values: list[float]) -> dict[str, float]:
    """Mean/std/min/max/median/percentiles for a list of values."""
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def profile_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: list[str] | tuple[str, ...],
    name: str = "",
    split: str = "train",
    intensity_sample: int = 300,
) -> DatasetStatistics:
    """Compute the full statistics profile for one split.

    Args:
        images_dir: Directory of images.
        labels_dir: Directory of YOLO ``.txt`` labels.
        class_names: Class names indexed by class id.
        name: Dataset name for the report.
        split: Split name for the report.
        intensity_sample: How many images to read for the intensity/contrast
            proxies. Reading every image of a 116k-image benchmark is wasteful;
            a fixed random-free prefix is used so results are reproducible.

    Returns:
        A populated :class:`DatasetStatistics`.
    """
    import cv2
    import numpy as np

    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    stats = DatasetStatistics(name=name, split=split, class_names=list(class_names))

    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    stats.num_images = len(images)

    counts: Counter = Counter()
    image_counts: Counter = Counter()
    per_image: list[int] = []
    widths, heights, aspects, sizes = [], [], [], []
    sizes_px: list[float] = []
    size_bins: Counter = Counter()
    image_sizes: Counter = Counter()
    density = np.zeros((10, 10), dtype=np.float64)

    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        rows = []
        if label_path.exists():
            rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        per_image.append(len(rows))
        w = h = None
        if img_path.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp"):
            data = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape[:2]
                image_sizes[f"{w}x{h}"] += 1
        for row in rows:
            if len(row) != 5:
                continue
            try:
                cid = int(float(row[0]))
                cx, cy, bw, bh = (float(v) for v in row[1:])
            except ValueError:
                continue
            label = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            counts[label] += 1
            image_counts[label] += 1
            if w and h:
                bw_px, bh_px = bw * w, bh * h
                widths.append(bw_px)
                heights.append(bh_px)
                aspects.append(bw_px / max(bh_px, 1e-9))
                sizes_px.append(float(np.sqrt(bw_px * bh_px)))
                area = bw_px * bh_px  # compared against COCO_SIZE_BINS, which are areas
                for bin_name, (lo, hi) in COCO_SIZE_BINS.items():
                    if lo <= area < hi:
                        size_bins[bin_name] += 1
                        break
                gy, gx = int(min(cy, 0.999) * 10), int(min(cx, 0.999) * 10)
                density[gy, gx] += 1

    stats.num_boxes = int(sum(counts.values()))
    stats.boxes_per_class = dict(sorted(counts.items()))
    stats.images_per_class = dict(sorted(image_counts.items()))
    stats.boxes_per_image = _stats([float(v) for v in per_image])
    stats.image_sizes = dict(sorted(image_sizes.items(), key=lambda kv: -kv[1])[:10])
    stats.box_width_px = _stats(widths)
    stats.box_height_px = _stats(heights)
    stats.aspect_ratio = _stats(aspects)
    stats.size_distribution = dict(size_bins)
    if counts:
        vals = [v for v in counts.values() if v > 0]
        stats.imbalance_ratio = float(max(vals) / max(min(vals), 1))

    # SAR difficulty proxies on a fixed prefix so the numbers are reproducible.
    intensities, contrasts, ratios = [], [], []
    for img_path in images[:intensity_sample]:
        data = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        arr = img.astype(np.float32) / 255.0
        intensities.append(float(arr.mean()))
        mu = cv2.blur(arr, (5, 5))
        contrasts.append(float(((arr - mu) ** 2).mean() ** 0.5))
        label_path = labels_dir / f"{img_path.stem}.txt"
        mask = np.zeros(arr.shape, dtype=bool)
        if label_path.exists():
            h, w = arr.shape
            for row in (line.split() for line in label_path.read_text().splitlines() if line.strip()):
                if len(row) != 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in row[1:])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                mask[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)] = True
        if mask.any() and (~mask).any():
            ratios.append(float(arr[mask].mean() / max(arr[~mask].mean(), 1e-6)))
    stats.intensity = _stats(intensities)
    stats.local_contrast = _stats(contrasts)
    stats.target_background_ratio = _stats(ratios)

    stats.histogram_data = {
        "box_width_px": _histogram(widths),
        "box_height_px": _histogram(heights),
        "aspect_ratio": _histogram(aspects),
        "object_equiv_diameter_px": _histogram(sizes_px),
        "boxes_per_image": _histogram([float(v) for v in per_image]),
        "density_grid": density.flatten().tolist(),
        "local_contrast": _histogram(contrasts),
        "intensity": _histogram(intensities),
    }
    return stats


def _histogram(values: list[float], bins: int = 40) -> list[float]:
    if not values:
        return []
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    return [int(c) for c in counts]


def save_statistics(stats: DatasetStatistics, out_dir: str | Path) -> Path:
    """Write statistics JSON to ``out_dir/<name>_<split>.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stats.name or 'dataset'}_{stats.split}.json"
    path.write_text(json.dumps(stats.to_dict(), indent=2))
    return path


def plot_statistics(stats: DatasetStatistics, out_dir: str | Path, dpi: int = 200) -> list[Path]:
    """Render the standard figure set. Returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    h = stats.histogram_data

    def _bar(values, labels, title, xlabel, fname, log=False):
        if not values:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color="#2c7fb8")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        if log:
            ax.set_yscale("log")
        fig.tight_layout()
        path = out / fname
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    if stats.boxes_per_class:
        _bar(list(stats.boxes_per_class.values()), list(stats.boxes_per_class), "Class distribution", "class", "class_distribution.png")
    _bar(h.get("box_width_px", []), [str(i) for i in range(len(h.get("box_width_px", [])))], "Box width", "bin", "box_width_hist.png")
    _bar(h.get("box_height_px", []), [str(i) for i in range(len(h.get("box_height_px", [])))], "Box height", "bin", "box_height_hist.png")
    _bar(h.get("aspect_ratio", []), [str(i) for i in range(len(h.get("aspect_ratio", [])))], "Aspect ratio", "bin", "aspect_ratio_hist.png")
    _bar(h.get("object_equiv_diameter_px", []), [str(i) for i in range(len(h.get("object_equiv_diameter_px", [])))], "Object size", "bin", "object_size_hist.png")
    _bar(h.get("boxes_per_image", []), [str(i) for i in range(len(h.get("boxes_per_image", [])))], "Objects per image", "objects", "objects_per_image_hist.png")
    _bar(h.get("local_contrast", []), [str(i) for i in range(len(h.get("local_contrast", [])))], "Local contrast (difficulty proxy)", "bin", "local_contrast_hist.png")

    if h.get("density_grid"):
        grid = np.asarray(h["density_grid"], dtype=np.float64).reshape(10, 10)
        fig, ax = plt.subplots(figsize=(5, 4.4))
        im = ax.imshow(grid, cmap="inferno")
        ax.set_title("Object centre density")
        ax.set_xlabel("image x decile")
        ax.set_ylabel("image y decile")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        path = out / "density_heatmap.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    if stats.size_distribution:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(list(stats.size_distribution), list(stats.size_distribution.values()), color="#e6550d")
        ax.set_title("COCO size distribution")
        ax.set_ylabel("count")
        fig.tight_layout()
        path = out / "size_distribution.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    return written
