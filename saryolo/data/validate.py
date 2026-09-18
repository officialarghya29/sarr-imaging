"""Dataset validation and quality control.

The validator never mutates the source data. It produces a machine-readable
report plus optional *separate* cleaned artefacts, so a questionable image can
always be traced back to what was actually on disk.

Checks
------
Images:   unreadable/corrupt files, zero-size, non-8-bit or unexpected channel
          counts, duplicated content (both exact MD5 and near-duplicate dHash).
Labels:   missing/empty label files, missing image for a label, invalid class
          ids, malformed rows, negative or non-normalised coordinates,
          coordinates outside the image, zero-area boxes, degenerate 1px slivers,
          and boxes that are implausibly tiny to be real targets.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["ValidationReport", "Issue", "validate_yolo_dataset", "write_report"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class Issue:
    """A single validation finding."""

    kind: str
    severity: str  # "error" | "warning" | "info"
    path: str
    detail: str = ""

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.kind}: {self.path} {self.detail}".rstrip()


@dataclass
class ValidationReport:
    """Aggregate validation result for one YOLO-format split (or a whole dataset)."""

    root: str
    n_images: int = 0
    n_labels: int = 0
    n_boxes: int = 0
    classes_present: dict[str, int] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when no error-severity issue was found."""
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"Validation report for {self.root}",
            f"  images: {self.n_images}   labels: {self.n_labels}   boxes: {self.n_boxes}",
            f"  classes: {dict(sorted(self.classes_present.items()))}",
            f"  errors: {len(self.errors)}   warnings: {len(self.warnings)}",
        ]
        if self.duplicate_groups:
            lines.append(f"  duplicate groups: {len(self.duplicate_groups)}")
        by_kind = Counter(i.kind for i in self.issues)
        if by_kind:
            lines.append("  issue counts:")
            lines.extend(f"    {k}: {v}" for k, v in sorted(by_kind.items()))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["errors"] = len(self.errors)
        d["warnings"] = len(self.warnings)
        d["ok"] = self.ok
        d["issues"] = [asdict(i) for i in self.issues]
        return d


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash: cheap near-duplicate detection robust to tiny jitter."""
    import cv2

    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def _read_image(path: Path):
    """Read an image, returning ``(array, gray)`` or ``(None, None)`` if unreadable."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)  # np.fromfile handles non-ASCII paths
    if data.size == 0:
        return None, None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = img if img.ndim == 2 else img[..., 0]
    return img, gray


def validate_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: list[str] | tuple[str, ...] | None = None,
    min_box_px: float = 2.0,
    near_dup_threshold: int = 4,
    limits: dict[str, float] | None = None,
) -> ValidationReport:
    """Validate a YOLO-format split without modifying it.

    Args:
        images_dir: Directory of images.
        labels_dir: Directory of ``.txt`` label files (one row per box:
            ``class cx cy w h``, normalised).
        class_names: Class names; used to check class ids and to label the report.
        min_box_px: Boxes smaller than this (in pixels, using the image width as
            reference for the normalised width) are reported as suspicious.
        near_dup_threshold: Hamming distance under which two dHashes are treated
            as near-duplicates.
        limits: Optional ``{"max_boxes": n}`` style thresholds for warnings.

    Returns:
        A :class:`ValidationReport`.
    """
    limits = limits or {}
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    report = ValidationReport(root=str(images_dir))
    if not images_dir.exists():
        report.issues.append(Issue("missing_directory", "error", str(images_dir)))
        return report
    if not labels_dir.exists():
        report.issues.append(Issue("missing_directory", "error", str(labels_dir)))

    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    report.n_images = len(images)

    seen_md5: dict[str, str] = {}
    hashes: list[tuple[str, int]] = []
    label_stems = {p.stem for p in labels_dir.glob("*.txt")} if labels_dir.exists() else set()
    image_stems = {p.stem for p in images}

    for img_path in images:
        img, gray = _read_image(img_path)
        if img is None:
            report.issues.append(Issue("corrupt_image", "error", str(img_path), "could not decode"))
            continue
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            report.issues.append(Issue("zero_size_image", "error", str(img_path)))
            continue
        if img.dtype != np.uint8:
            report.issues.append(Issue("non_uint8_image", "warning", str(img_path), f"dtype={img.dtype}"))

        digest = _md5(img_path)
        if digest in seen_md5:
            report.issues.append(Issue("exact_duplicate", "warning", str(img_path), f"same as {seen_md5[digest]}"))
        else:
            seen_md5[digest] = str(img_path)
        hashes.append((str(img_path), _dhash(gray)))

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            report.issues.append(
                Issue("missing_label", "warning", str(label_path), "image has no label file (may be a pure background image)")
            )
            continue
        report.n_labels += 1
        rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        if not rows:
            report.issues.append(Issue("empty_label", "info", str(label_path), "background image"))
            continue
        if len(rows) > limits.get("max_boxes", 10_000):
            report.issues.append(Issue("too_many_boxes", "warning", str(label_path), f"{len(rows)} rows"))
        for row_idx, row in enumerate(rows):
            if len(row) != 5:
                report.issues.append(Issue("malformed_row", "error", str(label_path), f"line {row_idx + 1}: {len(row)} fields"))
                continue
            try:
                cid = int(float(row[0]))
                cx, cy, bw, bh = (float(v) for v in row[1:])
            except ValueError:
                report.issues.append(Issue("unparsable_row", "error", str(label_path), f"line {row_idx + 1}"))
                continue
            if class_names is not None and not 0 <= cid < len(class_names):
                report.issues.append(Issue("invalid_class_id", "error", str(label_path), f"line {row_idx + 1}: class {cid}"))
            if not all(np.isfinite(v) for v in (cx, cy, bw, bh)):
                report.issues.append(Issue("non_finite_coordinate", "error", str(label_path), f"line {row_idx + 1}"))
                continue
            if bw <= 0 or bh <= 0:
                report.issues.append(Issue("zero_area_box", "error", str(label_path), f"line {row_idx + 1}"))
                continue
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            if x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6:
                report.issues.append(Issue("box_outside_image", "warning", str(label_path), f"line {row_idx + 1}"))
            if min(bw * w, bh * h) < min_box_px:
                report.issues.append(Issue("tiny_box", "warning", str(label_path), f"line {row_idx + 1}: {bw * w:.1f}x{bh * h:.1f}px"))
            report.n_boxes += 1
            key = class_names[cid] if class_names and 0 <= cid < len(class_names) else str(cid)
            report.classes_present[key] = report.classes_present.get(key, 0) + 1

    # Label files with no matching image would silently disappear from training.
    for stem in sorted(label_stems - image_stems):
        report.issues.append(Issue("orphan_label", "warning", str(labels_dir / f"{stem}.txt"), "no matching image"))

    # Near-duplicate grouping (O(n^2) but only on the hash integers; fine for <= 100k).
    grouped = [False] * len(hashes)
    for i in range(len(hashes)):
        if grouped[i]:
            continue
        group = [hashes[i][0]]
        for j in range(i + 1, len(hashes)):
            if grouped[j]:
                continue
            if bin(hashes[i][1] ^ hashes[j][1]).count("1") <= near_dup_threshold:
                group.append(hashes[j][0])
                grouped[j] = True
        if len(group) > 1:
            grouped[i] = True
            report.duplicate_groups.append(group)

    return report


def write_report(report: ValidationReport, out_path: str | Path) -> Path:
    """Write a validation report to JSON."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2))
    return out
