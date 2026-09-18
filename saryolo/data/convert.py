"""Converters from popular SAR annotation formats to YOLO format.

Supported inputs
----------------
``coco``  HRSID, SARDet-100K, and most modern benchmarks (``instances_*.json``).
``voc``   SSDD (one PASCAL-VOC XML per image).
``dota``  SAR-Ship-Dataset (oriented boxes) -> YOLO-OBB 8-coordinate format.
``yolo``  Already-converted data; passed through (validated but not rewritten).

Conversion writes into ``datasets/processed/<dataset>/`` and never touches the
raw download, so a conversion bug is always recoverable by re-running.
"""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .registry import DatasetSpec, get_dataset

__all__ = ["prepare_dataset", "coco_to_yolo", "voc_to_yolo", "dota_to_yolo_obb", "link_or_copy"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def link_or_copy(src: Path, dst: Path, mode: str = "symlink") -> None:
    """Place ``src`` at ``dst`` either by symlink (saves disk) or copy.

    Symlinks keep a 116k-image benchmark from duplicating tens of GB, but they
    break if the tree is moved, so ``copy`` is available and used by default when
    the source lives on a different filesystem (where ``os.link`` fails).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if mode == "copy":
        shutil.copyfile(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copyfile(src, dst)


def _index_images(images_dir: Path) -> dict[str, Path]:
    """Map every stem (and file name) under ``images_dir`` to its path."""
    index: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
            index.setdefault(path.name, path)
    return index


def _write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def coco_to_yolo(
    annotations: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    mode: str = "symlink",
    use_official_splits: bool = True,
) -> dict:
    """Convert COCO-format annotations to YOLO labels.

    Official COCO ``train``/``val``/``test`` splits are honoured when present:
    the project brief requires respecting an official split unless there is a
    strong methodological reason not to, and inventing a split here would make
    results incomparable with the dataset's published baselines.

    Args:
        annotations: Path to a COCO JSON file (or a directory of them).
        images_dir: Root directory containing the image files.
        out_dir: Output root; ``images/<split>`` and ``labels/<split>`` are created.
        mode: ``"symlink"`` or ``"copy"``.
        use_official_splits: Honour the COCO split field.

    Returns:
        A dict of per-split counts and the class names discovered.
    """
    ann_files = sorted(Path(annotations).glob("*.json")) if Path(annotations).is_dir() else [Path(annotations)]
    if not ann_files:
        raise FileNotFoundError(f"No COCO json found at {annotations}")

    out_dir = Path(out_dir)
    image_index = _index_images(Path(images_dir))
    counts: dict[str, Counter] = {}
    categories: dict[int, str] = {}

    for ann_file in ann_files:
        data = json.loads(ann_file.read_text())
        for cat in data.get("categories", []):
            categories[cat["id"]] = cat["name"]
        images = {img["id"]: img for img in data.get("images", [])}
        anns = data.get("annotations", [])
        by_image: dict[int, list[dict]] = {}
        for ann in anns:
            by_image.setdefault(ann["image_id"], []).append(ann)

        # Prefer the COCO split; fall back to the file name (e.g. instances_train.json).
        default_split = "train"
        for candidate in ("train", "val", "test"):
            if candidate in ann_file.stem.lower():
                default_split = candidate

        for img_id, meta in images.items():
            split = str(meta.get("split", default_split)) if use_official_splits else default_split
            split = {"valid": "val", "validation": "val", "testing": "test"}.get(split.lower(), split.lower())
            fname = meta["file_name"]
            src = image_index.get(Path(fname).stem) or image_index.get(Path(fname).name)
            if src is None:
                continue
            w, h = float(meta["width"]), float(meta["height"])
            link_or_copy(src, out_dir / "images" / split / Path(fname).name, mode)
            rows: list[str] = []
            for ann in by_image.get(img_id, []):
                x, y, bw, bh = (float(v) for v in ann["bbox"])
                if bw <= 0 or bh <= 0:
                    continue
                cid = int(ann["category_id"])
                names = sorted(categories)
                cid = names.index(cid) if cid in names else cid
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                rows.append(
                    f"{cid} {min(max(cx, 0.0), 1.0):.6f} {min(max(cy, 0.0), 1.0):.6f} "
                    f"{min(bw / w, 1.0):.6f} {min(bh / h, 1.0):.6f}"
                )
            _write_label(out_dir / "labels" / split / f"{Path(fname).stem}.txt", rows)
            counts.setdefault(split, Counter())["images"] += 1
            counts[split]["boxes"] += len(rows)

    return {
        "classes": [categories[k] for k in sorted(categories)],
        "splits": {k: dict(v) for k, v in sorted(counts.items())},
    }


def voc_to_yolo(
    voc_root: str | Path,
    out_dir: str | Path,
    mode: str = "symlink",
    official_split_files: dict[str, str] | None = None,
) -> dict:
    """Convert PASCAL-VOC XML annotations (SSDD) to YOLO labels.

    Args:
        voc_root: Root containing ``Annotations/`` and ``JPEGImages/``.
        out_dir: Output root.
        mode: ``"symlink"`` or ``"copy"``.
        official_split_files: Optional mapping ``{"train": "<path to txt>", ...}``
            listing image stems per split. SSDD ships such files and they should
            be used rather than a fresh random split.
    """
    voc_root, out_dir = Path(voc_root), Path(out_dir)
    ann_dir = voc_root / "Annotations"
    img_dir = voc_root / "JPEGImages"
    if not ann_dir.exists():
        raise FileNotFoundError(f"Expected VOC Annotations/ under {voc_root}")

    split_of: dict[str, str] = {}
    for split, list_path in (official_split_files or {}).items():
        for line in Path(list_path).read_text().split():
            split_of[Path(line.strip()).stem] = split
    has_official = bool(split_of)

    xml_files = sorted(ann_dir.glob("*.xml"))
    class_names: list[str] = []
    counts: dict[str, Counter] = {}
    unlisted: list[str] = []

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        fname = root.findtext("filename") or f"{xml_path.stem}.jpg"
        src = img_dir / fname
        if not src.exists():
            candidates = [p for p in img_dir.glob(f"{xml_path.stem}.*")]
            if not candidates:
                continue
            src = candidates[0]
        size = root.find("size")
        w, h = float(size.findtext("width")), float(size.findtext("height"))
        if has_official:
            # An official split was supplied: an entry outside it belongs to a split we were
            # not given. Skipping is safer than silently dumping it into train.
            split = split_of.get(xml_path.stem)
            if split is None:
                unlisted.append(xml_path.stem)
                continue
        else:
            split = "train"

        rows = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in class_names:
                class_names.append(name)
            cid = class_names.index(name)
            box = obj.find("bndbox")
            x1, y1 = float(box.findtext("xmin")), float(box.findtext("ymin"))
            x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x1 + bw / 2) / w, (y1 + bh / 2) / h
            rows.append(
                f"{cid} {min(max(cx, 0.0), 1.0):.6f} {min(max(cy, 0.0), 1.0):.6f} "
                f"{min(bw / w, 1.0):.6f} {min(bh / h, 1.0):.6f}"
            )
        link_or_copy(src, out_dir / "images" / split / src.name, mode)
        _write_label(out_dir / "labels" / split / f"{src.stem}.txt", rows)
        counts.setdefault(split, Counter())["images"] += 1
        counts[split]["boxes"] += len(rows)

    result = {"classes": class_names, "splits": {k: dict(v) for k, v in sorted(counts.items())}}
    if unlisted:
        result["unlisted_images"] = len(unlisted)
        result["unlisted_examples"] = unlisted[:5]
    return result


def dota_to_yolo_obb(dota_dir: str | Path, out_dir: str | Path, mode: str = "symlink") -> dict:
    """Convert DOTA-style oriented boxes to YOLO-OBB labels (8 normalised coords).

    Needed for the optional oriented-detection study (Component 6): SAR-Ship-Dataset
    annotates rotated boxes, and axis-aligned re-encoding would discard exactly the
    orientation information that study is meant to evaluate.
    """
    dota_dir, out_dir = Path(dota_dir), Path(out_dir)
    counts: dict[str, Counter] = {}
    class_names: list[str] = []
    for label_file in sorted(dota_dir.rglob("*.txt")):
        img = None
        for ext in IMAGE_SUFFIXES:
            candidate = label_file.with_suffix(ext)
            if candidate.exists():
                img = candidate
                break
        if img is None:
            continue
        w, h = _image_size(img)
        rows = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 9:
                continue
            name = parts[8]
            if name not in class_names:
                class_names.append(name)
            coords = [float(v) for v in parts[:8]]
            xs, ys = coords[0::2], coords[1::2]
            norm = []
            for x, y in zip(xs, ys):
                norm.extend((min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0)))
            rows.append(f"{class_names.index(name)} " + " ".join(f"{v:.6f}" for v in norm))
        split = "train"
        link_or_copy(img, out_dir / "images" / split / img.name, mode)
        _write_label(out_dir / "labels" / split / f"{img.stem}.txt", rows)
        counts.setdefault(split, Counter())["images"] += 1
        counts[split]["boxes"] += len(rows)
    return {"classes": class_names, "splits": {k: dict(v) for k, v in sorted(counts.items())}}


def _image_size(path: Path) -> tuple[int, int]:
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    return img.shape[1], img.shape[0]


def prepare_dataset(
    name: str,
    raw_dir: str | Path,
    out_root: str | Path = "datasets/processed",
    mode: str = "symlink",
    official_split_files: dict[str, str] | None = None,
) -> dict:
    """Convert a raw dataset download into the project's processed YOLO layout.

    Args:
        name: Registry key (``ssdd``, ``hrsid``, ``sardet100k``, ``sar_ship``).
        raw_dir: Directory holding the raw download.
        out_root: Root for processed datasets.
        mode: ``"symlink"`` or ``"copy"``.
        official_split_files: Split list files to honour, if the dataset ships them.

    Returns:
        Conversion summary including class names and per-split counts.
    """
    spec: DatasetSpec = get_dataset(name)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_root) / spec.name
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data for {spec.name!r} not found at {raw_dir}.\n"
            f"Download it from: {spec.source}\n"
            f"See docs/DATASETS.md for the licensed download routes."
        )

    if spec.annotation_format == "coco":
        jsons = sorted(raw_dir.rglob("*.json"))
        ann = next((j for j in jsons if "annot" in j.name.lower() or "instance" in j.name.lower()), None)
        if ann is None:
            raise FileNotFoundError(f"No COCO annotation json found under {raw_dir}")
        images_dir = ann.parent
        for candidate in ("images", "Images", "JPEGImages"):
            if (raw_dir / candidate).is_dir():
                images_dir = raw_dir / candidate
                break
        result = coco_to_yolo(ann, images_dir, out_dir, mode=mode)
    elif spec.annotation_format == "voc":
        result = voc_to_yolo(raw_dir, out_dir, mode=mode, official_split_files=official_split_files)
    elif spec.annotation_format == "dota":
        result = dota_to_yolo_obb(raw_dir, out_dir, mode=mode)
    elif spec.annotation_format == "yolo":
        result = {"classes": list(spec.classes), "splits": {}, "note": "already YOLO format"}
    else:
        raise ValueError(f"Unsupported annotation format {spec.annotation_format!r}")

    result["dataset"] = spec.name
    result["output"] = str(out_dir)
    result["nc"] = len(result.get("classes") or spec.classes)
    return result
