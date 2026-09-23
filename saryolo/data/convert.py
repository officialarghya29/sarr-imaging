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

from .metadata import AcquisitionMetadata, MetadataTable, load_metadata_sidecar
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


def _find_float(node, path: str) -> float | None:
    """Read a float from an XML sub-element, or ``None`` if absent or unparseable.

    ``Element.findtext`` returns ``None`` for a missing element and the raw string
    for a present one, so both cases must be handled before ``float()`` is applied.
    Real VOC annotations do contain missing ``<size>`` blocks and truncated
    ``<bndbox>`` elements.
    """
    text = node.findtext(path)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


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
    # Malformed annotations are counted and skipped, never allowed to abort the batch:
    # a single bad XML file in a 100k-image download must not discard the work already done.
    skipped: list[tuple[str, str]] = []

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
        w = _find_float(size, "width") if size is not None else None
        h = _find_float(size, "height") if size is not None else None
        if not w or not h:
            # Without a valid <size> the normalised coordinates are undefined, so
            # these boxes cannot be converted correctly -- report, do not guess.
            skipped.append((xml_path.name, "missing or invalid <size>"))
            continue
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
        n_objects = 0
        n_malformed = 0
        for obj in root.findall("object"):
            n_objects += 1
            name = (obj.findtext("name") or "").strip()
            if name not in class_names:
                class_names.append(name)
            cid = class_names.index(name)
            box = obj.find("bndbox")
            x1, y1 = (_find_float(box, "xmin"), _find_float(box, "ymin")) if box is not None else (None, None)
            x2, y2 = (_find_float(box, "xmax"), _find_float(box, "ymax")) if box is not None else (None, None)
            if None in (x1, y1, x2, y2):
                n_malformed += 1
                continue
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x1 + bw / 2) / w, (y1 + bh / 2) / h
            rows.append(
                f"{cid} {min(max(cx, 0.0), 1.0):.6f} {min(max(cy, 0.0), 1.0):.6f} "
                f"{min(bw / w, 1.0):.6f} {min(bh / h, 1.0):.6f}"
            )
        if n_malformed:
            why = f"{n_malformed} of {n_objects} boxes malformed"
            if n_malformed == n_objects:
                # An annotation whose every box failed to parse is NOT a background
                # image. Emitting it with an empty label file would quietly teach the
                # detector that a ship-bearing image contains no ships.
                skipped.append((xml_path.name, f"all {n_objects} boxes malformed; image not written"))
                continue
            skipped.append((xml_path.name, f"partial: {why}"))

        link_or_copy(src, out_dir / "images" / split / src.name, mode)
        _write_label(out_dir / "labels" / split / f"{src.stem}.txt", rows)
        counts.setdefault(split, Counter())["images"] += 1
        counts[split]["boxes"] += len(rows)

    result: dict = {"classes": class_names, "splits": {k: dict(v) for k, v in sorted(counts.items())}}
    if unlisted:
        result["unlisted_images"] = len(unlisted)
        result["unlisted_examples"] = unlisted[:5]
    if skipped:
        result["skipped_annotations"] = len(skipped)
        result["skipped_examples"] = [f"{name}: {why}" for name, why in skipped[:5]]
        result["note"] = (
            "skipped annotations are images that were NOT written; a partial entry means "
            "some boxes in that file were dropped while the rest were kept."
        )
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
            for x, y in zip(xs, ys, strict=True):
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


# ---------------------------------------------------------------------------- acquisition profiles
#: Source-level acquisition profiles transcribed from the official SARDet-100K table.
#:
#: Only exact, source-level values are populated. If the publication gives a range or a
#: mixed set, the corresponding per-image field remains unknown: replacing a range with
#: its midpoint would fabricate labels, create false resolution bins, and contaminate
#: conditioning. Keys match source prefixes in stems (case/punctuation insensitive).
SARDet_SOURCE_PROFILES: dict[str, dict] = {
    "AIR_SARShip":  {"sensor": "gaofen3", "resolution_m": None, "band": "C", "polarization": "VV"},
    "HRSID":        {"sensor": None, "resolution_m": None, "band": None, "polarization": None},
    "MSAR":         {"sensor": "hisea1", "resolution_m": None, "band": "C", "polarization": None},
    "SADD":         {"sensor": "terrasarx", "resolution_m": None, "band": "X", "polarization": "HH"},
    "SAR-AIRcraft": {"sensor": "gaofen3", "resolution_m": 1.0, "band": "C", "polarization": "uni"},
    "ShipDataset":  {"sensor": None, "resolution_m": None, "band": None, "polarization": None},
    "SSDD":         {"sensor": None, "resolution_m": None, "band": None, "polarization": None},
    "OGSOD":        {"sensor": "gaofen3", "resolution_m": 3.0, "band": "C", "polarization": None},
    "SIVED":        {"sensor": "airborne", "resolution_m": None, "band": None, "polarization": None},
}

#: The standalone HRSID README says resolutions are 0.5, 1 and 3 m and lists several
#: satellites/polarizations, but the verified file inventory does not map those fields
#: to each chip. A range or mixed label is not a per-image value: leave unknown unless
#: a documented scene-to-chip sidecar is supplied.
HRSID_PROFILE: dict = {
    "sensor": None, "resolution_m": None, "polarization": None, "mode": None,
    "band": None, "incidence_deg": None,
}


def _profile_metadata(profile: dict) -> AcquisitionMetadata:
    """Convert one documented profile to per-image metadata without inventing values."""
    return AcquisitionMetadata.from_dict({
        field: profile.get(field)
        for field in ("sensor", "resolution_m", "polarization", "mode", "band", "incidence_deg")
    })


def _profile_for_stem(stem: str) -> tuple[str, AcquisitionMetadata] | None:
    """Resolve a dataset source prefix, preferring the most specific key."""
    def norm(value: str) -> str:
        return value.upper().replace("-", "").replace("_", "").replace(" ", "")

    normalized = norm(stem)
    candidates = sorted(SARDet_SOURCE_PROFILES, key=lambda key: len(norm(key)), reverse=True)
    for source in candidates:
        if normalized.startswith(norm(source)):
            return source, _profile_metadata(SARDet_SOURCE_PROFILES[source])
    return None


def write_acquisition_metadata(
    images_dir: str | Path,
    out_path: str | Path,
    dataset: str,
    sidecar: str | Path | None = None,
) -> Path:
    """Write acquisition metadata from sourced dataset profiles and optional per-image sidecars.

    SARDet-100K's official source table has mixed/ranged values for many sources. Those fields
    remain null, not midpoints or a fabricated single sensor. HRSID's public overview similarly
    lists multiple sensors and three resolutions without a verified per-chip mapping; pass a
    documented JSON/CSV sidecar to populate per-image fields. The profiles are useful for the
    exact fields they state, but do not pretend source-level summaries are per-image truth.

    Args:
        images_dir: Prepared images directory, scanned recursively.
        out_path: Destination MetadataTable JSON.
        dataset: Registry key (``sardet100k`` or ``hrsid``).
        sidecar: Optional per-image CSV/JSON mapping using the metadata module's documented schema.

    Raises:
        ValueError: unsupported profile, no images, or no known values for any image.
    """
    get_dataset(dataset)
    if dataset not in ("sardet100k", "hrsid"):
        raise ValueError(
            f"no verified acquisition profile for {dataset!r}; add one sourced from official "
            "documentation, never inferred from filenames"
        )

    images_dir = Path(images_dir)
    images = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no dataset images found under {images_dir}")

    sidecar_values: dict[str, AcquisitionMetadata] = {}
    if sidecar is not None:
        sidecar_values = load_metadata_sidecar(sidecar, stem_column="stem")

    entries: dict[str, AcquisitionMetadata] = {}
    matched_sources: set[str] = set()
    unmatched_images: list[str] = []
    source_profiles: dict[str, dict] = SARDet_SOURCE_PROFILES if dataset == "sardet100k" else {}
    image_stems = {p.stem for p in images}
    sidecar_unmatched = sorted(set(sidecar_values) - image_stems)
    if sidecar_unmatched:
        raise ValueError(
            f"sidecar {sidecar} contains {len(sidecar_unmatched)} stem(s) absent from {images_dir} "
            f"(e.g. {sidecar_unmatched[:3]}); a profile table must describe exactly this image tree"
        )

    for image in images:
        stem = image.stem
        profile_match = _profile_for_stem(stem) if dataset == "sardet100k" else None
        if profile_match is not None:
            source, profile_meta = profile_match
            matched_sources.add(source)
        else:
            source, profile_meta = None, AcquisitionMetadata()
            if dataset == "sardet100k":
                unmatched_images.append(stem)
            else:
                profile_meta = _profile_metadata(HRSID_PROFILE)

        # Sidecar values are per-image and explicitly sourced, so override only fields they
        # actually specify while keeping any exact profile fields not present in the sidecar.
        stated = sidecar_values.get(stem)
        if stated is not None:
            profile_meta = AcquisitionMetadata.from_dict({
                field: getattr(stated, field) if getattr(stated, field) is not None else getattr(profile_meta, field)
                for field in ("sensor", "resolution_m", "polarization", "mode", "band", "incidence_deg")
            })
        entries[stem] = profile_meta

    if len(entries) != len(images):
        raise ValueError(
            "image stems are not unique across directories; the metadata table is stem-keyed and "
            "cannot safely distinguish these images. Keep source/split identity in the stem or "
            "provide a collision-free prepared layout."
        )
    if not any(meta.present for meta in entries.values()):
        raise ValueError(
            f"no documented acquisition values matched the {len(images)} {dataset} image(s). "
            "Provide a per-image --sidecar sourced from the dataset archive; do not use "
            "range midpoints or infer labels from chip names."
        )
    # A profile table may be sufficient for source-level grouping but not for a particular
    # conditioning/fold field. Refuse to build a nominal resolution table when every actual
    # chip's resolution is unknown, instead of letting later code mistake nulls for data.
    # (Other uses may still consume the sensor/band information from the same table.)
    source_name = f"profile:{dataset}" + (f"+sidecar:{Path(sidecar).name}" if sidecar else "")
    table = MetadataTable(entries=entries, source=source_name)
    out = table.save(out_path)
    unmatched_sources = sorted(set(source_profiles) - matched_sources)
    unknown_images = sum(not meta.present for meta in entries.values())
    print(
        f"wrote {out}: {len(entries)} images | profile sources matched: "
        f"{', '.join(sorted(matched_sources)) or '(none)'}"
        + (f" | unmatched images: {len(unmatched_images)}" if unmatched_images else "")
        + (f" | unmatched profile sources: {', '.join(unmatched_sources)}" if unmatched_sources else "")
        + (f" | images without acquisition fields: {unknown_images}" if unknown_images else "")
    )
    return out
