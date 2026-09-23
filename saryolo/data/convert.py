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

from .metadata import AcquisitionMetadata, MetadataTable
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
#: Verified acquisition profiles for the datasets this project conditions on.
#:
#: Sourced from the official dataset documentation (SARDet-100K's published source
#: table; HRSID's scene list), not inferred from filenames — a filename heuristic
#: would look like metadata while being a guess. Values are per-source constants:
#: they are the *coarsest true description* of each source, which is exactly the
#: signal the leave-one-source-out protocol needs, and exactly what the
#: representation probe tests for entanglement. Ranges are carried as the mid-point
#: of the published range plus the range itself in the profile, so nothing is
#: silently averaged away.
#: Keys are matched against image stems case-insensitively via prefix; the per-source
#: ``key_pattern`` states the stem shape.
SARDet_SOURCE_PROFILES: dict[str, dict] = {
    # name: (sensor, resolution range m, band, polarizations, satellites)
    "AIR_SARShip":  {"sensor": "gaofen3", "resolution_m": 2.0, "band": "C", "polarizations": ["VV"]},
    "HRSID":        {"sensor": "mixed", "resolution_m": 1.75, "resolution_range": [0.5, 3.0],
                     "band": "C/X", "polarizations": ["HH", "HV", "VH", "VV"]},
    "MSAR":         {"sensor": "hisea1", "resolution_m": 1.0, "band": "C", "polarizations": ["HH", "HV", "VH", "VV"]},
    "SADD":         {"sensor": "terrasarx", "resolution_m": 1.75, "resolution_range": [0.5, 3.0],
                     "band": "X", "polarizations": ["HH"]},
    "SAR-AIRcraft": {"sensor": "gaofen3", "resolution_m": 1.0, "band": "C", "polarizations": ["uni"]},
    "ShipDataset":  {"sensor": "mixed", "resolution_m": 14.0, "resolution_range": [3.0, 25.0],
                     "band": "C", "polarizations": ["HH", "VV", "VH", "HV"]},
    "SSDD":         {"sensor": "mixed", "resolution_m": 8.0, "resolution_range": [1.0, 15.0],
                     "band": "C/X", "polarizations": ["HH", "VV", "VH", "HV"]},
    "OGSOD":        {"sensor": "gaofen3", "resolution_m": 3.0, "band": "C", "polarizations": ["VV", "VH"]},
    "SIVED":        {"sensor": "airborne", "resolution_m": 0.2, "resolution_range": [0.1, 0.3],
                     "band": "Ka/Ku/X", "polarizations": ["VV", "HH"]},
}

#: HRSID standalone: three stated resolutions, Sentinel-1B / TerraSAR-X / TanDEM-X.
HRSID_PROFILE: dict = {
    "sensor": "mixed", "resolution_m": 1.5, "resolution_range": [0.5, 3.0],
    "band": "C/X", "polarizations": ["HH", "HV", "VH", "VV"],
}


def write_acquisition_metadata(
    images_dir: str | Path,
    out_path: str | Path,
    dataset: str,
) -> Path:
    """Write a :class:`MetadataTable` for a prepared dataset from its verified profile.

    This is the bridge between a downloaded dataset and the acquisition-conditioning
    experiments: without it, every conditioning arm and every resolution fold would
    need a hand-built sidecar CSV. The table is keyed by image stem, so it applies to
    the prepared (converted) layout directly.

    Args:
        images_dir: Prepared images directory (``images/<split>`` roots are scanned
            recursively, so one call covers all splits).
        out_path: Where to write the table JSON (consumed by ``loso --rule resolution``
            and by the conditioning trainer).
        dataset: Registry key whose profile to apply (``sardet100k``, ``hrsid``).

    Raises:
        ValueError: for a dataset with no verified profile, or a per-source profile
            that matched none of the images on disk (a table of unknowns would make
            every conditioning arm identical while the experiment still reported).
    """
    from .registry import get_dataset  # noqa: F401  (validates the registry key)

    get_dataset(dataset)
    profiles: dict[str, dict] = {}
    if dataset == "sardet100k":
        profiles = SARDet_SOURCE_PROFILES
    elif dataset == "hrsid":
        profiles = {}
    else:
        raise ValueError(
            f"no verified acquisition profile for {dataset!r}; add one sourced from the "
            "official dataset documentation, never inferred from filenames"
        )

    images_dir = Path(images_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    stems = sorted(p.stem for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not stems:
        raise ValueError(f"no dataset images found under {images_dir}")

    entries: dict[str, AcquisitionMetadata] = {}
    matched: set[str] = set()
    unmatched_images: list[str] = []

    def _norm(s: str) -> str:
        # Normalise BOTH sides, or "AIR_SARShip" (key) never matches "AIR_SARShip_1.0"
        # (stem): the key loses its underscore, the stem keeps it, and startswith fails
        # while looking like it ought to work.
        return s.upper().replace("-", "").replace("_", "").replace(" ", "")

    profile_keys = [(_norm(source), source, prof) for source, prof in profiles.items()]
    for stem in stems:
        fields: dict = {}
        if dataset == "hrsid":
            pols = HRSID_PROFILE["polarizations"]
            fields = {
                "sensor": HRSID_PROFILE["sensor"],
                "resolution_m": HRSID_PROFILE["resolution_m"],
                "polarization": (pols[0] if len(pols) == 1 else None),
                "mode": None,
                "band": HRSID_PROFILE["band"],
                "incidence_deg": None,
            }
        else:
            norm_stem = _norm(stem)
            for norm_key, source, prof in profile_keys:
                if norm_stem.startswith(norm_key):
                    matched.add(source)
                    fields = {
                        "sensor": prof["sensor"],
                        "resolution_m": prof["resolution_m"],
                        "polarization": (prof["polarizations"][0] if len(prof["polarizations"]) == 1 else None),
                        "mode": None,
                        "band": prof["band"],
                        "incidence_deg": None,
                    }
                    break
            if not fields:
                unmatched_images.append(stem)
        entries[stem] = AcquisitionMetadata.from_dict(fields)

    applied = matched if dataset == "sardet100k" else {"hrsid"}
    if not applied:
        raise ValueError(
            f"the {dataset} source profile matched none of the {len(stems)} image stems; "
            "check the per-source prefix layout before running the conditioning arms"
        )

    table = MetadataTable(entries=entries, source=f"profile:{dataset}")
    out = table.save(out_path)
    unmatched_sources = sorted(set(profiles) - matched) if dataset == "sardet100k" else []
    print(
        f"wrote {out}: {len(entries)} images | sources matched: "
        f"{', '.join(sorted(matched)) or '(standalone profile)'}"
        + (f" | UNMATCHED sources: {', '.join(unmatched_sources)}" if unmatched_sources else "")
        + (f" | UNMATCHED images: {len(unmatched_images)}" if unmatched_images else "")
    )
    return out
