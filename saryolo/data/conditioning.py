"""Acquisition metadata for conditioned training: source listing, vocabularies, guards.

Metadata is joined to a dataset *after* geometric transforms, keyed by the image's stable
stem, and that join is implemented **once**, in :mod:`saryolo.training.trainer`
(``_attach_metadata`` with ``_metadata_descriptor``), which is what the training and
validation paths call. This module holds the pieces around the join that are not
``DetectionTrainer``-specific: enumerating a training source into image paths, fitting the
training-only vocabularies, preparing the model config from them, and the guard below.

Why the mixing guard is here and where it is called
--------------------------------------------------
A per-image descriptor describes exactly one image. An image-mixing transform -- mosaic,
MixUp, CutMix, CopyPaste -- builds one training input out of several, so it would hand a
composite the acquisition label of a single source, which is not a degradation the model can
learn around but a wrong label. ``validate_metadata_augmentations`` refuses such a request, and
it is called from both places that can see one: the runner, before a run starts (so a config
that asks for mixing on conditioned data fails with the reason instead of quietly training a
different pipeline than the one it states), and the trainer, before it builds its datasets.

One join, not two
-----------------
This module used to carry a second, parallel implementation of the join -- its own
``YOLODataset`` subclass and an ``attach_metadata_table`` attachment function -- which nothing
outside its tests called. It could not express the metadata field mask the live join supports
(so the missing-metadata study could not have run through it), and two joins that can drift
apart are worse than one that cannot, so it was removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from saryolo.data.metadata import CATEGORICAL_FIELDS, MetadataTable, Vocabulary

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CATEGORICAL_ORDER = CATEGORICAL_FIELDS


def _listed_images(value: str | Path | list[str]) -> list[Path]:
    """Enumerate image files from the directory/file-list forms accepted by YOLO."""
    values = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for entry in values:
        path = Path(entry).expanduser()
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        elif path.is_file():
            for line in path.read_text().splitlines():
                row = line.strip()
                if not row:
                    continue
                image = Path(row)
                if not image.is_absolute():
                    image = (path.parent / image).resolve()
                if image.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(image)
        else:
            raise FileNotFoundError(f"training image source does not exist: {path}")
    unique = {image.resolve() for image in images}
    if not unique:
        raise ValueError(f"no images found under training source(s) {values}")
    stems = [p.stem for p in unique]
    if len(stems) != len(set(stems)):
        raise ValueError("training image stems are not unique; acquisition metadata joins would be ambiguous")
    return sorted(unique)


def training_metadata_vocabularies(data: dict[str, Any]) -> tuple[dict[str, Vocabulary], dict[str, Any]] | None:
    """Build frozen categorical vocabularies from training images only."""
    table_path = data.get("acquisition_metadata")
    if not table_path:
        return None
    table = MetadataTable.load(table_path)
    images = _listed_images(data["train"])
    stems = [image.stem for image in images]
    missing = sorted(set(stems) - set(table.entries))
    if missing:
        raise ValueError(
            f"acquisition metadata has no row for {len(missing)} training image(s) "
            f"(e.g. {missing[:3]}); rebuild the table for the exact training split"
        )
    if not any(table.entries[stem].present for stem in stems):
        raise ValueError("training images have no known acquisition fields; conditioning would be a no-op")
    vocabs = table.build_vocabularies(stems)
    return vocabs, {key: value.to_dict() for key, value in vocabs.items()}


def prepare_conditioned_config(cfg: Any, vocabularies: dict[str, Vocabulary]) -> dict[str, Any]:
    """Set adapter embedding sizes from training-only vocabularies before model/DDP setup."""
    import yaml

    config = yaml.safe_load(Path(cfg).read_text()) if isinstance(cfg, (str, Path)) else cfg
    if not isinstance(config, dict):
        raise TypeError(f"conditioned model config must be a YAML mapping, got {type(config).__name__}")
    import copy

    config = copy.deepcopy(config)
    found = False
    for section in ("backbone", "head"):
        rows = []
        for original in config.get(section, []):
            row = list(original)
            if row[2] == "AcquisitionConditionedAdapter":
                args = list(row[3])
                if len(args) < 2 or args[-1] is None:
                    raise ValueError("conditioned adapter YAML row is missing its vocabulary-size argument")
                args[-1] = [max(vocabularies[field].size, 2) for field in CATEGORICAL_ORDER]
                row[3] = args
                found = True
            rows.append(row)
        config[section] = rows
    if not found:
        raise ValueError("dataset config supplies acquisition_metadata but the model has no conditioned adapter")
    return config


def validate_metadata_augmentations(args: Any) -> None:
    """Reject image-mixing augmentation for a per-image conditioned model.

    ``args`` is anything carrying the four mixing keys as attributes -- the trainer's resolved
    argument namespace, or a namespace built from a run's overrides. The refusal names only the
    transforms that are switched on, so the message is the whole diagnosis.
    """
    active = {
        key: getattr(args, key, 0.0)
        for key in ("mosaic", "mixup", "cutmix", "copy_paste")
        if float(getattr(args, key, 0.0) or 0.0) > 0
    }
    if active:
        raise ValueError(
            f"acquisition-conditioned training cannot use image-mixing augmentations {active}; "
            "set mosaic, mixup, cutmix and copy_paste to 0 so each image keeps one valid acquisition label"
        )
