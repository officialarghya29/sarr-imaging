"""Attach sourced, per-image acquisition metadata to Ultralytics detection batches.

Metadata is joined after geometric transforms using the image's stable stem. Image-mixing
augmentations are incompatible with a single acquisition descriptor, so the conditioned
training path refuses them rather than assigning one source's metadata to a composite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics.data.dataset import YOLODataset

from saryolo.data.metadata import CATEGORICAL_FIELDS, MetadataTable, Vocabulary, encode_metadata

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CATEGORICAL_ORDER = CATEGORICAL_FIELDS


class AcquisitionMetadataYOLODataset(YOLODataset):
    """YOLO detection dataset that adds one encoded metadata record per image."""

    _metadata_by_stem: dict[str, tuple[list[float], list[int], list[float]]]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        stem = Path(self.im_files[index]).stem
        descriptor = self._metadata_by_stem.get(stem)
        if descriptor is None:
            raise KeyError(f"acquisition metadata has no row for image stem {stem!r}")
        continuous, categorical, availability = descriptor
        sample["metadata"] = {
            "continuous": torch.tensor(continuous, dtype=torch.float32),
            "categorical": torch.tensor(categorical, dtype=torch.long),
            "availability": torch.tensor(availability, dtype=torch.float32),
        }
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Use Ultralytics collation, then stack per-image acquisition vectors."""
        result = YOLODataset.collate_fn(batch)
        records = result.get("metadata")
        if records is not None:
            result["metadata"] = {
                field: torch.stack([record[field] for record in records], dim=0)
                for field in ("continuous", "categorical", "availability")
            }
        return result


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


def attach_metadata_table(
    dataset: YOLODataset,
    table: MetadataTable,
    vocabularies: dict[str, Vocabulary],
) -> None:
    """Join table fields to the constructed dataset using frozen training vocabularies."""
    if not isinstance(dataset, YOLODataset):
        raise TypeError(f"metadata conditioning supports YOLO detection datasets, got {type(dataset).__name__}")
    stems = [Path(path).stem for path in dataset.im_files]
    if len(stems) != len(set(stems)):
        raise ValueError("image stems are not unique in this split; metadata joins must be unambiguous")
    missing = sorted(set(stems) - set(table.entries))
    if missing:
        raise ValueError(
            f"acquisition metadata has no row for {len(missing)} dataset image(s) "
            f"(e.g. {missing[:3]}); rebuild the table for this exact split"
        )
    encoded = {stem: encode_metadata(table.get(stem), vocabularies) for stem in stems}
    try:
        dataset.__class__ = AcquisitionMetadataYOLODataset
    except TypeError as exc:
        raise TypeError(
            f"cannot attach acquisition metadata to {type(dataset).__name__}; "
            "the dataset class must support the metadata-aware YOLO subclass"
        ) from exc
    dataset._metadata_by_stem = encoded


def validate_metadata_augmentations(args: Any) -> None:
    """Reject image-mixing augmentation for a per-image conditioned model."""
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
