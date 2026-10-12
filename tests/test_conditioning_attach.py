"""The acquisition-metadata join the training path actually performs, and the guard around it.

The join is where a per-image descriptor meets the dataset that will emit it. Two failures are
quiet here and both are pinned:

* a split whose images are only *partially* covered by the table would attach metadata to some
  images and silently leave others to the model's unknown path -- so a partial table is refused
  rather than absorbed;
* an image-mixing augmentation would assign one acquisition label to a composite of two images,
  so the conditioned path refuses mosaic/mixup/cutmix/copy-paste instead of training on a label
  that is simply wrong.

These tests drive the join in :mod:`saryolo.training.trainer` -- the one the trainer and the
validator call -- rather than the copy that used to live in the data layer with no caller. That
copy could not express the metadata field mask, so the missing-metadata study could not have run
through it, and a second implementation of a join is a second thing that can drift; the tests
below therefore cover the live join *including* the mask, which is the capability that decided
which implementation stayed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ultralytics.data.dataset import YOLODataset

from saryolo.data.conditioning import _listed_images, validate_metadata_augmentations
from saryolo.data.field_mask import metadata_field_mask
from saryolo.data.metadata import AcquisitionMetadata, MetadataTable, Vocabulary
from saryolo.training.trainer import _attach_metadata


def _dataset(im_files: list[str]) -> YOLODataset:
    """A bare YOLODataset carrying only ``im_files``; enough to exercise the join."""
    ds = YOLODataset.__new__(YOLODataset)
    ds.im_files = list(im_files)
    return ds


# ------------------------------------------------------------------ the join

def test_attach_metadata_encodes_every_image_and_switches_the_class():
    table = MetadataTable(entries={
        "a": AcquisitionMetadata(sensor="sentinel1"),
        "b": AcquisitionMetadata(sensor="terrasarx"),
    })
    vocabs = table.build_vocabularies()
    ds = _dataset(["a.jpg", "b.jpg"])

    _attach_metadata(ds, table, vocabs)

    from saryolo.training.trainer import AcquisitionMetadataDetectionDataset

    assert isinstance(ds, AcquisitionMetadataDetectionDataset), "the dataset class was not switched"
    assert set(ds._acquisition_rows) == {"a", "b"}
    availability = ds._acquisition_rows["a"]["availability"]
    assert float(availability[-3]) == 1.0, "a recorded sensor must be flagged available"


def test_attach_metadata_uses_the_frozen_vocabulary_for_ids():
    table = MetadataTable(entries={
        "a": AcquisitionMetadata(sensor="sentinel1"),
        "b": AcquisitionMetadata(sensor="terrasarx"),
    })
    vocabs = table.build_vocabularies()
    ds = _dataset(["a.jpg", "b.jpg"])
    _attach_metadata(ds, table, vocabs)
    # CATEGORICAL_FIELDS order is (sensor, polarization, mode); entry 0 is the sensor.
    categorical = ds._acquisition_rows["a"]["categorical"]
    assert int(categorical[0]) == vocabs["sensor"].index("sentinel1")


def test_attach_metadata_honours_a_field_mask_the_dead_copy_could_not():
    """The missing-metadata study withholds fields at the data level, not the graph level.

    That is the capability that decided which of the two joins stayed: the live one can encode a
    record as though a field had never been recorded -- value zeroed *and* availability cleared,
    so it is indistinguishable from an absent field -- and the removed copy could not.
    """
    table = MetadataTable(entries={"a": AcquisitionMetadata(sensor="sentinel1", resolution_m=5.0)})
    vocabs = table.build_vocabularies()

    kept = _dataset(["a.jpg"])
    _attach_metadata(kept, table, vocabs)
    assert float(kept._acquisition_rows["a"]["availability"][0]) == 1.0
    assert float(kept._acquisition_rows["a"]["continuous"][0]) > 0.0

    masked = _dataset(["a.jpg"])
    _attach_metadata(masked, table, vocabs, field_mask=metadata_field_mask(("sensor",)))
    row = masked._acquisition_rows["a"]
    assert float(row["availability"][0]) == 0.0, "a withheld resolution must read as absent"
    assert float(row["continuous"][0]) == 0.0
    assert float(row["availability"][-3]) == 1.0, "the delivered sensor survives the mask"


def test_attach_metadata_refuses_a_non_yolo_dataset():
    with pytest.raises(TypeError, match="YOLO detection datasets"):
        _attach_metadata(object(), MetadataTable(), {})


def test_attach_metadata_refuses_a_partially_covered_split():
    """A missing row would leave that image on the unknown path without saying so."""
    table = MetadataTable(entries={"a": AcquisitionMetadata(sensor="sentinel1")})
    vocabs = table.build_vocabularies()
    with pytest.raises(ValueError, match="no row for"):
        _attach_metadata(_dataset(["a.jpg", "b.jpg"]), table, vocabs)


def test_attach_metadata_refuses_duplicate_stems():
    table = MetadataTable(entries={"a": AcquisitionMetadata(sensor="sentinel1")})
    vocabs = table.build_vocabularies()
    with pytest.raises(ValueError, match="stems are not unique"):
        _attach_metadata(_dataset(["a.jpg", "a.png"]), table, vocabs)


def test_vocabularies_are_a_frozen_interface():
    """``build_vocabularies`` returns the tables; they must be Vocabulary instances."""
    table = MetadataTable(entries={"a": AcquisitionMetadata(sensor="sentinel1")})
    vocabs = table.build_vocabularies()
    assert all(isinstance(v, Vocabulary) for v in vocabs.values())


# ------------------------------------------------------------------ augmentation guard

def test_validate_metadata_augmentations_passes_a_clean_pipeline():
    args = SimpleNamespace(mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0)
    validate_metadata_augmentations(args)  # must not raise


def test_validate_metadata_augmentations_refuses_an_image_mixing_transform():
    args = SimpleNamespace(mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0)
    with pytest.raises(ValueError, match="image-mixing"):
        validate_metadata_augmentations(args)


def test_validate_metadata_augmentations_only_names_the_active_transforms():
    args = SimpleNamespace(mosaic=0.0, mixup=0.3, cutmix=0.0, copy_paste=0.0)
    with pytest.raises(ValueError, match="mixup"):
        validate_metadata_augmentations(args)


# ------------------------------------------------------------------ image listing

def _write_image(path: Path) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))


def test_listed_images_reads_a_directory_recursively(tmp_path):
    _write_image(tmp_path / "images" / "a.jpg")
    _write_image(tmp_path / "images" / "sub" / "b.png")
    found = _listed_images(tmp_path / "images")
    assert {p.name for p in found} == {"a.jpg", "b.png"}


def test_listed_images_reads_a_list_file(tmp_path):
    _write_image(tmp_path / "a.jpg")
    listing = tmp_path / "train.txt"
    listing.write_text("a.jpg\n")
    assert [p.name for p in _listed_images(listing)] == ["a.jpg"]


def test_listed_images_refuses_a_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _listed_images(tmp_path / "absent")


def test_listed_images_refuses_an_empty_source(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no images"):
        _listed_images(tmp_path / "empty")


def test_listed_images_refuses_duplicate_stems(tmp_path):
    """Two images with the same stem would make a stem-keyed metadata join ambiguous."""
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b" / "a.png")
    with pytest.raises(ValueError, match="stems are not unique"):
        _listed_images(tmp_path)
