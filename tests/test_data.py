"""Unit tests for the dataset layer.

These test the checks that decide whether a dataset is usable: the validator must
actually catch malformed labels (otherwise it is just decoration), the size bins
must match the COCO convention used by the loss and metrics, and the synthetic
generator must produce a dataset the rest of the pipeline accepts.
"""

from __future__ import annotations

import numpy as np
import pytest

from saryolo.data.registry import DATASETS, RECOMMENDED_ORDER, get_dataset
from saryolo.data.statistics import COCO_SIZE_BINS, profile_dataset
from saryolo.data.synth import make_synthetic_dataset
from saryolo.data.validate import validate_yolo_dataset
from saryolo.data.yolo import (
    label_row_kind,
    load_data_config,
    project_root,
    resolve_data_yaml,
    split_dirs,
)

#: A valid YOLO-OBB row: class id followed by 8 normalised corner coordinates.
OBB_ROW = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> object:
    root = tmp_path_factory.mktemp("synthetic")
    make_synthetic_dataset(root, n_train=8, n_val=4, n_test=4, imgsz=96, seed=0)
    return root


# --------------------------------------------------------------------------- registry
def test_registry_entries_are_complete():
    for name, spec in DATASETS.items():
        assert spec.classes, f"{name} has no classes"
        assert spec.source.startswith("http"), f"{name} has no source URL"
        assert spec.license, f"{name} has no license recorded"
        assert spec.citation, f"{name} has no citation recorded"
        assert spec.tier in ("pilot", "benchmark")


def test_recommended_order_is_cheapest_first():
    """Development order must be pilot-tier first, then ascending in size.

    Both halves matter: the project rule is to prove the pipeline on a pilot
    dataset before spending GPU time on a benchmark, and within a tier each step
    should cost more than the last.
    """
    specs = [get_dataset(name) for name in RECOMMENDED_ORDER]
    tiers = [spec.tier for spec in specs]
    assert tiers == sorted(tiers, key={"pilot": 0, "benchmark": 1}.get), (
        f"all pilot-tier datasets must come before benchmark ones, got {tiers}"
    )
    for tier in ("pilot", "benchmark"):
        sizes = [spec.approx_images for spec in specs if spec.tier == tier]
        assert sizes == sorted(sizes), f"{tier} datasets should be ascending in size, got {sizes}"


# -------------------------------------------------------------------------- validator
def test_validator_accepts_generated_dataset(synthetic):
    report = validate_yolo_dataset(
        synthetic / "images" / "train", synthetic / "labels" / "train", class_names=["target"]
    )
    assert report.ok, report.summary()
    assert report.n_images == 8
    assert report.n_boxes > 0


def test_validator_catches_invalid_class_id(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    import cv2

    cv2.imwrite(str(images / "a.png"), np.zeros((32, 32), dtype=np.uint8))
    (labels / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n")  # class 7 does not exist
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    assert not report.ok
    assert any(i.kind == "invalid_class_id" for i in report.errors)


def test_validator_catches_malformed_rows(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    import cv2

    cv2.imwrite(str(images / "a.png"), np.zeros((32, 32), dtype=np.uint8))
    (labels / "a.txt").write_text("0 0.5 0.5 0.2\n")  # only 4 fields
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    assert any(i.kind == "malformed_row" for i in report.errors)


def test_validator_catches_zero_area_and_out_of_range(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    import cv2

    cv2.imwrite(str(images / "a.png"), np.zeros((64, 64), dtype=np.uint8))
    (labels / "a.txt").write_text("0 0.5 0.5 0.0 0.2\n0 2.0 0.5 0.2 0.2\n")
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    kinds = {i.kind for i in report.issues}
    assert "zero_area_box" in kinds
    assert "box_outside_image" in kinds


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (["0", "0.5", "0.5", "0.2", "0.2"], "detection"),
        (["0", "0.1", "0.1", "0.9", "0.1", "0.9", "0.9", "0.1", "0.9"], "oriented"),
        (["0", "0.5", "0.5", "0.2"], "malformed"),
        (["garbage"], "malformed"),
    ],
)
def test_label_row_kind(row, expected):
    """Five fields is detection, nine is oriented, anything else is malformed."""
    assert label_row_kind(row) == expected


def test_validator_diagnoses_oriented_labels_instead_of_calling_them_malformed(tmp_path):
    """A 9-field row is valid YOLO-OBB, not a broken detection label.

    Two silent failure modes are being guarded against. Reporting it as
    "malformed" buries the one actionable fact under thousands of identical
    errors, and dropping it entirely would let an oriented dataset validate as a
    perfectly good detection dataset while containing no usable boxes.
    """
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    import cv2

    cv2.imwrite(str(images / "a.png"), np.zeros((32, 32), dtype=np.uint8))
    (labels / "a.txt").write_text(OBB_ROW)

    report = validate_yolo_dataset(images, labels, class_names=["target"])
    kinds = {i.kind for i in report.issues}
    assert report.oriented_label_rows == 1
    assert "oriented_labels" in kinds
    assert "malformed_row" not in kinds, "oriented rows must not be reported as malformed"
    assert not report.ok, "an oriented dataset must not validate clean"
    assert report.n_boxes == 0


def test_validator_flags_orphan_labels(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (labels / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    assert any(i.kind == "orphan_label" for i in report.warnings)


# ------------------------------------------------------------------------- statistics
def test_coco_size_bins_are_area_based():
    """COCO thresholds are 32^2 and 96^2 in *area*; 32/96 are linear sizes."""
    assert COCO_SIZE_BINS["small"] == (0.0, 1024.0)
    assert COCO_SIZE_BINS["medium"] == (1024.0, 9216.0)
    assert COCO_SIZE_BINS["large"][1] == float("inf")


def test_statistics_classify_small_objects_correctly(synthetic):
    """A 6-26px synthetic target is COCO-small; misclassifying it as large would
    wrongly 'justify' the P2 head."""
    stats = profile_dataset(
        synthetic / "images" / "train",
        synthetic / "labels" / "train",
        ["target"],
        name="synthetic",
        split="train",
        intensity_sample=4,
    )
    assert stats.num_images == 8
    assert stats.num_boxes > 0
    assert stats.size_distribution.get("large", 0) == 0, "small synthetic targets were binned as large"


def test_statistics_does_not_silently_drop_oriented_labels(tmp_path):
    """An oriented dataset must not profile as a clean dataset with zero objects."""
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    import cv2

    cv2.imwrite(str(images / "a.png"), np.zeros((32, 32), dtype=np.uint8))
    (labels / "a.txt").write_text(OBB_ROW)

    stats = profile_dataset(images, labels, ["target"], intensity_sample=1)
    assert stats.num_boxes == 0
    assert stats.oriented_label_rows == 1
    assert "YOLO-OBB" in stats.summary(), "the profile must say why it found no boxes"
    assert stats.to_dict()["oriented_label_rows"] == 1


def test_statistics_reports_sar_difficulty_proxies(synthetic):
    stats = profile_dataset(
        synthetic / "images" / "train", synthetic / "labels" / "train", ["target"], intensity_sample=4
    )
    assert stats.local_contrast["mean"] > 0
    assert stats.target_background_ratio["mean"] > 1.0, "synthetic targets are brighter than background"


# ------------------------------------------------------------------------ converters
VALID_VOC_XML = """<annotation>
  <filename>good.jpg</filename>
  <size><width>64</width><height>64</height><depth>1</depth></size>
  <object>
    <name>ship</name>
    <bndbox><xmin>8</xmin><ymin>8</ymin><xmax>32</xmax><ymax>32</ymax></bndbox>
  </object>
</annotation>"""

#: A truncated download: the <size> block never made it into the file.
VOC_XML_NO_SIZE = """<annotation>
  <filename>nosize.jpg</filename>
  <object><name>ship</name><bndbox><xmin>8</xmin><ymin>8</ymin></bndbox></object>
</annotation>"""

#: <size> is fine but the bounding box is incomplete.
VOC_XML_TRUNCATED_BOX = """<annotation>
  <filename>truncated.jpg</filename>
  <size><width>64</width><height>64</height><depth>1</depth></size>
  <object><name>ship</name><bndbox><xmin>8</xmin><ymin>8</ymin></bndbox></object>
</annotation>"""


def test_voc_converter_survives_malformed_annotations(tmp_path):
    """A broken XML file must not abort a batch conversion.

    Real VOC downloads contain truncated annotations, and the converter runs over
    tens of thousands of files: raising part-way through used to discard every
    conversion already performed and surface an opaque 'NoneType has no attribute
    findtext' instead of naming the offending file.
    """
    import cv2

    from saryolo.data.convert import voc_to_yolo

    voc = tmp_path / "voc"
    (voc / "Annotations").mkdir(parents=True)
    (voc / "JPEGImages").mkdir(parents=True)
    for stem in ("good", "nosize", "truncated"):
        cv2.imwrite(str(voc / "JPEGImages" / f"{stem}.jpg"), np.zeros((64, 64), dtype=np.uint8))
    (voc / "Annotations" / "good.xml").write_text(VALID_VOC_XML)
    (voc / "Annotations" / "nosize.xml").write_text(VOC_XML_NO_SIZE)
    (voc / "Annotations" / "truncated.xml").write_text(VOC_XML_TRUNCATED_BOX)

    out = tmp_path / "out"
    result = voc_to_yolo(voc, out, mode="copy")

    assert result["splits"]["train"]["images"] == 1, "the valid annotation must still convert"
    assert result["skipped_annotations"] == 2
    assert any("nosize.xml" in ex for ex in result["skipped_examples"])
    # An image whose every box failed must not be written as a background image.
    assert not (out / "labels" / "train" / "truncated.txt").exists()
    assert not (out / "images" / "train" / "truncated.jpg").exists()
    assert any("truncated.xml" in ex for ex in result["skipped_examples"])
    assert (out / "labels" / "train" / "good.txt").exists()
    label = (out / "labels" / "train" / "good.txt").read_text().split()
    assert len(label) == 5, f"expected 'class cx cy w h', got {label}"
    # centre (20, 20) of 64x64, size 24x24 -> normalised
    assert float(label[1]) == pytest.approx(20 / 64, abs=1e-4)
    assert float(label[3]) == pytest.approx(24 / 64, abs=1e-4)


# ------------------------------------------------------------------------ data config
def test_project_root_is_detected():
    assert (project_root(__file__) / "pyproject.toml").exists()


def test_relative_path_resolves_without_touching_ultralytics_settings(tmp_path):
    """A relative `path` must resolve against the YAML, not ultralytics' datasets_dir."""
    dataset = tmp_path / "processed" / "demo"
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    cfg_dir = tmp_path / "configs" / "datasets"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "demo.yaml"
    cfg.write_text(
        "path: ../../processed/demo\ntrain: images/val\nval: images/val\nnc: 1\nnames: [x]\n"
    )
    resolved = resolve_data_yaml(cfg, root=tmp_path, out_dir=tmp_path / "resolved")
    root, loaded = load_data_config(resolved)
    assert root == dataset.resolve()
    images_dir, labels_dir = split_dirs(resolved)
    assert images_dir == dataset / "images" / "val"
    assert labels_dir == dataset / "labels" / "val"


# ---------------------------------------------------------------------- acquisition profiles
def test_sardet_source_profile_matches_the_real_stem_layouts():
    """The profiles must key on the actual SARDet-100K stem shapes; a prefix rule that
    matches in theory but not on real stems would leave sources without acquisition
    metadata and the conditioning arms comparing unknowns."""
    from saryolo.data.convert import write_acquisition_metadata
    from saryolo.data.metadata import MetadataTable

    stems = [
        "AIR_SARShip_1.0_001_0001", "HRSID_JPG_0001_0_800_10190_10990", "MSAR_000001",
        "SADD_0001", "SAR-AIRcraft_0001_0", "ShipDataset_000001", "SSDD_000001",
        "OGSOD_0001", "SIVED_0001",
    ]
    root = tmp_path_factory_factory()
    for s in stems:
        (root / f"{s}.jpg").write_bytes(b"x")
    out = write_acquisition_metadata(root, root / "meta.json", "sardet100k")
    table = MetadataTable.load(out)

    # Every declared source matched at least one stem...
    matched = {str(v.sensor) for v in table.entries.values() if v.sensor is not None}
    assert matched == {"gaofen3", "mixed", "hisea1", "terrasarx", "airborne"}
    # ...and every image carries a resolution, the field the cross-resolution rule bins.
    assert all(v.resolution_m is not None for v in table.entries.values())
    # Distinct resolutions exist, so resolution folds are actually buildable.
    assert len({v.resolution_m for v in table.entries.values()}) >= 4


def _tmp_root():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())


tmp_path_factory_factory = _tmp_root


def test_profile_refuses_when_nothing_matches():
    """A profile matching no stems must be a refusal, not an all-unknown table that
    silently makes every conditioning arm identical."""
    import pytest

    from saryolo.data.convert import write_acquisition_metadata

    root = _tmp_root()
    (root / "junk_001.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="matched none of"):
        write_acquisition_metadata(root, root / "meta.json", "sardet100k")


def test_profile_refuses_a_dataset_without_a_verified_profile():
    """Profiles are sourced from official documentation only; a dataset without one
    must fail loudly instead of inventing acquisition values."""
    import pytest

    from saryolo.data.convert import write_acquisition_metadata

    root = _tmp_root()
    (root / "x.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="no verified acquisition profile"):
        write_acquisition_metadata(root, root / "meta.json", "ssdd")


def test_hrsid_standalone_profile_applies_to_every_image():
    from saryolo.data.convert import write_acquisition_metadata
    from saryolo.data.metadata import MetadataTable

    root = _tmp_root()
    for s in ("P0001_0_800_10190_10990", "P0002_200_1000_11220_12020"):
        (root / f"{s}.jpg").write_bytes(b"x")
    out = write_acquisition_metadata(root, root / "meta.json", "hrsid")
    table = MetadataTable.load(out)
    assert all(v.resolution_m == 1.5 for v in table.entries.values())
