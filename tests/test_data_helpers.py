"""The data-layer helpers: registry, YOLO path helpers, reports, validation, metadata.

Most of this module's behaviour is exercised through ``test_data.py`` and friends; these
are the public entry points and dataclass surfaces those files never named. The ones that
matter most are the metadata rules -- a rule that silently derives nothing produces a
table of all-unknown acquisitions, which turns a conditioning ablation into a comparison
of two identical models while still reporting a result.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from saryolo.data.convert import link_or_copy
from saryolo.data.metadata import (
    AcquisitionMetadata,
    MetadataRule,
    MetadataTable,
    Vocabulary,
    build_metadata_table,
    encode_metadata,
    load_metadata_sidecar,
)
from saryolo.data.registry import DATASETS, RECOMMENDED_ORDER, DatasetSpec, get_dataset, list_datasets
from saryolo.data.splits import LeakageReport
from saryolo.data.validate import ValidationReport, validate_yolo_dataset, write_report
from saryolo.data.yolo import dataset_root, label_row_kind

# ------------------------------------------------------------------ registry

def test_get_dataset_returns_the_spec_named_by_the_key():
    spec = get_dataset("ssdd")
    assert isinstance(spec, DatasetSpec)
    assert spec.name == "ssdd" and spec.nc == 1


def test_get_dataset_refuses_an_unknown_key():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("not_a_dataset")


def test_list_datasets_follows_the_recommended_order():
    """The returned order is the development order the module documents."""
    assert [s.name for s in list_datasets()] == [k for k in RECOMMENDED_ORDER if k in DATASETS]


def test_every_registered_dataset_names_its_licence_and_citation():
    """A dataset with no licence line cannot be used responsibly."""
    for key, spec in DATASETS.items():
        assert spec.license and spec.citation and spec.source, key
        assert spec.annotation_format in ("voc", "coco", "yolo", "dota"), key


# ------------------------------------------------------------------ yolo helpers

def test_label_row_kind_classifies_detection_oriented_and_malformed():
    assert label_row_kind(["0", "0.5", "0.5", "0.2", "0.2"]) == "detection"
    assert label_row_kind(["0"] + ["0.5"] * 8) == "oriented"
    assert label_row_kind(["0", "0.5"]) == "malformed"


def test_dataset_root_resolves_relative_to_the_processed_root(tmp_path):
    root = dataset_root(tmp_path, "hrsid_real")
    assert root == (tmp_path / "hrsid_real").resolve()


# ------------------------------------------------------------------ leakage report

def test_leakage_report_is_clean_only_without_cross_split_duplicates():
    report = LeakageReport(n_files=3, split_sizes={"train": 2, "val": 1})
    assert report.clean
    assert "ACTION REQUIRED" not in report.summary()
    report.near_duplicate_pairs.append(("train:a", "val:b"))
    assert not report.clean
    assert "ACTION REQUIRED" in report.summary()
    assert report.to_dict()["clean"] is False


# ------------------------------------------------------------------ validation

def test_validation_report_ok_is_true_without_error_severity_issues():
    report = ValidationReport(root="x")
    assert report.ok and report.errors == [] and report.warnings == []
    summary = report.summary()
    assert "errors: 0" in summary


def test_validation_report_to_dict_folds_issues_into_counts():
    from saryolo.data.validate import Issue

    report = ValidationReport(root="x", issues=[
        Issue("malformed_row", "error", "a.txt"),
        Issue("tiny_box", "warning", "b.txt"),
    ])
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["errors"] == 1 and payload["warnings"] == 1
    assert len(payload["issues"]) == 2


def _write_chip(path, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_validate_yolo_dataset_flags_a_missing_label_and_a_clean_one(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    _write_chip(images / "a.jpg")
    _write_chip(images / "b.jpg")
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    report = validate_yolo_dataset(images, labels, class_names=["ship"])
    kinds = {i.kind for i in report.issues}
    assert "missing_label" in kinds, "b.jpg has no label and must be reported"
    assert report.ok, "a missing label is a warning, not an error"
    assert report.n_images == 2 and report.n_boxes == 1


def test_validate_yolo_dataset_reports_a_missing_directory(tmp_path):
    report = validate_yolo_dataset(tmp_path / "absent", tmp_path / "also_absent")
    assert not report.ok
    assert any(i.kind == "missing_directory" for i in report.issues)


def test_validate_yolo_dataset_flags_an_out_of_range_class(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    _write_chip(images / "a.jpg")
    labels.mkdir()
    (labels / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n")
    report = validate_yolo_dataset(images, labels, class_names=["ship"])
    assert any(i.kind == "invalid_class_id" for i in report.issues)
    assert not report.ok


def test_write_report_creates_the_parent_directory(tmp_path):
    report = ValidationReport(root="x", n_images=1)
    out = write_report(report, tmp_path / "nested" / "report.json")
    assert out.exists()
    assert json.loads(out.read_text())["n_images"] == 1


# ------------------------------------------------------------------ acquisition records

def test_acquisition_metadata_reports_presence_and_a_grouping_key():
    meta = AcquisitionMetadata(sensor="sentinel1", resolution_m=10.0)
    assert meta.present == ("sensor", "resolution_m")
    assert meta.missing(("sensor", "band")) == ("band",)
    assert "sensor=sentinel1" in meta.key()


def test_acquisition_metadata_key_of_an_empty_record_is_named():
    assert AcquisitionMetadata().key() == "no-metadata"


def test_acquisition_metadata_from_dict_rejects_an_unknown_field():
    with pytest.raises(KeyError, match="unknown acquisition field"):
        AcquisitionMetadata.from_dict({"resolution": 10.0})


def test_acquisition_metadata_refuses_a_non_positive_resolution():
    with pytest.raises(ValueError, match="resolution_m must be positive"):
        AcquisitionMetadata(resolution_m=0.0)


# ------------------------------------------------------------------ vocabulary

def test_vocabulary_builds_sorted_and_reserves_the_unknown_index():
    vocab = Vocabulary.build("sensor", ["terrasarx", "sentinel1", None, "sentinel1"])
    assert vocab.entries == ["sentinel1", "terrasarx"]
    assert vocab.size == 3
    assert vocab.index("sentinel1") == 1 and vocab.index(None) == 0
    assert vocab.index("gaofen3") == 0, "an unseen value maps to the reserved unknown index"
    assert vocab.is_known("sentinel1") and not vocab.is_known("gaofen3")
    assert not vocab.is_known(None)


def test_vocabulary_from_dict_refuses_a_foreign_unknown_index():
    """A table built elsewhere may index differently; accepting it would corrupt the encoding."""
    with pytest.raises(ValueError, match="unknown_index"):
        Vocabulary.from_dict({"name": "sensor", "entries": ["a"], "unknown_index": 9})


# ------------------------------------------------------------------ metadata table

def _table() -> MetadataTable:
    return MetadataTable(entries={
        "a": AcquisitionMetadata(sensor="sentinel1", resolution_m=10.0),
        "b": AcquisitionMetadata(sensor="terrasarx"),
        "c": AcquisitionMetadata(),
    })


def test_metadata_table_coverage_counts_recorded_fields():
    coverage = _table().coverage()
    assert coverage["sensor"] == pytest.approx(2 / 3)
    assert coverage["band"] == 0.0


def test_metadata_table_groups_images_by_an_acquisition_key():
    grouped = _table().groups(by="sensor")
    assert grouped["sensor=sentinel1"] == ["a"]
    assert grouped["sensor=terrasarx"] == ["b"]


def test_metadata_table_build_vocabularies_respects_the_given_stems():
    """The whole point of the argument: a held-out sensor must not enter the vocabulary."""
    table = _table()
    table.build_vocabularies(stems=["a", "c"])
    assert table.vocabularies["sensor"].entries == ["sentinel1"]
    assert not table.vocabularies["sensor"].is_known("terrasarx")


def test_metadata_table_round_trips_through_disk(tmp_path):
    table = _table()
    table.build_vocabularies()
    path = table.save(tmp_path / "meta.json")
    reloaded = MetadataTable.load(path)
    assert reloaded.get("a").sensor == "sentinel1"
    assert reloaded.vocabularies["sensor"].entries == ["sentinel1", "terrasarx"]


def test_metadata_table_get_returns_an_unknown_record_for_a_missing_stem():
    assert _table().get("absent") == AcquisitionMetadata()


# ------------------------------------------------------------------ metadata rules

def test_a_directory_rule_needs_a_root_a_depth_and_a_field_map():
    with pytest.raises(ValueError, match="needs a root"):
        MetadataRule(kind="directory", field_map={0: "sensor"})
    with pytest.raises(ValueError, match="field_map"):
        MetadataRule(kind="directory", root="/x")
    with pytest.raises(ValueError, match="depth"):
        MetadataRule(kind="directory", root="/x", depth=0, field_map={0: "sensor"})


def test_a_filename_rule_needs_a_named_group_naming_a_known_field():
    with pytest.raises(ValueError, match="named groups"):
        MetadataRule(kind="filename", pattern=r"\d+")
    with pytest.raises(ValueError, match="unknown acquisition field"):
        MetadataRule(kind="filename", pattern=r"(?P<sensr>[a-z]+)")


def test_a_filename_rule_reads_the_named_groups():
    rule = MetadataRule(kind="filename", pattern=r"(?P<sensor>[A-Za-z0-9]+)_")
    assert rule.read(__import__("pathlib").Path("/x/sentinel1_0001.png")) == {"sensor": "sentinel1"}
    assert rule.read(__import__("pathlib").Path("/x/nomatch.png")) == {}


def test_build_metadata_table_refuses_without_a_rule_or_sidecar():
    with pytest.raises(ValueError, match="rule or a sidecar"):
        build_metadata_table(["a.jpg"])


def test_build_metadata_table_refuses_a_table_of_only_unknowns(tmp_path):
    """An all-unknown table would make every conditioning arm identical."""
    img = tmp_path / "chip1.png"
    img.write_bytes(b"")
    rule = MetadataRule(kind="filename", pattern=r"(?P<sensor>zzz)_")
    with pytest.raises(ValueError, match="no acquisition field"):
        build_metadata_table([img], rule=rule)


def test_build_metadata_table_uses_a_sidecar_over_a_rule(tmp_path):
    img = tmp_path / "chip1.png"
    img.write_bytes(b"")
    sidecar = {"chip1": AcquisitionMetadata(sensor="gaofen3")}
    table = build_metadata_table([img], sidecar=sidecar)
    assert table.get("chip1").sensor == "gaofen3"


# ------------------------------------------------------------------ sidecar

def test_load_metadata_sidecar_reads_json(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"chip1": {"sensor": "sentinel1", "resolution_m": 10.0}}))
    values = load_metadata_sidecar(path)
    assert values["chip1"].sensor == "sentinel1"


def test_load_metadata_sidecar_reads_csv(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("stem,sensor,resolution_m\nchip1,sentinel1,10\n")
    values = load_metadata_sidecar(path)
    assert values["chip1"].resolution_m == 10.0


def test_load_metadata_sidecar_refuses_a_missing_stem_column(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("id,sensor\nchip1,sentinel1\n")
    with pytest.raises(ValueError, match="stem"):
        load_metadata_sidecar(path)


def test_load_metadata_sidecar_refuses_an_unrecognised_column(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("stem,sensr\nchip1,sentinel1\n")
    with pytest.raises(ValueError, match="unrecognised column"):
        load_metadata_sidecar(path)


def test_load_metadata_sidecar_raises_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata_sidecar(tmp_path / "absent.csv")


# ------------------------------------------------------------------ encoding

def test_encode_metadata_zeroes_the_value_and_clears_availability_when_absent():
    """'No incidence recorded' must not encode as 'zero incidence'."""
    continuous, categorical, availability = encode_metadata(AcquisitionMetadata())
    assert continuous[1] == 0.0, "an absent incidence angle encodes as zero *and* flagged off"
    assert availability[1] == 0.0
    assert availability[0] == 0.0 and availability[0] == 0.0


def test_encode_metadata_flags_a_recorded_but_zero_value_as_available():
    continuous, categorical, availability = encode_metadata(AcquisitionMetadata(incidence_deg=0.0))
    assert continuous[1] == 0.0 and availability[1] == 1.0


# ------------------------------------------------------------------ link_or_copy

def test_link_or_copy_places_the_file_at_the_destination(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dst = tmp_path / "out" / "dst.bin"
    link_or_copy(src, dst, mode="copy")
    assert dst.read_bytes() == b"payload"


def test_link_or_copy_replaces_an_existing_destination(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"new")
    dst = tmp_path / "dst.bin"
    dst.write_bytes(b"old")
    link_or_copy(src, dst)
    assert dst.read_bytes() == b"new"
