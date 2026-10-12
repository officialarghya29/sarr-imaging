"""Tests for the validator's detection branches and its report.

``tests/test_data.py`` proves the validator catches the *category* errors (bad class ids,
malformed rows, zero-area boxes, oriented labels, orphan labels). What was untested is most of
the individual finding branches -- a corrupt file, a zero-size image, a non-8-bit image, an
exact duplicate, a near-duplicate group, an empty label, a too-many-boxes warning, an
unparsable row, a non-finite coordinate, a sub-pixel box -- and the machine-readable report
itself. That matters because a validator fails *silently*: a branch that never fires returns a
clean report, and a clean report is indistinguishable from a clean dataset. So each branch is
driven here on an input that must trip it, and the report is checked to carry the finding it
found.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from saryolo.data.validate import validate_yolo_dataset, write_report


@pytest.fixture(autouse=True)
def _cv2():
    pytest.importorskip("cv2")


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    """An empty ``images``/``labels`` pair under the test's tmp directory."""
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return images, labels


def _write_image(path: Path, array: np.ndarray) -> None:
    import cv2

    cv2.imwrite(str(path), array)


# ------------------------------------------------------------------------- structural
def test_a_missing_images_directory_is_an_error_and_stops_early(tmp_path):
    report = validate_yolo_dataset(tmp_path / "nope", tmp_path / "labels")
    assert not report.ok
    assert [i.kind for i in report.errors] == ["missing_directory"]
    assert report.n_images == 0


def test_a_missing_labels_directory_is_an_error_beside_the_found_images(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "a.png", np.zeros((16, 16), np.uint8))
    report = validate_yolo_dataset(images, tmp_path / "no_labels")
    kinds = {i.kind for i in report.issues}
    assert "missing_directory" in kinds
    # The images were still found, and each is reported as lacking a label.
    assert report.n_images == 1
    assert "missing_label" in kinds


# ------------------------------------------------------------------------- image findings
def test_a_file_that_will_not_decode_is_a_corrupt_image_error(tmp_path):
    images, labels = _pair(tmp_path)
    (images / "broken.png").write_bytes(b"not an image")
    (labels / "broken.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    assert any(i.kind == "corrupt_image" for i in report.errors), report.summary()


def test_a_sixteen_bit_image_is_a_warning_not_a_silent_pass(tmp_path):
    images, labels = _pair(tmp_path)
    _write_image(images / "deep.png", (np.ones((16, 16), np.uint16) * 40000))
    (labels / "deep.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    kinds = {i.kind for i in report.warnings}
    assert "non_uint8_image" in kinds, report.summary()


def test_an_exact_duplicate_is_flagged_against_the_file_it_repeats(tmp_path):
    images, labels = _pair(tmp_path)
    rng = np.random.default_rng(0)
    array = (rng.random((24, 24)) * 255).astype(np.uint8)
    _write_image(images / "first.png", array)
    _write_image(images / "second.png", array)  # byte-identical
    for stem in ("first", "second"):
        (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    duplicates = [i for i in report.issues if i.kind == "exact_duplicate"]
    assert len(duplicates) == 1
    assert "first.png" in duplicates[0].detail, "the duplicate must name the file it repeats"


def test_near_duplicate_images_are_grouped_even_when_their_bytes_differ(tmp_path):
    """The dHash path catches content duplicates the MD5 check cannot see."""
    images, labels = _pair(tmp_path)
    ramp = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    _write_image(images / "a.png", ramp)
    jittered = ramp.copy()
    jittered[0, 0] = 255 - jittered[0, 0]  # one pixel changed: different md5, same dHash
    _write_image(images / "b.png", jittered)
    for stem in ("a", "b"):
        (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    assert not any(i.kind == "exact_duplicate" for i in report.issues), (
        "the two files were treated as byte-identical, so the near-duplicate path was not reached"
    )
    assert len(report.duplicate_groups) == 1
    assert {Path(p).stem for p in report.duplicate_groups[0]} == {"a", "b"}


# ------------------------------------------------------------------------- label findings
def test_an_empty_label_file_is_background_not_an_error(tmp_path):
    images, labels = _pair(tmp_path)
    _write_image(images / "bg.png", np.zeros((16, 16), np.uint8))
    (labels / "bg.txt").write_text("")
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    infos = [i for i in report.issues if i.kind == "empty_label"]
    assert len(infos) == 1 and infos[0].severity == "info"
    assert report.ok, "a background image must not fail the dataset"


def test_a_label_with_more_rows_than_the_limit_warns(tmp_path):
    images, labels = _pair(tmp_path)
    _write_image(images / "busy.png", np.zeros((32, 32), np.uint8))
    (labels / "busy.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.1 0.1\n")
    report = validate_yolo_dataset(images, labels, class_names=["target"], limits={"max_boxes": 1})
    assert any(i.kind == "too_many_boxes" for i in report.warnings)


def test_an_unparsable_row_and_a_non_finite_coordinate_are_distinct_errors(tmp_path):
    """A non-numeric field and a ``nan`` are different faults and must not collapse."""
    images, labels = _pair(tmp_path)
    _write_image(images / "a.png", np.zeros((32, 32), np.uint8))
    (labels / "a.txt").write_text("0 0.5 x 0.2 0.2\n0 nan 0.5 0.2 0.2\n")
    report = validate_yolo_dataset(images, labels, class_names=["target"])
    kinds = [i.kind for i in report.errors]
    assert "unparsable_row" in kinds
    assert "non_finite_coordinate" in kinds
    # Neither row is counted as a real box.
    assert report.n_boxes == 0


def test_a_sub_pixel_box_is_a_tiny_box_warning(tmp_path):
    images, labels = _pair(tmp_path)
    _write_image(images / "a.png", np.zeros((100, 100), np.uint8))
    (labels / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n")  # 1 px on a 100 px image
    report = validate_yolo_dataset(images, labels, class_names=["target"], min_box_px=2.0)
    assert any(i.kind == "tiny_box" for i in report.warnings), report.summary()
    assert report.n_boxes == 1


# ------------------------------------------------------------------------- the report itself
def test_the_report_serialises_issue_counts_and_round_trips_to_disk(tmp_path):
    """The report is consumed as JSON, so the counts must be in it and survive a write."""
    images, labels = _pair(tmp_path)
    _write_image(images / "a.png", np.zeros((32, 32), np.uint8))
    (labels / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n")  # invalid class id -> an error
    report = validate_yolo_dataset(images, labels, class_names=["target"])

    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["errors"] == len(report.errors) >= 1
    assert payload["warnings"] == len(report.warnings)
    assert payload["issues"] and all("kind" in i for i in payload["issues"])
    assert payload["n_images"] == 1

    out = write_report(report, tmp_path / "nested" / "report.json")
    assert out.exists() and out.parent.name == "nested"
    assert json.loads(out.read_text()) == payload

    text = report.summary()
    assert "invalid_class_id" in text, "the summary must name the issue kinds it counted"
    assert "images: 1" in text
