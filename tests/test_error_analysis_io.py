"""The failure taxonomy's bookkeeping and its two writers.

``tests/test_hard_examples.py`` drives ``analyse_failures``; this file covers the
:class:`FailureSummary` views the hard-example miner reads back (``per_image`` over the
*complete* case list, not the capped ``examples`` view), and the two helpers that turn a
split into a clutter proxy and a summary into a report. The contrast map is checked for
both split forms -- a directory and a `.txt` list -- because a list split silently produced
an empty map under the old directory-only code, and an empty map means the clutter proxy
stops working without any error.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from saryolo.data.yolo import write_data_yaml
from saryolo.evaluation.metrics import Detection
from saryolo.visualization.error_analysis import (
    FailureCase,
    FailureSummary,
    analyse_failures,
    image_contrast_map,
    write_failure_report,
)


def _write_image(path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


# ------------------------------------------------------------------ FailureCase

def test_failure_case_records_its_coordinates_and_serialises():
    case = FailureCase("img1", "false_negative", cls=0, iou=0.0, score=0.0, xyxy=(1, 2, 3, 4))
    payload = case.to_dict()
    assert payload["image"] == "img1" and payload["xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert "false_negative" in repr(case)


# ------------------------------------------------------------------ FailureSummary views

def test_summary_keeps_every_case_and_caps_only_the_examples_view():
    summary = FailureSummary()
    for _ in range(9):
        summary.add(FailureCase("img", "false_positive", 0, 0.0, 0.5))
    assert len(summary.all_cases()) == 9, "the full case list must not be truncated"
    assert len(summary.examples["false_positive"]) == 5, "the text report stays capped"


def test_summary_per_image_counts_the_full_case_list():
    summary = FailureSummary()
    summary.add(FailureCase("a", "false_positive", 0, 0.0, 0.1))
    summary.add(FailureCase("a", "small_object_miss", 0, 0.0, 0.0))
    summary.add(FailureCase("b", "true_positive", 0, 0.8, 0.9))
    per_image = summary.per_image()
    assert per_image["a"]["false_positive"] == 1 and per_image["a"]["small_object_miss"] == 1
    assert per_image["b"]["true_positive"] == 1


def test_summary_text_and_dict_agree_on_the_counts():
    summary = FailureSummary()
    summary.n_gt, summary.n_det = 2, 3
    summary.add(FailureCase("a", "true_positive", 0, 0.9, 0.9))
    assert "true_positive: 1" in summary.summary()
    payload = summary.to_dict()
    assert payload["counts"] == {"true_positive": 1}
    assert payload["n_gt"] == 2 and payload["n_det"] == 3


# ------------------------------------------------------------------ report

def test_write_failure_report_creates_the_parent_directory(tmp_path):
    summary = FailureSummary()
    summary.add(FailureCase("a", "false_negative", 0, 0.0, 0.0))
    out = write_failure_report(summary, tmp_path / "nested" / "failures.json")
    assert out.exists()
    assert json.loads(out.read_text())["counts"] == {"false_negative": 1}


# ------------------------------------------------------------------ contrast map

def test_image_contrast_map_reads_a_directory_split(tmp_path):
    images = tmp_path / "images" / "val"
    _write_image(images / "a.jpg", value=100)
    _write_image(images / "b.jpg", value=200)
    data_yaml = write_data_yaml(tmp_path / "data.yaml", tmp_path, ["ship"], splits=("train", "val"))

    contrast = image_contrast_map(data_yaml, split="val")
    assert set(contrast) == {"a", "b"}
    assert all(v >= 0.0 for v in contrast.values())


def test_image_contrast_map_reads_a_list_file_split(tmp_path):
    """A split given as a `.txt` list must still yield a contrast map, not an empty one."""
    _write_image(tmp_path / "images" / "val" / "a.jpg")
    listing = tmp_path / "val.txt"
    listing.write_text("images/val/a.jpg\n")
    data_yaml = write_data_yaml(tmp_path / "data.yaml", tmp_path, ["ship"], splits=("train", "val"))
    # Overwrite the 'val' entry to point at the list file.
    text = data_yaml.read_text().replace("val: images/val", f"val: {listing}")
    data_yaml.write_text(text)

    contrast = image_contrast_map(data_yaml, split="val")
    assert contrast, "the list-split contrast map is empty, so the clutter proxy is dead"
    assert "a" in contrast


def test_image_contrast_map_honours_the_limit(tmp_path):
    images = tmp_path / "images" / "val"
    for i in range(4):
        _write_image(images / f"img{i}.jpg")
    data_yaml = write_data_yaml(tmp_path / "data.yaml", tmp_path, ["ship"], splits=("train", "val"))
    assert len(image_contrast_map(data_yaml, split="val", limit=2)) == 2


# ------------------------------------------------------------------ taxonomy sanity

def test_analyse_failures_categorises_a_miss_and_a_hit():
    """A matched detection is one outcome and an unmatched ground truth is another."""
    gts = [Detection("a", 0, (0.0, 0.0, 10.0, 10.0)), Detection("a", 0, (50.0, 50.0, 60.0, 60.0))]
    preds = [Detection("a", 0, (0.0, 0.0, 10.0, 10.0), score=0.9)]
    summary = analyse_failures(preds, gts, iou_threshold=0.5)
    assert summary.counts["true_positive"] == 1
    # The second ground truth was never detected and is COCO-small, so it is a small-object miss.
    assert summary.counts["small_object_miss"] == 1
