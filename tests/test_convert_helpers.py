"""The COCO and DOTA converters, and the ``prepare_dataset`` dispatch.

``test_data.py`` covers the VOC path and the metadata sidecar. These are the other two
annotation formats the pipeline claims to support, plus the entry point that routes a
registry key to one of them. The properties checked are the ones that decide whether a
converted label file is usable: normalised centre coordinates, one row per object, and a
split that comes from the dataset rather than being invented.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from saryolo.data.convert import coco_to_yolo, dota_to_yolo_obb, prepare_dataset


def _write_image(path, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_coco_to_yolo_writes_normalised_labels_and_honours_the_coco_split(tmp_path):
    raw = tmp_path / "raw"
    _write_image(raw / "a.jpg")
    _write_image(raw / "b.jpg")
    ann = {
        "categories": [{"id": 1, "name": "ship"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64, "split": "train"},
            {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64, "split": "val"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [16, 16, 32, 32]},
            {"image_id": 2, "category_id": 1, "bbox": [0, 0, 32, 16]},
        ],
    }
    ann_path = tmp_path / "instances.json"
    ann_path.write_text(json.dumps(ann))

    out = tmp_path / "processed"
    result = coco_to_yolo(ann_path, raw, out, mode="copy")

    assert result["classes"] == ["ship"]
    assert set(result["splits"]) == {"train", "val"}
    train_label = out / "labels" / "train" / "a.txt"
    assert train_label.exists()
    fields = train_label.read_text().split()
    assert len(fields) == 5
    assert fields[0] == "0"
    numbers = [float(v) for v in fields[1:]]
    assert numbers == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_coco_to_yolo_raises_when_a_directory_holds_no_annotation_file(tmp_path):
    empty = tmp_path / "anns"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No COCO json"):
        coco_to_yolo(empty, tmp_path, tmp_path / "out")


def test_dota_to_yolo_obb_writes_eight_normalised_coordinates(tmp_path):
    dota = tmp_path / "dota"
    _write_image(dota / "chip.jpg", size=100)
    (dota / "chip.txt").write_text("10 10 40 10 40 30 10 30 ship\n")

    result = dota_to_yolo_obb(dota, tmp_path / "out", mode="copy")

    assert result["classes"] == ["ship"]
    label = tmp_path / "out" / "labels" / "train" / "chip.txt"
    fields = label.read_text().split()
    assert len(fields) == 9, "an OBB row is class + eight coordinates"
    assert fields[0] == "0"
    assert all(0.0 <= float(v) <= 1.0 for v in fields[1:]), fields


def test_prepare_dataset_refuses_a_missing_raw_download(tmp_path):
    """A missing download must name the source rather than fail deep inside a converter."""
    with pytest.raises(FileNotFoundError, match="Download it from"):
        prepare_dataset("ssdd", tmp_path / "absent")


def test_prepare_dataset_refuses_an_unsupported_registry_dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(KeyError):
        prepare_dataset("not_a_dataset", raw)


def test_prepare_dataset_dispatches_a_coco_dataset_and_records_the_output(tmp_path):
    raw = tmp_path / "raw"
    _write_image(raw / "images" / "a.jpg")
    ann = {
        "categories": [{"id": 1, "name": "ship"}],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64, "split": "train"}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [8, 8, 16, 16]}],
    }
    (raw / "instances_train.json").write_text(json.dumps(ann))

    result = prepare_dataset("hrsid", raw, out_root=tmp_path / "processed", mode="copy")
    assert result["dataset"] == "hrsid"
    assert result["output"].endswith("hrsid")
    assert result["classes"] == ["ship"]
