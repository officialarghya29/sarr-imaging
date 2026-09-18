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
from saryolo.data.yolo import load_data_config, project_root, resolve_data_yaml, split_dirs


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


def test_statistics_reports_sar_difficulty_proxies(synthetic):
    stats = profile_dataset(
        synthetic / "images" / "train", synthetic / "labels" / "train", ["target"], intensity_sample=4
    )
    assert stats.local_contrast["mean"] > 0
    assert stats.target_background_ratio["mean"] > 1.0, "synthetic targets are brighter than background"


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
        f"path: ../../processed/demo\ntrain: images/val\nval: images/val\nnc: 1\nnames: [x]\n"
    )
    resolved = resolve_data_yaml(cfg, root=tmp_path, out_dir=tmp_path / "resolved")
    root, loaded = load_data_config(resolved)
    assert root == dataset.resolve()
    images_dir, labels_dir = split_dirs(resolved)
    assert images_dir == dataset / "images" / "val"
    assert labels_dir == dataset / "labels" / "val"
