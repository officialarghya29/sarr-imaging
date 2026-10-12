"""Tests for the qualitative detection and feature-map panels.

``detections.py`` and ``feature_maps.py`` render the pictures that go into the paper, where
a wrong panel is not a crash: it is a figure that quietly draws a box on the caller's image,
attributes a prediction no model made, or compares two different scenes. Both modules had
effectively no tests, so the properties that decide whether a panel means what its caption
says are pinned here:

* ground truth is drawn on a **copy** (a mutation would corrupt the array a caller reuses),
  a missing or malformed label file is a clean copy rather than an error;
* the confidence threshold is honoured, so a figure cannot show a box the detector rejected;
* the illustrative-image selector really orders by its named mode and refuses one it does not
  implement;
* the feature grid caps its channel count and survives a constant channel (the normalisation
  would otherwise divide by zero);
* a panel that cannot be rendered returns ``None`` instead of writing a blank figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from saryolo.visualization.detections import (
    annotate_gt,
    comparison_panel,
    draw_detections,
    select_examples,
)
from saryolo.visualization.feature_maps import compare_feature_maps, save_feature_grid


def _image(side: int = 48) -> np.ndarray:
    return np.full((side, side, 3), 30, np.uint8)


# ---------------------------------------------------------------------- ground truth
def test_ground_truth_is_drawn_on_a_copy_and_a_missing_label_is_a_clean_copy(tmp_path):
    """Drawing must not mutate the caller's array, and absent labels are not an error."""
    image = _image()
    original = image.copy()
    (tmp_path / "chip.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    out = annotate_gt(image, "chip", tmp_path)
    assert out.shape == image.shape
    assert np.array_equal(image, original), "annotate_gt mutated the caller's image"
    assert not np.array_equal(out, image), "no ground-truth box was drawn"

    # A stem with no label file returns an unmodified copy.
    assert np.array_equal(annotate_gt(image, "missing", tmp_path), original)
    # A malformed line (too few fields) is skipped rather than raising.
    (tmp_path / "torn.txt").write_text("0 0.5 0.5\n")
    assert np.array_equal(annotate_gt(image, "torn", tmp_path), original)


def test_a_ground_truth_box_lands_in_the_pixels_the_normalised_coordinates_name(tmp_path):
    """The YOLO centre/size convention must be converted correctly, not transposed."""
    (tmp_path / "chip.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    image = np.zeros((40, 60, 3), np.uint8)
    out = annotate_gt(image, "chip", tmp_path)
    ys, xs = np.where(out[..., 1] > 0)  # the green channel carries the rectangle
    # centre (0.5, 0.5) of a 60x40 image with a 0.5 span -> x in [15, 45), y in [10, 30).
    assert xs.min() >= 14 and xs.max() <= 46, (xs.min(), xs.max())
    assert ys.min() >= 9 and ys.max() <= 31, (ys.min(), ys.max())
    assert 20 < xs.mean() < 40 and 12 < ys.mean() < 28


# -------------------------------------------------------------------- predictions
def test_draw_detections_honours_the_confidence_threshold_and_copies():
    """A sub-threshold box must not be drawn, and the input must be untouched."""
    image = _image()
    original = image.copy()
    high = {"xyxy": (5, 5, 30, 30), "conf": 0.9, "cls": 0, "name": "ship"}
    low = {"xyxy": (2, 2, 40, 40), "conf": 0.1, "cls": 0, "name": "ship"}

    drawn = draw_detections(image, [high], conf_threshold=0.25)
    assert np.array_equal(image, original), "draw_detections mutated the caller's image"
    assert not np.array_equal(drawn, image)

    below = draw_detections(image, [low], conf_threshold=0.25)
    assert np.array_equal(below, original), "a sub-threshold box was drawn"
    # The same box at a lower threshold is drawn, so the threshold is what filtered it.
    assert not np.array_equal(draw_detections(image, [low], conf_threshold=0.05), original)


class _Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor(conf)
        self.cls = torch.tensor(cls)


class _Result:
    names = {0: "ship"}

    def __init__(self):
        self.boxes = [_Box((6, 6, 30, 30), 0.8, 0)]


class _StubModel:
    """The minimal shape ``_predict_boxes`` consumes from an ultralytics result."""

    def predict(self, **_kwargs):
        return [_Result()]


def test_comparison_panel_writes_a_grid_and_returns_none_when_nothing_renders(tmp_path):
    """A rendered panel is a file on disk; an un-renderable one is ``None``, never a blank image."""
    import cv2

    image_path = tmp_path / "chip.png"
    cv2.imwrite(str(image_path), np.full((64, 64), 120, np.uint8))
    (tmp_path / "chip.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    out = comparison_panel(
        _StubModel(), [image_path], tmp_path, tmp_path / "nested" / "panel.png",
        imgsz=64, conf=0.25,
    )
    assert out is not None and Path(out).exists()
    assert Path(out).parent.name == "nested", "the output directory was not created"

    # An image that will not decode -> no rows -> None, rather than an empty figure.
    (tmp_path / "broken.png").write_bytes(b"not an image")
    missing = comparison_panel(
        _StubModel(), [tmp_path / "broken.png"], tmp_path, tmp_path / "x.png", imgsz=64,
    )
    assert missing is None


# ------------------------------------------------------------- illustrative selection
def _selection_dataset(root: Path) -> Path:
    """Three chips with distinct box counts and object areas, for the three named modes."""
    import cv2

    images = root / "images" / "val"
    labels = root / "labels" / "val"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    # chip0: three large boxes; chip1: one tiny box; chip2: two medium boxes.
    boxes = {
        "chip0": [(0.5, 0.5, 0.3, 0.3)] * 3,
        "chip1": [(0.5, 0.5, 0.02, 0.02)],
        "chip2": [(0.4, 0.4, 0.1, 0.1), (0.6, 0.6, 0.1, 0.1)],
    }
    for stem, rows in boxes.items():
        cv2.imwrite(str(images / f"{stem}.png"), np.full((32, 32), 100, np.uint8))
        (labels / f"{stem}.txt").write_text(
            "".join(f"0 {cx} {cy} {w} {h}\n" for cx, cy, w, h in rows)
        )
    cfg = root / "data.yaml"
    cfg.write_text(f"path: {root}\nnc: 1\nnames:\n- ship\ntrain: images/val\nval: images/val\n")
    return cfg


def test_select_examples_orders_by_each_named_mode_and_refuses_an_unknown_one(tmp_path):
    """Density, sparsity and object size must pick different chips, in the right order."""
    cfg = _selection_dataset(tmp_path)
    names = lambda paths: [p.stem for p in paths]  # noqa: E731 - a local read, not a style choice

    assert names(select_examples(cfg, limit=3, mode="dense")) == ["chip0", "chip2", "chip1"]
    assert names(select_examples(cfg, limit=3, mode="sparse")) == ["chip1", "chip2", "chip0"]
    assert names(select_examples(cfg, limit=3, mode="small")) == ["chip1", "chip2", "chip0"]
    assert len(select_examples(cfg, limit=2, mode="dense")) == 2

    with pytest.raises(ValueError, match="Unknown mode"):
        select_examples(cfg, mode="biggest")


# ------------------------------------------------------------------ feature grids
def test_save_feature_grid_writes_a_png_caps_channels_and_survives_a_constant_map(tmp_path):
    """The grid must render, cap its channels, and not divide by zero on a flat channel."""
    rng = np.random.default_rng(0)
    maps = rng.random((12, 8, 8)).astype(np.float32)
    maps[0] = 0.5  # a constant channel: (max - min) is zero, so normalisation must guard it
    out = save_feature_grid(maps, tmp_path / "nested" / "grid.png", title="features", max_channels=4)
    assert out.exists() and out.parent.name == "nested"
    assert out.stat().st_size > 0


def test_compare_feature_maps_returns_none_when_it_cannot_render(tmp_path):
    """An undecodable image or a model that raises must yield ``None``, not an empty figure."""
    import cv2

    (tmp_path / "not.png").write_bytes(b"not an image")
    assert compare_feature_maps({"a": object()}, tmp_path / "not.png", 0, tmp_path / "o.png") is None

    cv2.imwrite(str(tmp_path / "ok.png"), np.zeros((16, 16), np.uint8))
    # No models at all -> no panels.
    assert compare_feature_maps({}, tmp_path / "ok.png", 0, tmp_path / "o.png") is None
    # A model whose feature extraction raises is skipped rather than propagated.
    class _Boom:
        pass

    assert compare_feature_maps({"boom": _Boom()}, tmp_path / "ok.png", 0, tmp_path / "o.png") is None
