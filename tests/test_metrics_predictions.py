"""Reading predictions back: ``load_yolo_predictions`` and the list-split image index.

``evaluate_detections`` scores whatever this function returns, so a prediction that is
silently dropped here is a false negative in the metric without anything raising. The two
ways it can drop one are both pinned:

* the image cannot be found to look up its size, so the normalised box has no absolute
  coordinates -- the split may be a directory, or a ``.txt`` list whose images are elsewhere
  (the reason ``image_index`` exists);
* the row is too short to be a ``cls cx cy w h conf`` detection.
"""

from __future__ import annotations

import cv2
import numpy as np

from saryolo.evaluation.metrics import load_yolo_predictions


def _write_image(path, size=(40, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_predictions_are_scaled_to_absolute_pixels(tmp_path):
    _write_image(tmp_path / "images" / "a.jpg", size=(40, 80))  # h=40, w=80
    labels = tmp_path / "labels"
    labels.mkdir()
    # centre-normalised box: cx=.5 cy=.5 w=.25 h=.5 conf=.9
    (labels / "a.txt").write_text("0 0.5 0.5 0.25 0.5 0.9\n")

    dets = load_yolo_predictions(labels, tmp_path / "images")
    assert len(dets) == 1
    det = dets[0]
    assert det.image == "a"
    # x1 = (.5 - .125) * 80 = 30, y1 = (.5 - .25) * 40 = 10, x2 = 50, y2 = 30
    assert det.xyxy == (30.0, 10.0, 50.0, 30.0)
    assert det.score == 0.9


def test_predictions_for_an_unfindable_image_are_dropped(tmp_path):
    """No image size means no absolute box; the row cannot be scored."""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "missing.txt").write_text("0 0.5 0.5 0.2 0.2 0.5\n")
    assert load_yolo_predictions(labels, tmp_path / "images") == []


def test_predictions_use_an_explicit_image_index_when_given(tmp_path):
    """A list split has no images directory to probe, so the index supplies the sizes."""
    image = tmp_path / "elsewhere" / "a.png"
    _write_image(image)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.5 0.5 0.7\n")

    dets = load_yolo_predictions(labels, tmp_path / "no_such_dir", image_index={"a": image})
    assert len(dets) == 1 and dets[0].image == "a"


def test_a_short_row_is_skipped_not_scored_as_a_box():
    """A row without a confidence is not a detection; ``len(row) < 6`` must drop it."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_image(tmp / "images" / "a.jpg")
        labels = tmp / "labels"
        labels.mkdir()
        (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # no confidence
        assert load_yolo_predictions(labels, tmp / "images") == []
