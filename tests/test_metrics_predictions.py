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

from saryolo.evaluation.metrics import clear_stale_predictions, load_yolo_predictions


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


def test_a_second_run_into_one_output_directory_does_not_inherit_the_first_runs_boxes(
    tmp_path, monkeypatch
):
    """Defect D-32: ``save_txt`` appends, so a re-used ``--out`` double-counted detections.

    Ultralytics' own docstring is explicit -- "existing contents of the file will not be
    overwritten; new results are appended" -- and ``predict_to_labels`` reuses its output
    directory with ``exist_ok=True`` while ``load_yolo_predictions`` reads every ``*.txt`` under
    it. Measured on 4 val chips with the committed checkpoint before the fix: two calls each
    wrote 3 label files and the detections read back went **11 -> 22**, i.e. the second run's
    mAP would have been computed over a detection set half of which came from the first run.
    The fake model below appends exactly as ultralytics does, so the test is faithful to the
    mechanism rather than to one call site.
    """
    from pathlib import Path

    from saryolo.evaluation.metrics import predict_to_labels

    images = tmp_path / "images"
    _write_image(images / "a.jpg")
    _write_image(images / "b.jpg")
    out = tmp_path / "preds"

    class _AppendingPredictor:
        """Two detections per image, appended to whatever is already there."""

        def predict(self, **kwargs):
            labels = Path(kwargs["project"]) / "labels"
            labels.mkdir(parents=True, exist_ok=True)
            for image in sorted(Path(kwargs["source"]).glob("*.jpg")):
                with open(labels / f"{image.stem}.txt", "a") as handle:
                    handle.write("0 0.5 0.5 0.2 0.2 0.9\n0 0.4 0.4 0.1 0.1 0.8\n")

    monkeypatch.setattr("saryolo.training.trainer.load_model", lambda *a, **k: _AppendingPredictor())

    first = load_yolo_predictions(predict_to_labels("w.pt", images, out_dir=out), images)
    second = load_yolo_predictions(predict_to_labels("w.pt", images, out_dir=out), images)
    assert len(first) == 4, f"expected 2 images x 2 detections, read {len(first)}"
    assert len(second) == len(first), (
        f"the second run read back {len(second)} detections against the first run's {len(first)}: "
        "stale prediction files from the earlier call were scored as this one's"
    )


def test_the_stale_sweep_leaves_ground_truth_alone(tmp_path):
    """The sweep must be able to run on a directory that also holds a dataset's own labels.

    A prediction row carries a confidence column and a ground-truth row does not, which is the
    only difference the sweep needs to be safe: pointing ``--out`` at a dataset directory (an
    easy mistake, and previously a silent one) cannot delete the ground truth.
    """
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "prediction.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")
    (labels / "ground_truth.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    removed = clear_stale_predictions(tmp_path)
    assert [p.name for p in removed] == ["prediction.txt"]
    assert (labels / "ground_truth.txt").exists(), "the sweep deleted a dataset's own label file"


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
