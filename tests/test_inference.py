"""Tests for the single-image inference pipeline.

Why this file exists
--------------------
:mod:`saryolo.inference` is the one path a CLI invocation, the Streamlit demo, and any future
service all stand on. That makes it exactly the place a defect would be invisible: if it silently
served predictions from an untrained model, or drew a box the model never produced, or accepted a
600 MB upload, every consumer would look fine while showing a number that is not a result. So the
checks here are written against *those* failure modes rather than against the happy path alone:

* a bad upload is refused **before** any pixel work, with a message that names the fix;
* a missing or unreadable checkpoint is a hard error, never a substitute model -- untrained
  predictions look exactly like real ones, so there is no safe fallback;
* the annotation is pure (the caller's array is untouched) and marks nothing when there is
  nothing to mark;
* on a genuine chip the boxes are inside the image, at or above the stated threshold, and the
  result serialises to the format the CLI and the demo both print.

Everything that needs the real artefact -- the SSAC-001 checkpoint and the HRSID val chips -- is
skipped with a *named* reason, so an absent weight file or dataset reads as "not run here", never
as a pass. The default suite therefore still runs on a machine with neither.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from saryolo.evaluation.metrics import Detection
from saryolo.inference import (
    DEFAULT_WEIGHT_CANDIDATES,
    MAX_SIDE,
    MAX_UPLOAD_BYTES,
    CheckpointError,
    InferenceError,
    InferenceResult,
    InvalidImageError,
    annotate_image,
    class_names,
    load_detector,
    predict_image,
    resolve_weights,
    validate_and_decode,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "results" / "runs" / "SSAC-001" / "weights" / "best.pt"
VAL_IMAGES = REPO_ROOT / "datasets" / "processed" / "hrsid_real" / "images" / "val"

needs_checkpoint = pytest.mark.skipif(
    not WEIGHTS.exists(),
    reason="no trained SSAC-001 checkpoint present; real inference not run here",
)
needs_data = pytest.mark.skipif(
    not VAL_IMAGES.is_dir(),
    reason="HRSID subset not present; real-chip inference not run here",
)


def _png(array) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", array)
    assert ok, "the test fixture could not encode a PNG"
    return buf.tobytes()


def _gray(side: int, value: int = 90):
    import numpy as np

    return np.full((side, side), value, np.uint8)


# ------------------------------------------------------------------- input validation refusals
def test_the_validator_refuses_every_bad_upload_and_names_the_fix():
    """Each rejection must be cheap, typed, and actionable -- not a crash deep in a decode.

    The checks run cheapest-first, so a clearly bad upload never costs a full image decode. The
    messages are asserted on their *content* because a UI shows them verbatim: a refusal that says
    only "invalid image" leaves the user with nothing to change.
    """
    import numpy as np

    # Wrong type, before any decode is attempted.
    with pytest.raises(InvalidImageError, match="not a supported image type"):
        validate_and_decode(b"GIF89a", "scan.gif")
    with pytest.raises(InvalidImageError, match="not a supported image type"):
        validate_and_decode(b"", "no_extension")

    # Empty, and over the size limit.
    with pytest.raises(InvalidImageError, match="empty"):
        validate_and_decode(b"", "scan.png")
    oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    assert len(oversized) > MAX_UPLOAD_BYTES, "the fixture is not actually over the limit"
    with pytest.raises(InvalidImageError, match="over the"):
        validate_and_decode(oversized, "scan.png")

    # Decodable-suffix bytes that are not an image.
    with pytest.raises(InvalidImageError, match="could not be decoded"):
        validate_and_decode(b"this is not a picture", "scan.png")

    # Dimensions: smaller than the stride can carry, and larger than the memory guard allows.
    with pytest.raises(InvalidImageError, match="minimum"):
        validate_and_decode(_png(_gray(16)), "tiny.png")
    wide = np.zeros((64, MAX_SIDE + 1), np.uint8)
    with pytest.raises(InvalidImageError, match="larger than"):
        validate_and_decode(_png(wide), "wide.png")


def test_the_validator_replicates_one_measured_channel_to_the_three_the_backbone_takes():
    """A SAR chip is one measured channel; the pipeline replicates it, exactly as the timed path does.

    If the three channels carried *different* content the demo would still run, but it would run on
    an input no measured number was ever taken on. Pinning the replication keeps a displayed result
    comparable with the profiled one.
    """
    import numpy as np

    payload = _png(_gray(64, value=120))
    image = validate_and_decode(payload, "chip.png")
    assert (image.width, image.height) == (64, 64)
    assert image.size == (64, 64)
    assert image.array.shape == (64, 64, 3)
    assert image.array.dtype == np.uint8
    assert np.array_equal(image.array[:, :, 0], image.array[:, :, 1])
    assert np.array_equal(image.array[:, :, 1], image.array[:, :, 2])
    assert int(image.array[0, 0, 0]) == 120


def test_the_validator_handles_a_jpeg_like_input_the_same_way(tmp_path):
    """The accepted-suffix list must actually be accepted (anti-vacuity for the refusal checks)."""
    import cv2
    import numpy as np

    for suffix in (".png", ".jpg", ".bmp", ".tif"):
        path = tmp_path / f"chip{suffix}"
        ok = cv2.imwrite(str(path), np.full((48, 48), 70, np.uint8))
        assert ok, f"could not write a {suffix} fixture"
        image = validate_and_decode(path.read_bytes(), path.name)
        assert (image.width, image.height) == (48, 48), suffix


# ----------------------------------------------------------------------- checkpoint refusals
def test_a_missing_checkpoint_is_refused_and_never_replaced_by_an_untrained_model(tmp_path):
    """A fallback to a fresh model is the worst possible behaviour, so it is asserted against.

    Predictions from randomly-initialised weights are indistinguishable, to a reader, from real
    ones -- there is no error, no empty result, just plausible-looking boxes. The loader must
    therefore name the missing path instead.
    """
    missing = tmp_path / "no_such_weights.pt"
    with pytest.raises(CheckpointError, match="checkpoint not found"):
        load_detector(missing)
    assert "no_such_weights" in str(
        pytest.raises(CheckpointError, load_detector, missing).value
    )


def test_an_unreadable_checkpoint_is_refused_with_its_path_in_the_message(tmp_path):
    """Garbage at a real path is refused, not half-loaded; the message keeps the path for triage."""
    bad = tmp_path / "garbage.pt"
    bad.write_bytes(b"this is not a checkpoint")
    assert bad.read_bytes(), "the fixture wrote nothing, so the check would be vacuous"
    with pytest.raises(CheckpointError, match="could not load the checkpoint"):
        load_detector(bad)


def test_the_single_sar_class_is_the_documented_default():
    """With no names on the model, the pipeline must default to the one trained class, not crash."""
    assert class_names(None) == {0: "ship"}


# ---------------------------------------------------------- which checkpoint a deployment serves
def test_weights_are_resolved_from_the_environment_but_an_explicit_path_still_wins(monkeypatch):
    """`SARVO_WEIGHTS` is the documented deployment knob, and a CLI `--weights` must beat it.

    A deployment sets the variable once; an operator running `saryolo predict --weights ...` on
    the same host must not be silently redirected to the deployed checkpoint.
    """
    monkeypatch.setenv("SARVO_WEIGHTS", "/deployed/chosen.pt")
    assert resolve_weights() == "/deployed/chosen.pt"
    assert resolve_weights("/explicit.pt") == "/explicit.pt"


def test_a_configured_but_missing_path_is_returned_so_the_error_names_it(monkeypatch):
    """Configured-but-absent is echoed back, so the failure names the path that was asked for."""
    monkeypatch.setenv("SARVO_WEIGHTS", "/nonexistent/asked_for.pt")
    assert resolve_weights() == "/nonexistent/asked_for.pt"


def test_with_nothing_configured_resolution_returns_none_not_a_guess(monkeypatch, tmp_path):
    """`None` must mean "nothing configured", which is a different fix from "configured but missing"."""
    monkeypatch.delenv("SARVO_WEIGHTS", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_weights() is None


def test_the_committed_weights_location_is_discovered_with_no_configuration(monkeypatch, tmp_path):
    """The documented default must actually be found -- otherwise the deploy instructions are wrong."""
    monkeypatch.delenv("SARVO_WEIGHTS", raising=False)
    target = tmp_path / "weights" / "sarvo_ssac001.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"placeholder checkpoint")
    monkeypatch.chdir(tmp_path)
    assert resolve_weights() == "weights/sarvo_ssac001.pt"
    # Anti-vacuity: the candidate list is what makes this work, not an empty-path accident.
    assert "weights/sarvo_ssac001.pt" in DEFAULT_WEIGHT_CANDIDATES


# --------------------------------------------------------------------------- pure annotation
def test_the_annotation_is_pure_and_marks_only_when_there_is_something_to_mark():
    """Rendering must not mutate the caller's array, and must change it exactly when it draws.

    The array may be a cached, shared image (the demo re-renders it on every interaction), so an
    annotator that drew in place would corrupt the input for the *next* request. And an annotator
    that "drew" on an empty detection list would make a miss look like a hit, which is the one
    thing a detection display must never do.
    """
    import numpy as np

    chip = _gray(96, value=40)
    original = chip.copy()

    empty = annotate_image(chip, ())
    assert np.array_equal(chip, original), "the annotation mutated the caller's array"
    assert np.array_equal(empty, original), "an empty detection list still changed the image"

    boxes = (Detection(image="", cls=0, xyxy=(20.0, 30.0, 60.0, 70.0), score=0.8),)
    drawn = annotate_image(chip, boxes, {0: "ship"})
    assert np.array_equal(chip, original), "the annotation mutated the caller's array"
    assert drawn.shape == original.shape and drawn.dtype == original.dtype
    assert not np.array_equal(drawn, original), "a real detection drew nothing"
    # The mark must appear *at the box*, not merely somewhere -- and the label it draws sits in a
    # strip above the box, so "untouched" is asserted far from both, not adjacent to either.
    assert np.any(drawn[30:70, 20:60] != original[30:70, 20:60]), "nothing was drawn on the box"
    assert np.array_equal(drawn[80:, :], original[80:, :]), "the annotation drew far below the box"
    assert np.array_equal(drawn[:, :14], original[:, :14]), "the annotation drew far left of the box"
    assert np.array_equal(drawn[:, 90:], original[:, 90:]), "the annotation drew far right of the box"


# ------------------------------------------------------------- non-finite detector output
class _FakeBoxes:
    """The slice of the detector's ``boxes`` object the pipeline reads, with no model behind it."""

    def __init__(self, xyxy, conf, cls=0):
        import torch

        # A real detector emits one class id per box; broadcast a scalar so a single-box fake
        # is shaped like the real thing rather than as a 0-d tensor.
        if not isinstance(cls, (list, tuple)):
            cls = [cls] * len(conf)
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.conf.shape[0])


class _FakeModel:
    """A stand-in detector, so a numerical fault can be injected without a real checkpoint."""

    names = {0: "ship"}

    def __init__(self, xyxy, conf, cls=0, orig_shape=(64, 64)):
        class _Result:
            pass

        result = _Result()
        result.boxes = _FakeBoxes(xyxy, conf, cls)
        result.orig_shape = orig_shape
        self._results = [result]

    def predict(self, **_kwargs):
        return self._results


def test_a_finite_detector_output_is_converted_before_the_guard_is_trusted():
    """Non-vacuity: the stability guard must not fire on a healthy detector output."""
    import numpy as np

    model = _FakeModel(xyxy=[[10.0, 20.0, 30.0, 40.0]], conf=[0.9])
    result = predict_image(model, np.zeros((64, 64, 3), "uint8"), imgsz=64)
    assert result.n_detections == 1
    assert result.detections[0].xyxy == (10.0, 20.0, 30.0, 40.0)
    assert result.detections[0].score == pytest.approx(0.9)
    assert (result.image_height, result.image_width) == (64, 64)


def test_a_non_finite_detection_is_refused_rather_than_reported():
    """A NaN or inf box is not a detection; the pipeline must refuse it, not draw it.

    Both failure modes are silent by default: a ``nan`` coordinate renders as a stray mark and a
    ``nan`` score as an unreadable label, while the run still looks successful. Dropping the box
    would hide a real numerical fault in the forward pass, so the pipeline raises instead.
    """
    import numpy as np

    image = np.zeros((64, 64, 3), "uint8")
    for bad in ([[10.0, 20.0, float("nan"), 40.0]], [[0.0, 0.0, 10.0, float("inf")]]):
        with pytest.raises(InferenceError, match="non-finite"):
            predict_image(_FakeModel(xyxy=bad, conf=[0.9]), image, imgsz=64)
    with pytest.raises(InferenceError, match="non-finite"):
        predict_image(
            _FakeModel(xyxy=[[10.0, 20.0, 30.0, 40.0]], conf=[float("nan")]), image, imgsz=64
        )


def test_an_empty_detector_output_is_a_valid_result_not_an_error():
    """No detections is a real outcome for a pilot model and must not be treated as a failure."""
    import numpy as np

    model = _FakeModel(xyxy=[], conf=[])
    result = predict_image(model, np.zeros((64, 64, 3), "uint8"), imgsz=64)
    assert result.n_detections == 0
    assert result.as_rows() == []
    assert result.as_dict()["n_detections"] == 0


# --------------------------------------------------------------- real inference (gated)
def _val_chips(limit: int = 12) -> list[Path]:
    return sorted(VAL_IMAGES.glob("*.jpg"))[:limit]


@needs_checkpoint
@needs_data
def test_a_real_chip_produces_boxes_inside_the_image_at_the_stated_threshold():
    """The end-to-end contract: real weights, real pixels, geometry that is actually admissible.

    A detection box that leaves the image, or a score below the threshold the result advertises,
    would be a rendering/plumbing bug that reads as a model result. The check spans several chips
    and requires the *set* to contain detections, so it cannot pass by returning nothing.
    """
    model = load_detector(WEIGHTS, device="cpu")
    chips = _val_chips()
    assert chips, "no val chips were found, so the check would be vacuous"

    seen = 0
    for path in chips:
        image = validate_and_decode(path.read_bytes(), path.name)
        result = predict_image(
            model, image.array, imgsz=640, conf=0.25, iou=0.7, device="cpu", weights=str(WEIGHTS)
        )
        assert isinstance(result, InferenceResult)
        assert result.latency_ms > 0.0
        assert (result.image_width, result.image_height) == (image.width, image.height)
        assert result.conf == 0.25 and result.iou == 0.7 and result.imgsz == 640
        assert result.device == "cpu"
        for det in result.detections:
            x1, y1, x2, y2 = det.xyxy
            assert 0.0 <= x1 < x2 <= result.image_width, f"{path.name}: x outside the image {det.xyxy}"
            assert 0.0 <= y1 < y2 <= result.image_height, f"{path.name}: y outside the image {det.xyxy}"
            assert 0.25 <= det.score <= 1.0, f"{path.name}: score below the stated threshold"
            assert det.cls == 0, f"{path.name}: unexpected class id {det.cls}"
            assert det.area > 0.0
        seen += result.n_detections
    assert seen > 0, "no chip in the sample produced a detection; the pipeline returns nothing"


@needs_checkpoint
@needs_data
def test_the_result_serialises_to_the_format_the_cli_and_the_demo_both_show():
    """One representation, serialisable: a dict for the CLI, rows for the table, both consistent.

    If the CLI's JSON and the demo's table disagreed, the same run would read as two different
    results. The fields asserted here are the contract both consumers depend on.
    """
    model = load_detector(WEIGHTS, device="cpu")
    path = _val_chips(1)[0]
    image = validate_and_decode(path.read_bytes(), path.name)
    result = predict_image(
        model, image.array, imgsz=640, conf=0.25, iou=0.7, device="cpu", weights=str(WEIGHTS)
    )

    payload = result.as_dict()
    assert json.loads(json.dumps(payload)) == payload, "the result is not JSON-serialisable"
    assert set(payload) >= {
        "n_detections", "detections", "latency_ms", "image_width", "image_height",
        "imgsz", "conf", "iou", "device", "weights",
    }
    assert payload["n_detections"] == result.n_detections == len(result.detections)
    rows = result.as_rows()
    assert len(rows) == result.n_detections
    for row, det in zip(rows, result.detections, strict=True):
        assert row["class"] == det.cls
        assert row["confidence"] == round(det.score, 4)
        assert len(row["xyxy"]) == 4 and row["area_px"] > 0.0


@needs_checkpoint
@needs_data
def test_repeating_one_chip_gives_the_same_boxes():
    """Eval-mode inference is deterministic; a drift here would make every demo number suspect."""
    model = load_detector(WEIGHTS, device="cpu")
    path = _val_chips(1)[0]
    image = validate_and_decode(path.read_bytes(), path.name)

    def run():
        r = predict_image(
            model, image.array, imgsz=640, conf=0.25, iou=0.7, device="cpu", weights=str(WEIGHTS)
        )
        return [(d.cls, tuple(round(v, 3) for v in d.xyxy), round(d.score, 5)) for d in r.detections]

    first = run()
    assert first == run(), "repeated inference on one chip is not deterministic"


@needs_checkpoint
@needs_data
def test_a_path_and_an_array_reach_the_same_result():
    """The entry point accepts a path or an array; both must be the same forward pass.

    A path is decoded by the underlying library and an array by our own validator -- if those two
    disagreed, the CLI and the demo would be running different preprocessing.
    """
    model = load_detector(WEIGHTS, device="cpu")
    path = _val_chips(1)[0]
    image = validate_and_decode(path.read_bytes(), path.name)

    def boxes(result):
        return [(d.cls, tuple(round(v, 2) for v in d.xyxy), round(d.score, 4)) for d in result.detections]

    from_array = predict_image(model, image.array, imgsz=640, conf=0.25, device="cpu")
    from_path = predict_image(model, path, imgsz=640, conf=0.25, device="cpu")
    assert boxes(from_array) == boxes(from_path)
