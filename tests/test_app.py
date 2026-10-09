"""Tests for the Streamlit demo application.

Why this file exists
--------------------
The demo is the only surface a non-researcher touches, and it is the easiest place for a shown
number to drift from a measured one: a UI that drew its own boxes, or read its own confidence, or
quietly substituted a model when the checkpoint was missing, would display "a result" that the
model never produced. So the app is driven the way a *user* drives it -- startup, a missing
checkpoint, an unusable upload, and a full upload-through-display run -- and the end-to-end check
asserts the displayed detections are exactly what :func:`saryolo.inference.predict_image` returns
for the same chip.

Streamlit's own ``AppTest`` harness executes the real ``app.py`` in-process, so these are true
checks of the deployed entry point, not a re-implementation of it.

The checkpoint- and data-dependent cases skip with a *named* reason when the artefact is absent, so
the default suite still runs on a machine with neither the weights nor the dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="streamlit is not installed; the demo app is not tested here")

from streamlit.testing.v1 import AppTest  # noqa: E402  (after the importorskip)

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app.py"
WEIGHTS = REPO_ROOT / "results" / "runs" / "SSAC-001" / "weights" / "best.pt"
VAL_IMAGES = REPO_ROOT / "datasets" / "processed" / "hrsid_real" / "images" / "val"

needs_checkpoint = pytest.mark.skipif(
    not WEIGHTS.exists(),
    reason="no trained SSAC-001 checkpoint present; the demo's inference path is not run here",
)
needs_data = pytest.mark.skipif(
    not VAL_IMAGES.is_dir(),
    reason="HRSID subset not present; the upload path has no chip to run",
)


def _run(monkeypatch, weights: Path | str | None = None, timeout: int = 240) -> AppTest:
    """Start the real app, optionally pointing ``SARVO_WEIGHTS`` at a specific checkpoint.

    The environment is set *before* the first run because ``app.py`` reads its default checkpoint
    from the environment when it executes, which is what a deployment sets -- so this drives the
    same code path a hosted instance uses.
    """
    if weights is not None:
        monkeypatch.setenv("SARVO_WEIGHTS", str(weights))
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    return at


def _metric(at: AppTest, label: str) -> str:
    for m in at.metric:
        if m.label == label:
            return m.value
    return ""


def _texts(elements) -> str:
    return " ".join(getattr(e, "value", "") for e in elements)


# ------------------------------------------------------------------------------ startup
def test_the_app_starts_without_an_exception_and_renders_its_entry_point(monkeypatch):
    """Startup is the one thing every user hits; it must not raise, and it must offer the uploader.

    The app must also say *something* about the model -- the parameter panel when a checkpoint
    loads, or an explicit error when it does not. Showing neither would mean the page silently
    lost the model information a reader needs to interpret any result.
    """
    at = _run(monkeypatch)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("SARVO" in t.value for t in at.title)
    assert len(at.file_uploader) == 1, "the upload control is missing"
    assert at.metric or at.error, "no model information and no error: the page says nothing"


def test_the_app_with_no_checkpoint_says_so_and_offers_no_detection(monkeypatch, tmp_path):
    """A missing checkpoint is stated plainly and no detection control is offered.

    This is the anti-fabrication contract at the UI layer: untrained predictions look like real
    ones, so the app must not be able to produce boxes without weights.
    """
    at = _run(monkeypatch, tmp_path / "definitely_absent.pt")
    assert not at.exception, [str(e) for e in at.exception]
    assert "No checkpoint at" in _texts(at.error), "a missing checkpoint was not reported"
    assert "Recovery" in _texts(at.info), "the failure did not tell the user how to fix it"
    assert not at.metric, "the model panel rendered without a model"
    assert not any(b.label == "Run Detection" for b in at.button)


def test_the_app_says_when_no_checkpoint_is_configured_at_all(monkeypatch, tmp_path):
    """"Nothing configured" is a different message from "configured but missing".

    The distinction matters because the fix differs: the first needs ``SARVO_WEIGHTS`` set, the
    second needs a corrected path. Chdir'ing to an empty directory guarantees neither the
    environment nor the repository's default location supplies a checkpoint.
    """
    monkeypatch.delenv("SARVO_WEIGHTS", raising=False)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "No checkpoint is configured" in _texts(at.error), _texts(at.error)
    assert not any(b.label == "Run Detection" for b in at.button)


def test_the_app_refuses_a_checkpoint_that_cannot_be_loaded(monkeypatch, tmp_path):
    """A corrupt or non-checkpoint file is a reported error, not a crash and not a silent model."""
    bad = tmp_path / "garbage.pt"
    bad.write_bytes(b"this is not a checkpoint")
    at = _run(monkeypatch, bad)
    assert not at.exception, [str(e) for e in at.exception]
    assert "could not load the checkpoint" in _texts(at.error)
    assert not any(b.label == "Run Detection" for b in at.button)


# ------------------------------------------------------------------------ invalid uploads
def test_the_app_refuses_an_upload_that_is_not_an_image_and_says_how_to_fix_it(monkeypatch, tmp_path):
    """An unusable file gets an actionable message and no detection control -- never a guess.

    The checkpoint is deliberately absent so the model load is not part of this check: input
    validation happens *before* the model is consulted, which is what makes the refusal cheap.
    """
    at = _run(monkeypatch, tmp_path / "definitely_absent.pt")
    at.file_uploader[0].upload("scan.png", b"this is not a picture", "image/png")
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "cannot be used" in _texts(at.error), "an undecodable upload was not refused"
    assert "Recovery" in _texts(at.info)
    assert not any(b.label == "Run Detection" for b in at.button)


# ------------------------------------------------------------------ the full workflow (gated)
@needs_checkpoint
@needs_data
def test_an_uploaded_chip_runs_end_to_end_and_shows_the_pipeline_result(monkeypatch):
    """Upload → Run Detection → display, with the shown numbers equal to the pipeline's own.

    This is the acceptance criterion of the whole workflow: a valid image produces a visible
    result. The detections are compared against a direct :func:`predict_image` call on the same
    chip, so the test fails if the demo ever shows something the pipeline did not compute.
    """
    from saryolo.inference import load_detector, predict_image, validate_and_decode

    chip = sorted(VAL_IMAGES.glob("*.jpg"))[9]
    at = _run(monkeypatch, WEIGHTS)
    assert not at.exception, [str(e) for e in at.exception]

    at.file_uploader[0].upload(chip.name, chip.read_bytes(), "image/jpeg")
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any(b.label == "Run Detection" for b in at.button), "no detection control appeared"

    at.button[0].click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error, f"the app reported an error: {_texts(at.error)}"

    # What the app displayed...
    shown = int(_metric(at, "Detections"))
    assert _metric(at, "Parameters").endswith("M") and _metric(at, "Device") == "cpu"
    assert "ms" in _metric(at, "Latency"), "no measured latency was shown"

    # ...must be what the pipeline computes for the same chip and the same thresholds.
    model = load_detector(WEIGHTS, device="cpu")
    image = validate_and_decode(chip.read_bytes(), chip.name)
    expected = predict_image(
        model, image.array, imgsz=640, conf=0.25, iou=0.7, device="cpu", weights=str(WEIGHTS)
    )
    assert shown == expected.n_detections, "the demo's detection count is not the pipeline's"

    # Two images are rendered: the input and the annotated result.
    assert len(at.image) >= 2, "the input and/or the annotated result was not displayed"

    if expected.n_detections:
        assert len(at.dataframe) == 1, "detections were found but no table was shown"
        rows = at.dataframe[0].value.to_dict("records")
        assert len(rows) == expected.n_detections
        for row, want in zip(rows, expected.as_rows(), strict=True):
            assert int(row["class"]) == want["class"]
            assert float(row["confidence"]) == pytest.approx(want["confidence"], abs=1e-6)
