"""Tests for the corruption-sweep infrastructure.

``apply_corruption`` itself is covered by ``test_augmentation.py`` and ``test_cfar_scope.py``.
What had no test at all is the machinery around it: ``build_corrupted_split`` and
``robustness_sweep`` -- the part that decides whether a robustness number is measured on the
split it claims. The failure modes a running sweep would hide are:

* the corrupted split being built without its ground truth (labels sit beside the images), or
  with degraded labels, which would change where the targets are and not only how they look;
* a severity that fails being **dropped** from the result rather than recorded, so an arm that
  could not run reads as an arm with fewer severities instead of an unmeasured one;
* the result never being written where the analysis reads it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from saryolo.evaluation import robustness


def _tiny_split(root: Path, n: int = 3) -> tuple[Path, Path]:
    """A ``clean/images/val`` + ``clean/labels/val`` tree with distinct, non-flat chips."""
    import cv2

    images = root / "images" / "val"
    labels = root / "labels" / "val"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(images / f"chip_{i}.png"), np.full((24, 24), 40 + 50 * i, np.uint8))
        (labels / f"chip_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return images, labels


def test_build_corrupted_split_writes_images_and_places_labels_untouched(tmp_path):
    """Only the sensor signal is degraded; the target locations must be copied verbatim."""
    import cv2

    images, labels = _tiny_split(tmp_path / "clean")
    dest = robustness.build_corrupted_split(images, tmp_path / "out", "brightness", 0.5, seed=0)

    assert dest == tmp_path / "out" / "brightness" / "0.5" / "images" / "val"
    assert sorted(p.name for p in dest.glob("*.png")) == ["chip_0.png", "chip_1.png", "chip_2.png"]

    labels_dest = tmp_path / "out" / "brightness" / "0.5" / "labels" / "val"
    assert sorted(p.name for p in labels_dest.glob("*.txt")) == [
        "chip_0.txt", "chip_1.txt", "chip_2.txt",
    ]
    for name in ("chip_0.txt", "chip_1.txt", "chip_2.txt"):
        assert (labels_dest / name).read_text() == (labels / name).read_text(), (
            "ground truth was modified by a degradation that is supposed to be appearance-only"
        )

    # The images are genuinely degraded: gain 0.5 darkens, and the shape is preserved.
    clean = cv2.imread(str(images / "chip_0.png"), cv2.IMREAD_GRAYSCALE)
    corrupted = cv2.imread(str(dest / "chip_0.png"), cv2.IMREAD_GRAYSCALE)
    assert corrupted.shape == clean.shape
    assert corrupted.max() < clean.max(), "the corrupted chip is not darker than the clean one"


def test_build_corrupted_split_honours_the_limit(tmp_path):
    """``limit`` bounds the number of chips written, so a bounded sweep is bounded."""
    images, _ = _tiny_split(tmp_path / "clean", n=4)
    dest = robustness.build_corrupted_split(images, tmp_path / "out", "blur", 1.6, seed=0, limit=2)
    assert len(list(dest.glob("*.png"))) == 2


def _sweep_cfg(tmp_path: Path) -> Path:
    root = tmp_path / "clean"
    cfg = tmp_path / "data.yaml"
    cfg.write_text(f"path: {root}\nnc: 1\nnames:\n- ship\ntrain: images/val\nval: images/val\n")
    return cfg


def test_the_sweep_records_every_severity_it_measured(tmp_path, monkeypatch):
    """A measured severity must appear in the result and in the written report."""
    _tiny_split(tmp_path / "clean")
    cfg = _sweep_cfg(tmp_path)

    seen: list[str] = []

    def fake_eval(_weights, data_yaml, **_kwargs):
        seen.append(str(data_yaml))
        return {"mAP50": 0.4, "mAP50_95": 0.2}

    monkeypatch.setattr("saryolo.evaluation.metrics.evaluate_detections", fake_eval)

    result = robustness.robustness_sweep(
        "w.pt", cfg, out_dir=tmp_path / "out", imgsz=32, corruptions=("brightness",)
    )
    assert result["clean"] == {"mAP50": 0.4, "mAP50_95": 0.2}
    severities = result["corruptions"]["brightness"]
    assert sorted(severities) == sorted(str(s) for s in robustness.CORRUPTIONS["brightness"].severities)
    for row in severities.values():
        assert row == {"mAP50": 0.4, "mAP50_95": 0.2}
    assert len(seen) == 1 + len(robustness.CORRUPTIONS["brightness"].severities)

    written = json.loads((tmp_path / "out" / "robustness.json").read_text())
    assert written == result, "the sweep result on disk differs from the one returned"


def test_the_sweep_records_a_failed_severity_instead_of_dropping_it(tmp_path, monkeypatch):
    """A failing evaluation must leave a visible reason, not a missing row.

    The whole point of the sweep is to be read as a curve. A severity silently omitted would
    make the curve look complete while it had one fewer point, which is the failure the
    ``error`` key exists to prevent.
    """
    _tiny_split(tmp_path / "clean")
    cfg = _sweep_cfg(tmp_path)

    def fake_eval(_weights, data_yaml, **_kwargs):
        if "low_contrast" in str(data_yaml):
            raise RuntimeError("deliberate failure for the error-path test")
        return {"mAP50": 0.5, "mAP50_95": 0.25}

    monkeypatch.setattr("saryolo.evaluation.metrics.evaluate_detections", fake_eval)

    result = robustness.robustness_sweep(
        "w.pt", cfg, out_dir=tmp_path / "out", imgsz=32, corruptions=("low_contrast",)
    )
    severities = result["corruptions"]["low_contrast"]
    assert sorted(severities) == sorted(str(s) for s in robustness.CORRUPTIONS["low_contrast"].severities)
    for severity, row in severities.items():
        assert "error" in row, f"severity {severity} was dropped or fabricated: {row}"
        assert "deliberate failure" in row["error"]

    # The clean row is still a real measurement, so the failure did not erase the baseline.
    assert result["clean"] == {"mAP50": 0.5, "mAP50_95": 0.25}
    assert (tmp_path / "out" / "robustness.json").exists()


def test_the_sweep_refuses_an_unknown_corruption(tmp_path, monkeypatch):
    """A typo in the corruption list must raise, not silently sweep nothing."""
    _tiny_split(tmp_path / "clean")
    cfg = _sweep_cfg(tmp_path)
    monkeypatch.setattr(
        "saryolo.evaluation.metrics.evaluate_detections",
        lambda *_a, **_k: {"mAP50": 0.1, "mAP50_95": 0.05},
    )
    with pytest.raises(KeyError):
        robustness.robustness_sweep("w.pt", cfg, out_dir=tmp_path / "out", corruptions=("speckl",))
