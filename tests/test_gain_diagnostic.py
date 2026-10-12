"""Tests for the gain-collapse diagnostic's plumbing, not its statistics.

``test_cfar_scope.py`` pins the *numbers* ``summarise_gain`` returns. What had no test is the
machinery that produces them from a checkpoint, and it is the half that can silently measure
the wrong thing:

* the split reader must refuse an image it cannot decode. It used to leave the pre-allocated
  **zero** row in place, folding a degenerate black chip (the input the front end's ``log``
  statistic is least defined on) into the pooled collapse statistics under a real chip's name.
  The efficiency loader refuses the identical case, so this was also an inconsistency.
* the diagnostic must apply exactly the named synthetic shifts, skip ``identity`` as a
  redundant copy of the clean row, and write the report where the analysis reads it;
* it must refuse a checkpoint with no front end (reporting a constant zero gain for a stock
  detector would be a lie) and an empty split (an empty diagnostic is not a measurement).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import saryolo  # noqa: F401  (registers custom layers so a checkpoint unpickles)
from saryolo.evaluation.gain import (
    DEFAULT_SHIFTS,
    _read_gray,
    diagnose_gain,
    find_front_end,
    gain_maps,
)
from saryolo.nn.arch import VARIANTS, build_yaml_dict
from saryolo.nn.model import SARYOLODetectionModel


def _front_model():
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["cfar_n"]), ch=3, nc=1, verbose=False)
    model.eval()
    return model


def _save_checkpoint(model, path: Path) -> Path:
    """Persist a built model the way the trainer does, so ``load_model`` can read it back."""
    torch.save({"model": model.model, "epoch": -1}, path)
    return path


def _split(root: Path, n: int = 3) -> Path:
    """A small ``images/val`` split with non-flat chips and its ``data.yaml``."""
    import cv2

    images = root / "images" / "val"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(images / f"chip_{i}.png"), (rng.random((32, 32)) * 255).astype(np.uint8))
    cfg = root / "data.yaml"
    cfg.write_text(f"path: {root}\nnc: 1\nnames:\n- ship\ntrain: images/val\nval: images/val\n")
    return cfg


# ------------------------------------------------------------------- the split reader
def test_the_gray_loader_normalises_a_chip_and_refuses_an_undecodable_one(tmp_path):
    """A file that will not decode must raise, not leave a black tile in the batch."""
    import cv2

    cv2.imwrite(str(tmp_path / "ok.png"), np.full((20, 20), 200, np.uint8))
    out = _read_gray([tmp_path / "ok.png"], imgsz=16)
    assert out.shape == (1, 1, 16, 16) and out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert float(out.max()) > 0.0, "a real chip must not arrive as an all-zero tile"

    (tmp_path / "broken.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="could not decode"):
        _read_gray([tmp_path / "broken.png"], imgsz=16)


# --------------------------------------------------------------------- the front end
def test_find_front_end_accepts_the_facade_or_the_bare_module_and_refuses_a_stock_detector():
    """The locator must work from either object a caller holds, and say ``None`` for a stack."""
    model = _front_model()
    facade_front = find_front_end(model)
    bare_front = find_front_end(model.model)
    assert facade_front is not None
    assert bare_front is facade_front, (
        "the facade and the bare module must locate the same front end"
    )
    stock = SARYOLODetectionModel(build_yaml_dict(VARIANTS["baseline_n"]), ch=3, nc=1, verbose=False)
    assert find_front_end(stock) is None


def test_gain_maps_returns_one_row_per_image_and_the_map_is_bounded():
    """The collected gain must be one flattened map per image, and finite."""
    front = find_front_end(_front_model())
    images = np.random.default_rng(0).random((3, 1, 32, 32)).astype(np.float32)
    gains = gain_maps(front, images)
    assert gains.shape == (3, 32 * 32)
    assert np.isfinite(gains).all()


def test_the_default_shift_grid_has_an_identity_reference_and_named_shifts():
    """``identity`` is the unshifted reference; the other rows are the labelled synthetic shifts."""
    names = [name for name, _ in DEFAULT_SHIFTS]
    assert names[0] == "identity"
    assert {"brightness", "low_contrast", "speckle"} <= set(names), names


# -------------------------------------------------------------- the end-to-end diagnostic
def test_diagnose_gain_reports_the_clean_row_and_each_named_shift(tmp_path):
    """The full path: a checkpoint, a real split, and a report that names its provenance."""
    ckpt = _save_checkpoint(_front_model(), tmp_path / "cfar.pt")
    cfg = _split(tmp_path / "ds")

    result = diagnose_gain(
        ckpt, cfg, out_dir=tmp_path / "gain_out", imgsz=32,
        shifts=(("identity", 0.0), ("brightness", 0.5)),
    )
    assert result["weights"] == str(ckpt)
    assert result["mode"] == "cfar"
    assert result["synthetic"] is True, "a synthetic shift must be labelled as one"
    assert result["clean"]["images"] == 3
    # ``identity`` is the clean reference and must not be duplicated as a shift row.
    assert list(result["shifts"]) == ["brightness_0.5"]
    for row in (result["clean"], result["shifts"]["brightness_0.5"]):
        assert row["pixels_per_image"] == 32 * 32
        assert "collapsed" in row and "scene_scalar_dominated" in row

    written = json.loads((tmp_path / "gain_out" / "gain.json").read_text())
    assert written == result, "the report on disk differs from the one returned"


def test_diagnose_gain_refuses_a_detector_with_no_front_end(tmp_path):
    """A stock detector has no gain to diagnose; reporting zeros for it would be a lie."""
    stock = SARYOLODetectionModel(build_yaml_dict(VARIANTS["baseline_n"]), ch=3, nc=1, verbose=False)
    ckpt = _save_checkpoint(stock, tmp_path / "stock.pt")
    cfg = _split(tmp_path / "ds")
    with pytest.raises(ValueError, match="no RatioSpaceCFARFrontEnd"):
        diagnose_gain(ckpt, cfg, out_dir=tmp_path / "out", imgsz=32)


def test_diagnose_gain_refuses_an_empty_split(tmp_path):
    """An empty split resolved to zero images is not a measurement and must raise."""
    ckpt = _save_checkpoint(_front_model(), tmp_path / "cfar.pt")
    empty = tmp_path / "empty"
    (empty / "images" / "val").mkdir(parents=True)
    cfg = empty / "data.yaml"
    cfg.write_text(f"path: {empty}\nnc: 1\nnames:\n- ship\ntrain: images/val\nval: images/val\n")
    with pytest.raises(FileNotFoundError, match="no images"):
        diagnose_gain(ckpt, cfg, out_dir=tmp_path / "out", imgsz=32)
