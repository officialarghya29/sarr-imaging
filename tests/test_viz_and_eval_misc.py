"""Assorted public helpers that no test named: summaries, writers, and the criterion factory.

Grouped because each is a small entry point rather than a subsystem, and each had a silent
failure mode worth one test:

* ``frontier_summary`` is the README's prose about the published field, generated from the
  registry so the two cannot drift -- if it drops an entry, the text quietly understates
  the field;
* ``apply_corruption`` is shared by the training-time consistency term and the evaluation
  sweep, so an unknown name must fail loudly rather than return its input;
* ``overlay_heatmap`` and ``save_feature_grid`` are the figure writers;
* ``build_criterion`` decides whether a run is a plain detector or carries the SAR terms.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from saryolo.evaluation.reported_baselines import REPORTED_BASELINES, frontier_summary
from saryolo.evaluation.robustness import CORRUPTIONS, Corruption, apply_corruption

# ------------------------------------------------------------------ frontier summary

def test_frontier_summary_names_every_reported_entry_with_its_venue():
    """The summary is generated from the registry; dropping an entry is a silent omission."""
    text = frontier_summary()
    for entry in REPORTED_BASELINES.values():
        assert entry.name in text, f"{entry.name} is missing from the frontier summary"
        assert entry.venue in text
        assert str(entry.year) in text


def test_frontier_summary_warns_that_it_is_not_like_for_like():
    """The prose must not present published numbers as a reproduced comparison."""
    assert "positioning aid" in frontier_summary()


# ------------------------------------------------------------------ corruption registry

def test_corruption_is_a_named_parameterised_degradation():
    spec = Corruption("x", "p", (1.0, 2.0), "desc", physical=True)
    assert spec.param_name == "p" and spec.severities == (1.0, 2.0) and spec.physical


def test_every_registered_corruption_has_an_ordered_severity_grid():
    """The x-axis of the robustness figure is the severity grid, so it must be monotone."""
    for name, spec in CORRUPTIONS.items():
        assert spec.name == name
        assert spec.description and spec.param_name, name
        assert len(spec.severities) >= 2, f"{name}: a single point is not a curve"


def test_apply_corruption_refuses_an_unknown_name():
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError, match="Unknown corruption"):
        apply_corruption(np.zeros((16, 16), dtype=np.uint8), "nope", 1.0, rng)


def test_apply_corruption_identity_returns_uint8_unchanged():
    img = np.full((16, 16), 128, dtype=np.uint8)
    out = apply_corruption(img, "identity", 1.0, np.random.default_rng(0))
    assert out.dtype == np.uint8
    assert np.allclose(out, img, atol=1)


def test_apply_corruption_is_deterministic_given_a_seed():
    """Baseline and proposed must face byte-identical inputs, so the draw is reproducible."""
    img = np.full((16, 16), 100, dtype=np.uint8)
    a = apply_corruption(img, "speckle", 4.0, np.random.default_rng(7))
    b = apply_corruption(img, "speckle", 4.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ writers

def test_save_feature_grid_writes_a_figure(tmp_path):
    from saryolo.visualization.feature_maps import save_feature_grid

    maps = np.random.default_rng(0).random((6, 16, 16)).astype(np.float32)
    out = save_feature_grid(maps, tmp_path / "nested" / "grid.png", title="features")
    assert out.exists() and out.stat().st_size > 0


def test_overlay_heatmap_returns_a_three_channel_image_of_the_input_size():
    from saryolo.visualization.attention_maps import overlay_heatmap

    image = np.zeros((24, 32, 3), dtype=np.uint8)
    heat = np.linspace(0, 1, 24 * 32).reshape(24, 32)
    out = overlay_heatmap(image, heat)
    assert out.shape == image.shape


def test_overlay_heatmap_accepts_a_grayscale_image_and_resizes_a_mismatched_heatmap():
    from saryolo.visualization.attention_maps import overlay_heatmap

    image = np.zeros((20, 20), dtype=np.uint8)
    heat = np.zeros((5, 5), dtype=np.float32)
    out = overlay_heatmap(image, heat)
    assert out.shape[:2] == (20, 20)


def test_save_attention_panel_returns_none_for_an_undecodable_image(tmp_path):
    """A file that is not an image must return None rather than draw a blank panel."""
    from saryolo.visualization.attention_maps import save_attention_panel

    bad = tmp_path / "not_an_image.png"
    bad.write_bytes(b"this is not a png")
    assert save_attention_panel(None, bad, tmp_path / "out.png") is None
    assert not (tmp_path / "out.png").exists()


# ------------------------------------------------------------------ criterion factory

def _baseline_model():
    """A model with the training arguments Ultralytics' criterion expects.

    ``build_criterion`` constructs ``SARAwareDetectionLoss`` directly, and ``v8DetectionLoss``
    reads ``model.args`` for its box/cls/dfl gains. A model built here has no training
    arguments, so they are set explicitly -- the same defaults the repository's own
    ``init_criterion`` falls back to.
    """
    from types import SimpleNamespace

    from saryolo.nn.arch import VARIANTS, build_yaml_dict
    from saryolo.nn.model import SARYOLODetectionModel

    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["baseline"]), ch=3, nc=1, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    return model


def test_build_criterion_returns_a_sar_aware_loss_for_a_classic_head():
    from saryolo.nn.losses import SARAwareDetectionLoss, build_criterion

    criterion = build_criterion(_baseline_model(), {"w_sep": 0.5})
    assert isinstance(criterion, SARAwareDetectionLoss)
    assert criterion.w_sep == 0.5


def test_build_criterion_with_no_weights_is_still_the_sar_criterion_at_zero():
    """Off-by-default must mean *numerically* stock, not a different class."""
    from saryolo.nn.losses import SARAwareDetectionLoss, build_criterion

    criterion = build_criterion(_baseline_model())
    assert isinstance(criterion, SARAwareDetectionLoss)
    assert (criterion.w_sep, criterion.w_small, criterion.w_smooth) == (0.0, 0.0, 0.0)


def test_build_criterion_refuses_an_unknown_sar_option():
    from saryolo.nn.losses import build_criterion

    with pytest.raises(ValueError, match="Unknown SAR loss option"):
        build_criterion(_baseline_model(), {"w_typo": 1.0})
