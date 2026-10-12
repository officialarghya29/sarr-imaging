"""Tests for the shared primitives every SAR-YOLO module is built from.

``saryolo/nn/modules/_common.py`` carries the contract that makes a custom module droppable
into an Ultralytics YAML: channel preservation and identity at initialisation. The primitives
themselves are exercised through the modules that use them, but their *guards* were never hit,
and the guard that matters is :func:`resolve_c1`. ``parse_model`` passes the running channel
list (``"ch"``) with an explicit ``source`` index whenever a YAML row's ``from`` is not ``-1``,
and the module's own docstring names the failure: reading the wrong entry silently builds the
module at the wrong width and every downstream number describes a model nobody chose. So the
resolution rules and their refusals are pinned directly here, against the primitives rather
than against one consumer.
"""

from __future__ import annotations

import math

import pytest
import torch

from saryolo.nn.modules._common import (
    ChannelDescriptor,
    LocalStats,
    ZeroGate,
    interp_weights,
    radial_band_index,
    resolve_c1,
)


# --------------------------------------------------------------------- channel resolution
def test_resolve_c1_reads_the_parser_channel_list_at_the_named_source():
    """``ch[-1]`` and ``ch[source]`` must name different layers, as the docstring warns."""
    ch = [16, 32, 64, 128]
    assert resolve_c1(ch) == 128, "without a source the last entry is the most recent layer"
    assert resolve_c1(ch, source=0) == 16
    assert resolve_c1(ch, source=-2) == 64
    assert resolve_c1(ch, source=-1) == 128
    # A plain int passes straight through; the parser substitutes an int once the graph is built.
    assert resolve_c1(48) == 48


def test_resolve_c1_refuses_every_malformed_channel_argument():
    """Each refusal is a boundary where a wrong guess would build the module at a wrong width."""
    with pytest.raises(ValueError, match="empty channel list"):
        resolve_c1([])
    with pytest.raises(IndexError, match="outside the channel list"):
        resolve_c1([16, 32], source=5)
    with pytest.raises(IndexError, match="outside the channel list"):
        resolve_c1([16, 32], source=-3)
    # ``bool`` is an ``int`` subclass, so it must be rejected explicitly rather than read as 1.
    with pytest.raises(TypeError, match="bool"):
        resolve_c1(True)
    with pytest.raises(TypeError, match="int or list"):
        resolve_c1(16.0)
    with pytest.raises(TypeError, match="int or list"):
        resolve_c1("ch")


# ------------------------------------------------------------------------- zero gate
def test_zero_gate_is_an_exact_identity_at_init_and_names_its_value_beside_it():
    """The zero init is what keeps a fresh SAR-YOLO identical to its baseline."""
    gate = ZeroGate()
    assert float(gate()) == 0.0
    assert "+0.0000" in gate.extra_repr()
    with torch.no_grad():
        gate.raw.fill_(100.0)  # tanh saturates well inside (+/-1)
    assert float(gate()) == pytest.approx(1.0, abs=1e-6)
    assert "+1.0000" in gate.extra_repr(), gate.extra_repr()


# ------------------------------------------------------------------- local statistics
def test_local_stats_returns_consistent_first_and_second_order_terms():
    """``sd`` and ``highpass`` must be the exact functions of ``mu`` the module documents."""
    torch.manual_seed(0)
    stats = LocalStats(k=3, eps=1e-5)
    x = torch.randn(2, 4, 16, 16)
    mu, var, sd, highpass = stats(x)
    assert mu.shape == x.shape and highpass.shape == x.shape
    assert torch.allclose(highpass, x - mu)
    assert torch.allclose(sd, torch.sqrt(var + 1e-5), atol=1e-6)
    # Variance is floored at zero, so a constant field has no negative variance and no NaN.
    flat = torch.full((1, 1, 8, 8), 0.4)
    _, var_flat, sd_flat, _ = stats(flat)
    assert float(var_flat.min()) >= 0.0 and torch.isfinite(sd_flat).all()


def test_channel_descriptor_stacks_mean_max_and_std_per_channel():
    """The descriptor's three statistics must be per-channel and correctly concatenated."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8, 8)
    out = ChannelDescriptor()(x)
    assert out.shape == (2, 15)
    flat = x.flatten(2)
    assert torch.allclose(out[:, :5], flat.mean(-1), atol=1e-6)
    assert torch.allclose(out[:, 5:10], flat.amax(-1), atol=1e-6)
    assert torch.allclose(out[:, 10:], flat.std(-1, unbiased=False), atol=1e-6)


# ------------------------------------------------------------------- band interpolation
def test_interp_weights_returns_a_continuous_two_band_split_within_range():
    """Coefficients must interpolate between two bands, never fall outside them."""
    radius = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    lo, hi, weight = interp_weights(radius, bands=5)
    assert lo.shape == hi.shape == weight.shape == radius.shape
    assert (weight >= 0.0).all() and (weight <= 1.0).all()
    assert (lo >= 0).all() and (hi <= 4).all()
    # A midpoint sits halfway between its two bands rather than snapping to one.
    mid = interp_weights(torch.tensor([0.25]), bands=3)  # pos = 0.5 -> bands 0 and 1, w = 0.5
    assert mid[0].item() == 0 and mid[1].item() == 1
    assert abs(mid[2].item() - 0.5) < 1e-6
    # The top radius clamps to the last band, with no out-of-range index.
    top_lo, top_hi, _ = interp_weights(torch.tensor([1.0]), bands=5)
    assert top_lo.item() == 4 and top_hi.item() == 4


def test_radial_band_index_covers_every_rfft2_bin():
    """The band weights must be indexed per ``rfft2`` bin, or the spectral filter mis-sizes."""
    h, w, bands = 8, 16, 6
    lo, hi, weight = radial_band_index(h, w, bands, device=torch.device("cpu"), dtype=torch.float32)
    assert lo.numel() == h * (w // 2 + 1)
    assert hi.numel() == weight.numel() == lo.numel()
    assert (lo >= 0).all() and (hi <= bands - 1).all()
    # The corner bin (highest radial frequency) must reach the top band, and the DC bin the floor.
    matrix = weight.reshape(h, w // 2 + 1)
    assert matrix[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert int(lo.reshape(h, w // 2 + 1)[h // 2, w // 2]) == bands - 1
    assert math.isclose(float(matrix.max()), float(matrix.max()))  # finite, no NaN
