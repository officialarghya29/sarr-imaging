"""The small module-level APIs: builders, inspection methods, and the parsing helpers.

Each of these is a public entry point that no test named before, so a wrong return shape
or a silently defaulted argument could sit here unnoticed. They fall into three groups:

* the attention registry (``build_attention``) and the per-variant ``attention_map`` -- the
  map is what every attention figure and every CAM target ultimately reads;
* the spectral gain (``spectral_gain``) and the conditioning vocabulary helpers
  (``parse_vocab_sizes``, ``embedding_size``), which decide a module's parameter count;
* SSAC's ``evidence_logit``, the pre-sigmoid quantity the whole allocation is built on.
"""

from __future__ import annotations

import pytest
import torch

from saryolo.nn.modules.attention import ATTENTION_BUILDERS, build_attention
from saryolo.nn.modules.conditioning import AcquisitionEncoder, parse_vocab_sizes
from saryolo.nn.modules.frequency import SpatialFrequencyRepresentation
from saryolo.nn.modules.ssac import ScatterSelectiveRefinement

# ------------------------------------------------------------------ attention registry

def test_build_attention_returns_each_registered_variant():
    c = 8
    for name in ATTENTION_BUILDERS:
        module = build_attention(name, c)
        assert isinstance(module, torch.nn.Module)


def test_build_attention_refuses_an_unknown_name():
    """A typo in an ablation YAML must fail, not silently fall back to a default block."""
    with pytest.raises(KeyError, match="Unknown attention"):
        build_attention("se_typo", 8)


@pytest.mark.parametrize("name", list(ATTENTION_BUILDERS))
def test_every_attention_map_is_broadcastable_over_the_feature(name):
    """All variants share one residual contract, so the map must broadcast to (B, C, H, W)."""
    module = build_attention(name, 8).eval()
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        m = module.attention_map(x)
    assert m.shape == x.shape, f"{name}: attention map {tuple(m.shape)} is not {tuple(x.shape)}"


def test_the_identity_attention_map_is_all_ones():
    """The control variant must be exactly identity, or the attention comparison is offset."""
    module = build_attention("none", 8).eval()
    x = torch.randn(1, 8, 8, 8)
    with torch.no_grad():
        m = module.attention_map(x)
    assert torch.equal(m, torch.ones_like(x))


def test_every_attention_variant_is_identity_at_initialisation():
    """``alpha`` starts at zero, so the residual is a no-op whatever the map says."""
    for name in ATTENTION_BUILDERS:
        module = build_attention(name, 8).eval()
        x = torch.randn(1, 8, 12, 12)
        with torch.no_grad():
            assert torch.equal(module(x), x), f"{name} is not identity at init"


# ------------------------------------------------------------------ spectral gain

def test_static_spectral_gain_has_the_rfft2_shape_and_is_positive():
    """A static gain is channel-wide: shape (1, C, H, W//2+1), and exp(g) > 0 always."""
    module = SpatialFrequencyRepresentation(6, mode="static", bands=4).eval()
    x = torch.randn(1, 6, 16, 24)
    with torch.no_grad():
        g = module.spectral_gain(x)
    assert g.shape == (1, 6, 16, 24 // 2 + 1)
    assert torch.all(g > 0), "exp(g) must be strictly positive"


def test_sff_spectral_gain_is_per_sample():
    """The proposed arm adapts per sample, so its gain carries the batch dimension."""
    module = SpatialFrequencyRepresentation(6, mode="sff", bands=4).eval()
    x = torch.randn(3, 6, 16, 16)
    with torch.no_grad():
        g = module.spectral_gain(x)
    assert g.shape == (3, 6, 16, 16 // 2 + 1)


def test_highpass_gain_increases_with_radius():
    """The classical control must actually be a high-pass: gain rises with frequency."""
    module = SpatialFrequencyRepresentation(4, mode="highpass", bands=8).eval()
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        g = module.spectral_gain(x)[0, 0]  # (H, W//2+1)
    # Compare the DC corner against the far (high-frequency) corner of the same row.
    assert float(g[0, 0]) < float(g[-1, -1]), "the high-pass gain does not rise with frequency"


# ------------------------------------------------------------------ conditioning helpers

def test_parse_vocab_sizes_accepts_mapping_list_and_string():
    assert parse_vocab_sizes({"sensor": 5, "mode": 3}) == {"sensor": 5, "mode": 3}
    # A positional list is what a model YAML uses; the field order is fixed.
    assert parse_vocab_sizes([5, 3, 2]) == {"sensor": 5, "polarization": 3, "mode": 2}
    assert parse_vocab_sizes("sensor=5,mode=3") == {"sensor": 5, "mode": 3}


def test_parse_vocab_sizes_returns_empty_for_none():
    assert parse_vocab_sizes(None) == {}


def test_parse_vocab_sizes_refuses_a_positional_list_of_the_wrong_length():
    """A short list would silently truncate the field mapping rather than fail."""
    with pytest.raises(ValueError, match="categorical fields"):
        parse_vocab_sizes([5, 3])


def test_parse_vocab_sizes_refuses_an_unknown_field_name():
    """A typo'd field would encode a real field as unknown without anything saying so."""
    with pytest.raises(ValueError, match="unknown categorical field"):
        parse_vocab_sizes("sensr=5")


def test_parse_vocab_sizes_refuses_a_non_integer_size():
    with pytest.raises(ValueError, match="not an integer"):
        parse_vocab_sizes("sensor=big")


def test_parse_vocab_sizes_refuses_a_missing_equals_sign():
    with pytest.raises(ValueError, match="field=size"):
        parse_vocab_sizes("sensor:")


def test_acquisition_encoder_embedding_size_counts_the_used_tables():
    encoder = AcquisitionEncoder(
        continuous_dim=3,
        vocab_sizes={"sensor": 5, "polarization": 3, "mode": 2},
        fields=("sensor", "polarization", "mode"),
    )
    assert encoder.embedding_size == 5 + 3 + 2


def test_acquisition_encoder_refuses_an_index_wider_than_its_table():
    """Mapping an unseen sensor onto a trained row is the confusion the design prevents."""
    encoder = AcquisitionEncoder(
        continuous_dim=3,
        vocab_sizes={"sensor": 2, "polarization": 2, "mode": 2},
        fields=("sensor",),
    )
    continuous = torch.zeros(1, 3)
    categorical = torch.tensor([[5, 0, 0]])  # index 5 into a 2-row table
    with pytest.raises(ValueError, match="embedding table has"):
        encoder(continuous, categorical, torch.ones(1, 6))


# ------------------------------------------------------------------ SSAC evidence

def test_ssac_evidence_logit_keeps_the_spatial_layout_in_adaptive_modes():
    module = ScatterSelectiveRefinement(8, mode="adaptive").eval()
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        logit = module.evidence_logit(x)
    assert logit.shape == (2, 1, 16, 16)


def test_ssac_evidence_logit_is_spatially_averaged_in_fixed_mode():
    """``fixed`` is the ablation: one difficulty scalar per image, same parameter count."""
    module = ScatterSelectiveRefinement(8, mode="fixed").eval()
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        logit = module.evidence_logit(x)
    assert logit.shape == (2, 1, 1, 1)


def test_ssac_evidence_logit_feeds_the_gain_map():
    """The gain map is the sigmoid of the evidence, so the two must agree."""
    module = ScatterSelectiveRefinement(8, mode="adaptive").eval()
    x = torch.randn(1, 8, 16, 16)
    with torch.no_grad():
        logit = module.evidence_logit(x)
        gain = module.gain_map(x)
    assert torch.allclose(gain, torch.sigmoid(logit / module.tau), atol=1e-6)
