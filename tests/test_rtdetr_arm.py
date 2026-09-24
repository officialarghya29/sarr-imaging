"""Tests for the RT-DETR conditioning feasibility arm.

Scope (deliberately narrow, honestly so): these tests prove the conditioned
adapter *inserts into an RT-DETR graph and consumes the metadata context* — the
prerequisite for the architecture-generality claim (red-team W6). They do **not**
measure accuracy, and nothing here may be quoted as a cross-architecture result.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANT = REPO_ROOT / "configs" / "models" / "rtdetr" / "rtdetr_s_cond_film.yaml"


def _vocabs():
    from saryolo.data.metadata import Vocabulary

    return {
        "sensor": Vocabulary.build("sensor", ["g3", "s1"]),
        "polarization": Vocabulary.build("polarization", []),
        "mode": Vocabulary.build("mode", []),
    }


def _context(meta, vocabs):
    from saryolo.data.metadata import encode_metadata

    continuous, categorical, availability = encode_metadata(meta, vocabs)
    return (
        torch.as_tensor([continuous], dtype=torch.float32),
        torch.as_tensor([categorical], dtype=torch.int64),
        torch.as_tensor([availability], dtype=torch.float32),
    )


def _prepared_model(vocabs):
    """The model exactly as training builds it: vocab sizes rewritten from the table."""
    from saryolo.data.conditioning import prepare_conditioned_config
    from saryolo.nn.model import SARYOLORTDetectionModel

    prepared = prepare_conditioned_config(VARIANT, vocabs)
    torch.manual_seed(0)
    return SARYOLORTDetectionModel(prepared, ch=3, nc=1, verbose=False)


# ---------------------------------------------------------------- generated file


def test_variant_yaml_matches_its_generator():
    """The committed RT-DETR YAML is generator output, not a hand edit."""
    import subprocess
    import sys

    from saryolo.nn import register  # noqa: F401  (ensures the repo is importable)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_rtdetr_variant.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_variant_has_exactly_one_adapter_and_intact_decoder():
    """One adapter row, inserted before the decoder, and the decoder still exists."""
    data = yaml.safe_load(VARIANT.read_text())
    rows = data["backbone"] + data["head"]
    adapters = [r for r in rows if r[2] == "AcquisitionConditionedAdapter"]
    assert len(adapters) == 1
    assert data["head"][-1][2] == "RTDETRDecoder"
    # The decoder's third source must be the layer directly after the P5 C2f —
    # i.e. the adapter — otherwise the decoder never sees conditioned features.
    assert isinstance(data["head"][-1][0], list) and len(data["head"][-1][0]) == 3


# ------------------------------------------------------------------ facade path


def test_rtdetr_facade_carries_the_metadata_context():
    """The facade attaches the same context machinery as the YOLO facade."""
    model = _prepared_model(_vocabs())
    assert model.n_conditioned_adapters == 1
    assert model.metadata_context is not None


def test_conditioning_reaches_the_decoder_output():
    """With the gate open (as training leaves it), a different sensor must change
    the model output, and a held-out sensor must land on the unknown row — the
    exact property the LOSO protocol needs from this architecture."""
    from saryolo.data.metadata import AcquisitionMetadata

    vocabs = _vocabs()
    model = _prepared_model(vocabs)
    model.eval()
    img = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__ == "AcquisitionConditionedAdapter":
                module.alpha.raw.fill_(0.5)
        model.metadata_context.set(*_context(AcquisitionMetadata(sensor="s1", resolution_m=5.0), vocabs))
        out_s1 = model(img)
        model.metadata_context.set(*_context(AcquisitionMetadata(sensor="g3", resolution_m=5.0), vocabs))
        out_g3 = model(img)
        model.metadata_context.set(*_context(AcquisitionMetadata(sensor="held_out", resolution_m=5.0), vocabs))
        out_held = model(img)

    assert not torch.allclose(out_s1[0], out_g3[0]), (
        "two known sensors produced identical outputs: the adapter is not consuming the context"
    )
    assert not torch.allclose(out_s1[0], out_held[0]), (
        "the held-out sensor (unknown row) produced the same output as a known sensor"
    )


def test_vocabulary_mismatch_is_refused_loudly():
    """Encoding with a vocabulary whose *indices* exceed the table must raise, not
    clamp — the same loud failure the YOLO facade gives (a silent clamp would map
    an unseen sensor onto a trained one). The mismatched vocabulary's entries sort
    differently, so sensor "d" encodes to index 4 against a 3-row table."""
    from saryolo.data.metadata import AcquisitionMetadata, Vocabulary

    vocabs = _vocabs()
    model = _prepared_model(vocabs)  # built with the 2-source vocabulary
    model.eval()
    img = torch.rand(1, 3, 64, 64)

    mismatched = dict(vocabs)
    mismatched["sensor"] = Vocabulary.build("sensor", ["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="does not match"), torch.no_grad():
        model.metadata_context.set(*_context(AcquisitionMetadata(sensor="d"), mismatched))
        model(img)


def test_gate_closed_keeps_the_baseline_bit_exact():
    """At init (gate = 0) the conditioned RT-DETR graph must equal the stock graph
    numerically — the identity contract every other module in this repo obeys."""
    from ultralytics.nn.tasks import RTDETRDetectionModel

    from saryolo.nn.model import SARYOLORTDetectionModel

    torch.manual_seed(7)
    conditioned = SARYOLORTDetectionModel(VARIANT, ch=3, nc=1, verbose=False)
    conditioned.eval()
    torch.manual_seed(7)
    stock = RTDETRDetectionModel(VARIANT, ch=3, nc=1, verbose=False)
    stock.eval()

    img = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out_conditioned = conditioned(img)
        out_stock = stock(img)
    assert torch.allclose(out_conditioned[0], out_stock[0], atol=1e-6), (
        "conditioned graph differs from stock at init: the gate is not the identity mechanism"
    )
