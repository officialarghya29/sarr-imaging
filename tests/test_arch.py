"""Architecture tests: baseline parity with stock YOLO11, wiring, and identity-at-init.

These are the tests that keep the ablation table honest. If the baseline the
builder emits ever drifts from stock YOLO11, or if a SAR module stops being an
exact identity at initialisation, the reported gains would stop being
attributable to the modules and these tests fail first.
"""

from __future__ import annotations

import pytest
import torch

import saryolo  # noqa: F401  (registers custom layers with ultralytics)
from saryolo.nn.arch import BASELINE_PARAMS, VARIANTS, ModelSpec, build_yaml_dict, variant_filename
from saryolo.nn.model import SARYOLODetectionModel
from saryolo.nn.register import registered_modules

torch.manual_seed(0)


def _build(spec: ModelSpec, nc: int = 1, ch: int = 3) -> SARYOLODetectionModel:
    model = SARYOLODetectionModel(build_yaml_dict(spec), ch=ch, nc=nc, verbose=False)
    model.eval()
    return model


def _num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- registration
def test_modules_are_registered():
    """Importing saryolo must publish every custom layer for YAML resolution."""
    names = registered_modules()
    for expected in (
        "SARFeatureEnhancement",
        "SpeckleAwareFeatureModule",
        "SARAdaptiveAttention",
        "AdaptiveMultiScaleFusion",
        "SEAttention",
        "ECAAttention",
        "CBAMAttention",
    ):
        assert expected in names, f"{expected} was not registered"


def test_registration_is_idempotent():
    assert saryolo.register_modules() == saryolo.ensure_registered()


# ------------------------------------------------------------------------ baseline parity
@pytest.mark.parametrize("scale", ["n", "s"])
def test_baseline_matches_stock_yolo11_parameter_count(scale):
    """The no-module variant must be bit-for-bit the stock YOLO11 architecture.

    Published counts come from ultralytics' own ``cfg/models/11/yolo11.yaml``
    header comments, so an exact match proves the builder reproduces YOLO11
    rather than merely resembling it.
    """
    spec = VARIANTS[f"baseline_{scale}"]
    model = _build(spec, nc=80)
    assert _num_params(model) == BASELINE_PARAMS[scale], (
        f"baseline_{scale} has {_num_params(model):,} params, expected {BASELINE_PARAMS[scale]:,}"
    )


def test_variant_filenames_encode_scale():
    """Every emitted file name must let ultralytics recover its scale.

    ``yaml_model_load`` overwrites the YAML's ``scale`` key with
    ``guess_model_scale(path)``, which only recognises ``yolo<digits><scale>`` in
    the file name. A file it cannot parse silently builds at scale ``'n'`` — so a
    model reported as YOLO11s would actually be YOLO11n. This test pins the
    invariant against ultralytics' own function rather than a copy of its regex.
    """
    from ultralytics.nn.tasks import guess_model_scale

    for name, spec in VARIANTS.items():
        filename = variant_filename(spec)
        assert guess_model_scale(filename) == spec.scale, (
            f"{name}: file name {filename!r} does not encode scale {spec.scale!r}; "
            "ultralytics would silently build it at scale 'n'"
        )


def test_emitted_filenames_are_unique():
    """Distinct variants must not collide on the same output file name."""
    names: dict[str, str] = {}
    for name, spec in VARIANTS.items():
        filename = variant_filename(spec)
        assert filename not in names, f"{name} and {names[filename]} both map to {filename}"
        names[filename] = name


def test_baseline_has_no_sar_modules():
    spec = VARIANTS["baseline"]
    assert spec.module_names == []
    module_types = {m.type for m in _build(spec).model}
    assert not any("SAR" in t or "Speckle" in t for t in module_types)


# ------------------------------------------------------------------------- variant wiring
def test_every_variant_builds_and_forwards():
    """Every declared variant must construct and run a forward pass.

    Each variant is built at scale ``n`` because wiring correctness is
    scale-independent and ``n`` is the cheapest to instantiate. Variants are
    *not* filtered by their declared scale — doing so once silently skipped every
    ablation variant (they default to scale ``s``) and hid a real bug.
    """
    import dataclasses

    x = torch.randn(1, 3, 64, 64)
    tested = 0
    for name, spec in sorted(VARIANTS.items()):
        model = _build(dataclasses.replace(spec, scale="n"), nc=2)
        with torch.no_grad():
            out = model(x)
        assert out is not None, f"{name} produced no output"
        assert all(torch.isfinite(p).all() for p in model.parameters()), f"{name} has non-finite params"
        tested += 1
    assert tested == len(VARIANTS), f"only {tested} of {len(VARIANTS)} variants were exercised"


def test_declared_scale_variants_build():
    """Variants at their declared scale must build with the expected Detect levels."""
    names = ("baseline_n", "baseline_s", "baseline_m", "baseline_l",
             "full_s", "full_m", "full_l", "full_p35_l")
    for name in names:
        model = _build(VARIANTS[name], nc=1)
        has_p2_head = name.startswith("full_") and "p35" not in name
        assert len(model.model[-1].stride) == (4 if has_p2_head else 3), name
        assert _num_params(model) > 0


def test_ablation_variants_differ_from_full():
    """Ablation variants must actually change the graph, not silently be no-ops."""
    full = _build(VARIANTS["full"], nc=2)
    signatures = {}
    for name in ("att_none", "att_se", "att_cbam", "fus_static", "fus_concat", "spk_none", "pre_log"):
        model = _build(VARIANTS[name], nc=2)
        signatures[name] = _num_params(model)
    assert len(set(signatures.values())) > 1, f"ablation variants are indistinguishable: {signatures}"
    for name, n in signatures.items():
        assert n < _num_params(full), f"{name} should remove parameters relative to the full model"


# ---------------------------------------------------------------- identity-at-initialisation
def test_each_module_is_exactly_identity_at_init():
    """Every module must return its input bit-for-bit before training.

    This is the guarantee the whole ablation rests on: a module that perturbs the
    feature at initialisation would confound "the module helps" with "the extra
    layers changed the initial function". Exact equality (not approximate) is
    asserted deliberately.
    """
    import saryolo.nn.modules as M

    c = 32
    x = torch.randn(2, c, 16, 16)
    modules = [
        M.SARFeatureEnhancement(c),
        M.SpeckleAwareFeatureModule(c),
        M.SARAdaptiveAttention(c),
        M.SARAdaptiveAttention(c, gate="static"),
        M.AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2], mode="amf"),
        M.AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2], mode="static"),
        M.IdentityAttention(c),
        M.SEAttention(c),
        M.ECAAttention(c),
        M.CBAMAttention(c),
    ]
    for module in modules:
        module.eval()
        out = module(x)
        assert out.shape == x.shape
        assert torch.equal(out, x), f"{type(module).__name__} is not an exact identity at init"


def test_models_output_identically_to_baseline_at_init():
    """With the baseline's weights in place, SAR-YOLO must predict identically.

    Inserting modules shifts ``Sequential`` indices, so the baseline state dict
    cannot be loaded positionally. Instead the *stock* layers are matched in
    order (insertions are additive, so the stock rows keep their relative order)
    and copied one by one. The two models must then agree exactly.
    """
    from saryolo.nn.modules import CUSTOM_MODULES

    baseline = _build(VARIANTS["baseline_s"], nc=1, ch=3)
    saryolo_model = _build(VARIANTS["full_p35_s"], nc=1, ch=3)

    custom = tuple(CUSTOM_MODULES.values())
    ref_stock = [m for m in baseline.model if not isinstance(m, custom)]
    got_stock = [m for m in saryolo_model.model if not isinstance(m, custom)]
    assert len(ref_stock) == len(got_stock), (
        f"stock layer count differs: baseline {len(ref_stock)} vs SAR-YOLO {len(got_stock)}"
    )
    with torch.no_grad():
        for ref_layer, got_layer in zip(ref_stock, got_stock):
            assert type(ref_layer) is type(got_layer), (type(ref_layer), type(got_layer))
            got_layer.load_state_dict(ref_layer.state_dict())

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        ref = baseline(x)
        got = saryolo_model(x)

    ref_t = ref[0] if isinstance(ref, (tuple, list)) else ref
    got_t = got[0] if isinstance(got, (tuple, list)) else got
    assert ref_t.shape == got_t.shape, f"shape mismatch: {ref_t.shape} vs {got_t.shape}"
    max_diff = (ref_t - got_t).abs().max().item()
    assert max_diff == 0.0, f"modules are not identity at init (max abs diff {max_diff:.3e})"


def test_p2_head_adds_a_detection_level():
    """The P2 variant must expose four prediction levels rather than three."""
    p3 = _build(VARIANTS["baseline_s"], nc=1)
    p2 = _build(VARIANTS["p2"], nc=1)
    assert len(p3.model[-1].stride) == 3
    assert len(p2.model[-1].stride) == 4
    assert p2.model[-1].stride.tolist() == [4.0, 8.0, 16.0, 32.0]


def test_amf_group_split_matches_concatenation():
    """AMF reads its group sizes from the parser's channel list; verify they sum to its input."""
    from saryolo.nn.modules import AdaptiveMultiScaleFusion

    # `ch` as the parser would present it at the AMF row: prior layer widths, with the
    # concat width last (the AMF row is wired with `from: -1`, straight after the Concat).
    ch = [64, 128, 256, 512, 384]
    amf = AdaptiveMultiScaleFusion(ch, sources=[2, 1], mode="amf")
    assert amf.groups == [256, 128]
    assert amf.c1 == 384
    x = torch.randn(2, 384, 8, 8)
    assert amf(x).shape == x.shape

    # A stale `sources` list must fail loudly rather than silently mis-wire.
    with pytest.raises(ValueError, match="sum to"):
        AdaptiveMultiScaleFusion(ch, sources=[3, 2], mode="amf")


def test_channel_preservation_is_enforced_by_construction():
    """Every module must return the channel count the parser attributed to it."""
    from saryolo.nn.modules import (
        AdaptiveMultiScaleFusion,
        SARAdaptiveAttention,
        SARFeatureEnhancement,
        SpeckleAwareFeatureModule,
    )

    c = 48
    x = torch.randn(2, c, 16, 16)
    for module in (
        SARFeatureEnhancement(c),
        SpeckleAwareFeatureModule(c),
        SARAdaptiveAttention(c),
        AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2]),
    ):
        assert module(x).shape == x.shape, f"{type(module).__name__} changed the channel count"
