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
        "TargetPriorModulation",
        "SpatialFrequencyRepresentation",
        "ContextAggregation",
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
             "full_s", "full_m", "full_l", "full_p35_l",
             "v2_full_s", "v2_full_m", "v2_full_l", "v2_full_p35_s")
    for name in names:
        model = _build(VARIANTS[name], nc=1)
        # Stock baselines and the explicit P3-P5 variants have three levels; every other
        # declared-scale variant carries the P2 small-object level.
        has_p2_head = not name.startswith("baseline_") and "p35" not in name
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
#: Modes that must be exact identities at initialisation, per module class.
#:
#: The classical *operator-replacement* arms (Lee filter, fixed low-pass, log compression,
#: local standardisation) are deliberately excluded: they replace the operator instead of
#: gating a residual, so they are not identity at init and are not meant to be. Only the
#: proposed mechanisms and their controlled ablations are required to start exactly at the
#: baseline function -- that is what makes a measured gain attributable to the mechanism.
IDENTITY_MODES: dict[str, tuple[str, ...]] = {
    "SARFeatureEnhancement": ("sfe", "identity"),
    "SpeckleAwareFeatureModule": ("sfm", "sfm_clutter"),
    "TargetPriorModulation": ("learned", "channel", "static", "cfar", "none"),
    "SpatialFrequencyRepresentation": ("sff", "static", "highpass", "none"),
    "ContextAggregation": ("multi", "local", "regional", "none"),
}

#: Control modes that have no residual to learn from: they either return the input
#: unchanged (``"none"``) or are a pure pass-through with no gate at all (``"identity"``).
#: A stationary gate is the correct behaviour for these, so they are excluded from the
#: gradient-flow test -- running a backward pass through them is meaningless.
UNGATED_MODES = frozenset({"none", "identity"})

def _declared_vocabularies() -> dict[str, tuple[str, ...]]:
    """Every string this package can hand to ``parse_model`` as a mode argument.

    The attention slot is included via its builder registry because
    ``SARAdaptiveAttention`` selects its behaviour with a ``gate`` argument rather than a
    ``MODES`` tuple, so its vocabulary lives in ``ATTENTION_BUILDERS``.
    """
    import saryolo.nn.modules as M

    vocab = {
        name: tuple(getattr(M, name).MODES)
        for name in ("SARFeatureEnhancement", "SpeckleAwareFeatureModule", "AdaptiveMultiScaleFusion",
                     "TargetPriorModulation", "SpatialFrequencyRepresentation", "ContextAggregation")
    }
    vocab["attention slot"] = tuple(M.ATTENTION_BUILDERS)
    vocab["attention gate"] = ("adaptive", "static")
    return vocab


#: The subset whose residual gate must be able to move.
LEARNABLE_MODES: dict[str, tuple[str, ...]] = {
    cls: tuple(m for m in modes if m not in UNGATED_MODES) for cls, modes in IDENTITY_MODES.items()
}


def _cases_by_mode(c: int = 32) -> list[tuple[str, torch.nn.Module]]:
    """One instance per declared mode of every component, so new modes are covered free."""
    import saryolo.nn.modules as M

    return [
        (f"{cls_name}({mode})", getattr(M, cls_name)(c, mode=mode))
        for cls_name, modes in IDENTITY_MODES.items()
        for mode in modes
    ]


def test_each_module_is_exactly_identity_at_init():
    """Every module must return its input bit-for-bit before training.

    This is the guarantee the whole ablation rests on: a module that perturbs the
    feature at initialisation would confound "the module helps" with "the extra
    layers changed the initial function". Exact equality (not approximate) is
    asserted deliberately.

    Every declared mode of every component is exercised, so adding a mode cannot
    quietly escape the invariant.
    """
    import saryolo.nn.modules as M

    c = 32
    x = torch.randn(2, c, 16, 16)
    modules = [
        M.SARAdaptiveAttention(c),
        M.SARAdaptiveAttention(c, gate="static"),
        M.AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2], mode="amf"),
        M.AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2], mode="static"),
        M.IdentityAttention(c),
        M.SEAttention(c),
        M.ECAAttention(c),
        M.CBAMAttention(c),
        *[m for _, m in _cases_by_mode(c)],
    ]
    for module in modules:
        module.eval()
        out = module(x)
        assert out.shape == x.shape
        assert torch.equal(out, x), f"{type(module).__name__} is not an exact identity at init"


@pytest.mark.parametrize("variant", ["full_p35_s", "v2_full_p35_s"])
def test_models_output_identically_to_baseline_at_init(variant):
    """With the baseline's weights in place, SAR-YOLO must predict identically.

    Inserting modules shifts ``Sequential`` indices, so the baseline state dict
    cannot be loaded positionally. Instead the *stock* layers are matched in
    order (insertions are additive, so the stock rows keep their relative order)
    and copied one by one. The two models must then agree exactly.
    """
    from saryolo.nn.modules import CUSTOM_MODULES

    baseline = _build(VARIANTS["baseline_s"], nc=1, ch=3)
    saryolo_model = _build(VARIANTS[variant], nc=1, ch=3)

    custom = tuple(CUSTOM_MODULES.values())
    ref_stock = [m for m in baseline.model if not isinstance(m, custom)]
    got_stock = [m for m in saryolo_model.model if not isinstance(m, custom)]
    assert len(ref_stock) == len(got_stock), (
        f"stock layer count differs: baseline {len(ref_stock)} vs SAR-YOLO {len(got_stock)}"
    )
    with torch.no_grad():
        for ref_layer, got_layer in zip(ref_stock, got_stock, strict=True):
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
    modules = [
        SARFeatureEnhancement(c),
        SpeckleAwareFeatureModule(c),
        SARAdaptiveAttention(c),
        AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2]),
        *[m for _, m in _cases_by_mode(c)],
    ]
    for module in modules:
        assert module(x).shape == x.shape, f"{type(module).__name__} changed the channel count"


# ------------------------------------------------------------- gradient flow at init
def test_no_module_is_frozen_at_init():
    """Every component's gate must receive gradient and actually move when trained.

    Identity at initialisation is provided by the zero-initialised gate, and that is
    sufficient *on its own*. This test exists because the residual branch used to be
    zero-initialised too, which looks harmless and is not: with ``branch == 0`` the gate
    gradient ``dL/dalpha = <dL/dout, branch>`` is identically zero, so ``alpha`` can never
    leave 0, and ``dL/d(branch) = alpha * ...`` is zero for the same reason. Both vanish
    together and the component stays a permanent no-op while the identity test passes.

    That bug was present in Component 1 (``SARFeatureEnhancement``): its learned
    enhancement branch never trained, so "+SFE" measured only its affine scalars. The
    assertion is deliberately made on the gradient as well as on the value, because the
    gradient is the property that must never be zero.
    """
    import saryolo.nn.modules as M

    c = 32
    x = torch.randn(2, c, 16, 16)
    cases = [
        ("SARAdaptiveAttention", M.SARAdaptiveAttention(c)),
        ("SARAdaptiveAttention(static)", M.SARAdaptiveAttention(c, gate="static")),
        ("SEAttention", M.SEAttention(c)),
        ("ECAAttention", M.ECAAttention(c)),
        ("CBAMAttention", M.CBAMAttention(c)),
        ("AMF(amf)", M.AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2], mode="amf")),
        ("AMF(static)", M.AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2], mode="static")),
        *[
            (f"{cls}({mode})", getattr(M, cls)(c, mode=mode))
            for cls, modes in LEARNABLE_MODES.items()
            for mode in modes
        ],
    ]
    for name, module in cases:
        module.train()
        module.zero_grad(set_to_none=True)
        module(x).pow(2).mean().backward()
        grad = module.alpha.raw.grad
        assert grad is not None and grad.abs().item() > 0.0, (
            f"{name}: the gate receives zero gradient at init, so it can never switch on. "
            "The residual branch must not be zero-initialised -- identity comes from the gate."
        )

        # And confirm the gate actually moves under an optimiser, not just that its
        # gradient tensor is non-zero.
        opt = torch.optim.SGD(module.parameters(), lr=0.5)
        for _ in range(3):
            opt.zero_grad()
            module(x).pow(2).mean().backward()
            opt.step()
        moved = abs(float(module.alpha().detach()))
        assert moved > 1e-6, f"{name}: gate stayed at {moved:.2e} after 3 optimiser steps"


# ------------------------------------------------------------------ parse_model contract
def test_module_mode_names_are_safe_for_parse_model():
    """Mode names must survive ``parse_model``'s argument resolution.

    ``parse_model`` resolves each string argument through
    ``ast.literal_eval`` inside ``contextlib.suppress(ValueError)``. A plain identifier
    raises ``ValueError``, which is suppressed, so the string survives; but a Python
    *keyword* raises ``SyntaxError``, which is **not** suppressed, and model construction
    dies with a bare "SyntaxError: invalid syntax" from inside ultralytics. A context mode
    originally named ``global`` did exactly that.

    The same code path can also *silently* substitute a mode whose name collides with a
    local variable of ``parse_model`` (``ch``, ``f``, ``n``, ``m``, ``args``, ...), which is
    worse than an error, so both hazards are guarded here.
    """
    import keyword

    #: Non-exhaustive: locals of ``ultralytics.nn.tasks.parse_model`` that a mode must never equal.
    parse_model_locals = {"d", "ch", "f", "n", "m", "args", "i", "j", "a", "c1", "c2",
                          "width", "depth", "max_channels", "scales", "restricted", "legacy"}
    for slot, modes in _declared_vocabularies().items():
        for mode in modes:
            assert mode.isidentifier(), f"{slot}: mode {mode!r} is not an identifier"
            assert not keyword.iskeyword(mode), (
                f"{slot}: mode {mode!r} is a Python keyword; parse_model's ast.literal_eval "
                "raises an unsuppressed SyntaxError for it"
            )
            assert mode not in parse_model_locals, (
                f"{slot}: mode {mode!r} collides with a parse_model local variable and would "
                "be silently substituted by it"
            )


def test_every_new_slot_mode_is_exercised_by_a_variant():
    """Each declared mode of the new slots must be covered by at least one variant.

    Wire-level errors in these modules are invisible to the unit-level identity test: they
    only appear when ``parse_model`` substitutes channel counts and layer indices for the
    real graph. Coverage by a variant is therefore what actually tests the wiring.
    """
    from saryolo.nn.modules import ContextAggregation, SpatialFrequencyRepresentation, TargetPriorModulation

    covered = {
        "prior": {s.prior for s in VARIANTS.values() if s.prior},
        "frequency": {s.frequency for s in VARIANTS.values() if s.frequency},
        "context": {s.context for s in VARIANTS.values() if s.context},
    }
    expected = {
        "prior": set(TargetPriorModulation.MODES),
        "frequency": set(SpatialFrequencyRepresentation.MODES),
        "context": set(ContextAggregation.MODES),
    }
    for slot, modes in expected.items():
        missing = modes - covered[slot]
        assert not missing, f"{slot}: modes no variant exercises: {sorted(missing)}"


# ------------------------------------------------------------------------ v2 components
def test_v2_ladder_adds_one_component_per_row():
    """Each v2 ladder row must add exactly one thing over the previous row.

    The clutter row is the exception by design: clutter-awareness is a *mode change* on the
    existing speckle slot rather than an additional module, so it adds no slot and the test
    pins that explicitly instead of letting it pass vacuously.
    """
    rows: dict[str, str | None] = {
        "v2_clutter": None,  # mode change, not a new slot
        "v2_prior": "prior",
        "v2_freq": "frequency",
        "v2_full": "context",
    }
    assert VARIANTS["full"].speckle == "sfm"
    assert VARIANTS["v2_clutter"].speckle == "sfm_clutter"

    prev = set(VARIANTS["full"].module_names)
    for name, expected_new in rows.items():
        added = set(VARIANTS[name].module_names) - prev
        if expected_new is None:
            assert added == set(), f"{name} should only change a mode, but added {sorted(added)}"
        else:
            assert added == {expected_new}, f"{name} added {sorted(added)}, expected '{expected_new}'"
        prev = set(VARIANTS[name].module_names)


def test_new_slots_appear_in_the_built_graph():
    """The new slots must materialise as layers, not be silently dropped."""
    from saryolo.nn.modules import ContextAggregation, SpatialFrequencyRepresentation, TargetPriorModulation

    types = {type(m) for m in _build(VARIANTS["v2_full"], nc=2).model}
    for cls in (TargetPriorModulation, SpatialFrequencyRepresentation, ContextAggregation):
        assert cls in types, f"{cls.__name__} is missing from the built v2_full model"


def test_v2_removal_ablation_actually_removes_something():
    """Every removal row must cost fewer parameters than the full model."""
    full = _num_params(_build(VARIANTS["v2_full"], nc=2))
    for name in ("v2_noprior", "v2_nofreq", "v2_noctx", "v2_noclutter"):
        n = _num_params(_build(VARIANTS[name], nc=2))
        assert n < full, f"{name} ({n}) removes nothing relative to v2_full ({full})"


def test_target_prior_arms_are_capacity_matched_where_claimed():
    """``tp_channel`` must be the same size as the proposed arm, or the comparison is void.

    The prior ablation claims that ``learned`` vs ``channel`` isolates *spatial selectivity*
    rather than capacity. That claim is only true if the two arms have identical parameter
    counts, which is exactly what this asserts; ``learned`` vs ``static`` is the coarser,
    deliberately non-matched comparison.
    """
    counts = {
        name: _num_params(_build(VARIANTS[name], nc=2))
        for name in ("tp_none", "tp_static", "tp_channel", "v2_full")
    }
    assert counts["tp_channel"] == counts["v2_full"], (
        f"channel and learned priors must be capacity matched: {counts}"
    )
    assert counts["tp_none"] < counts["tp_static"] < counts["tp_channel"], counts


def test_frequency_slot_arms_differ_as_documented():
    """Fixed spectral filters cost no parameters; learnable bands and adaptivity do."""
    counts = {
        name: _num_params(_build(VARIANTS[name], nc=2))
        for name in ("fr_none", "fr_highpass", "fr_static", "v2_full")
    }
    # The high-pass profile is a registered buffer, not a parameter: the arm differs from
    # the disabled one in behaviour, not in size.
    assert counts["fr_highpass"] == counts["fr_none"], counts
    # Learnable bands add parameters; the input-adaptive head adds more on top.
    assert counts["fr_static"] > counts["fr_none"], counts
    assert counts["v2_full"] > counts["fr_static"], counts


def test_context_slot_arms_differ_as_documented():
    """Both context extents must add parameters, and the combination both of them."""
    counts = {
        name: _num_params(_build(VARIANTS[name], nc=2))
        for name in ("cx_none", "cx_local", "cx_regional", "v2_full")
    }
    assert counts["cx_none"] < counts["cx_local"], counts
    assert counts["cx_none"] < counts["cx_regional"], counts
    assert counts["v2_full"] > counts["cx_local"], counts
    assert counts["v2_full"] > counts["cx_regional"], counts


# ------------------------------------------------------------- new-module unit contracts
def test_target_prior_map_contract_holds_for_every_mode():
    """``prior_map`` must be (B, 1, H, W) in (-1, 1) for every arm.

    The figure pipeline renders the prior for any arm, so the shape and range cannot be
    mode-dependent. The uniform arms return a spatially constant map -- that is what those
    arms assert, not a defect.
    """
    from saryolo.nn.modules import TargetPriorModulation

    x = torch.randn(2, 16, 12, 12)
    for mode in TargetPriorModulation.MODES:
        module = TargetPriorModulation(16, mode=mode).eval()
        with torch.no_grad():
            pm = module.prior_map(x)
        assert pm.shape == (2, 1, 12, 12), (mode, tuple(pm.shape))
        assert -1.0 <= float(pm.min()) <= float(pm.max()) <= 1.0, (mode, float(pm.min()), float(pm.max()))

    # The uniform arms are uniform by construction; the proposed arm is not.
    uniform = TargetPriorModulation(16, mode="static").eval()
    with torch.no_grad():
        pm = uniform.prior_map(x)
    assert torch.allclose(pm, pm[..., :1, :1].expand_as(pm))


def test_frequency_module_is_resolution_independent():
    """Radial bands must make one module valid at any input size, including odd ones.

    A per-frequency-bin mask could not do this, and multi-resolution testing at 512/640/
    800/1024 is a planned experiment, so the parameterisation has to survive it.
    """
    from saryolo.nn.modules import SpatialFrequencyRepresentation

    module = SpatialFrequencyRepresentation(16, mode="sff", bands=8).eval()
    for shape in ((16, 16), (32, 32), (24, 40), (17, 23)):
        x = torch.randn(1, 16, *shape)
        with torch.no_grad():
            out = module(x)
        assert out.shape == x.shape, (shape, tuple(out.shape))
        assert torch.isfinite(out).all()


def test_frequency_module_survives_half_precision():
    """AMP: ``torch.fft`` has no fp16 CPU kernel, so the spectral path must upcast.

    Under autocast a float16 activation reaches this module. Calling ``torch.fft`` on it
    directly raises, so the internal upcast is required for mixed-precision training, not
    merely tidy.
    """
    from saryolo.nn.modules import SpatialFrequencyRepresentation

    module = SpatialFrequencyRepresentation(16, mode="sff", bands=8).eval()
    x = torch.randn(1, 16, 16, 16, dtype=torch.float16)
    with torch.no_grad():
        out = module(x)
    assert out.dtype == torch.float16, out.dtype
    assert torch.isfinite(out).all()
