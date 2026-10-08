"""The SSAC prototype: the ten functional checks the master workflow requires.

Phase 4 of ``docs/ssac_design.md`` lists ten functional tests a minimal independent
prototype must pass — initialisation, a synthetic forward pass, output shape, gradient
flow through the core mechanism, loss, an optimiser update, numerical stability,
batch/size handling, checkpoint save+restore, and basic inference. They are all here,
plus the two claims that make the design falsifiable rather than merely functional:

* the **matched fixed-computation control** is parameter-identical to the proposal by
  construction, so "adaptive beats fixed" cannot be explained by capacity; and
* the proposal's allocation is genuinely *spatial* while the control's is not, which is
  the difference the whole mechanism claims and which a collapsed gate would erase.

The one deliberate asymmetry, shared with ``test_cfar_frontend.py``: identity at
initialisation comes from the zero-initialised residual gate (``alpha``), so the
sub-modules legitimately receive zero gradient at step 0 and non-zero gradient once
``alpha`` has moved. That is asserted in both directions so it cannot be mistaken for a
frozen branch — the bug this repository already hit once in Component 1.

A second layer of checks covers the cost ablation: sparse tile execution with a derived
halo and the ``keep = 1`` equivalence that proves the halo is right, the sparsity-penalty
arm, and the argument-order contract that ties the generated YAML to the constructor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from saryolo.nn.arch import SSAC_EXECUTIONS, SSAC_MODES, VARIANTS, ModelSpec, build_yaml_dict, variant_filename
from saryolo.nn.model import SARYOLODetectionModel
from saryolo.nn.modules import ScatterSelectiveRefinement

REPO_ROOT = Path(__file__).resolve().parents[1]


def _block(mode: str = "adaptive", c: int = 16, **kwargs) -> ScatterSelectiveRefinement:
    return ScatterSelectiveRefinement(c, mode=mode, **kwargs)


def _detection_batch(size: int = 64) -> dict:
    """A two-image synthetic batch with one box each, matching the repo's loss tests."""
    return {
        "img": torch.rand(2, 3, size, size),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.08], [0.3, 0.7, 0.05, 0.05]]),
    }


# ------------------------------------------------------------------ 1. initialisation
def test_the_block_initialises_and_declares_its_modes_and_gate():
    """Every declared mode must build, and the identity contract must be declared."""
    for mode in ScatterSelectiveRefinement.MODES:
        block = _block(mode)
        assert block.mode == mode
        assert isinstance(block.alpha, torch.nn.Module)
        assert block.identity_at_init is True
    assert tuple(SSAC_MODES) == tuple(ScatterSelectiveRefinement.MODES), (
        "the builder literal and the module's own MODES list have drifted apart"
    )


def test_the_block_is_an_exact_identity_at_initialisation_in_every_mode():
    """``alpha == 0`` must give ``out == x`` bit-for-bit, not merely approximately."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 16, 16)
    for mode in ScatterSelectiveRefinement.MODES:
        assert torch.equal(_block(mode).eval()(x), x), f"{mode}: not an exact identity at init"


def test_the_identity_is_not_because_the_branch_is_degenerate():
    """Moving the gate has to change the output, or the identity test proves nothing."""
    block = _block("adaptive").eval()
    x = torch.randn(2, 16, 16, 16)
    with torch.no_grad():
        block.alpha.raw.fill_(1.0)
    out = block(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x), "the residual gate has no effect on the output"


# ---------------------------------------------------------------- 2-3. forward + shape
def test_forward_pass_on_synthetic_input_keeps_shape_and_stays_finite():
    """Single input, single output, channel and spatial shape preserved."""
    torch.manual_seed(0)
    for mode in ScatterSelectiveRefinement.MODES:
        block = _block(mode).eval()
        x = torch.randn(3, 16, 20, 24)
        out = block(x)
        assert out.shape == x.shape, f"{mode}: {tuple(out.shape)} != {tuple(x.shape)}"
        assert torch.isfinite(out).all(), f"{mode}: non-finite output"


def test_the_gain_map_has_the_declared_shape_and_a_valid_range():
    """The allocation is a per-region map in (0, 1), shape ``(B, 1, H, W)``."""
    block = _block("adaptive").eval()
    x = torch.randn(2, 16, 12, 12)
    g = block.gain_map(x)
    assert g.shape == (2, 1, 12, 12)
    assert (g > 0).all() and (g < 1).all()


# ------------------------------------------------- 4. gradient flow through the mechanism
def test_the_core_mechanism_receives_gradient_at_initialisation():
    """The gate must get gradient at init, and the branch once it has moved.

    Identity at init comes from ``alpha == 0`` alone. With ``alpha == 0`` the sub-modules
    legitimately see ``dL/d(branch) = alpha * ... = 0``, so the assertion is made twice:
    the gate's gradient is non-zero immediately, and the scorer's gradient becomes
    non-zero the moment ``alpha`` leaves zero. A branch that stayed at zero gradient
    *after* ``alpha`` moved would be the frozen-component bug.
    """
    torch.manual_seed(0)
    block = _block("adaptive").train()
    x = torch.randn(2, 16, 16, 16)
    block(x).pow(2).mean().backward()
    assert block.alpha.raw.grad is not None and block.alpha.raw.grad.abs().item() > 0, (
        "the residual gate receives zero gradient at init and can never switch on"
    )
    assert block.head[0].weight.grad is not None and block.head[0].weight.grad.abs().sum() == 0, (
        "the scorer should be masked by alpha==0 at step 0; if it is not, this test no "
        "longer documents the intended pattern"
    )
    with torch.no_grad():
        block.alpha.raw.fill_(1.0)
    block.zero_grad(set_to_none=True)
    block(x).pow(2).mean().backward()
    assert block.head[0].weight.grad.abs().sum() > 0, (
        "the assessment scorer never receives gradient: the allocation is not trained"
    )


def test_the_allocation_is_spatial_in_the_proposal_and_constant_in_the_control():
    """The mechanism is *per-region*; the control is the same model without that property.

    Both modes carry exactly the same parameters. The only difference is whether the
    evidence logit is per-location (proposal) or averaged over the map (control), so this
    pair is what separates adaptivity from capacity.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 16, 20, 20)
    adaptive = _block("adaptive").eval().gain_map(x)
    fixed = _block("fixed").eval().gain_map(x)
    # Range rather than std: the control's map is (B, 1, 1, 1), where an unbiased std over
    # a single element is NaN and the comparison would be meaningless.
    span = lambda g: (g.amax(dim=(2, 3)) - g.amin(dim=(2, 3)))  # noqa: E731
    assert span(adaptive).mean().item() > 0, "the proposal's allocation does not vary spatially"
    assert span(fixed).max().item() == 0.0, "the control's allocation varies spatially"
    # And the control still varies *between images*: it is a learned scalar, not a constant.
    assert (fixed.max() - fixed.min()).item() > 0


# ------------------------------------------------------------------ 5-6. loss + update
def test_the_arm_carries_only_the_core_mechanism():
    """The comparison is attributable only if the arm differs in one place."""
    assert VARIANTS["ssac_s"].module_names == ["ssac"]
    assert VARIANTS["ssac_raw_s"].module_names == ["ssac"]
    assert VARIANTS["ssac_fixed_s"].module_names == ["ssac"]
    assert VARIANTS["ssac_n"].scale == "n" and VARIANTS["ssac_fixed_n"].scale == "n"


def test_the_control_is_parameter_identical_to_the_proposal():
    """Real parity, not approximate: ``adaptive`` and ``fixed`` must match exactly.

    If the counts differed, a difference between the arms could be the capacity rather
    than the allocation — which is the confound the control exists to remove. The
    assessment alternative differs on purpose (it reads the raw feature, a different
    width), and is asserted to differ so the three arms cannot have collapsed together.
    """
    def count(variant: str) -> int:
        spec = VARIANTS[variant]
        assert (REPO_ROOT / "configs" / "models" / variant_filename(spec)).exists(), (
            f"{variant}: missing generated YAML; run `python -m saryolo arch --variant all`"
        )
        model = SARYOLODetectionModel(build_yaml_dict(spec), ch=3, nc=1, verbose=False)
        return sum(p.numel() for p in model.parameters())

    proposal, control, alternative = count("ssac_s"), count("ssac_fixed_s"), count("ssac_raw_s")
    assert proposal == control, (
        f"the matched fixed-computation control is not parameter-matched: {proposal} vs {control}"
    )
    assert proposal != alternative, "the assessment alternative collapsed onto the proposal"


@pytest.mark.parametrize(
    "variant",
    ["ssac_s", "ssac_raw_s", "ssac_fixed_s", "ssac_sparse_s", "ssac_e1_s", "ssac_pen_s"],
)
def test_the_graph_builds_forwards_and_losses_without_an_error(variant):
    """Phase 4 requires forward *and* loss on the real graph, not just the module."""
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS[variant]), ch=3, nc=1, verbose=False)
    model.train()
    blocks = [m for m in model.modules() if isinstance(m, ScatterSelectiveRefinement)]
    assert len(blocks) == 3, f"expected one block per detection level, found {len(blocks)}"
    loss, items = model.loss(_detection_batch())
    assert bool(torch.isfinite(loss).all()), f"{variant}: loss is not finite ({loss})"
    assert float(loss.sum()) > 0.0
    assert "box_loss" in items and "cls_loss" in items


def test_the_prototype_takes_an_optimiser_step_through_the_gate():
    """Gradient flow must reach the optimiser, not merely exist on a tensor."""
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_s"]), ch=3, nc=1, verbose=False)
    model.train()
    gate = next(m for m in model.modules() if isinstance(m, ScatterSelectiveRefinement)).alpha
    batch = _detection_batch()
    model.loss(batch)  # builds the criterion lazily
    optimiser = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2, momentum=0.9
    )
    loss, _ = model.loss(batch)
    assert bool(torch.isfinite(loss).all())
    optimiser.zero_grad()
    loss.sum().backward()
    before = gate.raw.detach().clone()
    optimiser.step()
    assert not torch.equal(before, gate.raw.detach()), (
        "the core mechanism's gate did not move after an optimiser step"
    )


# --------------------------------------------------------- 7-8. stability + size handling
def test_the_statistics_are_finite_on_degenerate_inputs():
    """Flat and all-zero maps are the cases where a log and a ratio go wrong.

    Open sea and uniform desert are locally flat enough to reach this, so a NaN here would
    poison a batch and look like a bad learning rate.
    """
    for mode in ScatterSelectiveRefinement.MODES:
        block = _block(mode).eval()
        for x in (torch.full((1, 16, 16, 16), 0.3), torch.zeros(1, 16, 16, 16)):
            assert torch.isfinite(block.statistics(x)).all(), f"{mode}: non-finite statistic"
            assert torch.isfinite(block(x)).all(), f"{mode}: non-finite output"


@pytest.mark.parametrize(("batch", "size"), [(1, 32), (4, 64), (2, 40)])
def test_batch_size_and_input_size_are_handled(batch, size):
    """Odd sizes stress the padded pooling; small batches stress the per-image allocation."""
    block = _block("adaptive").eval()
    x = torch.randn(batch, 16, size, size)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_an_unknown_mode_and_a_degenerate_window_are_refused():
    """Modes, windows and the expand factor are validated where the mistake is made."""
    with pytest.raises(ValueError, match="mode must be one of"):
        _block("adaptive_v2")
    with pytest.raises(ValueError, match="odd"):
        _block(scales=(3, 4))
    with pytest.raises(ValueError, match="empty"):
        _block(scales=())
    with pytest.raises(ValueError, match="expand"):
        _block(expand=0)
    with pytest.raises(ValueError, match="select"):
        _block(select=1.0)


# ------------------------------------------------------ 9. checkpoint save + restore
def test_checkpoint_save_and_restore_preserves_the_output(tmp_path):
    """A saved block must reload and reproduce its output exactly."""
    torch.manual_seed(0)
    block = _block("adaptive").eval()
    with torch.no_grad():  # train it a little so the state is not the trivial init
        for _ in range(3):
            block.alpha.raw.fill_(0.5)
            block.head[-1].weight.add_(0.01)
    x = torch.randn(2, 16, 16, 16)
    expected = block(x)
    path = tmp_path / "ssac.pt"
    torch.save(block.state_dict(), path)

    restored = _block("adaptive").eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(restored(x), expected), "the restored block does not reproduce the output"


def test_a_restored_model_reproduces_its_detections(tmp_path):
    """The end-to-end checkpoint path: save the whole model, reload, same output."""
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_n"]), ch=3, nc=1, verbose=False)
    model.eval()
    # Move the gates off zero so the checkpoint is not the trivial identity.
    for m in model.modules():
        if isinstance(m, ScatterSelectiveRefinement):
            with torch.no_grad():
                m.alpha.raw.fill_(0.4)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ref = model(x)[0]
    torch.save(model.state_dict(), tmp_path / "model.pt")
    fresh = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_n"]), ch=3, nc=1, verbose=False)
    fresh.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    fresh.eval()
    with torch.no_grad():
        got = fresh(x)[0]
    assert torch.equal(ref, got), "the restored model does not reproduce its detections"


# ------------------------------------------------------------------ 10. basic inference
def test_basic_inference_is_finite_and_shape_stable():
    """Inference on a real-sized input returns a finite detection tensor."""
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_n"]), ch=3, nc=1, verbose=False)
    model.eval()
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        out = model(x)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    assert pred.shape[0] == 1 and pred.shape[1] == 5  # 4 box + 1 class
    assert torch.isfinite(pred).all()


def test_the_core_mechanism_cost_is_measured_and_reported(tmp_path):
    """The overhead is measured here rather than asserted in prose.

    The mechanism is dense in this implementation, so it *adds* compute; that price is
    stated rather than hidden, because the master workflow forbids an efficiency claim
    that rests on theory. What is asserted is the honest shape of the cost: more
    parameters and more FLOPs than the stock detector, parameter-parity with the control,
    and a non-empty selected fraction to report.
    """
    from saryolo.evaluation.efficiency import profile_yaml

    def profile(variant: str) -> dict:
        path = REPO_ROOT / "configs" / "models" / variant_filename(VARIANTS[variant])
        assert path.exists(), f"{variant}: missing generated YAML {path.name}"
        return profile_yaml(path, imgsz=320, nc=1)

    baseline = profile("baseline_s")
    proposal = profile("ssac_s")
    control = profile("ssac_fixed_s")
    assert proposal["params"] > baseline["params"], "the mechanism reports no parameters"
    assert proposal["flops_G"] > baseline["flops_G"], (
        "the mechanism reports no added compute; the cost is being hidden, not paid"
    )
    assert control["params"] == proposal["params"], (
        "the fixed-computation control and the proposal disagree on parameter count"
    )


# ============================================================ execution and cost ablation
# The checks below cover the layer the cost claim rests on. Each of them is a way the
# ablation could be *wrong while still running*: a sparse mode that skipped the wrong tiles,
# a `keep` the parser read as a `tile`, an "unpenalised" arm that quietly carried a penalty.
# A build that merely trains and reports a latency would not distinguish any of those.


def test_the_execution_vocabulary_is_declared_once():
    """``arch.SSAC_EXECUTIONS`` and the module's own list must not drift apart."""
    assert tuple(SSAC_EXECUTIONS) == tuple(ScatterSelectiveRefinement.EXECUTIONS)
    assert set(SSAC_EXECUTIONS) <= {"dense", "sparse"}


def test_the_yaml_argument_order_matches_the_module_signature():
    """The generated YAML's positional arguments must land on the fields they name.

    ``parse_model`` calls the module positionally, so inserting a constructor argument in
    the middle of the list would not fail — it would silently build a different mechanism
    (a ``keep`` read as a ``tile``, say) and every number downstream would describe a model
    nobody chose. The order is therefore read back out of the generated file and compared
    field by field, against the same spec that generated it.
    """
    import yaml

    spec = VARIANTS["ssac_sparse_s"]
    text = (REPO_ROOT / "configs" / "models" / variant_filename(spec)).read_text()
    rows = yaml.safe_load(text)
    rows = [r for r in rows["head"] if r[2] == "ScatterSelectiveRefinement"]
    assert len(rows) == len(spec.levels), "expected one SSAC row per detection level"
    call = list(rows[0][3])
    assert call[0] == "ch", "the first YAML argument must be the parser's channel list"
    call[0] = [64] * 32  # the parser substitutes its own ch list before calling us
    block = ScatterSelectiveRefinement(*call)
    assert block.c1 == 64, "the source index did not resolve against the channel list"
    assert block.mode == spec.ssac
    assert block.scales == tuple(spec.ssac_scales)
    assert block.hidden == spec.ssac_hidden
    assert block.expand == spec.ssac_expand
    assert block.execution == spec.ssac_execution
    assert block.tile == spec.ssac_tile
    assert block.keep == spec.ssac_keep
    assert block.penalty == spec.ssac_penalty
    assert block.tau == spec.ssac_tau
    assert block.select == spec.ssac_select


@pytest.mark.parametrize("size", [16, 40, 37])
def test_sparse_execution_reproduces_dense_execution_when_every_tile_is_selected(size):
    """The condition that makes the sparse mode checkable: ``keep = 1`` in ``eval`` mode.

    Every tile selected, halos in place, and the halo-padded gather plus the tile-wise
    BatchNorm statistics (eval mode uses the running estimates, so a tile sees the same
    normalisation it would see in the full map). 37 is not a multiple of the tile size, so
    this also covers the padded bottom/right edge.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 16, size, size)
    dense = _block("adaptive", execution="dense").eval()
    sparse = _block("adaptive", execution="sparse", keep=1.0, tile=16).eval()
    sparse.load_state_dict(dense.state_dict())
    with torch.no_grad():
        for block in (dense, sparse):
            block.alpha.raw.fill_(0.5)
        a, b = dense(x), sparse(x)
    drift = float((a - b).abs().max())
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (
        f"sparse execution with every tile selected differs from dense execution by {drift}"
    )


def test_sparse_execution_leaves_unselected_tiles_at_the_cheap_path():
    """Skipping must be *real* skipping, and the skipped tiles must be left alone.

    The mix in ``forward`` gives an unselected location the cheap path's value anyway, so a
    sparse build that refined everything and then masked would produce the same output —
    and would be measuring nothing. The check is therefore made on the routing itself: the
    selected tile is refined exactly as the dense pass refines it, and the others come back
    bit-for-bit equal to the cheap features they were given.
    """
    torch.manual_seed(0)
    block = _block("adaptive", c=8, execution="sparse", keep=0.25, tile=8).eval()
    f = torch.randn(1, 8, 16, 16)
    g = torch.zeros(1, 1, 16, 16)
    g[..., :8, :8] = 1.0  # exactly one tile carries all the evidence
    selected = block._selected_tiles(block._tile_scores(g))
    assert selected.tolist() == [[0]], f"the routing did not pick the evidenced tile: {selected}"
    with torch.no_grad():
        rich = block._refine_sparse(f, g)
        dense_rich = block.rich(f)
    assert torch.equal(rich[..., :8, :8], dense_rich[..., :8, :8]), (
        "the selected tile was not refined exactly as the dense pass refines it"
    )
    assert torch.equal(rich[..., 8:, :], f[..., 8:, :]), "an unselected tile was still refined"
    assert torch.equal(rich[..., :8, 8:], f[..., :8, 8:]), "an unselected tile was still refined"


def test_the_routing_report_states_what_a_sparse_build_skips_and_what_it_costs_in_halo():
    """``routing_stats`` is the number the efficiency story is built on, so it is asserted.

    ``executed_rich_fraction`` includes the halo, because a tile gathered with its context
    is genuinely more arithmetic than the tile alone -- a routing report that quoted the
    tile fraction alone would overstate the saving.
    """
    torch.manual_seed(0)
    block = _block("adaptive", c=8, execution="sparse", keep=0.25, tile=8).eval()
    stats = block.routing_stats(torch.randn(1, 8, 32, 32))
    assert stats["tiles"] == 16 and stats["selected_tiles"] == 4
    assert stats["selected_tile_fraction"] == pytest.approx(0.25, abs=1e-4)
    halo = ((8 + 2 * stats["halo"]) / 8) ** 2
    assert stats["executed_rich_fraction"] == pytest.approx(0.25 * halo, abs=1e-3)
    assert 0.0 <= stats["gate_mean"] <= 1.0
    assert 0.0 <= stats["gate_selected_fraction"] <= 1.0
    assert stats["halo"] == 2


def test_the_halo_is_derived_from_the_expensive_path_kernels():
    """The halo must follow the expensive path, not a constant someone remembered.

    If the halo were too small the sparse pass would refine a tile with the wrong context
    and the ``keep = 1`` equivalence above would fail; if it were hard-coded, changing the
    kernel size would silently reintroduce that error. It is computed here, so this test
    changes the kernel and asserts that the radius follows.
    """
    from torch import nn

    block = _block("adaptive", c=8)
    assert block.halo == 2, "1x1 -> DW5x5 -> 1x1 has a receptive radius of 2"
    dw = block.rich[1].dw
    block.rich[1].dw = nn.Conv2d(
        dw.in_channels, dw.out_channels, 7, padding=3, groups=dw.groups, bias=False
    )
    assert block._receptive_halo() == 3, "the halo no longer follows the expensive path"


def test_gradient_flows_through_the_sparse_gather_and_scatter():
    """Gathering and scattering must not detach the expensive path or the scorer."""
    torch.manual_seed(0)
    block = _block("adaptive", execution="sparse", keep=0.5, tile=8).train()
    with torch.no_grad():
        block.alpha.raw.fill_(1.0)
    block(torch.randn(2, 16, 16, 16)).pow(2).mean().backward()
    assert block.rich[0].conv.weight.grad.abs().sum() > 0, "the expensive path is detached"
    assert block.cheap.dw.weight.grad.abs().sum() > 0, "the cheap path is detached"
    assert block.head[0].weight.grad.abs().sum() > 0, "the allocation is detached"


@pytest.mark.parametrize(("batch", "size"), [(1, 16), (2, 33), (1, 7)])
def test_sparse_execution_handles_odd_and_single_tile_maps(batch, size):
    """Sizes that are not a multiple of the tile, and maps smaller than one tile."""
    torch.manual_seed(0)
    block = _block("adaptive", execution="sparse", keep=0.25, tile=16).eval()
    x = torch.randn(batch, 16, size, size)
    out = block(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert block.routing_stats(x)["selected_tiles"] >= 1, "routing selected nothing at all"


def test_sparse_execution_is_stable_on_degenerate_input():
    """Flat and zero maps: the routing still has to select something finite."""
    block = _block("adaptive", execution="sparse", keep=0.25, tile=8).eval()
    for x in (torch.full((1, 16, 16, 16), 0.3), torch.zeros(1, 16, 16, 16)):
        assert torch.isfinite(block(x)).all()


def test_sparse_execution_changes_no_parameter():
    """A sparse arm and its dense twin must be the same model, differently executed.

    This is what makes a cost difference attributable to the execution rather than to
    capacity, and it is also why a dense-trained checkpoint loads into a sparse build
    without a missing or unexpected key.
    """
    dense = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_n"]), ch=3, nc=1, verbose=False)
    sparse = SARYOLODetectionModel(
        build_yaml_dict(VARIANTS["ssac_sparse_n"]), ch=3, nc=1, verbose=False
    )
    assert sum(p.numel() for p in dense.parameters()) == sum(
        p.numel() for p in sparse.parameters()
    ), "sparse execution changed the parameter count"
    assert list(dense.state_dict()) == list(sparse.state_dict())
    sparse.load_state_dict(dense.state_dict())  # strict: raises on any missing key
    assert VARIANTS["ssac_n"].ssac_execution == "dense"
    assert VARIANTS["ssac_sparse_n"].ssac_execution == "sparse"


def test_a_restored_sparse_model_reproduces_its_detections(tmp_path):
    """The sparse build's checkpoint path, end to end."""
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_sparse_n"]), ch=3, nc=1, verbose=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, ScatterSelectiveRefinement):
            with torch.no_grad():
                m.alpha.raw.fill_(0.4)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ref = model(x)[0]
    torch.save(model.state_dict(), tmp_path / "sparse.pt")
    fresh = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_sparse_n"]), ch=3, nc=1, verbose=False)
    fresh.load_state_dict(torch.load(tmp_path / "sparse.pt", weights_only=True))
    fresh.eval()
    with torch.no_grad():
        got = fresh(x)[0]
    assert torch.equal(ref, got), "the restored sparse model does not reproduce its detections"


def test_a_sparse_config_is_refused_when_it_would_skip_nothing():
    """A sparse arm at ``keep = 1`` pays the routing and can save nothing."""
    with pytest.raises(ValueError, match="refines every tile"):
        ModelSpec(
            name="ssac_sparse_bogus", ssac="adaptive", ssac_execution="sparse", ssac_keep=1.0
        )


def test_the_execution_and_tiling_arguments_are_validated():
    """Bad routing parameters are refused where they are written, not at run time."""
    with pytest.raises(ValueError, match="execution must be one of"):
        _block("adaptive", execution="batched")
    with pytest.raises(ValueError, match="tile must be"):
        _block("adaptive", tile=0)
    with pytest.raises(ValueError, match="keep must be"):
        _block("adaptive", keep=0.0)
    with pytest.raises(ValueError, match="keep must be"):
        _block("adaptive", keep=1.5)
    with pytest.raises(ValueError, match="penalty must be"):
        _block("adaptive", penalty=-0.1)


# ================================================================== sparsity penalty
# The penalty is deliberately *not* part of the proposal: penalising the allocation to be
# small is circular when sparsity is the mechanism's own claim. It exists as an arm so the
# premise a sparse build relies on -- that the mechanism can be pushed sparse without losing
# the detection -- is a measurement instead of an assumption.


def test_the_penalty_is_off_in_the_proposal_and_on_only_in_its_own_arm():
    """The switch has to be where the arm claims it is, and the default has to be off."""
    from saryolo.nn.losses import SAR_LOSS_DEFAULTS

    assert SAR_LOSS_DEFAULTS["w_ssac_sparsity"] == 0.0
    assert VARIANTS["ssac_n"].ssac_penalty == 0.0
    assert not VARIANTS["ssac_n"].sar_loss
    assert VARIANTS["ssac_pen_n"].ssac_penalty > 0.0
    assert VARIANTS["ssac_pen_n"].sar_loss == {"w_ssac_sparsity": VARIANTS["ssac_pen_n"].ssac_penalty}


def test_the_sparsity_penalty_enters_the_loss_and_reaches_the_allocation_at_step_zero():
    """The penalty must (a) be in the loss and (b) make the allocation trainable at init.

    (b) is the interesting part. Every SSAC mode is an exact identity at initialisation, so
    with ``alpha == 0`` the scorer's gradient is masked to zero and the allocation only
    starts learning once the residual gate has moved. A sparsity penalty acts on the mean
    allocation directly, so it trains the scorer from step 0 — a different optimisation
    trajectory for the same architecture, which is exactly what this arm is for.
    """
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_pen_n"]), ch=3, nc=1, verbose=False)
    model.train()
    blocks = [m for m in model.modules() if isinstance(m, ScatterSelectiveRefinement)]
    assert len(blocks) == 3
    loss, items = model.loss(_detection_batch())
    assert "ssac_sparsity" in items, "the arm declares a penalty but the loss does not carry it"
    means = torch.stack([m.last_gate_mean for m in blocks])
    assert all(m is not None for m in means)
    expected = VARIANTS["ssac_pen_n"].ssac_penalty * means.mean()
    assert float((items["ssac_sparsity"] - expected).abs()) < 1e-6, (
        f"the logged term {float(items['ssac_sparsity'])} is not the weight times the mean "
        f"allocation ({float(expected)})"
    )
    loss.sum().backward()
    assert blocks[0].head[0].weight.grad.abs().sum() > 0, (
        "the penalty does not train the allocation at initialisation, so it is not the "
        "mechanism's allocation that is being penalised"
    )


def test_the_unpenalised_arm_does_not_carry_the_term():
    """The contrast the penalty arm is read against: no term, and no early scorer gradient."""
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["ssac_n"]), ch=3, nc=1, verbose=False)
    model.train()
    loss, items = model.loss(_detection_batch())
    assert "ssac_sparsity" not in items
    block = next(m for m in model.modules() if isinstance(m, ScatterSelectiveRefinement))
    assert block.last_gate_mean is None, "an unpenalised block recorded an allocation"
    loss.sum().backward()
    assert block.head[0].weight.grad.abs().sum() == 0, (
        "the scorer is expected to be masked by alpha == 0 at step 0 in the unpenalised arm"
    )


def test_the_cost_ablation_arms_are_all_exercised_by_a_variant():
    """Every ablation the docs name must exist as a buildable arm, or its row is un-runnable."""
    for arm in ("ssac_sparse_s", "ssac_sparse_n", "ssac_e1_s", "ssac_e1_n", "ssac_pen_s", "ssac_pen_n"):
        assert arm in VARIANTS, f"missing ablation arm {arm}"
        model = SARYOLODetectionModel(build_yaml_dict(VARIANTS[arm]), ch=3, nc=1, verbose=False)
        blocks = [m for m in model.modules() if isinstance(m, ScatterSelectiveRefinement)]
        assert len(blocks) == 3
        assert min(m.numel() for m in model.parameters() if m.numel()) > 0
    # The expand ablation's whole point is that it is *smaller* than the proposal and still
    # larger than the stock detector: a build that came out equal to either would make the
    # row answer a different question than the one it is labelled with.
    def params(arm: str) -> int:
        model = SARYOLODetectionModel(build_yaml_dict(VARIANTS[arm]), ch=3, nc=1, verbose=False)
        return sum(p.numel() for p in model.parameters())

    assert params("baseline_n") < params("ssac_e1_n") < params("ssac_n")
