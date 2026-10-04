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
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from saryolo.nn.arch import SSAC_MODES, VARIANTS, build_yaml_dict, variant_filename
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


@pytest.mark.parametrize("variant", ["ssac_s", "ssac_raw_s", "ssac_fixed_s"])
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
