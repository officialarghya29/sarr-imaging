"""The SARVO prototype front end: forward, identity, gradient, loss and cost.

`docs/architecture_proposals.md` §2 recommends the radar-statistic input
representation and names the experiment that could kill it. This file is the
Phase-5 prototype contract: the module is the first representation, it is an
*exact* identity until it learns something, its gradient is non-zero at
initialisation (the failure that silently freezes a component), it survives the
real detection objective and an optimiser step, and its cost is measured rather
than estimated.

The one deliberate asymmetry: the ``fixed`` mode is *not* an identity at
initialisation. It is the control that separates "the statistic helps" from "the
learnable gain helps", and a control that equals the baseline measures nothing.
That is asserted explicitly, so the exemption cannot be mistaken for an oversight.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from saryolo.nn.arch import CFAR_MODES, VARIANTS, build_yaml_dict, variant_filename
from saryolo.nn.model import SARYOLODetectionModel
from saryolo.nn.modules import RatioSpaceCFARFrontEnd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _front(mode: str = "cfar", **kwargs) -> RatioSpaceCFARFrontEnd:
    return RatioSpaceCFARFrontEnd(3, mode=mode, **kwargs)


def _detection_batch(size: int = 64) -> dict:
    """A two-image synthetic batch with one box each, matching the repo's loss tests."""
    return {
        "img": torch.rand(2, 3, size, size),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.08], [0.3, 0.7, 0.05, 0.05]]),
    }


def test_the_proposed_mode_is_an_exact_identity_at_initialisation():
    """``g == 0`` must give ``out == x`` bit-for-bit, not merely approximately.

    This is the property the whole ablation rests on: any measured gain has to be
    attributable to what the statistic learned, never to a first representation that
    perturbed the input at step 0.
    """
    torch.manual_seed(0)
    front = _front().eval()
    x = torch.rand(2, 3, 32, 32)
    assert torch.equal(front(x), x), "the cfar front end is not an exact identity at init"


def test_the_front_end_stops_being_an_identity_once_the_gain_moves():
    """The test above must not be passing because the gain is disconnected.

    A module that can only ever return its input would satisfy the identity check and
    be useless. Moving one gain parameter off zero has to change the output.
    """
    front = _front().eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        front.gain[-1].bias.fill_(0.7)
    out = front(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x), "the gain has no effect on the output"


def test_the_statistic_stack_is_finite_on_a_constant_image():
    """A flat region is the degenerate case: zero variance, undefined ratio.

    Real SAR sea and desert are locally flat enough to reach it, so a NaN here would
    poison a whole batch during training and the failure would look like a bad
    learning rate.
    """
    front = _front().eval()
    x = torch.full((1, 3, 16, 16), 0.3)
    stats = front.statistics(x)
    expected_channels = 1 + 2 * len(front.scales)
    assert stats.shape[:2] == (1, expected_channels)
    assert torch.isfinite(stats).all(), "the statistic stack has a non-finite entry"
    assert torch.isfinite(front(x)).all()


def test_the_statistic_stack_carries_the_ratio_and_the_texture_channels():
    """The stack must actually contain ``log-ratio`` and coefficient-of-variation pairs.

    The design claim is that a learned threshold over the CFAR decision variable is what
    the module contributes. If the stack silently collapsed to one channel per scale
    there would be nothing to threshold.
    """
    front = _front(scales=(3, 7)).eval()
    x = torch.rand(1, 3, 24, 24)
    stats = front.statistics(x)
    assert stats.shape[1] == 5  # intensity + (ratio, contrast) x 2 scales
    ratios, contrasts = stats[:, 1::2], stats[:, 2::2]
    assert ratios.shape[1] == contrasts.shape[1] == 2
    # A local coefficient of variation is non-negative by construction.
    assert (contrasts >= 0).all()


def test_the_gain_parameters_receive_gradient_at_initialisation():
    """Identity at init must come from the zero init, not from a dead branch.

    A zero-initialised *branch* under a zero-initialised gate has a gate gradient of
    exactly zero and never trains while looking healthy -- the bug this repository
    already hit once in Component 1. The assertion is therefore on the gradient, not
    only on the value.
    """
    torch.manual_seed(0)
    front = _front().train()
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    front(x).sum().backward()
    last = front.gain[-1]
    assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0
    assert last.bias.grad is not None and last.bias.grad.abs().sum() > 0

    # The first layer legitimately has a *zero* gradient at step 0, and that is a
    # property of zero-initialising the last layer rather than a freeze: the path to
    # ``gain[0]`` runs through ``gain[-1]``'s weights, which are zero, so it cannot
    # receive gradient until they move. The next two assertions separate the benign
    # version of that from the frozen-branch bug -- the last layer moves, and once it
    # has, the first layer's gradient becomes non-zero.
    first = front.gain[0]
    assert first.weight.grad is not None and first.weight.grad.abs().sum() == 0
    with torch.no_grad():
        last.weight.add_(0.1)
    front.zero_grad()
    front(x).sum().backward()
    assert first.weight.grad.abs().sum() > 0, (
        "the first layer never receives gradient, so the statistic is not trained"
    )


def test_the_matched_cost_control_has_exact_parameter_parity_and_is_identity_at_init():
    """The control that separates "the statistic" from "the extra parameters".

    Real parameter *parity*, not approximate: the gain network has identical layer shapes
    in both modes, so the only difference between the two arms is what it reads. If the
    counts differed, a difference between the arms could be the capacity rather than the
    representation -- which is exactly the confound the control exists to remove.
    """
    proposed = _front("cfar")
    control = _front("conv")
    assert sum(p.numel() for p in proposed.parameters()) == sum(
        p.numel() for p in control.parameters()
    ), "the matched-cost control is not parameter-matched"
    assert control.identity_at_init is True
    x = torch.rand(2, 3, 32, 32)
    assert torch.equal(control.eval()(x), x)


def test_the_two_controls_answer_different_questions():
    """One holds the budget fixed and removes the statistic; the other does the reverse.

    Neither substitutes for the other, so a mode that collapsed into the other would leave
    the design with only half a falsifier while both tests still passed.
    """
    matched = _front("conv")
    fixed = _front("fixed")
    assert sum(p.numel() for p in matched.parameters()) > 0
    assert sum(p.numel() for p in fixed.parameters()) == 0
    assert matched.identity_at_init is True and fixed.identity_at_init is False


def test_the_fixed_threshold_control_has_no_learnable_parameters_and_is_not_identity():
    """The control must isolate the statistic, so nothing in it may be learned.

    And it must perturb the input: a control whose output equals the baseline would
    attribute any difference to nothing at all.
    """
    front = _front("fixed").eval()
    assert sum(p.numel() for p in front.parameters()) == 0
    assert front.identity_at_init is False
    assert front.identity_at_init != _front("cfar").identity_at_init
    x = torch.rand(2, 3, 32, 32)
    assert not torch.equal(front(x), x)


def test_an_unknown_mode_and_a_degenerate_window_are_refused():
    """Modes and windows are validated where the mistake is made, not at build time.

    A ``2`` window has no centred padding and an even-sized pool shifts the statistic by
    half a pixel; both produce a plausible-looking model with a subtly wrong
    representation, which is worse than a raise.
    """
    with pytest.raises(ValueError, match="mode must be one of"):
        _front("cfar_v2")
    with pytest.raises(ValueError, match="odd"):
        _front(scales=(3, 4))
    with pytest.raises(ValueError, match="empty"):
        _front(scales=())


def test_the_builder_and_the_module_declare_the_same_modes():
    """The literal in ``arch.py`` and the module's own list must not drift apart."""
    assert tuple(CFAR_MODES) == tuple(RatioSpaceCFARFrontEnd.MODES)


def test_the_prototype_arm_sits_on_the_image_and_carries_nothing_else():
    """The comparison is attributable only if the arm differs in exactly one place."""
    spec = VARIANTS["cfar_s"]
    assert spec.cfar == "cfar"
    assert spec.module_names == ["cfar"], (
        f"the prototype arm carries {spec.module_names}; it must differ from the stock "
        "detector in the first representation and nothing else"
    )
    assert VARIANTS["cfar_fixed_s"].module_names == ["cfar"]
    assert VARIANTS["cfar_fixed_s"].cfar == "fixed"
    assert VARIANTS["cfar_conv_s"].module_names == ["cfar"]
    assert VARIANTS["cfar_conv_s"].cfar == "conv"
    assert VARIANTS["cfar_n"].scale == "n" and VARIANTS["cfar_conv_n"].scale == "n"


@pytest.mark.parametrize("variant", ["cfar_s", "cfar_conv_s", "cfar_fixed_s"])
def test_the_prototype_graph_builds_forwards_and_losses_without_an_error(variant):
    """Phase 5 requires forward *and* loss on the real graph, not just the module."""
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS[variant]), ch=3, nc=1, verbose=False)
    model.train()
    fronts = [m for m in model.modules() if isinstance(m, RatioSpaceCFARFrontEnd)]
    assert len(fronts) == 1, f"expected one front end, found {len(fronts)}"
    batch = _detection_batch()
    loss, items = model.loss(batch)
    assert bool(torch.isfinite(loss).all()), f"{variant}: loss is not finite ({loss})"
    assert float(loss.sum()) > 0.0
    assert "box_loss" in items and "cls_loss" in items


def test_the_prototype_takes_an_optimiser_step_through_the_gain():
    """Gradient flow has to reach the optimiser, not merely exist on a tensor.

    The check is deliberately on the *parameter* after ``step()``: a non-zero ``.grad``
    that never moves the value is how a component can report a gradient and still be
    a permanent no-op.
    """
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["cfar_s"]), ch=3, nc=1, verbose=False)
    model.train()
    front = next(m for m in model.modules() if isinstance(m, RatioSpaceCFARFrontEnd))
    batch = _detection_batch()
    model.loss(batch)  # builds the criterion lazily
    optimiser = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2, momentum=0.9
    )
    loss, _ = model.loss(batch)
    assert bool(torch.isfinite(loss).all())
    optimiser.zero_grad()
    loss.sum().backward()
    before = front.gain[-1].bias.detach().clone()
    optimiser.step()
    assert not torch.equal(before, front.gain[-1].bias.detach()), (
        "the front end's gain did not move after an optimiser step"
    )


def test_the_front_end_cost_is_measured_and_inside_the_budget_it_claims():
    """The parameter overhead is measured here rather than asserted in prose.

    The proposals document puts the cost under the 0.5 % budget the repository already
    holds conditioning to. That is a claim about a measurement, so it is made by the
    profiler on the generated YAML, and the compute is required to be *visible* rather
    than hidden: the module adds FLOPs, and the test says so.
    """
    from saryolo.evaluation.efficiency import profile_yaml

    def profile(variant: str) -> dict:
        path = REPO_ROOT / "configs" / "models" / variant_filename(VARIANTS[variant])
        assert path.exists(), f"{variant}: missing generated YAML {path.name}"
        return profile_yaml(path, imgsz=320, nc=1)

    baseline = profile("baseline_s")
    prototype = profile("cfar_s")
    # Compared on the integer count, not on the rounded ``params_M``: the front end is
    # ~200 parameters against ~9.4 M, so a three-decimal Megaparameter column rounds it
    # to zero and the check would compare 0.000 with 0.000.
    overhead_pct = (prototype["params"] - baseline["params"]) / baseline["params"] * 100
    assert 0.0 < overhead_pct < 0.5, (
        f"the front end costs {overhead_pct:.4f} % of the baseline's parameters, "
        "outside the budget the proposal claims"
    )
    assert prototype["flops_G"] > baseline["flops_G"], (
        "the front end reports no added compute; the cost is being hidden, not paid"
    )
    # The matched-cost control must match on parameters exactly, and must be *no more*
    # expensive in compute than the arm it controls for -- otherwise "matched cost" is a
    # description of neither model.
    matched = profile("cfar_conv_s")
    assert matched["params"] == prototype["params"], (
        "the matched-cost control and the prototype disagree on parameter count"
    )
    assert matched["flops_G"] <= prototype["flops_G"]
