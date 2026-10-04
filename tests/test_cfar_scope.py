"""Advanced property and scope tests for the SARVO prototype front end.

`tests/test_cfar_frontend.py` is the Phase-5 contract: it proves the module is an
exact identity at init, that its gradient is alive, and that it costs what the
proposal says. This file is the *scope and efficiency* deep-scan on top of that
contract, and it exists to catch three classes of mistake that the contract does
not:

* **A mechanism that is not actually the claimed mechanism.** The design says the
  log-ratio channel is a CFAR decision variable, which is only true if it is
  invariant to a global radiometric gain (the acquisition variable the statistic
  is supposed to transfer across). That is a property of the *numbers*, so it is
  tested on the numbers here rather than asserted in prose.
* **A scope that drifted.** Each declared mode must be reachable as a real arm, at
  both scales, and every arm must still differ from the stock detector in exactly
  one place. A mode that quietly stopped being generated would leave the design
  with one fewer falsifier while every existing test still passed.
* **An efficiency claim that only holds at one scale.** The parameter overhead is a
  function of the front end alone, so it must be the same at ``n`` and ``s`` and
  inside the budget at both -- not a coincidence at the scale that happened to be
  profiled first.

Finally, the gain-collapse diagnostic the proposal names as its own falsifier is
unit-tested here: a detector that reports a per-pixel gain it never varies must be
caught, and a genuinely varying gain must not be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from saryolo.evaluation.gain import find_front_end, summarise_gain
from saryolo.nn.arch import VARIANTS, build_yaml_dict, variant_filename
from saryolo.nn.model import SARYOLODetectionModel
from saryolo.nn.modules import CUSTOM_MODULES, RatioSpaceCFARFrontEnd

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_FRONT_END_ARMS = ("cfar_n", "cfar_s", "cfar_conv_n", "cfar_conv_s", "cfar_fixed_n", "cfar_fixed_s")


def _front(mode: str = "cfar", **kwargs) -> RatioSpaceCFARFrontEnd:
    return RatioSpaceCFARFrontEnd(3, mode=mode, **kwargs)


# --------------------------------------------------------- the mechanism, on numbers
def test_the_log_ratio_channel_is_invariant_to_a_global_radiometric_gain():
    """A CFAR decision variable must transfer across absolute brightness.

    The acquisition shift that matters is a change of sensor gain: the same scene
    arrives at a different absolute level. ``r_k = log x - box(log x)`` cancels a
    constant multiplicative gain exactly, which is what makes a learned threshold
    over ``r_k`` meaningful across acquisitions. If this were false, the proposal's
    central motivation would be false with it, so it is checked directly.
    """
    torch.manual_seed(0)
    front = _front().eval()
    x = torch.rand(2, 3, 32, 32) + 0.05
    base = front.statistics(x)
    for gain in (0.3, 3.0):
        shifted = front.statistics(x * gain)
        assert torch.allclose(shifted[:, 1::2], base[:, 1::2], atol=1e-4), (
            f"the log-ratio channels moved under a x{gain} radiometric gain; the statistic "
            "is not the dimensionless quantity the proposal claims"
        )
        # The intensity channel must track the gain exactly, or the invariance above
        # would be trivial (a statistic stack that discarded the image entirely).
        assert torch.allclose(shifted[:, 0], base[:, 0] * gain, rtol=1e-4), (
            "the intensity channel did not scale with the radiometric gain"
        )


def test_the_statistic_stack_is_not_purely_dimensional_under_a_gain_shift():
    """The *whole* stack must not be invariant, or the pilot's gain could not respond.

    Only the ratio channels are gain-invariant; the local coefficient of variation
    is a contrast relative to the local log-mean, so a change of absolute level
    moves it. This is checked so the diagnostic's expectation is honest: the gain
    network is *allowed* to react to an acquisition shift, and does not have to be
    frozen to prove anything.
    """
    torch.manual_seed(0)
    front = _front().eval()
    x = torch.rand(2, 3, 32, 32) + 0.05
    base = front.statistics(x)
    shifted = front.statistics(x * 0.3)
    assert not torch.allclose(shifted[:, 2::2], base[:, 2::2], atol=1e-3), (
        "the coefficient-of-variation channels are invariant to the gain, so the stack "
        "carries no information about the acquisition level at all"
    )


def test_forward_applies_exactly_the_gain_map_it_reports():
    """``forward`` and the diagnostic must read the same quantity.

    The diagnostic can only be trusted if the tensor it measures is the one that is
    applied. A refactor that computed the gain twice -- once for the output and once
    for the report -- could make the report describe a gain the detector never uses.
    """
    torch.manual_seed(0)
    front = _front().eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        front.gain[-1].bias.fill_(0.4)
    assert torch.allclose(front(x), x * (1.0 + front.gain_map(x)), atol=1e-6)


# ------------------------------------------------------------------------- scope
def test_every_declared_mode_is_reachable_as_a_real_arm():
    """Each mode must exist as a generated arm, not only as a constructor argument."""
    declared = set(RatioSpaceCFARFrontEnd.MODES)
    reachable = {spec.cfar for spec in VARIANTS.values() if spec.cfar is not None}
    assert declared == reachable, (
        f"modes {sorted(declared - reachable)} are declared but no arm selects them"
    )


@pytest.mark.parametrize("variant", ALL_FRONT_END_ARMS)
def test_every_front_end_arm_diffs_from_the_stock_detector_in_one_place(variant):
    """Six arms, one insertion point: the comparison stays attributable at every scale."""
    spec = VARIANTS[variant]
    assert spec.cfar in RatioSpaceCFARFrontEnd.MODES
    assert spec.module_names == ["cfar"], (
        f"{variant} carries {spec.module_names}; a front-end arm must differ from the "
        "baseline in the first representation and nothing else"
    )
    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(spec), ch=3, nc=1, verbose=False)
    fronts = [m for m in model.modules() if isinstance(m, RatioSpaceCFARFrontEnd)]
    assert len(fronts) == 1, f"{variant}: expected exactly one front end, found {len(fronts)}"
    assert fronts[0].mode == spec.cfar


def test_the_front_end_is_registered_so_a_checkpoint_can_rebuild_it():
    """A checkpoint is only loadable if the builder knows the class by name."""
    assert CUSTOM_MODULES.get("RatioSpaceCFARFrontEnd") is RatioSpaceCFARFrontEnd


# ------------------------------------------------------------- the efficiency scope
@pytest.mark.parametrize("scale", ["n", "s"])
def test_the_parameter_overhead_is_scale_independent_and_inside_the_budget(scale):
    """The overhead is the front end's, so it cannot depend on the backbone's size.

    Measured on the generated YAML rather than the module: the claim is about the
    assembled model, and a front end that was inserted more than once would show up
    here even though the module's own parameter count looked right.
    """
    from saryolo.evaluation.efficiency import profile_yaml

    def profile(variant: str) -> dict:
        path = REPO_ROOT / "configs" / "models" / variant_filename(VARIANTS[variant])
        assert path.exists(), f"{variant}: missing generated YAML {path.name}"
        return profile_yaml(path, imgsz=320, nc=1)

    baseline = profile(f"baseline_{scale}")
    prototype = profile(f"cfar_{scale}")
    matched = profile(f"cfar_conv_{scale}")
    fixed = profile(f"cfar_fixed_{scale}")

    overhead = prototype["params"] - baseline["params"]
    # The same front end sits on both scales, so the *absolute* overhead must be the
    # same integer at n and s. A mismatch means something else changed with the scale.
    assert overhead == 217, f"scale {scale}: front end costs {overhead} parameters, expected 217"
    pct = overhead / baseline["params"] * 100
    assert 0.0 < pct < 0.5, f"scale {scale}: {pct:.4f} % overhead is outside the 0.5 % budget"

    assert matched["params"] == prototype["params"], (
        f"scale {scale}: matched-cost control is not parameter-identical to the prototype"
    )
    assert fixed["params"] == baseline["params"], (
        f"scale {scale}: the fixed-threshold control must add exactly zero parameters"
    )


# ---------------------------------------------------- the diagnostic's own honesty
def test_the_collapse_report_flags_a_constant_gain_and_clears_a_varied_one():
    """The diagnostic must fail the case the proposal names, and pass a live one."""
    constant = summarise_gain(np.zeros((3, 64), dtype=np.float64))
    assert constant["collapsed"] is True
    assert constant["active_fraction"] == 0.0
    assert constant["pooled_std"] == 0.0

    rng = np.random.default_rng(0)
    varied = summarise_gain(rng.uniform(-0.9, 0.9, size=(3, 64)))
    assert varied["collapsed"] is False
    assert varied["active_fraction"] > 0.5
    assert varied["within_image_std_mean"] > 0.1


def test_the_collapse_report_separates_a_per_scene_scalar_from_a_per_pixel_decision():
    """A gain that varies only *between* scenes is not the per-pixel path the design claims.

    Each row constant at a different value: the pooled std is large, so the naive
    collapsed flag is ``False``, but no pixel inside any scene differs from its
    neighbours. The report has to say so, or it would certify a per-scene scalar as a
    per-pixel decision.
    """
    per_scene = np.stack([np.full(64, v) for v in (-0.5, 0.0, 0.5)])
    report = summarise_gain(per_scene)
    assert report["collapsed"] is False
    assert report["within_image_std_mean"] == 0.0
    assert report["scene_scalar_dominated"] is True


def test_a_stock_detector_has_no_front_end_to_diagnose():
    """The diagnostic must refuse a model it does not apply to, not report zeros."""
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["baseline_s"]), ch=3, nc=1, verbose=False)
    assert find_front_end(model) is None


def test_the_brightness_shift_is_an_identity_at_gain_one_and_darkens_below_it():
    """The synthetic acquisition shift must be exactly the radiometric gain it claims."""
    from saryolo.evaluation.robustness import apply_corruption

    rng = np.random.default_rng(1)
    img = (rng.random((32, 32)) * 255).astype(np.uint8)
    same = apply_corruption(img, "brightness", 1.0, np.random.default_rng(0))
    assert np.array_equal(same, img), "gain 1.0 must be byte-exact, not a re-quantised copy"
    darker = apply_corruption(img, "brightness", 0.5, np.random.default_rng(0))
    assert darker.max() < img.max(), "gain 0.5 did not darken the image"
    assert darker.shape == img.shape


def test_the_anisotropic_shift_degrades_one_axis_only():
    """An anisotropy shift must change the x axis and leave the range axis alone.

    This is the difference between it and isotropic ``low_resolution``, which blurs both
    axes. Tested on the one image class that makes the distinction observable: a scene
    constant along x (identical columns) has no azimuthal content, so along-track-only
    resampling must return it unchanged, while the isotropic corruption must not.
    """
    from saryolo.evaluation.robustness import apply_corruption

    rng = np.random.default_rng(2)
    column = (rng.random((32, 1)) * 255).astype(np.uint8)
    img = np.repeat(column, 32, axis=1)  # constant along x, varies along y

    same = apply_corruption(img, "anisotropic", 1.0, np.random.default_rng(0))
    assert np.array_equal(same, img)
    kept = apply_corruption(img, "anisotropic", 0.25, np.random.default_rng(0))
    assert np.array_equal(kept, img), (
        "a scene with no azimuthal content changed under along-track resampling; the shift "
        "is touching the range axis as well"
    )
    isotropic = apply_corruption(img, "low_resolution", 0.25, np.random.default_rng(0))
    assert not np.array_equal(isotropic, img), (
        "the isotropic control did not blur a y-varying scene, so the comparison above "
        "would not be discriminating"
    )


def test_the_gain_report_refuses_an_empty_split():
    """An empty diagnostic must raise, not return a plausible-looking ``nan`` report."""
    with pytest.raises(ValueError, match="no gain values"):
        summarise_gain(np.zeros((0, 16), dtype=np.float64))
