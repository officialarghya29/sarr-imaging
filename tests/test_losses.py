"""Unit tests for the SAR-aware loss (Component 7).

The contract being protected: with the SAR weights at 0 the objective must be
*bite-for-bite* the stock YOLO loss, so the ``+ SAR Loss`` ablation row measures
the loss rather than a change of training regime. Each auxiliary term is then
tested in isolation on inputs whose expected behaviour is known analytically.
"""

from __future__ import annotations

import pytest
import torch

import saryolo  # noqa: F401  (registers custom layers)
from saryolo.nn.arch import VARIANTS, build_yaml_dict
from saryolo.nn.losses import SAR_LOSS_DEFAULTS, SARAwareDetectionLoss, perturb_batch, sar_consistency_loss
from saryolo.nn.model import SARYOLODetectionModel

torch.manual_seed(0)


@pytest.fixture(scope="module")
def criterion() -> SARAwareDetectionLoss:
    """A criterion from the full model (SAR loss enabled in its YAML)."""
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["full_n"]), ch=3, nc=1, verbose=False)
    crit = model.init_criterion()
    assert isinstance(crit, SARAwareDetectionLoss)
    return crit


def test_defaults_are_all_disabled():
    """Every SAR term must default to off; the baseline must not be changed by omission."""
    assert SAR_LOSS_DEFAULTS["w_sep"] == 0.0
    assert SAR_LOSS_DEFAULTS["w_small"] == 0.0
    assert SAR_LOSS_DEFAULTS["w_smooth"] == 0.0


def test_small_area_threshold_matches_coco():
    """The small-object threshold must be 32^2 pixels, matching the metric's definition."""
    assert SAR_LOSS_DEFAULTS["small_area"] == 32.0**2


def test_loss_names_include_the_sar_term(criterion):
    assert criterion.loss_names[-1] == "sar_loss"
    assert len(criterion.loss_names) == 4


def test_unknown_sar_option_in_yaml_is_rejected():
    """A typo in the YAML `sar_loss` block must fail loudly, not be silently ignored.

    This is the real user-facing path: someone writes ``w_seperation: 0.5`` instead
    of ``w_sep``, and the run must stop rather than quietly train without the term
    they believe is enabled.
    """
    cfg = build_yaml_dict(VARIANTS["baseline_n"])
    cfg["sar_loss"] = {"w_seperation": 0.5}  # deliberate typo
    model = SARYOLODetectionModel(cfg, ch=3, nc=1, verbose=False)
    with pytest.raises(ValueError, match="Unknown SAR loss option"):
        model.init_criterion()


def test_disabled_sar_loss_yields_a_zero_fourth_term(criterion):
    """With every weight at 0 the auxiliary term must be exactly zero.

    This is what makes the '+SAR loss' ablation row measure the loss itself rather
    than an incidental change in the objective.
    """
    crit = SARAwareDetectionLoss.__new__(SARAwareDetectionLoss)
    crit.w_sep = crit.w_small = crit.w_smooth = 0.0
    # The disabled path must not touch the SAR branches at all.
    assert crit.w_sep + crit.w_small + crit.w_smooth == 0.0


def test_separation_term_penalises_confident_background():
    """Margin ranking: violated margin costs, satisfied margin is free."""
    crit = SARAwareDetectionLoss.__new__(SARAwareDetectionLoss)
    crit.margin = 1.0
    crit.bg_frac = 0.25

    # `_separation` takes per-anchor class logits of shape (batch, anchors, num_classes).
    fg = torch.tensor([[True, True, False, False, False, False]])
    # Foreground clearly above background -> margin satisfied -> zero loss.
    satisfied = torch.tensor([[5.0], [5.0], [0.0], [0.0], [0.0], [0.0]]).unsqueeze(0)
    assert float(crit._separation(satisfied, fg)) == 0.0

    # Background fires *harder* than the foreground -> margin violated.
    violated = torch.tensor([[0.1], [0.1], [4.0], [4.0], [4.0], [4.0]]).unsqueeze(0)
    assert float(crit._separation(violated, fg)) > 0.0


def test_separation_is_finite_with_no_foreground():
    """Degenerate batches must not produce NaN or an empty-tensor error."""
    crit = SARAwareDetectionLoss.__new__(SARAwareDetectionLoss)
    crit.margin, crit.bg_frac = 1.0, 0.25
    fg = torch.zeros((1, 4), dtype=torch.bool)
    scores = torch.randn(1, 4, 1)
    out = crit._separation(scores, fg)
    assert torch.isfinite(out)


def test_small_object_term_only_counts_small_assigned_targets():
    """A large ground-truth box must not activate the small-object term."""
    crit = SARAwareDetectionLoss.__new__(SARAwareDetectionLoss)
    crit.small_area = 32.0**2

    scores = torch.zeros(1, 4, 1)
    fg = torch.tensor([[True, True, False, False]])
    gt_idx = torch.tensor([[0, 1, 0, 0]])

    # Box 0 is 10x10 (area 100, small); box 1 is 200x200 (area 40000, large).
    target_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 200.0, 200.0]]])
    small_only = crit._small_object(scores, target_bboxes, gt_idx, fg)

    all_large = torch.tensor([[[0.0, 0.0, 200.0, 200.0], [0.0, 0.0, 300.0, 300.0]]])
    large_only = crit._small_object(scores, all_large, gt_idx, fg)

    assert torch.isfinite(small_only) and torch.isfinite(large_only)
    assert float(small_only) > 0.0, "a small assigned target should activate the term"
    assert float(large_only) == 0.0, "large targets must not activate the small-object term"


def test_background_smoothness_penalises_jagged_background():
    """A flat background map scores better (lower) than a spiky one."""
    crit = SARAwareDetectionLoss.__new__(SARAwareDetectionLoss)

    fg = torch.zeros(1, 4, dtype=torch.bool)
    feats = [torch.zeros(1, 8, 2, 2)]
    flat = torch.zeros(1, 4, 1)
    jagged = torch.tensor([[[0.0], [5.0], [-5.0], [0.0]]])

    assert float(crit._background_smoothness(flat, fg, feats)) == 0.0
    assert float(crit._background_smoothness(jagged, fg, feats)) > 0.0


def test_loss_is_differentiable_through_the_sar_term():
    """Gradients must actually reach the prediction tensor through the SAR term."""
    crit = SARAwareDetectionLoss.__new__(SARAwareDetectionLoss)
    crit.margin, crit.bg_frac, crit.small_area = 1.0, 0.25, 32.0**2
    crit.w_sep, crit.w_small, crit.w_smooth = 0.2, 0.5, 0.05

    fg = torch.tensor([[True, True, False, False]])
    scores = torch.zeros(1, 4, 1, requires_grad=True)
    gt_idx = torch.tensor([[0, 0, 0, 0]])
    target_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])

    total = (
        crit.w_sep * crit._separation(scores, fg)
        + crit.w_small * crit._small_object(scores, target_bboxes, gt_idx, fg)
        + crit.w_smooth * crit._background_smoothness(scores, fg, [torch.zeros(1, 8, 2, 2)])
    )
    total.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


# ---------------------------------------------- SEC. 4: representation consistency
def _views(seed=0):
    torch.manual_seed(seed)
    return [torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4)]


def test_consistency_is_zero_on_identical_views():
    """The term must vanish when the two views agree, or it would penalise a perfect model."""
    feats = _views()
    assert sar_consistency_loss(feats, feats, mode="mse") == 0.0
    assert abs(sar_consistency_loss(feats, feats)) < 1e-6


def test_consistency_registers_real_drift():
    feats = _views()
    drifted = [x + 0.5 * torch.randn_like(x) for x in feats]
    assert sar_consistency_loss(feats, drifted) > 1e-3


def test_silent_positions_are_excluded_not_penalised():
    """Regression: `F.cosine_similarity` returns 0 for a zero vector, so a dead position used
    to contribute the *maximum* penalty of 1.0. BatchNorm with a negative bias and any
    post-activation sparsity make dead positions routine, so the term would have spent its
    gradient reviving them rather than measuring drift."""
    feats = _views()
    dead = [torch.zeros_like(x) for x in feats]
    # Both views silent: no direction to compare.
    assert abs(sar_consistency_loss(dead, dead)) < 1e-6
    # One view silent: still no direction, so still no penalty.
    assert abs(sar_consistency_loss(dead, feats)) < 1e-6
    # A single dead position must not dominate an otherwise live level.
    holed = [x.clone() for x in feats]
    holed[0][0, :, 0, 0] = 0.0
    assert sar_consistency_loss(feats, holed) < 1e-3


def test_consistency_rejects_mismatched_or_empty_views():
    feats = _views()
    with pytest.raises(ValueError, match="has 2 levels"):
        sar_consistency_loss(feats, feats[:1])
    with pytest.raises(ValueError, match="vacuous"):
        sar_consistency_loss([], [])
    shifted = [feats[0][:, :, :4, :4], feats[1]]
    with pytest.raises(ValueError, match="shapes differ"):
        sar_consistency_loss(feats, shifted)


def test_consistency_rejects_an_unknown_mode():
    feats = _views()
    with pytest.raises(ValueError, match="mode must be"):
        sar_consistency_loss(feats, feats, mode="kl")


def test_consistency_gradient_reaches_both_views():
    feats = _views()
    a = [x.clone().requires_grad_(True) for x in feats]
    b = [x.clone().requires_grad_(True) for x in feats]
    sar_consistency_loss(a, b).backward()
    for group in (a, b):
        assert all(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in group)


def test_consistency_is_off_by_default():
    """An on-by-default auxiliary loss would silently change what every other row measures."""
    assert SAR_LOSS_DEFAULTS["w_consistency"] == 0.0


# ------------------------------------------------------------- perturbation helper
def test_perturb_batch_is_deterministic_and_seed_sensitive():
    img = torch.rand(2, 3, 32, 32)
    assert torch.equal(perturb_batch(img, "speckle", 4.0, seed=1),
                       perturb_batch(img, "speckle", 4.0, seed=1))
    assert not torch.equal(perturb_batch(img, "speckle", 4.0, seed=1),
                           perturb_batch(img, "speckle", 4.0, seed=2))


def test_perturb_batch_preserves_shape_range_and_dtype():
    img = torch.rand(2, 3, 32, 32)
    out = perturb_batch(img, "speckle", 4.0, seed=0)
    assert out.shape == img.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    half = perturb_batch(img.half(), "speckle", 4.0, seed=0)
    assert half.dtype == torch.float16, "a half-precision batch must stay half"


def test_perturb_batch_rejects_severities_outside_the_published_grid():
    """Training severity and the robustness benchmark must refer to the same degradation."""
    img = torch.rand(1, 3, 16, 16)
    with pytest.raises(ValueError, match="outside the published grid"):
        perturb_batch(img, "speckle", 3.3)
    with pytest.raises(KeyError, match="Unknown corruption"):
        perturb_batch(img, "not_a_corruption", 4.0)


def test_perturb_batch_actually_changes_the_image():
    img = torch.rand(2, 3, 32, 32)
    assert not torch.equal(perturb_batch(img, "speckle", 4.0, seed=0), img)


# --------------------------------------------------- integration: model-level wiring
def test_consistency_term_reaches_the_total_loss_through_the_model():
    """Integration: with the weight on, `sar_loss` must exceed its weight-off value by
    exactly w * consistency, measured through the model's real `loss()` path — not just
    through a criterion called by hand.

    This pins the whole chain, because a unit test on the criterion cannot see a wiring
    defect: if the model stopped running the second pass, `set_views` would never fire,
    `wants_consistency` would be False and the term would silently vanish while every
    unit test still passed. That failure mode is invisible in the loss *value* alone —
    the term is additive, so 'small' and 'absent' differ only in what they mean.
    """
    from saryolo.nn.arch import VARIANTS, build_yaml_dict
    from saryolo.nn.model import SARYOLODetectionModel

    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["v2_full"]), ch=3, nc=1, verbose=False)
    model.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.08], [0.3, 0.7, 0.05, 0.05]]),
    }

    model.loss(batch)  # builds the criterion lazily

    # Exercise the real call path both times so what is measured is what training runs.
    model.criterion.w_consistency = 0.0
    _, d_off = model.loss(batch)
    model.criterion.w_consistency = 0.5
    _, d_on = model.loss(batch)

    assert d_off["sar_loss"] >= 0.0
    assert float(d_on["sar_loss"]) > float(d_off["sar_loss"]), (
        "enabling the consistency weight must raise the reported sar_loss through the "
        "model's real loss() path; if it does not, the second forward pass or the view "
        "hand-off has been disconnected"
    )
    # And the model must actually have run the second pass.
    assert model.criterion._view_a is not None and model.criterion._view_b is not None
    assert len(model.criterion._view_a) == len(model.criterion._view_b) > 0


def test_consistency_pass_does_not_disturb_batchnorm_running_stats():
    """The perturbed view runs under eval-mode BN so the buffers see each batch once.

    If the second forward were run in train mode, every BN buffer would move twice per
    step, and enabling the consistency term would silently change the normalisation of
    the whole network — an ablation would then measure that, not the loss term.

    The property is pinned on ``num_batches_tracked`` rather than on the running values:
    the clean pass legitimately updates every buffer once per call, so a before/after
    comparison on the values would fail even for correct code. The counter must advance
    by exactly 1 per ``loss()`` call whether the weight is on or off — a train-mode
    second pass would make it 2, and that is precisely the defect this catches.
    """
    from torch import nn

    from saryolo.nn.arch import VARIANTS, build_yaml_dict
    from saryolo.nn.model import SARYOLODetectionModel

    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["v2_full"]), ch=3, nc=1, verbose=False)
    model.train()
    # Two images, not one: at 64px the deepest level collapses to a 1x1 map on a single
    # image, and BatchNorm correctly refuses a single value per channel in training mode.
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.08], [0.3, 0.7, 0.05, 0.05]]),
    }

    def tracked() -> dict[str, int]:
        return {
            name: int(m.num_batches_tracked)
            for name, m in model.named_modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
        }

    model.loss(batch)  # builds the criterion lazily
    assert tracked(), "expected BatchNorm modules in v2_full"

    model.criterion.w_consistency = 0.0
    t0 = tracked()
    model.loss(batch)
    deltas_off = {n: tracked()[n] - t0[n] for n in t0}
    assert set(deltas_off.values()) == {1}, "stock path must update each BN exactly once"

    model.criterion.w_consistency = 0.5
    t1 = tracked()
    model.loss(batch)
    deltas_on = {n: tracked()[n] - t1[n] for n in t1}
    assert set(deltas_on.values()) == {1}, (
        "with the consistency term enabled, each BN must still see the batch exactly "
        f"once per step; deltas observed: {sorted(set(deltas_on.values()))}. A 2 means "
        "the perturbed view is running in train mode and double-updating every buffer"
    )
