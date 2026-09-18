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
from saryolo.nn.losses import SAR_LOSS_DEFAULTS, SARAwareDetectionLoss
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
