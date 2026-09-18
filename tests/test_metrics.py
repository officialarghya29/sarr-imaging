"""Unit tests for the COCO-style AP implementation.

These matter more than they look. In the smoke run every metric came out as
``0.0`` — which is the expected value for two epochs at 128px on synthetic data,
but it is *indistinguishable* from a metric function that is simply broken and
always returns zero. These tests establish, on hand-constructed inputs where the
answer is known analytically, that the metric responds correctly, including the
scale-wise routing the paper's central claim depends on.
"""

from __future__ import annotations

import pytest

from saryolo.evaluation.metrics import AREA_RANGES, Detection, compute_ap


def _det(image: str, cls: int, xyxy, score: float = 0.9) -> Detection:
    return Detection(image, cls, tuple(float(v) for v in xyxy), score=score)


def test_perfect_predictions_give_ap_one():
    """Ground truth exactly reproduced must score AP = 1.0 at every IoU threshold."""
    gts = [_det("a", 0, (10, 10, 50, 50)) for _ in [0]]
    dets = [_det("a", 0, (10, 10, 50, 50), score=0.99)]
    stats = compute_ap(dets, gts, nc=1, area_range=AREA_RANGES["all"])
    assert stats["AP"] == pytest.approx(1.0)
    assert stats["AP50"] == pytest.approx(1.0)
    assert stats["n_positives"] == 1.0


def test_no_predictions_give_ap_zero():
    gts = [_det("a", 0, (10, 10, 50, 50))]
    stats = compute_ap([], gts, nc=1, area_range=AREA_RANGES["all"])
    assert stats["AP"] == pytest.approx(0.0)


def test_no_ground_truth_gives_none_not_zero():
    """An area range with no ground truth is *not measurable*.

    Returning 0.0 here would be actively misleading: a dataset with no large
    objects would appear to score AP_large = 0, which a reader would interpret as
    the model failing on large objects.
    """
    gts = [_det("a", 0, (0, 0, 10, 10))]  # area 100 -> small
    stats = compute_ap([_det("a", 0, (0, 0, 10, 10))], gts, nc=1, area_range=AREA_RANGES["large"])
    assert stats["n_positives"] == 0.0
    assert stats["AP"] is None


def test_missed_detection_is_a_false_negative():
    gts = [_det("a", 0, (10, 10, 50, 50))]
    dets = [_det("a", 0, (200, 200, 240, 240), score=0.9)]
    stats = compute_ap(dets, gts, nc=1, area_range=AREA_RANGES["all"])
    assert stats["AP"] == pytest.approx(0.0)


def test_half_iou_does_not_count_as_ap75():
    """A box overlapping at ~0.5 IoU passes AP50 but must fail the higher thresholds."""
    gts = [_det("a", 0, (0, 0, 100, 100))]
    dets = [_det("a", 0, (0, 0, 100, 50), score=0.9)]  # IoU = 0.5 exactly
    stats = compute_ap(dets, gts, nc=1, area_range=AREA_RANGES["all"])
    assert stats["AP50"] == pytest.approx(1.0)
    assert stats["AP"] < 1.0


def test_scale_ranges_route_by_area():
    """Small/medium/large bins must partition ground truth by pixel area.

    The thresholds are 32^2 and 96^2. Binning *linear* sizes against 32/96 (an
    easy mistake) would put a 40x40 object in 'large' instead of 'medium'.
    """
    small = _det("a", 0, (0, 0, 20, 20))       # area 400
    medium = _det("b", 0, (0, 0, 50, 50))      # area 2500
    large = _det("c", 0, (0, 0, 150, 150))     # area 22500
    gts = [small, medium, large]
    dets = [
        _det("a", 0, (0, 0, 20, 20)),
        _det("b", 0, (0, 0, 50, 50)),
        _det("c", 0, (0, 0, 150, 150)),
    ]
    assert compute_ap(dets, gts, 1, AREA_RANGES["small"])["AP"] == pytest.approx(1.0)
    assert compute_ap(dets, gts, 1, AREA_RANGES["medium"])["AP"] == pytest.approx(1.0)
    assert compute_ap(dets, gts, 1, AREA_RANGES["large"])["AP"] == pytest.approx(1.0)
    assert compute_ap(dets, gts, 1, AREA_RANGES["small"])["n_positives"] == 1.0


def test_scale_range_boundary_is_exclusive_at_top():
    """A 32x32 object (area exactly 1024) belongs to 'medium', per the COCO convention."""
    exact = _det("a", 0, (0, 0, 32, 32))
    assert compute_ap([], [exact], 1, AREA_RANGES["small"])["n_positives"] == 0.0
    assert compute_ap([], [exact], 1, AREA_RANGES["medium"])["n_positives"] == 1.0


def test_confidence_ordering_affects_precision_curve():
    """A higher-scoring true positive must beat a lower-scoring false positive."""
    gts = [_det("a", 0, (0, 0, 100, 100))]
    good = [_det("a", 0, (0, 0, 100, 100), score=0.9), _det("a", 0, (500, 500, 520, 520), score=0.1)]
    bad = [_det("a", 0, (0, 0, 100, 100), score=0.1), _det("a", 0, (500, 500, 520, 520), score=0.9)]
    assert compute_ap(good, gts, 1, AREA_RANGES["all"])["AP"] > compute_ap(bad, gts, 1, AREA_RANGES["all"])["AP"]


def test_class_mismatch_is_not_a_true_positive():
    gts = [_det("a", 0, (0, 0, 100, 100))]
    dets = [_det("a", 1, (0, 0, 100, 100), score=0.9)]
    stats = compute_ap(dets, gts, nc=2, area_range=AREA_RANGES["all"])
    assert stats["AP"] == pytest.approx(0.0)
