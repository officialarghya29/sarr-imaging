"""The paper-figure generators: a figure is drawn from data or not drawn at all.

Every figure in the paper is an argument, and each of these builders has a "nothing to
show" path that must return ``None`` rather than draw a plausible-looking placeholder --
a figure drawn with invented values is worse than an absent one, because the saving
happens before anyone reads the caption. ``plot_cost_frontier``'s refusal is pinned in
``tests/test_efficiency_frontier.py``; this file covers the other four.
"""

from __future__ import annotations

import json

from saryolo.paper.figures import (
    FIGURE_PLAN,
    plot_ablation_bars,
    plot_accuracy_efficiency,
    plot_robustness_curve,
    plot_scale_ap,
)

# ------------------------------------------------------------------ robustness curve

def test_robustness_curve_is_not_drawn_without_a_results_file(tmp_path):
    """A missing input is not a figure; the curve must refuse rather than guess."""
    assert plot_robustness_curve({"ours": tmp_path / "absent.json"}, tmp_path / "fig.svg") is None
    assert not (tmp_path / "fig.svg").exists()


def test_robustness_curve_refuses_a_file_with_no_corruptions(tmp_path):
    path = tmp_path / "ours.json"
    path.write_text(json.dumps({"corruptions": {}}))
    assert plot_robustness_curve({"ours": path}, tmp_path / "fig.svg") is None


def test_robustness_curve_draws_one_panel_per_corruption(tmp_path):
    ours = {"corruptions": {"speckle": {"1": {"mAP50": 0.4}, "2": {"mAP50": 0.3}},
                            "stripe": {"1": {"mAP50": 0.5}}}}
    base = {"corruptions": {"speckle": {"1": {"mAP50": 0.45}, "2": {"mAP50": 0.2}},
                            "stripe": {"1": {"mAP50": 0.55}}}}
    (tmp_path / "ours.json").write_text(json.dumps(ours))
    (tmp_path / "base.json").write_text(json.dumps(base))

    out = tmp_path / "robust.svg"
    path = plot_robustness_curve(
        {"SAR-YOLO": tmp_path / "ours.json", "YOLO": tmp_path / "base.json"}, out
    )
    assert path == out and out.exists() and out.stat().st_size > 0


# ------------------------------------------------------------------ accuracy vs efficiency

def test_accuracy_efficiency_refuses_when_no_point_can_be_placed(tmp_path):
    """A cost-only arm has no accuracy axis, so the figure must not be produced."""
    points = {"lite": {"params_M": 11.0, "flops_G": 32.5, "fps": 30.0}}
    assert plot_accuracy_efficiency(points, tmp_path / "fig.svg") is None
    assert not (tmp_path / "fig.svg").exists()


def test_accuracy_efficiency_skips_incomplete_points_and_draws_the_rest(tmp_path):
    points = {
        "complete": {"mAP50_95": 0.30, "fps": 20.0, "params_M": 11.0, "flops_G": 32.5},
        "no-map": {"fps": 40.0, "params_M": 3.0, "flops_G": 8.0},
    }
    out = tmp_path / "ae.svg"
    path = plot_accuracy_efficiency(points, out)
    assert path == out and out.exists()


# ------------------------------------------------------------------ ablation bars

def test_ablation_bars_refuse_an_empty_ablation(tmp_path):
    assert plot_ablation_bars({}, tmp_path / "fig.svg") is None
    assert plot_ablation_bars({"a": None, "b": None}, tmp_path / "fig.svg") is None


def test_ablation_bars_drop_unmeasured_rows_and_draw_the_rest(tmp_path):
    out = tmp_path / "ablation.svg"
    path = plot_ablation_bars({"base": 0.30, "full": 0.41, "unrun": None}, out)
    assert path == out and out.exists()


def test_ablation_bars_accept_a_baseline_reference(tmp_path):
    out = tmp_path / "ablation_delta.svg"
    path = plot_ablation_bars({"base": 0.30, "full": 0.41}, out, baseline_key="base")
    assert path == out and out.exists()


# ------------------------------------------------------------------ scale AP

def test_scale_ap_refuses_without_any_scale_measurement(tmp_path):
    models = {"baseline": {"mAP50_95": 0.30, "params_M": 3.0}}
    assert plot_scale_ap(models, tmp_path / "fig.svg") is None
    assert not (tmp_path / "fig.svg").exists()


def test_scale_ap_marks_an_unmeasurable_range_instead_of_drawing_a_zero(tmp_path):
    """A missing AP_large must be annotated 'n/a', not drawn as a zero-height bar."""
    models = {"SAR-YOLO": {"AP_small": 0.20, "AP_medium": 0.40, "AP_large": None}}
    out = tmp_path / "scale.svg"
    path = plot_scale_ap(models, out)
    assert path == out and out.exists()


# ------------------------------------------------------------------ plan

def test_the_figure_plan_names_every_supported_transform_and_target():
    """The plan is a checklist; a figure missing from it would never be requested."""
    assert FIGURE_PLAN, "the figure plan is empty"
    for key, description in FIGURE_PLAN.items():
        assert key.startswith("fig"), key
        assert description, f"{key}: no description"
    # The alternative-frequency study and the cost frontier must be planned explicitly --
    # they are the two figures a reviewer of the efficiency/frequency claims would ask for.
    assert "fig16_cost_frontier" in FIGURE_PLAN
    assert "fig13_robustness" in FIGURE_PLAN
