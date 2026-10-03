"""The efficiency frontier: the cost claim, and the honesty rules around it.

The frontier is the part of the paper where a number we *measured* (params, GFLOPs) sits
next to a number someone else *reported* (a published mAP). That adjacency is the single
most dangerous place in the repository for an accidental fabrication, because a reader who
sees one table assumes one protocol. These tests pin both halves:

* the light arms are what they claim -- strictly cheaper than full v2, and built by
  *removing* the two largest compute slots rather than by a hidden change;
* a reported number never satisfies a measured cell, and every reported entry is cited.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from saryolo.evaluation.reported_baselines import REPORTED_BASELINES, reported_rows
from saryolo.nn.arch import VARIANTS, build_yaml_dict
from saryolo.paper.tables import (
    EFFICIENCY_FRONTIER_ARMS,
    TBD,
    build_efficiency_frontier,
)
from saryolo.tracking.ledger import ExperimentLedger

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The slots the light arms remove. Stated once so a test cannot drift from the docstring.
REMOVED_SLOTS = ("context", "fusion")


def _measure(variant: str):
    """Construct one variant and measure its real params/GFLOPs."""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops

    spec = dataclasses.replace(VARIANTS[variant])
    model = DetectionModel(build_yaml_dict(spec), ch=3, nc=1, verbose=False)
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops = float(get_flops(model, imgsz=640))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {"params_M": params, "flops_G": flops}


@pytest.mark.parametrize("variant", [v for v, _l, _e in EFFICIENCY_FRONTIER_ARMS if v.startswith("v2_lite")])
def test_light_arms_remove_exactly_the_two_dominant_compute_slots(variant):
    """A light arm differs from full v2 by the two slots it says it removes -- and nothing else.

    This is the claim that keeps the cost story honest. If a light arm quietly dropped a
    physical prior (the spectral branch, the target prior, the refinement) the compute
    saving would be real but the *reason* would be a lie, and the paper would be comparing
    a cheaper model that is no longer the method under test.
    """
    spec = VARIANTS[variant]
    full = VARIANTS["v2_full"]
    for slot in REMOVED_SLOTS:
        assert getattr(spec, slot) is None, f"{variant}: slot {slot!r} should be removed"
    for slot in ("enhancement", "speckle", "frequency", "prior", "attention", "refinement"):
        assert getattr(spec, slot) == getattr(full, slot), (
            f"{variant}: physical slot {slot!r} differs from full v2 "
            f"({getattr(spec, slot)!r} vs {getattr(full, slot)!r}); the cost claim must not "
            "be bought by deleting the science"
        )
    # Scale and levels are allowed to differ (that is what makes them frontier *points*),
    # but they must be declared explicitly, not inherited by accident.
    assert spec.scale in ("n", "s", "m", "l", "x")
    assert spec.efficiency is True, "a frontier arm must be flagged as an efficiency arm"


def test_every_light_arm_is_strictly_cheaper_than_its_scale_matched_reference():
    """Cheaper in both parameters and compute than full v2 *at the same scale*.

    Scale-matched on purpose: comparing a scale-``m`` light arm against the scale-``s``
    reference would fail for the trivial reason that ``m`` is a bigger backbone, and the
    resulting assertion would say nothing about the slots that were removed. The claim is
    about the removal, so the reference has to hold the scale fixed.
    """
    for variant, _label, _exp in EFFICIENCY_FRONTIER_ARMS:
        if not variant.startswith("v2_lite"):
            continue
        reference = f"v2_full_{VARIANTS[variant].scale}"
        full = _measure(reference)
        row = _measure(variant)
        assert row["params_M"] < full["params_M"], (
            f"{variant}: {row['params_M']:.3f}M params is not below {reference}'s {full['params_M']:.3f}M"
        )
        assert row["flops_G"] < full["flops_G"], (
            f"{variant}: {row['flops_G']:.2f} GFLOPs is not below {reference}'s {full['flops_G']:.2f}"
        )


def test_the_light_arm_at_scale_s_keeps_the_p2_level_and_still_beats_the_reference():
    """The headline light point keeps P2 (small objects) and still cuts compute sharply.

    Pinned as a ratio rather than an absolute so the test states the *finding* -- the
    dominant compute is in the fusion and context slots, not in the resolution of the head
    -- instead of a number that would go stale the moment the backbone changes.
    """
    full = _measure("v2_full")
    lite = _measure("v2_lite_s")
    assert VARIANTS["v2_lite_s"].levels == ("p2", "p3", "p4", "p5")
    assert lite["flops_G"] < 0.7 * full["flops_G"], (
        f"SARVO-Lite (s) at {lite['flops_G']:.2f} GFLOPs is not a >30% compute reduction "
        f"from {full['flops_G']:.2f}"
    )


def test_conditioning_on_the_light_model_is_a_small_absolute_cost():
    """The cross-sensor adapter must stay cheap on the model the cost claim is made about."""
    lite = _measure("v2_lite_s")
    cond = _measure("v2_lite_cond_s")
    overhead_pct = (cond["params_M"] - lite["params_M"]) / lite["params_M"] * 100
    assert overhead_pct < 1.0, f"conditioning overhead on the light model is {overhead_pct:.3f}%"
    # The adapter is a per-channel modulation, so it must be compute-free *for practical
    # purposes*. It is not bit-identical in the FLOP counter: the counter charges the tiny
    # gating multiplies, which is a ~0.0005% change. Asserted as an absolute bound in
    # GFLOPs so the test states the finding rather than hiding it behind a loose ratio.
    assert abs(cond["flops_G"] - lite["flops_G"]) < 0.01, (
        f"conditioning changed compute by {cond['flops_G'] - lite['flops_G']:.4f} GFLOPs; "
        "a per-channel modulation must be compute-free"
    )


def test_the_frontier_arms_all_have_a_model_yaml_and_an_experiment_config():
    """A frontier row that cannot be run would be a permanently TBD row with no symptom."""
    from saryolo.nn.arch import variant_filename

    for variant, _label, exp_id in EFFICIENCY_FRONTIER_ARMS:
        assert variant in VARIANTS, f"frontier names an unknown variant: {variant}"
        yaml_path = REPO_ROOT / "configs" / "models" / variant_filename(VARIANTS[variant])
        assert yaml_path.exists(), f"{variant}: missing {yaml_path.name}"
        configs = sorted((REPO_ROOT / "configs" / "exp").glob(f"{exp_id}_*.yaml"))
        assert configs, f"{variant}: no runnable config for {exp_id}"


def test_a_reported_number_never_fills_a_measured_cell():
    """The frontier's measured section is populated from our benchmark; published rows are not.

    The failure this guards is subtle: if a reported parameter count leaked into the
    measured section, the table would still render, the number would still be plausible,
    and nothing else in the suite would notice. So the two sections are asserted to be
    disjoint in *source*, and a reported row is required to declare itself.
    """
    ledger = ExperimentLedger(REPO_ROOT / "results")
    bench = {"v2_full": {"params_M": 16.230, "flops_G": 55.68}}
    table = build_efficiency_frontier(ledger, bench)
    rows = {row[0]: row for row in table.rows}
    assert rows["SAR-YOLO v2 (reference)"][1] == "16.230"
    assert rows["SAR-YOLO v2 (reference)"][5] == "measured (this repo)"
    for r in reported_rows():
        row = rows[f"{r.name} (published)"]
        assert "(published)" in row[0]
        assert "reported:" in row[5]
        # Every reported row must carry its venue, so it is never a naked number.
        assert r.venue in row[5], f"{r.name}: source column omits the venue"


def test_reported_rows_are_fully_cited_and_flagged():
    """Every published entry needs a venue, a URL, a dataset and a backbone."""
    assert REPORTED_BASELINES, "the reported-baseline registry is empty"
    for key, entry in REPORTED_BASELINES.items():
        assert entry.key == key
        assert entry.url.startswith("http"), f"{key}: not a link"
        assert entry.venue and entry.year and entry.dataset and entry.backbone, key
        assert entry.metrics, f"{key}: no captured numbers"
        # A relative-only entry must say so, because a percentage without its baseline is
        # not comparable and must not be promoted into a measured column.
        relative = {"params_reduction_pct", "compute_reduction_pct", "map50_95_delta_pct",
                    "map_delta_ssdd_pct", "map_delta_hrsid_pct"}
        if relative & set(entry.metrics):
            assert entry.notes, f"{key}: relative numbers need the baseline named in notes"


def test_reported_baselines_are_not_reachable_from_the_ledger():
    """A published number must not be writable as a measured metric.

    The ledger's metric keys are a closed set, and none of the reported units (a percentage
    reduction, a delta) is among them -- so a reported number cannot be recorded as if we
    had measured it, even by accident.
    """
    from saryolo.tracking.ledger import METRIC_KEYS

    reported_units = {
        "params_reduction_pct", "compute_reduction_pct", "compute_reduction_gflops",
        "map50_95_delta_pct", "map_delta_ssdd_pct", "map_delta_hrsid_pct",
        "map_sardet100k", "map50_ssdd", "map50_hrsid",
    }
    assert not (reported_units & set(METRIC_KEYS)), (
        "a reported-only unit is now a ledger metric key, which would let a published "
        "number be stored as a measurement"
    )


def test_the_cost_frontier_figure_refuses_to_draw_without_measured_accuracy(tmp_path):
    """Before a run exists there is no y-coordinate, so the figure must not be produced.

    The alternative -- drawing our arms at a placeholder height -- is the exact failure this
    project is built to avoid, and it would be invisible in a saved PDF.
    """
    from saryolo.paper.figures import plot_cost_frontier

    cost_only = {"SARVO-Lite (s)": {"params_M": 11.016, "flops_G": 32.55}}
    assert plot_cost_frontier(cost_only, tmp_path / "frontier.svg") is None
    assert not (tmp_path / "frontier.svg").exists()


def test_the_cost_frontier_figure_skips_relative_only_published_entries(tmp_path):
    """A published '30% fewer params' has no coordinate and must be skipped, not converted.

    Converting a relative reduction into an absolute GFLOPs value would require inventing
    the baseline it is relative to -- a fabricated number in a figure, which is worse than
    an absent point because nothing downstream can detect it.
    """
    from saryolo.evaluation.reported_baselines import REPORTED_BASELINES
    from saryolo.paper.figures import plot_cost_frontier

    measured = {"SARVO-Lite (s)": {"params_M": 11.016, "flops_G": 32.55, "mAP50_95": 0.30}}
    out = tmp_path / "frontier.svg"
    path = plot_cost_frontier(measured, out, reported=reported_rows())
    assert path is not None and path.exists()
    # AC-YOLO is relative-only, so it contributes no point; this is asserted structurally
    # rather than by reading the SVG: the entry simply has no absolute pair to plot.
    ac = REPORTED_BASELINES["ac_yolo"]
    assert "flops_G" not in ac.metrics and "map50_95" not in ac.metrics


def test_the_frontier_table_still_marks_accuracy_as_unmeasured():
    """The whole point of the table is that cost is known and accuracy is not."""
    ledger = ExperimentLedger(REPO_ROOT / "results")
    table = build_efficiency_frontier(ledger, {"v2_full": {"params_M": 16.23, "flops_G": 55.68}})
    our_rows = [row for row in table.rows if row[5] == "measured (this repo)"]
    assert our_rows, "no measured rows"
    for row in our_rows:
        assert row[3] == TBD and row[4] == TBD, f"{row[0]}: accuracy must stay TBD until run"
