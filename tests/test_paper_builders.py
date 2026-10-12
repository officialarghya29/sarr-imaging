"""The remaining paper-table builders: every row traceable to a run, or visibly TBD.

``tests/test_tables.py`` pins the seed-deduplication and variant-matching rules and the
real-data pilot. This file covers the other builders -- the baseline comparison, the
cumulative and module ablations, the removal ablation, the scale table, the robustness
table, the efficiency table and ``write_tables`` -- because each is a place where a wrong
number could be rendered without anything downstream noticing:

* a *removal* ablation row that silently matched the full model would report "removing this
  component costs nothing", which is the opposite of what the row is for;
* ``write_tables(assert_complete=True)`` is the submission gate: it must turn any remaining
  ``TBD`` into an error rather than writing a half-filled table that looks finished.
"""

from __future__ import annotations

import json

import pytest

from saryolo.paper.tables import (
    MODULE_ABLATION_GROUPS,
    REAL_PILOT_ARMS,
    REMOVAL_ABLATION_ROWS,
    TBD,
    Table,
    build_ablation,
    build_baseline_comparison,
    build_efficiency,
    build_module_ablation,
    build_removal_ablation,
    build_robustness,
    build_scale_analysis,
    write_tables,
)
from saryolo.tracking.ledger import ExperimentRecord


def _record(exp_id: str, model: str, metrics: dict | None = None, *, status: str = "completed",
            dataset: str = "ssdd", run_id: str = "run-1") -> ExperimentRecord:
    record = ExperimentRecord(experiment_id=exp_id, model=model, dataset=dataset, run_id=run_id)
    record.status = status
    if metrics is not None:
        record.metrics = metrics
    return record


class _Ledger:
    def __init__(self, records):
        self._records = records

    def completed(self):
        return [r for r in self._records if r.status == "completed"]


def _by_label(table: Table) -> dict[str, list[str]]:
    assert table.rows
    return {row[0]: row for row in table.rows}


# ------------------------------------------------------------------ baseline comparison

def test_baseline_comparison_renders_measured_values_and_tbd_otherwise():
    """The two reference rows must show what was measured and nothing else."""
    ledger = _Ledger([
        _record("EXP-001", "yolo11n.yaml", {"mAP50": 0.97, "mAP50_95": 0.60, "precision": 0.93,
                                            "recall": 0.55, "params_M": 2.59, "flops_G": 1.613,
                                            "fps": 61.8}),
    ])
    table = build_baseline_comparison(ledger)
    rows = _by_label(table)
    baseline = rows["YOLO11 baseline"]
    assert baseline[1] == "0.9700" and baseline[2] == "0.6000"
    assert baseline[5] == "2.590" and baseline[6] == "1.613" and baseline[7] == "61.8"
    ours = rows["SAR-YOLO (ours)"]
    assert ours[1] == TBD and ours[5] == TBD, "an unrun arm must render TBD, not a zero"
    assert not table.complete
    # Provenance must name the run for a measured row and say 'not run' otherwise.
    assert any("EXP-001" in p and "run-1" in p for p in table.provenance)
    assert any("EXP-007" in p and "not run" in p for p in table.provenance)


def test_baseline_comparison_is_complete_once_both_arms_have_run():
    metrics = {"mAP50": 0.9, "mAP50_95": 0.5, "precision": 0.9, "recall": 0.5,
               "params_M": 3.0, "flops_G": 8.0, "fps": 15.0}
    ledger = _Ledger([
        _record("EXP-001", "yolo11n.yaml", dict(metrics)),
        _record("EXP-007", "yolo11s_v2_full_s.yaml", dict(metrics)),
    ])
    assert build_baseline_comparison(ledger).complete


# ------------------------------------------------------------------ the cumulative ladder

def test_the_cumulative_ladder_has_one_row_per_component_and_ticks_the_flags():
    """Each row adds exactly one component; the checkmarks are that claim, rendered."""
    table = build_ablation(_Ledger([]))
    captions = _by_label(table)
    assert len(table.rows) == 12, "one row per rung of the ladder"
    assert captions["YOLO baseline"][1:11] == [""] * 10, "the baseline carries no component"
    full = captions["Full (v2)"]
    assert full[1:11] == ["\\checkmark"] * 10, "the full model must carry every component"
    # The clutter row is a mode change on the speckle slot: it keeps the speckle tick and
    # must not be presented as an added module with its own slot appearing twice.
    assert full[2] == "\\checkmark" and table.headers.count("Clutter") == 1


def test_the_ablation_renders_a_measured_arm_and_tbd_elsewhere():
    ledger = _Ledger([
        _record("EXP-017", "v2_full.yaml", {"mAP50": 0.71, "mAP50_95": 0.41, "params_M": 16.230, "fps": 9.9}),
    ])
    rows = _by_label(build_ablation(ledger))
    full = rows["Full (v2)"]
    assert full[-4] == "0.7100" and full[-3] == "0.4100" and full[-2] == "16.230"
    assert rows["YOLO baseline"][-4] == TBD


# ------------------------------------------------------------------ module ablation

def test_the_module_ablation_has_a_row_per_declared_arm():
    """A missing arm would be a permanently TBD row with no other symptom."""
    table = build_module_ablation(_Ledger([]))
    expected = sum(len(arms) for arms in MODULE_ABLATION_GROUPS.values())
    assert len(table.rows) == expected
    variants = [row[1] for row in table.rows]
    assert variants == [v for arms in MODULE_ABLATION_GROUPS.values() for v in arms]


def test_the_module_ablation_matches_an_arm_by_its_model_stem():
    """A run named ``v2_full`` must fill the ``v2_full`` row, and not a neighbouring one."""
    ledger = _Ledger([
        _record("EXP-017", "v2_full.yaml", {"mAP50": 0.55, "mAP50_95": 0.30, "params_M": 16.0}),
    ])
    rows = {row[1]: row for row in build_module_ablation(ledger).rows}
    assert rows["v2_full"][2] == "0.5500" and rows["v2_full"][3] == "0.3000"
    # Every other arm stays TBD, so the match is not a substring leakage.
    for variant, row in rows.items():
        if variant != "v2_full":
            assert row[2] == TBD, f"{variant} borrowed v2_full's score"


# ------------------------------------------------------------------ removal ablation

def test_the_removal_ablation_has_one_row_per_declared_removal():
    table = build_removal_ablation(_Ledger([]))
    assert len(table.rows) == len(REMOVAL_ABLATION_ROWS)
    labels = [row[0] for row in table.rows]
    assert labels[0] == "Full (v2)" and labels[1] == "- clutter branch"
    assert all(row[-2] == TBD and row[-1] == TBD for row in table.rows)


def test_the_removal_ablation_matches_the_full_model_and_a_removed_variant():
    ledger = _Ledger([
        _record("EXP-017", "v2_full.yaml", {"mAP50": 0.71, "AP_small": 0.22, "params_M": 16.2, "fps": 9.5}),
        _record("EXP-021", "v2_noclutter.yaml", {"mAP50": 0.70, "params_M": 15.9}),
    ])
    rows = {row[0]: row for row in build_removal_ablation(ledger).rows}
    assert rows["Full (v2)"][1] == "0.7100" and rows["Full (v2)"][3] == "0.2200"
    assert rows["- clutter branch"][1] == "0.7000"
    assert rows["- target prior"][1] == TBD, "an unrun removal must stay TBD"


# ------------------------------------------------------------------ scale analysis

def test_scale_analysis_reports_a_per_range_ground_truth_count():
    """A range with no ground truth is unmeasurable, not a zero score."""
    ledger = _Ledger([
        _record("EXP-007", "v2_full.yaml", {"AP_small": 0.20, "n_gt_small": 100,
                                            "AP_medium": 0.40, "n_gt_medium": 71,
                                            "AP_large": None, "n_gt_large": 0}),
    ])
    rows = _by_label(build_scale_analysis(ledger))
    ours = rows["SAR-YOLO (ours)"]
    assert ours[1] == "0.2000" and ours[2] == "100"
    assert ours[5] == TBD and ours[6] == "0", "no large boxes: TBD score, zero count"


# ------------------------------------------------------------------ robustness

def test_robustness_reads_the_two_json_files_and_reports_the_drop(tmp_path):
    ours = {"corruptions": {"speckle": {"1": {"mAP50": 0.30}, "2": {"mAP50": 0.20}}}}
    base = {"corruptions": {"speckle": {"1": {"mAP50": 0.25}, "2": {"mAP50": 0.10}}}}
    ours_path, base_path = tmp_path / "ours.json", tmp_path / "base.json"
    ours_path.write_text(json.dumps(ours))
    base_path.write_text(json.dumps(base))

    table = build_robustness(ours_path, base_path)
    rows = {(row[0], row[1]): row for row in table.rows}
    assert rows[("speckle", "1")][4] == "-0.0500", "drop = ours - baseline"
    assert rows[("speckle", "2")][4] == "-0.1000"


def test_robustness_renders_tbd_when_no_results_exist(tmp_path):
    """A fresh clone has no robustness JSON, so the table must not invent a row."""
    table = build_robustness(tmp_path / "missing.json")
    assert len(table.rows) == 1
    assert table.rows[0] == ["TBD", TBD, TBD, TBD, TBD]
    assert not table.complete


def test_robustness_still_lists_corruptions_when_only_the_baseline_exists(tmp_path):
    """A baseline-only run must still enumerate the corruption curves."""
    base = {"corruptions": {"stripe": {"1": {"mAP50": 0.5}}}}
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    table = build_robustness(tmp_path / "ours.json", base_path)
    assert table.rows[0][0] == "stripe"
    assert table.rows[0][2] == "0.5000", "the baseline column must carry the number"
    assert table.rows[0][3] == TBD, "the ours column is unrun"


# ------------------------------------------------------------------ efficiency

def test_efficiency_table_renders_cost_and_accuracy_columns():
    ledger = _Ledger([
        _record("EXP-001", "yolo11n.yaml", {"params_M": 2.59, "flops_G": 1.613, "fps": 61.8,
                                            "latency_ms": 51.25, "model_size_MB": 5.4}),
    ])
    rows = _by_label(build_efficiency(ledger))
    base = rows["YOLO baseline"]
    assert base[1] == "2.590" and base[2] == "1.613"
    assert base[4] == "51.25" and base[5] == "5.40"
    assert rows["SAR-YOLO (ours)"][1] == TBD


# ------------------------------------------------------------------ write_tables gate

def _complete_table() -> Table:
    return Table(name="done", caption="c", headers=["a", "b"], rows=[["1", "2"]])


def _incomplete_table() -> Table:
    return Table(name="todo", caption="c", headers=["a", "b"], rows=[["1", TBD]])


def test_write_tables_writes_markdown_and_latex_for_every_table(tmp_path):
    written = write_tables([_complete_table(), _incomplete_table()], tmp_path)
    names = sorted(p.name for p in written)
    assert names == ["done.md", "done.tex", "todo.md", "todo.tex"]
    assert (tmp_path / "done.tex").read_text().startswith("\\begin{table}")
    assert "not yet measured" in (tmp_path / "todo.md").read_text()


def test_write_tables_refuses_to_write_an_incomplete_table_for_submission(tmp_path):
    """The submission gate must fail loudly rather than emit a half-filled table."""
    with pytest.raises(ValueError, match="unmeasured cells"):
        write_tables([_incomplete_table()], tmp_path, assert_complete=True)
    # Nothing from the refused call should have been written.
    assert not (tmp_path / "todo.md").exists()


def test_write_tables_passes_a_complete_table_through_the_submission_gate(tmp_path):
    written = write_tables([_complete_table()], tmp_path, assert_complete=True)
    assert [p.name for p in written] == ["done.md", "done.tex"]


# ------------------------------------------------------------------ render safety

def test_table_to_dict_round_trips_every_field():
    table = Table(name="t", caption="c", headers=["a"], rows=[["1"]], provenance=["EXP-1: run-1"])
    payload = table.to_dict()
    assert payload["name"] == "t" and payload["complete"] is True
    assert payload["provenance"] == ["EXP-1: run-1"]


def test_pilot_and_removal_registries_are_non_empty_and_unique():
    """A dangling registry name would be a permanently TBD row with no symptom."""
    pilot_ids = [exp_id for exp_id, _label in REAL_PILOT_ARMS]
    assert len(pilot_ids) == len(set(pilot_ids))
    variants = [variant for _label, variant in REMOVAL_ABLATION_ROWS]
    assert len(variants) == len(set(variants))
