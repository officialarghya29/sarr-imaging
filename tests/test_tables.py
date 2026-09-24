"""Tests for the paper-table module.

A paper table is the last place a silent wrong number can hide: the ledger can be
appended to by any crashed run, so the aggregation rules here (one row per seed,
variant matching at word boundaries, TBD for anything unmeasured) are pinned by
test rather than by convention.
"""

from __future__ import annotations

import pytest

from saryolo.paper.tables import TBD, Table, build_multi_seed
from saryolo.tracking.ledger import ExperimentRecord


def _record(exp_id: str, seed: int, map5095: float | None, *, model: str = "yolo11s_full_s") -> ExperimentRecord:
    record = ExperimentRecord(experiment_id=exp_id, model=model, dataset="ssdd", train_seed=seed)
    if map5095 is not None:
        record.status = "completed"
        record.metrics = {"mAP50_95": map5095}
    return record


class _Ledger:
    """Minimal ledger stand-in over a fixed record list (append order)."""

    def __init__(self, records: list[ExperimentRecord]):
        self._records = records

    def completed(self):
        return [r for r in self._records if r.status == "completed"]


def _rows(table: Table) -> list[list[str]]:
    assert table.rows, "expected at least one row"
    return table.rows


# --------------------------------------------------------------------------- TBD


def test_no_measured_seeds_renders_tbd_not_zero():
    """An unrun multi-seed table must show TBD everywhere, never a fabricated 0.0."""
    table = build_multi_seed(_Ledger([_record("EXP-012", 0, None)]))
    (row,) = _rows(table)
    assert row[1] == TBD and row[2] == TBD and row[3] == TBD and row[4] == TBD
    assert not table.complete


def test_table_incompleteness_is_detectable():
    """``complete`` must be False while any TBD cell remains (submission guard)."""
    table = Table(name="t", caption="c", headers=["a"], rows=[[TBD]])
    assert not table.complete
    assert "WARNING" in table.to_latex()
    assert "not yet measured" in table.to_markdown()


# ------------------------------------------------------------------ seed dedup


def test_restarted_seed_is_counted_once():
    """A crashed-and-rerun seed leaves two completed rows; only the latest counts.

    The ledger is append-only, so this is the normal state after a restart.
    Counting both would drag the mean and inflate ``n seeds`` with a phantom run.
    """
    ledger = _Ledger([
        _record("EXP-012", 1, 0.40),  # superseded restart of seed 1
        _record("EXP-012", 1, 0.50),  # the seed's real number
        _record("EXP-012", 2, 0.52),
    ])
    row = _rows(build_multi_seed(ledger))[0]
    assert row[5] == "2", "n seeds must count distinct train_seed values, not rows"
    assert "0.5000" in row[6] and "0.5200" in row[6] and "0.4000" not in row[6]


def test_mean_std_best_worst_from_three_seeds():
    """The SARVO Phase-40 reporting contract: mean, std, best, worst — all present."""
    ledger = _Ledger([
        _record("EXP-012", 0, 0.50),
        _record("EXP-012", 1, 0.52),
        _record("EXP-012", 2, 0.60),
    ])
    row = _rows(build_multi_seed(ledger))[0]
    assert row[0] == "mAP50:95"
    assert row[1] == "0.5400"  # mean
    assert row[2] == "0.0529"  # sample std of (0.50, 0.52, 0.60)
    assert row[3] == "0.6000"  # best
    assert row[4] == "0.5000"  # worst
    assert row[5] == "3"
    assert table_complete(build_multi_seed(ledger))


def test_other_experiment_ids_and_failures_are_ignored():
    """Only the requested experiment's completed rows feed the table."""
    ledger = _Ledger([
        _record("EXP-007", 0, 0.99),  # different experiment
        _record("EXP-012", 0, None),  # failed run (no metric recorded)
        _record("EXP-012", 1, 0.55),
    ])
    row = _rows(build_multi_seed(ledger))[0]
    # The lone seed's measured value still shows per-seed, but mean/std stay TBD:
    # one run is not validation.
    assert row[5] == "1" and "0.5500" in row[6] and row[1] == TBD


def test_single_seed_stays_tbd():
    """One seed is not validation; the table refuses to present it as such."""
    ledger = _Ledger([_record("EXP-012", 0, 0.55)])
    (row,) = _rows(build_multi_seed(ledger))
    assert row[1] == TBD and row[2] == TBD


# --------------------------------------------------------- variant matching


def test_variant_match_at_word_boundary():
    """``v2_full`` must not match ``v2_full_s``: substring matching would report
    another model's score inside the ablation table."""
    from saryolo.paper.tables import _match_variant

    records = [_record("EXP-007", 0, 0.60, model="yolo11s_v2_full_s")]
    assert _match_variant(records, "v2_full") is None
    assert _match_variant(records, "v2_full_s") is records[0]
    assert _match_variant(records, "v2_full_l") is None


def table_complete(table: Table) -> bool:
    return not any(TBD in cell for row in table.rows for cell in row)


# ------------------------------------------------------------ render safety


def test_markdown_rows_render_all_columns():
    """Every header must have a value in every row — a ragged row in LaTeX is a
    build failure at paper time, so catch it here."""
    ledger = _Ledger([
        _record("EXP-012", 0, 0.50),
        _record("EXP-012", 1, 0.52),
    ])
    table = build_multi_seed(ledger)
    width = len(table.headers)
    for row in _rows(table):
        assert len(row) == width, f"ragged row: {row}"
    md = table.to_markdown()
    assert "0.5100" in md  # mean of 0.50/0.52


@pytest.mark.parametrize("value,expected", [(None, TBD), ("", TBD), (0.5, "0.5000"), ("x", "x")])
def test_fmt_contract(value, expected):
    from saryolo.paper.tables import _fmt

    assert _fmt(value) == expected
