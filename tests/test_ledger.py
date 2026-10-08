"""The experiment ledger's own contract: append-only, concurrent-safe, measurement-only.

Why this file exists
--------------------
Every accuracy number in the paper is read back out of ``results/experiments.jsonl``, and the
README's pilot table is *checked against* it. The ledger is therefore load-bearing in the
no-fabrication argument, and it is the one artefact written by a process that is running for
hours on a machine where other runs may be doing the same thing at the same time.

Three properties are asserted here, each of which has a failure mode that would be invisible
in the output:

* **Append-only.** A rerun must add a row, never replace one, or a table can silently change
  meaning between two reads of the same history.
* **Concurrent-safe.** ``_exclusive`` serialises writers across processes and the derived CSV
  is rewritten through a per-process temp file. Both were added in response to a measured
  failure (two processes racing on a *shared* temp path, one raising ``FileNotFoundError``
  while the other replaced it). Four real subprocesses are used rather than threads, because
  the lock is an ``fcntl`` file lock and threads in one interpreter do not exercise it.
* **Measurement-only.** A metric that was not measured must be *absent*, and render as an
  empty cell in the CSV, not as ``0``. A zero is a claim; an empty cell is not.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from saryolo.tracking.ledger import METRIC_KEYS, ExperimentLedger

REPO_ROOT = Path(__file__).resolve().parents[1]


def _start(ledger: ExperimentLedger, exp_id: str = "LEDGER-001", seed: int = 0):
    return ledger.start(
        exp_id, "yolo11n_baseline_n.yaml", "hrsid_real.yaml", {"train": {"epochs": 1}},
        train_seed=seed, epochs=1, batch=4, imgsz=320, optimizer="auto", notes="unit test row",
    )


def test_a_run_is_visible_while_it_is_still_running(tmp_path):
    """A crash mid-run must leave a trace, which is the reason the row is written up front."""
    ledger = ExperimentLedger(tmp_path)
    started = _start(ledger)
    rows = ledger.load()
    assert len(rows) == 1
    assert rows[0].status == "running" and rows[0].metrics == {}
    finished = ledger.finish(started, {"mAP50_95": 0.3012})
    assert len(ledger.load()) == 2, "finishing a run replaced its row instead of appending one"
    assert finished.run_id == started.run_id
    assert finished.status == "completed" and finished.metrics["mAP50_95"] == 0.3012


def test_only_measured_metrics_are_stored_and_the_rest_stay_empty_in_the_csv(tmp_path):
    """An unmeasured metric must be absent, and must not become a zero in the derived view."""
    ledger = ExperimentLedger(tmp_path)
    row = ledger.finish(_start(ledger), {"mAP50": 0.571, "mAP50_95": None, "fps": 61.8})
    assert row.metrics["mAP50_95"] is None
    assert row.metrics["mAP50"] == 0.571
    with (tmp_path / "experiments.csv").open(newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 2  # the running row and the finished one
    finished = csv_rows[-1]
    assert finished["mAP50"] == "0.571"
    assert finished["mAP50_95"] == "", "an unmeasured metric was written as a number"
    for key in METRIC_KEYS:
        assert key in finished, f"the CSV view dropped the metric column {key!r}"


def test_concurrent_writers_cannot_corrupt_the_ledger_or_its_csv(tmp_path):
    """Four processes, the pattern the pilot runs are actually launched with."""
    script = textwrap.dedent(
        """
        import sys
        from saryolo.tracking.ledger import ExperimentLedger

        root, index = sys.argv[1], int(sys.argv[2])
        ledger = ExperimentLedger(root)
        record = ledger.start(
            f"CONC-{index}", "m.yaml", "d.yaml", {"i": index},
            train_seed=index, epochs=1, batch=1, imgsz=320, optimizer="auto", notes="concurrent",
        )
        ledger.finish(record, {"mAP50_95": 0.1 * index, "fps": 10.0 + index})
        """
    )
    workers = 4
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(i)],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for i in range(workers)
    ]
    for proc in procs:
        _out, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, f"a concurrent writer failed:\n{err}"

    text = (tmp_path / "experiments.jsonl").read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2 * workers, f"expected {2 * workers} rows, found {len(lines)}"
    for line in lines:
        json.loads(line)  # a torn or interleaved line raises here

    ledger = ExperimentLedger(tmp_path)
    ids = sorted({r.experiment_id for r in ledger.completed()})
    assert ids == [f"CONC-{i}" for i in range(workers)]
    # One run id per run, shared by its `running` and `completed` rows *on purpose*: that is
    # what makes a crash mid-run attributable to the run that started, rather than looking
    # like an orphan. So the count is the number of runs, not the number of rows.
    assert len({r.run_id for r in ledger.load()}) == workers, "two writers shared a run id"

    with (tmp_path / "experiments.csv").open(newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 2 * workers, (
        "the derived CSV lost rows written concurrently; the rewrite raced"
    )
    assert {r["experiment_id"] for r in csv_rows} == set(ids)
    assert not list(tmp_path.glob("*.tmp")), "a CSV temp file was left behind"


def test_the_csv_is_derived_and_can_be_rebuilt_from_the_jsonl(tmp_path):
    """The CSV is a view: deleting it must not lose anything, and rebuilding must be faithful."""
    ledger = ExperimentLedger(tmp_path)
    ledger.finish(_start(ledger, "LEDGER-002"), {"mAP50_95": 0.28})
    (tmp_path / "experiments.csv").unlink()
    ledger.finish(_start(ledger, "LEDGER-003", seed=1), {"mAP50_95": 0.29})
    with (tmp_path / "experiments.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["experiment_id"] for r in rows] == [
        "LEDGER-002", "LEDGER-002", "LEDGER-003", "LEDGER-003",
    ], "the rebuilt CSV does not reflect the JSONL's append order"


def test_a_failed_run_keeps_its_cause_and_never_becomes_a_result(tmp_path):
    """A crash must be legible and must stay out of the completed set tables read from."""
    ledger = ExperimentLedger(tmp_path)
    failed = ledger.fail(_start(ledger, "LEDGER-004"), "CUDA out of memory")
    assert failed.status == "failed"
    assert "CUDA out of memory" in failed.notes
    assert ledger.completed() == []
    assert ledger.find("LEDGER-004")[-1].status == "failed"
    assert ledger.best("LEDGER-004") is None
