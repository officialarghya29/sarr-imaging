"""The runner's pure helpers: reading a run's artefacts back without inventing anything.

These three functions are what turn a finished run into a ledger row. Each has a silent
failure mode that a training smoke test would not catch:

* ``read_results_csv`` feeds the "epochs completed" count -- an unstripped key would make
  the count read zero while the run is fine;
* ``find_best_weights`` decides which checkpoint a metric is measured on;
* ``extract_val_metrics`` is the boundary where a metric either exists or stays ``None``.
  A ``None`` that becomes ``0.0`` here would be published as a real score.
"""

from __future__ import annotations

import pytest
import torch

from saryolo.tracking.ledger import ExperimentRecord
from saryolo.training.runner import (
    _offline_finish,
    extract_val_metrics,
    find_best_weights,
    read_results_csv,
)

# ------------------------------------------------------------------ read_results_csv

def test_read_results_csv_returns_empty_for_a_missing_file(tmp_path):
    assert read_results_csv(tmp_path / "absent.csv") == []


def test_read_results_csv_strips_the_column_names(tmp_path):
    """Ultralytics writes ``epoch, train/box_loss, ...``; the keys must be usable."""
    path = tmp_path / "results.csv"
    path.write_text("epoch, train/box_loss, metrics/mAP50\n1, 0.9, 0.4\n2, 0.8, 0.5\n")
    rows = read_results_csv(path)
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[1]["metrics/mAP50"].strip() == "0.5"
    assert all(" " not in key for r in rows for key in r), "an unstripped key broke the count"


# ------------------------------------------------------------------ find_best_weights

def test_find_best_weights_prefers_best_then_last(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "last.pt").write_bytes(b"x")
    assert find_best_weights(tmp_path).name == "best.pt"
    (tmp_path / "best.pt").unlink()
    assert find_best_weights(tmp_path).name == "last.pt"


def test_find_best_weights_honours_the_preference(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "last.pt").write_bytes(b"x")
    assert find_best_weights(tmp_path, prefer="last").name == "last.pt"


def test_find_best_weights_falls_back_to_any_checkpoint(tmp_path):
    nested = tmp_path / "weights"
    nested.mkdir()
    (nested / "epoch5.pt").write_bytes(b"x")
    assert find_best_weights(tmp_path).name == "epoch5.pt"


def test_find_best_weights_returns_none_when_nothing_was_written(tmp_path):
    """No checkpoint means no metric; the caller must see ``None``, not a path that is not there."""
    assert find_best_weights(tmp_path) is None


# ------------------------------------------------------------------ extract_val_metrics

class _Box:
    def __init__(self, **kwargs):
        self.map50 = kwargs.get("map50")
        self.map = kwargs.get("map")
        self.mp = kwargs.get("mp")
        self.mr = kwargs.get("mr")
        self.maps = kwargs.get("maps")
        self.ap_small = kwargs.get("ap_small")
        self.ap_medium = kwargs.get("ap_medium")
        self.ap_large = kwargs.get("ap_large")


class _Metrics:
    def __init__(self, box):
        self.box = box


def test_extract_val_metrics_returns_all_none_when_there_is_no_box():
    """``box is None`` is an unmeasured run, and every field must stay ``None``."""
    out = extract_val_metrics(_Metrics(None))
    assert out["mAP50"] is None and out["mAP50_95"] is None
    assert out["precision"] is None and out["recall"] is None and out["f1"] is None
    assert out["per_class_AP50_95"] is None
    assert "AP_small" not in out, "a scale key must not appear when it was never measured"


def test_extract_val_metrics_computes_f1_and_keeps_absent_scales_absent():
    box = _Box(map50=0.90, map=0.50, mp=0.80, mr=0.40, maps=torch.tensor([0.5, 0.4]))
    out = extract_val_metrics(_Metrics(box))
    assert out["mAP50"] == 0.90 and out["mAP50_95"] == 0.50
    # F1 = 2pr/(p+r) = 2*0.8*0.4 / 1.2 = 0.5333...
    assert abs(out["f1"] - (2 * 0.8 * 0.4 / 1.2)) < 1e-9
    assert out["per_class_AP50_95"] == pytest.approx([0.5, 0.4])
    assert "AP_small" not in out, "no scale AP was exposed, so none may be reported"


def test_extract_val_metrics_leaves_f1_none_when_a_component_is_missing():
    """F1 needs both precision and recall; faking one from the other is not allowed."""
    out = extract_val_metrics(_Metrics(_Box(map50=0.9, mp=0.8)))
    assert out["precision"] == 0.8 and out["recall"] is None
    assert out["f1"] is None


def test_extract_val_metrics_handles_a_zero_precision_and_recall_pair():
    """Both zero is a real measurement (`f1` 0.0), not a division error."""
    out = extract_val_metrics(_Metrics(_Box(map50=0.0, map=0.0, mp=0.0, mr=0.0)))
    assert out["f1"] == 0.0


def test_extract_val_metrics_reports_scale_ap_when_the_validator_supplied_it():
    box = _Box(map50=0.9, map=0.5, ap_small=0.2, ap_medium=0.4, ap_large=0.6)
    out = extract_val_metrics(_Metrics(box))
    assert out["AP_small"] == 0.2 and out["AP_medium"] == 0.4 and out["AP_large"] == 0.6


# ------------------------------------------------------------------ offline finish

def test_offline_finish_marks_completed_without_touching_the_ledger():
    """A smoke run returns a completed record with no run directory and no ledger write."""
    record = ExperimentRecord(experiment_id="EXP-X", model="m.yaml", dataset="d.yaml")
    done = _offline_finish(record, {"mAP50": 0.5})
    assert done.status == "completed" and done.metrics == {"mAP50": 0.5}
    assert done.save_dir is None, "an offline finish must not claim a run directory"
