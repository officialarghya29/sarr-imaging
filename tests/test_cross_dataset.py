"""Cross-dataset evaluation: the module that decides when a domain-gap number means anything.

Why this file exists
--------------------
`saryolo/evaluation/cross_dataset.py` had no test file at all, and its guards are exactly the
kind that fail silently if they are wrong: evaluating a ship-only detector against a dataset
whose classes the model cannot predict produces a *low mAP* that reads as a domain gap rather
than as a class-space mismatch. The guards are pinned here, and they are reachable without a
checkpoint -- each raises before any model is loaded -- so the default suite runs them.

``domain_gap_table`` is checked too, because it is the step that turns two metric dicts into
table cells: a missing measurement must stay ``None`` and render ``TBD``, never become a
fabricated drop.
"""

from __future__ import annotations

import json

import pytest

from saryolo.evaluation.cross_dataset import (
    class_compatibility,
    cross_dataset_eval,
    domain_gap_table,
)


def _data_yaml(path, names: list[str]):
    """Write a minimal data config whose only relevant field is ``names``."""
    import yaml

    path.write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    return path


def test_class_compatibility_reports_the_shared_and_the_solely_owned_classes(tmp_path):
    """Overlap is reported in both directions, and names are compared case-insensitively.

    The docstring once claimed ``compatible`` required the source classes to be a *subset* of
    the target's, which the code does not do (and should not -- evaluation is scored on target
    ground truth). This pins the real contract: any shared class makes the pair comparable.
    """
    source = _data_yaml(tmp_path / "source.yaml", ["Ship", "Plane"])
    target = _data_yaml(tmp_path / "target.yaml", ["ship", "harbor"])
    compat = class_compatibility(source, target)
    assert compat["source_classes"] == ["ship", "plane"]
    assert compat["target_classes"] == ["ship", "harbor"]
    assert compat["shared"] == ["ship"]
    assert compat["source_only"] == ["plane"]
    assert compat["target_only"] == ["harbor"]
    assert compat["compatible"] is True

    disjoint = class_compatibility(source, _data_yaml(tmp_path / "other.yaml", ["car"]))
    assert disjoint["compatible"] is False
    assert disjoint["shared"] == []


def test_cross_dataset_eval_refuses_a_pair_with_no_class_overlap(tmp_path):
    """No shared class means the score would measure nothing, so it must raise, not report."""
    source = _data_yaml(tmp_path / "source.yaml", ["ship"])
    target = _data_yaml(tmp_path / "target.yaml", ["car"])
    with pytest.raises(ValueError, match="No class overlap"):
        cross_dataset_eval("ignored.pt", target, source_data_yaml=source, out_dir=tmp_path / "out")
    # The refusal happens before any evaluation, so nothing is written for a meaningless run.
    assert not (tmp_path / "out" / "cross_dataset.json").exists()


def test_cross_dataset_eval_refuses_target_only_classes_unless_partial_overlap_is_allowed(
    tmp_path, monkeypatch
):
    """A target class the model cannot predict is a hard stop unless the limitation is accepted.

    The allowed path is exercised with a *stub* evaluator rather than a real checkpoint, so the
    test pins the guard's own behaviour (what it refuses, and what the returned dict carries)
    rather than re-testing the detector.
    """
    source = _data_yaml(tmp_path / "source.yaml", ["ship"])
    target = _data_yaml(tmp_path / "target.yaml", ["ship", "harbor"])
    with pytest.raises(ValueError, match="cannot predict"):
        cross_dataset_eval("ignored.pt", target, source_data_yaml=source, out_dir=tmp_path / "out")

    import saryolo.evaluation.metrics as metrics_mod

    calls: dict = {}

    def fake_evaluate(weights, data, **kwargs):
        calls["weights"] = weights
        calls["data"] = data
        return {"mAP50": 0.41, "mAP50_95": 0.22}

    monkeypatch.setattr(metrics_mod, "evaluate_detections", fake_evaluate)
    out = tmp_path / "allowed"
    result = cross_dataset_eval(
        "w.pt", target, source_data_yaml=source, out_dir=out, allow_partial_overlap=True
    )
    assert calls["weights"] == "w.pt" and calls["data"] == target
    assert result["mAP50"] == 0.41
    assert result["compatibility"]["target_only"] == ["harbor"]
    assert result["domain_shift"] is True
    # The limitation must survive to the written artefact, not only the return value.
    written = json.loads((out / "cross_dataset.json").read_text())
    assert written["compatibility"]["target_only"] == ["harbor"]


def test_cross_dataset_eval_skips_the_class_guard_when_no_source_config_is_given(tmp_path, monkeypatch):
    """With no source config there is nothing to compare, and the guard is documented as optional.

    This asserts the documented behaviour rather than a new one: the result then carries
    ``compatibility = None`` so a reader can see the pairing was not checked, instead of the
    module inventing a compatibility verdict it never measured.
    """
    target = _data_yaml(tmp_path / "target.yaml", ["ship"])
    import saryolo.evaluation.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "evaluate_detections", lambda *a, **k: {"mAP50": 0.5})
    result = cross_dataset_eval("w.pt", target, out_dir=tmp_path / "out")
    assert result["compatibility"] is None
    # ``domain_shift`` now follows the same convention as ``compatibility``: with no source
    # config there is nothing to compare, so the pairing is *unestablished* (``None``) rather
    # than asserted true. The flag is decided from the two resolved data roots when a source is
    # given, and the roots travel in the payload -- see
    # ``test_cross_dataset_eval_does_not_label_an_in_domain_evaluation_a_domain_shift`` below,
    # which is the case that was measured wrong (defect D-33).
    assert result["domain_shift"] is None
    assert result["same_dataset_as_source"] is None


def test_cross_dataset_eval_does_not_label_an_in_domain_evaluation_a_domain_shift(tmp_path, monkeypatch):
    """Pointing ``--source`` and ``--target`` at one dataset is not a domain shift.

    Measured before the fix: ``cross-dataset --weights sarvo_ssac001.pt --source
    configs/datasets/hrsid_real.yaml --target configs/datasets/hrsid_real.yaml`` returned
    ``domain_shift: true`` and ``compatible: true`` for the dataset the checkpoint was trained
    on. The flag was the literal ``True`` on every call, so the field said nothing about the two
    datasets it appeared to compare. Both directions are pinned here -- same dataset, and a
    genuinely different one -- so the fix cannot be a constant either way.
    """
    import yaml

    import saryolo.evaluation.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "evaluate_detections", lambda *a, **k: {"mAP50": 0.4})

    def _config(path, names, root):
        path.write_text(yaml.safe_dump({"names": names, "nc": len(names), "path": str(root)}))
        return path

    shared = tmp_path / "one"
    shared.mkdir()
    same_a = _config(tmp_path / "a.yaml", ["ship"], shared)
    same_b = _config(tmp_path / "b.yaml", ["ship"], shared)
    other = tmp_path / "two"
    other.mkdir()
    different = _config(tmp_path / "c.yaml", ["ship"], other)

    in_domain = cross_dataset_eval("w.pt", same_b, source_data_yaml=same_a, out_dir=tmp_path / "in")
    assert in_domain["domain_shift"] is False, "an in-domain evaluation was labelled a domain shift"
    assert in_domain["same_dataset_as_source"] is True
    written = json.loads((tmp_path / "in" / "cross_dataset.json").read_text())
    assert written["domain_shift"] is False, "the written artefact must carry the same verdict"

    shifted = cross_dataset_eval("w.pt", different, source_data_yaml=same_a, out_dir=tmp_path / "out")
    assert shifted["domain_shift"] is True
    assert shifted["same_dataset_as_source"] is False


def test_domain_gap_table_computes_the_drop_and_carries_missing_measurements_as_none():
    """The table's arithmetic is pinned, and an absent measurement stays absent.

    ``absolute_drop`` and ``relative_drop_pct`` are the numbers a reader quotes, so they are
    checked against their definitions; and a metric missing on either side must stay ``None``
    rather than become a drop computed against an assumed zero.
    """
    rows = domain_gap_table(
        {"mAP50": 0.60, "mAP50_95": 0.30}, {"mAP50": 0.45, "mAP50_95": 0.18}
    )
    by_metric = {row["metric"]: row for row in rows}
    assert set(by_metric) == {"mAP50", "mAP50:95"}
    assert by_metric["mAP50"]["absolute_drop"] == pytest.approx(0.15)
    assert by_metric["mAP50"]["relative_drop_pct"] == pytest.approx(25.0)
    assert by_metric["mAP50:95"]["relative_drop_pct"] == pytest.approx(40.0)

    missing = {row["metric"]: row for row in domain_gap_table({"mAP50": 0.6}, {"mAP50": 0.5})}
    assert missing["mAP50:95"]["in_domain"] is None
    assert missing["mAP50:95"]["cross_domain"] is None
    assert missing["mAP50:95"]["absolute_drop"] is None
    assert missing["mAP50:95"]["relative_drop_pct"] is None
