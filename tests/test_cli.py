"""CLI behaviour tests: a command must never succeed while doing nothing.

The commands below all used to iterate over requested work, ``continue`` past anything
missing, and return 0. A wrong path, or a ``data.yaml`` passed where a dataset *directory*
is expected, therefore printed nothing and exited successfully — indistinguishable from a
clean pass for anything that checks the exit status. These tests pin the loud failure, and
pin that the correct invocation still succeeds, so the fix cannot be "make it always fail".
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from saryolo.cli import main


def _tiny_dataset(root: Path, splits: tuple[str, ...] = ("train", "val")) -> Path:
    """A minimal but *valid* YOLO dataset: one 32x32 image and one box per split."""
    for split in splits:
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / "a.png"), np.full((32, 32), 90, dtype=np.uint8))
        (labels / "a.txt").write_text("0 0.500000 0.500000 0.250000 0.250000\n")
    return root


# ------------------------------------------------------------------- check-data
# NOTE: these paths raise SystemExit rather than returning a code, matching how `arch`
# already reports an unusable argument. Through the interpreter both arrive as exit 1 with
# the message on stderr, which is what a shell checks; here they surface as the exception.
def test_check_data_fails_loudly_on_a_missing_path(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["check-data", "--dataset", str(tmp_path / "nope")])
    assert str(exc.value.code).strip(), "the failure must explain itself"
    assert "does not exist" in str(exc.value.code)


def test_check_data_rejects_a_data_yaml_with_an_actionable_message(tmp_path, capsys):
    """Passing the YAML is the natural mistake; the error must say what to pass instead."""
    dataset = _tiny_dataset(tmp_path / "ds")
    data_yaml = tmp_path / "ds" / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames: {0: ship}\n")

    with pytest.raises(SystemExit) as exc:
        main(["check-data", "--dataset", str(data_yaml)])
    message = str(exc.value.code)
    assert "DIRECTORY" in message
    # The suggested directory is the one that actually contains the splits.
    assert str(dataset) in message


def test_check_data_succeeds_on_a_valid_dataset(tmp_path, capsys):
    code = main(["check-data", "--dataset", str(_tiny_dataset(tmp_path / "ds")), "--report", ""])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "images: 1" in out, out


def test_check_data_reports_splits_it_did_not_look_at(tmp_path, capsys):
    """Absent splits are skipped, but never silently: silence reads as approval."""
    dataset = _tiny_dataset(tmp_path / "ds", splits=("train",))
    code = main(["check-data", "--dataset", str(dataset), "--report", ""])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "not found and therefore not checked" in out
    assert "val" in out


# ----------------------------------------------------------------------- stats
def test_stats_fails_loudly_on_a_missing_path(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["stats", "--dataset", str(tmp_path / "nope"), "--classes", "ship",
              "--out", str(tmp_path / "s")])
    assert "does not exist" in str(exc.value.code)


# ----------------------------------------------------------------------- bench
def test_bench_fails_when_a_requested_variant_cannot_be_resolved(capsys):
    """A typo must not leave the caller a success status and a shorter table."""
    from saryolo.cli import main as cli_main

    code = cli_main(["bench", "--variants", "baseline_s", "definitely_not_a_variant"])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "could not be resolved" in out
    # The resolvable variant is still profiled: the failure reports the gap, it does not
    # hide the work that did succeed.
    assert "baseline_s" in out


def test_bench_succeeds_for_a_resolvable_variant(capsys):
    code = main(["bench", "--variants", "baseline_s"])
    assert code == 0, capsys.readouterr().out


# ----------------------------------------------------------------------- train
def test_failed_train_reports_the_cause_not_just_the_status(capsys, monkeypatch):
    """A failed run must say why. The status alone is undiagnosable.

    The runner catches the training exception and stores it in ``notes``; a CLI that prints
    only ``failed: EXP-019 -> None`` hands the user no exception, no message, and a run
    directory of ``None``, because the failure happened before one existed. This was the
    behaviour observed when an override was passed twice.
    """
    from saryolo.cli import main as cli_main
    from saryolo.tracking.ledger import ExperimentRecord

    def fake_run(*_args, **_kwargs):
        record = ExperimentRecord(experiment_id="EXP-019", model="m.yaml", dataset="d.yaml",
                                  train_seed=0, epochs=1, batch=2, imgsz=640, optimizer="auto",
                                  lr0=None, notes="purpose text | ERROR: boom: the real cause")
        record.status = "failed"
        return record

    monkeypatch.setattr("saryolo.training.runner.run_experiment", fake_run)
    code = cli_main(["train", "--exp", "configs/exp/EXP-019_v2_cons.yaml", "--no-ledger"])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "the real cause" in out, out
    assert "ERROR" not in out.split("reason:")[-1], "the marker itself should not leak"


def test_failure_reason_survives_a_note_without_an_error_marker():
    """An older or truncated record must still print something usable, not an empty line."""
    from saryolo.cli import _failure_reason

    assert _failure_reason("note | ERROR: cause") == "cause"
    assert _failure_reason("  only a note  ") == "only a note"
    assert _failure_reason("two | ERROR: first | ERROR: second") == "first | ERROR: second"


# ---------------------------------------------------------------------- loso
def _sensor_images(root, sensors: dict[str, int]):
    root.mkdir(parents=True, exist_ok=True)
    for sensor, count in sensors.items():
        d = root / sensor / "images"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (d / f"{sensor}_{i:03d}.png").write_bytes(b"x")
    return root


def test_metadata_command_builds_a_table_the_resolution_rule_accepts(tmp_path, capsys):
    """The missing third input of the conditioning experiments: archives that state
    acquisition only in a CSV. The command's output has to be consumable by ``loso
    --rule resolution`` without hand-editing, so that is exactly what this asserts."""
    root = tmp_path / "images"
    root.mkdir()
    rows = ["stem,sensor,resolution_m,polarization,mode,band,incidence_deg"]
    for i in range(20):
        stem = f"sentinel1_{i:03d}"
        (root / f"{stem}.png").write_bytes(b"x")
        rows.append(f"{stem},sentinel1,10.0,VV,stripmap,C,33.0")
    for i in range(20):
        stem = f"gaofen3_{i:03d}"
        (root / f"{stem}.png").write_bytes(b"x")
        rows.append(f"{stem},gaofen3,3.0,VV,stripmap,C,33.0")
    csv_path = tmp_path / "acquisition.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    table_path = tmp_path / "metadata.json"

    code = main(["metadata", "--images", str(root), "--sidecar", str(csv_path),
                 "--out", str(table_path)])
    assert code == 0, capsys.readouterr().out
    assert table_path.exists()
    printed = capsys.readouterr().out
    assert "40 images" in printed
    assert "sensor=100%" in printed

    # The table feeds the cross-resolution rule directly -- the hand-off that has to work.
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(f"path: {root}\nnc: 1\nnames:\n- ship\nval: images\n")
    code = main(["loso", "--images", str(root), "--data", str(data_cfg), "--rule", "resolution",
                 "--metadata", str(table_path), "--edges", "5", "--min-test-images", "5",
                 "--out", str(tmp_path / "folds")])
    assert code == 0, capsys.readouterr().out
    assert sorted(p.name for p in (tmp_path / "folds").iterdir() if p.is_dir()) == [
        "resolution_m<=5", "resolution_m>5"
    ]


def test_metadata_command_refuses_a_sidecar_matching_nothing(tmp_path):
    """A table of all-unknowns would make every conditioning arm identical while the
    experiment still reports a result -- the quietest way to kill the ablation."""
    root = tmp_path / "images"
    root.mkdir()
    for i in range(4):
        (root / f"chip_{i}.png").write_bytes(b"x")
    csv_path = tmp_path / "acquisition.csv"
    csv_path.write_text("stem,sensor,resolution_m\nother_0,sentinel1,10.0\n")

    with pytest.raises(SystemExit, match="cannot build the metadata table"):
        main(["metadata", "--images", str(root), "--sidecar", str(csv_path),
              "--out", str(tmp_path / "metadata.json")])


def test_metadata_command_needs_real_inputs(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "chip_0.png").write_bytes(b"x")
    with pytest.raises(SystemExit, match="is not a directory"):
        main(["metadata", "--images", str(tmp_path / "nope"),
              "--sidecar", str(tmp_path / "s.csv"), "--out", str(tmp_path / "m.json")])
    with pytest.raises(SystemExit, match="does not exist"):
        main(["metadata", "--images", str(root), "--sidecar", str(tmp_path / "nope.csv"),
              "--out", str(tmp_path / "m.json")])
    empty = tmp_path / "empty"
    empty.mkdir()
    csv_path = tmp_path / "s.csv"
    csv_path.write_text("stem,sensor\n")
    with pytest.raises(SystemExit, match="no dataset images"):
        main(["metadata", "--images", str(empty), "--sidecar", str(csv_path),
              "--out", str(tmp_path / "m.json")])


def test_loso_writes_one_runnable_fold_per_source(tmp_path, capsys):
    root = _sensor_images(tmp_path / "raw", {"s1": 20, "g3": 20})
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(f"path: {root}\nnc: 1\nnames:\n- ship\nval: s1/images\n")
    out = tmp_path / "loso"

    code = main(["loso", "--images", str(root), "--data", str(data_cfg),
                 "--min-test-images", "5", "--out", str(out)])
    assert code == 0, capsys.readouterr().out
    printed = capsys.readouterr().out
    assert "2 source(s)" in printed
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["g3", "s1"]
    for fold in ("s1", "g3"):
        assert (out / fold / "data.yaml").exists()
        assert (out / fold / "eval_holdout.yaml").exists()


def test_loso_refuses_a_rule_that_keys_on_the_split_layout(tmp_path, capsys):
    """Pointing the rule one level too high is the mistake that silently fakes the claim."""
    root = tmp_path / "processed" / "images"
    for split, count in (("train", 6), ("val", 3), ("test", 3)):
        d = root / split
        d.mkdir(parents=True)
        for i in range(count):
            (d / f"{split}_{i}.png").write_bytes(b"x")

    with pytest.raises(SystemExit) as exc:
        main(["loso", "--images", str(root), "--out", str(tmp_path / "loso")])
    assert "split names, not sources" in str(exc.value.code)


def test_loso_needs_a_rule_it_can_actually_apply(tmp_path):
    root = _sensor_images(tmp_path / "raw", {"s1": 20, "g3": 20})
    with pytest.raises(SystemExit, match="requires --pattern"):
        main(["loso", "--images", str(root), "--rule", "regex", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="requires --sidecar"):
        main(["loso", "--images", str(root), "--rule", "sidecar", "--out", str(tmp_path / "o")])
    # The resolution rule needs both of its inputs stated, and neither is guessable: the bin
    # width is a modelling choice, and the resolution is not in the filename.
    with pytest.raises(SystemExit, match="requires --metadata"):
        main(["loso", "--images", str(root), "--rule", "resolution", "--out", str(tmp_path / "o")])
    table = _metadata_table(tmp_path / "meta.json", {p.stem: 5.0 for p in root.rglob("*.png")})
    with pytest.raises(SystemExit, match="requires --edges"):
        main(["loso", "--images", str(root), "--rule", "resolution", "--metadata", str(table),
              "--out", str(tmp_path / "o")])


def _metadata_table(path, resolutions: dict[str, float | None]):
    """Minimal metadata table JSON, in the shape ``MetadataTable.load`` expects."""
    entries = {
        stem: {"sensor": "sentinel1", "resolution_m": value, "polarization": "VV",
               "mode": None, "band": "C", "incidence_deg": 35.0}
        for stem, value in resolutions.items()
    }
    path.write_text(json.dumps({"source": "test", "entries": entries, "vocabularies": {}}))
    return path


def test_loso_resolution_rule_builds_cross_resolution_folds(tmp_path, capsys):
    """Experiment D end to end: resolution bins become the held-out 'sources'."""
    root = tmp_path / "raw"
    root.mkdir(parents=True)
    resolutions = {}
    for i in range(40):
        stem = f"chip_{i:03d}"
        (root / f"{stem}.png").write_bytes(b"x")
        resolutions[stem] = 5.0 if i < 20 else 20.0
    table = _metadata_table(tmp_path / "meta.json", resolutions)
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(f"path: {root}\nnc: 1\nnames:\n- ship\nval: images\n")
    out = tmp_path / "crossres"

    code = main(["loso", "--images", str(root), "--data", str(data_cfg), "--rule", "resolution",
                 "--metadata", str(table), "--edges", "10", "--min-test-images", "5",
                 "--out", str(out)])
    assert code == 0, capsys.readouterr().out
    assert "2 source(s)" in capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [
        "resolution_m<=10", "resolution_m>10"
    ]
    for fold in ("resolution_m<=10", "resolution_m>10"):
        assert (out / fold / "eval_holdout.yaml").exists()


def test_loso_resolution_rule_refuses_an_image_it_cannot_bin(tmp_path):
    """An image with no metadata row would vanish from every fold without a word."""
    root = tmp_path / "raw"
    root.mkdir(parents=True)
    resolutions = {}
    for i in range(20):
        stem = f"chip_{i:03d}"
        (root / f"{stem}.png").write_bytes(b"x")
        resolutions[stem] = 5.0 if i < 10 else 20.0
    del resolutions["chip_000"]
    table = _metadata_table(tmp_path / "meta.json", resolutions)

    with pytest.raises(SystemExit, match="no row in"):
        main(["loso", "--images", str(root), "--rule", "resolution",
              "--metadata", str(table), "--edges", "10", "--out", str(tmp_path / "o")])


def test_loso_resolution_rule_refuses_bins_that_measure_nothing(tmp_path):
    """One populated bin means there is no resolution shift, so the claim is untestable."""
    root = tmp_path / "raw"
    root.mkdir(parents=True)
    resolutions = {}
    for i in range(20):
        stem = f"chip_{i:03d}"
        (root / f"{stem}.png").write_bytes(b"x")
        resolutions[stem] = 3.0
    table = _metadata_table(tmp_path / "meta.json", resolutions)

    with pytest.raises(SystemExit, match="single bin"):
        main(["loso", "--images", str(root), "--rule", "resolution",
              "--metadata", str(table), "--edges", "10", "--out", str(tmp_path / "o")])


# ------------------------------------------------------------------- arch/registry
def test_arch_rejects_an_unknown_variant(tmp_path):
    with pytest.raises(SystemExit):
        main(["arch", "--variant", "not_a_variant", "--out", str(tmp_path)])
