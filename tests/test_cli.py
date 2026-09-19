"""CLI behaviour tests: a command must never succeed while doing nothing.

The commands below all used to iterate over requested work, ``continue`` past anything
missing, and return 0. A wrong path, or a ``data.yaml`` passed where a dataset *directory*
is expected, therefore printed nothing and exited successfully — indistinguishable from a
clean pass for anything that checks the exit status. These tests pin the loud failure, and
pin that the correct invocation still succeeds, so the fix cannot be "make it always fail".
"""

from __future__ import annotations

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


# ------------------------------------------------------------------- arch/registry
def test_arch_rejects_an_unknown_variant(tmp_path):
    with pytest.raises(SystemExit):
        main(["arch", "--variant", "not_a_variant", "--out", str(tmp_path)])
