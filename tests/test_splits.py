"""Leakage-controlled splitting: the module that decides what a result may be read as.

Why this file exists
--------------------
`saryolo/data/splits.py` had no test file at all, and it decides two things a result depends
on: which chips are allowed to inform which split, and whether a dataset's own split is
honoured rather than re-drawn. Its failure modes are quiet ones -- a scene straddling a
boundary inflates mAP, and an unreadable split file that is accepted anyway turns "verified
the official split" into a no-op -- so they are pinned here rather than left to a run.

The tests use tiny synthetic images and no network or dataset download, in keeping with the
low-resource rule that the default suite must run on the development host.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from saryolo.data.splits import (
    leakage_report,
    read_official_split,
    split_files,
    write_splits,
)


def _files(n: int = 20) -> list[str]:
    return [f"chip{i:02d}" for i in range(n)]


def _write(path: Path, array) -> None:
    import cv2

    ok = cv2.imwrite(str(path), array)
    assert ok, f"the fixture could not write {path}"


# ------------------------------------------------------------------- the splitter itself
def test_split_files_is_deterministic_and_partitions_every_file_exactly_once():
    """Same seed, same split; every file in exactly one split; a different seed reshuffles.

    The partition property is the load-bearing one: a file appearing in two splits (or none)
    is precisely the leakage the module exists to prevent, and it would be invisible in the
    downstream counts.
    """
    files = _files(20)
    first = split_files(files, seed=0)
    assert split_files(files, seed=0) == first, "the same seed produced a different split"

    assert set(first) == {"train", "val", "test"}
    everything = first["train"] + first["val"] + first["test"]
    assert sorted(everything) == sorted(files), "a file was dropped or duplicated"
    assert len(everything) == len(set(everything)), "a file landed in two splits"
    for split, names in first.items():
        assert names == sorted(names), f"{split} is not emitted in a stable order"

    other = split_files(files, seed=1)
    assert sorted(other["train"] + other["val"] + other["test"]) == sorted(files)
    assert other["train"] != first["train"], "the seed had no effect on the assignment"


def test_split_files_refuses_ratios_that_do_not_sum_to_one():
    """A miscounted ratio must raise, not silently mis-size every split."""
    with pytest.raises(ValueError, match="sum to 1"):
        split_files(_files(10), ratios=(0.7, 0.2, 0.2))


def test_a_scene_keyed_split_never_puts_one_scene_on_both_sides_of_a_boundary():
    """The whole point of ``scene_key``: no acquisition may straddle a split boundary.

    Chips cut from one acquisition are near-identical, so a random chip split puts copies of
    the same scene in train and test and reports a number that measures memorisation.
    """
    files = [f"{scene}_{i}" for scene in ("s1", "s2", "s3", "s4", "s5") for i in range(4)]
    splits = split_files(
        files, ratios=(0.6, 0.2, 0.2), seed=0, scene_key=lambda f: f.split("_", 1)[0]
    )
    seen: dict[str, str] = {}
    for split, names in splits.items():
        for name in names:
            scene = name.split("_", 1)[0]
            assert scene not in seen or seen[scene] == split, (
                f"scene {scene} straddles {seen.get(scene)} and {split}"
            )
            seen[scene] = split
    assert set(seen) == {"s1", "s2", "s3", "s4", "s5"}, "a scene went missing from every split"


# ------------------------------------------------------------------- official splits
def test_read_official_split_reads_a_directory_and_normalises_valid_to_val(tmp_path):
    """A dataset that ships ``valid.txt`` must be read as the ``val`` split, not a new one."""
    d = tmp_path / "splits"
    d.mkdir()
    (d / "train.txt").write_text("a\nb\n\nc\n")  # a blank line is skipped, not a file named ""
    (d / "valid.txt").write_text("d\ne\n")
    assert read_official_split(d) == {"train": ["a", "b", "c"], "val": ["d", "e"]}


def test_read_official_split_refuses_a_single_file_that_names_no_split(tmp_path):
    """A single file must name a real split, and an empty directory must be refused.

    The defect this pins: the directory branch filtered to the three known splits and raised
    when none was usable, but the single-file branch accepted *any* stem -- so
    ``read_official_split("splits.txt")`` returned ``{"splits": [...]}``, a split no consumer
    reads, turning a wrong file into a silent no-op instead of an error.
    """
    stray = tmp_path / "splits.txt"
    stray.write_text("a\nb\n")
    with pytest.raises(FileNotFoundError, match="does not name a split"):
        read_official_split(stray)

    # A single file that *does* name a split is still read, and normalised the same way.
    val = tmp_path / "validation.txt"
    val.write_text("x\ny\n")
    assert read_official_split(val) == {"val": ["x", "y"]}

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No usable split"):
        read_official_split(empty)


def test_write_splits_round_trips_through_read_official_split(tmp_path):
    """What is written must be readable back, or the persisted split is not the one used."""
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    out = write_splits(splits, tmp_path / "splits")
    assert (out / "split_summary.json").is_file()
    assert read_official_split(out) == splits


# ------------------------------------------------------------------- leakage report
def test_leakage_report_flags_a_chip_shared_across_splits_and_clears_a_disjoint_one(tmp_path):
    """An identical chip listed in two splits must be flagged; disjoint splits must be clean.

    The report is the evidence that a split is safe to report from, so a false "clean" is the
    expensive direction: it is what lets a duplicated chip inflate a number without a trace.
    """
    import numpy as np

    shared = np.zeros((64, 64), np.uint8)
    shared[16:48, 16:48] = 200
    disjoint = np.full((64, 64), 255, np.uint8)
    disjoint[40:60, 40:60] = 0
    _write(tmp_path / "train_a.png", shared)
    _write(tmp_path / "test_a.png", shared)  # identical bytes under a different name
    _write(tmp_path / "test_b.png", disjoint)

    report = leakage_report(
        {"train": ["train_a.png"], "val": [], "test": ["test_a.png", "test_b.png"]}, tmp_path
    )
    assert report.split_sizes == {"train": 1, "val": 0, "test": 2}
    assert report.n_files == 3, "an image that could not be read was silently dropped"
    assert report.exact_duplicate_pairs, "an identical chip across two splits was not flagged"
    assert not report.clean
    assert "ACTION REQUIRED" in report.summary()

    clean = leakage_report(
        {"train": ["train_a.png"], "val": [], "test": ["test_b.png"]}, tmp_path
    )
    assert clean.clean and clean.exact_duplicate_pairs == [] and clean.near_duplicate_pairs == []
    assert clean.to_dict()["clean"] is True


def test_leakage_report_honours_the_near_duplicate_threshold(tmp_path):
    """The near-duplicate budget is a knob: permissive flags a pair exact comparison would miss.

    A hash-based near-duplicate check is only as meaningful as its threshold, so the threshold
    is exercised rather than assumed. At 64 bits (the full dHash width) no cross-split pair can
    be "far", so two byte-different images must be reported as a near-duplicate.
    """
    import numpy as np

    a = np.zeros((64, 64), np.uint8)
    a[8:56, 8:56] = 120
    b = np.full((64, 64), 255, np.uint8)
    b[0:24, 0:24] = 0
    _write(tmp_path / "x_a.png", a)
    _write(tmp_path / "x_b.png", b)

    splits = {"train": ["x_a.png"], "test": ["x_b.png"]}
    permissive = leakage_report(splits, tmp_path, near_dup_threshold=64)
    flagged = len(permissive.exact_duplicate_pairs) + len(permissive.near_duplicate_pairs)
    assert flagged >= 1, "with a 64-bit threshold every cross-split pair should be near"
    assert not permissive.clean
