"""Cross-source grouping and leave-one-source-out folds.

These tests exist because every failure mode of a LOSO protocol is *silent*: a rule that
matches nothing yields a single group and an in-domain number labelled cross-source; a fold
that leaks its held-out source still trains and still fills in a table. So the guards are
tested as hard as the happy path.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from saryolo.data.groups import (
    Fold,
    SourceRule,
    binned_rule,
    discover_sources,
    leave_one_out_folds,
    write_loso_splits,
)


# --------------------------------------------------------------------------- fixtures
def _sensor_tree(root, layout: dict[str, int]):
    """Create ``root/<sensor>/<i>.png`` for each sensor, returning the paths."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for sensor, count in layout.items():
        d = root / sensor
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            p = d / f"{sensor}_{i:04d}.png"
            p.write_bytes(b"not really a png")
            paths.append(p)
    return paths


# ---------------------------------------------------------------------- SourceRule
def test_parent_rule_keys_on_the_first_directory():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"sentinel1": 3, "gaofen3": 5})
        groups = discover_sources(paths, SourceRule.parent(root))
        assert groups.sizes() == {"gaofen3": 5, "sentinel1": 3}


def test_regex_rule_keys_on_the_filename():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        for name in ("S1A_0001.png", "S1A_0002.png", "G3_0001.png"):
            (root / name).write_bytes(b"x")
        paths = sorted(root.glob("*.png"))
        groups = discover_sources(paths, SourceRule.regex(r"^(S1A|G3)_"))
        assert groups.sizes() == {"G3": 1, "S1A": 2}


def test_regex_rule_requires_exactly_one_capture_group():
    """Without a group there is no key; with two, which one is the source is ambiguous."""
    with pytest.raises(ValueError, match="exactly one capture group"):
        SourceRule.regex(r"^S1A_")
    with pytest.raises(ValueError, match="exactly one capture group"):
        SourceRule.regex(r"^(S1A)_(\\d+)")


def test_parent_rule_requires_a_root_and_a_positive_depth():
    with pytest.raises(ValueError, match="needs a root"):
        SourceRule(kind="parent")
    with pytest.raises(ValueError, match="depth must be"):
        SourceRule.parent("/tmp", depth=0)


def test_unknown_rule_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown rule kind"):
        SourceRule(kind="vibes")


# ------------------------------------------------------------------ degeneracy guards
def test_a_rule_that_matches_nothing_raises_instead_of_returning_one_group():
    """The central failure mode: no key found must not become 'one source'.

    A single group is a *valid-looking* answer -- the folds build, the files are written,
    and the held-out "source" is a random chip split. Raising is the only safe behaviour.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"sentinel1": 4})
        with pytest.raises(ValueError, match="does not match this layout"):
            discover_sources(paths, SourceRule.regex(r"^(NOPE)_"))


def test_a_single_source_raises_because_there_is_nothing_to_hold_out():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"sentinel1": 10})
        with pytest.raises(ValueError, match="at least two"):
            discover_sources(paths, SourceRule.parent(root))


def test_split_named_groups_are_refused():
    """Keying on the processed layout holds out a split, not a sensor.

    This is the mistake the rule makes when pointed one level too high, and it produces
    three plausible-looking groups with plausible sizes, so nothing else would catch it.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"train": 6, "val": 2, "test": 2})
        with pytest.raises(ValueError, match="split names, not sources"):
            discover_sources(paths, SourceRule.parent(root))


def test_unmatched_images_are_refused_unless_explicitly_allowed():
    """An unkeyed image is absent from every fold, which nobody chose."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        for name in ("S1A_1.png", "G3_1.png", "readme.txt"):
            (root / name).write_bytes(b"x")
        paths = sorted(root.glob("*"))
        with pytest.raises(ValueError, match="could not be keyed"):
            discover_sources(paths, SourceRule.regex(r"^(S1A|G3)_"))
        groups = discover_sources(paths, SourceRule.regex(r"^(S1A|G3)_"), allow_unmatched=True)
        assert groups.n_unmatched == 1
        assert groups.sizes() == {"G3": 1, "S1A": 1}


# ------------------------------------------------------------------------ LOSO folds
def test_every_fold_holds_out_exactly_one_source_and_never_leaks_it():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        layout = {"sentinel1": 40, "gaofen3": 50, "terrasar": 60}
        paths = _sensor_tree(root, layout)
        groups = discover_sources(paths, SourceRule.parent(root))
        folds = leave_one_out_folds(groups, min_test_images=10)

        assert [f.name for f in folds] == ["gaofen3", "sentinel1", "terrasar"]
        for fold in folds:
            held_out = set(groups.groups[fold.name])
            assert set(fold.test) == held_out, "the test set must be the whole held-out source"
            assert not (set(fold.train) & held_out), f"{fold.name}: held-out source leaks into train"
            assert not (set(fold.val) & held_out), f"{fold.name}: held-out source leaks into val"

            # Every chip accounted for exactly once across the three splits.
            pooled = list(fold.train) + list(fold.val) + list(fold.test)
            assert len(pooled) == len(set(pooled)) == sum(layout.values())
            assert len(fold.test) == layout[fold.name]


def test_a_source_too_small_to_measure_is_refused():
    """mAP from a handful of chips is noise dressed as evidence."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"sentinel1": 100, "tiny": 3})
        groups = discover_sources(paths, SourceRule.parent(root))
        with pytest.raises(ValueError, match="too small to evaluate"):
            leave_one_out_folds(groups, min_test_images=30)
        # And it must be the *threshold* that refused, not the grouping: lowering it works.
        folds = leave_one_out_folds(groups, min_test_images=3)
        assert {f.name for f in folds} == {"sentinel1", "tiny"}


def test_folds_are_deterministic_given_the_seed_and_the_test_set_never_moves():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"s1": 60, "g3": 60, "tsx": 60})
        groups = discover_sources(paths, SourceRule.parent(root))
        a = leave_one_out_folds(groups, seed=0, min_test_images=10)
        b = leave_one_out_folds(groups, seed=0, min_test_images=10)
        c = leave_one_out_folds(groups, seed=1, min_test_images=10)

        assert [f.train for f in a] == [f.train for f in b], "same seed must reproduce the split"
        assert [f.test for f in a] == [f.test for f in c], "the held-out source is seed-independent"
        assert [f.train for f in a] != [f.train for f in c], "the seed must actually do something"


def test_validation_is_never_source_held_out_and_says_so():
    """The reported number comes from test; val only early-stops, and the fold records it."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"s1": 40, "g3": 40})
        groups = discover_sources(paths, SourceRule.parent(root))
        for fold in leave_one_out_folds(groups, min_test_images=10):
            assert fold.val_is_source_held_out is False
            assert fold.val, "a non-held-out val set should still exist here"


def test_a_fold_that_leaks_is_refused_when_written():
    """Bookkeeping errors must fail at the point of writing, not show up as a good number."""

    from saryolo.data.groups import _assert_disjoint

    leaky = Fold(name="s1", test=("a.png", "b.png"), train=("b.png", "c.png"), val=())
    with pytest.raises(ValueError, match="both train and test"):
        _assert_disjoint(leaky)
    with pytest.raises(ValueError, match="held-out source is empty"):
        _assert_disjoint(Fold(name="s1", test=(), train=("a.png",), val=()))
    with pytest.raises(ValueError, match="both val and test"):
        _assert_disjoint(Fold(name="s1", test=("a.png",), train=("b.png",), val=("a.png",)))


# ------------------------------------------------------------------------- persistence
def test_written_folds_carry_a_manifest_that_records_the_grouping(tmp_path):
    """A cross-source number must be traceable to the grouping that produced it."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"s1": 30, "g3": 30})
        groups = discover_sources(paths, SourceRule.parent(root))
        folds = leave_one_out_folds(groups, min_test_images=10)
        out = write_loso_splits(folds, tmp_path / "loso", groups=groups, seed=7)

        for fold in folds:
            for split in ("train", "val", "test"):
                text = (out / fold.name / f"{split}.txt").read_text()
                written = [line for line in text.splitlines() if line]
                assert written == list(getattr(fold, split))

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["protocol"] == "leave-one-source-out"
        assert manifest["seed"] == 7
        assert manifest["grouping"]["n_groups"] == 2
        assert manifest["grouping"]["sizes"] == {"g3": 30, "s1": 30}
        assert [f["name"] for f in manifest["folds"]] == [f.name for f in folds]
        assert all(f["val_is_source_held_out"] is False for f in manifest["folds"])


def test_written_folds_carry_runnable_configs_with_val_pointing_at_the_held_out_source(tmp_path):
    """Each fold must be trainable *and* measurable, and measuring must use the held-out list.

    ``evaluate_detections`` reads the ``val`` entry of a data config. A fold config that
    pointed ``val`` at ``val.txt`` would therefore report a score measured on sources the model
    trained on -- an in-domain number labelled cross-source. So there are two configs and the
    difference between them is the entire protocol.
    """
    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"s1": 30, "g3": 40})
        groups = discover_sources(paths, SourceRule.parent(root))
        folds = leave_one_out_folds(groups, min_test_images=10)
        out = write_loso_splits(folds, Path(tmp) / "loso", groups=groups, names=["ship"])

        for fold in folds:
            train_cfg = yaml.safe_load((out / fold.name / "data.yaml").read_text())
            eval_cfg = yaml.safe_load((out / fold.name / "eval_holdout.yaml").read_text())
            assert train_cfg["names"] == ["ship"] and train_cfg["nc"] == 1
            assert train_cfg["train"].endswith("train.txt")
            assert train_cfg["val"].endswith("val.txt"), "training validates on val.txt"
            assert eval_cfg["val"].endswith("test.txt"), "measurement must read the held-out list"
            assert "train" not in eval_cfg

        # Without class names there is nothing to train against, so no configs are invented.
        bare = write_loso_splits(folds, Path(tmp) / "bare", groups=groups)
        assert not (bare / folds[0].name / "data.yaml").exists()


def test_a_metadata_field_restriction_is_written_into_every_fold_config(tmp_path):
    """The missing-metadata protocol must survive fold generation.

    A restriction delivered to ``write_loso_splits`` but dropped from the fold configs would
    let one fold train with full metadata while its neighbour trained with fields withheld --
    and the per-fold table would compare numbers measured under different protocols. Both
    configs per fold must carry the restriction, and the CLI-level validation has already
    refused unknown field names by the time this runs.
    """
    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _sensor_tree(root, {"s1": 30, "g3": 40})
        groups = discover_sources(paths, SourceRule.parent(root))
        folds = leave_one_out_folds(groups, min_test_images=10)
        out = write_loso_splits(
            folds, Path(tmp) / "loso", groups=groups, names=["ship"],
            metadata_fields=["sensor"],
        )

        for fold in folds:
            for config_name in ("data.yaml", "eval_holdout.yaml"):
                cfg = yaml.safe_load((out / fold.name / config_name).read_text())
                assert cfg["metadata_fields"] == ["sensor"], (fold.name, config_name)

        # No restriction in, no restriction written: a full-metadata run must not gain a
        # stale key just because the writer knows how to write one.
        plain = write_loso_splits(folds, Path(tmp) / "plain", groups=groups, names=["ship"])
        cfg = yaml.safe_load((plain / folds[0].name / "data.yaml").read_text())
        assert "metadata_fields" not in cfg


def test_binned_rule_splits_a_continuous_field_into_resolution_groups():
    """The cross-resolution protocol: resolution is a number, so it is binned, not keyed.

    A cross-resolution experiment that silently produced one bin would build folds and report a
    number that has nothing to do with resolution, which is why the bin count is asserted here
    as well as the labels.
    """
    values = {"low": 3.0, "mid": 8.0, "high": 15.0, "higher": 40.0}
    rule = binned_rule(values, [5, 10, 20], field="resolution_m")
    assert rule.mapping == {
        "low": "resolution_m<=5",
        "mid": "5<resolution_m<=10",
        "high": "10<resolution_m<=20",
        "higher": "resolution_m>20",
    }
    assert len(set(rule.mapping.values())) == 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5.0, "resolution_m<=5"),        # the edge itself belongs to the lower bin
        (5.0001, "5<resolution_m<=10"),
        (10.0, "5<resolution_m<=10"),
        (10.0001, "10<resolution_m<=20"),
        (20.0, "10<resolution_m<=20"),
        (20.0001, "resolution_m>20"),
        (0.001, "resolution_m<=5"),      # below the first edge, so not dropped
        (1e6, "resolution_m>20"),        # far above the last
    ],
)
def test_bins_are_half_open_and_unbounded_at_both_ends(value, expected):
    """An out-of-range resolution must land in a bin, never fall out of the grouping.

    A closed range would silently drop the extremes -- and those are exactly the resolutions a
    cross-resolution experiment cares about.
    """
    # ``low`` and ``high`` anchor the extremes so at least two bins are always populated; the
    # single-bin guard would otherwise fire for the cases under test and mask the boundary
    # behaviour being checked.
    rule = binned_rule({"a": value, "low": 1.0, "high": 1e6}, [5, 10, 20])
    assert rule.mapping["a"] == expected


def test_binned_rule_refuses_an_image_it_cannot_place():
    """An unbinnable image must be an error, because its absence is invisible downstream."""
    with pytest.raises(ValueError, match="have no resolution_m recorded"):
        binned_rule({"a": 1.0, "b": None, "c": 9.0}, [5])
    # Explicit opt-in is allowed, and it is then the caller's stated choice.
    rule = binned_rule({"a": 1.0, "b": None, "c": 9.0}, [5], allow_missing=True)
    assert "b" not in rule.mapping
    # And if *everything* is missing there is nothing left to group.
    with pytest.raises(ValueError, match="no image has a recorded"):
        binned_rule({"a": None}, [5], allow_missing=True)


@pytest.mark.parametrize(
    ("values", "edges", "match"),
    [
        ({"a": 1.0, "b": 9.0}, [], "at least one bin edge"),
        ({"a": 1.0, "b": 9.0}, [10, 5], "strictly ascending"),
        ({"a": 1.0, "b": 9.0}, [5, 5], "strictly ascending"),
        ({"a": 1.0, "b": 2.0}, [5, 10], "single bin"),
        ({"a": float("nan"), "b": 9.0}, [5], "cannot be binned"),
        ({"a": float("inf"), "b": 9.0}, [5], "cannot be binned"),
    ],
)
def test_binned_rule_refuses_degenerate_bins(values, edges, match):
    """Each of these would produce a run that completes while measuring the wrong thing.

    NaN is the subtle one: it compares false against every edge, so it would fall through to the
    last bin and be reported as the *highest* resolution class with nothing looking wrong.
    """
    with pytest.raises(ValueError, match=match):
        binned_rule(values, edges)


def test_a_single_bin_is_refused_before_folds_are_built():
    """The one failure that is easy to miss: folds still build, a number is still reported.

    Checked through ``discover_sources`` as well, because a caller may hand the rule on rather
    than constructing it here -- the refusal has to survive the whole path.
    """
    values = {f"chip_{i}": 3.0 for i in range(10)}
    with pytest.raises(ValueError, match="single bin"):
        binned_rule(values, [20])

    # Constructed directly, a one-bin sidecar rule reaches the two-source guard instead.
    one_bin = SourceRule.sidecar({f"chip_{i}": "res<=20" for i in range(10)})
    with pytest.raises(ValueError, match="Only one source"):
        discover_sources([f"/x/chip_{i}.png" for i in range(10)], one_bin)


def test_binned_rule_labels_are_stable_and_stripped_of_float_noise():
    """Labels go into directory names and fold ids, so they must not carry ``10.0`` style noise.

    A label like ``10.0<resolution_m<=20.0`` would also make the fold directory names awkward
    on the command line, which is where they are used.
    """
    rule = binned_rule({"a": 1.0, "b": 15.0, "c": 50.0}, [10.0, 20.0])
    assert sorted(set(rule.mapping.values())) == [
        "10<resolution_m<=20", "resolution_m<=10", "resolution_m>20"
    ]
    assert not any(".0" in label for label in rule.mapping.values())


def test_sidecar_rule_keys_from_an_explicit_mapping():
    """The honest fallback for archives that encode the sensor nowhere."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        names = [f"chip_{i}.png" for i in range(6)]
        for name in names:
            (root / name).write_bytes(b"x")
        mapping = {f"chip_{i}": ("s1" if i < 4 else "g3") for i in range(6)}
        groups = discover_sources(sorted(root.glob("*.png")), SourceRule.sidecar(mapping))
        assert groups.sizes() == {"g3": 2, "s1": 4}
        with pytest.raises(ValueError, match="needs a mapping"):
            SourceRule.sidecar({})
