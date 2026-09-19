"""Hard-example mining (SEC. 6): the miner must agree with the failure taxonomy.

The miner exists to oversample the images the model failed on. Its whole value depends on
two properties that are easy to get wrong and invisible in a training log:

1. **It must count the same failures the failure table counts.** An earlier version carried
   its own IoU matcher, so the miner and the paper's failure analysis could disagree about
   the same model. These tests pin that they are derived from one taxonomy.
2. **It must not lose ground truth.** The score is normalised by the ground-truth count, so
   an image whose ground truth failed to load would silently score as easy (division by a
   larger number) or as infinitely hard. Both are wrong, and neither raises.
"""

from __future__ import annotations

import pytest

from saryolo.evaluation.metrics import Detection
from saryolo.training.hard_examples import (
    DIFFICULTY_WEIGHTS,
    ImageDifficulty,
    rank_examples,
    score_examples,
    write_hard_data_config,
    write_oversampled_list,
)
from saryolo.visualization.error_analysis import analyse_failures


def _box(image, cls, xyxy, score=1.0):
    return Detection(image, cls, xyxy, score=score)


# ---------------------------------------------------------------- taxonomy agreement
def test_score_totals_equal_the_failure_taxonomy_counts():
    """Every outcome the taxonomy reports is weighted, and the counts match exactly."""
    gts = [
        _box("a", 0, (10, 10, 20, 20)),      # matched
        _box("a", 0, (50, 50, 58, 58)),      # small, missed
        _box("b", 0, (10, 10, 30, 30)),      # missed
    ]
    preds = [
        _box("a", 0, (10, 10, 20, 20)),      # true positive
        _box("a", 0, (90, 90, 99, 99)),      # false positive
    ]
    rows = score_examples(preds, gts)
    summary = analyse_failures(preds, gts)

    by_image: dict[str, dict[str, int]] = {}
    for case in summary.all_cases():
        by_image.setdefault(case.image, {})
        by_image[case.image][case.kind] = by_image[case.image].get(case.kind, 0) + 1

    for row in rows:
        assert row.counts == by_image.get(row.image, {}), (
            f"{row.image}: miner counts {row.counts} disagree with taxonomy {by_image.get(row.image)}"
        )
    # And the totals reconcile: nothing was dropped on the way through.
    assert sum(sum(r.counts.values()) for r in rows) == len(summary.all_cases())


def test_every_taxonomy_outcome_has_a_weight():
    """A taxonomy that grows an outcome must fail here, not silently score zero."""
    gts = [_box("a", 0, (10, 10, 20, 20))]
    preds = [_box("a", 0, (10, 10, 20, 20)), _box("a", 1, (10, 10, 20, 20), score=2.0)]
    rows = score_examples(preds, gts)
    produced = {k for r in rows for k in r.counts}
    assert produced, "the fixture must actually produce some outcomes for this to test anything"
    assert produced <= set(DIFFICULTY_WEIGHTS)


def test_perfect_detector_is_not_hard():
    gts = [_box("a", 0, (10, 10, 20, 20)), _box("b", 0, (0, 0, 4, 4))]
    preds = [gts[0], gts[1]]
    rows = score_examples(preds, gts)
    assert [r.score for r in rows] == [0.0, 0.0]
    assert all(r.missed == 0 and r.spurious == 0 for r in rows)


def test_small_object_miss_outweighs_a_plain_miss():
    """The headline claim is small-object recall, so its weight must actually dominate."""
    small = [_box("a", 0, (0, 0, 8, 8))]                 # area 64 < 32^2
    large = [_box("a", 0, (0, 0, 200, 200))]             # area 40000 > 32^2
    small_rows = score_examples([], small)
    large_rows = score_examples([], large)
    assert small_rows[0].counts.get("small_object_miss") == 1
    assert large_rows[0].counts.get("false_negative") == 1
    assert small_rows[0].score > large_rows[0].score


def test_one_error_in_a_dense_scene_is_less_hard_than_the_same_error_alone():
    """Normalisation: the score is a per-image failure *rate*, not a raw error count.

    One missed target in a five-target scene is a worse detector *rate* than one missed
    target in a one-target scene, so the sparse image must rank higher. Deliberately not
    phrased as "more ground truth always scores lower": when every object is missed the
    rate is 1.0 for both, which is the correct behaviour and not what this test is about.
    """
    alone = score_examples([], [_box("a", 0, (0, 0, 50, 50))])[0]
    # Five dense objects, one of which is missed; the other four are detected.
    gts = [_box("b", 0, (i * 60, 0, i * 60 + 50, 50)) for i in range(5)]
    dense = score_examples(gts[:4], gts)[0]
    assert dense.n_gt == 5 and alone.n_gt == 1
    assert dense.counts.get("false_negative") == 1
    assert dense.score < alone.score


def test_all_objects_missed_scores_a_rate_of_one_whatever_the_density():
    """The complement of the above: a total miss is a rate of 1.0 at any density."""
    one = score_examples([], [_box("a", 0, (0, 0, 50, 50))])[0]
    four = score_examples([], [_box("a", 0, (0, 0, 50, 50)) for _ in range(4)])[0]
    assert one.score == four.score == pytest.approx(1.0)


def test_an_image_with_no_ground_truth_still_scores_its_false_positives():
    """Pure clutter has no denominator, which is exactly where the FP count *is* the signal."""
    rows = score_examples([_box("clutter", 0, (5, 5, 25, 25))], [])
    assert len(rows) == 1
    assert rows[0].n_gt == 0
    assert rows[0].score > 0.0


def test_unknown_weight_key_is_rejected():
    with pytest.raises(KeyError, match="Unknown difficulty weight"):
        score_examples([], [_box("a", 0, (0, 0, 9, 9))], weights={"not_an_outcome": 1.0})


def test_weight_override_does_not_require_restating_the_defaults():
    gts = [_box("a", 0, (0, 0, 8, 8))]  # small miss
    default_score = score_examples([], gts)[0].score
    heavier = score_examples([], gts, weights={"small_object_miss": 90.0})[0].score
    assert heavier > default_score


# ------------------------------------------------------------------------ ranking
def test_rank_examples_is_deterministic_and_hardest_first():
    rows = [
        ImageDifficulty("z", score=0.0),
        ImageDifficulty("a", score=2.0),
        ImageDifficulty("m", score=2.0),
    ]
    hard, easy = rank_examples(rows, top_frac=0.5)
    assert [r.image for r in hard] == ["a", "m"]   # tie broken by name, not by input order
    assert [r.image for r in easy] == ["z"]


def test_rank_examples_respects_min_count_on_a_tiny_split():
    rows = [ImageDifficulty(f"i{i}", score=float(i)) for i in range(3)]
    hard, _ = rank_examples(rows, top_frac=0.0, min_count=2)
    assert len(hard) == 2


def test_rank_examples_can_return_no_hard_set():
    rows = [ImageDifficulty("a", score=1.0)]
    hard, easy = rank_examples(rows, top_frac=0.0, min_count=0)
    assert hard == [] and len(easy) == 1


# ------------------------------------------------------------------- oversampled list
def test_oversampled_list_keeps_every_image_and_repeats_only_the_hard_ones(tmp_path):
    out = write_oversampled_list(["a", "b", "c"], ["b"], tmp_path / "train.txt", repeats=3)
    lines = out.read_text().split()
    assert sorted(lines) == ["a", "b", "b", "b", "c"], "hard images are added, never substituted"
    assert set(lines) >= {"a", "b", "c"}, "a dropped image would make this a different dataset"


def test_oversampled_list_with_one_repeat_is_the_plain_list(tmp_path):
    """`repeats=1` is the honest control: no oversampling at all."""
    out = write_oversampled_list(["a", "b"], ["a"], tmp_path / "t.txt", repeats=1)
    assert out.read_text().split() == ["a", "b"]


def test_oversampled_list_repeats_nothing_when_no_image_is_hard(tmp_path):
    out = write_oversampled_list(["a", "b"], [], tmp_path / "t.txt", repeats=4)
    assert out.read_text().split() == ["a", "b"]


def test_hard_image_from_another_split_is_an_error_not_a_silent_no_op(tmp_path):
    """Regression: mining `val` and passing those paths wrote a list containing none of them.

    The filter that dropped them made the whole strategy a no-op that still printed a repeat
    count and exited 0 -- a run that looks like hard-example training and is byte-identical to
    the baseline. A hard image outside the training list is always a mistake, so it raises.
    """
    with pytest.raises(ValueError, match="not in the training list"):
        write_oversampled_list(["/ds/images/train/a.png"], ["/ds/images/val/b.png"],
                               tmp_path / "t.txt", repeats=2)


def test_the_no_op_error_counts_only_the_images_that_are_actually_missing(tmp_path):
    """`b` is in both lists and must not be reported; only `c` and `d` are the problem."""
    with pytest.raises(ValueError) as exc:
        write_oversampled_list(["b"], ["b", "c", "d"], tmp_path / "t.txt", repeats=2)
    message = str(exc.value)
    assert "2 hard image(s)" in message
    assert "c" in message and "d" in message


# --------------------------------------------------- the contract ultralytics must honour
def test_the_oversampled_list_is_what_the_dataset_loader_actually_reads(tmp_path):
    """The whole strategy rests on ultralytics reading a ``.txt`` list as a split.

    That assumption is load-bearing and would otherwise be untested: if the loader ignored the
    list, the hard-example run would silently train on the original 64 images and the
    comparison against the baseline would be a no-op reported as a result. So the count is
    measured through the real dataset class, not asserted from the file's line count.
    """
    import cv2
    import numpy as np
    from ultralytics.data.dataset import YOLODataset

    from saryolo.data.yolo import load_data_config

    ds = tmp_path / "ds"
    images, labels = ds / "images" / "train", ds / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rng = np.random.default_rng(0)
    base = []
    for i in range(4):
        p = images / f"i{i}.png"
        cv2.imwrite(str(p), (rng.random((32, 32)) * 255).astype(np.uint8))
        (labels / f"i{i}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        base.append(str(p))
    (ds / "images" / "val").mkdir(parents=True)
    (ds / "labels" / "val").mkdir(parents=True)

    src = ds / "data.yaml"
    src.write_text(
        f"path: {ds}\nnc: 1\nnames: [target]\ntrain: images/train\nval: images/val\n"
    )

    hard = base[:2]
    listing = write_oversampled_list(base, hard, tmp_path / "train_hard.txt", repeats=3)
    cfg_path = write_hard_data_config(src, listing, tmp_path / "data_hard.yaml")

    _root, cfg = load_data_config(cfg_path)
    dataset = YOLODataset(img_path=str(listing), data=dict(cfg), task="detect", augment=False)
    # 4 base images + 2 hard images repeated twice more = 8 entries.
    assert len(dataset.im_files) == 8, (
        f"the loader read {len(dataset.im_files)} images; the oversampled list has 8 lines. "
        "If this is 4, the list file is being ignored and hard-example mining is a no-op."
    )


def test_the_control_list_loads_exactly_the_base_images(tmp_path):
    """`repeats=1` must load the original count, so the control is the baseline dataset."""
    import cv2
    import numpy as np
    from ultralytics.data.dataset import YOLODataset

    from saryolo.data.yolo import load_data_config

    ds = tmp_path / "ds"
    images, labels = ds / "images" / "train", ds / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    base = []
    for i in range(3):
        p = images / f"i{i}.png"
        cv2.imwrite(str(p), np.zeros((32, 32), dtype=np.uint8))
        (labels / f"i{i}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        base.append(str(p))
    (ds / "images" / "val").mkdir(parents=True)
    (ds / "labels" / "val").mkdir(parents=True)
    src = ds / "data.yaml"
    src.write_text(f"path: {ds}\nnc: 1\nnames: [target]\ntrain: images/train\nval: images/val\n")

    listing = write_oversampled_list(base, base[:1], tmp_path / "t.txt", repeats=1)
    cfg_path = write_hard_data_config(src, listing, tmp_path / "d.yaml")
    _root, cfg = load_data_config(cfg_path)
    dataset = YOLODataset(img_path=str(listing), data=dict(cfg), task="detect", augment=False)
    assert len(dataset.im_files) == 3, "repeats=1 must not oversample anything"


# --------------------------------------------------------------------- data config
def test_hard_data_config_moves_only_train(tmp_path):
    yaml = pytest.importorskip("yaml")
    ds = tmp_path / "ds"
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "images" / "val").mkdir(parents=True)
    src = ds / "data.yaml"
    src.write_text(
        yaml.safe_dump({
            "path": str(ds), "nc": 2, "names": ["ship", "boat"],
            "train": "images/train", "val": "images/val",
        })
    )
    train_list = write_oversampled_list(["x", "y"], ["y"], tmp_path / "train.txt", repeats=2)
    out = write_hard_data_config(src, train_list, tmp_path / "hard.yaml")

    original = yaml.safe_load(src.read_text())
    # The generated comment header is ignored by the YAML loader, so the file stays loadable
    # with the same call `load_data_config` makes.
    produced = yaml.safe_load(out.read_text())
    assert produced["train"] == str(train_list.resolve())
    assert produced["val"] == original["val"], "the evaluation protocol must not move"
    assert produced["nc"] == original["nc"]
    assert produced["names"] == original["names"]
