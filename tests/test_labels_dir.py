"""Where a split's ground truth lives, when the dataset root carries the word "images".

Why this file exists
--------------------
A YOLO dataset pairs ``images/<split>`` with ``labels/<split>``, so every consumer derives the
labels directory from the images directory by string replacement -- six call sites across the
data, evaluation, visualisation and augmentation layers. The replacement rewrote the *first*
occurrence of the substring anywhere in the resolved path, while the rule that actually defines
the layout (and the one Ultralytics' own ``img2label_paths`` implements) is to replace the last
``images`` path *component*.

The two agree until a parent directory contains the word, which is not exotic: ``/data/images``,
``~/my_images``, a clone inside ``~/projects/images/``. There the substring form produces a
labels directory that does not exist, and nothing raises -- a missing label file is a legal YOLO
state (a background image), so the split simply reads as object-free, the ground truth is empty,
and the metrics come back ``None`` on a run that reports success. The evaluation helper that maps
an *image* to its label file had already noticed and worked around it; the directory rule is
shared now, and these tests hold the pipeline to it through the public entry points: the dataset
helper, the evaluator's ground-truth reader, the example selector, the corruption builder and the
SAR-augmentation writer.

Every layout below is deliberately pathological -- the root directory contains the word -- because
a test that only used well-behaved paths passes against either rule, which is why the defect
survived the earlier review passes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from saryolo.data.yolo import labels_dir_for, split_dirs, write_data_yaml


def _chip(path, size: int = 48) -> None:
    """A tiny but genuinely decodable image: the readers all ask for its pixel size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((size, size, 3), 110, np.uint8))


def _dataset(root, splits=("train", "val"), per_split: int = 1):
    """A complete YOLO layout under ``root``, with one labelled chip per split."""
    for split in splits:
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(per_split):
            _chip(root / "images" / split / f"chip{i}.jpg")
            (root / "labels" / split / f"chip{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


# ------------------------------------------------------------------ the rule itself

def test_the_labels_directory_replaces_the_component_and_not_the_first_substring():
    """The rule, stated once: the last ``images`` component moves and nothing else does.

    Both cases below were resolved wrongly before, and the first is the one that matters -- the
    root's own ``images`` component was rewritten while the split's was left in place.
    """
    assert labels_dir_for("data/images/hrsid/images/val") == Path("data/images/hrsid/labels/val")
    assert labels_dir_for("my_images/ds/images/train") == Path("my_images/ds/labels/train")
    assert labels_dir_for("/a/b/images") == Path("/a/b/labels")


def test_a_path_with_no_images_component_keeps_the_substring_fallback():
    """Nothing that used to resolve differently may start resolving differently.

    A path with no ``images`` component (a ``.txt`` split list, or an unconventionally named
    directory) has no component to move, so the old substring rule still applies. Kept relative on
    purpose: an absolute expectation would also depend on what the temporary directory is called,
    which is not what this test is about.
    """
    assert labels_dir_for("ds_images/train") == Path("ds_labels/train")


# ------------------------------------------------------------------ the consumers

def test_split_dirs_resolves_the_true_labels_directory_under_a_wordy_root(tmp_path):
    root = _dataset(tmp_path / "images" / "hrsid")
    data_yaml = write_data_yaml(tmp_path / "data.yaml", root, ["ship"], splits=("train", "val"))

    images_dir, labels_dir = split_dirs(data_yaml, "val")
    assert images_dir == root / "images" / "val"
    assert labels_dir == root / "labels" / "val"
    assert labels_dir.is_dir(), "the labels directory a split declares must exist"
    assert list(labels_dir.glob("*.txt")), "the split's label files must be found"


def test_ground_truth_is_read_from_a_root_that_carries_the_word(tmp_path):
    """The observable contract: evaluation finds the boxes it is supposed to score.

    This is precisely what the defect broke -- an empty ground-truth list, therefore ``None``
    metrics, and no error anywhere. Asserting the count, not just the directory, is what makes the
    test fail on the old rule for the right reason.
    """
    from saryolo.evaluation.metrics import load_yolo_ground_truth

    root = _dataset(tmp_path / "images" / "hrsid", per_split=2)
    data_yaml = write_data_yaml(tmp_path / "data.yaml", root, ["ship"], splits=("train", "val"))

    gts = load_yolo_ground_truth(data_yaml, split="val")
    assert len(gts) == 2, "two labelled val chips must yield two ground-truth boxes"
    assert {g.image for g in gts} == {"chip0", "chip1"}
    assert all(g.area > 0 for g in gts)


def test_the_label_file_for_an_image_is_the_one_beside_its_own_split(tmp_path):
    """The image-to-label helper resolves to the file that exists beside its own split."""
    from saryolo.evaluation.metrics import _label_path_for

    root = _dataset(tmp_path / "my_images" / "ds", per_split=1)
    label = _label_path_for(root / "images" / "val" / "chip0.jpg")
    assert label == root / "labels" / "val" / "chip0.txt"
    assert label.exists()


def test_the_example_selector_finds_chips_under_a_wordy_root(tmp_path):
    """Illustrative-chip selection reads the same split, so it must survive the same layout."""
    from saryolo.visualization.detections import select_examples

    root = _dataset(tmp_path / "images" / "hrsid", per_split=1)
    data_yaml = write_data_yaml(tmp_path / "data.yaml", root, ["ship"], splits=("train", "val"))

    chosen = select_examples(data_yaml, limit=1)
    assert len(chosen) == 1, "a labelled split must yield an illustrative chip"
    assert chosen[0].name == "chip0.jpg"


def test_the_corrupted_split_links_labels_from_the_images_component(tmp_path):
    """The robustness sweep must evaluate against the same ground truth it degraded."""
    from saryolo.evaluation.robustness import build_corrupted_split

    root = _dataset(tmp_path / "images" / "hrsid", splits=("val",), per_split=2)
    out = tmp_path / "corrupted"

    dest = build_corrupted_split(root / "images" / "val", out, "brightness", 0.5, seed=0)
    assert dest.is_dir() and len(list(dest.glob("*.jpg"))) == 2
    linked = sorted(p.name for p in (out / "brightness" / "0.5" / "labels" / "val").glob("*.txt"))
    assert linked == ["chip0.txt", "chip1.txt"], (
        "ground truth was not carried to the corrupted copy; the sweep would score against an "
        "empty reference"
    )


def test_the_sar_augmented_split_emits_views_with_their_labels(tmp_path):
    """The augmentation writer skips any image whose label it cannot find, so a wrong labels
    directory turns the whole augmented training split into an empty manifest."""
    from saryolo.augmentation.sar import build_augmented_train_split, default_plan

    root = _dataset(tmp_path / "images" / "hrsid", splits=("train",), per_split=2)
    aug = tmp_path / "aug"
    manifest = build_augmented_train_split(
        root / "images" / "train", aug, default_plan(views=1, seed=0)
    )
    assert manifest["n_emitted"] > 0, "every source image was skipped: the labels were not found"
    assert manifest["skipped"] == []
    for entry in manifest["entries"]:
        label = aug / "labels" / "train" / f"{Path(entry['image']).stem}.txt"
        assert label.exists(), f"{entry['image']} was emitted without ground truth"
        assert label.read_text().strip() == "0 0.5 0.5 0.2 0.2"


def test_the_two_rules_still_agree_on_a_conventional_layout(tmp_path):
    """The fix must be invisible where the old rule was already right, which is nearly everywhere."""
    root = _dataset(tmp_path / "datasets" / "processed" / "hrsid_real")
    data_yaml = write_data_yaml(tmp_path / "data.yaml", root, ["ship"], splits=("train", "val"))

    images_dir, labels_dir = split_dirs(data_yaml, "train")
    assert images_dir == root / "images" / "train"
    assert labels_dir == root / "labels" / "train"
    assert labels_dir == Path(str(images_dir).replace("images", "labels", 1)), (
        "on a conventional layout both rules must give the same directory"
    )


@pytest.mark.parametrize("split", ["train", "val"])
def test_both_splits_of_a_wordy_root_resolve(tmp_path, split):
    root = _dataset(tmp_path / "images" / "ds")
    data_yaml = write_data_yaml(tmp_path / "data.yaml", root, ["ship"], splits=("train", "val"))

    images_dir, labels_dir = split_dirs(data_yaml, split)
    assert images_dir == root / "images" / split
    assert labels_dir == root / "labels" / split
