"""SAR-specific augmentation (SEC. 5).

The augmentation copies label files verbatim, with no box transformation. That is only
correct because every corruption is *appearance-only* -- it changes pixel values and never
moves a target. These tests pin that property directly, because if it ever stops holding the
failure is silent: training would proceed on boxes that no longer point at anything.

The rest pins the reproducibility claims the module's docstring makes, so the docstring cannot
drift away from the code: same seed gives byte-identical output, the manifest accounts for
every emitted file, and severities must come from the published robustness grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from saryolo.augmentation.sar import (
    SAR_AUGMENTATIONS,
    AugmentationPlan,
    build_augmented_train_split,
    default_plan,
)
from saryolo.evaluation.robustness import CORRUPTIONS, apply_corruption


def _dataset(root: Path, n: int = 3, size: int = 48) -> Path:
    """A tiny train split with a deterministic image and one box per image."""
    images, labels = root / "images" / "train", root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = (rng.random((size, size)) * 200 + 40).astype(np.uint8)
        cv2.imwrite(str(images / f"img{i}.png"), img)
        (labels / f"img{i}.txt").write_text("0 0.500000 0.500000 0.250000 0.250000\n")
    return root


# ------------------------------------------------------- the property labels depend on
def test_every_corruption_preserves_image_shape():
    """Appearance-only is the entire justification for not transforming labels."""
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64)) * 255).astype(np.uint8)
    for name, spec in CORRUPTIONS.items():
        out = apply_corruption(img, name, float(spec.severities[-1]), np.random.default_rng(1))
        assert out.shape == img.shape, f"{name} resized the image; labels would be invalid"
        assert out.dtype == np.uint8


def test_identity_corruption_is_byte_exact():
    """The clean view must be the original image, not a re-quantised copy of it."""
    rng = np.random.default_rng(3)
    img = (rng.random((32, 32)) * 255).astype(np.uint8)
    out = apply_corruption(img, "identity", 0.0, np.random.default_rng(0))
    assert np.array_equal(out, img)


def test_clean_view_is_byte_identical_to_its_source(tmp_path):
    dataset = _dataset(tmp_path / "ds")
    build_augmented_train_split(dataset / "images" / "train", tmp_path / "aug", default_plan(views=2))
    out = tmp_path / "aug" / "images" / "train"
    for src in sorted((dataset / "images" / "train").glob("*.png")):
        clean = out / f"{src.stem}__v0_clean.png"
        assert clean.exists(), f"no clean view emitted for {src.name}"
        assert np.array_equal(cv2.imread(str(clean), cv2.IMREAD_GRAYSCALE),
                              cv2.imread(str(src), cv2.IMREAD_GRAYSCALE))


def test_labels_are_copied_verbatim_for_every_view(tmp_path):
    dataset = _dataset(tmp_path / "ds")
    manifest = build_augmented_train_split(
        dataset / "images" / "train", tmp_path / "aug", AugmentationPlan(views=3, seed=1)
    )
    labels = tmp_path / "aug" / "labels" / "train"
    assert manifest["n_emitted"] > 0
    for entry in manifest["entries"]:
        stem = Path(entry["image"]).stem
        text = (labels / f"{stem}.txt").read_text()
        assert text == "0 0.500000 0.500000 0.250000 0.250000\n", (
            f"{entry['image']} did not inherit its label unchanged"
        )


# ------------------------------------------------------------------- manifest honesty
def test_every_emitted_image_has_a_label_and_a_manifest_entry(tmp_path):
    dataset = _dataset(tmp_path / "ds")
    manifest = build_augmented_train_split(dataset / "images" / "train", tmp_path / "aug", default_plan())
    images = sorted(p.name for p in (tmp_path / "aug" / "images" / "train").glob("*.png"))
    labels = sorted(p.stem for p in (tmp_path / "aug" / "labels" / "train").glob("*.txt"))
    recorded = sorted(e["image"] for e in manifest["entries"])
    assert len(images) == len(labels) == len(recorded) == manifest["n_emitted"]
    assert recorded == images
    on_disk = json.loads((tmp_path / "aug" / "manifest.json").read_text())
    assert on_disk["n_emitted"] == manifest["n_emitted"]


def test_plan_emits_one_clean_view_plus_the_requested_views(tmp_path):
    dataset = _dataset(tmp_path / "ds", n=1)
    manifest = build_augmented_train_split(
        dataset / "images" / "train", tmp_path / "aug", AugmentationPlan(views=2, seed=0)
    )
    assert len(manifest["entries"]) == 3  # clean + 2 views
    kinds = [e["corruption"] for e in manifest["entries"]]
    assert kinds[0] is None, "the clean view is first, so it is easy to find"
    assert sum(k is not None for k in kinds) == 2


def test_manifest_records_the_parameter_name_for_each_degradation(tmp_path):
    dataset = _dataset(tmp_path / "ds", n=1)
    manifest = build_augmented_train_split(
        dataset / "images" / "train", tmp_path / "aug", AugmentationPlan(views=4, seed=5)
    )
    for entry in manifest["entries"]:
        if entry["corruption"] is None:
            assert entry["severity"] is None and entry["param"] is None
        else:
            assert entry["param"] == CORRUPTIONS[entry["corruption"]].param_name
            assert entry["severity"] in CORRUPTIONS[entry["corruption"]].severities


def test_image_without_a_label_file_is_reported_not_silently_trained_on(tmp_path):
    dataset = _dataset(tmp_path / "ds", n=2)
    (dataset / "labels" / "train" / "img1.txt").unlink()
    manifest = build_augmented_train_split(dataset / "images" / "train", tmp_path / "aug", default_plan())
    assert any(s["image"] == "img1.png" and s["reason"] == "no_label_file" for s in manifest["skipped"])
    assert all("img1" not in e["image"] for e in manifest["entries"])


def test_missing_label_directory_is_not_an_empty_success(tmp_path):
    """A wrong labels path must not produce an augmentation of nothing."""
    dataset = _dataset(tmp_path / "ds", n=2)
    manifest = build_augmented_train_split(
        dataset / "images" / "train", tmp_path / "aug", default_plan(), labels_dir=tmp_path / "absent"
    )
    assert manifest["n_emitted"] == 0
    assert manifest["n_source"] == 2
    assert len(manifest["skipped"]) == 2


# --------------------------------------------------------------------- determinism
def test_same_seed_reproduces_byte_identical_output(tmp_path):
    dataset = _dataset(tmp_path / "ds")
    a = build_augmented_train_split(dataset / "images" / "train", tmp_path / "a", AugmentationPlan(views=3, seed=7))
    b = build_augmented_train_split(dataset / "images" / "train", tmp_path / "b", AugmentationPlan(views=3, seed=7))
    assert [e["image"] for e in a["entries"]] == [e["image"] for e in b["entries"]]
    assert [e["corruption"] for e in a["entries"]] == [e["corruption"] for e in b["entries"]]
    for entry in a["entries"]:
        pa = tmp_path / "a" / "images" / "train" / entry["image"]
        pb = tmp_path / "b" / "images" / "train" / entry["image"]
        assert np.array_equal(cv2.imread(str(pa), 0), cv2.imread(str(pb), 0)), entry["image"]


def test_a_different_seed_produces_a_different_draw(tmp_path):
    dataset = _dataset(tmp_path / "ds", n=6)
    a = build_augmented_train_split(dataset / "images" / "train", tmp_path / "a", AugmentationPlan(views=4, seed=1))
    b = build_augmented_train_split(dataset / "images" / "train", tmp_path / "b", AugmentationPlan(views=4, seed=2))
    assert [e["corruption"] for e in a["entries"]] != [e["corruption"] for e in b["entries"]]


# ------------------------------------------------------------------- plan validation
def test_severity_outside_the_published_grid_is_rejected():
    """Training severities must come from the robustness grid, or the two stop being comparable."""
    plan = AugmentationPlan(kinds=("speckle",), severities={"speckle": (3.3,)})
    with pytest.raises(ValueError, match="outside the published grid"):
        plan.grid("speckle")


def test_restricting_to_a_subset_of_the_grid_is_allowed():
    plan = AugmentationPlan(kinds=("speckle",), severities={"speckle": (8.0, 4.0)})
    assert plan.grid("speckle") == (8.0, 4.0)


def test_unknown_augmentation_kind_is_rejected():
    with pytest.raises(KeyError, match="Unknown corruption"):
        AugmentationPlan(kinds=("not_a_corruption",)).grid("not_a_corruption")


def test_clutter_is_available_but_not_in_the_default_plan():
    """Clutter injects unlabelled bright blobs; it must be an explicit opt-in."""
    assert "clutter" in CORRUPTIONS
    assert "clutter" not in SAR_AUGMENTATIONS
    assert "clutter" not in default_plan().kinds
    assert set(SAR_AUGMENTATIONS) <= set(CORRUPTIONS)
