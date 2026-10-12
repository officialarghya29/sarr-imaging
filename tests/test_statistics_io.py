"""Writing dataset statistics out: a JSON report and the standard figure set.

``profile_dataset`` (covered in ``tests/test_data.py``) computes the numbers; these two
functions persist them. They are checked here because the failure is quiet: a statistics
run that writes nothing still exits 0, and the missing figure is noticed only at paper
time. The naming rule matters too -- ``<name>_<split>.json`` is what lets a reader tell a
train profile from a val one.
"""

from __future__ import annotations

import json

from saryolo.data.statistics import DatasetStatistics, plot_statistics, save_statistics


def _stats(name: str = "hrsid", split: str = "val") -> DatasetStatistics:
    return DatasetStatistics(
        name=name,
        split=split,
        num_images=10,
        num_boxes=25,
        class_names=["ship"],
        boxes_per_class={"ship": 25},
        boxes_per_image={"mean": 2.5, "max": 6.0},
        size_distribution={"small": 20, "medium": 5},
        histogram_data={
            "box_width_px": [3, 5, 8, 9],
            "box_height_px": [4, 6, 7, 8],
            "aspect_ratio": [2, 4, 6],
            "object_equiv_diameter_px": [1, 2, 3, 4],
            "boxes_per_image": [1, 3, 4, 2],
            "local_contrast": [5, 4, 3],
            "density_grid": [float(i % 5) for i in range(100)],
        },
        local_contrast={"mean": 0.031},
    )


# ------------------------------------------------------------------ save_statistics

def test_save_statistics_names_the_file_by_dataset_and_split(tmp_path):
    path = save_statistics(_stats("hrsid", "val"), tmp_path)
    assert path == tmp_path / "hrsid_val.json"
    payload = json.loads(path.read_text())
    assert payload["name"] == "hrsid" and payload["split"] == "val"
    assert payload["num_boxes"] == 25


def test_save_statistics_falls_back_to_a_generic_name(tmp_path):
    """An unnamed profile still writes; the file is 'dataset_<split>.json', not '_.json'."""
    path = save_statistics(_stats("", "train"), tmp_path)
    assert path.name == "dataset_train.json"


def test_save_statistics_creates_the_directory(tmp_path):
    target = tmp_path / "nested" / "stats"
    assert save_statistics(_stats(), target).exists()


# ------------------------------------------------------------------ plot_statistics

def test_plot_statistics_writes_the_expected_figure_set(tmp_path):
    written = plot_statistics(_stats(), tmp_path)
    names = sorted(p.name for p in written)
    for expected in ("class_distribution.png", "box_width_hist.png", "aspect_ratio_hist.png",
                     "object_size_hist.png", "objects_per_image_hist.png",
                     "local_contrast_hist.png", "density_heatmap.png", "size_distribution.png"):
        assert expected in names, f"{expected} was not rendered"
    assert all(p.exists() and p.stat().st_size > 0 for p in written), "an empty figure was written"


def test_plot_statistics_skips_panels_it_has_no_data_for(tmp_path):
    """A profile with no histograms must render nothing, not eight blank axes."""
    stats = DatasetStatistics(name="empty", split="test")
    assert plot_statistics(stats, tmp_path) == []


def test_plot_statistics_returns_the_paths_it_wrote(tmp_path):
    written = plot_statistics(_stats(name="x"), tmp_path)
    assert written == sorted(written) or len(written) == len(set(written))
    assert all(p.parent == tmp_path for p in written)
