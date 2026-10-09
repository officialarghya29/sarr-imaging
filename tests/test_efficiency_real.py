"""Advanced efficiency tests: timing on **real** images, not random noise.

`tests/test_efficiency.py` pins the profiling contract. This file adds the check that
matters for a model whose first layer is *data-dependent*: the CFAR front end takes
``log(x + eps)`` and a local mean, and on zero-mean Gaussian noise most of the image
clamps to ``eps``, so a latency taken on ``torch.randn`` times the **degenerate** branch
of the front end rather than the branch a real SAR image hits. A cost number is only
meaningful alongside the input distribution it was measured on, so the profiler now
records that distribution, and these tests hold it to it.

The real-image checks are skipped when the HRSID subset is not present, because the
default suite must run with no dataset download. The skip is explicit and named, so an
absent dataset reads as "not run here", never as a pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import saryolo  # noqa: F401  (registers custom layers with ultralytics)
from saryolo.evaluation.efficiency import (
    load_image_batch,
    measure_latency,
    profile_model,
)
from saryolo.nn.arch import VARIANTS, build_yaml_dict
from saryolo.nn.model import SARYOLODetectionModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "configs" / "datasets" / "hrsid_real.yaml"
VAL_IMAGES = REPO_ROOT / "datasets" / "processed" / "hrsid_real" / "images" / "val"
CFAR_WEIGHTS = REPO_ROOT / "results" / "runs" / "REAL-004" / "weights" / "best.pt"

needs_data = pytest.mark.skipif(
    not (DATA_YAML.exists() and VAL_IMAGES.is_dir()),
    reason="HRSID subset not present; real-input efficiency check not run here",
)
needs_cfar = pytest.mark.skipif(
    not CFAR_WEIGHTS.exists(),
    reason="no trained CFAR checkpoint present; real-input profile not run here",
)


def _cfar_model():
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["cfar_n"]), ch=3, nc=1, verbose=False)
    model.eval()
    return model


# --------------------------------------------------------------------- the loader itself
@needs_data
def test_the_real_image_batch_loader_returns_normalised_three_channel_images():
    """Shape, dtype and range must match what the front end and the backbone expect."""
    x = load_image_batch(VAL_IMAGES, imgsz=64, batch=4)
    assert tuple(x.shape) == (4, 3, 64, 64)
    assert x.dtype == torch.float32
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    # A real batch is not a constant: a loader that returned zeros would look valid but
    # would time the degenerate branch this file exists to avoid.
    assert float(x.max()) > 0.0
    assert float(x.std()) > 0.0


def test_an_empty_image_directory_is_refused_rather_than_timed_on_noise(tmp_path):
    """With no images the loader must raise, not silently fall back to a synthetic batch."""
    with pytest.raises(FileNotFoundError, match="no images"):
        load_image_batch(tmp_path / "empty", imgsz=32, batch=2)


def test_the_loader_replicates_one_channel_and_fills_a_batch_from_few_files(tmp_path):
    """Two loader contracts the dataset-backed check cannot pin without a download.

    A SAR chip is one measured channel, so the loader replicates it to the three the backbone
    takes -- if it instead fed different content per channel, the timing would still "work"
    while timing a different input than the one the model sees. And a batch larger than the
    number of files must reuse images rather than leave zero rows, because a zero row is
    exactly the degenerate first-layer branch this file exists to avoid. The same reasoning
    makes a file that will not decode a hard error: skipping it would leave a zero row in a
    batch labelled "real".
    """
    import cv2
    import numpy as np

    for i, side in enumerate((24, 40)):
        cv2.imwrite(str(tmp_path / f"chip_{i}.png"), np.full((side, side), 60 + 60 * i, np.uint8))

    x = load_image_batch(tmp_path, imgsz=32, batch=10)
    assert tuple(x.shape) == (10, 3, 32, 32) and x.dtype == torch.float32
    # One measured channel, replicated: the three channels must be identical, not different.
    assert torch.equal(x[:, 0], x[:, 1]) and torch.equal(x[:, 1], x[:, 2])
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    # Fewer files than the batch: every slot carries a real image, none is left blank.
    assert all(float(x[i].max()) > 0.0 for i in range(10)), "a batch slot was left blank"
    # The bytes are normalised by 255 and the distinct source files are actually distinct.
    assert x[0].max() != x[1].max(), "the loader repeated one image for every slot"
    assert 0.45 < float(x[1].max()) < 0.49, "the image was not normalised to [0, 1]"

    # A file that will not decode must raise, not leave a zero row in a batch labelled "real".
    (tmp_path / "zz_broken.png").write_bytes(b"not a real image")
    with pytest.raises(ValueError, match="could not decode"):
        load_image_batch(tmp_path, imgsz=32, batch=10)


def test_a_single_run_reports_no_spread_and_a_multi_run_reports_its_spread():
    """A cost number is only trustworthy if the profiler can say how stable it is.

    One timed block cannot measure drift between blocks, so it must *not* report a spread
    of zero -- that would read as perfect stability rather than as "not measured here".
    More than one block must report the median with its min/max, and the median must be
    inside that range.
    """
    model = _cfar_model()
    single = measure_latency(model, imgsz=32, warmup=1, repeats=2)
    assert "latency_runs" not in single
    assert "latency_ms_spread_pct" not in single

    multi = measure_latency(model, imgsz=32, warmup=1, repeats=2, runs=3)
    assert multi["latency_runs"] == 3
    assert multi["latency_ms_min"] <= multi["latency_ms"] <= multi["latency_ms_max"]
    assert multi["latency_ms_spread_pct"] >= 0.0
    # The FPS/timing identity must hold for the *reported* median, not the last block.
    assert multi["fps"] == pytest.approx(1000.0 / multi["latency_per_image_ms"], rel=1e-3)


def test_the_reported_spread_is_the_block_range_over_the_reported_median():
    """The spread is the number a reader compares a claimed saving against, so pin its formula.

    ``latency_ms_spread_pct`` must be the min-to-max range of the timed blocks divided by the
    *reported median* -- the same quantity a reader subtracts a saving from. A spread taken
    against the mean, or against the last block, would understate or overstate the noise and
    quietly change whether a measured saving is separable from it, which is exactly the
    judgement the 320 px / 640 px efficiency result turns on.
    """
    model = _cfar_model()
    row = measure_latency(model, imgsz=32, warmup=1, repeats=3, runs=4)
    assert row["latency_runs"] == 4
    assert row["latency_ms_min"] <= row["latency_ms"] <= row["latency_ms_max"]
    expected = (row["latency_ms_max"] - row["latency_ms_min"]) / row["latency_ms"] * 100.0
    # The median is rounded to 3 dp and the spread to 1 dp, so allow exactly that rounding.
    assert row["latency_ms_spread_pct"] == pytest.approx(expected, abs=0.15), (row, expected)


def test_measure_latency_rejects_a_batch_that_is_not_rgb_shaped():
    """A malformed batch must fail loudly; a wrong shape would be timed as a wrong model."""
    model = _cfar_model()
    with pytest.raises(ValueError, match="B, 3, H, W"):
        measure_latency(model, imgsz=32, warmup=1, repeats=1, inputs=torch.zeros(2, 1, 32, 32))


# --------------------------------------------------------------- the provenance contract
def test_the_profile_says_which_input_distribution_it_timed():
    """Every latency number must carry its input source, even when it is synthetic."""
    model = _cfar_model()
    row = measure_latency(model, imgsz=32, warmup=1, repeats=1)
    assert row["latency_source"] == "synthetic"
    assert row["latency_input_max"] > 0.0
    # ``torch.randn`` is unbounded; the recorded range must reflect that, which is exactly
    # the reason a random batch is the wrong distribution for this model.
    assert row["latency_input_min"] < 0.0


@needs_data
def test_a_real_batch_is_labelled_real_and_carries_a_plausible_range():
    """A real SAR batch is bounded to ``[0, 1]`` and must be labelled as such."""
    model = _cfar_model()
    x = load_image_batch(VAL_IMAGES, imgsz=64, batch=2)
    row = measure_latency(model, imgsz=64, warmup=1, repeats=2, inputs=x, source="real")
    assert row["latency_source"] == "real"
    assert 0.0 <= row["latency_input_min"] <= row["latency_input_max"] <= 1.0
    assert row["latency_batch"] == 2, "the batch size must come from the batch, not the argument"
    assert row["fps"] == pytest.approx(1000.0 / row["latency_per_image_ms"], rel=1e-3)


@needs_data
def test_real_and_synthetic_timings_are_both_finite_and_the_same_order():
    """Timing on a real batch must not change the *magnitude* of the cost by an order.

    The point of real inputs is correctness (which branch is exercised), not a different
    cost structure; if the two disagreed by orders of magnitude, one of them would be
    timing something other than a forward pass. The bound is deliberately loose: CPU
    timing is noisy and the two batches differ.
    """
    model = _cfar_model()
    real = measure_latency(
        model, imgsz=64, warmup=2, repeats=5,
        inputs=load_image_batch(VAL_IMAGES, imgsz=64, batch=2), source="real",
    )
    synthetic = measure_latency(model, imgsz=64, warmup=2, repeats=5, batch=2, source="synthetic")
    for row in (real, synthetic):
        assert row["latency_ms"] > 0 and row["fps"] > 0
    ratio = real["latency_ms"] / synthetic["latency_ms"]
    assert 0.2 < ratio < 5.0, f"real/synthetic latency ratio {ratio:.2f} is implausible"


@needs_cfar
@needs_data
def test_profiling_a_trained_checkpoint_on_real_images_records_the_source():
    """The end-to-end path: a real checkpoint, a real batch, a profile that says so."""
    profile = profile_model(
        CFAR_WEIGHTS, imgsz=64, data_yaml=DATA_YAML, batch=2,
    )
    assert "latency_error" not in profile, profile.get("latency_error")
    assert profile["latency_source"] == "real"
    assert 0.0 <= profile["latency_input_min"] <= profile["latency_input_max"] <= 1.0
    assert profile["latency_batch"] == 2
    assert profile["params"] > 0
