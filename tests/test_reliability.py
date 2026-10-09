"""Reliability and resource-stability tests: the failure paths and the "does it stay stable" checks.

Why this file exists
--------------------
The rest of the suite proves the repository computes the *right* thing on the happy path. This
file holds the properties the master directive makes explicit and that "the code ran once" does
not establish:

* **A corrupt artefact is refused, not half-loaded.** Loading a truncated or garbage checkpoint
  must raise; a partial load that returns a model is the most dangerous outcome, because every
  number measured on it afterwards looks like a result.
* **An invalid numerical state stops the run.** A non-finite training loss is never something to
  continue from -- its gradient is corrupt -- so the model must refuse to return it. The guard
  is only meaningful if it *does not* fire on a healthy batch, so that is asserted first.
* **Repeated inference is deterministic and does not grow the process.** A leak that only shows
  after many passes is invisible to a single-forward test, and this project measures latency by
  repeating forwards.
* **The profiler records the practical memory peak.** With no GPU, the CPU high-water RSS is the
  memory figure a reader needs next to the latency, and its absence would be a silent gap.

Everything here runs on CPU with no dataset download, in keeping with the low-resource rule that
basic testing must stay functional on the machine the work is done on.
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path

import pytest
import torch

import saryolo  # noqa: F401  (registers custom layers with ultralytics)
from saryolo.evaluation.efficiency import measure_latency, peak_rss_mb
from saryolo.nn.arch import VARIANTS, build_yaml_dict
from saryolo.nn.model import SARYOLODetectionModel
from saryolo.training.trainer import load_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build(variant: str):
    from ultralytics.nn.tasks import DetectionModel

    spec = dataclasses.replace(VARIANTS[variant])
    model = DetectionModel(build_yaml_dict(spec), ch=3, nc=spec.nc, verbose=False)
    model.eval()
    return model


def _tensors(obj) -> list[torch.Tensor]:
    """Every tensor inside a nested forward output, so two outputs can be compared elementwise."""
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in obj:
            out.extend(_tensors(item))
        return out
    return []


# ------------------------------------------------------------------ corrupt artefact is refused
def test_a_corrupt_checkpoint_raises_instead_of_loading_partially(tmp_path):
    """Garbage in place of a checkpoint must raise; a partial load would be measured as a model.

    The failure mode this guards is not a crash on a bad path -- it is a *silent* one: a
    loader that returns a partially-initialised model, or falls back to a freshly built one,
    would let every downstream number be taken on weights that were never trained.
    """
    for name, payload in (
        ("garbage.pt", b"this is not a checkpoint"),
        ("truncated.pt", b"PK\x03\x04" + b"\x00" * 8),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        assert path.read_bytes(), "the fixture wrote nothing, so the check would be vacuous"
        with pytest.raises(Exception) as excinfo:
            load_model(str(path))
        # The raise must be a real failure with a message, not a bare SystemExit(0) or a None.
        assert str(excinfo.value) != "", f"{name}: raised without any message"


# ---------------------------------------------------------- invalid numerical state stops a run
def test_a_finite_batch_gives_a_finite_loss_before_the_guard_is_trusted():
    """Non-vacuity: the numerical guard must not fire on a healthy batch.

    A guard that raised unconditionally would pass the next test while breaking every real run,
    so the healthy case is asserted first, on the same model and batch shape.
    """
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["baseline_n"]), ch=3, nc=1, verbose=False)
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.08], [0.3, 0.7, 0.05, 0.05]]),
    }
    loss, _items = model.loss(batch)
    assert torch.isfinite(loss).all(), f"a healthy batch produced a non-finite loss: {loss}"


def test_a_non_finite_loss_is_refused_rather_than_backpropagated():
    """A NaN/inf loss must stop the step, not be handed to the optimizer.

    Once the loss is non-finite its gradient is corrupt, so continuing writes garbage into every
    weight while the run keeps printing plausible numbers. The directive is explicit that the
    model must not continue in an invalid numerical state, so this raises and names the value.
    """
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["baseline_n"]), ch=3, nc=1, verbose=False)
    poisoned = {
        "img": torch.full((2, 3, 64, 64), float("nan")),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.08], [0.3, 0.7, 0.05, 0.05]]),
    }
    with pytest.raises(RuntimeError, match="not finite"):
        model.loss(poisoned)


# ------------------------------------------------------ repeated inference stability (CPU)
def test_repeated_inference_is_deterministic_and_does_not_grow_the_process():
    """Twenty repeats must give the same output and must not grow the process's resident set.

    A single-forward check cannot see a leak, and this project's latency numbers come from
    repeating forwards -- so the stability of the repetition *is* part of the measurement's
    validity. The growth bound is generous (64 MB) because a process legitimately faults in
    code; what it catches is unbounded growth, which is what a leak looks like.
    """
    psutil = pytest.importorskip("psutil")
    model = _build("v2_lite_n")
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        reference = _tensors(model(x))
        assert reference, "the model returned no tensors, so determinism cannot be checked"
        for _ in range(3):  # let lazy allocation and cache growth settle before the baseline
            model(x)
        gc.collect()
        before = psutil.Process().memory_info().rss
        for _ in range(20):
            again = _tensors(model(x))
            assert len(again) == len(reference)
            assert all(torch.equal(a, b) for a, b in zip(again, reference, strict=True)), (
                "repeated inference on the same input is not deterministic"
            )
        gc.collect()
        after = psutil.Process().memory_info().rss
    growth = after - before
    assert growth < 64 * 1024 * 1024, f"process RSS grew {growth / 1e6:.1f} MB over 20 repeats"


# ------------------------------------------------------------------ practical memory figure
def test_the_profiler_returns_a_plausible_process_peak_memory():
    """The CPU peak-memory figure must be present, positive, and consistent with the live RSS.

    ``peak_rss_mb`` is the process high-water mark, so it can only be at or above the current
    resident set -- a value below it would mean the units or the platform branch is wrong, and
    that is exactly the kind of silent unit error a cost table cannot survive.
    """
    psutil = pytest.importorskip("psutil")
    peak = peak_rss_mb()
    if peak is None:
        pytest.skip("resource.getrusage is unavailable on this platform")
    assert peak > 0.0
    rss_mb = psutil.Process().memory_info().rss / 1024**2
    assert peak >= rss_mb * 0.9, f"peak {peak} MB is below the current RSS {rss_mb:.1f} MB"
    assert peak < 1024 * 1024, f"peak {peak} MB is implausible; the unit conversion is likely wrong"

    # And every latency row must carry it, because a cost number is quoted next to its memory.
    row = measure_latency(_build("v2_lite_n"), imgsz=64, warmup=1, repeats=2, runs=2)
    assert row["latency_peak_rss_mb"] is not None and row["latency_peak_rss_mb"] > 0.0
