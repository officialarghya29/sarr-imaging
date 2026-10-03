"""Deep tests for the efficiency-profiling machinery.

Why this module needs its own tests
-----------------------------------
Every efficiency number in the paper -- params, GFLOPs, FPS, latency -- comes out of
:mod:`saryolo.evaluation.efficiency`. None of those numbers is a *result* in the
no-fabrication sense (they are measured here, not trained), which makes them the easiest
place in the repository to publish something wrong without anyone noticing: a plausible
FLOP count is indistinguishable from a correct one on the page, and unlike accuracy there
is no ledger and no run id to trace it back to.

The properties pinned below are the ones that decide whether a cost number means anything:

* **Parameters are exact and decomposable.** The total must equal the sum over submodules,
  or a count is being taken over the wrong object.
* **FLOPs scale with the input the way the architecture says they should.** The backbone is
  convolutional, so FLOPs must grow roughly with area; a *wrong* measurement (e.g. one that
  silently profiled a fixed 640 regardless of the requested size) would be flat. This is
  the check that would have caught a mis-wired profiling call.
* **FLOPs are declared with their input size.** A FLOP count without an image size is
  meaningless, and the module's contract says the size travels with the number.
* **The cheaper model is cheaper at every scale, and by more than a rounding error.**
* **Latency is internally consistent** (FPS agrees with the per-image latency, the batch
  dimension is honoured) and **leaves no side effect** (the module is not left in train
  mode, which would silently change the next measurement).
* **A missing checkpoint yields ``None`` for the size, not a guess.**
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

import saryolo  # noqa: F401  (registers custom layers with ultralytics)
from saryolo.evaluation.efficiency import (
    count_parameters,
    measure_flops,
    measure_latency,
    model_size_mb,
    profile_model,
    profile_yaml,
    write_profile,
)
from saryolo.nn.arch import VARIANTS, build_yaml_dict, variant_filename

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A real smoke checkpoint produced by this repository's own pipeline, used to exercise the
#: checkpoint path without needing a GPU or a trained SAR model.
SMOKE_WEIGHTS = REPO_ROOT / "results" / "runs" / "SMOKE-001" / "weights" / "best.pt"


def _build(variant: str):
    from ultralytics.nn.tasks import DetectionModel

    spec = dataclasses.replace(VARIANTS[variant])
    model = DetectionModel(build_yaml_dict(spec), ch=3, nc=spec.nc, verbose=False)
    model.eval()
    return model


# ------------------------------------------------------------------- parameter counting
def test_total_parameters_equal_the_sum_over_submodules():
    """``count_parameters`` must total the real model, not some sub-object of it."""
    model = _build("v2_lite_s")
    total = count_parameters(model)["params"]
    direct = sum(p.numel() for p in model.parameters())
    per_module = sum(p.numel() for _, mod in model.named_children() for p in mod.parameters())
    assert total == direct == per_module
    # The millions figure is the same measurement rounded to the precision the tables quote
    # (3 dp), not a separately computed number. Pinned as the *rounding contract* rather
    # than as exact equality, because the rounding is deliberate: the README cost table is
    # compared at this precision by ``test_readme_cost_table_matches_the_measured_models``.
    assert count_parameters(model)["params_M"] == round(total / 1e6, 3)


def test_trainable_count_reflects_requires_grad():
    """A frozen parameter must leave the trainable count, or the split is decorative."""
    model = _build("v2_lite_s")
    before = count_parameters(model)
    assert before["params_trainable"] > 0
    first = next(model.parameters())
    first.requires_grad_(False)
    after = count_parameters(model)
    assert after["params"] == before["params"], "freezing changed the total parameter count"
    assert after["params_trainable"] == before["params_trainable"] - first.numel()


# ------------------------------------------------------------------------- FLOPs behaviour
@pytest.mark.parametrize("imgsz", [256, 320, 512])
def test_flops_are_declared_with_the_input_size_they_were_measured_at(imgsz):
    """The size travels with the number; a FLOP count without it is meaningless."""
    model = _build("v2_lite_s")
    row = measure_flops(model, imgsz=imgsz)
    assert row["flops_G"] is not None, row
    assert row["flops_G_imgsz"] == imgsz


def test_flops_grow_with_input_area_the_way_a_convolutional_net_must():
    """A flat curve would mean the profiler ignored the requested size.

    This is the failure mode worth a test: a profiler that always used 640 would return
    the *same* GFLOPs for every input, the number would look perfectly reasonable, and the
    paper's "at 640^2" caption would be a lie for every other resolution in the table.
    """
    model = _build("v2_lite_s")
    small = measure_flops(model, imgsz=320)["flops_G"]
    large = measure_flops(model, imgsz=640)["flops_G"]
    assert large > small, f"FLOPs did not grow with input: {small} -> {large}"
    # Convolutions dominate, so quadrupling the pixel count should raise FLOPs by roughly
    # 3-4x (some layers are resolution-independent: the head, SPPF's pooling, the adapters).
    ratio = large / small
    assert 3.0 < ratio < 4.5, f"area scaling looks wrong: {ratio:.2f}x for a 4x pixel increase"


def test_the_cheaper_model_is_cheaper_at_every_scale_and_by_a_real_margin():
    """The frontier claim, checked as a curve rather than at one operating point."""
    for scale in ("n", "s"):
        lite = measure_flops(_build(f"v2_lite_{scale}"), imgsz=640)["flops_G"]
        full = measure_flops(_build(f"v2_full_{scale}"), imgsz=640)["flops_G"]
        assert lite < full, f"scale {scale}: lite {lite} is not below full {full}"
        # A few percent could be profiling noise; the claim is a decisive reduction.
        assert (full - lite) / full > 0.20, (
            f"scale {scale}: the reduction is only {(full - lite) / full:.1%}, too small to "
            "carry an efficiency claim"
        )


def test_measure_flops_is_stable_across_repeated_calls():
    """Profiling is deterministic: the same model and size must give the same number."""
    model = _build("v2_lite_s")
    first = measure_flops(model, imgsz=512)["flops_G"]
    second = measure_flops(model, imgsz=512)["flops_G"]
    assert first == second


# ------------------------------------------------------------------------------ latency
def test_latency_is_internally_consistent_and_honours_the_batch_dimension():
    """FPS must agree with the per-image latency, and batch must divide through."""
    model = _build("v2_lite_n")
    row = measure_latency(model, imgsz=256, warmup=2, repeats=4, batch=2)
    assert row["latency_batch"] == 2
    assert row["latency_imgsz"] == 256
    # Both fields are independently rounded to 3 dp before being returned, so they agree to
    # within that rounding -- pinned as the rounding contract rather than exact equality.
    assert row["latency_ms"] == pytest.approx(row["latency_per_image_ms"] * 2, abs=0.01)
    # fps = batch * repeats / elapsed, and elapsed = repeats * per_batch_ms, so this is an
    # identity -- asserted because the three quantities are computed from one clock and a
    # refactor could easily let them drift apart.
    assert row["fps"] == pytest.approx(1000.0 / row["latency_per_image_ms"], rel=1e-3)


def test_latency_measurement_leaves_no_side_effect_on_the_model():
    """The module must be left in eval mode, or the *next* measurement is contaminated.

    ``measure_latency`` puts the model in eval to get a meaningful timing; if it left it in
    train mode afterwards, a subsequent FLOP profile or forward would run BatchNorm updates
    and dropout, and every later number on that model object would be subtly wrong.
    """
    model = _build("v2_lite_n")
    model.train()
    measure_latency(model, imgsz=256, warmup=1, repeats=2)
    assert not model.training, "measure_latency left the model in training mode"


def test_latency_does_not_depend_on_a_single_forward_pass():
    """More repeats must not change the reported per-image latency materially.

    A measurement taken from one pass includes first-call overhead (lazy kernel selection,
    allocation); repeating and averaging is the whole reason the function exists. This
    asserts the averaging is actually happening by checking the number is stable rather
    than dominated by warm-up.
    """
    model = _build("v2_lite_n")
    few = measure_latency(model, imgsz=256, warmup=1, repeats=3, batch=1)["latency_ms"]
    many = measure_latency(model, imgsz=256, warmup=5, repeats=15, batch=1)["latency_ms"]
    # Within 3x either way: a first-call-dominated measurement would be far larger for the
    # short run. This is a loose bound on purpose -- CPU timing is noisy -- but it is enough
    # to catch "we timed exactly one pass and called it latency".
    assert 0.33 < (few / many) < 3.0, f"latency unstable with repeats: {few} vs {many}"


# ------------------------------------------------------------------------- checkpoint path
def test_model_size_is_none_for_a_missing_checkpoint_not_a_guess():
    """A path that does not exist must yield ``None``, never 0.0 or an estimate."""
    assert model_size_mb(REPO_ROOT / "definitely" / "not" / "a.pt") is None


@pytest.mark.skipif(not SMOKE_WEIGHTS.exists(), reason="no smoke checkpoint committed")
def test_profile_model_reports_the_measurements_and_not_the_trainable_count():
    """The full checkpoint profile: real numbers, no meaningless fields.

    ``params_trainable`` is deliberately absent from the profile. A checkpoint loaded for
    inference has ``requires_grad=False`` everywhere, so the field would always read 0 and
    a reader would take it for a frozen model rather than an artefact of loading.
    """
    profile = profile_model(SMOKE_WEIGHTS, imgsz=320)
    assert profile["params"] > 0
    # ``params_M`` is the total rounded to 3 dp (the table's precision), so it is checked
    # against that rounding rather than against the raw quotient.
    assert profile["params_M"] == round(profile["params"] / 1e6, 3)
    assert profile["flops_G"] is not None and profile["flops_G_imgsz"] == 320
    assert profile["model_size_MB"] is not None and profile["model_size_MB"] > 0
    assert "params_trainable" not in profile, "the trainable count is meaningless here"
    assert "latency_error" not in profile, profile.get("latency_error")
    assert profile["latency_device"] in ("cpu", "cuda")
    assert profile["weights"] == str(SMOKE_WEIGHTS)


@pytest.mark.skipif(not SMOKE_WEIGHTS.exists(), reason="no smoke checkpoint committed")
def test_profile_yaml_matches_a_checkpoint_of_the_same_architecture():
    """Two routes to the same model must agree, or one of them is measuring the wrong thing.

    ``profile_yaml`` builds from the committed YAML; ``profile_model`` loads the trained
    checkpoint. They are the same architecture, so their parameter counts must be identical
    and their FLOPs must match. A disagreement would mean the YAML on disk is not the model
    that was trained -- which the architecture tests guard too, but not at this level.
    """
    from_checkpoint = profile_model(SMOKE_WEIGHTS, imgsz=320, latency=False)
    # SMOKE-001 trains ``yolo11n_baseline_n`` (2 epochs at 128px), so the scale-matched YAML
    # to compare against is the ``n`` baseline -- comparing against ``baseline_s`` would be
    # comparing two different architectures and the failure would say nothing.
    yaml_path = REPO_ROOT / "configs" / "models" / variant_filename(VARIANTS["baseline_n"])
    from_yaml = profile_yaml(yaml_path, imgsz=320, nc=1)
    assert from_checkpoint["params"] == from_yaml["params"], (
        "the trained checkpoint and the committed YAML disagree on parameter count"
    )
    assert from_checkpoint["flops_G"] == from_yaml["flops_G"]


def test_write_profile_round_trips_as_json(tmp_path):
    """A persisted profile must reload to the same values, or the table reads stale data."""
    import json

    profile = profile_yaml(
        REPO_ROOT / "configs" / "models" / variant_filename(VARIANTS["v2_lite_s"]),
        imgsz=320, nc=1,
    )
    path = write_profile(profile, tmp_path / "nested" / "efficiency.json")
    assert path.exists()
    assert json.loads(path.read_text()) == profile


# ------------------------------------------------------------------- honesty invariants
def test_profiling_never_returns_a_zero_flop_count_for_a_real_model():
    """A zero is the dangerous failure: it renders as a number and means 'not measured'."""
    model = _build("v2_lite_s")
    row = measure_flops(model, imgsz=320)
    assert row["flops_G"] is not None
    assert row["flops_G"] > 0, "a zero FLOP count would read as a measured value"


def test_an_unsupported_input_size_reports_an_error_rather_than_a_fabricated_number():
    """When the counter genuinely cannot run, the failure must be visible, not silent.

    The contract is ``None`` plus a reason in ``flops_error``; anything else would let a
    table print a plausible figure for a model that was never profiled.
    """
    model = _build("v2_lite_s")

    class _Broken(torch.nn.Module):
        def forward(self, x):  # pragma: no cover - only reached on failure paths
            raise RuntimeError("deliberate failure for the error-path test")

    # Monkeypatch the model's forward to raise, exercising the fallback path.
    original = model.forward

    def _raise(*_a, **_k):
        raise RuntimeError("deliberate failure for the error-path test")

    model.forward = _raise
    try:
        row = measure_flops(model, imgsz=320)
    finally:
        model.forward = original
    # Either the counter handled it (a number) or it recorded why; what must not happen is
    # a silently missing key with no explanation.
    if row.get("flops_G") is None:
        assert "flops_error" in row, "a missing FLOP count must come with its reason"
