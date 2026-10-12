"""Efficiency and scope-of-improvement deep scan for the cost machinery.

`tests/test_efficiency.py` pins the profiling contract and `tests/test_efficiency_real.py`
pins the real-input timing; this file adds the checks that decide whether a *cost number*,
and the *scope* of the mechanism's saving, mean what the table says:

* **The provenance fields describe what was measured, not what was asked for.** The one
  place this was wrong is a real defect: ``half=True`` on a CPU-only host (this repository's
  development and CI environment) reported ``latency_half = True`` while the model stayed
  ``float32``. A profile taken in float32 would therefore have been published under an
  ``fp16`` label. The fix reports the *applied* precision; the regression is pinned below.
* **A cost table can only quote the checkpoint profile it is handed.** The execution
  override that produces the dense/sparse ablation has to reach every level, record its
  own provenance, and refuse to leave the levels disagreeing -- otherwise the dense and
  sparse rows are two different models, not two amounts of arithmetic.
* **A failed measurement is a recorded failure, not a missing row.** A timing that raises
  must leave the measured fields in place and record the reason, so an un-runnable arm is
  visible in the table instead of silently absent.
* **The routing report is the execution's own partition.** ``routing_stats`` counts tiles
  with ``ceil`` while the gather pools with a floor; they agree only because execution pads
  first. On a map that is not a multiple of the tile this is checked against the batch the
  expensive path actually received -- so the number a cost table quotes is the number the
  gather produced.
* **The reporting threshold is a diagnostic, not a switch.** ``select`` must move the
  reported fraction and nothing else; a hidden execution switch there would make the
  efficiency story depend on a number that also changed the forward pass.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

import saryolo  # noqa: F401  (registers custom layers with ultralytics)
from saryolo.evaluation.efficiency import (
    measure_latency,
    merge_repeated_profiles,
    override_ssac_execution,
    profile_model,
    profile_yaml,
    ssac_routing,
)
from saryolo.nn.arch import VARIANTS, build_yaml_dict, variant_filename
from saryolo.nn.modules.ssac import ScatterSelectiveRefinement

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The one checkpoint this repository commits, which carries three SSAC blocks -- the object
#: an execution-mode ablation switches. ``results/**`` is gitignored, so this is the only
#: weight guaranteed to exist in a clean clone.
SSAC_WEIGHTS = REPO_ROOT / "weights" / "sarvo_ssac001.pt"

needs_ssac_weights = pytest.mark.skipif(
    not SSAC_WEIGHTS.exists(), reason="committed SSAC checkpoint not present"
)


def _build(variant: str):
    from ultralytics.nn.tasks import DetectionModel

    spec = dataclasses.replace(VARIANTS[variant])
    model = DetectionModel(build_yaml_dict(spec), ch=3, nc=spec.nc, verbose=False)
    model.eval()
    return model


def _tiny_val_split(root: Path, n: int = 3) -> Path:
    """A three-chip ``images/val`` split and its ``data.yaml``, for the real-input paths."""
    import cv2
    import numpy as np

    images = root / "images" / "val"
    images.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(images / f"chip_{i}.png"), np.full((32, 32), 40 + 60 * i, np.uint8))
    cfg = root / "data.yaml"
    cfg.write_text(f"path: {root}\nnc: 1\nnames:\n- ship\ntrain: images/val\nval: images/val\n")
    return cfg


# ================================================== the reported precision is the applied one
def test_the_half_precision_flag_reports_what_was_applied_not_what_was_requested():
    """A CPU host cannot run FP16, so a ``half=True`` request must not be reported as applied.

    This was the defect: the field was ``bool(half)`` -- what the caller asked for -- while the
    conversion was gated on CUDA. On a host without CUDA every ``--half`` profile therefore
    carried ``latency_half = True`` beside a ``float32`` timing, which is exactly the kind of
    unlabelled number this repository bans: the number would be compared against an FP16
    baseline as though it were one.
    """
    model = _build("v2_lite_n")
    before = next(model.parameters()).dtype
    row = measure_latency(model, imgsz=64, warmup=1, repeats=2, half=True)

    if row["latency_device"] == "cpu":
        assert row["latency_half"] is False, (
            "the profile reports half precision that this CPU run never used"
        )
        # ...and the model really was left in the precision it ran in.
        assert next(model.parameters()).dtype == before == torch.float32
    else:  # pragma: no cover - the development host has no CUDA
        assert row["latency_half"] is True
        assert next(model.parameters()).dtype == torch.float16

    # The un-requested case is unambiguous on either host.
    assert measure_latency(model, imgsz=64, warmup=1, repeats=1)["latency_half"] is False


# ============================================ what the execution override records and refuses
@needs_ssac_weights
def test_the_profile_records_an_execution_override_and_omits_it_when_not_given():
    """``not overridden`` and ``overridden to dense`` are different facts, and both are named.

    The whole point of the execution ablation is that one checkpoint is timed two ways. If the
    profile did not record *whether* it overrode the execution, a reader could not tell a
    checkpoint's own configuration from a mode the profiler imposed on it -- and the two builds
    are parameter-identical, so nothing else distinguishes them.
    """
    base = profile_model(SSAC_WEIGHTS, imgsz=64, latency=False)
    assert not any(key.startswith("ssac_") for key in base), (
        "a profile with no execution override must not carry ssac_* provenance"
    )
    assert "params_trainable" not in base

    sparse = profile_model(
        SSAC_WEIGHTS, imgsz=64, latency=False, ssac_execution="sparse", ssac_keep=0.25, ssac_tile=16
    )
    assert sparse["ssac_blocks"] == 3
    assert sparse["ssac_execution"] == "sparse"
    assert sparse["ssac_keep"] == 0.25
    assert sparse["ssac_tile"] == 16
    assert sparse["ssac_halo"] == 2, "the halo the gather used must be recorded with the mode"
    # Switching execution must not change capacity -- otherwise the cost difference is a
    # different model rather than a different amount of arithmetic.
    assert sparse["params"] == base["params"]
    assert sparse["model_size_MB"] == base["model_size_MB"]


def test_the_execution_override_refuses_to_leave_the_levels_disagreeing(monkeypatch):
    """A partial override would compare a dense level against a sparse one.

    The guard is unreachable through the normal path (the override writes every block), which
    is exactly why it is tested with a block that *ignores* the write: a level that silently
    kept its old mode would make the ablation measure a mixture and report one label.
    """

    class _Obedient:
        def __init__(self):
            self.execution = "dense"
            self.keep = 0.5
            self.tile = 16
            self.halo = 2
            self.c1 = 8

    class _Stubborn:
        # Class attributes, and a setter that refuses every assignment: the override appears
        # to apply but the level keeps its previous mode.
        execution = "dense"
        keep = 0.5
        tile = 16
        halo = 2
        c1 = 8

        def __setattr__(self, _name, _value):  # noqa: D105 - deliberate no-op
            pass

    monkeypatch.setattr(
        "saryolo.evaluation.efficiency.ssac_blocks", lambda _model: [_Obedient(), _Stubborn()]
    )
    with pytest.raises(ValueError, match="different execution modes"):
        override_ssac_execution(object(), execution="sparse")

    # With two blocks that both accept the write, the positive path still returns provenance.
    monkeypatch.setattr(
        "saryolo.evaluation.efficiency.ssac_blocks", lambda _model: [_Obedient(), _Obedient()]
    )
    applied = override_ssac_execution(object(), execution="sparse", keep=0.25, tile=8)
    assert applied["ssac_blocks"] == 2 and applied["ssac_execution"] == "sparse"


def test_the_execution_override_refuses_a_model_with_nothing_to_override():
    """A ``--ssac-*`` flag on a stock checkpoint must raise, not silently profile a stock run."""
    with pytest.raises(ValueError, match="no ScatterSelectiveRefinement blocks"):
        override_ssac_execution(_build("baseline_n"), execution="sparse")


# ============================================================= a failed timing is recorded
@needs_ssac_weights
def test_a_failed_latency_measurement_is_recorded_and_keeps_the_measured_fields(monkeypatch):
    """An un-runnable timing must leave a visible reason, not a missing row.

    ``profile_model`` wraps the timing so a model or host that cannot be timed still yields the
    parameters, FLOPs and size it *did* measure, plus the failure. Dropping the row instead --
    or omitting the reason -- would make the arm look like one nobody ran rather than one that
    could not run here.
    """
    import saryolo.evaluation.efficiency as efficiency

    def _boom(*_args, **_kwargs):
        raise RuntimeError("deliberate failure for the error-path test")

    monkeypatch.setattr(efficiency, "measure_latency", _boom)
    profile = profile_model(SSAC_WEIGHTS, imgsz=64, latency=True)

    assert "latency_error" in profile and "deliberate failure" in profile["latency_error"]
    assert "latency_ms" not in profile, "a failed timing must not leave a fabricated latency"
    # The measurements that did not depend on the timing survive.
    assert profile["params"] > 0
    assert profile["model_size_MB"] > 0
    assert profile["weights"] == str(SSAC_WEIGHTS)


# =================================================== routing only on the batch it was timed on
@needs_ssac_weights
def test_the_profile_records_routing_only_on_the_real_batch_it_timed(tmp_path):
    """A routing report on synthetic noise would describe a distribution the model is not on.

    The mechanism's efficiency case rests on *where* the allocation concentrates, which is a
    property of real chips (the CFAR statistic the allocation is built on is undefined on
    zero-mean noise). So the report must appear only when the timing used real images, and it
    must describe the mechanism the checkpoint actually carries.
    """
    cfg = _tiny_val_split(tmp_path)

    synthetic = profile_model(SSAC_WEIGHTS, imgsz=32, latency=True, batch=2)
    assert synthetic["latency_source"] == "synthetic"
    assert "ssac_routing" not in synthetic, (
        "routing was reported for a synthetic batch, which is not the deployed distribution"
    )

    real = profile_model(SSAC_WEIGHTS, imgsz=32, data_yaml=cfg, latency=True, batch=2)
    assert real["latency_source"] == "real"
    assert "latency_error" not in real, real.get("latency_error")
    levels = real["ssac_routing"]["levels"]
    assert len(levels) == 3, "the routing report must cover every SSAC level"
    for row in levels:
        assert row["tiles"] >= 1 and 1 <= row["selected_tiles"] <= row["tiles"]
        assert 0.0 <= row["gate_selected_fraction"] <= 1.0


# ========================================= the reported tile count is the executed partition
def test_the_reported_tile_count_is_the_partition_sparse_execution_actually_runs():
    """The number a cost table quotes must be the number the gather produced.

    ``routing_stats`` counts tiles as ``ceil(h/tile) * ceil(w/tile)`` while ``_tile_scores``
    (which the *execution* uses) pools with a floor. They coincide only because execution pads
    the map up to a whole number of tiles first. On a map that is not a multiple of the tile
    this is checked directly: the batch handed to the expensive path must be
    ``images x selected_tiles``. A ``ceil``/floor divergence would make the report quote a tile
    count the gather never gathered.
    """
    torch.manual_seed(0)
    block = ScatterSelectiveRefinement(8, execution="sparse", keep=0.5, tile=16).eval()
    x = torch.randn(2, 8, 20, 20)  # 20 is deliberately not a multiple of 16

    captured: dict = {}

    def _capture(_module, args):
        captured["batch"] = int(args[0].shape[0])

    handle = block.rich.register_forward_pre_hook(_capture)
    try:
        with torch.no_grad():
            block(x)
    finally:
        handle.remove()

    stats = block.routing_stats(x)
    # ceil(20 / 16) ** 2 = 4 tiles; keep = 0.5 -> ceil(0.5 * 4) = 2 selected per image.
    assert stats["tiles"] == 4, stats
    assert stats["selected_tiles"] == 2, stats
    assert captured["batch"] == 2 * stats["selected_tiles"], (
        f"execution gathered {captured['batch']} tiles but the report says "
        f"{stats['selected_tiles']} per image"
    )


# ============================================= the reporting threshold is not an execution knob
def test_the_selected_fraction_is_a_reporting_knob_that_cannot_change_the_output():
    """``select`` reports the allocation's concentration; it must not gate the forward pass.

    The design's efficiency story reads the selected fraction, so if ``select`` also changed
    what the block computed, the quoted number would depend on a switch with a computational
    effect -- an unlabelled confound. It is asserted to move the reported fraction and nothing
    else, and to agree with the routing report's ``gate_selected_fraction``.
    """
    torch.manual_seed(0)
    block = ScatterSelectiveRefinement(8, execution="sparse", keep=0.5, tile=8).eval()
    with torch.no_grad():
        block.alpha.raw.fill_(0.5)
    x = torch.randn(2, 8, 32, 32)
    with torch.no_grad():
        g = block.gain_map(x)

    for select in (0.2, 0.5, 0.8):
        block.select = select
        expected = float((g > select).float().mean())
        assert block.selected_fraction(x) == pytest.approx(expected, abs=1e-6)
        # The report and the diagnostic must read the same tensor.
        assert block.routing_stats(x)["gate_selected_fraction"] == pytest.approx(expected, abs=1e-5)

    block.select = 0.2
    with torch.no_grad():
        first = block(x)
    block.select = 0.8
    with torch.no_grad():
        second = block(x)
    assert torch.equal(first, second), "the reporting threshold changed the forward pass"


def test_the_block_repr_states_the_execution_so_a_checkpoint_reads_sensibly():
    """A checkpoint's ``repr`` is how a reader checks which execution it was built with."""
    block = ScatterSelectiveRefinement(8, mode="fixed", execution="sparse", tile=8, keep=0.5)
    text = repr(block)
    for token in ("mode=fixed", "execution=sparse", "tile=8", "keep=0.5"):
        assert token in text, f"the block repr omits {token!r}: {text}"


# ============================================================ architecture profile on a device
def test_profile_yaml_on_an_explicit_device_measures_the_same_architecture():
    """A device override must not change the architecture, only where it is placed."""
    path = REPO_ROOT / "configs" / "models" / variant_filename(VARIANTS["v2_lite_n"])
    cpu = profile_yaml(path, imgsz=128, nc=1)
    on_device = profile_yaml(path, imgsz=128, nc=1, device="cpu")
    assert on_device["params"] == cpu["params"]
    assert on_device["flops_G"] == cpu["flops_G"]


# ================== the report describes the pass that ran, not the pass that could have run
def test_the_routing_report_states_what_the_mode_executed_not_what_it_could_have_skipped():
    """A dense timing must not be described by a sparse allocation's arithmetic.

    ``routing_stats`` answers the hypothetical -- "what would a sparse build skip here?" -- and
    that is the right question for a what-if table. It is the wrong answer for the per-level
    report attached to a *timing*, whose column means "the expensive path this pass executed".
    Dense execution runs the expensive path everywhere: no tiles, no halo, no selection. The
    defect this pins is the report carrying the settlement's fraction under a dense latency, so
    a row that paid for 1.00 printed 0.44 -- the reader then sees a saving the run never took,
    and the same checkpoint appears to execute less than a dense pass while timing the same.
    """
    model = _build("ssac_n")
    x = torch.rand(1, 3, 320, 320)
    dense = ssac_routing(model, x)["levels"]
    assert len(dense) == 3
    for row in dense:
        assert row["executed_rich_fraction"] == 1.0, (
            f"a dense pass reported {row['executed_rich_fraction']} of the expensive path "
            "executed; dense execution runs all of it"
        )
        assert row["selected_tiles"] == row["tiles"], row
        assert row["selected_tile_fraction"] == 1.0

    # ...and the same report, on the same model and batch, follows the execution switch: a
    # sparse build reports the budget it actually applied. Both halves are needed, or the fix
    # could be a constant.
    override_ssac_execution(model, execution="sparse", keep=0.25, tile=16)
    sparse = ssac_routing(model, x)["levels"]
    assert [r["tiles"] for r in sparse] == [r["tiles"] for r in dense], (
        "switching execution changed the map geometry, which no execution mode can do"
    )
    for after in sparse:
        assert after["selected_tiles"] <= after["tiles"]
        if after["tiles"] > 1:
            assert after["executed_rich_fraction"] < 1.0, after


# =========================================== one process, all plans: the matched-sweep merge
def _profile(latency: float, **extra) -> dict:
    """A minimal profile of the shape ``profile_model`` returns, for the merge contract."""
    row = {
        "latency_ms": latency,
        "latency_batch": 4,
        "latency_imgsz": 320,
        "latency_source": "real",
        "latency_device": "cpu",
        "ssac_execution": "sparse",
        "ssac_keep": 0.25,
        "ssac_routing": {"levels": [{"tiles": 9}]},
        "fps": round(1000.0 * 4 / latency, 2),
    }
    row.update(extra)
    return row


def test_repeated_cycles_merge_to_the_median_with_the_spread_across_all_of_them():
    """The merged row must report the drift across cycles, not hide it behind one cycle.

    This is the whole reason a matched sweep measures dense and sparse in one process: the
    between-process drift on this host is bigger than the effect under test, so the number that
    may be quoted is a median over interleaved cycles with the span kept beside it.
    """
    cycles = [
        _profile(210.0, latency_runs=3, latency_ms_min=208.0, latency_ms_max=212.0,
                 latency_peak_rss_mb=400.0),
        _profile(200.0, latency_runs=3, latency_ms_min=199.0, latency_ms_max=203.0,
                 latency_peak_rss_mb=420.0),
        _profile(220.0, latency_runs=3, latency_ms_min=218.0, latency_ms_max=221.0,
                 latency_peak_rss_mb=410.0),
    ]
    merged = merge_repeated_profiles(cycles)
    assert merged["latency_ms"] == 210.0, "the median of the per-cycle medians"
    assert merged["latency_ms_min"] == 199.0 and merged["latency_ms_max"] == 221.0
    assert merged["latency_runs"] == 9, "every timed block, not just the reported cycle's"
    assert merged["latency_cycles"] == 3
    assert merged["fps"] == pytest.approx(1000.0 * 4 / 210.0, abs=0.01)
    assert merged["latency_peak_rss_mb"] == 420.0, "the peak across the cycles, not the first"
    # Every non-timing field comes through unchanged, routing included.
    assert merged["ssac_routing"] == cycles[0]["ssac_routing"]


def test_a_single_cycle_is_returned_untouched_and_a_mismatch_is_refused():
    """Merging must not invent fields a single measurement never produced.

    A one-cycle sweep is still a matched sweep, so it must pass through unchanged rather than
    be reshaped into the multi-cycle form (which would report a spread of zero by construction
    and read as stability). And a merge across *different* plans -- a different budget, or a
    dense cycle mixed with a sparse one -- would report a number no run produced.
    """
    single = _profile(200.0, latency_runs=3)
    assert merge_repeated_profiles([single]) is single
    assert "latency_cycles" not in merge_repeated_profiles([single])

    with pytest.raises(ValueError, match="not the same plan"):
        merge_repeated_profiles([_profile(200.0), _profile(210.0, ssac_keep=0.5)])
    with pytest.raises(ValueError, match="not the same plan"):
        merge_repeated_profiles([_profile(200.0, ssac_execution="dense"), _profile(210.0)])
    with pytest.raises(ValueError, match="at least one profile"):
        merge_repeated_profiles([])


@needs_ssac_weights
def test_the_cli_sweep_writes_a_dense_baseline_and_the_budgets_from_one_process(tmp_path):
    """The CLI's sweep must produce a *pair*, both halves timed in the same process.

    The defect this closes is a comparison across processes: the dense baseline used to be a
    separate invocation, and two invocations of this host disagree by more than the effect the
    sweep is measuring. So a sweep now writes ``efficiency.json`` (the dense baseline) and
    ``ssac_execution_sweep.json`` (one row per budget) from one run, both marked with the cycles
    that were interleaved -- and the sparse rows carry the routing of the pass that was timed.
    """
    from saryolo.cli import main

    cfg = _tiny_val_split(tmp_path)
    out = tmp_path / "sweep"
    code = main([
        "efficiency", "--weights", str(SSAC_WEIGHTS), "--imgsz", "32", "--real",
        "--data", str(cfg), "--batch", "2", "--runs", "1", "--cycles", "2",
        "--ssac-execution", "sparse", "--ssac-keep", "0.25", "--ssac-tile", "16",
        "--out", str(out),
    ])
    assert code == 0

    dense = json.loads((out / "efficiency.json").read_text())
    sparse = json.loads((out / "ssac_execution_sweep.json").read_text())
    assert isinstance(sparse, list) and len(sparse) == 1
    for row in (dense, sparse[0]):
        assert row["latency_cycles"] == 2, "the row does not say how many cycles it merges"
        assert row["latency_runs"] == 2, "the timed blocks of both cycles must be counted"
        assert row["latency_source"] == "real"
        assert row["weights"] == str(SSAC_WEIGHTS)
    # Both halves carry the same provenance, so the pair cannot be two different checkpoints.
    assert dense["ssac_execution"] == "dense" and sparse[0]["ssac_execution"] == "sparse"
    assert sparse[0]["ssac_keep"] == 0.25
    for row in dense["ssac_routing"]["levels"]:
        assert row["executed_rich_fraction"] == 1.0
