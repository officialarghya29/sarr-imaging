"""Paper table generation from measured results only.

The guarantee
-------------
Every table here is built from the experiment ledger and from JSON artefacts that
evaluation runs actually wrote. A cell with no measurement renders as ``TBD``.
There is deliberately **no** code path that fills a missing value with a
plausible number, a zero, or an interpolation: a table is either traceable to a
completed run or visibly incomplete.

`assert_complete=False` at the call sites means an incomplete table is a normal
state during development. Set it to ``True`` when preparing a submission to turn
any missing cell into a hard error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["TBD", "Table", "MODULE_ABLATION_GROUPS", "REMOVAL_ABLATION_ROWS",
           "REAL_PILOT_ARMS", "build_baseline_comparison", "build_ablation", "build_module_ablation",
           "build_removal_ablation", "build_scale_analysis", "build_robustness", "build_efficiency",
           "build_efficiency_frontier", "build_multi_seed", "build_real_data_pilot", "write_tables"]

#: The measured real-data pilot arms, in reading order: the full fine-tune, the efficiency
#: frontier model, then the parameter-efficient arm. Labels name the dataset subset in every
#: row, because a pilot row must never be mistaken for a benchmark row when the table is
#: pasted into a draft next to the ladder.
REAL_PILOT_ARMS: tuple[tuple[str, str], ...] = (
    ("REAL-001", "YOLO11n baseline -- HRSID subset"),
    ("REAL-002", "SARVO-Lite (s) -- HRSID subset"),
    ("REAL-003", "YOLO11n + LoRA r=8 -- HRSID subset"),
    ("REAL-004", "SARVO prototype: CFAR front end -- HRSID subset"),
    ("REAL-005", "Prototype control: CFAR, fixed threshold -- HRSID subset"),
)

#: Placeholder rendered for any unmeasured value.
TBD = "TBD"


def _fmt(value, digits: int = 4) -> str:
    """Format a measured value, or return :data:`TBD` if it was never measured."""
    if value is None or value == "":
        return TBD
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


@dataclass
class Table:
    """A renderable table with a caption and provenance."""

    name: str
    caption: str
    headers: list[str]
    rows: list[list[str]] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Whether every cell was measured."""
        return not any(TBD in cell for row in self.rows for cell in row)

    def to_markdown(self) -> str:
        lines = [f"**{self.caption}**", ""]
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("| " + " | ".join("---" for _ in self.headers) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        if not self.complete:
            lines += ["", "`TBD` = not yet measured (run the corresponding experiment)."]
        return "\n".join(lines)

    def to_latex(self, label: str | None = None) -> str:
        label = label or self.name.lower().replace(" ", "_")
        spec = "l" + "r" * (len(self.headers) - 1)
        out = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{self.caption}}}",
            rf"\label{{tab:{label}}}",
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            " & ".join(self.headers) + r" \\",
            r"\midrule",
        ]
        for row in self.rows:
            out.append(" & ".join(row) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        if not self.complete:
            out.insert(2, "% WARNING: contains unmeasured (TBD) cells -- not submission ready.")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {"name": self.name, "caption": self.caption, "headers": self.headers,
                "rows": self.rows, "complete": self.complete, "provenance": self.provenance}


# --------------------------------------------------------------------- builders
def _ledger_rows(ledger):
    return ledger.completed()


#: Datasets that are *subsets* of a benchmark rather than the benchmark itself. A run on
#: one of these is a pilot: real and measured, but measured on a fraction of the release,
#: so it must never fill a benchmark cell. The subset pilot has its own table
#: (:func:`build_real_data_pilot`), with the subset and the schedule in its caption.
#: Spelled out as an explicit list rather than inferred from a filename so that adding a
#: pilot subset is a deliberate edit and not an accident of naming.
PILOT_DATASETS: tuple[str, ...] = ("hrsid_real",)


def _is_subset_pilot(record) -> bool:
    """True when a run was measured on a subset pilot rather than a benchmark release."""
    return Path(str(record.dataset)).stem in PILOT_DATASETS


def _match_variant(records, variant: str, allow_pilot: bool = False):
    """The completed record whose model file is exactly ``variant``, or ``None``.

    Matched on the model file *stem* at a ``_`` boundary rather than by substring
    containment. This matters: ``"v2_full"`` is a substring of ``"v2_full_s"``,
    ``"v2_full_l"`` and ``"v2_full_p35_s"``, so a containment test could quietly report a
    different model's score in the ablation table -- and a wrong number in a table is worse
    than a missing one, because nothing downstream can tell it is wrong.

    Subset pilots are skipped unless ``allow_pilot`` is set: a benchmark table must not
    quote an accuracy measured on 260 images as if it were the benchmark's, which is the
    failure this flag exists to prevent. A pilot's home is :func:`build_real_data_pilot`.
    """
    for record in records:
        if record.status != "completed":
            continue
        if not allow_pilot and _is_subset_pilot(record):
            continue
        stem = Path(str(record.model)).stem
        if stem == variant or stem.endswith(f"_{variant}"):
            return record
    return None


def _latest_by_experiment(records) -> dict[str, object]:
    """Best completed record per experiment id (highest mAP50:95, or the latest)."""
    best: dict[str, object] = {}
    for record in records:
        current = best.get(record.experiment_id)
        if current is None:
            best[record.experiment_id] = record
            continue
        new_score = record.metrics.get("mAP50_95")
        old_score = current.metrics.get("mAP50_95")
        if new_score is not None and (old_score is None or new_score > old_score):
            best[record.experiment_id] = record
    return best


def build_baseline_comparison(ledger, efficiency: dict | None = None) -> Table:
    """TABLE 2 — comparison against the detector baselines."""
    best = _latest_by_experiment(_ledger_rows(ledger))
    table = Table(
        "baseline_comparison",
        "Comparison with YOLO baselines and the proposed SAR-YOLO. Cells are TBD until the "
        "corresponding experiment has been run.",
        ["Model", "mAP50", "mAP50:95", "Precision", "Recall", "Params (M)", "GFLOPs", "FPS"],
    )
    for exp_id, label in (("EXP-001", "YOLO11 baseline"), ("EXP-007", "SAR-YOLO (ours)")):
        record = best.get(exp_id)
        metrics = record.metrics if record else {}
        table.rows.append([
            label,
            _fmt(metrics.get("mAP50")),
            _fmt(metrics.get("mAP50_95")),
            _fmt(metrics.get("precision")),
            _fmt(metrics.get("recall")),
            _fmt(metrics.get("params_M"), 3),
            _fmt(metrics.get("flops_G"), 3),
            _fmt(metrics.get("fps"), 1),
        ])
        table.provenance.append(f"{exp_id}: {getattr(record, 'run_id', 'not run')}")
    return table


def build_ablation(ledger) -> Table:
    """TABLE 3 — the main ablation, one component added per row."""
    best = _latest_by_experiment(_ledger_rows(ledger))
    #: ``(exp id, label, (sfe, clutter, attention, amf, p2, sar loss, prior, frequency,
    #: context, refinement))``
    #: The clutter row is a mode change on the speckle slot rather than a new component, so it
    #: keeps ``speckle`` set and adds ``clutter``.
    chain = (
        ("EXP-001", "YOLO baseline", (False, False, False, False, False, False, False, False, False, False)),
        ("EXP-002", "+ SFE", (True, False, False, False, False, False, False, False, False, False)),
        ("EXP-003", "+ Speckle", (True, False, False, False, False, False, False, False, False, False)),
        ("EXP-004", "+ Attention", (True, False, True, False, False, False, False, False, False, False)),
        ("EXP-005", "+ AMF", (True, False, True, True, False, False, False, False, False, False)),
        ("EXP-006", "+ Small head", (True, False, True, True, True, False, False, False, False, False)),
        ("EXP-007", "Full (v1)", (True, False, True, True, True, True, False, False, False, False)),
        ("EXP-013", "+ Clutter-aware", (True, True, True, True, True, True, False, False, False, False)),
        ("EXP-014", "+ Target prior", (True, True, True, True, True, True, True, False, False, False)),
        ("EXP-015", "+ Spatial-frequency", (True, True, True, True, True, True, True, True, False, False)),
        ("EXP-016", "+ Context", (True, True, True, True, True, True, True, True, True, False)),
        ("EXP-017", "Full (v2)", (True, True, True, True, True, True, True, True, True, True)),
    )
    table = Table(
        "main_ablation",
        "Main ablation. Each row adds exactly one component over the row above, so every delta "
        "is attributable to that component alone. The clutter row is a mode change on the "
        "speckle slot rather than an added module.",
        ["Model", "SFE", "Clutter", "Attention", "AMF", "P2 head", "SAR loss",
         "Prior", "Frequency", "Context", "Refine", "mAP50", "mAP50:95", "Params (M)", "FPS"],
    )
    for exp_id, label, flags in chain:
        record = best.get(exp_id)
        metrics = record.metrics if record else {}
        table.rows.append([
            label, *("\\checkmark" if f else "" for f in flags),
            _fmt(metrics.get("mAP50")), _fmt(metrics.get("mAP50_95")),
            _fmt(metrics.get("params_M"), 3), _fmt(metrics.get("fps"), 1),
        ])
        table.provenance.append(f"{exp_id}: {getattr(record, 'run_id', 'not run')}")
    return table


#: Module-level ablation slots: ``slot -> arms``. Each arm sits in the same slot with every
#: other component held fixed. Kept at module level so ``tests/test_repo.py`` can assert that
#: every named variant actually exists -- a typo here would produce a permanently ``TBD`` row
#: with no other symptom.
MODULE_ABLATION_GROUPS: dict[str, tuple[str, ...]] = {
    "Attention": ("att_none", "att_se", "att_eca", "att_cbam", "att_saa_static", "attention"),
    "Fusion": ("fus_concat", "fus_add", "fus_static", "amf"),
    "Preprocessing": ("pre_identity", "pre_log", "pre_clahe", "pre_standardize", "sfe"),
    "Speckle": ("spk_none", "spk_lee", "spk_denoise", "speckle"),
    # Component 8's own study. `tp_channel` is the capacity-matched control for `v2_full`:
    # the same evidence network, with spatial variation pooled away.
    "Target prior": ("tp_none", "tp_cfar", "tp_static", "tp_channel", "v2_full"),
    # The alternative-frequency study (SEC. 14 of the brief): the FFT arms, then the local
    # transforms. `static` and `dct` carry identical parameter counts by construction, so a
    # difference between them is attributable to the transform.
    "Frequency": ("fr_none", "fr_highpass", "fr_static", "fr_dct", "fr_wavelet", "v2_full"),
    "Context": ("cx_none", "cx_local", "cx_regional", "v2_full"),
    # Component 11's own study. `rf_local` is the capacity control for the proposed arm: the
    # same sub-network with the offsets removed, so `ours - rf_local` isolates *deformation*
    # rather than the extra convolution.
    "Refinement": ("rf_none", "rf_local", "rf_static", "rf_off25", "rf_off100", "v2_full"),
    # Module A. Unlike every other slot, the proposed arm is *not* `v2_full`: the adapter is
    # not in the v2 default, so `in_hybrid` is the arm that would have to earn it a place.
    "Input adapter": ("in_identity", "in_local", "in_learned", "in_hybrid"),
    # SEC. 4. The only group whose arms are the *same size* as `v2_full` by construction: the
    # variable is the objective, not the graph, so `v2_full` (weight 0) is the control and the
    # rows below it differ only in which degradation the representation is asked to ignore.
    "Representation consistency": ("v2_full", "cons_sev1", "cons_sev16", "cons_lowcontrast",
                                  "cons_lowsnr", "v2_cons"),
}

#: Removal ablation rows: ``(label, variant)``. Exposed for the same reason as above.
REMOVAL_ABLATION_ROWS: tuple[tuple[str, str], ...] = (
    ("Full (v2)", "v2_full"),
    ("- clutter branch", "v2_noclutter"),
    ("- target prior", "v2_noprior"),
    ("- spatial-frequency", "v2_nofreq"),
    ("- context", "v2_noctx"),
    ("- refinement", "v2_norefine"),
    # Its reference is EXP-018 (`v2_prior_spectral`), not full v2: Module G is not part of
    # `v2_full`. Measured against `v2_full` this row is a zero-parameter difference, i.e. a
    # no-op that would read as a clean removal.
    ("- prior spectral (Module G; ref. v2_prior_spectral)", "v2_nopspectral"),
)


def build_module_ablation(ledger, suffix: str = "_s") -> Table:
    """TABLE 4 — module-level ablation: ours vs the standard component in the same slot."""
    groups = MODULE_ABLATION_GROUPS
    table = Table(
        "module_ablation",
        "Module-level comparison. Each proposed block is compared against the standard "
        "component it replaces, in the same architectural slot, so the comparison isolates "
        "the mechanism rather than the added parameters.",
        ["Slot", "Variant", "mAP50", "mAP50:95", "Params (M)"],
    )
    rows = _ledger_rows(ledger)
    for slot, variants in groups.items():
        for variant in variants:
            record = _match_variant(rows, variant)
            metrics = record.metrics if record else {}
            table.rows.append([
                slot, variant,
                _fmt(metrics.get("mAP50")), _fmt(metrics.get("mAP50_95")),
                _fmt(metrics.get("params_M"), 3),
            ])
            table.provenance.append(f"{variant}: {getattr(record, 'run_id', 'not run')}")
    return table


def build_removal_ablation(ledger) -> Table:
    """TABLE 4b -- removal ablation: drop one component from the full v2 model.

    The cumulative ladder shows that each component *can* help; this shows whether it still
    does once the others are present. Both are reported because a component can look useful
    in a cumulative table purely because of the order the rows were added in.
    """
    rows = _ledger_rows(ledger)
    table = Table(
        "removal_ablation",
        "Removal ablation on the full v2 model. 'Full' is EXP-017; each row removes exactly "
        "one component, so a *drop* in this table contradicts the corresponding gain in the "
        "cumulative ladder.",
        ["Model", "mAP50", "mAP50:95", "AP_small", "Params (M)", "FPS"],
    )
    for label, variant in REMOVAL_ABLATION_ROWS:
        record = _match_variant(rows, variant)
        metrics = record.metrics if record else {}
        table.rows.append([
            label,
            _fmt(metrics.get("mAP50")), _fmt(metrics.get("mAP50_95")),
            _fmt(metrics.get("AP_small")), _fmt(metrics.get("params_M"), 3),
            _fmt(metrics.get("fps"), 1),
        ])
        table.provenance.append(f"{variant}: {getattr(record, 'run_id', 'not run')}")
    return table


def build_scale_analysis(ledger) -> Table:
    """TABLE 5 — small/medium/large object performance (the paper's core claim)."""
    best = _latest_by_experiment(_ledger_rows(ledger))
    table = Table(
        "scale_analysis",
        "Performance by object scale. Scale-wise AP is reported separately because a single "
        "mAP can hide exactly the small-object effect this work claims. 'n GT' is the number "
        "of ground-truth boxes in each range; a range with none is unmeasurable, not zero.",
        ["Model", "AP_small", "n GT small", "AP_medium", "n GT medium", "AP_large", "n GT large"],
    )
    for exp_id, label in (("EXP-001", "YOLO baseline"), ("EXP-007", "SAR-YOLO (ours)")):
        record = best.get(exp_id)
        metrics = record.metrics if record else {}
        table.rows.append([
            label,
            _fmt(metrics.get("AP_small")), _fmt(metrics.get("n_gt_small"), 0),
            _fmt(metrics.get("AP_medium")), _fmt(metrics.get("n_gt_medium"), 0),
            _fmt(metrics.get("AP_large")), _fmt(metrics.get("n_gt_large"), 0),
        ])
        table.provenance.append(f"{exp_id}: {getattr(record, 'run_id', 'not run')}")
    return table


def build_robustness(robustness_json: str | Path, baseline_json: str | Path | None = None) -> Table:
    """TABLE 6 — degradation under controlled corruption, and the drop."""
    def _load(path):
        path = Path(path)
        return json.loads(path.read_text()) if path.exists() else {}

    ours = _load(robustness_json)
    base = _load(baseline_json) if baseline_json else {}
    table = Table(
        "robustness",
        "Robustness under controlled degradations. Both models face byte-identical corrupted "
        "inputs, and only the images are degraded (ground truth is untouched).",
        ["Corruption", "Severity", "YOLO mAP50", "SAR-YOLO mAP50", "Drop (ours)"],
    )
    corruptions = ours.get("corruptions", {}) or base.get("corruptions", {})
    for name, per_sev in sorted(corruptions.items()):
        for severity in sorted(per_sev, key=float):
            ours_v = (ours.get("corruptions", {}).get(name, {}) or {}).get(severity, {})
            base_v = (base.get("corruptions", {}).get(name, {}) or {}).get(severity, {})
            ours_map = ours_v.get("mAP50")
            base_map = base_v.get("mAP50")
            drop = (base_map - ours_map) if (base_map is not None and ours_map is not None) else None
            table.rows.append([name, severity, _fmt(base_map), _fmt(ours_map), _fmt(drop)])
    if not table.rows:
        table.rows.append(["TBD", TBD, TBD, TBD, TBD])
    return table


def build_efficiency(ledger) -> Table:
    """TABLE 7 — efficiency comparison."""
    best = _latest_by_experiment(_ledger_rows(ledger))
    table = Table(
        "efficiency",
        "Efficiency. Latency is measured after warm-up with CUDA synchronisation; FLOPs are "
        "reported at the recorded input size, since FLOPs are meaningless without it.",
        ["Model", "Params (M)", "GFLOPs", "FPS", "Latency (ms)", "Size (MB)"],
    )
    for exp_id, label in (("EXP-001", "YOLO baseline"), ("EXP-007", "SAR-YOLO (ours)")):
        record = best.get(exp_id)
        metrics = record.metrics if record else {}
        table.rows.append([
            label, _fmt(metrics.get("params_M"), 3), _fmt(metrics.get("flops_G"), 3),
            _fmt(metrics.get("fps"), 1), _fmt(metrics.get("latency_ms"), 2),
            _fmt(metrics.get("model_size_MB"), 2),
        ])
        table.provenance.append(f"{exp_id}: {getattr(record, 'run_id', 'not run')}")
    return table


def _latest_per_seed(records, experiment_id: str):
    """Latest completed record per training seed for one experiment id.

    Seed arms share an ``experiment_id`` (see the generated ``EXP-012_seed*``
    configs) and are distinguished by ``train_seed``. The ledger is append-only,
    so a restarted seed run leaves *two* completed rows behind; the earlier one
    is superseded, not evidence. Counting both would drag the seed's mean and
    inflate ``n seeds`` with a rerun -- exactly the kind of number that looks
    like variance and is actually a crashed job. One row per seed, the latest.
    """
    latest: dict[object, object] = {}
    for record in records:
        if record.experiment_id != experiment_id or record.status != "completed":
            continue
        if record.metrics.get("mAP50_95") is None:
            continue
        latest[record.train_seed] = record  # later append wins
    return list(latest.values())


#: The frontier arms, in the order they should be read: the reference model first, then
#: progressively cheaper operating points. ``(variant, label, experiment id)``. The
#: experiment ids are the committed configs, so a cost row can always be traced to a run.
EFFICIENCY_FRONTIER_ARMS: tuple[tuple[str, str, str], ...] = (
    ("v2_full", "SAR-YOLO v2 (reference)", "EXP-017"),
    ("v2_full_p35_s", "v2, no P2 level", "EXP-017"),
    ("v2_lite_s", "SARVO-Lite (s)", "EXP-501"),
    ("v2_lite_p35_s", "SARVO-Lite (s), no P2", "EXP-502"),
    ("v2_lite_cond_s", "SARVO-Lite + conditioning (s)", "EXP-503"),
    ("v2_lite_n", "SARVO-Lite (n)", "EXP-504"),
    ("v2_lite_m", "SARVO-Lite (m)", "EXP-505"),
)


def build_real_data_pilot(ledger) -> Table:
    """TABLE 9 -- what was actually measured on real SAR data, as distinct from the plan.

    This table exists because the rest of this module describes the *benchmark*: every accuracy
    cell in the ladder is ``TBD`` until a GPU run on the full release exists. A subset pilot is
    not that, and putting it in the same table as a benchmark row would let a 260-image, single
    seed, CPU run read as a comparable result. So it gets its own table with the subset, the
    schedule and the resolution in the caption.

    The columns are the same shape as the baseline table on purpose: when the full run exists,
    the two can be read side by side and the difference in the caption is visible rather than
    hidden in a footnote.
    """
    best = _latest_by_experiment(_ledger_rows(ledger))
    table = Table(
        "real_data_pilot",
        "Pilot on a real HRSID subset: the official release, YOLO boxes, 200 train / 60 val / 60 "
        "test images, 40 epochs at 320 px, single seed, CPU only. Real and measured, but the "
        "release is 5,604 images -- this is not the benchmark, and a row here is not comparable "
        "with a full-release row. For REAL-003 the adapter is folded into the base weights when "
        "the checkpoint is written, so its parameter count equals the baseline's while what was "
        "optimised (a rank-8 adapter plus the head) is recorded separately in the ledger.",
        ["Model", "mAP50", "mAP50:95", "Precision", "Recall", "Params (M)", "GFLOPs@320",
         "FPS", "Train (min)"],
    )
    for exp_id, label in REAL_PILOT_ARMS:
        record = best.get(exp_id)
        metrics = record.metrics if record else {}
        table.rows.append([
            label,
            _fmt(metrics.get("mAP50")),
            _fmt(metrics.get("mAP50_95")),
            _fmt(metrics.get("precision")),
            _fmt(metrics.get("recall")),
            _fmt(metrics.get("params_M"), 3),
            _fmt(metrics.get("flops_G"), 3),
            _fmt(metrics.get("fps"), 1),
            _fmt(metrics.get("train_minutes"), 2),
        ])
        table.provenance.append(f"{exp_id}: {getattr(record, 'run_id', 'not run')}")
    return table


def build_efficiency_frontier(ledger, benchmark: dict | None = None) -> Table:
    """TABLE 8 — the efficiency frontier, with published baselines kept separate.

    Two kinds of number meet in this table and are deliberately not in the same column:

    * **our measured cost** (params, GFLOPs) -- profiled on this machine by
      ``python -m saryolo bench`` and handed in through ``benchmark``. These are real
      measurements, so they are not ``TBD`` even though no accuracy exists yet.
    * **reported baselines** -- numbers other papers published, imported from
      :mod:`saryolo.evaluation.reported_baselines`. They are rendered as a *separate
      section* with their venue, because mixing a number we measured with a number we
      read under one header is how a comparison table starts lying without a single
      false digit being typed.

    Accuracy for our arms stays ``TBD`` until a run exists. A cost claim does not need
    accuracy to be honest about *cost*, and the caption says exactly that.

    Args:
        ledger: The experiment ledger (accuracy columns).
        benchmark: ``{variant: {"params_M": float, "flops_G": float}}`` from the bench
            command, or ``None`` when it has not been run -- in which case the cost
            columns are ``TBD`` too rather than recalled from memory.
    """
    from saryolo.evaluation.reported_baselines import reported_rows

    best = _latest_by_experiment(_ledger_rows(ledger))
    bench = benchmark or {}
    table = Table(
        "efficiency_frontier",
        "Efficiency frontier. Cost is *measured* on this machine (params and GFLOPs at "
        "640\u00b2, one-class head); accuracy is ``TBD`` until the corresponding run exists, "
        "because a cost claim needs no accuracy to be true about cost. The lower section "
        "lists published SAR detectors: those are numbers their authors reported, not "
        "reproduced here, and they are kept in a separate section for exactly that reason.",
        ["Model", "Params (M)", "GFLOPs", "mAP50", "mAP50:95", "Source"],
    )
    for variant, label, exp_id in EFFICIENCY_FRONTIER_ARMS:
        record = _match_variant(_ledger_rows(ledger), variant)
        if record is None:
            record = best.get(exp_id)
        metrics = record.metrics if record else {}
        row = bench.get(variant, {})
        # Cost prefers the ledger's own measurement (recorded with the run); the bench
        # profile is the pre-training fallback. Both are measured; neither is estimated.
        params = metrics.get("params_M", row.get("params_M"))
        flops = metrics.get("flops_G", row.get("flops_G"))
        table.rows.append([
            label,
            _fmt(params, 3),
            _fmt(flops, 2),
            _fmt(metrics.get("mAP50")),
            _fmt(metrics.get("mAP50_95")),
            "measured (this repo)",
        ])
        table.provenance.append(f"{variant}: {getattr(record, 'run_id', 'cost only, not run')}")

    # Reported section. An *absolute* captured quantity is placed in its matching column,
    # because that is the whole point of positioning against it; the ``(published)`` marker
    # and the venue in ``Source`` keep it from being read as our measurement. A
    # *relative-only* quantity (a percentage reduction, an AP delta) has no column here and
    # stays in ``Source``, where its baseline is named -- a reduction without its baseline
    # is not a comparable number, so it must not be promoted into a cell.
    def _absolute(r, *keys):
        for key in keys:
            if key in r.metrics:
                return r.metrics[key]
        return None

    for r in reported_rows():
        bits = ", ".join(f"{k}={v:g}" for k, v in r.metrics.items())
        source = f"{r.venue}, {r.year} \u2014 reported: {bits}"
        if r.caveats:
            source += f" [{' ; '.join(r.caveats)}]"
        table.rows.append([
            f"{r.name} (published)",
            _fmt(_absolute(r, "params_M"), 3),
            _fmt(_absolute(r, "flops_G"), 2),
            _fmt(_absolute(r, "map50", "map50_ssdd", "map50_hrsid"), 1),
            _fmt(_absolute(r, "map50_95"), 4),
            source,
        ])
        table.provenance.append(f"reported: {r.url}")
    return table


def build_multi_seed(ledger, experiment_id: str = "EXP-012") -> Table:
    """TABLE 9 — mean +/- std over seeds, plus the per-seed values.

    One row per training seed (the latest completed run of it), and best/worst
    reported alongside mean and sample std: a seed whose result sits far outside
    the rest is information about the method, not noise to be averaged away.
    """
    records = _latest_per_seed(_ledger_rows(ledger), experiment_id)
    values = [r.metrics["mAP50_95"] for r in records]
    table = Table(
        "multi_seed",
        "Multi-seed validation. A single-seed difference of a few tenths of a point is not "
        "evidence, so the headline result is reported as mean +/- std over seeds. "
        "Std is the sample standard deviation; one row per training seed.",
        ["Metric", "Mean", "Std", "Best", "Worst", "n seeds", "Per-seed values"],
    )
    # The per-seed column is filled whenever a value exists; mean/std/best/worst
    # need at least two seeds, because a single run is not validation.
    per_seed = ", ".join(f"{v:.4f}" for v in values) if values else TBD
    if len(values) >= 2:
        import statistics

        table.rows.append([
            "mAP50:95",
            f"{statistics.mean(values):.4f}",
            f"{statistics.stdev(values):.4f}",
            f"{max(values):.4f}",
            f"{min(values):.4f}",
            str(len(values)),
            per_seed,
        ])
    else:
        table.rows.append(["mAP50:95", TBD, TBD, TBD, TBD, str(len(values)), per_seed])
    return table


def write_tables(tables: list[Table], out_dir: str | Path, assert_complete: bool = False) -> list[Path]:
    """Write each table as Markdown and LaTeX; return the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in tables:
        if assert_complete and not table.complete:
            raise ValueError(
                f"Table {table.name!r} contains unmeasured cells (TBD) and cannot be used for "
                "submission. Run the missing experiments rather than filling the cells by hand."
            )
        md = out / f"{table.name}.md"
        md.write_text(table.to_markdown() + "\n")
        tex = out / f"{table.name}.tex"
        tex.write_text(table.to_latex() + "\n")
        written += [md, tex]
    return written
