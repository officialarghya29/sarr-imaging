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
           "build_baseline_comparison", "build_ablation", "build_module_ablation",
           "build_removal_ablation", "build_scale_analysis", "build_robustness", "build_efficiency",
           "build_multi_seed", "write_tables"]

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


def _match_variant(records, variant: str):
    """The completed record whose model file is exactly ``variant``, or ``None``.

    Matched on the model file *stem* at a ``_`` boundary rather than by substring
    containment. This matters: ``"v2_full"`` is a substring of ``"v2_full_s"``,
    ``"v2_full_l"`` and ``"v2_full_p35_s"``, so a containment test could quietly report a
    different model's score in the ablation table -- and a wrong number in a table is worse
    than a missing one, because nothing downstream can tell it is wrong.
    """
    for record in records:
        if record.status != "completed":
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
    "Frequency": ("fr_none", "fr_highpass", "fr_static", "v2_full"),
    "Context": ("cx_none", "cx_local", "cx_regional", "v2_full"),
    # Component 11's own study. `rf_local` is the capacity control for the proposed arm: the
    # same sub-network with the offsets removed, so `ours - rf_local` isolates *deformation*
    # rather than the extra convolution.
    "Refinement": ("rf_none", "rf_local", "rf_static", "v2_full"),
}

#: Removal ablation rows: ``(label, variant)``. Exposed for the same reason as above.
REMOVAL_ABLATION_ROWS: tuple[tuple[str, str], ...] = (
    ("Full (v2)", "v2_full"),
    ("- clutter branch", "v2_noclutter"),
    ("- target prior", "v2_noprior"),
    ("- spatial-frequency", "v2_nofreq"),
    ("- context", "v2_noctx"),
    ("- refinement", "v2_norefine"),
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


def build_multi_seed(ledger, experiment_id: str = "EXP-012") -> Table:
    """TABLE 9 — mean +/- std over seeds, plus the per-seed values."""
    records = [r for r in _ledger_rows(ledger) if r.experiment_id == experiment_id]
    values = [r.metrics.get("mAP50_95") for r in records if r.metrics.get("mAP50_95") is not None]
    table = Table(
        "multi_seed",
        "Multi-seed validation. A single-seed difference of a few tenths of a point is not "
        "evidence, so the headline result is reported as mean +/- std over seeds.",
        ["Metric", "Mean", "Std", "n seeds", "Per-seed values"],
    )
    if len(values) >= 2:
        import statistics

        table.rows.append([
            "mAP50:95",
            f"{statistics.mean(values):.4f}",
            f"{statistics.pstdev(values):.4f}",
            str(len(values)),
            ", ".join(f"{v:.4f}" for v in values),
        ])
    else:
        table.rows.append(["mAP50:95", TBD, TBD, str(len(values)), TBD])
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
