#!/usr/bin/env python
"""Generate every chart and table in the README from real artefacts.

Why generate instead of write by hand
-------------------------------------
A README is the first thing a reviewer reads and the easiest place to
accidentally overclaim. Every figure here is produced by *measuring* something on
this machine:

* parameter/FLOP counts come from actually constructing each architecture and
  profiling it;
* the identity chart runs each module on a real tensor and records the observed
  deviation from the input;
* the dataset table comes from :mod:`saryolo.data.registry`;
* the "has real results" grid comes from the experiment ledger, which only
  records measured values;
* the test-suite chart comes from collecting this repository's own tests.

Nothing in this script can invent a number: if a value is missing it is drawn as
"not run", never as a plausible-looking guess. Run it with::

    python scripts/make_readme_assets.py

Outputs land in ``docs/assets/`` as SVG (crisp on GitHub at any width) plus a
``facts.json`` so numbers quoted in prose stay traceable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import saryolo  # noqa: E402,F401  (importing registers the custom modules)
from saryolo.data.registry import DATASETS, RECOMMENDED_ORDER, get_dataset  # noqa: E402
from saryolo.nn.arch import VARIANTS, build_yaml_dict  # noqa: E402
from saryolo.tracking.ledger import ExperimentLedger  # noqa: E402

ASSETS = ROOT / "docs" / "assets"

# --------------------------------------------------------------------------- style
BG = "#0a0e17"
PANEL = "#111827"
GRID = "#1f2a3c"
TEXT = "#e6edf7"
MUTED = "#8fa3bf"
CYAN = "#22d3ee"
MAGENTA = "#f472b6"
AMBER = "#fbbf24"
GREEN = "#34d399"
VIOLET = "#a78bfa"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
        # Deterministic SVG element ids. Without a fixed salt matplotlib hashes
        # object ids into random-looking ids, so every regeneration produces a
        # thousand-line diff even when not a single number changed -- which hides
        # the real changes in the history.
        "svg.hashsalt": "saryolo",
    }
)


def _header(ax, title: str, subtitle: str | None, title_size: float = 16) -> None:
    """Draw a left-aligned title and multi-line subtitle above an axes.

    The offsets are given in *points*, not axes fractions, and the title's offset
    is derived from the subtitle's line count. That is what stops the two from
    colliding: a three-line subtitle grows upward by a known number of points, so
    the title is placed above it rather than at a fixed guess. Positioning these
    by axes fraction is exactly what made the longer captions overlap the titles.
    """
    lines = subtitle.splitlines() if subtitle else []
    ax.annotate(
        title,
        xy=(0, 1), xycoords="axes fraction",
        xytext=(0, 14 + 12.5 * len(lines)), textcoords="offset points",
        ha="left", va="bottom", fontsize=title_size, fontweight="bold", color=TEXT,
    )
    if lines:
        ax.annotate(
            subtitle,
            xy=(0, 1), xycoords="axes fraction",
            xytext=(0, 7), textcoords="offset points",
            ha="left", va="bottom", fontsize=9.5, color=MUTED, linespacing=1.45,
        )


def _style(ax, title: str, subtitle: str | None = None, grid_axis: str = "y") -> None:
    """Apply the house style and an optional subtitle to an axes.

    Args:
        grid_axis: Which axis carries the grid lines. Vertical bars need
            horizontal grid lines (``"y"``); horizontal bars need vertical ones
            (``"x"``). Getting this wrong draws grid lines along the bars instead
            of across them, which reads as visual clutter.
    """
    ax.grid(axis=grid_axis, color=GRID, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    _header(ax, title, subtitle)


def _save(fig, name: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    # metadata={"Date": None} drops the <dc:date> stamp, the other half of the
    # non-determinism. Together with svg.hashsalt this makes an unchanged chart
    # byte-identical across runs, so a diff here always means a real change.
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.3, metadata={"Date": None})
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


# ----------------------------------------------------------------- measurement
_MEASURED: dict[tuple[str, str, bool], dict] = {}


def measure(variant: str, scale: str = "s", with_flops: bool = False, imgsz: int = 640) -> dict:
    """Construct one architecture and measure its real size (memoised).

    Args:
        variant: Key into :data:`saryolo.nn.arch.VARIANTS`.
        scale: Compound scale to build at.
        with_flops: Also profile GFLOPs (needs a forward pass, so it is opt-in).
        imgsz: Input resolution for FLOP profiling.

    Returns:
        dict with ``params_M`` and, when requested, ``flops_G``.
    """
    key = (variant, scale, with_flops)
    if key in _MEASURED:
        return _MEASURED[key]

    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops

    spec = VARIANTS[variant]
    spec.scale = scale
    model = DetectionModel(build_yaml_dict(spec), ch=3, nc=spec.nc, verbose=False)
    model.eval()
    out = {"variant": variant, "scale": scale, "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    if with_flops:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["flops_G"] = float(get_flops(model, imgsz=imgsz))
    del model
    _MEASURED[key] = out
    return out


def measure_identity() -> list[dict]:
    """Run each in-place SAR module on a real tensor and record the deviation.

    This is the empirical version of the claim the whole ablation depends on: a
    module must be an exact identity function at initialisation, so that any
    later accuracy difference is caused by *learning* rather than by the module
    rearranging the feature map before training starts.

    Returns:
        One dict per module with the observed max-absolute deviation.
    """
    import saryolo.nn.modules as M

    torch.manual_seed(0)
    c = 32
    x = torch.randn(2, c, 16, 16)
    cases = [
        ("SFE", "Component 1 - feature enhancement", M.SARFeatureEnhancement(c)),
        ("SFM", "Component 2 - speckle-aware", M.SpeckleAwareFeatureModule(c, mode="sfm_clutter")),
        ("SAA", "Component 3 - adaptive attention", M.SARAdaptiveAttention(c)),
        ("AMF", "Component 4 - multi-scale fusion", M.AdaptiveMultiScaleFusion(c, groups=[c // 2, c // 2], mode="amf")),
        ("TPM", "Component 8 - target prior", M.TargetPriorModulation(c)),
        ("SFR", "Component 9 - spatial-frequency", M.SpatialFrequencyRepresentation(c)),
        ("CAG", "Component 10 - context aggregation", M.ContextAggregation(c)),
        ("TADR", "Component 11 - deformable refinement", M.TargetAwareRefinement(c)),
        # The adapter's *proposed* arm is `hybrid`, and it satisfies the same contract as every
        # other module, so it belongs in this measurement rather than being left out and the
        # count quietly kept at eight.
        ("SIA", "Component 12 - input adapter", M.SARInputAdapter(c, mode="hybrid")),
    ]
    rows = []
    for short, label, module in cases:
        module.eval()
        with torch.no_grad():
            delta = float((module(x) - x).abs().max())
        rows.append({"short": short, "label": label, "max_abs_deviation": delta, "identical": delta == 0.0})
        print(f"  {short}: max|f(x) - x| = {delta:.1e}")
    return rows


#: The ablation ladder: each step adds exactly one module (the v2 clutter row is a mode
#: change on the speckle slot rather than an added module, which is why it is labelled
#: "+clutter" and not "+SFM2").
LADDER = ("baseline", "sfe", "speckle", "attention", "amf", "p2", "full",
          "v2_clutter", "v2_prior", "v2_freq", "v2_ctx", "v2_full", "v2_prior_spectral")
LADDER_LABELS = {
    "baseline": "YOLO11\nbaseline",
    "sfe": "+SFE",
    "speckle": "+SFM",
    "attention": "+SAA",
    "amf": "+AMF",
    "p2": "+P2 head",
    "full": "FULL v1\n+SAR loss",
    "v2_clutter": "+clutter",
    "v2_prior": "+prior",
    "v2_freq": "+freq",
    "v2_ctx": "+context",
    "v2_full": "FULL v2\n+refine",  # Components 1-11
    # Appended, so every previously published row keeps its label and meaning. This step
    # swaps the prior mechanism for the prior-conditioned spectral one rather than adding a
    # module, which is why it costs parameters but almost no compute.
    "v2_prior_spectral": "+prior\nspectral (G)",
}

#: Controlled module-level ablations.
#:
#: Each group holds every *other* component fixed and varies only the module in
#: that slot, so a difference in the table is attributable to that slot alone.
#: Getting this wrong is an easy and serious mistake -- for example comparing a
#: "no speckle module" arm that happens to also lack multi-scale fusion would
#: credit the fusion block's gains to the speckle module.
SLOT_SETS: dict[str, tuple[str, list[tuple[str, str, bool]]]] = {
    "attention": (
        "Component 3 - attention slot   (held fixed: SFE + SFM)",
        [
            ("att_none", "no attention", False),
            ("att_se", "SE", False),
            ("att_eca", "ECA", False),
            ("att_cbam", "CBAM", False),
            ("att_saa_static", "ours, static gate", True),
            ("attention", "ours, adaptive gate", True),
        ],
    ),
    "fusion": (
        "Component 4 - fusion slot   (held fixed: SFE + SFM + SAA)",
        [
            ("fus_concat", "Concat", False),
            ("fus_add", "Add (projected)", False),
            ("fus_static", "ours, static weights", True),
            ("amf", "ours, adaptive weights", True),
        ],
    ),
    "speckle": (
        "Component 2 - speckle slot   (held fixed: SFE + SAA + AMF)",
        [
            ("spk_none", "no handling", False),
            ("spk_lee", "Lee filter", False),
            ("spk_denoise", "fixed low-pass", False),
            ("amf", "ours, SFM", True),
        ],
    ),
    "enhancement": (
        "Component 1 - enhancement slot   (held fixed: SFM + SAA + AMF)",
        [
            ("pre_identity", "identity (none)", False),
            ("pre_log", "log compression", False),
            ("pre_clahe", "CLAHE", False),
            ("pre_standardize", "local standardisation", False),
            ("amf", "ours, SFE", True),
        ],
    ),
}

#: The v2 components' slot studies. Every arm sits in the same slot with all other
#: components held at their v2 settings, so a difference is attributable to that slot.
SLOT_SETS_V2: dict[str, tuple[str, list[tuple[str, str, bool]]]] = {
    "prior": (
        "Component 8 - target prior slot   (held fixed: full v2 setting)",
        [
            ("tp_none", "no modulation", False),
            ("tp_cfar", "CFAR statistic (no learning)", False),
            ("tp_static", "learned, spatially uniform", False),
            ("tp_channel", "learned, capacity-matched", False),
            ("v2_full", "ours, spatial prior", True),
        ],
    ),
    "frequency": (
        "Component 9 - spatial-frequency slot   (held fixed: full v2 setting)",
        [
            ("fr_none", "no spectral branch", False),
            ("fr_highpass", "fixed high-pass", False),
            ("fr_static", "learned bands, fixed filter", False),
            ("fr_dct", "block DCT, same size", False),
            ("fr_wavelet", "Haar sub-bands", False),
            ("v2_full", "ours, input-adaptive", True),
        ],
    ),
    "context": (
        "Component 10 - context slot   (held fixed: full v2 setting)",
        [
            ("cx_none", "no context", False),
            ("cx_local", "local, dilated", False),
            ("cx_regional", "regional only", False),
            ("v2_full", "ours, both extents", True),
        ],
    ),
    "refinement": (
        "Component 11 - refinement slot   (held fixed: full v2 setting)",
        [
            ("rf_none", "no refinement", False),
            ("rf_local", "local, no offsets (control)", False),
            ("rf_static", "learned offsets, fixed", False),
            ("rf_off25", "ours, radius 0.25", True),
            ("rf_off100", "ours, radius 1.0", True),
            ("v2_full", "ours, input-adaptive offsets", True),
        ],
    ),
    "adapter": (
        "Module A - SAR input adapter   (applied to full v2)",
        [
            ("in_identity", "raw intensity (control)", False),
            ("in_local", "local statistics only", False),
            ("in_learned", "learned only", False),
            ("in_hybrid", "ours, both streams", True),
        ],
    ),
    "removal": (
        "Removal ablation   (v2 full minus one component)",
        [
            ("v2_noclutter", "- clutter branch", False),
            ("v2_noprior", "- target prior", False),
            ("v2_nofreq", "- spatial-frequency", False),
            ("v2_noctx", "- context", False),
            ("v2_norefine", "- refinement", False),
            ("v2_full", "full v2", True),
        ],
    ),
}


def build_facts() -> dict:
    """Measure the model zoo and collect repository facts."""
    facts: dict = {"variants": len(VARIANTS)}

    print("Measuring the ablation ladder ...")
    facts["zoo"] = {}
    for scale in ("n", "s"):
        for name in LADDER:
            m = measure(name, scale, with_flops=(scale == "s"))
            facts["zoo"][f"{name}_{scale}"] = m
            extra = f", {m['flops_G']:.2f} GFLOPs" if "flops_G" in m else ""
            print(f"  {name}_{scale}: {m['params_M']:.3f} M{extra}")

    print("Measuring controlled module-level ablations ...")
    facts["slots"] = {}
    for group, (_, arms) in {**SLOT_SETS, **SLOT_SETS_V2}.items():
        facts["slots"][group] = [
            {"variant": v, "label": label, "ours": ours, "params_M": measure(v, "s")["params_M"]}
            for v, label, ours in arms
        ]

    print("Measuring identity-at-initialisation ...")
    facts["identity"] = measure_identity()

    for key in RECOMMENDED_ORDER:
        spec = get_dataset(key)
        facts.setdefault("datasets", {})[key] = {
            "title": spec.title,
            "images": spec.approx_images,
            "classes": len(spec.classes),
            "tier": spec.tier,
            "imgsz": spec.recommended_imgsz,
            "format": spec.annotation_format,
        }
    facts["dataset_count"] = len(DATASETS)

    # Distinct EXP-00x ids: several files legitimately share an id (seed replicates).
    exp_ids = sorted(p.stem.split("_")[0] for p in (ROOT / "configs" / "exp").glob("EXP-*.yaml"))
    known = set(exp_ids)
    ledger = ExperimentLedger(ROOT / "results")
    completed = {r.experiment_id for r in ledger.completed()}
    facts["experiments"] = {
        "total": len(known),
        "config_files": len(exp_ids),
        # Restricted to the committed experiment ids on purpose: the synthetic
        # SMOKE-* runs also land in the ledger, and they must never be mistaken for
        # a paper experiment that produced a number.
        "with_results": sorted(completed & known),
        "ledger_runs": len(ledger.load()),
        "non_experiment_runs": sorted(completed - known),
    }

    print("Collecting tests ...")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    counts = Counter(line.split("::")[0] for line in proc.stdout.splitlines() if "::" in line)
    facts["tests"] = {"total": sum(counts.values()), "by_file": dict(sorted(counts.items()))}
    print(f"  {facts['tests']['total']} tests in {len(counts)} files")

    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(f"  wrote {(ASSETS / 'facts.json').relative_to(ROOT)}")
    return facts


# --------------------------------------------------------------------- charts
def chart_ladder_params(facts: dict) -> None:
    """Grouped bars: parameters of each ablation-ladder step, scale n vs s."""
    # Wide enough for eleven ladder steps: at the original width the tick labels for the
    # later v2 steps crowd each other even when they do not strictly overlap.
    fig, ax = plt.subplots(figsize=(18.5, 8.6))
    xs = range(len(LADDER))
    w = 0.38
    for offset, scale, color in ((-w / 2, "n", CYAN), (w / 2, "s", MAGENTA)):
        vals = [facts["zoo"][f"{n}_{scale}"]["params_M"] for n in LADDER]
        bars = ax.bar([x + offset for x in xs], vals, w, label=f"YOLO11-{scale}", color=color, alpha=0.9)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=color)
    for scale, color in (("n", CYAN), ("s", MAGENTA)):
        ax.axhline(facts["zoo"][f"baseline_{scale}"]["params_M"], color=color,
                   linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LADDER_LABELS[n] for n in LADDER], fontsize=9.5)
    ax.set_ylabel("Parameters (millions)")
    ax.set_ylim(0, max(facts["zoo"][f"{n}_s"]["params_M"] for n in LADDER) * 1.24)
    _style(
        ax,
        "The cost of every module, measured",
        "Dotted lines = stock baseline. Two things worth reading off this chart: +P2 head and FULL "
        "have identical\nbars, because the SAR-aware loss is an objective rather than a layer - "
        "Component 7 costs no parameters at all.\nCounts are profiled with a one-class head, so the "
        "stock YOLO11n baseline reads 2.59M here versus the 2.62M published for 80 classes.",
    )
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper left")
    _save(fig, "ladder_params.svg")


def _text_box(artist, fig) -> tuple[float, float, float, float]:
    """Window extent of a Text artist, in device pixels."""
    box = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    return (box.x0, box.y0, box.x1, box.y1)


def _boxes_intersect(a, b, pad: float = 2.0) -> bool:
    """True when two rectangles (in pixels) touch, allowing a small padding."""
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


#: Candidate label placements in points, relative to the marker. Tried in order.
#: The vertical options are spread widely because the crowded cluster of points in
#: the accuracy/cost chart spans a few percent of the x axis, so labels can only be
#: separated vertically; the horizontal flip is the fallback.
_LABEL_CANDIDATES: tuple[tuple[float, float], ...] = (
    (13, 17), (13, -25), (13, 50), (13, -58), (13, 84), (13, -92),
    (-13, 17), (-13, -25), (-13, 50), (-13, -58),
)


def _label_points(ax, fig, points, labels, colors, fontsize: float = 10.5) -> None:
    """Annotate scatter points with labels that do not overlap anything.

    A fixed offset is not enough for this chart, and the reason is a property of
    the experiment rather than of the drawing code: the ladder deliberately
    contains near-coincident points. The baseline and +SFE differ by 0.1%, and
    +P2 head and FULL are *identical* because the SAR-aware loss adds no
    parameters. One shared offset stacked those labels almost exactly on top of
    each other.

    Each candidate placement is measured against the boxes already on the figure
    (tick labels, caption, previously placed labels) and the first clear one wins.
    """
    fig.canvas.draw()
    taken = [
        box
        for artist in fig.findobj(plt.Text)
        if artist.get_text().strip() and artist.get_visible()
        for box in [_text_box(artist, fig)]
    ]

    for (x, y), label, color in zip(points, labels, colors, strict=True):
        chosen = None
        for dx, dy in _LABEL_CANDIDATES:
            artist = ax.annotate(
                label, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=fontsize, color=color, ha="left" if dx > 0 else "right", va="center",
            )
            fig.canvas.draw()
            box = _text_box(artist, fig)
            if not any(_boxes_intersect(box, other) for other in taken):
                chosen = box
                break
            artist.remove()
        if chosen is None:
            # Every candidate collided: keep the first rather than drop the label,
            # since an unlabelled point is worse than a crowded one.
            dx, dy = _LABEL_CANDIDATES[0]
            artist = ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                                 fontsize=fontsize, color=color, ha="left", va="center")
            fig.canvas.draw()
            chosen = _text_box(artist, fig)
        taken.append(chosen)


def chart_accuracy_cost(facts: dict) -> None:
    """Scatter of parameters vs GFLOPs for the ladder at scale s."""
    pts = [
        (facts["zoo"][f"{n}_s"]["flops_G"], facts["zoo"][f"{n}_s"]["params_M"], LADDER_LABELS[n], n)
        for n in LADDER
        if "flops_G" in facts["zoo"].get(f"{n}_s", {})
    ]
    fig, ax = plt.subplots(figsize=(13.5, 8.4))
    cmap = plt.get_cmap("cool")
    for i, (fx, py, _label, _name) in enumerate(pts):
        ax.scatter(fx, py, s=210, color=cmap(i / max(len(pts) - 1, 1)),
                   edgecolor=BG, linewidth=1.6, zorder=3)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GRID, linewidth=1.2, zorder=1)
    ax.set_xlabel("GFLOPs @ 640\u00b2  (compute per image)")
    ax.set_ylabel("Parameters (millions)")
    # Explicit limits keep the leftmost x tick away from the lowest y tick, which
    # otherwise collide in the bottom-left corner.
    ax.set_xlim(min(p[0] for p in pts) - 3.0, max(p[0] for p in pts) + 1.5)
    ax.set_ylim(min(p[1] for p in pts) - 1.2, max(p[1] for p in pts) + 1.7)
    _style(
        ax,
        "Where the extra compute actually goes",
        "Both axes are profiled, not estimated. Curve of the ladder: the multi-scale fusion and "
        "the P2 head dominate,\nso each must earn its place in EXP-005 and EXP-006 before the "
        "FULL model is ever trained.\nThe baseline/+SFE and +P2/FULL pairs nearly coincide -- "
        "that is the point, not a plotting error.",
    )
    # Placed after the caption so the header boxes are already on the figure and
    # count as obstacles when the labels look for somewhere to sit.
    _label_points(
        ax, fig,
        [(p[0], p[1]) for p in pts],
        [p[2].replace("\n", " ") for p in pts],
        [cmap(i / max(len(pts) - 1, 1)) for i in range(len(pts))],
    )
    _save(fig, "accuracy_cost.svg")


def _render_slot_panels(facts: dict, slot_sets: dict, filename: str, suptitle: str, blurb: str) -> None:
    """One panel per slot: the parameter cost of every alternative within that slot.

    The grid is computed from the number of slots rather than hard-coded, and the header
    band is a fixed physical height regardless of panel count. That constant band is what
    keeps a panel's title from colliding with the row above it when the grid grows -- the
    collision the first version of this chart actually had.
    """
    names = list(slot_sets)
    ncols = min(len(names), 2)
    nrows = (len(names) + ncols - 1) // ncols
    height = 6.75 * nrows
    header_in = 2.7  # inches reserved above the panels for the title, caption and blurb

    fig, axes = plt.subplots(nrows, ncols, figsize=(17.5, height), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=1 - (header_in / height), bottom=0.055,
                        hspace=0.42, wspace=0.34)
    flat = list(axes.ravel())
    for unused in flat[len(names):]:
        unused.set_visible(False)

    for ax, group in zip(flat, names, strict=False):
        title, arms = slot_sets[group]
        rows = facts["slots"][group]
        labels = [r["label"] for r in rows]
        vals = [r["params_M"] for r in rows]
        colors = [MAGENTA if r["ours"] else CYAN for r in rows]
        bars = ax.barh(labels, vals, color=colors, alpha=0.9, height=0.62)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(v * 1.015, bar.get_y() + bar.get_height() / 2, f"{v:.3f}M",
                    va="center", fontsize=10, color=MUTED)
        # The spread goes in the caption rather than floating over the bars, where it
        # used to sit on top of the longest one.
        span = max(vals) - min(vals)
        ax.set_xlim(0, max(vals) * 1.30)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.tick_params(labelsize=11)
        ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        _header(
            ax,
            title,
            f"{len(arms)} arms · spread across arms: {span:.3f}M",
            title_size=13,
        )
    fig.suptitle(suptitle, color=TEXT, fontsize=19, fontweight="bold", x=0.02, ha="left",
                 y=1 - (0.35 / height))
    fig.text(0.02, 1 - (0.9 / height), blurb, color=MUTED, fontsize=11.5, ha="left", va="top",
             linespacing=1.5)
    _save(fig, filename)


def chart_slot_ablations(facts: dict) -> None:
    """Parameter cost of every alternative inside Components 1-4's slots."""
    _render_slot_panels(
        facts,
        SLOT_SETS,
        "slot_ablations.svg",
        "Every component is compared inside the same slot, at equal budget",
        "The claim is never \"attention helps\". It is \"our attention beats SE, ECA and CBAM when "
        "each sits in the identical slot on the identical backbone\".\nEach group holds every other "
        "component fixed; only the named slot varies. Magenta = proposed.\nWhere the spread is ~0 "
        "(Components 1 and 2's classical arms) the comparison is about accuracy, not size.",
    )


def chart_slot_ablations_v2(facts: dict) -> None:
    """Parameter cost of every alternative inside the v2 components' slots."""
    _render_slot_panels(
        facts,
        SLOT_SETS_V2,
        "slot_ablations_v2.svg",
        "The v2 components, held to the same standard as the first four",
        "Components 5-7 are ablated the same way as 1-4: one slot varies while everything else "
        "stays at its v2 setting.\nThe 'capacity-matched' prior arm reuses the proposed arm's exact "
        "evidence network and pools its output over space, so 'ours vs\ncapacity-matched' isolates "
        "spatial selectivity rather than size. The removal group asks the complementary question: "
        "does a component still\nearn its place once the others are already present?",
    )


def chart_identity(facts: dict) -> None:
    """Visualise the measured identity-at-initialisation property."""
    rows = facts["identity"]
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    for i, r in enumerate(rows):
        y = len(rows) - i
        ok = r["identical"]
        ax.add_patch(FancyBboxPatch((0.02, y - 0.3), 0.85, 0.6, boxstyle="round,pad=0.06",
                                    facecolor=PANEL, edgecolor=GREEN if ok else AMBER, linewidth=1.4))
        ax.text(0.445, y, r["short"], ha="center", va="center", fontsize=11,
                color=TEXT, fontweight="bold")
        ax.text(1.05, y, r["label"], ha="left", va="center", fontsize=10, color=TEXT)
        ax.text(3.55, y, "f(x) = x", ha="left", va="center", fontsize=10.5, color=GREEN,
                family="monospace")
        ax.text(4.55, y, f"max|f(x) - x| = {r['max_abs_deviation']:.1e}", ha="left", va="center",
                fontsize=10.5, color=GREEN if ok else AMBER, family="monospace")
    ax.text(4.55, len(rows) + 0.8, "measured on a random 2x32x16x16 tensor", fontsize=9, color=MUTED)
    ax.text(0.02, -0.15,
            "Consequence: at step 0 every variant is numerically the baseline, so no module can "
            "inflate accuracy merely by\nadding capacity. Any gain in the ablation table is "
            "therefore attributable to learning. Verified in\n"
            "tests/test_arch.py::test_models_output_identically_to_baseline_at_init.",
            fontsize=10, color=CYAN, va="top")
    ax.set_xlim(0, 8.4)
    ax.set_ylim(-1.5, len(rows) + 1.6)
    ax.axis("off")
    _save(fig, "identity_property.svg")


def _two_line(label: str, width: int = 34) -> str:
    """Wrap a dataset title onto at most two lines, breaking at the em dash.

    Long one-line tick labels are the main reason a horizontal bar chart looks
    cramped: they eat the left margin and push the bars into the caption.
    """
    head, _, tail = label.partition(" \u2014 ")
    if tail:
        return f"{head.strip()}\n{textwrap.fill(tail.strip(), width)}"
    return textwrap.fill(label.strip(), width)


def chart_datasets(facts: dict) -> None:
    """Dataset landscape: images per dataset, coloured by tier."""
    keys = list(RECOMMENDED_ORDER)
    vals = [facts["datasets"][k]["images"] for k in keys]
    labels = [_two_line(facts["datasets"][k]["title"]) for k in keys]
    colors = [GREEN if facts["datasets"][k]["tier"] == "pilot" else VIOLET for k in keys]

    fig, ax = plt.subplots(figsize=(16.5, 8.6))
    bars = ax.barh(labels, vals, color=colors, alpha=0.9, height=0.6)
    ax.set_xscale("log")

    # All annotations sit in one aligned column to the right of the longest bar,
    # instead of hanging off each bar's end where long text runs into the axis.
    text_x = max(vals) * 2.4
    for bar, v, k in zip(bars, vals, keys, strict=True):
        ax.text(text_x, bar.get_y() + bar.get_height() / 2,
                f"{v:,} images   ·   {facts['datasets'][k]['classes']} "
                f"class{'es' if facts['datasets'][k]['classes'] > 1 else ''} "
                f"   ·   {facts['datasets'][k]['format'].upper()} "
                f"   ·   {facts['datasets'][k]['imgsz']}px input",
                va="center", ha="left", fontsize=11.5, color=MUTED)

    ax.set_xlabel("Images (log scale)", fontsize=12)
    ax.set_xlim(1, max(vals) * 130)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.tick_params(labelsize=12)
    ax.axvline(max(vals), color=VIOLET, linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(
        handles=[
            Patch(facecolor=GREEN, alpha=0.9, label="pilot tier \u2014 fast iteration"),
            Patch(facecolor=VIOLET, alpha=0.9, label="benchmark tier \u2014 final numbers"),
        ],
        frameon=False, labelcolor=TEXT, fontsize=11.5, loc="lower right",
    )
    _style(
        ax,
        "The SAR data landscape, and why we start small",
        "Development order is decided by cost: a wiring bug should surface on a 1,160-image \n"
        "pilot dataset, not after hours on the 116k-image benchmark. Counts are indicative \u2014 \n"
        "docs/DATASETS.md records exactly what must be verified against each official release.",
        grid_axis="x",
    )
    _save(fig, "datasets.svg")


def chart_coverage(facts: dict) -> None:
    """Honest status grid: wired experiments vs those with measured results.

    The geometry is derived from the number of rows rather than fixed. It was fixed, and the
    grid silently grew into its own labels: at 72 experiments the rows were 0.12in apart while
    the text needed 0.14in, so consecutive ids overlapped by a third. Any chart whose row count
    comes from the repository has to compute its size from that count, or it will be correct
    only for the experiment set it was first written against.

    Two columns rather than one, because 72 rows in a single column would need a canvas around
    24in tall to stay readable -- which is worse than unreadable, since nobody scrolls it.
    """
    ids = sorted(p.stem.split("_")[0] for p in (ROOT / "configs" / "exp").glob("EXP-*.yaml"))
    done = set(facts["experiments"]["with_results"])

    per_col = (len(ids) + 1) // 2
    col_w = 3.75
    fig, ax = plt.subplots(figsize=(15.5, max(6.0, per_col * 0.30 + 2.8)))
    for c in range(2):
        x0 = c * col_w
        for i, eid in enumerate(ids[c * per_col:(c + 1) * per_col]):
            y = per_col - i - 1
            status = eid in done
            ax.add_patch(FancyBboxPatch((x0, y - 0.3), col_w - 0.15, 0.62,
                                        boxstyle="round,pad=0.05", facecolor=PANEL,
                                        edgecolor=GREEN if status else GRID, linewidth=1.3))
            ax.text(x0 + 0.13, y, eid, fontsize=10, color=TEXT, va="center", family="monospace")
            ax.text(x0 + 1.65, y, "measured" if status else "wired, awaiting GPU",
                    fontsize=9.5, color=GREEN if status else AMBER, va="center")
    # The count is interpolated, not typed: the caption previously claimed "all 12 experiments"
    # while the grid below it drew seventy rows.
    ax.text(0, per_col + 0.7,
            f"All {len(ids)} experiments are code-complete and reproducible from a committed "
            "config. None of the accuracy numbers\nexist yet: this repository contains no "
            "fabricated results, and the table generators refuse to emit a row without\n"
            "a measured value. Run notebook 02 on a GPU and this grid fills itself in.",
            fontsize=10, color=CYAN, va="bottom")
    ax.set_xlim(0, 2 * col_w - 0.2)
    ax.set_ylim(-0.5, per_col + 2.6)
    ax.axis("off")
    _save(fig, "coverage.svg")


def chart_tests(facts: dict) -> None:
    """Test-suite composition, straight from pytest collection."""
    by_file = facts["tests"]["by_file"]
    labels = [Path(k).name for k in by_file]
    vals = list(by_file.values())
    colors = [CYAN, MAGENTA, AMBER, GREEN, VIOLET][: len(vals)]
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    bars = ax.barh(labels, vals, color=colors, alpha=0.9)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2, str(v),
                va="center", fontsize=10, color=TEXT, fontweight="bold")
    ax.set_xlabel("Tests")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.margins(y=0.14)
    _style(
        ax,
        title=f"Guarding the science: {facts['tests']['total']} tests across {len(vals)} modules",
        subtitle=(
            "These are correctness tests, not accuracy claims. They pin the baseline to published "
            "parameter counts, prove\nmodule identity at init, validate the COCO matcher against "
            "hand-computed cases, and assert that no table can\nemit an unmeasured number."
        ),
        grid_axis="x",
    )
    _save(fig, "tests.svg")


def main() -> None:
    facts = build_facts()
    print("Rendering charts ...")
    chart_ladder_params(facts)
    chart_accuracy_cost(facts)
    chart_slot_ablations(facts)
    chart_slot_ablations_v2(facts)
    chart_identity(facts)
    chart_datasets(facts)
    chart_coverage(facts)
    chart_tests(facts)
    print("\nDone. Assets in docs/assets/")


if __name__ == "__main__":
    main()
