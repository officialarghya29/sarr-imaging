"""Published SAR-detector results, kept in a separate namespace from measured ones.

Why this module exists
----------------------
The efficiency claim in this project is comparative: "a better accuracy/cost point than
the current SAR detectors". Making that comparison honestly needs two different kinds of
number in one table:

* **measured** -- our own params/FLOPs/mAP, produced by this repository;
* **reported** -- numbers other papers published, which we did not run and cannot rerun
  here (no GPU, and the datasets are not on disk).

Mixing those two silently is the single easiest way to fabricate a result without typing a
false digit: a reader sees one column of numbers and assumes one protocol. So reported
values live here, every entry carries its venue and URL, and the table builder marks those
rows and footnotes them. Nothing in this module is ever written into the experiment
ledger, and nothing here is allowed to satisfy a measured cell.

A second discipline applies inside this module. Papers report different things:
some give absolute parameter counts, some only a percentage *reduction* against their own
baseline, and the reduction is meaningless without the baseline. Each entry therefore
records exactly which quantities were captured and in which units, and a quantity that was
not captured is simply absent rather than inferred. ``metrics`` keys are explicit about
their unit:

``params_M`` / ``flops_G``
    Absolute, as published.
``params_reduction_pct`` / ``compute_reduction_pct``
    Relative to the entry's own stated baseline; the baseline is named in ``notes``.
``map50`` / ``map50_95``
    Absolute accuracy, with the dataset named in ``dataset``.
``map50_95_delta``
    An accuracy *difference* the paper reports against its own baseline.

Numbers here were captured from the abstract or the paper's own text on 2026-10-03; each
entry states the source. They are not a substitute for a measured comparison under our
protocol -- they are the positioning that decides which operating point we have to beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ReportedBaseline",
    "REPORTED_BASELINES",
    "reported_rows",
    "frontier_summary",
]


@dataclass(frozen=True)
class ReportedBaseline:
    """One published SAR detector, with the numbers that were actually captured.

    Attributes:
        key: Stable short identifier, used by the table builder and tests.
        name: The model's name as its authors write it.
        venue: Where it was published, or ``"arXiv preprint"`` when unrefereed.
        year: Publication year.
        url: Canonical link (DOI, publisher page, or arXiv).
        code: Public implementation, when the paper states one.
        dataset: The dataset the accuracy numbers refer to.
        backbone: The detector family it is built on -- the comparison is only
            meaningful against the same family, so this is required rather than optional.
        metrics: Captured quantities, keyed with their unit (see the module docstring).
        notes: What the numbers are relative to, and any protocol caveat.
        caveats: Reasons the number may not be comparable to ours, stated rather than
            buried. An empty tuple means the captured quantity is a plain absolute.
    """

    key: str
    name: str
    venue: str
    year: int
    url: str
    dataset: str
    backbone: str
    metrics: dict[str, float]
    code: str | None = None
    notes: str = ""
    caveats: tuple[str, ...] = field(default_factory=tuple)


#: The published field this project positions against. Every entry carries a URL; the
#: capture date is the module docstring's, and re-verify before submission, because a
#: reported number is only as good as the version of the paper it was read from.
REPORTED_BASELINES: dict[str, ReportedBaseline] = {
    "ac_yolo": ReportedBaseline(
        key="ac_yolo",
        name="AC-YOLO",
        venue="PLOS ONE 20(7): e0327362",
        year=2025,
        url="https://doi.org/10.1371/journal.pone.0327362",
        code="https://github.com/He-ship-sar/ACYOLO",
        dataset="SSDD / HRSID",
        backbone="YOLO11",
        metrics={
            # The abstract states reductions and AP *deltas* only -- no absolute counts.
            "params_reduction_pct": 30.0,
            "compute_reduction_pct": 15.6,
            "map_delta_ssdd_pct": 1.2,
            "map_delta_hrsid_pct": 1.5,
        },
        notes=(
            "Reductions and AP deltas are against their own YOLO11 baseline on SSDD; "
            "absolute parameter and FLOP counts are not stated in the abstract."
        ),
        caveats=(
            "relative-only: no absolute params/FLOPs captured",
            "AP delta is 'AP', not mAP50:95 -- the threshold is not stated in the abstract",
        ),
    ),
    "rle_yolo": ReportedBaseline(
        key="rle_yolo",
        name="RLE-YOLO",
        venue="IEEE JSTARS",
        year=2025,
        url="https://ieeexplore.ieee.org/document/10924247",
        dataset="SSDD / HRSID",
        backbone="YOLOv8",
        metrics={
            "map50_ssdd": 93.9,
            "map50_hrsid": 98.4,
            "params_reduction_pct": 43.9,
            "compute_reduction_pct": 34.5,
        },
        notes=(
            "mAP50 on each dataset; reductions are against their improved YOLOv8 baseline. "
            "This is the strongest published mAP50 anchor on the two pilot datasets."
        ),
        caveats=(
            "single-class ship detection, so mAP50 is a plain AP",
            "YOLOv8 family, not YOLO11 -- a different baseline than ours",
        ),
    ),
    "sarlite": ReportedBaseline(
        key="sarlite",
        name="SARLite",
        venue="Scientific Reports 16, article s41598-026-49143-5",
        year=2026,
        url="https://www.nature.com/articles/s41598-026-49143-5",
        dataset="SARDet-100K",
        backbone="YOLO",
        metrics={
            "params_reduction_pct": 17.0,
            "compute_reduction_gflops": 1.0,
            "map50_95_delta_pct": 3.4,
        },
        notes=(
            "On SARDet-100K: ~3.4% mAP@50:95 improvement with ~17% fewer parameters and "
            "~1 GFLOP less than its baseline. Relative-only."
        ),
        caveats=(
            "relative-only: absolute params/FLOPs and mAP not captured",
            "multi-class benchmark, so mAP50:95 is comparable in kind to ours",
        ),
    ),
    "edge_yolo": ReportedBaseline(
        key="edge_yolo",
        name="Edge-optimized lightweight YOLO",
        venue="Remote Sensing 17(13): 2168",
        year=2025,
        url="https://www.mdpi.com/2072-4292/17/13/2168",
        dataset="SARDet-100K",
        backbone="YOLO",
        metrics={
            "params_M": 1.9,
            "map_sardet100k": 87.7,
        },
        notes=(
            "Reports 87.7 mAP on SARDet-100K with only 1.9 M parameters. The single "
            "clearest absolute anchor for a sub-2M-parameter SAR detector."
        ),
        caveats=(
            "the metric is reported as 'mAP' without the threshold; treat as mAP50 until "
            "the table in the paper is checked",
            "no FLOPs captured",
        ),
    ),
}


def reported_rows() -> list[ReportedBaseline]:
    """All reported baselines, ordered by name for a stable table."""
    return sorted(REPORTED_BASELINES.values(), key=lambda r: r.name.lower())


def frontier_summary() -> str:
    """A one-paragraph, citable summary of the published efficiency landscape.

    Kept in code rather than in prose so the README and the paper cannot drift from the
    entries above: adding a baseline without updating the text is not possible.
    """
    lines = [
        "Published SAR detectors report their cost relative to their own baseline rather "
        "than in absolute terms, which is why the frontier below is a positioning aid and "
        "not a like-for-like comparison:",
    ]
    for r in reported_rows():
        bits = ", ".join(f"{k}={v}" for k, v in r.metrics.items())
        lines.append(f"* {r.name} ({r.venue}, {r.year}, {r.backbone}) — {bits}")
    return "\n".join(lines)
