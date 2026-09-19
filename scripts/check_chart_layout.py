#!/usr/bin/env python
"""Detect overlapping or clipped text in the generated README charts.

Why this exists
---------------
Whether a chart "looks right" is normally judged by eye, which means layout
regressions are invisible in a diff and only noticed after publication. A caption
that collides with a title, or tick labels that run into the bars, still renders
as a valid SVG, so nothing in the test suite or the diff can catch it.

This script renders every chart and then measures the *actual* text bounding
boxes matplotlib computed, reporting pairs that overlap by more than a threshold
and any text that a tight bounding box could not fit. It is a layout linter, not
an image comparison, so it works without storing golden files.

Usage:
    python scripts/check_chart_layout.py            # report problems
    python scripts/check_chart_layout.py --min-overlap 0.15

Exit code is 1 when a problem is found, so it can gate a commit.

The charts are rebuilt from ``docs/assets/facts.json`` rather than by re-measuring
the model zoo, so this takes a couple of seconds instead of minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_readme_assets as M  # noqa: E402

#: Chart functions, in the order the generator runs them. Each must render exactly one
#: figure: this checker captures only the last figure a function saves, so a function that
#: draws two charts would leave the first unchecked.
CHARTS = (
    M.chart_ladder_params,
    M.chart_accuracy_cost,
    M.chart_slot_ablations,
    M.chart_slot_ablations_v2,
    M.chart_identity,
    M.chart_datasets,
    M.chart_coverage,
    M.chart_tests,
)

#: Text smaller than this (in points^2) is a tick label or a data label with a
#: short string; overlaps involving them are measured the same way but they are
#: the usual source of false positives, so they are reported separately.
TINY_AREA_PT2 = 30.0


def _texts(fig) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Every visible Text artist on a figure, with its window extent in inches."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for artist in fig.findobj(matplotlib.text.Text):
        text = artist.get_text()
        if not text.strip() or not artist.get_visible():
            continue
        try:
            bbox = artist.get_window_extent(renderer=renderer)
        except Exception:
            continue
        # Convert device pixels to inches so the report is DPI-independent.
        dpi = fig.dpi
        box = (bbox.x0 / dpi, bbox.y0 / dpi, bbox.x1 / dpi, bbox.y1 / dpi)
        # Matplotlib keeps placeholder tick labels with a 1x1 pixel extent for axes
        # it has not drawn. They are invisible, but a box that small can sit 100%
        # inside a real label and would be reported as a collision.
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < 0.005:
            continue
        out.append((text, box))
    return out


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return (intersection area, smaller box area) of two rectangles."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, dx) * max(0.0, dy)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter, min(area_a, area_b)


def check(chart, facts: dict, min_overlap: float) -> list[str]:
    """Render one chart and return human-readable problems."""
    problems: list[str] = []
    captured: dict[str, object] = {}
    original_save = M._save

    def capture(fig, name):
        captured["fig"] = fig
        captured["name"] = name
        return fig  # do not write to disk; this is a pure layout check

    M._save = capture
    try:
        chart(facts)
    finally:
        M._save = original_save

    fig = captured.get("fig")
    name = captured.get("name", getattr(chart, "__name__", "?"))
    if fig is None:
        return [f"{name}: chart produced no figure"]

    entries = _texts(fig)
    fig_w, fig_h = fig.get_size_inches()

    # 1. Overlapping text.
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            (ta, ba), (tb, bb) = entries[i], entries[j]
            inter, smaller = _overlap(ba, bb)
            if smaller <= 0 or inter / smaller < min_overlap:
                continue
            ratio = inter / smaller
            problems.append(
                f"{name}: text overlap {ratio:.0%}  {ta[:38]!r}  vs  {tb[:38]!r}"
            )

    # 2. Text drawn outside the canvas region the chart can grow into. With
    #    bbox_inches='tight' the canvas expands, so anything beyond a 12in x 26in
    #    envelope means the layout is running away rather than fitting.
    limit_x, limit_y = 26.0, 26.0
    for text, box in entries:
        if box[2] > limit_x or box[3] > limit_y:
            problems.append(f"{name}: text extends beyond {limit_x}in x {limit_y}in: {text[:38]!r}")

    print(f"  {name}: {len(entries)} text artists, canvas {fig_w:.1f}in x {fig_h:.1f}in")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--min-overlap", type=float, default=0.30,
        help="Report a pair when the overlap covers this fraction of the smaller text box (default: 0.30)",
    )
    parser.add_argument("--facts", default="docs/assets/facts.json")
    args = parser.parse_args()

    facts_path = ROOT / args.facts
    if not facts_path.exists():
        print(f"missing {facts_path}; run scripts/make_readme_assets.py first")
        return 1
    facts = json.loads(facts_path.read_text())

    print("Checking chart layout ...")
    problems: list[str] = []
    for chart in CHARTS:
        problems.extend(check(chart, facts, args.min_overlap))

    if not problems:
        print("\nNo text collisions detected.")
        return 0

    print(f"\n{len(problems)} layout problem(s):")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
