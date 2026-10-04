#!/usr/bin/env python
"""Render the qualitative real-data figure: HRSID chips with GT and predictions.

Why a separate script
---------------------
``scripts/make_readme_assets.py`` measures *architecture*, which needs no trained weights and
therefore always works on a clean checkout. This figure needs a checkpoint, so it lives apart
and is allowed to be absent: a fresh clone has no ``results/`` directory, and a figure
generator that fails there would make the whole chart set unreproducible.

What it draws
-------------
For the best measured real-data arm, a ``ground truth | predictions | overlay`` grid on the
hardest test chips, so false negatives and false positives are visible side by side instead of
being inferred from a single mAP number. The panels come from
:mod:`saryolo.visualization.detections`, which is also what a reader can call directly.

Every number printed and drawn here is read from the experiment ledger and from the checkpoint
that the ledger names. Nothing is typed by hand, and a missing checkpoint is reported as a
missing checkpoint rather than skipped silently.

Usage::

    python scripts/make_real_figures.py                     # best arm by mAP50-95
    python scripts/make_real_figures.py --experiment REAL-001 --rows 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_readme_assets as M  # noqa: E402  (sets the house style for every figure)

from saryolo.tracking.ledger import ExperimentLedger  # noqa: E402
from saryolo.visualization.detections import comparison_panel, select_examples  # noqa: E402

ASSETS = ROOT / "docs" / "assets"


def _real_arms() -> dict:
    """The measured real-data arms, read from the ledger."""
    return M._real_arms_facts(ExperimentLedger(ROOT / "results"))


def _best_arm(arms: dict, preferred: str | None = None) -> str | None:
    """The arm with the highest measured mAP50-95, or the requested one if it has run."""
    if not arms:
        return None
    if preferred:
        if preferred not in arms:
            raise SystemExit(f"{preferred} has no completed run in the ledger")
        return preferred
    ranked = [eid for eid, row in arms.items() if row.get("mAP50_95") is not None]
    if not ranked:
        raise SystemExit("no real-data arm recorded mAP50-95; nothing to illustrate")
    return max(ranked, key=lambda eid: arms[eid]["mAP50_95"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None, help="REAL-* id to illustrate (default: best mAP50-95)")
    parser.add_argument("--rows", type=int, default=3, help="how many test chips to draw")
    parser.add_argument("--mode", default="small", choices=("dense", "sparse", "small"),
                        help="which test chips to pick")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--out", default=str(ASSETS / "real_detections.png"))
    args = parser.parse_args()

    arms = _real_arms()
    exp_id = _best_arm(arms, args.experiment)
    if exp_id is None:
        print("No real-data arm has a completed run, so there is nothing to illustrate.")
        print("Run one, e.g.:  python -m saryolo train --exp configs/exp/REAL-001_hrsid_baseline.yaml")
        return 0

    row = arms[exp_id]
    weights = ROOT / "results" / "runs" / exp_id / "weights" / "best.pt"
    if not weights.is_file():
        print(f"{exp_id} is in the ledger but its checkpoint is missing: {weights}")
        print("The figure is left untouched rather than replaced by a drawing of a different run.")
        return 1

    data_yaml = ROOT / "configs" / "datasets" / "hrsid_real.yaml"
    imgsz = args.imgsz or int(row.get("imgsz") or 640)
    chips = select_examples(data_yaml, limit=args.rows, mode=args.mode)
    if not chips:
        print(f"no test chip with labels could be selected from {data_yaml}")
        return 1

    from saryolo.training.trainer import load_model

    model = load_model(str(weights))
    labels_dir = data_yaml.parent.parent / "datasets" / "processed" / "hrsid_real" / "labels" / "test"
    out = Path(args.out)
    titles = (
        "SAR chip · ground truth",
        f"{exp_id} predictions (conf ≥ {args.conf:g})",
        "overlay · GT green, prediction orange",
    )
    written = comparison_panel(
        model, chips, labels_dir, out,
        imgsz=imgsz, conf=args.conf, titles=titles, dpi=170,
    )
    if written is None:
        print("the panel produced no rows; check that the chips and labels line up")
        return 1

    scored = ", ".join(
        f"{key} {row[key]:.3f}" for key in ("mAP50", "mAP50_95", "precision", "recall") if key in row
    )
    print(f"wrote {written.relative_to(ROOT)}")
    print(f"  arm {exp_id} ({row['model']}, {row.get('epochs')} epochs, imgsz {imgsz}): {scored}")
    print(f"  chips: {[p.name for p in chips]}")
    print("  every value above was read from results/experiments.jsonl, not typed here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
