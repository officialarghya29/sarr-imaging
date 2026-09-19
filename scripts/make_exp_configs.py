#!/usr/bin/env python
"""Generate the EXPERIMENT MATRIX configs under ``configs/exp/``.

The experiment matrix is the backbone of the paper: EXP-001..016 walk from the
baseline to the full model, then ablation, robustness, efficiency,
generalization and multi-seed validation. Generating them from one table instead
of hand-writing dozens of YAMLs keeps them consistent (same seeds, same schedule,
same dataset) — which is what makes the ablation table a controlled comparison
rather than a pile of unrelated runs.

Two id ranges are emitted:

``EXP-0xx``
    The main ladder and the evaluation studies, one config per row of
    :data:`MATRIX`.
``EXP-2xx``
    Module-level ablation arms. These used to have model YAMLs but no configs, so
    the ablation tables had nothing to consume: a table can only be filled by a
    run, and a run needs a config. The scheme is ``EXP-2<slot><arm>``, e.g.
    ``EXP-251``..``EXP-255`` for the target-prior slot, which keeps every arm
    individually traceable in the ledger.

Usage
-----
    python scripts/make_exp_configs.py --dataset ssdd
    python scripts/make_exp_configs.py --dataset sardet100k --scale m --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saryolo.nn.arch import VARIANTS, variant_filename  # noqa: E402

#: ``(id, model variant, display name, what it isolates)``.
MATRIX: tuple[tuple[str, str, str, str], ...] = (
    ("EXP-001", "baseline", "YOLO baseline", "Engineering baseline; every other row is measured against this."),
    ("EXP-002", "sfe", "+ SAR Feature Enhancement", "Component 1 alone."),
    ("EXP-003", "speckle", "+ Speckle-Aware Module", "Component 2 on top of Component 1."),
    ("EXP-004", "attention", "+ SAR-Adaptive Attention", "Component 3 on top of Components 1-2."),
    ("EXP-005", "amf", "+ Adaptive Multi-Scale Fusion", "Component 4 on top of Components 1-3."),
    ("EXP-006", "p2", "+ P2 small-object head", "High-resolution detection level, justified by dataset stats."),
    ("EXP-007", "full", "FULL SAR-YOLO", "All modules + P2 head + SAR-aware loss."),
    ("EXP-008", "full_noloss", "SAR-YOLO w/o SAR loss", "Isolates the loss (Component 7) from the architecture."),
    ("EXP-009", "full", "Robustness study", "Uses the EXP-007 checkpoint; no separate training is needed."),
    ("EXP-010", "full", "Efficiency study", "Uses the EXP-007 checkpoint; no separate training is needed."),
    ("EXP-011", "full", "Cross-dataset generalization", "Uses the EXP-007 checkpoint; no separate training is needed."),
    ("EXP-012", "full", "Multi-seed validation", "Three seeds for mean +/- std on the headline metric."),
    # NOTE: EXP-012's seed repeats are emitted below against `full_<scale>`. For the v2
    # headline number the same study must be repeated against `v2_full_<scale>`; that is a
    # deliberate manual step because it doubles the compute and belongs in the paper's
    # compute budget rather than in a generator's default.
    # --- v2 extension: the target-prior / clutter / frequency / context components, appended
    # rather than renumbered, so the already-published EXP-001..008 keep their exact meaning.
    ("EXP-013", "v2_clutter", "+ Clutter-aware representation",
     "Component 2 extension: clutter modelled separately from speckle."),
    ("EXP-014", "v2_prior", "+ Target prior modulation", "Component 8: the project's central hypothesis."),
    ("EXP-015", "v2_freq", "+ Spatial-frequency representation", "Component 9: explicit spectral branch."),
    ("EXP-016", "v2_ctx", "+ Context aggregation", "Component 10: multi-extent context."),
    ("EXP-017", "v2_full", "FULL v2 SAR-YOLO", "Component 11: target-aware deformable refinement, completing the v2 model."),
)

#: Experiment ids that are evaluated rather than trained.
EVAL_ONLY = {"EXP-009", "EXP-010", "EXP-011"}

#: Module-level ablation slots. Each arm sits in the *same slot* with every other
#: component held fixed, so a difference between arms is attributable to that slot's
#: mechanism rather than to a change elsewhere in the graph. The last entry of each slot
#: is the proposed arm (except the removal slot, where every row is a removal).
ABLATION_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attention", ("att_none", "att_se", "att_eca", "att_cbam", "att_saa_static", "attention")),
    ("fusion", ("fus_concat", "fus_add", "fus_static", "amf")),
    ("speckle", ("spk_none", "spk_lee", "spk_denoise", "speckle")),
    ("enhancement", ("pre_identity", "pre_log", "pre_clahe", "pre_standardize", "sfe")),
    ("target prior", ("tp_none", "tp_cfar", "tp_static", "tp_channel", "v2_full")),
    ("frequency", ("fr_none", "fr_highpass", "fr_static", "v2_full")),
    ("context", ("cx_none", "cx_local", "cx_regional", "v2_full")),
    ("removal", ("v2_noclutter", "v2_noprior", "v2_nofreq", "v2_noctx", "v2_norefine")),
    # Appended rather than inserted, so the ids of the slots above keep their meaning
    # (an id is referenced by the ledger and by the paper tables).
    ("refinement", ("rf_none", "rf_local", "rf_static", "v2_full")),
)


def _write(path: Path, payload: dict, note: str) -> None:
    header = (
        "# GENERATED by scripts/make_exp_configs.py -- edit the generator, not this file.\n"
        f"# {note}\n"
    )
    path.write_text(header + yaml.safe_dump(payload, sort_keys=False))


def _train_block(args: argparse.Namespace) -> dict:
    train = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seeds[0],
        "optimizer": "auto",
        "cos_lr": True,
        "patience": 30,
        "workers": 8,
        "deterministic": True,
    }
    if args.scale in ("m", "l", "x"):
        train["batch"] = max(args.batch // 2, 1)
    return train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="ssdd", help="Dataset key used for the data config name")
    parser.add_argument("--scale", default="s", choices=["n", "s", "m", "l", "x"], help="Model scale for the matrix")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds for EXP-012")
    parser.add_argument("--models-dir", default="configs/models")
    parser.add_argument("--out", default="configs/exp")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written, missing = [], []

    def model_rel(variant: str) -> str | None:
        """Fractional model path for a variant, or ``None`` if its YAML was never emitted."""
        model_file = Path(args.models_dir) / variant_filename(VARIANTS[variant])
        if not model_file.exists():
            missing.append(str(model_file))
            return None
        return str(Path("..") / "models" / model_file.name)

    rel_dataset = str(Path("..") / "datasets" / f"{args.dataset}.yaml")

    for exp_id, variant, name, purpose in MATRIX:
        rel_model = model_rel(variant)
        if rel_model is None:
            continue
        payload = {
            "experiment": {"id": exp_id, "name": name, "description": purpose},
            "model": rel_model,
            "dataset": rel_dataset,
            "train": _train_block(args),
            "evaluation_only": exp_id in EVAL_ONLY,
            "notes": purpose,
        }
        _write(out / f"{exp_id}_{variant}.yaml", payload, purpose)
        written.append(f"{exp_id}_{variant}.yaml")

    # EXP-012 gets one config per seed so each run is independently traceable.
    for seed in args.seeds[1:]:
        payload = {
            "experiment": {"id": "EXP-012", "name": f"Multi-seed validation (seed {seed})",
                           "description": "Repeat of EXP-007 with a different seed."},
            "model": f"../models/{variant_filename(VARIANTS[f'full_{args.scale}'])}",
            "dataset": rel_dataset,
            "train": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "seed": seed,
                      "cos_lr": True, "patience": 30, "workers": 8, "deterministic": True},
            "notes": f"EXP-007 repeated with seed {seed}; EXP-012 reports mean +/- std over all seeds.",
        }
        _write(out / f"EXP-012_seed{seed}_full_{args.scale}.yaml", payload, f"multi-seed seed {seed}")
        written.append(f"EXP-012_seed{seed}_full_{args.scale}.yaml")

    # Module-level ablation arms: EXP-2<slot><arm>.
    for slot_index, (slot, variants) in enumerate(ABLATION_SLOTS, start=1):
        for arm_index, variant in enumerate(variants, start=1):
            exp_id = f"EXP-2{slot_index}{arm_index}"
            rel_model = model_rel(variant)
            if rel_model is None:
                continue
            purpose = f"Module ablation -- {slot} slot, arm {arm_index}/{len(variants)} ({variant})."
            payload = {
                "experiment": {"id": exp_id, "name": f"{slot}: {variant}", "description": purpose},
                "model": rel_model,
                "dataset": rel_dataset,
                "train": _train_block(args),
                "notes": purpose,
            }
            _write(out / f"{exp_id}_{slot.replace(' ', '_')}_{variant}.yaml", payload, purpose)
            written.append(f"{exp_id}_{slot.replace(' ', '_')}_{variant}.yaml")

    print(f"wrote {len(written)} configs to {out}")
    for name in written:
        print(f"  {name}")
    if missing:
        print("\nMISSING model YAMLs (run `python -m saryolo arch --variant all` first):")
        for path in sorted(set(missing)):
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
