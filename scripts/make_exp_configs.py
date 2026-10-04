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
    ("EXP-018", "v2_prior_spectral", "+ prior-conditioned spectral selection",
     "Module G: the target prior chooses the radial frequency bands. Added rather than folded "
     "into EXP-017 so the ladder row and the slot study measure the same graph."),
    ("EXP-019", "v2_cons", "FULL v2 + representation consistency (SEC. 4)",
     "The SEC. 4 term switched on: identical graph to EXP-017, different objective. Its own "
     "experiment because a loss cannot be ablated by removing parameters."),
)

#: The efficiency frontier (EXP-501..505). One runnable config per frontier arm, so a
#: cost row in the frontier table can always be traced to a run rather than to a profile
#: taken at a different time. ``(id, variant, display name, purpose)``.
#:
#: These are *not* a new mechanism. Each one is the full v2 graph with the two largest
#: compute slots (context aggregation and adaptive multi-scale fusion) removed, plus the
#: scale/conditioning/P2 points of the frontier -- i.e. exactly the subset a removal
#: ablation would select if those slots fail to repay their measured share of the compute.
FRONTIER: tuple[tuple[str, str, str, str], ...] = (
    ("EXP-501", "v2_lite_s", "SARVO-Lite (s)",
     "Efficiency frontier: full v2 minus context aggregation and AMF, at scale s."),
    ("EXP-502", "v2_lite_p35_s", "SARVO-Lite (s), no P2",
     "Efficiency frontier without the P2 level: the cheapest point that keeps every physical prior."),
    ("EXP-503", "v2_lite_cond_s", "SARVO-Lite + conditioning (s)",
     "Efficiency frontier with the acquisition-conditioned adapter: is the cross-sensor claim affordable?"),
    ("EXP-504", "v2_lite_n", "SARVO-Lite (n)",
     "Efficiency frontier at scale n: the edge-deployment point."),
    ("EXP-505", "v2_lite_m", "SARVO-Lite (m)",
     "Efficiency frontier at scale m: does the frontier hold when capacity is restored?"),
)

#: Parameter-efficient adaptation (master Direction B / Stage 10). The LoRA arms exist so the
#: proposed adapter is compared against the *standard* cheap-adaptation method rather than
#: against a full fine-tune. Every arm shares one model graph and one dataset with its
#: reference, so the only difference is the ``peft`` block -- which is what makes the
#: comparison a comparison of adaptation methods.
#:
#: ``(id, variant, display name, peft block, purpose)``. ``peft`` is ``None`` for the
#: reference arm (a full fine-tune of the same graph).
PEFT: tuple[tuple[str, str, str, dict | None, str], ...] = (
    ("EXP-601", "baseline", "Full fine-tune (reference)", None,
     "Adaptation reference: the whole detector is trained. Every LoRA arm below is compared against this."),
    ("EXP-602", "baseline", "LoRA r=4", {"method": "lora", "rank": 4, "alpha": 8.0},
     "Parameter-efficient adaptation: rank-4 low-rank updates on every convolution, base frozen."),
    ("EXP-603", "baseline", "LoRA r=8", {"method": "lora", "rank": 8, "alpha": 16.0},
     "Parameter-efficient adaptation at the conventional rank; the arm most papers report."),
    ("EXP-604", "baseline", "LoRA r=16", {"method": "lora", "rank": 16, "alpha": 32.0},
     "Does the low-rank budget bind? A larger rank tests whether r=8 was capacity-limited."),
    ("EXP-605", "baseline", "LoRA r=8, backbone frozen",
     {"method": "lora", "rank": 8, "alpha": 16.0, "freeze_base": True},
     "Strict adapter-only optimisation (a low-rank linear probe). Not comparable with the arms "
     "above: it freezes the head too, so it measures a different budget."),
)

#: Data-efficiency sweep (master Direction E). The fractions are the labelled *training* data
#: used, subsampled with one shared seed so every arm and every fraction sees the same images
#: -- otherwise a difference between fractions would be confounded with a difference between
#: draws. ``(id, fraction, display name)``.
DATA_FRACTIONS: tuple[tuple[str, float, str], ...] = (
    ("EXP-701", 0.01, "1% of training data"),
    ("EXP-702", 0.05, "5% of training data"),
    ("EXP-703", 0.10, "10% of training data"),
    ("EXP-704", 0.25, "25% of training data"),
    ("EXP-705", 0.50, "50% of training data"),
)

#: SARVO prototype arms (master Phase 5, `docs/architecture_proposals.md` §2). The first is
#: the recommended direction; the second is the fixed-threshold control that separates the
#: statistic from the learned gain. Emitted at the *baseline* model, so the only difference
#: from EXP-001 is the first representation. ``(id, variant, display name, purpose)``.
PROTOTYPE: tuple[tuple[str, str, str, str], ...] = (
    ("EXP-801", "cfar_s", "Ratio-space CFAR front end",
     "SARVO prototype: the analytic multi-scale CFAR statistic as the detector's first "
     "representation, with a zero-initialised per-pixel gain. Compared against EXP-001 at the "
     "same scale, seed and schedule, so any difference is the representation."),
    ("EXP-802", "cfar_fixed_s", "CFAR statistic, fixed threshold (control)",
     "Control for EXP-801: the same statistic stack with an analytic threshold and zero "
     "learnable parameters. Separates 'the statistics help' from 'the learned gain helps'."),
)

#: Experiment ids that are evaluated rather than trained.
EVAL_ONLY = {"EXP-009", "EXP-010", "EXP-011"}

#: Module-level ablation slots: ``(slot name, experiment-id prefix, arms)``.
#:
#: Each arm sits in the *same slot* with every other component held fixed, so a difference
#: between arms is attributable to that slot's mechanism rather than to a change elsewhere in
#: the graph. The last entry of each slot is the proposed arm (except the removal slot, where
#: every row is a removal).
#:
#: The id prefix is written out rather than derived from the slot's position. Deriving it
#: (``f"EXP-2{slot_index}{arm_index}"``) silently produced ``EXP-2101`` once a tenth slot was
#: added, i.e. a four-digit id that no longer matched the documented ``EXP-2<slot><arm>``
#: scheme and that a reader could not place. An explicit prefix also means appending a slot
#: can never renumber the ids of the slots above it.
ABLATION_SLOTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("attention", "21", ("att_none", "att_se", "att_eca", "att_cbam", "att_saa_static", "attention")),
    ("fusion", "22", ("fus_concat", "fus_add", "fus_static", "amf")),
    ("speckle", "23", ("spk_none", "spk_lee", "spk_denoise", "speckle")),
    ("enhancement", "24", ("pre_identity", "pre_log", "pre_clahe", "pre_standardize", "sfe")),
    # The two `tp_spectral*` arms answer Module G of the brief. They are capacity matched --
    # same band head, same band count, same descriptor width -- so the only difference is
    # whether the head is fed the *prior evidence* or the raw feature. Without that control,
    # "the prior conditions the spectrum" would be indistinguishable from "any input-adaptive
    # filter does".
    # Appended *after* `v2_full`, not inserted before it: the generator numbers arms by
    # position, so inserting the new arms would have moved `v2_full` off EXP-255 and silently
    # repointed that id at a different graph. Ids are referenced by the ledger and the paper
    # tables, so a renumber is indistinguishable from a changed result.
    ("target prior", "25", ("tp_none", "tp_cfar", "tp_static", "tp_channel", "v2_full",
                            "tp_spectral_feat", "tp_spectral", "v2_prior_spectral")),
    # Appended (not inserted) so the arms above keep their ids: SEC. 14 of the brief asks for
    # the alternative-frequency study explicitly, so the transform arms join this slot.
    ("frequency", "26", ("fr_none", "fr_highpass", "fr_static", "fr_dct", "fr_wavelet", "v2_full")),
    ("context", "27", ("cx_none", "cx_local", "cx_regional", "v2_full")),
    # `v2_nopspectral` appended so the ids above keep their meaning: its reference model is
    # v2_prior_spectral (EXP-018), not v2_full, so "removal" here means dropping Module G's
    # spectral selection and reverting the prior to spatial-only -- one edit, the same slot.
    ("removal", "28", ("v2_noclutter", "v2_noprior", "v2_nofreq", "v2_noctx", "v2_norefine", "v2_nopspectral")),
    # SEC. 4 is a *loss* slot, so it is kept out of the removal slot above: those arms are
    # verified by a strict parameter drop, which a loss term can never produce -- every arm
    # here is the same size as `v2_full` by construction. The control is `v2_full` itself
    # (`w_consistency: 0`), and the sweep asks whether the *choice of corruption* is what
    # carries the gain, which is the difference between a principle and a tuned constant.
    ("consistency", "32", ("v2_full", "cons_sev1", "cons_sev16", "cons_lowcontrast",
                            "cons_lowsnr", "v2_cons")),
    # Acquisition conditioning -- the cross-sensor claim. Two sub-studies share one slot because
    # they vary the same insertion point: the adapter *design* (scale / shift / film / spatial) and
    # the *metadata field set* (sensor-only through continuous-only). `v2_full` is the control, so
    # this slot answers "does any conditioning help?" as well as "which design and which fields?".
    ("conditioning", "33", ("v2_full", "cond_gain", "cond_shift", "cond_sensor",
                            "cond_resolution", "cond_continuous", "cond_film", "cond_spatial")),
    # Appended rather than inserted, so the ids of the slots above keep their meaning
    # (an id is referenced by the ledger and by the paper tables).
    ("refinement", "29", ("rf_none", "rf_local", "rf_static", "rf_off25", "rf_off100", "v2_full")),
    # Module A. The proposed arm is `in_hybrid` rather than `v2_full` because the adapter is
    # deliberately *not* part of the v2 default: it has to earn that place here.
    ("input adapter", "31", ("in_identity", "in_local", "in_learned", "in_hybrid")),
)


#: For each removal arm, the model graph it is removed *from*.
#:
#: This is declared rather than assumed, because "removal" is only meaningful relative to a
#: reference, and one arm's reference is not ``v2_full``:
#:
#: * ``v2_nopspectral`` removes Module G (prior-conditioned spectral selection), which is not
#:   part of ``v2_full`` -- it is the EXP-018 ladder step. Comparing it against ``v2_full``
#:   would report a no-op as a clean removal, and the arm would then "prove" that Module G is
#:   free while measuring nothing at all. Against ``v2_prior_spectral`` it is a strict removal.
#:
#: Kept next to the slot that uses it, and cross-checked below, so an arm cannot be added to
#: the removal slot without someone stating what it is a removal *from*.
REMOVAL_REFERENCES: dict[str, str] = {
    "v2_noclutter": "v2_full",
    "v2_noprior": "v2_full",
    "v2_nofreq": "v2_full",
    "v2_noctx": "v2_full",
    "v2_norefine": "v2_full",
    "v2_nopspectral": "v2_prior_spectral",
}


def _removal_arms() -> tuple[str, ...]:
    """The variants in the removal slot, read from the slot table itself."""
    for slot, _prefix, variants in ABLATION_SLOTS:
        if slot == "removal":
            return variants
    raise SystemExit("no 'removal' slot is declared in ABLATION_SLOTS; the removal table is empty")


def check_removal_references() -> None:
    """Every removal arm must have a declared reference, and both must agree with the slot.

    Without this the failure mode is silent: an arm with no reference is simply never
    compared, so it appears in the table and in every config while being unchecked.
    """
    arms = set(_removal_arms())
    declared = set(REMOVAL_REFERENCES)
    if arms - declared:
        raise SystemExit(
            f"removal arms without a declared reference in REMOVAL_REFERENCES: "
            f"{sorted(arms - declared)}. State what each one is a removal from."
        )
    if declared - arms:
        raise SystemExit(
            f"REMOVAL_REFERENCES names {sorted(declared - arms)}, which is not in the "
            f"removal slot {sorted(arms)}; a stale reference would never be exercised."
        )
    for arm, ref in REMOVAL_REFERENCES.items():
        if arm == ref:
            raise SystemExit(f"removal arm {arm!r} is its own reference; that is not a removal")


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

    check_removal_references()

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

    # Phase 3 -- the RGB-pretraining baseline, as three arms whose *only* stated
    # difference is the initialisation. The model YAML and every train arg are shared, so
    # a later accuracy difference is attributable to the init stage and nothing else.
    # EXP-401 is the control that also documents EXP-404's default: without an `init`
    # key, the runner starts from a fresh build of the model YAML.
    for exp_id, init_val, note in (
        ("EXP-401", "none", "random init (fresh build from the model YAML; the Phase-3 control)"),
        ("EXP-402", "coco11", "COCO-pretrained yolo11s.pt fine-tuned on SAR (the standard route)"),
        ("EXP-403", {"weights": "PATH/TO/msfa_or_sar_checkpoint.pt"},
         "SAR-pretrained checkpoint fine-tuned on SAR (e.g. MSFA-style weights); state the path before running"),
    ):
        payload = {
            "experiment": {"id": exp_id, "name": f"RGB/SAR pretraining baseline ({note.split(' (')[0]})",
                           "description": f"Phase 3: {note}"},
            "model": model_rel("baseline"),
            "dataset": rel_dataset,
            "init": init_val,
            "train": _train_block(args),
            "notes": f"Phase 3 init comparison; shared config with EXP-401..403 except the init stage. {note}",
        }
        _write(out / f"{exp_id}_init_baseline.yaml", payload, f"Phase 3 init: {note}")
        written.append(f"{exp_id}_init_baseline.yaml")

    # The efficiency frontier. Emitted from its own table rather than from MATRIX so the
    # frontier can grow a scale without renumbering anything above it.
    for exp_id, variant, name, purpose in FRONTIER:
        rel_model = model_rel(variant)
        if rel_model is None:
            continue
        payload = {
            "experiment": {"id": exp_id, "name": name, "description": purpose},
            "model": rel_model,
            "dataset": rel_dataset,
            "train": _train_block(args),
            "notes": purpose,
        }
        _write(out / f"{exp_id}_{variant}.yaml", payload, purpose)
        written.append(f"{exp_id}_{variant}.yaml")

    # Parameter-efficient adaptation (Direction B / Stage 10). Emitted from the PEFT table so
    # a new rank is one row rather than a hand-written config that drifts from the others.
    for exp_id, variant, name, peft, purpose in PEFT:
        rel_model = model_rel(variant)
        if rel_model is None:
            continue
        payload = {
            "experiment": {"id": exp_id, "name": name, "description": purpose},
            "model": rel_model,
            "dataset": rel_dataset,
            "train": _train_block(args),
            "notes": purpose,
        }
        if peft is not None:
            payload["peft"] = dict(peft)
        slug = "full" if peft is None else f"lora_r{peft.get('rank', 'x')}" + (
            "_frozen" if peft.get("freeze_base") else ""
        )
        _write(out / f"{exp_id}_peft_{slug}.yaml", payload, purpose)
        written.append(f"{exp_id}_peft_{slug}.yaml")

    # SARVO prototype arms (master Phase 5). Emitted from the PROTOTYPE table so the arm and
    # its control cannot drift apart, and so the recommended direction is one entry rather
    # than a hand-written config next to tests that assume it exists.
    for exp_id, variant, name, purpose in PROTOTYPE:
        rel_model = model_rel(variant)
        if rel_model is None:
            continue
        payload = {
            "experiment": {"id": exp_id, "name": name, "description": purpose},
            "model": rel_model,
            "dataset": rel_dataset,
            "train": _train_block(args),
            "notes": purpose,
        }
        _write(out / f"{exp_id}_{variant}.yaml", payload, purpose)
        written.append(f"{exp_id}_{variant}.yaml")

    # Data-efficiency sweep (Direction E). The fraction is written into the config as an
    # explicit ``data_fraction`` key rather than being baked into a separate dataset directory,
    # so every arm reads the same split and the subset is drawn at train time. The runner turns
    # it into Ultralytics' own ``fraction`` argument, which sorts the file list and takes a
    # prefix -- nested at every fraction, so a difference between two arms is the fraction
    # rather than a different sample.
    for exp_id, fraction, name in DATA_FRACTIONS:
        rel_model = model_rel("v2_full" if args.scale == "s" else f"v2_full_{args.scale}")
        if rel_model is None:
            continue
        purpose = (
            f"Data-efficiency sweep: {name}, drawn as a prefix of the sorted training split so "
            f"the fractions are nested and comparable."
        )
        payload = {
            "experiment": {"id": exp_id, "name": name, "description": purpose},
            "model": rel_model,
            "dataset": rel_dataset,
            "train": _train_block(args),
            "data_fraction": fraction,
            "notes": purpose,
        }
        _write(out / f"{exp_id}_datafrac_{int(round(fraction * 100)):02d}.yaml", payload, purpose)
        written.append(f"{exp_id}_datafrac_{int(round(fraction * 100)):02d}.yaml")

    # Module-level ablation arms: EXP-<prefix><arm>.
    seen_ids: dict[str, str] = {}
    for slot, prefix, variants in ABLATION_SLOTS:
        for arm_index, variant in enumerate(variants, start=1):
            exp_id = f"EXP-{prefix}{arm_index}"
            rel_model = model_rel(variant)
            if rel_model is None:
                continue
            # Two configs claiming one experiment id would make the ledger ambiguous: the
            # table builders key on the id, so whichever ran last would silently win. This
            # actually happened when a slot grew and its `v2_full` arm moved id.
            key = exp_id
            if key in seen_ids:
                raise SystemExit(
                    f"duplicate experiment id {exp_id}: {seen_ids[key]} and {slot}/{variant}"
                )
            seen_ids[key] = f"{slot}/{variant}"
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

    # Prune configs this generator owns but no longer emits. Without this, an arm that moves
    # id leaves its old file behind: the directory then contains two configs for the same
    # experiment, and `--only EXP-294` would run whichever sorted first. Only `EXP-*` files
    # are touched (`_smoke_*.yaml` is hand-written and must survive).
    emitted = set(written)
    pruned = [p.name for p in sorted(out.glob("EXP-*.yaml")) if p.name not in emitted]
    for name in pruned:
        (out / name).unlink()

    print(f"wrote {len(written)} configs to {out}")
    for name in written:
        print(f"  {name}")
    if pruned:
        print(f"\npruned {len(pruned)} stale config(s) this run no longer emits:")
        for name in pruned:
            print(f"  {name}")
    if missing:
        print("\nMISSING model YAMLs (run `python -m saryolo arch --variant all` first):")
        for path in sorted(set(missing)):
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
