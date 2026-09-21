"""SAR-YOLO command line interface.

A single entry point for the whole pipeline, so the Colab notebooks and the
local smoke tests run exactly the same code paths::

    saryolo datasets                      # list supported datasets
    saryolo synth-data --out datasets/processed/smoke
    saryolo check-data --dataset datasets/processed/ssdd
    saryolo stats --dataset datasets/processed/ssdd --name ssdd
    saryolo arch --variant all --nc 1
    saryolo train --exp configs/exp/EXP-001_baseline.yaml
    saryolo eval --weights runs/EXP-001/weights/best.pt --data configs/datasets/ssdd.yaml
    saryolo robustness --weights ... --data ...
    saryolo efficiency --weights ... 
    saryolo cross-dataset --weights ... --source A.yaml --target B.yaml
    saryolo bench --variants baseline_s full_s
    saryolo ledger
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cmd_datasets(args) -> int:
    from saryolo.data import list_datasets

    for spec in list_datasets():
        print(f"{spec.name:<12} {spec.tier:<10} nc={spec.nc}  images~{spec.approx_images:<7} imgsz={spec.recommended_imgsz}")
        print(f"  {spec.title}")
        print(f"  source: {spec.source}")
        print(f"  license: {spec.license}")
        if spec.notes:
            print(f"  note: {spec.notes}")
        print()
    return 0


def _cmd_synth(args) -> int:
    from saryolo.data import make_synthetic_dataset, write_data_yaml

    info = make_synthetic_dataset(
        args.out,
        n_train=args.train,
        n_val=args.val,
        n_test=args.test,
        imgsz=args.imgsz,
        class_names=tuple(args.classes),
        looks=args.looks,
        clutter_rate=args.clutter,
        seed=args.seed,
    )
    yaml_path = write_data_yaml(
        Path(args.out) / "data.yaml", args.out, info["class_names"], splits=("train", "val", "test")
    )
    print(json.dumps(info, indent=2))
    print(f"data config -> {yaml_path}")
    print("\nNOTE: synthetic data is for pipeline testing only and must never be reported as a result.")
    return 0


def _dataset_splits(dataset: Path, splits: list[str]) -> list[tuple[str, Path, Path]]:
    """Resolve ``images/<split>`` directories, failing loudly instead of silently.

    Both commands below used to ``continue`` past any split whose ``images/<split>``
    directory was absent and then return success. A wrong path, or a ``data.yaml`` passed
    where a dataset *directory* is expected, therefore printed nothing and exited 0 -- so a
    CI step or a shell ``if ! saryolo check-data ...`` would report "dataset verified"
    having examined zero images. Silence is indistinguishable from approval, which makes it
    a worse failure than a crash.

    Args:
        dataset: Dataset root, expected to contain ``images/<split>`` and ``labels/<split>``.
        splits: Split names to resolve.

    Returns:
        One ``(split, images_dir, labels_dir)`` tuple per split that exists.

    Raises:
        SystemExit: If the root does not exist, is a file, or no requested split is present.
    """
    if dataset.is_file():
        raise SystemExit(
            f"--dataset expects a dataset DIRECTORY, but {dataset} is a file.\n"
            f"A data.yaml names the splits rather than containing them; pass the directory\n"
            f"it points at, e.g. --dataset {dataset.parent}."
        )
    if not dataset.is_dir():
        raise SystemExit(
            f"--dataset {dataset} does not exist.\n"
            f"Expected a directory containing images/<split> and labels/<split>."
        )

    found = [(s, dataset / "images" / s, dataset / "labels" / s) for s in splits
             if (dataset / "images" / s).exists()]
    if not found:
        present = sorted(p.name for p in (dataset / "images").glob("*") if p.is_dir()) \
            if (dataset / "images").is_dir() else []
        raise SystemExit(
            f"none of the requested splits {splits} exist under {dataset / 'images'}.\n"
            f"Splits actually present: {present or 'none (no images/ directory)'}"
        )

    missing = [s for s in splits if s not in {name for name, _i, _l in found}]
    if missing:
        # Not an error by default: plenty of SAR datasets ship no test split. Reported so the
        # reader can tell "checked and clean" from "never looked at".
        print(f"NOTE: splits not found and therefore not checked: {missing}\n")
    return found


def _cmd_check_data(args) -> int:
    from saryolo.data import validate_yolo_dataset, write_report

    dataset = Path(args.dataset)
    failed = False
    for split, images, labels in _dataset_splits(dataset, list(args.splits)):
        report = validate_yolo_dataset(images, labels, class_names=args.classes)
        print(report.summary())
        if args.report:
            out = write_report(report, Path(args.report) / f"{dataset.name}_{split}.json")
            print(f"  report -> {out}")
        failed = failed or not report.ok
        print()
    return 1 if failed else 0


def _cmd_stats(args) -> int:
    from saryolo.data import plot_statistics, profile_dataset, save_statistics

    dataset = Path(args.dataset)
    for split, images, labels in _dataset_splits(dataset, list(args.splits)):
        stats = profile_dataset(
            images, labels, args.classes, name=args.name or dataset.name, split=split,
            intensity_sample=args.sample,
        )
        print(stats.summary())
        save_statistics(stats, args.out)
        figures = plot_statistics(stats, args.out)
        print(f"  wrote {len(figures)} figures to {args.out}\n")
    return 0


def _cmd_arch(args) -> int:
    from saryolo.nn import arch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(arch.VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        if name not in arch.VARIANTS:
            raise SystemExit(f"Unknown variant {name!r}. Available: {', '.join(sorted(arch.VARIANTS))}")
        spec = arch.VARIANTS[name]
        spec.nc = args.nc
        path = out / arch.variant_filename(spec)
        path.write_text(arch.build_yaml_text(spec))
        print(f"wrote {path}")
    return 0


def _failure_reason(notes: str) -> str:
    """The cause of a failed run, as the runner recorded it.

    The runner catches the training exception and appends it to ``notes`` as
    ``... | ERROR: <cause>``. That is the only place the cause survives, so a caller that
    prints just the status turns a diagnosable failure into "failed: EXP-019 -> None" --
    no exception, no message, and the run directory is ``None`` because the failure happened
    before one was created. This extracts the cause, falling back to the whole note.
    """
    marker = "| ERROR:"
    if marker in notes:
        return notes.split(marker, 1)[1].strip()
    return notes.strip()


def _cmd_train(args) -> int:
    from saryolo.training.runner import run_experiment

    record = run_experiment(
        args.exp,
        ledger_root=args.ledger,
        project=args.project,
        device=args.device,
        extra_overrides=json.loads(args.overrides) if args.overrides else None,
        validate_after=not args.no_val,
        skip_ledger=args.no_ledger,
    )
    print(f"{record.status}: {record.experiment_id} -> {record.save_dir}")
    for key, value in record.metrics.items():
        print(f"  {key}: {value}")
    if record.status != "completed":
        print(f"  reason: {_failure_reason(record.notes)}")
    return 0 if record.status == "completed" else 1


def _cmd_eval(args) -> int:
    from saryolo.evaluation.metrics import evaluate_detections, write_metrics

    metrics = evaluate_detections(
        args.weights, args.data, imgsz=args.imgsz, conf=args.conf, device=args.device, out_dir=args.out
    )
    print(json.dumps(metrics, indent=2, default=float))
    write_metrics(metrics, Path(args.out) / "metrics.json")
    return 0


def _cmd_robustness(args) -> int:
    from saryolo.evaluation.robustness import robustness_sweep

    results = robustness_sweep(
        args.weights, args.data, out_dir=args.out, imgsz=args.imgsz,
        corruptions=tuple(args.corruptions), seed=args.seed, limit=args.limit, device=args.device,
    )
    print(json.dumps(results, indent=2, default=float))
    return 0


def _cmd_efficiency(args) -> int:
    from saryolo.evaluation.efficiency import profile_model, write_profile

    profile = profile_model(args.weights, imgsz=args.imgsz, device=args.device)
    print(json.dumps(profile, indent=2, default=float))
    if args.out:
        write_profile(profile, Path(args.out) / "efficiency.json")
    return 0


def _cmd_cross_dataset(args) -> int:
    from saryolo.evaluation.cross_dataset import cross_dataset_eval

    result = cross_dataset_eval(
        args.weights, args.target, source_data_yaml=args.source, imgsz=args.imgsz,
        device=args.device, out_dir=args.out, allow_partial_overlap=args.allow_partial_overlap,
    )
    print(json.dumps(result, indent=2, default=float))
    return 0


def _cmd_mine_hard(args) -> int:
    """Score the validation split by difficulty and write an oversampled train list."""
    from saryolo.evaluation.metrics import (
        load_yolo_ground_truth,
        load_yolo_predictions,
        predict_to_labels,
    )
    from saryolo.training.hard_examples import (
        rank_examples,
        score_examples,
        write_hard_data_config,
        write_oversampled_list,
    )
    from saryolo.visualization.error_analysis import image_contrast_map

    data = Path(args.data).resolve()
    if not data.exists():
        raise SystemExit(f"--data {data} does not exist")

    from saryolo.data.yolo import split_dirs

    # Hard examples must be mined from the split that will be trained on. Mining the
    # validation split and then oversampling those images would train on the evaluation data,
    # so `train` is the default and any other split has to be asked for explicitly.
    mined_dir, _ = split_dirs(data, args.split)
    train_images_dir, _ = split_dirs(data, "train")
    if not mined_dir.exists():
        raise SystemExit(f"{args.split} images not found at {mined_dir}")
    if args.split != "train":
        print(
            f"WARNING: mining the '{args.split}' split. Those images are now in the training "
            "list, so "
            f"'{args.split}' can no longer be used as an evaluation split for the model "
            "trained from it -- it is contaminated. Report metrics from a split that was "
            "never mined."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Inference is the expensive step and its output is reusable, so an existing run is
    # reused rather than silently repeated -- and when it is reused the caller is told, so
    # they are not left believing fresh predictions were produced.
    labels = out / "predictions" / "labels"
    if labels.exists() and any(labels.glob("*.txt")):
        print(f"reusing existing predictions from {labels}")
    else:
        labels = predict_to_labels(
            args.weights, mined_dir, imgsz=args.imgsz, conf=args.conf,
            out_dir=out / "predictions", device=args.device,
        )

    preds = load_yolo_predictions(labels, mined_dir)
    gts = load_yolo_ground_truth(data)
    if not gts:
        raise SystemExit(
            f"no ground truth found for the val split of {data}; "
            "difficulty scoring would be meaningless"
        )

    contrast = image_contrast_map(data, split=args.split, limit=args.contrast_limit)
    rows = score_examples(preds, gts, contrast_by_image=contrast)
    hard, easy = rank_examples(rows, top_frac=args.top_frac, min_count=args.min_count)

    out.mkdir(parents=True, exist_ok=True)
    (out / "difficulty.json").write_text(
        json.dumps({"n_images": len(rows), "hard": [r.as_row() for r in hard],
                    "all": [r.as_row() for r in rows]}, indent=2)
    )
    print(f"scored {len(rows)} images; {len(hard)} marked hard (top_frac={args.top_frac})")
    for row in hard[:10]:
        print(f"  {row.score:6.3f}  {row.image}  missed={row.missed} spurious={row.spurious}")

    # Only two things differ from a baseline run: the image list and the config pointing at it.
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    train_images = sorted(
        str(p) for p in train_images_dir.rglob("*") if p.suffix.lower() in extensions
    )
    # Detections are keyed by file *stem*, so the hard set is resolved back to a path through
    # the mined split's own index. A stem that resolves to nothing is an error rather than a
    # warning: dropping it would quietly reduce the mining to a no-op.
    mined_index = {p.stem: str(p) for p in mined_dir.rglob("*") if p.suffix.lower() in extensions}
    unresolved = [r.image for r in hard if r.image not in mined_index]
    if unresolved:
        raise SystemExit(
            f"{len(unresolved)} hard image(s) could not be mapped back to a file path under "
            f"{mined_dir}: {unresolved[:3]}. The training list would silently omit them."
        )
    hard_paths = [mined_index[r.image] for r in hard]
    if not train_images:
        raise SystemExit(
            f"no training images found under {train_images_dir}; the oversampled list would "
            "be empty and must not be used to train"
        )

    train_list = write_oversampled_list(train_images, hard_paths, out / "train_hard.txt", repeats=args.repeats)
    cfg = write_hard_data_config(data, train_list, out / "data_hard.yaml",
                                 note=f"source={data} split={args.split} repeats={args.repeats} "
                                      f"top_frac={args.top_frac}")
    # Reported from the file that was actually written, not from the intended arithmetic: the
    # two disagreed, and only one of them affects training.
    written = [line for line in train_list.read_text().splitlines() if line.strip()]
    extra = len(written) - len(train_images)
    print(f"\nwrote {train_list}: {len(written)} entries ({len(train_images)} base + {extra} repeats)")
    print(f"wrote {cfg}")
    if extra == 0:
        # Not a hard failure -- `repeats=1` legitimately means no oversampling -- but a run
        # that silently oversamples nothing is the baseline, and should not look like mining.
        print(
            "\nNOTE: no image was oversampled. This run is identical to the baseline unless "
            "the hard set was empty by choice."
        )
        return 1
    return 0


def _cmd_augment(args) -> int:
    """Build a multi-view, SAR-augmented copy of a training split (SEC. 5)."""
    from saryolo.augmentation.sar import AugmentationPlan, build_augmented_train_split

    images_dir = Path(args.images) if args.images else None
    if images_dir is None:
        from saryolo.data.yolo import split_dirs

        images_dir, _ = split_dirs(args.data, "train")
    if not images_dir.exists():
        raise SystemExit(f"training images not found at {images_dir}")

    plan = AugmentationPlan(
        kinds=tuple(args.kinds), views=args.views, include_clean=not args.no_clean, seed=args.seed,
        severities={k: tuple(v) for k, v in _parse_severities(args.severity).items()},
    )
    manifest = build_augmented_train_split(images_dir, args.out, plan, limit=args.limit)
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))
    for kind in plan.kinds:
        n = sum(1 for e in manifest["entries"] if e["corruption"] == kind)
        print(f"  {kind:<16} {n} images")
    if manifest["skipped"]:
        print(f"\n{len(manifest['skipped'])} image(s) skipped; see manifest.json for the reason")
    if manifest["n_emitted"] == 0:
        print("\nNothing was emitted; the destination must not be used for training.")
        return 1
    print(f"\nwrote {args.out} ({manifest['n_emitted']} images + manifest.json)")
    return 0


def _parse_severities(specs: list[str] | None) -> dict[str, tuple[float, ...]]:
    """Parse ``--severity speckle=8,4`` into ``{'speckle': (8.0, 4.0)}``."""
    out: dict[str, tuple[float, ...]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--severity expects KIND=v1,v2 (got {spec!r})")
        kind, values = spec.split("=", 1)
        try:
            out[kind.strip()] = tuple(float(v) for v in values.split(",") if v.strip())
        except ValueError as exc:
            raise SystemExit(f"--severity {spec!r} has a non-numeric value: {exc}") from exc
    return out


def _cmd_bench(args) -> int:
    """Architecture-level params/FLOPs benchmark; needs no training and no GPU."""
    from saryolo.evaluation.efficiency import profile_yaml
    from saryolo.nn.arch import VARIANTS, variant_filename

    rows = []
    unresolved = []
    for variant in args.variants:
        # Accept a variant key ('full_s'), a plain name ('full'), or an explicit path.
        path = Path(args.models) / f"{variant}.yaml"
        if not path.exists() and variant in VARIANTS:
            path = Path(args.models) / variant_filename(VARIANTS[variant])
        if not path.exists():
            candidates = sorted(Path(args.models).glob(f"*{variant}*.yaml"))
            if not candidates:
                # Collected rather than printed-and-forgotten: a mistyped variant must not
                # leave the caller with a success status and a shorter table than asked for.
                unresolved.append(variant)
                print(f"UNRESOLVED {variant}: no yaml under {args.models}")
                continue
            path = candidates[0]
        row = profile_yaml(path, imgsz=args.imgsz, nc=args.nc)
        rows.append(row)
        print(f"{variant:<20} params={row['params_M']:>8.3f}M  GFLOPs={row['flops_G']}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "architecture_benchmark.json").write_text(json.dumps(rows, indent=2, default=float))
    if unresolved:
        print(f"\n{len(unresolved)} requested variant(s) could not be resolved: {unresolved}")
        return 1
    return 0


def _cmd_assets(args) -> int:
    """Generate the paper's tables (Markdown + LaTeX) from measured results only."""
    from saryolo.paper import (
        build_ablation,
        build_baseline_comparison,
        build_efficiency,
        build_module_ablation,
        build_multi_seed,
        build_removal_ablation,
        build_robustness,
        build_scale_analysis,
        write_tables,
    )
    from saryolo.tracking.ledger import ExperimentLedger

    ledger = ExperimentLedger(args.ledger)
    robustness_root = Path(args.robustness)
    tables = [
        build_baseline_comparison(ledger),
        build_ablation(ledger),
        build_module_ablation(ledger),
        build_removal_ablation(ledger),
        build_scale_analysis(ledger),
        build_robustness(robustness_root / "robustness.json", robustness_root / "robustness_baseline.json"),
        build_efficiency(ledger),
        build_multi_seed(ledger),
    ]
    written = write_tables(tables, args.out, assert_complete=args.require_complete)
    for table in tables:
        state = "complete" if table.complete else "INCOMPLETE (TBD cells)"
        print(f"{table.name:<24} {state}")
    print(f"\nwrote {len(written)} files to {args.out}")
    print("Unmeasured cells render as TBD; nothing is filled in automatically.")
    return 0


def _cmd_ledger(args) -> int:
    from saryolo.tracking.ledger import ExperimentLedger

    ledger = ExperimentLedger(args.ledger)
    records = ledger.load()
    if not records:
        print("ledger is empty")
        return 0
    print(f"{'id':<10} {'model':<22} {'dataset':<18} {'status':<10} {'mAP50':>7} {'mAP50-95':>9} {'seed':>5}")
    for r in records:
        m50 = r.metrics.get("mAP50")
        m = r.metrics.get("mAP50_95")
        print(
            f"{r.experiment_id:<10} {r.model:<22} {r.dataset:<18} {r.status:<10} "
            f"{(f'{m50:.4f}' if m50 is not None else 'TBD'):>7} "
            f"{(f'{m:.4f}' if m is not None else 'TBD'):>9} {r.train_seed:>5}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="saryolo", description="SAR-YOLO research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("datasets", help="list supported SAR datasets")
    p.set_defaults(func=_cmd_datasets)

    p = sub.add_parser("synth-data", help="generate a synthetic dataset for PIPELINE TESTING")
    p.add_argument("--out", default="datasets/processed/synthetic_smoke")
    p.add_argument("--train", type=int, default=64)
    p.add_argument("--val", type=int, default=16)
    p.add_argument("--test", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--classes", nargs="+", default=["target"])
    p.add_argument("--looks", type=float, default=6.0)
    p.add_argument("--clutter", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_synth)

    p = sub.add_parser("check-data", help="validate a processed YOLO dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--report", default="validation_reports")
    p.set_defaults(func=_cmd_check_data)

    p = sub.add_parser("stats", help="dataset statistics and figures")
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="dataset_statistics")
    p.add_argument("--sample", type=int, default=300)
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser("arch", help="emit SAR-YOLO model YAMLs")
    p.add_argument("--variant", default="all")
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--out", default="configs/models")
    p.set_defaults(func=_cmd_arch)

    p = sub.add_parser("train", help="train one experiment config")
    p.add_argument("--exp", required=True)
    p.add_argument("--ledger", default="results")
    p.add_argument("--project", default="results/runs")
    p.add_argument("--device", default=None)
    p.add_argument("--overrides", default=None, help="JSON dict of extra Ultralytics args")
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--no-ledger", action="store_true")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="evaluate a checkpoint (overall + scale-wise AP)")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/eval")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("robustness", help="corruption robustness sweep")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--corruptions", nargs="+",
                   default=["speckle", "low_contrast", "blur", "low_resolution", "clutter"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/robustness")
    p.set_defaults(func=_cmd_robustness)

    p = sub.add_parser("efficiency", help="params/FLOPs/latency/FPS for a checkpoint")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_efficiency)

    p = sub.add_parser("cross-dataset", help="domain-shift evaluation on another dataset")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", default=None, help="training dataset data.yaml (enables the class guard)")
    p.add_argument("--target", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/generalization")
    p.add_argument("--allow-partial-overlap", action="store_true")
    p.set_defaults(func=_cmd_cross_dataset)

    p = sub.add_parser("mine-hard", help="score val images by difficulty and write an oversampled train list")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="data.yaml of the dataset the checkpoint was trained on")
    p.add_argument("--split", default="train",
                   help="split to mine; must be the one that will be trained on (default: train)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--top-frac", type=float, default=0.2, dest="top_frac")
    p.add_argument("--min-count", type=int, default=0, dest="min_count")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--contrast-limit", type=int, default=None, dest="contrast_limit")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/hard_examples")
    p.set_defaults(func=_cmd_mine_hard)

    p = sub.add_parser("augment", help="build a multi-view SAR-augmented training split (SEC. 5)")
    p.add_argument("--data", required=True, help="data.yaml to read the train split from")
    p.add_argument("--images", default=None, help="override: images directory to augment")
    p.add_argument("--kinds", nargs="+", default=["speckle", "low_contrast", "blur", "low_resolution", "low_snr"])
    p.add_argument("--views", type=int, default=2, help="augmented copies per image (clean copy is extra)")
    p.add_argument("--severity", nargs="*", default=None, help="restrict a grid, e.g. speckle=8,4")
    p.add_argument("--no-clean", action="store_true", help="omit the undeformed copy")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="datasets/augmented/train")
    p.set_defaults(func=_cmd_augment)

    p = sub.add_parser("bench", help="architecture-level params/FLOPs table (no training)")
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--models", default="configs/models")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--out", default="results/tables")
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("ledger", help="show the experiment ledger")
    p.add_argument("--ledger", default="results")
    p.set_defaults(func=_cmd_ledger)

    p = sub.add_parser("assets", help="generate paper tables from measured results")
    p.add_argument("--ledger", default="results")
    p.add_argument("--robustness", default="results/robustness")
    p.add_argument("--out", default="paper/tables")
    p.add_argument("--require-complete", action="store_true",
                   help="Fail if any cell is unmeasured (use when preparing a submission)")
    p.set_defaults(func=_cmd_assets)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
