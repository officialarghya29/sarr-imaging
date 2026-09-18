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


def _cmd_check_data(args) -> int:
    from saryolo.data import validate_yolo_dataset, write_report

    dataset = Path(args.dataset)
    failed = False
    for split in args.splits:
        images, labels = dataset / "images" / split, dataset / "labels" / split
        if not images.exists():
            continue
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
    for split in args.splits:
        images, labels = dataset / "images" / split, dataset / "labels" / split
        if not images.exists():
            continue
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


def _cmd_bench(args) -> int:
    """Architecture-level params/FLOPs benchmark; needs no training and no GPU."""
    from saryolo.evaluation.efficiency import profile_yaml
    from saryolo.nn.arch import VARIANTS, variant_filename

    rows = []
    for variant in args.variants:
        # Accept a variant key ('full_s'), a plain name ('full'), or an explicit path.
        path = Path(args.models) / f"{variant}.yaml"
        if not path.exists() and variant in VARIANTS:
            path = Path(args.models) / variant_filename(VARIANTS[variant])
        if not path.exists():
            candidates = sorted(Path(args.models).glob(f"*{variant}*.yaml"))
            if not candidates:
                print(f"skip {variant}: no yaml under {args.models}")
                continue
            path = candidates[0]
        row = profile_yaml(path, imgsz=args.imgsz, nc=args.nc)
        rows.append(row)
        print(f"{variant:<20} params={row['params_M']:>8.3f}M  GFLOPs={row['flops_G']}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "architecture_benchmark.json").write_text(json.dumps(rows, indent=2, default=float))
    return 0


def _cmd_assets(args) -> int:
    """Generate the paper's tables (Markdown + LaTeX) from measured results only."""
    from saryolo.paper import (
        build_ablation,
        build_baseline_comparison,
        build_efficiency,
        build_module_ablation,
        build_multi_seed,
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
