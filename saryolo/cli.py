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

from saryolo.data.groups import IMAGE_SUFFIXES


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


def _cmd_metadata(args) -> int:
    """Build an acquisition-metadata table from a per-image CSV sidecar.

    The missing third input of the conditioning experiments. The ``--rule resolution``
    branch of ``loso`` needs a saved :class:`MetadataTable` JSON, and ``build_metadata_table``
    provides no CLI route to one for archives that state acquisition only in a table (SAR-Ship-Dataset,
    chip releases of SARDet-100K). This command is that route, and is deliberately small:
    the CSV is stated, never scraped from filenames, because a scraped value would look like
    metadata while being a guess.
    """
    from saryolo.data.metadata import build_metadata_table, load_metadata_sidecar

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise SystemExit(f"--images {images_dir} is not a directory")
    sidecar_path = Path(args.sidecar)
    if not sidecar_path.is_file():
        raise SystemExit(f"--sidecar {sidecar_path} does not exist")
    sidecar = load_metadata_sidecar(sidecar_path)
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no dataset images found under {images_dir} (tried {', '.join(IMAGE_SUFFIXES)})")
    try:
        table = build_metadata_table(images, sidecar=sidecar)
    except ValueError as exc:
        # An all-unknown table (sidecar matching no stems) raises ValueError; a traceback
        # would still point at the cause, but SystemExit keeps every CLI refusal in the
        # same shape the loso command uses.
        raise SystemExit(f"cannot build the metadata table: {exc}") from None
    out = table.save(args.out)
    print(f"wrote {out}: {len(table)} images, coverage "
          + ", ".join(f"{field}={cov:.0%}" for field, cov in table.coverage().items()))
    return 0


def _cmd_probe(args) -> int:
    """Representation diagnosis on a frozen checkpoint (§14 of the master plan).

    The evidence step the conditioning claim rests on: hook the head's per-level
    maps, pool each image, and measure what a *linear* probe can recover from the
    frozen features alone. A ``--field sensor`` probe far above chance — with a
    ``--class-field class`` probe that is not — says the representation is
    dominated by acquisition appearance, which is the failure the adapter exists
    to address. On an untrained checkpoint the numbers are placeholders; this is
    diagnosis, not evidence, until a trained checkpoint exists.
    """
    import numpy as np
    import torch
    import yaml as pyyaml
    from PIL import Image

    from saryolo.evaluation.probes import (
        FeatureExtraction,
        collect_head_features,
        representation_report,
    )
    from saryolo.training.trainer import load_model

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"--weights {weights} does not exist")
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"--data {data_path} does not exist")
    cfg = pyyaml.safe_load(data_path.read_text())
    root = Path(cfg.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    split_dir = root / cfg.get(args.split, "images/" + args.split)
    if not split_dir.is_dir():
        raise SystemExit(
            f"split '{args.split}' resolves to {split_dir}, which does not exist; "
            "the data config's paths and 'path' key must describe this layout"
        )

    names = cfg.get("names") or [f"class_{i}" for i in range(int(cfg.get("nc", 1)))]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    table = None
    if args.metadata:
        from saryolo.data.metadata import MetadataTable

        table_path = Path(args.metadata)
        if not table_path.exists():
            raise SystemExit(f"--metadata {table_path} does not exist")
        table = MetadataTable.load(table_path)

    extension_set = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(
        p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in extension_set
    )
    if not images:
        raise SystemExit(f"no dataset images found under {split_dir}")
    if args.limit and len(images) > args.limit:
        # First N rather than a random subset: deterministic, and the manifest records
        # exactly which images were probed instead of a seed pretending it was a draw.
        images = images[: args.limit]

    # ---- acquisition field labels (the probe target) ----
    if table is not None:
        values: list[str | None] = [getattr(table.get(p.stem), args.field) for p in images]
    elif args.stem_pattern:
        import re

        pattern = re.compile(args.stem_pattern)
        values = []
        for p in images:
            m = pattern.search(p.stem)
            values.append(m.group(1) if m and m.re.groups >= 1 else None)
    else:
        raise SystemExit(
            "give --metadata (a table built by `saryolo metadata`/write_acquisition_metadata) "
            "or --stem-pattern '(...)'; without one the probe has nothing to predict"
        )
    known = [v for v in values if v is not None]
    if len(set(known)) < 2:
        raise SystemExit(
            f"field '{args.field}' has {len(set(known))} distinct known value(s) across "
            f"{len(images)} images; a probe needs >= 2 (a one-class probe would report "
            "perfect accuracy while measuring nothing)"
        )
    vocab = {v: i for i, v in enumerate(sorted(set(known)))}
    field_labels = torch.tensor(
        [vocab.get(v, -1) for v in values], dtype=torch.long
    )

    # ---- class labels (the contrast probe) ----
    class_labels: torch.Tensor | None = None
    class_names_present: dict[int, str] = {}
    if args.class_field:
        from saryolo.data.yolo import load_data_config, split_dirs

        try:
            _, data_cfg = load_data_config(data_path)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            raise SystemExit(f"cannot read the data config for class labels: {exc}") from None
        try:
            _images_dir, labels_dir = split_dirs(root, args.split)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"cannot resolve the label split for class labels: {exc}") from None
        class_ids: list[int] = []
        for p in images:
            label_path = labels_dir / (p.stem + ".txt")
            first = None
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    if line.strip():
                        first = int(line.split()[0])
                        break
            class_ids.append(first if first is not None else -1)
        class_labels = torch.tensor(class_ids, dtype=torch.long)
        class_names_present = {int(c): names[c] for c in sorted(set(class_ids)) if c >= 0 and c < len(names)}

    # ---- feature extraction on the frozen checkpoint ----
    model = load_model(str(weights))
    inner = model.model if hasattr(model, "model") else model
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    inner = inner.to(device).eval()

    per_level_features: dict[int, list] = {}
    per_level_labels: dict[str, list] = {}
    batch_size = args.batch
    level_shapes: dict[int, tuple[int, int]] = {}
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        arr = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in chunk])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
        # Letterbox to a common size so a batch can mix image dimensions; the probe
        # pools over space anyway, but the model graph needs a rectangular input.
        x = _letterbox_batch(x, args.imgsz)
        chunk_field = field_labels[start : start + batch_size]
        chunk_labels = {args.field: chunk_field}
        if args.class_field:
            chunk_labels["class"] = class_labels[start : start + batch_size]
        extraction = collect_head_features(inner, x, labels=chunk_labels)
        level_shapes = extraction.level_shapes
        for level in extraction.levels():
            per_level_features.setdefault(level, []).append(extraction.features[level])
        for name in chunk_labels:
            per_level_labels.setdefault(name, []).append(chunk_labels[name])

    full = FeatureExtraction(
        features={lvl: torch.cat(parts) for lvl, parts in sorted(per_level_features.items())},
        labels={name: torch.cat(parts) for name, parts in sorted(per_level_labels.items())},
        level_shapes=level_shapes,
    )
    if not full.features:
        raise SystemExit("no per-level features were collected; the forward pass produced nothing readable")

    report = representation_report(full, class_field="class" if args.class_field else None, seed=args.seed)
    report["checkpoint"] = str(weights)
    report["data"] = str(data_path)
    report["split"] = args.split
    report["n_images_used"] = len(images)
    report["field"] = args.field
    report["field_vocab"] = {v: k for k, v in vocab.items()}
    if class_names_present:
        report["class_vocab"] = {str(k): v for k, v in class_names_present.items()}
    report["chance_rate"] = 1.0 / len(vocab)
    report["note"] = (
        "diagnosis, not evidence: on an untrained checkpoint these numbers are placeholders; "
        "compare probe accuracy against the stated chance rate, not against 1.0"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "representation_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"probe -> {out / 'representation_report.json'}")
    for level in sorted(report["levels"]):
        for key, val in report["levels"][level].items():
            if isinstance(val, dict) and val.get("accuracy") is not None:
                print(f"  level {level}  {key}: acc={val['accuracy']:.3f}  (chance {report['chance_rate']:.3f})")
    if "within_class_group_distance" in report:
        dist = report["within_class_group_distance"]
        if "reason" not in dist:
            pairs = ", ".join(f"{k}={v:.3f}" for k, v in sorted(dist.items()))
            print(f"  within-class cross-group drift: {pairs}")
    if "linear_cka_top_level" in report:
        cka = report["linear_cka_top_level"]
        if cka and "reason" not in cka:
            pairs = ", ".join(f"{k}={v:.3f}" for k, v in sorted(cka.items()) if v is not None)
            print(f"  linear CKA (top level): {pairs}")
    return 0


def _letterbox_batch(x, size: int):
    """Resize a ``(n, C, H, W)`` batch to ``(n, C, size, size)`` (probe-only pooling makes this safe)."""
    import torch

    return torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


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


def _cmd_loso(args) -> int:
    """Build leave-one-source-out folds for the cross-source generalisation claim."""
    from saryolo.data.groups import (
        IMAGE_SUFFIXES,
        SourceRule,
        discover_sources,
        leave_one_out_folds,
        write_loso_splits,
    )

    images_root = Path(args.images).resolve()
    if not images_root.exists():
        raise SystemExit(f"--images {images_root} does not exist")

    paths = sorted(
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(
            f"no images with {IMAGE_SUFFIXES} found under {images_root}; point --images at "
            f"the directory holding the images, not the dataset root"
        )

    if args.rule == "parent":
        root = Path(args.root).resolve() if args.root else images_root
        rule = SourceRule.parent(root, depth=args.depth)
    elif args.rule == "regex":
        if not args.pattern:
            raise SystemExit("--rule regex requires --pattern (with one capture group)")
        rule = SourceRule.regex(args.pattern)
    elif args.rule == "resolution":
        # The cross-resolution protocol. Both inputs are stated rather than inferred: the bin
        # edges are a modelling choice, and the metadata table is where the resolution lives.
        # Guessing either would produce a run that "completes" while measuring nothing.
        from saryolo.data.groups import binned_rule
        from saryolo.data.metadata import MetadataTable

        if not args.metadata:
            raise SystemExit(
                "--rule resolution requires --metadata (a metadata table JSON); resolution is "
                "not in the filename, so it has to be read from the table built by "
                "`python -m saryolo datasets`/`saryolo.data.metadata`"
            )
        table_path = Path(args.metadata).resolve()
        if not table_path.exists():
            raise SystemExit(f"--metadata {table_path} does not exist")
        if not args.edges:
            raise SystemExit(
                "--rule resolution requires --edges, e.g. --edges 5,10,20. There is no default: "
                "the bin width is a modelling choice that decides what 'cross-resolution' means"
            )
        try:
            edges = [float(e) for e in str(args.edges).replace(",", " ").split()]
        except ValueError:
            raise SystemExit(f"--edges {args.edges!r} is not a list of numbers") from None

        table = MetadataTable.load(table_path)
        values = {
            stem: getattr(table.get(stem), "resolution_m", None)
            for stem in table.entries
        }
        # Only images that are actually present count. A metadata row for an image outside
        # --images would otherwise populate a bin that has nothing to evaluate on.
        present = {p.stem for p in paths}
        values = {stem: value for stem, value in values.items() if stem in present}
        absent = sorted(present - set(values))
        if absent:
            raise SystemExit(
                f"{len(absent)} image(s) have no row in {table_path.name} "
                f"(e.g. {absent[:3]}). They cannot be binned, so they would be absent from "
                f"every fold without anything reporting it. Rebuild the metadata table over "
                f"this directory."
            )
        # A bad binning is a user error, not a crash: the ValueError carries the diagnosis, and
        # a traceback would bury it. Same convention as the fold-building path below.
        try:
            rule = binned_rule(
                values, edges, field="resolution_m", allow_missing=args.allow_unmatched
            )
        except ValueError as exc:
            raise SystemExit(f"cannot build a resolution rule: {exc}") from None
    else:
        if not args.sidecar:
            raise SystemExit("--rule sidecar requires --sidecar (a JSON image -> source mapping)")
        side = Path(args.sidecar).resolve()
        if not side.exists():
            raise SystemExit(f"--sidecar {side} does not exist")
        rule = SourceRule.sidecar(json.loads(side.read_text()))

    # A bad rule is a user error, not a crash: report what was found and how to fix it
    # without a traceback, so the message is the whole diagnosis.
    try:
        groups = discover_sources(paths, rule, allow_unmatched=args.allow_unmatched)
        print(groups.summary())
        folds = leave_one_out_folds(
            groups, val_ratio=args.val_ratio, seed=args.seed,
            min_test_images=args.min_test_images,
        )
    except ValueError as exc:
        raise SystemExit(f"cannot build leave-one-source-out folds: {exc}") from None
    # Class names come from the dataset's own data config, so a fold config is directly
    # trainable and cannot disagree with the dataset about the label space.
    names: list[str] | None = None
    if args.data:
        from saryolo.data.yolo import load_data_config

        data_cfg_path = Path(args.data).resolve()
        if not data_cfg_path.exists():
            raise SystemExit(f"--data {data_cfg_path} does not exist")
        _, data_cfg = load_data_config(data_cfg_path)
        resolved = data_cfg.get("names") or [f"class_{i}" for i in range(int(data_cfg.get("nc", 1)))]
        names = [resolved[k] for k in sorted(resolved)] if isinstance(resolved, dict) else list(resolved)

    out_dir = write_loso_splits(folds, args.out, groups=groups, seed=args.seed, names=names)

    print(f"\nleave-one-source-out folds -> {out_dir}")
    print(f"  {'held-out source':<28}{'train':>8}{'val':>8}{'test':>8}")
    for fold in folds:
        sizes = fold.sizes
        print(f"  {fold.name:<28}{sizes['train']:>8}{sizes['val']:>8}{sizes['test']:>8}")

    if args.leakage:
        # Source-grouped folds remove *source* leakage by construction. Duplicate chips
        # shared across sources are a separate risk and are not addressed by grouping, so
        # this is opt-in: it hashes every chip in the fold and is quadratic in the pair
        # comparison, which is far too slow to run by default on a benchmark-tier dataset.
        from saryolo.data.splits import leakage_report

        for fold in folds:
            splits = {"train": list(fold.train), "val": list(fold.val), "test": list(fold.test)}
            report = leakage_report(splits, images_root)
            print(report.summary())

    print(
        f"\nnote: validation shares sources with training in every fold, so it drives early "
        f"stopping only. Report numbers from test.txt ({out_dir}/manifest.json records the "
        f"rule and the per-fold sizes)."
    )
    return 0


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

    p = sub.add_parser("metadata", help="build an acquisition-metadata table from a per-image CSV sidecar")
    p.add_argument("--images", required=True, dest="images_dir",
                   help="directory of dataset images the table describes")
    p.add_argument("--sidecar", required=True,
                   help="CSV with a stem column plus acquisition fields (sensor, resolution_m, ...)")
    p.add_argument("--out", default="datasets/metadata.json")
    p.set_defaults(func=_cmd_metadata)

    p = sub.add_parser("probe", help="representation diagnosis: linear probes on frozen features (master-plan §14)")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="data.yaml of the dataset to probe on")
    p.add_argument("--field", default="sensor", help="metadata field the probe predicts (default: sensor)")
    p.add_argument("--class-field", action="store_true", dest="class_field",
                   help="also probe the object class (from the label files) as the contrast task")
    p.add_argument("--metadata", default=None,
                   help="metadata table JSON keyed by image stem (saryolo.data.metadata)")
    p.add_argument("--stem-pattern", default=None, dest="stem_pattern",
                   help="alternative to --metadata: regex with one capture group applied to each stem")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="probe only the first N images (deterministic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/probes")
    p.set_defaults(func=_cmd_probe)

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

    p = sub.add_parser("loso", help="leave-one-source-out folds for cross-source generalisation")
    p.add_argument("--images", required=True, help="directory holding the images to group by source")
    p.add_argument("--data", default=None,
                   help="dataset data.yaml; supplies class names so each fold gets a runnable config")
    p.add_argument("--rule", default="parent", choices=["parent", "regex", "sidecar", "resolution"],
                   help="how the source key is derived; stated, never guessed (default: parent). "
                        "'resolution' bins a continuous metadata field -- the cross-resolution "
                        "protocol, which 'parent'/'regex' cannot express")
    p.add_argument("--root", default=None, help="for --rule parent: root the key is relative to")
    p.add_argument("--depth", type=int, default=1, help="for --rule parent: path components kept")
    p.add_argument("--pattern", default=None, help="for --rule regex: pattern with one capture group")
    p.add_argument("--sidecar", default=None, help="for --rule sidecar: JSON image -> source mapping")
    p.add_argument("--metadata", default=None, dest="metadata",
                   help="for --rule resolution: metadata table JSON (saryolo.data.metadata)")
    p.add_argument("--edges", default=None, dest="edges",
                   help="for --rule resolution: comma-separated ascending bin edges in metres, "
                        "e.g. '5,10,20'; bins are unbounded at both ends")
    p.add_argument("--allow-unmatched", action="store_true",
                   help="permit unkeyed images (default: refuse, they would silently shrink the test set)")
    p.add_argument("--val-ratio", type=float, default=0.2, dest="val_ratio")
    p.add_argument("--min-test-images", type=int, default=30, dest="min_test_images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--leakage", action="store_true", help="also run the duplicate check per fold (slow)")
    p.add_argument("--out", default="datasets/splits/loso")
    p.set_defaults(func=_cmd_loso)

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
