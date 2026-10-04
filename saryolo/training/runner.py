"""Experiment runner: train one config and record it in the ledger.

The runner is the only place that writes an experiment row. It records a
``running`` row *before* training so a crash mid-run is still visible, and only
writes measured metrics afterwards.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from saryolo.tracking.ledger import ExperimentLedger, ExperimentRecord

from .config import ExperimentConfig, load_experiment
from .peft import LoRAConfig
from .trainer import apply_init, load_model

__all__ = [
    "run_experiment",
    "read_results_csv",
    "extract_val_metrics",
    "find_best_weights",
    "validate_peft",
    "training_overrides",
]


def validate_peft(payload) -> dict | None:
    """Validate a run's ``peft`` block, returning the mapping the trainer should receive.

    ``None`` (no ``peft`` key) means a full fine-tune, which is the default and needs no
    summary -- reporting ``{"method": "none"}`` for every ordinary run would add noise to
    the ledger without adding a fact. Any other value is checked here rather than at train
    time, so a typo in the block costs one command instead of a wasted run, and the validated
    block is then passed to ``SARYOLOTrainer`` through the training overrides.

    Why the adapter is not injected here
    ------------------------------------
    It used to be, and the first real LoRA arm proved it wrong. ``YOLO.train()`` rebuilds the
    model from its config inside the trainer whenever the facade has no checkpoint, so an
    adapter injected into the loaded facade is discarded before the first optimiser step --
    while the summary built here was still written to the ledger, recording a parameter-
    efficient arm that was in fact a full fine-tune. The injection therefore lives in
    ``SARYOLOTrainer.get_model``, where it is applied to the model that is actually trained and
    before the optimiser is built; the runner only validates the block and forwards it.
    """
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"the 'peft' block must be a mapping, got {type(payload).__name__}")
    method = str(payload.get("method", "lora")).lower()
    if method not in ("lora", "none"):
        raise ValueError(
            f"unsupported peft method {method!r}; supported: 'lora', 'none'. "
            f"Add the method to saryolo/training/peft.py rather than spelling it differently here."
        )
    if method == "none":
        return None
    LoRAConfig.from_mapping({k: v for k, v in payload.items() if k != "method"})
    return dict(payload)


def training_overrides(cfg: ExperimentConfig, extra_overrides: dict | None = None,
                       device: str | None = None) -> dict:
    """Assemble the Ultralytics training arguments for one run.

    A named function rather than inline code because two of its entries are translations rather
    than pass-through, and each kind of mistake here is silent:

    * ``data_fraction`` -> the library's own ``fraction`` (never the private key: see the
      comment at the translation for the measured failure that produced this rule);
    * ``peft`` -> forwarded for the trainer to apply, validated first so a bad block costs one
      command instead of a run.

    Kept importable and side-effect free so tests can assert the mapping without training.
    """
    overrides = dict(cfg.train)
    overrides.update(extra_overrides or {})
    if device:
        overrides["device"] = device
    if cfg.data_fraction < 1.0:
        # Direction E (data efficiency). The sweep is a property of the *run*, not of the
        # dataset, so it is stated in the config -- and it becomes Ultralytics' own `fraction`.
        #
        # It was first implemented as a private `data_fraction` override handled by this
        # repository's trainer, on the stated belief that "ultralytics shuffles the image list
        # with its own seed", so the subset had to be drawn here to be reproducible. That belief
        # was wrong, and the consequence was measured: for a *stock* model YAML the run resolves
        # to the plain `YOLO` facade, this repository's trainer never sees the override, and
        # Ultralytics refuses it with "'data_fraction' is not a valid YOLO argument" -- so every
        # baseline arm of the sweep, the arm the sweep is measured against, could not run at
        # all. ``BaseDataset.get_img_files`` sorts the file list and takes a prefix, which is
        # deterministic and nested at every fraction, and ``get_split_fraction`` applies it to
        # ``train`` only unless the caller passes a per-split list. Those are exactly the
        # properties the private override existed to provide, so it was redundant as well as
        # broken; the native argument works for both facades and is the behaviour a reader
        # already has. The nesting is pinned by
        # ``tests/test_peft.py::test_the_native_fraction_is_nested_and_deterministic``, because a
        # library upgrade that changed the selection order would silently turn the sweep into a
        # comparison of independent draws.
        overrides["fraction"] = cfg.data_fraction
    # The adapter arm is validated up front -- a bad block costs one command instead of a run --
    # and then forwarded through the trainer, which is where it is applied: see `validate_peft`.
    peft_payload = validate_peft(cfg.extra.get("peft"))
    if peft_payload is not None:
        overrides["peft"] = peft_payload
    return overrides


def _set_seed(seed: int) -> None:
    """Seed python/numpy/torch for reproducible runs (best effort)."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_results_csv(path: str | Path) -> list[dict]:
    """Read an Ultralytics ``results.csv`` into a list of dicts (stripped keys)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as fh:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(fh)]


def find_best_weights(save_dir: str | Path, prefer: str = "best") -> Path | None:
    """Locate ``best.pt``/``last.pt`` under a run directory."""
    save_dir = Path(save_dir)
    for name in (f"{prefer}.pt", "last.pt", "best.pt"):
        candidate = save_dir / name
        if candidate.exists():
            return candidate
    found = sorted(save_dir.rglob("*.pt"))
    return found[0] if found else None


def extract_val_metrics(metrics) -> dict:
    """Pull the standard detection metrics out of an Ultralytics metrics object.

    Returns only what was actually produced; absent fields stay ``None`` so
    downstream table generation marks them ``TBD`` instead of inventing a value.
    """
    out: dict = {"mAP50": None, "mAP50_95": None, "precision": None, "recall": None, "f1": None, "per_class_AP50_95": None}
    box = getattr(metrics, "box", None)
    if box is None:
        return out
    out["mAP50"] = _f(getattr(box, "map50", None))
    out["mAP50_95"] = _f(getattr(box, "map", None))
    out["precision"] = _f(getattr(box, "mp", None))
    out["recall"] = _f(getattr(box, "mr", None))
    if out["precision"] is not None and out["recall"] is not None:
        denom = out["precision"] + out["recall"]
        out["f1"] = (2 * out["precision"] * out["recall"] / denom) if denom > 0 else 0.0
    maps = getattr(box, "maps", None)
    if maps is not None:
        out["per_class_AP50_95"] = [float(v) for v in maps]
    # Ultralytics exposes scale-wise AP when the validator is asked for it.
    for key, attr in (("AP_small", "ap_small"), ("AP_medium", "ap_medium"), ("AP_large", "ap_large")):
        value = getattr(box, attr, None)
        if value is not None:
            out[key] = _f(value)
    return out


def _f(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def run_experiment(
    config_path: str | Path,
    ledger_root: str | Path = "results",
    project: str | Path = "results/runs",
    device: str | None = None,
    extra_overrides: dict | None = None,
    validate_after: bool = True,
    efficiency: bool = True,
    skip_ledger: bool = False,
) -> ExperimentRecord:
    """Train one experiment, record it, and return the ledger row.

    Args:
        config_path: Path to an ``EXP-*.yaml`` experiment config.
        ledger_root: Where ``experiments.jsonl`` lives.
        project: Ultralytics run directory root.
        device: Torch device string; ``None`` lets Ultralytics choose.
        extra_overrides: Additional Ultralytics training arguments.
        validate_after: Run a final validation on the best checkpoint and record metrics.
        efficiency: Measure params/FLOPs/FPS/latency after training.
        skip_ledger: Return the record without writing it (useful for smoke tests).

    Returns:
        The completed (or failed) :class:`ExperimentRecord`.
    """
    cfg: ExperimentConfig = load_experiment(config_path)
    ledger = ExperimentLedger(ledger_root)
    _set_seed(cfg.seed)

    overrides = training_overrides(cfg, extra_overrides, device)
    peft_payload = overrides.get("peft")
    # Absolute paths keep ultralytics from nesting runs under `runs/detect/<project>`.
    project = Path(project).resolve()

    record = ExperimentRecord(
        experiment_id=cfg.experiment_id,
        model=cfg.model_path.name,
        dataset=cfg.dataset_path.name,
        train_seed=cfg.seed,
        epochs=int(overrides.get("epochs", 0)),
        batch=int(overrides.get("batch", 0)) if str(overrides.get("batch", "")).lstrip("-").isdigit() else None,
        imgsz=cfg.imgsz,
        optimizer=str(overrides.get("optimizer", "auto")),
        lr0=_f(overrides.get("lr0")),
        notes=cfg.notes or cfg.description,
    )
    if not skip_ledger:
        record = ledger.start(cfg.experiment_id, record.model, record.dataset, cfg.to_dict(),
                              train_seed=cfg.seed, epochs=record.epochs, batch=record.batch,
                              imgsz=record.imgsz, optimizer=record.optimizer, lr0=record.lr0,
                              notes=record.notes)

    started = time.time()
    try:
        # Ultralytics resolves a relative `path` against its global datasets_dir, not the
        # YAML's own directory, so the data config is resolved explicitly and portably first.
        from saryolo.data.yolo import resolve_data_yaml
        from saryolo.training.trainer import metadata_augmentation_overrides

        data_yaml = resolve_data_yaml(cfg.dataset_path)
        # The detector and conditioning ablations must see identical geometric/photometric
        # pipelines. For an acquisition-conditioned dataset only, mixing transforms are
        # removed because a composite image cannot truthfully carry one acquisition label.
        overrides.update(metadata_augmentation_overrides(data_yaml))
        # A parameter-efficient arm must use the custom trainer even when its graph is a stock
        # YAML (that is exactly what makes it a fair comparison: same graph as the full
        # fine-tune, different thing optimised), because the adapter is injected there.
        model = load_model(str(cfg.model_path), prefer_custom=peft_payload is not None)
        # Phase 3: the init stage is stated in the config (default `none`, i.e. a fresh
        # build from the model YAML) and recorded, so a fine-tuned number can never be
        # quoted as if it were trained from scratch -- or the reverse. Both config
        # spellings are accepted: `init: coco11` or `init: {weights: coco11}`.
        init_cfg = cfg.extra.get("init", "none")
        init_weights = init_cfg if isinstance(init_cfg, str) else str(init_cfg.get("weights", "none"))
        init_summary = apply_init(model, init_weights)
        record.extra.update(init_summary)
        if init_summary.get("init_checkpoint"):
            # The facade must be reloaded from the init checkpoint: Ultralytics' train()
            # calls get_model(weights=self.model if self.ckpt else None), so a YAML-built
            # facade's inner model would be silently rebuilt and any transfer lost. A
            # checkpoint-built facade takes the standard .pt fine-tuning path instead.
            model = load_model(init_summary["init_checkpoint"], prefer_custom=peft_payload is not None)
        model.train(
            data=str(data_yaml),
            project=str(project),
            name=cfg.experiment_id,
            exist_ok=True,
            **overrides,
        )
        train_minutes = (time.time() - started) / 60.0
        save_dir = Path(getattr(model.trainer, "save_dir", project) or project)
        # The adapter summary is read back off the trainer, because the trainer is what applied
        # it. Reading it from anywhere else would let the two disagree, and the disagreement
        # this prevents was measured: a LoRA arm whose ledger claimed 87 wrapped layers while
        # the trained model contained no `lora_` tensor at all.
        if peft_payload is not None:
            peft_summary = getattr(getattr(model, "trainer", None), "peft_summary", None) or {}
            if not peft_summary:
                raise RuntimeError(
                    "this run requested a parameter-efficient method but the trainer reported no "
                    "adapter; the trained model is not the model the config describes"
                )
            record.extra.update(peft_summary)
        record.save_dir = str(save_dir)
        record.extra["train_minutes"] = round(train_minutes, 2)

        # Training-curve summary, straight from the run's own results.csv.
        epochs_rows = read_results_csv(save_dir / "results.csv")
        record.extra["epochs_completed"] = len(epochs_rows)

        metrics: dict = {}
        best = find_best_weights(save_dir)
        if best is not None:
            record.weights = str(best)
        if validate_after and best is not None:
            metrics.update(extract_val_metrics(model.val(data=str(data_yaml))))
        if efficiency and best is not None:
            try:
                from saryolo.evaluation.efficiency import profile_model

                metrics.update(profile_model(str(best), imgsz=cfg.imgsz))
            except Exception as exc:  # efficiency profiling is optional
                record.extra["efficiency_error"] = str(exc)
        metrics["train_minutes"] = round(train_minutes, 2)

        record = ledger.finish(record, metrics) if not skip_ledger else _offline_finish(record, metrics)
        return record
    except Exception as exc:
        if skip_ledger:
            record.status = "failed"
            record.notes = f"{record.notes} | ERROR: {exc}".strip(" |")
            return record
        return ledger.fail(record, str(exc))


def _offline_finish(record: ExperimentRecord, metrics: dict) -> ExperimentRecord:
    """Apply the terminal state without touching the ledger (smoke tests)."""
    record.status = "completed"
    record.metrics = metrics
    return record
