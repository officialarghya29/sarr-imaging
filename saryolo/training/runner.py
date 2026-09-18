"""Experiment runner: train one config and record it in the ledger.

The runner is the only place that writes an experiment row. It records a
``running`` row *before* training so a crash mid-run is still visible, and only
writes measured metrics afterwards.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from saryolo.tracking.ledger import ExperimentLedger, ExperimentRecord

from .config import ExperimentConfig, load_experiment
from .trainer import load_model

__all__ = ["run_experiment", "read_results_csv", "extract_val_metrics", "find_best_weights"]


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

    overrides = dict(cfg.train)
    overrides.update(extra_overrides or {})
    if device:
        overrides["device"] = device
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

        data_yaml = resolve_data_yaml(cfg.dataset_path)
        model = load_model(str(cfg.model_path))
        model.train(
            data=str(data_yaml),
            project=str(project),
            name=cfg.experiment_id,
            exist_ok=True,
            **overrides,
        )
        train_minutes = (time.time() - started) / 60.0
        save_dir = Path(getattr(model.trainer, "save_dir", project) or project)
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
