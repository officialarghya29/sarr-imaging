"""Experiment configuration.

One YAML per experiment (``configs/exp/EXP-00x.yaml``) so that a run is fully
described by a committed file, and a table row can always be traced back to the
config, seed and commit that produced it.

Example
-------
.. code-block:: yaml

    experiment:
      id: EXP-001
      name: yolo11s baseline
      description: Stock YOLO11s, no SAR modules.
    model: ../models/baseline_s.yaml      # relative to this file
    dataset: ../datasets/ssdd.yaml        # relative to this file
    train:                                # passed straight to Ultralytics
      epochs: 100
      imgsz: 640
      batch: 16
      seed: 0
    notes: ""
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ExperimentConfig", "load_experiment", "list_experiments", "DEFAULT_TRAIN_ARGS"]

#: Conservative defaults; every experiment overrides these explicitly anyway.
DEFAULT_TRAIN_ARGS: dict[str, Any] = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "seed": 0,
    "deterministic": True,
    "workers": 8,
    "cache": False,
    "cos_lr": True,
    "patience": 30,
    "save_period": -1,
    "val": True,
    "plots": True,
}


@dataclass
class ExperimentConfig:
    """A parsed, path-resolved experiment definition."""

    experiment_id: str
    name: str
    model_path: Path
    dataset_path: Path
    train: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    notes: str = ""
    source_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def seed(self) -> int:
        return int(self.train.get("seed", 0))

    @property
    def imgsz(self) -> int:
        return int(self.train.get("imgsz", 640))

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "model_path": str(self.model_path),
            "dataset_path": str(self.dataset_path),
            "train": self.train,
            "description": self.description,
            "notes": self.notes,
        }


def _resolve(base: Path, value: str | Path) -> Path:
    """Resolve a possibly-relative path against the config file's directory."""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config.

    Raises:
        FileNotFoundError: if the model or dataset YAML does not exist. Failing
            here is intentional: a run that silently used a missing dataset would
            waste GPU hours.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    meta = raw.get("experiment") or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: 'experiment' must be a mapping")
    exp_id = str(meta.get("id") or path.stem)
    name = str(meta.get("name") or exp_id)

    if "model" not in raw:
        raise ValueError(f"{path}: missing required key 'model'")
    if "dataset" not in raw:
        raise ValueError(f"{path}: missing required key 'dataset'")

    model_path = _resolve(path.parent, raw["model"])
    dataset_path = _resolve(path.parent, raw["dataset"])
    for label, resolved in (("model", model_path), ("dataset", dataset_path)):
        if not resolved.exists():
            raise FileNotFoundError(f"{path}: {label} path does not exist: {resolved}")

    train = {**DEFAULT_TRAIN_ARGS, **(raw.get("train") or {})}
    # Reproducibility guard: the seed must be explicit, not inherited by accident.
    if "seed" not in (raw.get("train") or {}):
        raise ValueError(f"{path}: 'train.seed' must be set explicitly for a reproducible run")

    known = {"experiment", "model", "dataset", "train", "notes", "description"}
    return ExperimentConfig(
        experiment_id=exp_id,
        name=name,
        model_path=model_path,
        dataset_path=dataset_path,
        train=train,
        description=str(raw.get("description") or meta.get("description") or ""),
        notes=str(raw.get("notes") or ""),
        source_path=path,
        extra={k: v for k, v in raw.items() if k not in known},
    )


def list_experiments(exp_dir: str | Path = "configs/exp") -> list[Path]:
    """All experiment configs, sorted, ignoring the template and generated files."""
    exp_dir = Path(exp_dir)
    return sorted(p for p in exp_dir.glob("*.yaml") if not p.name.startswith(("_", ".")))
