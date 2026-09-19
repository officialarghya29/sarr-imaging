"""Training package: SAR-YOLO trainer, experiment runner and config handling."""

from __future__ import annotations

from .config import ExperimentConfig, list_experiments, load_experiment
from .hard_examples import (
    ImageDifficulty,
    rank_examples,
    score_examples,
    write_hard_data_config,
    write_oversampled_list,
)
from .runner import run_experiment
from .trainer import SARYOLO, SARYOLOTrainer, load_model, resolve_model_class

__all__ = [
    "SARYOLO",
    "SARYOLOTrainer",
    "load_model",
    "resolve_model_class",
    "ExperimentConfig",
    "load_experiment",
    "list_experiments",
    "run_experiment",
    # hard-example mining (SEC. 6): an offline sampling strategy, not a module
    "ImageDifficulty",
    "score_examples",
    "rank_examples",
    "write_oversampled_list",
    "write_hard_data_config",
]
