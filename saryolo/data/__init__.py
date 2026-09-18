"""Dataset layer: registry, conversion, validation, statistics, splitting.

Nothing in this package mutates a raw dataset. Conversion output goes to
``datasets/processed``, splits to ``datasets/splits``, and reports to
``validation_reports`` / ``dataset_statistics``.
"""

from __future__ import annotations

from .convert import coco_to_yolo, dota_to_yolo_obb, prepare_dataset, voc_to_yolo
from .registry import DATASETS, RECOMMENDED_ORDER, DatasetSpec, get_dataset, list_datasets
from .splits import leakage_report, read_official_split, split_files, write_splits
from .statistics import COCO_SIZE_BINS, DatasetStatistics, plot_statistics, profile_dataset, save_statistics
from .synth import make_synthetic_dataset
from .validate import ValidationReport, validate_yolo_dataset, write_report
from .yolo import dataset_root, load_data_config, project_root, resolve_data_yaml, split_dirs, write_data_yaml

__all__ = [
    # registry
    "DATASETS",
    "RECOMMENDED_ORDER",
    "DatasetSpec",
    "get_dataset",
    "list_datasets",
    # conversion
    "prepare_dataset",
    "coco_to_yolo",
    "voc_to_yolo",
    "dota_to_yolo_obb",
    # yolo config
    "write_data_yaml",
    "dataset_root",
    "resolve_data_yaml",
    "load_data_config",
    "split_dirs",
    "project_root",
    # validation
    "validate_yolo_dataset",
    "write_report",
    "ValidationReport",
    # splits
    "split_files",
    "leakage_report",
    "read_official_split",
    "write_splits",
    # statistics
    "profile_dataset",
    "plot_statistics",
    "save_statistics",
    "DatasetStatistics",
    "COCO_SIZE_BINS",
    # synthetic (pipeline testing only)
    "make_synthetic_dataset",
]
