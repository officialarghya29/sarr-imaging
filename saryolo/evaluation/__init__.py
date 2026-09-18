"""Evaluation package: metrics, scale analysis, robustness, efficiency, generalization."""

from __future__ import annotations

from .cross_dataset import class_compatibility, cross_dataset_eval, domain_gap_table
from .efficiency import (
    count_parameters,
    measure_flops,
    measure_latency,
    model_size_mb,
    profile_model,
    profile_yaml,
    write_profile,
)
from .metrics import (
    AREA_RANGES,
    Detection,
    compute_ap,
    evaluate_detections,
    load_yolo_ground_truth,
    load_yolo_predictions,
    predict_to_labels,
    write_metrics,
)
from .robustness import CORRUPTIONS, Corruption, apply_corruption, build_corrupted_split, robustness_sweep

__all__ = [
    # metrics
    "evaluate_detections",
    "compute_ap",
    "load_yolo_ground_truth",
    "load_yolo_predictions",
    "predict_to_labels",
    "write_metrics",
    "Detection",
    "AREA_RANGES",
    # efficiency
    "profile_model",
    "profile_yaml",
    "count_parameters",
    "measure_flops",
    "measure_latency",
    "model_size_mb",
    "write_profile",
    # robustness
    "robustness_sweep",
    "apply_corruption",
    "build_corrupted_split",
    "CORRUPTIONS",
    "Corruption",
    # generalization
    "cross_dataset_eval",
    "class_compatibility",
    "domain_gap_table",
]
