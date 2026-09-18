"""Visualization package: detections, attention/feature maps and failure analysis."""

from __future__ import annotations

from .attention_maps import find_layers, gradcam, overlay_heatmap, save_attention_panel
from .detections import annotate_gt, comparison_panel, draw_detections, select_examples
from .error_analysis import FailureCase, FailureSummary, analyse_failures, image_contrast_map, write_failure_report
from .feature_maps import compare_feature_maps, feature_maps, save_feature_grid

__all__ = [
    "gradcam",
    "save_attention_panel",
    "overlay_heatmap",
    "find_layers",
    "draw_detections",
    "annotate_gt",
    "comparison_panel",
    "select_examples",
    "feature_maps",
    "save_feature_grid",
    "compare_feature_maps",
    "analyse_failures",
    "image_contrast_map",
    "write_failure_report",
    "FailureCase",
    "FailureSummary",
]
