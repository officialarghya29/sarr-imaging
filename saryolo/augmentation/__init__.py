"""Augmentation package: standard Ultralytics transforms plus SAR-specific ones (SEC. 5).

The SAR-specific side is offline and lives in :mod:`saryolo.augmentation.sar`. It reuses the
corruption model from :mod:`saryolo.evaluation.robustness` on purpose, so that what a model is
trained under and what it is evaluated under are the same physical model.
"""

from __future__ import annotations

from .sar import SAR_AUGMENTATIONS, AugmentationPlan, build_augmented_train_split, default_plan

__all__ = [
    "SAR_AUGMENTATIONS",
    "AugmentationPlan",
    "build_augmented_train_split",
    "default_plan",
]
