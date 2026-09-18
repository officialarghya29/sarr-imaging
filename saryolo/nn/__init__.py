"""SAR-YOLO neural network package.

Exposes the SAR-specific modules (Components 1-4), the symbolic architecture
builder, the registration hook into Ultralytics, the model class and the
SAR-aware loss (Component 7).
"""

from __future__ import annotations

from . import arch
from .arch import ModelSpec, build_yaml_dict, build_yaml_text
from .losses import SAR_LOSS_DEFAULTS, SARAwareDetectionLoss, build_criterion
from .model import SARYOLODetectionModel
from .modules import (
    AdaptiveMultiScaleFusion,
    CBAMAttention,
    ECAAttention,
    IdentityAttention,
    SARAdaptiveAttention,
    SARFeatureEnhancement,
    SEAttention,
    SpeckleAwareFeatureModule,
)
from .register import (
    allowlist_safe_load,
    ensure_registered,
    register_modules,
    registered_modules,
)

__all__ = [
    "arch",
    "ModelSpec",
    "build_yaml_dict",
    "build_yaml_text",
    "SARAwareDetectionLoss",
    "SAR_LOSS_DEFAULTS",
    "build_criterion",
    "SARYOLODetectionModel",
    "SARFeatureEnhancement",
    "SpeckleAwareFeatureModule",
    "SARAdaptiveAttention",
    "AdaptiveMultiScaleFusion",
    "IdentityAttention",
    "SEAttention",
    "ECAAttention",
    "CBAMAttention",
    "register_modules",
    "ensure_registered",
    "registered_modules",
    "allowlist_safe_load",
]
