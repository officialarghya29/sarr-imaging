"""SAR-YOLO neural modules (Components 1-4 of the SAR-YOLO architecture).

All modules here are single-input, single-output and channel-preserving, which is
what allows them to be dropped into an Ultralytics YOLO YAML without patching
``parse_model``. See ``saryolo/nn/modules/_common.py`` for the full contract.
"""

from __future__ import annotations

from .attention import (
    ATTENTION_BUILDERS,
    CBAMAttention,
    ECAAttention,
    IdentityAttention,
    SARAdaptiveAttention,
    SEAttention,
    build_attention,
)
from .enhancement import SARFeatureEnhancement
from .fusion import AdaptiveMultiScaleFusion
from .speckle import SpeckleAwareFeatureModule

__all__ = [
    # Components
    "SARFeatureEnhancement",
    "SpeckleAwareFeatureModule",
    "SARAdaptiveAttention",
    "AdaptiveMultiScaleFusion",
    # Attention baselines / registry
    "IdentityAttention",
    "SEAttention",
    "ECAAttention",
    "CBAMAttention",
    "ATTENTION_BUILDERS",
    "build_attention",
    #: Module name -> class, injected into ``ultralytics.nn.tasks`` for YAML resolution.
    "CUSTOM_MODULES",
]

#: The names below are exactly what may appear in a model YAML's ``module`` column.
CUSTOM_MODULES = {
    "SARFeatureEnhancement": SARFeatureEnhancement,
    "SpeckleAwareFeatureModule": SpeckleAwareFeatureModule,
    "SARAdaptiveAttention": SARAdaptiveAttention,
    "AdaptiveMultiScaleFusion": AdaptiveMultiScaleFusion,
    "IdentityAttention": IdentityAttention,
    "SEAttention": SEAttention,
    "ECAAttention": ECAAttention,
    "CBAMAttention": CBAMAttention,
}
