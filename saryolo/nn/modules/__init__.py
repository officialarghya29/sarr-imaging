"""SAR-YOLO neural modules (the architecture's research components).

All modules here are single-input, single-output and channel-preserving, which is
what allows them to be dropped into an Ultralytics YOLO YAML without patching
``parse_model``. See ``saryolo/nn/modules/_common.py`` for the full contract.

Two invariants are shared by every module and are enforced by tests:

1. **Exact identity at initialisation** — the residual gate starts at 0, so a
   freshly built SAR-YOLO is numerically identical to its YOLO baseline. Every
   reported gain is therefore attributable to what training learned, not to extra
   capacity that perturbs the function at step 0.
2. **Every parameter receives gradient at init** — identity comes from the gate
   *alone*. The residual branch is deliberately kept non-degenerate, because a
   zero-initialised branch on top of a zero-initialised gate has
   ``dL/dalpha = <dL/dout, branch> = 0`` and stays switched off forever while
   looking perfectly healthy to an identity test.
   See ``tests/test_arch.py::test_no_module_is_frozen_at_init``.
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
from .context import ContextAggregation
from .enhancement import SARFeatureEnhancement
from .frequency import SpatialFrequencyRepresentation
from .fusion import AdaptiveMultiScaleFusion
from .refinement import TargetAwareRefinement
from .speckle import SpeckleAwareFeatureModule
from .target_prior import TargetPriorModulation

__all__ = [
    # Components
    "SARFeatureEnhancement",
    "SpeckleAwareFeatureModule",
    "SARAdaptiveAttention",
    "AdaptiveMultiScaleFusion",
    "TargetPriorModulation",
    "SpatialFrequencyRepresentation",
    "ContextAggregation",
    "TargetAwareRefinement",
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
    "TargetPriorModulation": TargetPriorModulation,
    "SpatialFrequencyRepresentation": SpatialFrequencyRepresentation,
    "ContextAggregation": ContextAggregation,
    "TargetAwareRefinement": TargetAwareRefinement,
    "IdentityAttention": IdentityAttention,
    "SEAttention": SEAttention,
    "ECAAttention": ECAAttention,
    "CBAMAttention": CBAMAttention,
}
