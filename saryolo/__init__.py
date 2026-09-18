"""SAR-YOLO: speckle-aware feature representation for robust SAR object detection.

This package augments a YOLO baseline with evidence-driven, SAR-specific modules
and a reproducible research pipeline (dataset validation, ablation, robustness,
efficiency and cross-dataset studies).

Importing this package registers the SAR modules with Ultralytics
---------------------------------------------------------------
``import saryolo`` publishes the custom layers into ``ultralytics.nn.tasks`` so
they can be named in a model YAML. This happens automatically and idempotently
because it is also required for *unpickling* a trained checkpoint: loading
``best.pt`` re-imports ``saryolo.nn.modules.*``, which imports this module first.

Ultralytics itself is imported lazily via ``register_modules`` so that pure
data-pipeline use (for example ``saryolo data stats``) does not require torch.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "register_modules", "ensure_registered", "registered_modules"]


def register_modules(verbose: bool = False):
    """Register SAR-YOLO layers with ``ultralytics.nn.tasks`` (idempotent)."""
    from .nn.register import register_modules as _register

    return _register(verbose=verbose)


def ensure_registered():
    """Register SAR-YOLO layers if that has not happened yet in this process."""
    from .nn.register import ensure_registered as _ensure

    return _ensure()


def registered_modules():
    """Names currently registered with Ultralytics in this process."""
    from .nn.register import registered_modules as _registered

    return _registered()


def __getattr__(name: str):
    """Expose ``saryolo.nn`` lazily so heavy imports stay opt-in."""
    if name == "nn":
        from . import nn as _nn

        return _nn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _auto_register() -> None:
    """Register custom layers on import so YAML resolution and checkpoint unpickling just work."""
    try:
        register_modules()
    except ImportError as exc:
        if "ultralytics" not in str(exc):
            raise
        import warnings

        warnings.warn(
            f"SAR-YOLO custom layers were not registered because ultralytics is unavailable ({exc}). "
            "Model building and training require it; the data pipeline does not.",
            RuntimeWarning,
            stacklevel=2,
        )


_auto_register()
