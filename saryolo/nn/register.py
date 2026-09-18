"""Register SAR-YOLO modules with Ultralytics so they can be named in a model YAML.

How this works (verified against ultralytics 8.4.155)
-----------------------------------------------------
``ultralytics.nn.tasks.parse_model`` resolves a YAML row's ``module`` column with
``globals()[m]``, where ``globals()`` is the ``ultralytics.nn.tasks`` module
namespace. Publishing our classes as attributes of that module is therefore all
that is required to make ``- [-1, 1, SARAdaptiveAttention, ["ch", 16]]`` legal.

We deliberately do **not** monkeypatch ``parse_model``. Our modules satisfy the
parser's assumptions natively (single input, channel preserving — see
``saryolo/nn/modules/_common.py``), so:
  * ``c2 = ch[f]`` stays correct automatically;
  * the stock ultralytics parser, model summary, FLOPs counter and validator all
    work unmodified;
  * this survives ultralytics upgrades far better than a patched parser.

Checkpoint loading
------------------
Ultralytics checkpoints pickle the model object, so unpickling imports our
classes by their ``saryolo.nn.modules.*`` path. Importing ``saryolo`` (which runs
this registration) is therefore sufficient to load a trained SAR-YOLO checkpoint
with plain ``YOLO("best.pt")``. Restricted (``ULTRALYTICS_SAFE_LOAD``)
``weights_only=True`` loading additionally needs ``allowlist_safe_load()``.
"""

from __future__ import annotations

import os

from .modules import CUSTOM_MODULES

__all__ = ["register_modules", "registered_modules", "allowlist_safe_load", "ensure_registered"]

_REGISTERED: tuple[str, ...] = ()


def register_modules(verbose: bool = False) -> tuple[str, ...]:
    """Publish SAR-YOLO modules into ``ultralytics.nn.tasks`` (idempotent).

    Returns the sorted tuple of names that are now resolvable from a model YAML.
    """
    global _REGISTERED
    import ultralytics.nn.tasks as tasks

    for name, cls in CUSTOM_MODULES.items():
        setattr(tasks, name, cls)
    _REGISTERED = tuple(sorted(CUSTOM_MODULES))
    if verbose:
        print(f"Registered {len(_REGISTERED)} SAR-YOLO modules: {', '.join(_REGISTERED)}")
    return _REGISTERED


def registered_modules() -> tuple[str, ...]:
    """Names registered so far in this process (empty until ``register_modules`` runs)."""
    return _REGISTERED


def ensure_registered() -> tuple[str, ...]:
    """Register if not already done; safe to call from any entry point."""
    return _REGISTERED or register_modules()


def allowlist_safe_load() -> tuple[str, ...]:
    """Allow SAR-YOLO classes in ``torch.load(weights_only=True)`` checkpoints.

    Only needed when ``ULTRALYTICS_SAFE_LOAD`` is enabled. Registration is
    process-global and additive, matching the semantics ultralytics itself uses.
    """
    from torch.serialization import add_safe_globals

    classes = [cls for cls in CUSTOM_MODULES.values()] + [
        __import__("saryolo.nn.losses", fromlist=["SARAwareDetectionLoss"]).SARAwareDetectionLoss,
    ]
    add_safe_globals(classes)
    return tuple(f"{cls.__module__}.{cls.__qualname__}" for cls in classes)


def safe_load_enabled() -> bool:
    """Whether ultralytics restricted checkpoint loading is switched on."""
    return os.environ.get("ULTRALYTICS_SAFE_LOAD", "").lower() in ("1", "true", "yes", "on")
