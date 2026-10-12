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
``weights_only=True`` loading additionally needs ``allowlist_safe_load()``, which
allowlists not only this repository's classes but also the torch/ultralytics blocks
the graph is built from -- both are required, and the committed checkpoint fails to
load under restricted loading without the second set.
"""

from __future__ import annotations

import os

from .modules import CUSTOM_MODULES

__all__ = ["register_modules", "registered_modules", "allowlist_safe_load", "ensure_registered"]

_REGISTERED: tuple[str, ...] = ()

#: Cache for :func:`allowlist_safe_load`, which derives its class set by building graphs. The
#: derivation is the expensive part and its result cannot change within a process, so it runs at
#: most once -- which is what lets the checkpoint-load path call it unconditionally.
_ALLOWLISTED: tuple[str, ...] = ()


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


def _sarvo_classes() -> list[type]:
    """Every SAR-YOLO class that can appear in a checkpoint, not only the YAML-addressable ones.

    ``CUSTOM_MODULES`` is the set namable in a model YAML, which is *not* the same set that ends
    up pickled: the shared building blocks in ``saryolo.nn.modules._common``
    (``ConvBNAct``/``DWConvBNAct``/``ZeroGate``/...) are composed **inside** those modules, so
    they appear in every checkpoint while never appearing in a YAML row. Missing them was one of
    the reasons restricted loading failed on a checkpoint built entirely from this repository.
    """
    from torch import nn

    from .modules import _common

    classes: list[type] = list(CUSTOM_MODULES.values())
    classes += [
        obj for obj in vars(_common).values()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module
    ]
    classes.append(
        __import__("saryolo.nn.losses", fromlist=["SARAwareDetectionLoss"]).SARAwareDetectionLoss
    )
    return classes


def _framework_classes() -> list[type]:
    """The torch/ultralytics classes a SAR-YOLO checkpoint is built from, derived not listed.

    A checkpoint pickles the whole graph, so it contains the framework's blocks and containers
    (``DetectionModel``, ``C3k2``, ``SPPF``, ``Detect``, ``nn.Sequential``, ``nn.ModuleList``,
    ...) as well as this repository's classes. Deriving them rather than hand-writing them means
    an ultralytics upgrade that renames or adds a block is picked up automatically. Two sources
    are unioned because neither alone is complete: a real build contributes the torch containers
    (which no YAML names), and the YAML vocabulary of every registered variant contributes any
    block a particular scale or level set adds.
    """
    import ultralytics.nn.tasks as tasks
    from torch import nn
    from ultralytics.nn.tasks import DetectionModel

    from .arch import VARIANTS, build_yaml_dict

    classes: set[type] = {DetectionModel, nn.Module}
    # Build the smallest set of graphs whose union exercises every module slot, richest first, so
    # the torch containers and leaves *composed inside* the custom modules -- ``AvgPool2d`` in the
    # statistics pools, ``ModuleList`` in the band banks -- are collected too. Those classes
    # appear in no YAML, and a single baseline graph does not contain them, which is why one
    # build was not enough.
    covered: set[str] = set()
    for spec in sorted(VARIANTS.values(), key=lambda s: -len(s.module_names)):
        slugs = set(spec.module_names)
        if slugs <= covered:
            continue
        covered |= slugs
        built = DetectionModel(build_yaml_dict(spec), ch=3, nc=1, verbose=False)
        classes.update(type(module) for module in built.modules())
    if len(classes) == 2:  # nothing was built (an empty registry); fall back to a base graph
        built = DetectionModel(build_yaml_dict(VARIANTS["baseline_n"]), ch=3, nc=1, verbose=False)
        classes.update(type(module) for module in built.modules())
    # Every module name any registered variant can emit, resolved through the same namespace the
    # YAML parser resolves against.
    for spec in VARIANTS.values():
        config = build_yaml_dict(spec)
        for section in ("backbone", "head"):
            for row in config[section]:
                candidate = getattr(tasks, row[2], None)
                if isinstance(candidate, type):
                    classes.add(candidate)
    return sorted(classes, key=lambda cls: f"{cls.__module__}.{cls.__qualname__}")


def allowlist_safe_load() -> tuple[str, ...]:
    """Allow every class a SAR-YOLO checkpoint contains into ``torch.load(weights_only=True)``.

    Only needed when ``ULTRALYTICS_SAFE_LOAD`` is enabled. Registration is process-global and
    additive, matching the semantics ultralytics itself uses.

    The earlier version allowlisted only ``CUSTOM_MODULES`` plus the loss, which is necessary but
    not sufficient, and the shortfall was invisible because nothing exercised it: restricted
    loading of the committed checkpoint still failed, first on ``_common.DWConvBNAct`` (a class
    of this repository's own that was simply not in the list) and then on the framework's blocks
    and containers. The complete set is derived here and pinned by a test that loads a real
    checkpoint with ``weights_only=True``.
    """
    global _ALLOWLISTED
    if _ALLOWLISTED:
        return _ALLOWLISTED
    from torch.serialization import add_safe_globals

    seen: list[type] = []
    for cls in [*_sarvo_classes(), *_framework_classes()]:
        if cls not in seen:
            seen.append(cls)
    add_safe_globals(seen)
    _ALLOWLISTED = tuple(f"{cls.__module__}.{cls.__qualname__}" for cls in seen)
    return _ALLOWLISTED


def safe_load_enabled() -> bool:
    """Whether ultralytics restricted checkpoint loading is switched on."""
    return os.environ.get("ULTRALYTICS_SAFE_LOAD", "").lower() in ("1", "true", "yes", "on")
