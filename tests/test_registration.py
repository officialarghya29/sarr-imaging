"""Tests for ultralytics layer registration and restricted checkpoint loading.

Two mechanisms live here. The first, :func:`register_modules`, is what makes a YAML row like
``[-1, 1, ScatterSelectiveRefinement, [...]]`` legal: the parser resolves the module name with
``globals()[m]`` against ``ultralytics.nn.tasks``, so publishing the classes there is the whole
of it. The second, :func:`allowlist_safe_load`, is what is supposed to make
``torch.load(weights_only=True)`` -- the ``ULTRALYTICS_SAFE_LOAD`` path -- accept a checkpoint
that contains SAR-YOLO layers.

The second was broken and nothing caught it, because it was exported, documented, and never
called or tested: restricted loading of the committed checkpoint still failed, and its
allowlist omitted even some of this repository's own classes (the shared
``_common`` building blocks composed inside the YAML-addressable modules) as well as the
framework's blocks the graph is built from. The functional check below loads a real checkpoint
with ``weights_only=True``, which is exactly the contract the function advertises.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import saryolo  # noqa: F401  (registration runs on import)
from saryolo.nn import register
from saryolo.nn.modules import CUSTOM_MODULES

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CHECKPOINT = REPO_ROOT / "weights" / "sarvo_ssac001.pt"


# ------------------------------------------------------------------- layer registration
def test_registration_publishes_every_custom_module_into_the_parser_namespace():
    """Publishing the classes there is the *entire* mechanism the YAML parser relies on."""
    import ultralytics.nn.tasks as tasks

    names = register.register_modules()
    assert names == tuple(sorted(CUSTOM_MODULES)), "the returned list is not the registry"
    assert names, "nothing is registered, so no SAR-YOLO YAML could resolve"
    for name in names:
        assert getattr(tasks, name, None) is CUSTOM_MODULES[name], (
            f"{name} is not resolvable from ultralytics.nn.tasks, so a YAML naming it cannot build"
        )


def test_ensure_registered_is_idempotent_and_verbose_registration_reports_the_count(capsys):
    """Registration is process-global and additive; a second call must not duplicate work."""
    first = register.ensure_registered()
    assert first == register.registered_modules() == register.register_modules()
    register.register_modules(verbose=True)
    printed = capsys.readouterr().out
    assert f"Registered {len(first)} SAR-YOLO modules" in printed


# ------------------------------------------------------------------- safe-load enablement
@pytest.mark.parametrize("value", ["1", "true", "YES", "on", "On"])
def test_safe_load_enabled_recognises_the_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("ULTRALYTICS_SAFE_LOAD", value)
    assert register.safe_load_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "nope"])
def test_safe_load_enabled_is_off_for_everything_else(monkeypatch, value):
    monkeypatch.setenv("ULTRALYTICS_SAFE_LOAD", value)
    assert register.safe_load_enabled() is False


def test_safe_load_is_off_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("ULTRALYTICS_SAFE_LOAD", raising=False)
    assert register.safe_load_enabled() is False


# ------------------------------------------------------------------- the allowlist itself
def test_the_allowlist_covers_this_repository_classes_that_are_never_named_in_a_yaml():
    """The shared building blocks are composed inside the modules, so they are in every
    checkpoint while appearing in no YAML row -- omitting them was the first reason restricted
    loading failed."""
    names = register.allowlist_safe_load()
    assert names, "the allowlist is empty"
    qualified = set(names)
    for name in ("saryolo.nn.modules._common.ConvBNAct", "saryolo.nn.modules._common.DWConvBNAct",
                 "saryolo.nn.modules._common.ZeroGate"):
        assert name in qualified, f"{name} is in no checkpoint allowlist: restricted load fails"
    for cls in CUSTOM_MODULES.values():
        assert f"{cls.__module__}.{cls.__qualname__}" in qualified
    assert any("SARAwareDetectionLoss" in name for name in qualified)


def test_the_allowlist_covers_the_framework_blocks_and_containers_in_the_graph():
    """A checkpoint pickles the whole graph, so the framework's own types are required too."""
    names = set(register.allowlist_safe_load())
    for required in (
        "torch.nn.modules.container.Sequential",
        "torch.nn.modules.container.ModuleList",
        "torch.nn.modules.pooling.AvgPool2d",
        "torch.nn.modules.conv.Conv2d",
        "ultralytics.nn.tasks.DetectionModel",
    ):
        assert required in names, f"{required} is missing from the allowlist"


@pytest.mark.skipif(not COMMITTED_CHECKPOINT.exists(), reason="committed checkpoint not present")
def test_a_real_checkpoint_loads_under_weights_only_after_the_allowlist():
    """The contract the function advertises, checked on the checkpoint the repository ships.

    ``allowlist_safe_load`` promises that a SAR-YOLO checkpoint can be read with
    ``weights_only=True`` (the restricted ``ULTRALYTICS_SAFE_LOAD`` path). Before the fix this
    raised ``UnpicklingError`` -- the allowlist was necessary but far from sufficient -- so this
    is the regression that keeps it honest, and it exercises the primitive rather than the
    list of names.
    """
    register.allowlist_safe_load()
    payload = torch.load(COMMITTED_CHECKPOINT, weights_only=True)
    model = payload["model"] if isinstance(payload, dict) else payload
    assert type(model).__name__ in ("DetectionModel", "SARYOLODetectionModel")
    assert sum(p.numel() for p in model.parameters()) > 0


def test_the_allowlist_is_cached_so_it_is_not_re_derived_on_every_call():
    """The load path calls it unconditionally when safe loading is on, so repeats must be cheap."""
    first = register.allowlist_safe_load()
    assert register.allowlist_safe_load() is first, "the allowlist was re-derived"


@pytest.mark.skipif(not COMMITTED_CHECKPOINT.exists(), reason="committed checkpoint not present")
def test_load_model_prepares_restricted_loading_only_when_it_is_switched_on(monkeypatch):
    """The documented opt-in must be reachable from the load path, and cost nothing when off.

    ``ULTRALYTICS_SAFE_LOAD`` is only useful if something allowlists the graph before the load;
    nothing did, so the variable made the shipped checkpoint unloadable. This pins both halves:
    the path prepares the unpickler when the variable is set, and does not touch it otherwise.
    """
    from saryolo.training.trainer import load_model

    calls: list[int] = []
    real = register.allowlist_safe_load
    monkeypatch.setattr(register, "allowlist_safe_load", lambda: calls.append(1) or real())

    monkeypatch.setenv("ULTRALYTICS_SAFE_LOAD", "1")
    load_model(str(COMMITTED_CHECKPOINT))
    assert calls, "load_model did not prepare restricted loading under ULTRALYTICS_SAFE_LOAD"

    calls.clear()
    monkeypatch.delenv("ULTRALYTICS_SAFE_LOAD", raising=False)
    load_model(str(COMMITTED_CHECKPOINT))
    assert not calls, "an ordinary run paid for an allowlist it does not need"
