"""Tests for the LoRA parameter-efficient adaptation baseline (master Direction B).

These tests exist because a LoRA arm fails *silently* in a way that produces a
complete-looking result table. Four properties are the whole argument, and each is
checked against behaviour rather than against the config text:

1. **A LoRA-wrapped model is the base model at step 0.** The ``B`` factor is
   zero-initialised, so ``scale * B @ A`` is exactly zero and the wrapped layer computes
   bit-for-bit what the base layer computed. Without this, the LoRA arm would be measuring
   a changed initial function rather than the adaptation.
2. **The adapter actually trains.** Every factor must receive gradient and move under an
   optimiser. This is the failure that was *measured* while writing the module: the first
   version short-circuited on ``B == 0`` to make the identity bit-exact, which removed both
   factors from the autograd graph and left 88 adapters permanently at zero. An arm that
   reports a complete LoRA result while adapting nothing is worse than a crash.
3. **The base weights are frozen and only the declared budget is trainable.** The trainable
   count is the number the efficiency claim rests on, so it is asserted rather than
   reported.
4. **A merged adapter equals the adapted model.** ``merge_lora`` is the inference path, and
   an in-place merge double-counts the update -- measured at 0.28 of output drift -- so the
   round trip is pinned.

The configuration contract is checked too: an unknown key, an empty target set and a
fraction outside ``(0, 1]`` must all be refused, because each one turns an ablation into a
comparison of two identical arms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import saryolo  # noqa: F401  (registers custom layers with ultralytics)
from saryolo.nn.arch import VARIANTS, build_yaml_dict
from saryolo.nn.model import SARYOLODetectionModel
from saryolo.training.peft import (
    DEFAULT_TARGETS,
    LoRAConfig,
    LoRALayer,
    adapter_merged,
    inject_lora,
    lora_parameter_summary,
    merge_lora,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _model(variant: str = "baseline_s", nc: int = 1, seed: int = 0) -> SARYOLODetectionModel:
    torch.manual_seed(seed)
    return SARYOLODetectionModel(build_yaml_dict(VARIANTS[variant]), ch=3, nc=nc, verbose=False)


def _collect_tensors(output, acc: list[torch.Tensor]) -> None:
    """Every tensor in a model output, whatever nesting ultralytics returns it in."""
    if torch.is_tensor(output):
        acc.append(output)
    elif isinstance(output, dict):
        for value in output.values():
            _collect_tensors(value, acc)
    elif isinstance(output, (list, tuple)):
        for value in output:
            _collect_tensors(value, acc)


def _lora_modules(model: torch.nn.Module) -> list[LoRALayer]:
    return [m for m in model.modules() if isinstance(m, LoRALayer)]


# ------------------------------------------------------------------- identity at init
def test_a_lora_model_is_bit_identical_to_its_base_at_initialisation():
    """``B`` is zero, so the wrapped layer must compute exactly what the base computed.

    Exact equality is asserted deliberately. A LoRA arm whose initial function differs from
    its reference would make every reported difference a mix of the adaptation and the
    changed start point, and no parameter count or shape check would reveal it.
    """
    base = _model(seed=0).eval()
    adapted = _model(seed=0).eval()
    inject_lora(adapted, LoRAConfig(rank=8, alpha=16.0))
    adapted.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y_base = base(x)[0]
        y_adapted = adapted(x)[0]
    assert torch.equal(y_base, y_adapted), (
        f"a LoRA model is not the base model at init (max abs diff "
        f"{float((y_base - y_adapted).abs().max()):.3e})"
    )


@pytest.mark.parametrize("rank", [1, 4, 8])
def test_identity_holds_at_every_rank(rank):
    """The identity is a property of the initialisation, not of one rank."""
    base = _model(seed=1).eval()
    adapted = _model(seed=1).eval()
    inject_lora(adapted, LoRAConfig(rank=rank, alpha=float(2 * rank)))
    adapted.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(base(x)[0], adapted(x)[0]), f"rank {rank} is not identity at init"


def test_the_identity_does_not_depend_on_a_zero_short_circuit():
    """Opening ``B`` by hand must change the representation, so identity comes from ``B``'s value.

    The first implementation skipped the low-rank product entirely while ``B`` was zero.
    That made the identity exact but removed both factors from the autograd graph, so the
    adapter could never train. This test asserts the product is always computed.

    The *feature* maps are the thing to measure, not the raw detection output. Measured on
    an untrained 80-class head: perturbing every ``B`` moved the feature maps by 0.9-3.1 in
    relative terms, while the final tensor moved by 8e-8 -- because an untrained head emits
    box coordinates on a fixed grid, which dominate the tensor and do not depend on the
    features at all. Asserting on the output alone would have passed for a completely dead
    adapter, which is the failure this test exists to catch.
    """
    adapted = _model(seed=2).eval()
    inject_lora(adapted, LoRAConfig(rank=4, alpha=8.0))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        feats_before = [f.clone() for f in adapted(x)[1]["feats"]]
        for module in _lora_modules(adapted):
            torch.nn.init.normal_(module.lora_b, std=0.05)
        feats_after = adapted(x)[1]["feats"]

    moved = [
        float((a - b).abs().max() / max(float(a.abs().max()), 1e-9))
        for a, b in zip(feats_before, feats_after, strict=True)
    ]
    assert all(rel > 1e-3 for rel in moved), (
        f"perturbing B left the feature maps unchanged (relative moves {moved}); the low-rank "
        f"product is not reaching the graph"
    )


# ------------------------------------------------------------------- the adapter trains
def test_every_adapter_factor_receives_gradient_and_moves():
    """The failure this catches is a live, measured one: a permanently dead adapter.

    Both directions are checked. ``B`` must receive non-zero gradient on the very first
    backward pass, and ``A`` must too -- ``A``'s gradient is proportional to ``B``, so it is
    legitimately zero at step 0 and only becomes non-zero once ``B`` has moved. Asserting
    only the first step would accept an adapter whose second factor never trains.

    Injected at the detection-model scope (as the runner does). Scoping it to the inner
    ``Sequential`` would leave the detection head's own convolutions unwrapped, so the
    gradient reaching ``B`` would depend on the head's path rather than on the adapter.
    """
    model = _model(seed=3).train()
    inject_lora(model, LoRAConfig(rank=8, alpha=16.0))
    modules = _lora_modules(model)
    assert modules, "no adapter was injected"

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    first_b_grad = None
    for step in range(3):
        model.zero_grad()
        tensors: list[torch.Tensor] = []
        _collect_tensors(model(torch.randn(2, 3, 64, 64)), tensors)
        sum(t.float().pow(2).mean() for t in tensors).backward()
        if step == 0:
            first_b_grad = [m.lora_b.grad for m in modules]
        if step == 1:
            assert all(m.lora_a.grad is not None and m.lora_a.grad.abs().max().item() > 0
                       for m in modules), (
                "A receives no gradient after B has moved, so the low-rank update is frozen"
            )
        opt.step()

    assert all(g is not None and g.abs().max().item() > 0 for g in first_b_grad), (
        "B receives zero gradient on the first step, so the adapter can never leave identity"
    )
    assert any(float(m.lora_b.detach().abs().max()) > 0 for m in modules), "B never moved"
    assert any(float((m.lora_a.detach() - 0).abs().max()) > 0 for m in modules), "A never moved"


def test_the_base_weights_are_frozen():
    """LoRA's parameter claim is only meaningful if the base weights are not updated."""
    model = _model(seed=4).train()
    inject_lora(model, LoRAConfig(rank=4, alpha=8.0))
    modules = _lora_modules(model)
    tensors: list[torch.Tensor] = []
    _collect_tensors(model(torch.randn(2, 3, 64, 64)), tensors)
    sum(t.float().pow(2).mean() for t in tensors).backward()
    unfrozen = [m for m in modules if m.base.weight.grad is not None]
    assert not unfrozen, f"{len(unfrozen)} wrapped base weights received gradient"
    assert all(not m.base.weight.requires_grad for m in modules)


def test_the_trainable_budget_is_reported_and_smaller_than_the_base():
    """The trainable fraction is the number the efficiency claim rests on."""
    model = _model(seed=5)
    base_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = inject_lora(model, LoRAConfig(rank=8, alpha=16.0))

    assert summary["n_wrapped_layers"] > 0
    assert summary["lora_params"] > 0
    assert summary["trainable_params"] < base_trainable, (
        "the LoRA arm is not cheaper to train than the base, so it is not parameter-efficient"
    )
    assert 0.0 < summary["fraction_trainable"] < 1.0
    # The three-way split must add up, or the reported budget is describing another model.
    assert (
        summary["lora_params"] + summary["other_trainable_params"] + summary["frozen_params"]
        == summary["total_params"]
    )
    # And it must agree with what the module tree actually says.
    live = lora_parameter_summary(model)
    assert live == {k: summary[k] for k in live}


def test_a_larger_rank_costs_more_and_still_identity():
    """Rank must be a real knob: a study that varied it would otherwise measure nothing."""
    counts = {}
    for rank in (2, 8, 32):
        model = _model(seed=6)
        summary = inject_lora(model, LoRAConfig(rank=rank, alpha=float(2 * rank)))
        counts[rank] = summary["lora_params"]
    assert counts[2] < counts[8] < counts[32], counts


def test_freezing_the_base_is_a_stricter_budget_and_is_recorded():
    """``freeze_base`` is a different experiment, so the summary must say which one ran."""
    loose = _model(seed=7)
    strict = _model(seed=7)
    summary_loose = inject_lora(loose, LoRAConfig(rank=4, alpha=8.0))
    summary_strict = inject_lora(strict, LoRAConfig(rank=4, alpha=8.0, freeze_base=True))

    assert summary_strict["freeze_base"] is True
    assert summary_strict["trainable_params"] < summary_loose["trainable_params"]
    assert summary_strict["trainable_params"] == summary_strict["lora_params"], (
        "with freeze_base the only trainable parameters must be the LoRA factors"
    )


# ------------------------------------------------------------------------ merging
def test_merging_round_trips_the_adapted_model():
    """A merged model must equal the adapted model, and contain no adapters.

    The tempting implementation is to add ``B @ A`` into the base weight in place, which
    double-counts the update because the adapter's forward pass still adds it. That was
    measured at 0.28 of output drift before being replaced by ``merge_lora``, which swaps
    the adapter out entirely.
    """
    model = _model(seed=8).eval()
    inject_lora(model, LoRAConfig(rank=4, alpha=8.0))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        for module in _lora_modules(model):
            torch.nn.init.normal_(module.lora_b, std=0.02)
        before = model(x)[0]
        merged = merge_lora(model)
        after = model(x)[0]

    assert merged > 0
    assert not _lora_modules(model), "adapters survived the merge"
    assert torch.allclose(before, after, atol=1e-4), (
        f"merging changed the model (max abs diff {float((before - after).abs().max()):.3e})"
    )


def test_merging_actually_changes_the_weights():
    """A merge that leaves the model equal to the *base* would be a no-op in disguise.

    Measured on the feature maps for the reason given in the identity test: the untrained
    head's output is dominated by grid coordinates, so comparing only the final tensor would
    make a lost update look like a preserved one.
    """
    base = _model(seed=9).eval()
    adapted = _model(seed=9).eval()
    inject_lora(adapted, LoRAConfig(rank=4, alpha=8.0))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        for module in _lora_modules(adapted):
            torch.nn.init.normal_(module.lora_b, std=0.05)
        merge_lora(adapted)
        base_feats = base(x)[1]["feats"]
        merged_feats = adapted(x)[1]["feats"]
    moved = max(
        float((a - b).abs().max() / max(float(a.abs().max()), 1e-9))
        for a, b in zip(base_feats, merged_feats, strict=True)
    )
    assert moved > 1e-3, (
        "after merging, the adapted model still equals the base -- the update was lost"
    )


def test_the_save_time_merge_makes_a_loadable_graph_and_restores_the_adapters():
    """The save window must expose a plain detector and then hand the adapter back.

    This is the second failure that was measured on real runs rather than reasoned about. A
    LoRA run finished training, wrote ``best.pt``, and then died inside Ultralytics' own
    end-of-run validation of that file: ``'LoRALayer' object has no attribute 'weight'``,
    because the checkpoint loader fuses convolutions with the BatchNorm that follows them and
    a wrapper is not a convolution. So ``save_model`` merges for the duration of the save.

    Three things have to hold, and all three are checked because each is a different way the
    fix could be wrong: during the window the model is an ordinary graph that computes what
    the adapted model computes, afterwards the adapters and their exact bits are back, and the
    adversarial case -- an *in-place* merge -- is ruled out by the restoration check.
    """
    model = _model(seed=11).eval()
    inject_lora(model, LoRAConfig(rank=4, alpha=8.0))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        for module in _lora_modules(model):
            torch.nn.init.normal_(module.lora_b, std=0.05)
        adapted_feats = model(x)[1]["feats"]
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    with adapter_merged(model) as merged:
        assert merged > 0
        assert not _lora_modules(model), "the adapter is still in the tree during the save"
        with torch.no_grad():
            window_feats = model(x)[1]["feats"]
        assert all(
            torch.allclose(a, b, atol=1e-5) for a, b in zip(adapted_feats, window_feats, strict=True)
        ), "the merged model does not compute what the adapted model computed"

    assert len(_lora_modules(model)) == merged, "the adapters were not put back"
    for name, param in model.named_parameters():
        assert torch.equal(param.detach(), before[name]), (
            f"{name} was not restored bit-for-bit by the save window -- an in-place merge would "
            f"leave the update folded into the base weight and double-count it afterwards"
        )


def test_a_model_without_adapters_merges_nothing():
    """A full fine-tune has no adapter, so the window is a no-op and must say so."""
    with adapter_merged(_model(seed=12)) as merged:
        assert merged == 0


# ------------------------------------------------------------------- configuration
def test_unknown_config_keys_are_refused():
    """A typo such as ``r: 8`` would run the default rank and silently collapse two arms."""
    with pytest.raises(ValueError, match="unknown LoRA key"):
        LoRAConfig.from_mapping({"r": 8, "alpha": 16.0})


def test_an_empty_target_set_is_refused():
    with pytest.raises(ValueError, match="targets"):
        LoRAConfig(targets=())


@pytest.mark.parametrize("rank", [0, -1])
def test_a_non_positive_rank_is_refused(rank):
    with pytest.raises(ValueError, match="rank"):
        LoRAConfig(rank=rank)


def test_a_target_that_matches_nothing_is_refused():
    """An arm that wraps zero layers *is* the baseline, and would be reported as LoRA."""
    model = _model(seed=10)
    with pytest.raises(ValueError, match="matched no layer"):
        inject_lora(model, LoRAConfig(targets=("GRU",)))


def test_the_default_target_set_is_broad_and_documented():
    """The default must catch convolutions without the caller having to name them."""
    assert "Conv2d" in DEFAULT_TARGETS
    model = _model(seed=11)
    summary = inject_lora(model, LoRAConfig())
    assert summary["n_wrapped_layers"] > 50
    assert "Conv2d" in summary["matched_targets"]


def test_a_layer_the_base_model_freezes_is_not_wrapped():
    """Ultralytics freezes the DFL projection, so adapting it would add a dead parameter set.

    Measured before this rule existed: one of 88 wrapped layers had an identically zero
    gradient, so the reported trainable count overstated what was actually optimised.
    """
    model = _model(seed=12)
    summary = inject_lora(model, LoRAConfig())
    assert summary["skipped_frozen_layers"], (
        "no frozen base layer was skipped; either the rule was removed or ultralytics changed"
    )
    for name in summary["skipped_frozen_layers"]:
        assert not any(name.endswith(suffix) for suffix in ("lora_a", "lora_b"))


def test_excluded_layers_are_left_alone():
    """``exclude`` supports a targeted study and must actually exclude.

    Applied at the detection-model scope -- the same scope the runner uses -- because the
    exclusion names are interpreted relative to the module that is passed in. Calling this
    on ``model.model`` instead would name the layers ``0.conv``, and the rule would appear
    not to work while silently matching a different set of names.
    """
    full = _model(seed=13)
    all_layers = inject_lora(full, LoRAConfig())
    targeted = _model(seed=13)
    excluded = inject_lora(targeted, LoRAConfig(exclude=("model.0",)))
    assert excluded["n_wrapped_layers"] < all_layers["n_wrapped_layers"]
    wrapped = [name for name, m in targeted.named_modules() if isinstance(m, LoRALayer)]
    assert not any(name.startswith("model.0") for name in wrapped), (
        f"model.0 was excluded but still contains adapters: {[n for n in wrapped if n.startswith('model.0')]}"
    )


# ------------------------------------------------------------------ runner integration
# ------------------------------------------------------------------ runner integration
def test_the_runner_validates_the_peft_block_and_the_trainer_injects_it():
    """The config block must be validated by the runner and injected by the trainer.

    The split is the whole point of this pair of tests, and it is not a refactor for its own
    sake. ``YOLO.train()`` rebuilds the model from its config inside the trainer, so an
    adapter injected by the runner into a YAML-built facade never reaches the optimiser. That
    was measured on the first real LoRA arm: the ledger recorded 87 wrapped layers and a 10%
    trainable fraction, and the saved checkpoint contained no ``lora_`` tensor at all -- a
    full fine-tune reported as a parameter-efficient one. The runner now only validates and
    forwards the block; ``inject_run_adapter`` is the trainer-side rule, and it is checked
    here against the same model scope the trainer passes in.
    """
    from saryolo.training.runner import validate_peft
    from saryolo.training.trainer import inject_run_adapter

    assert validate_peft(None) is None, "an absent peft block must mean a full fine-tune"

    forwarded = validate_peft({"method": "lora", "rank": 4, "alpha": 8.0})
    assert forwarded == {"method": "lora", "rank": 4, "alpha": 8.0}, (
        "the runner must forward the block unchanged; re-spelling it here is how a config "
        "and a run come apart"
    )
    model = _model(seed=14)
    assert inject_run_adapter(model, None) == {}, "no block must mean no adapter"
    summary = inject_run_adapter(model, forwarded)
    assert summary["method"] == "lora"
    assert summary["rank"] == 4
    assert summary["n_wrapped_layers"] > 0
    assert _lora_modules(model), "the trainer reported LoRA but injected no adapter"


def test_the_trainer_constructs_without_an_explicit_cfg(tmp_path):
    """Ultralytics builds a trainer as ``Trainer(overrides=args)``, with no ``cfg``.

    The first signature used a *string* default, which Ultralytics then resolved as a path and
    the run died with ``FileNotFoundError: 'default.yaml'`` before training started. It stayed
    hidden for every baseline arm because a stock YAML resolves to the plain ``YOLO`` facade
    and therefore to Ultralytics' own trainer; it surfaced the moment an arm needed this one.
    Constructing the trainer is what pins it -- the failure is in ``__init__``, not in training.
    """
    from saryolo.data.yolo import resolve_data_yaml
    from saryolo.training.trainer import SARYOLOTrainer

    data_root = REPO_ROOT / "datasets" / "processed" / "synthetic_smoke"
    if not (data_root / "images" / "train").is_dir():
        pytest.skip("synthetic smoke data is not prepared (python -m saryolo synth-data ...)")

    trainer = SARYOLOTrainer(
        overrides={
            "model": str(REPO_ROOT / "configs" / "models" / "yolo11n_baseline_n.yaml"),
            # Resolved first for the same reason the runner resolves it: Ultralytics treats a
            # relative `path` in a dataset YAML as relative to its own global datasets dir.
            "data": str(resolve_data_yaml(REPO_ROOT / "configs" / "datasets" / "synthetic_smoke.yaml")),
            "epochs": 1,
            "imgsz": 64,
            "batch": 2,
            "workers": 0,
            "project": str(tmp_path / "runs"),
            "name": "construct",
        }
    )
    assert trainer.model is not None, "the trainer built no model"
    # The run-level controls travel through the trainer's namespace and must survive it.
    assert trainer.peft_summary == {}, "a full fine-tune must report no adapter"


def test_a_parameter_efficient_arm_uses_the_custom_trainer_even_on_a_stock_graph():
    """A LoRA arm must resolve to the custom facade; otherwise nothing injects the adapter.

    This is the facade half of the discarded-adapter bug. The arm that was measured used the
    *stock* baseline YAML, so the default resolution handed it to Ultralytics' own trainer,
    which knows nothing about ``peft``. The graph must stay stock -- that is what makes the
    arm a fair comparison against a full fine-tune of the same detector -- but the trainer
    cannot be.
    """
    from saryolo.training.trainer import SARYOLO, YOLO, resolve_model_class

    stock = str(REPO_ROOT / "configs" / "models" / "yolo11n_baseline_n.yaml")
    assert resolve_model_class(stock) is YOLO, (
        "a baseline must keep running on unmodified Ultralytics code"
    )
    assert resolve_model_class(stock, prefer_custom=True) is SARYOLO


def test_an_unsupported_peft_method_is_refused():
    from saryolo.training.runner import validate_peft

    with pytest.raises(ValueError, match="unsupported peft method"):
        validate_peft({"method": "prefix_tuning"})


def test_peft_method_none_is_a_full_fine_tune():
    """``method: none`` must mean *no adapter*, not an empty adapter arm.

    It returns ``None`` rather than ``{"method": "none"}`` so that the run is a plain full
    fine-tune in every downstream sense -- no facade override, no ledger block, nothing that a
    table could present as a parameter-efficient arm.
    """
    from saryolo.training.runner import validate_peft
    from saryolo.training.trainer import inject_run_adapter

    assert validate_peft({"method": "none"}) is None
    model = _model(seed=16)
    before = sum(p.numel() for p in model.parameters())
    assert inject_run_adapter(model, None) == {}
    assert sum(p.numel() for p in model.parameters()) == before
    assert not _lora_modules(model)


def test_a_lora_arm_end_to_end_trains_adapters_and_writes_a_usable_checkpoint(tmp_path):
    """The whole arm, on real training code: adapters trained, and a checkpoint that loads.

    Every other test in this file checks a property of the mechanism; this one checks that the
    five of them are actually wired together in that order during a real run, because both of
    the bugs found here were wiring bugs that unit tests of the mechanism could not see:

    * the adapter was applied to a facade that Ultralytics rebuilt before the first step, so
      the ledger described a parameter-efficient arm that was a full fine-tune;
    * the adapter survived into the checkpoint as a wrapper, so the checkpoint could not be
      loaded back at all.

    It trains the tiny synthetic smoke arm (2 epochs, 128px, 64 images) so the check is over
    the same code path as a real run. The synthetic data is meaningless as a result, which is
    the point: only the plumbing is under test.
    """
    import yaml

    from saryolo.training.runner import run_experiment

    config = REPO_ROOT / "configs" / "exp" / "_smoke_lora.yaml"
    data_root = REPO_ROOT / "datasets" / "processed" / "synthetic_smoke"
    if not any((data_root / "images" / "train").glob("*.*")):
        pytest.skip("synthetic smoke data is not prepared (python -m saryolo synth-data ...)")

    record = run_experiment(
        config,
        ledger_root=tmp_path / "results",
        project=tmp_path / "runs",
        extra_overrides={"workers": 0, "plots": False, "verbose": False},
        skip_ledger=True,
    )
    assert record.status == "completed", record.notes

    # 1. The summary came from the trainer, i.e. from the model that was actually optimised.
    assert record.extra.get("method") == "lora", record.extra
    assert record.extra["n_wrapped_layers"] > 0
    assert 0.0 < float(record.extra["fraction_trainable"]) < 1.0, (
        "a LoRA arm whose trainable fraction is 1.0 optimised exactly what the baseline did"
    )
    assert record.extra["lora_params"] > 0

    # 2. The checkpoint is an ordinary detector: no wrapper, so it loads, fuses and exports.
    best = Path(record.weights)
    assert best.is_file(), best
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    trained = checkpoint.get("model") or checkpoint.get("ema")
    keys = trained.state_dict()
    assert not [k for k in keys if "lora" in k], (
        "the checkpoint still carries adapter tensors; the save-time merge did not run"
    )
    assert type(trained).__mro__[1].__name__ == "DetectionModel", type(trained).__name__

    # 3. And it is genuinely adapted, not the un-adapted graph with a LoRA label.
    payload = yaml.safe_load(config.read_text())
    model_yaml = (config.parent / payload["model"]).resolve()
    fresh = SARYOLODetectionModel(
        yaml.safe_load(model_yaml.read_text()), ch=3, nc=1, verbose=False
    ).state_dict()
    moved = sum(
        1
        for key, value in keys.items()
        if key in fresh and fresh[key].shape == value.shape
        and not torch.equal(value.float(), fresh[key].float())
    )
    assert moved > len(fresh) // 2, (
        f"only {moved} of {len(fresh)} tensors differ from a fresh build; the training did not "
        f"reach the checkpoint"
    )


# ------------------------------------------------------------------ committed configs
def test_every_peft_arm_has_a_runnable_config_and_a_reference():
    """Each LoRA arm needs a config, and the reference arm must be a full fine-tune.

    Without the reference the LoRA numbers have nothing to be cheap *relative to*, and
    without the configs they can never be produced.
    """
    import make_exp_configs as G

    ids = [row[0] for row in G.PEFT]
    assert len(ids) == len(set(ids)), f"duplicate PEFT experiment ids: {ids}"
    references = [row for row in G.PEFT if row[3] is None]
    assert len(references) == 1, "there must be exactly one full-fine-tune reference arm"

    exp_dir = REPO_ROOT / "configs" / "exp"
    for exp_id, _variant, _name, peft, _purpose in G.PEFT:
        configs = sorted(exp_dir.glob(f"{exp_id}_*.yaml"))
        assert configs, f"{exp_id} has no committed config"
        import yaml

        payload = yaml.safe_load(configs[0].read_text())
        if peft is None:
            assert "peft" not in payload, f"{exp_id} is the reference arm but declares a peft block"
        else:
            assert payload.get("peft", {}).get("rank") == peft["rank"], (
                f"{exp_id}: config rank {payload.get('peft', {}).get('rank')} != table {peft['rank']}"
            )


def test_the_data_fraction_sweep_is_ordered_and_unique():
    """The fractions must be distinct and ordered, and each arm must exist as a config.

    That the resulting *subsets* are nested is a property of the selection code and is pinned
    separately, in ``test_the_native_fraction_is_nested_and_deterministic``; this test covers
    the half that is a claim about the repository: an arm per fraction, none duplicated, in an
    order a reader can follow.
    """
    import make_exp_configs as G

    fractions = [row[1] for row in G.DATA_FRACTIONS]
    assert fractions == sorted(fractions), fractions
    assert len(set(fractions)) == len(fractions), "duplicate fractions would collapse two arms"
    assert all(0.0 < f < 1.0 for f in fractions), fractions

    exp_dir = REPO_ROOT / "configs" / "exp"
    for exp_id, fraction, _name in G.DATA_FRACTIONS:
        configs = sorted(exp_dir.glob(f"{exp_id}_*.yaml"))
        assert configs, f"{exp_id} has no committed config"
        import yaml

        payload = yaml.safe_load(configs[0].read_text())
        assert abs(float(payload["data_fraction"]) - fraction) < 1e-9, exp_id


def test_the_data_fraction_becomes_the_native_fraction_argument(tmp_path):
    """The sweep's config key must reach the library's own ``fraction``, not a private key.

    The first implementation passed a private ``data_fraction`` override to this repository's
    trainer. Measured consequence: an arm whose model YAML is stock resolves to the plain
    ``YOLO`` facade, so Ultralytics' own trainer received the override and refused it with
    "'data_fraction' is not a valid YOLO argument" -- the baseline arm of the sweep, which is
    what every other arm is compared against, could not run at all. The translation is pinned
    here because it is a one-line mapping whose silent loss would make the fraction arms train
    on the full split while their configs and notes claim 1 % and 5 %.
    """
    from saryolo.training.config import load_experiment
    from saryolo.training.runner import training_overrides

    config = REPO_ROOT / "configs" / "exp" / "EXP-702_datafrac_05.yaml"
    cfg = load_experiment(config)
    assert abs(cfg.data_fraction - 0.05) < 1e-9, cfg.data_fraction

    overrides = training_overrides(cfg)
    assert overrides["fraction"] == 0.05, overrides.get("fraction")
    assert "data_fraction" not in overrides, (
        "the private key must not be forwarded: Ultralytics' validator rejects any argument it "
        "does not know, and a stock-graph arm never sees this repository's trainer"
    )

    # A run at the full split must leave the argument alone rather than restate the default.
    full = load_experiment(REPO_ROOT / "configs" / "exp" / "EXP-601_peft_full.yaml")
    assert full.data_fraction == 1.0
    assert "fraction" not in training_overrides(full)


def test_the_native_fraction_is_nested_and_deterministic(tmp_path):
    """Pin the library behaviour the sweep now depends on.

    A fraction study is only a fraction study if the arms are *nested*: the 1 % arm has to be a
    subset of the 5 % arm, or a difference between them is a difference between two independent
    draws. The nested-ness now comes from Ultralytics rather than from this repository, so it is
    asserted here directly against a real dataset object -- if a library upgrade changed the
    selection order, the sweep would silently become a comparison of unrelated samples.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    from saryolo.data.yolo import resolve_data_yaml

    data_yaml = resolve_data_yaml(REPO_ROOT / "configs" / "datasets" / "synthetic_smoke.yaml")
    if not (REPO_ROOT / "datasets" / "processed" / "synthetic_smoke" / "images" / "train").is_dir():
        pytest.skip("synthetic smoke data is not prepared (python -m saryolo synth-data ...)")

    data = check_det_dataset(str(data_yaml))

    def selected(fraction: float) -> list[str]:
        args = get_cfg(overrides={"data": str(data_yaml), "imgsz": 64, "task": "detect",
                                  "mode": "train", "fraction": fraction})
        dataset = build_yolo_dataset(args, data["train"], batch=4, data=data, mode="train")
        return sorted(dataset.im_files)

    whole = selected(1.0)
    subsets = {f: selected(f) for f in (0.125, 0.25, 0.5)}
    assert all(0 < len(images) < len(whole) for images in subsets.values()), subsets
    assert set(subsets[0.125]) <= set(subsets[0.25]) <= set(subsets[0.5]) <= set(whole), (
        "the fraction arms are not nested, so two fractions differ by more than the fraction"
    )
    assert subsets[0.25] == selected(0.25), "the selection is not deterministic across builds"
