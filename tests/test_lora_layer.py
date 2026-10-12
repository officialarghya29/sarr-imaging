"""The LoRA adapter's own layer: its shape bookkeeping, its low-rank update, its merge.

``tests/test_peft.py`` covers the end-to-end arm (identity at init, the adapter trains, the
checkpoint is usable). This file covers the pieces of :class:`LoRALayer` those tests reach
only in passing: the reported shapes, the ``delta_weight`` a merge folds in, and the fact
that merging returns a *new* base rather than mutating the one being adapted -- an in-place
merge double-counts the update, which was measured at 0.28 of output drift.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from saryolo.training.peft import LoRAConfig, LoRALayer, _parent_of, merge_lora

# ------------------------------------------------------------------ construction contract

def test_lora_refuses_a_layer_it_cannot_treat_as_linear():
    """The adapter is a low-rank update to a matrix; a non-linear layer has no such update."""
    with pytest.raises(TypeError, match="nn.Conv2d/nn.Linear"):
        LoRALayer(nn.ReLU(), rank=4, alpha=8.0)


def test_lora_refuses_a_non_positive_rank():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        LoRALayer(nn.Linear(4, 4), rank=0, alpha=1.0)


# ------------------------------------------------------------------ reported shapes

def test_linear_adapter_reports_the_base_in_and_out_features():
    layer = LoRALayer(nn.Linear(6, 3), rank=4, alpha=8.0)
    assert layer.in_features == 6 and layer.out_features == 3
    assert layer.n_lora_parameters == 4 * 6 + 3 * 4


def test_conv_adapter_factors_the_flattened_kernel():
    """A conv's 'in features' is the flattened kernel, not the channel count."""
    base = nn.Conv2d(4, 8, kernel_size=3)
    layer = LoRALayer(base, rank=2, alpha=4.0)
    assert layer.in_features == 3 * 3 * 4  # kh * kw * (in_channels / groups)
    assert layer.out_features == 8
    assert layer.delta_weight().shape == base.weight.shape


# ------------------------------------------------------------------ the update

def test_delta_weight_is_exactly_zero_at_initialisation():
    """B is zero-initialised, so the folded update is the zero matrix of the base's shape."""
    layer = LoRALayer(nn.Linear(5, 4), rank=3, alpha=6.0)
    delta = layer.delta_weight()
    assert delta.shape == layer.base.weight.shape
    assert torch.equal(delta, torch.zeros_like(delta))


def test_delta_weight_matches_scale_times_b_at_a():
    layer = LoRALayer(nn.Linear(5, 4), rank=3, alpha=6.0)
    with torch.no_grad():
        layer.lora_b.normal_(std=0.1)
    expected = (layer.scale * (layer.lora_b @ layer.lora_a)).view_as(layer.base.weight)
    assert torch.allclose(layer.delta_weight(), expected)


# ------------------------------------------------------------------ merged base

def test_merged_base_returns_a_new_module_and_leaves_the_adapter_local():
    """Merging must not mutate the layer being adapted: the forward still adds B @ A."""
    layer = LoRALayer(nn.Linear(5, 4), rank=3, alpha=6.0)
    original = layer.base.weight.detach().clone()
    merged = layer.merged_base()
    assert isinstance(merged, nn.Linear)
    assert torch.equal(layer.base.weight.detach(), original), "the base weight was mutated in place"


def test_merged_base_output_equals_the_adapted_forward_at_step_zero():
    """At init the update is zero, so the merged layer computes what the adapter computes."""
    layer = LoRALayer(nn.Linear(5, 4), rank=3, alpha=6.0).eval()
    x = torch.randn(2, 5)
    merged = layer.merged_base().eval()
    with torch.no_grad():
        assert torch.allclose(layer(x), merged(x), atol=1e-6)


# ------------------------------------------------------------------ parent lookup

def test_parent_of_finds_the_owning_module_and_none_for_the_root():
    model = nn.Sequential(nn.Linear(2, 2))
    assert _parent_of(model, "0") is model
    assert _parent_of(model, "") is None


def test_merging_a_root_level_adapter_is_refused():
    """A LoRALayer has no parent to replace it with, so merging it is an error, not a no-op."""
    layer = LoRALayer(nn.Linear(2, 2), rank=1, alpha=1.0)
    with pytest.raises(ValueError, match="cannot merge the root module"):
        merge_lora(layer)


# ------------------------------------------------------------------ config surface

def test_config_scale_is_alpha_over_rank():
    assert LoRAConfig(rank=8, alpha=16.0).scale == 2.0


def test_config_describe_marks_the_method_and_records_the_block():
    described = LoRAConfig(rank=4, alpha=8.0, freeze_base=True).describe()
    assert described["method"] == "lora"
    assert described["rank"] == 4 and described["scale"] == 2.0
    assert described["freeze_base"] is True


def test_from_mapping_converts_targets_and_exclude_to_tuples():
    """A YAML list arrives as a list; the dataclass stores tuples so it is hashable/stable."""
    config = LoRAConfig.from_mapping({"targets": ["Linear"], "exclude": ["model.23"]})
    assert config.targets == ("Linear",) and config.exclude == ("model.23",)


def test_config_refuses_a_dropout_outside_the_unit_interval():
    with pytest.raises(ValueError, match="dropout"):
        LoRAConfig(dropout=1.0)
