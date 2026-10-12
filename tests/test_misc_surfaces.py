"""A last sweep of public surfaces that had no test: context wiring, parser, config, repr.

Each is small, and each is the kind of API whose silent breakage is invisible: the
metadata context is what makes conditioning reachable at all, so a walk that attaches to
zero adapters would leave a conditioned run unconditioned and still pass every shape and
look-forward check; the CLI parser is the entry point a user actually types.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from saryolo.cli import build_parser
from saryolo.inference import ValidImage
from saryolo.nn.arch import _Builder
from saryolo.nn.model import SARYOLODetectionModel, attach_metadata_context, set_batch_metadata
from saryolo.training.config import ExperimentConfig
from saryolo.training.hard_examples import ImageDifficulty


def _build(variant: str):
    import saryolo  # noqa: F401  (registers custom layers)
    from saryolo.nn.arch import VARIANTS, build_yaml_dict

    torch.manual_seed(0)
    return SARYOLODetectionModel(build_yaml_dict(VARIANTS[variant]), ch=3, nc=1, verbose=False)


# ------------------------------------------------------------------ metadata context

def test_attach_metadata_context_wires_every_conditioned_adapter_to_one_context():
    model = _build("cond_film")
    attached = attach_metadata_context(model)
    assert attached > 0, "a conditioned model has adapters that must be wired"
    assert model.metadata_context is not None
    from saryolo.nn.modules.conditioning import AcquisitionConditionedAdapter

    adapters = [m for m in model.modules() if isinstance(m, AcquisitionConditionedAdapter)]
    assert all(m.context is model.metadata_context for m in adapters), (
        "the adapters do not share the model's context, so setting the batch would condition some"
    )


def test_attach_metadata_context_is_zero_for_an_unconditioned_model():
    """Zero is a valid answer: a plain detector has nothing to condition."""
    assert attach_metadata_context(_build("baseline")) == 0


def test_set_batch_metadata_reports_whether_conditioning_is_consumed():
    """``False`` means metadata was handed to a model that cannot use it."""
    conditioned = _build("cond_film")
    assert set_batch_metadata(conditioned, None) is True
    assert conditioned.metadata_context.is_set is False

    plain = _build("baseline")
    assert set_batch_metadata(plain, None) is False, "a baseline consumes no metadata"


def test_set_batch_metadata_accepts_a_mapping_of_descriptors():
    model = _build("cond_film")
    descriptor = {
        "continuous": torch.zeros(1, 3),
        "categorical": torch.zeros(1, 3, dtype=torch.long),
        "availability": torch.ones(1, 6),
    }
    assert set_batch_metadata(model, descriptor) is True
    assert model.metadata_context.is_set is True
    set_batch_metadata(model, None)
    assert model.metadata_context.is_set is False


# ------------------------------------------------------------------ CLI parser

def test_the_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_the_parser_recognises_the_datasets_command():
    args = build_parser().parse_args(["datasets"])
    assert args.command == "datasets" and callable(args.func)


def test_the_parser_parses_a_train_invocation():
    args = build_parser().parse_args(["train", "--exp", "configs/exp/EXP-001_baseline.yaml"])
    assert args.command == "train" and args.exp.endswith(".yaml")


# ------------------------------------------------------------------ model helper surfaces

def test_builder_counter_matches_the_rows_it_appends():
    builder = _Builder()
    assert builder.n_rows == 0
    builder.add("backbone", -1, 1, "Conv", [3, 32, 3, 2])
    idx = builder.add("head", 0, 1, "Detect", [1])
    assert builder.n_rows == 2 and idx == 1
    assert len(builder.backbone) == 1 and len(builder.head) == 1


def test_valid_image_size_is_width_by_height():
    valid = ValidImage(array=None, height=50, width=80)
    assert valid.size == (80, 50)


def test_ssac_extra_repr_names_the_mode_and_execution():
    from saryolo.nn.modules.ssac import ScatterSelectiveRefinement

    repr_text = repr(ScatterSelectiveRefinement(8, mode="fixed", execution="sparse", tile=8, keep=0.5))
    for fragment in ("mode=fixed", "execution=sparse", "tile=8", "keep=0.5"):
        assert fragment in repr_text, f"{fragment!r} missing from the module repr: {repr_text}"


# ------------------------------------------------------------------ experiment config

def _config(**extra) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="EXP-X",
        name="test",
        model_path=Path("m.yaml"),
        dataset_path=Path("d.yaml"),
        train={"seed": 7, "imgsz": 320},
        extra=extra,
    )


def test_experiment_config_reads_seed_and_imgsz_from_the_train_block():
    cfg = _config()
    assert cfg.seed == 7 and cfg.imgsz == 320


def test_data_fraction_defaults_to_one_when_absent():
    assert _config().data_fraction == 1.0


def test_data_fraction_refuses_a_zero_or_oversized_fraction():
    with pytest.raises(ValueError, match="data_fraction must be in"):
        _ = _config(data_fraction=0.0).data_fraction
    with pytest.raises(ValueError, match="data_fraction must be in"):
        _ = _config(data_fraction=2.0).data_fraction


def test_data_fraction_refuses_a_non_numeric_value():
    with pytest.raises(ValueError, match="must be a number"):
        _ = _config(data_fraction="half").data_fraction


def test_config_to_dict_reports_the_resolved_fraction():
    payload = _config(data_fraction=0.5).to_dict()
    assert payload["data_fraction"] == 0.5
    assert payload["experiment_id"] == "EXP-X"


# ------------------------------------------------------------------ difficulty row

def test_image_difficulty_row_separates_misses_from_spurious_firings():
    difficulty = ImageDifficulty(
        image="img1", n_gt=3, n_det=2,
        counts={"small_object_miss": 2, "clutter_false_positive": 1, "true_positive": 1},
        score=1.25,
    )
    row = difficulty.as_row()
    assert row["missed"] == 2 and row["spurious"] == 1
    assert row["score"] == 1.25
    assert row["counts"]["true_positive"] == 1
