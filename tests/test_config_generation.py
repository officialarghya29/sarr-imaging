"""Tests for the model-YAML generator and the arm-spec validation.

Every architecture in the repository is described by a committed YAML under
``configs/models/``, and every one of those files is produced by
:func:`saryolo.nn.arch.build_yaml_text`. Two guards already existed for the *body*: the files
must exist and the parsed ``backbone``/``head`` rows must equal the builder's output. What was
unguarded is everything else the generator emits -- the provenance **header** (which scale the
file encodes, which module slug was enabled, whether the arm exists for the cost claim, and
which loss block it carries) and the CLI that writes the files at all. A header that drifted
would mislabel a model whose weights are fine, which is exactly the kind of claim this
repository treats as a defect.

The `ModelSpec` validation is the second half: those refusals are what stop a mis-specified
arm from being built and then measured, so each one is pinned here rather than left to the
arm that happens to trip it first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from saryolo.nn import arch
from saryolo.nn.arch import VARIANTS, ModelSpec, build_yaml_text, variant_filename

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "configs" / "models"


# ============================================================ the committed files vs the generator
def test_every_committed_model_yaml_is_byte_identical_to_the_generator():
    """The committed YAML -- header included -- must be exactly what the generator emits.

    The existing body guard parses the file and compares ``backbone``/``head``, which ignores
    the provenance header. The header is a claim in its own right: it names the scale the file
    encodes (the one thing ultralytics reads from the *file name*, not the body), the enabled
    module slug, and whether the arm exists for the cost claim. Comparing the whole text ties
    all of that to the spec, so a hand-edited comment cannot survive next to a correct graph.
    """
    specs = sorted(VARIANTS.items())
    assert specs, "the variant registry is empty, so this guard would pass vacuously"
    for name, spec in specs:
        path = MODELS_DIR / variant_filename(spec)
        assert path.exists(), f"{name}: missing generated YAML {path.name}"
        assert path.read_text() == build_yaml_text(spec), (
            f"{path.name}: the committed file differs from the generator output. Regenerate "
            "with `python -m saryolo arch --variant all` instead of editing it by hand."
        )


def test_the_yaml_header_states_the_scale_modules_arm_purpose_and_loss():
    """The header fields a reader trusts must name the spec's own values, not defaults."""
    spec = VARIANTS["cfar_n"]
    text = build_yaml_text(spec)
    assert "GENERATED FILE, do not edit by hand" in text
    assert f"# scale={spec.scale}  levels={'-'.join(spec.levels)}  modules=cfar" in text
    assert "# efficiency_arm=False" in text
    assert "# sar_loss=off" in text
    assert f"# {spec.notes}" in text, "the provenance note was dropped from the header"

    # A stock baseline has no modules: the header must say so rather than print an empty list.
    assert "modules=none (stock YOLO11)" in build_yaml_text(VARIANTS["baseline_n"])

    # An arm that exists for the cost claim advertises itself, and a loss-carrying arm names
    # its block -- the two provenance bits the efficiency table and the criterion keys on.
    efficiency = next(s for s in VARIANTS.values() if s.efficiency)
    assert "efficiency_arm=True" in build_yaml_text(efficiency)
    loss_arm = next(s for s in VARIANTS.values() if s.sar_loss)
    assert f"# sar_loss={loss_arm.sar_loss}" in build_yaml_text(loss_arm)
    assert "sar_loss" in yaml.safe_load(build_yaml_text(loss_arm))
    assert "sar_loss" not in yaml.safe_load(build_yaml_text(VARIANTS["cfar_n"]))


def test_variant_filename_accepts_a_variant_name_or_a_spec():
    """The string form is what the CLI and callers pass; both routes must agree."""
    assert variant_filename("cfar_n") == variant_filename(VARIANTS["cfar_n"])
    assert variant_filename("cfar_n").startswith("yolo11n_"), (
        "the scale prefix is what ultralytics reads to pick the compound scale"
    )


def test_module_names_lists_the_adapter_slot_and_is_empty_for_a_stock_arm():
    """The slug list is the header's description of the graph, so it must match it."""
    # The adapter is input-side, so it leads the slug list; the arm also carries a neck stack.
    adapter_slugs = VARIANTS["in_identity"].module_names
    assert adapter_slugs[0] == "adapter", adapter_slugs
    assert "adapter" in adapter_slugs
    assert VARIANTS["baseline_n"].module_names == []
    assert VARIANTS["cfar_n"].module_names == ["cfar"]
    assert VARIANTS["ssac_n"].module_names == ["ssac"]


# ============================================================================ the CLI
def test_the_arch_cli_writes_one_variant_and_refuses_an_unknown_one(tmp_path, monkeypatch):
    """The CLI is how the committed files are produced, so it must write and must refuse."""
    spec = VARIANTS["cfar_n"]
    original_nc = spec.nc
    try:
        monkeypatch.setattr(
            sys, "argv", ["arch", "--variant", "cfar_n", "--nc", "3", "--out", str(tmp_path)]
        )
        arch._main()
        written = tmp_path / variant_filename(spec)
        assert written.exists()
        assert yaml.safe_load(written.read_text())["nc"] == 3

        monkeypatch.setattr(
            sys, "argv", ["arch", "--variant", "not_a_variant", "--out", str(tmp_path)]
        )
        with pytest.raises(SystemExit, match="Unknown variant"):
            arch._main()
    finally:
        spec.nc = original_nc  # _main mutates the shared spec; leave the registry as found


def test_the_arch_cli_writes_every_variant_for_the_all_selector(tmp_path, monkeypatch):
    """``--variant all`` must emit one file per registered arm, at the default class count."""
    monkeypatch.setattr(sys, "argv", ["arch", "--variant", "all", "--nc", "1", "--out", str(tmp_path)])
    arch._main()
    written = {p.name for p in tmp_path.glob("*.yaml")}
    expected = {variant_filename(spec) for spec in VARIANTS.values()}
    assert written == expected, f"missing: {sorted(expected - written)}"


# ======================================================== arm-spec validation
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cfar": "bogus"}, "cfar must be one of"),
        ({"ssac": "bogus"}, "ssac must be one of"),
        ({"ssac_execution": "batched"}, "ssac_execution must be one of"),
        ({"ssac": "adaptive", "ssac_scales": ()}, "ssac_scales must not be empty"),
        ({"ssac": "adaptive", "ssac_tile": 0}, "ssac_tile must be >= 1"),
        ({"ssac": "adaptive", "ssac_keep": 0.0}, "ssac_keep must be a fraction"),
        ({"ssac": "adaptive", "ssac_keep": 1.5}, "ssac_keep must be a fraction"),
        ({"ssac": "adaptive", "ssac_execution": "sparse", "ssac_keep": 1.0}, "refines every tile"),
        ({"scale": "z"}, "scale must be one of"),
        ({"levels": ("p2", "p4")}, "P2 level must come with"),
        ({"levels": ("p3",)}, "contiguous"),
        ({"levels": ("p4", "p5")}, "contiguous"),
    ],
)
def test_a_model_spec_refuses_every_field_it_validates(kwargs, match):
    """Each refusal stops a mis-specified arm before it can be built and measured.

    This matters more than it looks: the variant registry is the definition of every experiment,
    so a field that fails validation only at model-build time would surface as an opaque
    parse error deep in ultralytics, or -- worse for a field with no downstream check at all --
    as a running arm nobody intended.
    """
    with pytest.raises(ValueError, match=match):
        ModelSpec("spec_under_test", **kwargs)


def test_a_valid_arm_spec_builds_and_reports_its_declared_fields():
    """The refusals above must not be reaching a spec that is actually well-formed."""
    spec = ModelSpec("probe", scale="s", cfar="cfar", ssac="adaptive", levels=("p2", "p3", "p4", "p5"))
    assert spec.has_p2 is True
    assert spec.module_names == ["cfar", "ssac", "p2_head"]
    text = build_yaml_text(spec)
    assert "levels=p2-p3-p4-p5" in text
