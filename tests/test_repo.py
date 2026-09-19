"""Repository integrity tests: generated artefacts and notebooks.

These exist because both failure modes are silent and expensive:

* a malformed ``.ipynb`` only fails when a user opens it in Colab, after the
  repository has been published;
* a malformed generated model YAML only fails when training starts, after GPU
  time has been booked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from saryolo.nn.arch import VARIANTS, build_yaml_dict, variant_filename

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "configs" / "models"
EXP_DIR = REPO_ROOT / "configs" / "exp"


def test_notebooks_are_valid_json():
    """Every notebook must parse as JSON and carry the nbformat structure Colab needs."""
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks, "no notebooks found"
    for path in notebooks:
        data = json.loads(path.read_text())
        assert data["nbformat"] == 4, f"{path.name}: unexpected nbformat"
        assert isinstance(data["cells"], list) and data["cells"], f"{path.name}: no cells"
        for i, cell in enumerate(data["cells"]):
            assert cell["cell_type"] in ("markdown", "code"), f"{path.name} cell {i}: bad type"
            assert isinstance(cell["source"], (str, list)), f"{path.name} cell {i}: bad source"
            if cell["cell_type"] == "code":
                assert "outputs" in cell, f"{path.name} cell {i}: code cell missing 'outputs'"


def test_notebooks_do_not_contain_credentials():
    """Guard against a token or key ever being pasted into a notebook."""
    import re

    patterns = (r"ghp_[A-Za-z0-9]{20,}", r"gho_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{20,}",
                r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
    for path in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")) + [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs").glob("*.md")),
    ]:
        text = path.read_text()
        for pattern in patterns:
            assert not re.search(pattern, text), f"{path.name}: looks like it contains a credential"


def test_generated_model_yamls_match_the_builder():
    """The committed YAMLs must be exactly what the builder produces.

    If they drift, a run described by a committed config would not be the model
    the builder (and therefore the tests) verified.
    """
    generated = sorted(MODELS_DIR.glob("*.yaml"))
    assert generated, "no generated model YAMLs found; run `python -m saryolo arch --variant all`"
    expected_names = {variant_filename(spec) for spec in VARIANTS.values()}
    committed_names = {p.name for p in generated}
    missing = expected_names - committed_names
    assert not missing, f"model YAMLs missing from configs/models: {sorted(missing)}"

    for name in sorted(expected_names):
        path = MODELS_DIR / name
        text = path.read_text()
        assert "GENERATED FILE" in text, f"{name}: missing the generated-file header"
        # The file name must encode the scale, or ultralytics silently builds scale 'n'.
        data = yaml.safe_load(text)
        assert "backbone" in data and "head" in data, f"{name}: missing backbone/head"
        assert data["head"][-1][2] == "Detect", f"{name}: last head row must be Detect"


def test_model_yamls_are_not_hand_edited_out_of_sync():
    """A YAML whose body differs from the builder's output means someone edited it by hand."""
    for _name, spec in sorted(VARIANTS.items()):
        path = MODELS_DIR / variant_filename(spec)
        if not path.exists():
            continue
        # Compare the parsed architecture, ignoring the spec's mutable nc field.
        spec.nc = int(spec.nc)
        committed = yaml.safe_load(path.read_text())
        rebuilt = build_yaml_dict(spec)
        for key in ("backbone", "head"):
            assert committed[key] == rebuilt[key], (
                f"{path.name}: {key} differs from the builder output. "
                "Regenerate with `python -m saryolo arch --variant all` instead of editing by hand."
            )


def test_experiment_configs_reference_existing_files():
    """Every EXP config must point at model and dataset files that exist."""
    configs = sorted(p for p in EXP_DIR.glob("*.yaml"))
    assert configs, "no experiment configs found"
    for path in configs:
        raw = yaml.safe_load(path.read_text())
        for key in ("model", "dataset"):
            if key not in raw:
                continue
            target = (path.parent / raw[key]).resolve()
            assert target.exists(), f"{path.name}: {key} path does not exist: {target}"


def test_experiment_configs_declare_a_seed():
    """A run without an explicit seed is not reproducible, so it must not load."""
    from saryolo.training.config import load_experiment

    for path in sorted(EXP_DIR.glob("*.yaml")):
        cfg = load_experiment(path)
        assert isinstance(cfg.seed, int)


def test_readme_references_only_existing_assets():
    """Every image and local link in the README must resolve.

    A broken chart path is invisible in a diff and only shows up as a torn image
    on the project page, which is a bad first impression for a README that is the
    entry point to the research.
    """
    import re

    readme = (REPO_ROOT / "README.md").read_text()
    referenced = set(
        re.findall(r"<img\s+src=\"([^\"]+)\"", readme)
        + re.findall(r"\]\((?!https?://|#)([^)]+)\)", readme)
    )
    assert referenced, "README references no assets; did the chart links get dropped?"
    missing = sorted(ref for ref in referenced if not (REPO_ROOT / ref).exists())
    assert not missing, f"README points at files that do not exist: {missing}"


def test_readme_generated_charts_exist():
    """The generated figures must be committed, not just produced locally."""
    charts = sorted((REPO_ROOT / "docs" / "assets").glob("*.svg"))
    assert len(charts) >= 7, f"expected the full chart set, found {[p.name for p in charts]}"
    assert (REPO_ROOT / "docs" / "assets" / "facts.json").exists()


def test_readme_charts_have_no_text_collisions():
    """Charts must be readable, not merely valid.

    A caption that overlaps a title, or an axis label sitting on top of a data
    label, still renders as a perfectly valid SVG -- so nothing else in this suite
    can detect it, and the problem only becomes visible to a reader. This runs the
    layout checker, which measures the text bounding boxes matplotlib actually
    computed rather than trusting the source.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_chart_layout.py")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "chart layout problems detected:\n" + (proc.stdout or "") + (proc.stderr or "")
    )


def test_generated_charts_are_reproducible():
    """A chart must not change when nothing changed.

    Matplotlib stamps a creation date into SVG output by default, so every
    regeneration of an unchanged chart produced a real diff. That buries genuine
    changes in noise and makes it impossible to tell whether a committed figure
    still corresponds to the code that produced it.
    """
    charts = sorted((REPO_ROOT / "docs" / "assets").glob("*.svg"))
    assert charts
    stamped = [p.name for p in charts if "<dc:date>" in p.read_text()]
    assert not stamped, (
        f"{stamped} carry a creation timestamp; pass metadata={{'Date': None}} to savefig "
        "and set rcParams['svg.hashsalt'] so regenerating an unchanged chart is byte-identical"
    )


def test_every_registry_dataset_has_a_matching_data_config():
    """A dataset the registry advertises must ship a data config that agrees with it.

    Drift here is silent and expensive: the registry, the README table and
    docs/DATASETS.md all list the dataset, the code happily returns its classes,
    and the failure only appears as a confusing "dataset not found" once someone
    actually tries to train on it. Class ids must also match, since a reordered
    name list silently relabels every annotation.
    """
    from saryolo.data.registry import DATASETS

    for key, spec in sorted(DATASETS.items()):
        path = REPO_ROOT / "configs" / "datasets" / f"{key}.yaml"
        assert path.exists(), f"{key}: missing {path.relative_to(REPO_ROOT)}"
        data = yaml.safe_load(path.read_text())
        assert data["nc"] == len(spec.classes), (
            f"{key}: data config declares nc={data['nc']} but the registry has {len(spec.classes)} classes"
        )
        assert list(data["names"]) == list(spec.classes), (
            f"{key}: class names/order differ from the registry ({list(data['names'])} vs {list(spec.classes)})"
        )


@pytest.mark.parametrize("exp_id", ["EXP-001", "EXP-007", "EXP-016"])
def test_core_experiments_exist(exp_id):
    """The experiments every table depends on must be present."""
    matches = [p for p in EXP_DIR.glob("*.yaml") if p.name.startswith(exp_id)]
    assert matches, f"missing experiment config for {exp_id}"


def test_every_ablation_arm_has_a_runnable_experiment_config():
    """Each ablation arm needs a config, or its table row can never be filled.

    The arms previously had model YAMLs but no experiment configs, so the ablation tables
    had nothing to consume: a cell can only be filled by a run, and a run needs a committed
    config. This pins the generator's coverage against the variant registry so a newly added
    arm cannot be quietly unrunnable.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import make_exp_configs as G

    arms = sorted({variant for _slot, variants in G.ABLATION_SLOTS for variant in variants})
    assert arms, "the ablation matrix is empty"
    for variant in arms:
        assert variant in VARIANTS, f"ablation arm {variant!r} is not a declared model variant"
        model = MODELS_DIR / variant_filename(VARIANTS[variant])
        assert model.exists(), f"ablation arm {variant!r} has no model YAML: {model.name}"
        configs = sorted(EXP_DIR.glob(f"*_{variant}.yaml"))
        assert configs, f"ablation arm {variant!r} has no experiment config, so it can never be run"


def test_paper_table_rows_reference_real_models():
    """Every variant named in a paper table must exist in the variant registry.

    The table builders match variants by name, so a typo produces a row that is
    permanently ``TBD`` while looking entirely healthy -- the worst kind of error in a
    results table, because nothing about the output suggests a bug.
    """
    from saryolo.paper.tables import MODULE_ABLATION_GROUPS, REMOVAL_ABLATION_ROWS

    named = [v for arms in MODULE_ABLATION_GROUPS.values() for v in arms]
    named += [v for _label, v in REMOVAL_ABLATION_ROWS]
    unknown = sorted({v for v in named if v not in VARIANTS})
    assert not unknown, f"paper tables name variants that do not exist: {unknown}"


# ------------------------------------------------------------------- README numbers
#: README cost-table row label -> model variant. The README's cost table is the number a
#: reader quotes first, and it is hand-maintained prose sitting next to generated charts, so
#: it is exactly the kind of figure that drifts without anyone noticing. It did: adding
#: Component 11 to the full model changed every "ours" row of the v2 slot table, and nothing
#: in the suite could see it.
README_COST_ROWS: dict[str, str] = {
    "YOLO11-s": "baseline",
    "+ SFE": "sfe",
    "+ SFM": "speckle",
    "+ SAA": "attention",
    "+ AMF": "amf",
    "+ P2 head": "p2",
    "FULL v1": "full",
    "+ clutter": "v2_clutter",
    "+ prior": "v2_prior",
    "+ freq": "v2_freq",
    "+ context": "v2_ctx",
    "FULL v2": "v2_full",
}


def _readme_cost_table() -> dict[str, tuple[float, float]]:
    """Parse ``| label | ... | params | ... | GFLOPs | ... |`` rows out of the README."""
    rows: dict[str, tuple[float, float]] = {}
    for line in (REPO_ROOT / "README.md").read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] not in README_COST_ROWS:
            continue
        try:
            rows[cells[0]] = (float(cells[2]), float(cells[4]))
        except ValueError:  # header separator row, or a placeholder such as an em dash
            continue
    return rows


def test_readme_cost_table_matches_the_measured_models():
    """The README's cost table must agree with the measured model zoo.

    Compared against ``docs/assets/facts.json``, which the asset generator recomputes from
    live ``parse_model`` builds rather than from anything typed by hand.
    """
    facts = json.loads((REPO_ROOT / "docs" / "assets" / "facts.json").read_text())
    zoo = facts["zoo"]
    readme = _readme_cost_table()
    assert len(readme) == len(README_COST_ROWS), (
        f"README cost-table rows not found: {sorted(set(README_COST_ROWS) - set(readme))}"
    )
    for label, variant in sorted(README_COST_ROWS.items()):
        params, flops = readme[label]
        expected_params = zoo[f"{variant}_s"]["params_M"]
        expected_flops = zoo[f"{variant}_s"]["flops_G"]
        # Compared at the README's own precision (3 dp for params, 2 dp for GFLOPs), so the
        # test neither demands more digits than the table shows nor tolerates a real drift.
        assert round(params, 3) == round(expected_params, 3), (
            f"{label}: README says {params} M params, measured {expected_params:.6f} M"
        )
        assert round(flops, 2) == round(expected_flops, 2), (
            f"{label}: README says {flops} GFLOPs, measured {expected_flops:.4f}"
        )


def test_readme_documents_every_ladder_step():
    """A ladder step that is missing from the README table is an undocumented experiment.

    The measured zoo is generated from ``scripts/make_readme_assets.py``'s ``LADDER``, so it
    is the authoritative list of steps; this fails when a step is added without a row.
    """
    facts = json.loads((REPO_ROOT / "docs" / "assets" / "facts.json").read_text())
    ladder = {key.rsplit("_", 1)[0] for key in facts["zoo"] if key.endswith("_s")}
    documented = set(README_COST_ROWS.values())
    assert ladder == documented, (
        f"README cost table and the measured ladder disagree -- "
        f"undocumented: {sorted(ladder - documented)}, stale rows: {sorted(documented - ladder)}"
    )
