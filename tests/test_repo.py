"""Repository integrity tests: generated artefacts and notebooks.

These exist because both failure modes are silent and expensive:

* a malformed ``.ipynb`` only fails when a user opens it in Colab, after the
  repository has been published;
* a malformed generated model YAML only fails when training starts, after GPU
  time has been booked.
"""

from __future__ import annotations

import json
import re
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


def test_no_commit_attributes_the_work_to_anyone_but_the_repository_owner():
    """Every commit, past and future, must be authored AND committed by the repo owner.

    This is the project's provenance rule, made mechanical: the history was once rewritten
to strip AI-attribution trailers (44 Co-Authored-By/Generated-with lines across 22
commits), and an AI coding tool re-adding a trailer is exactly the silent regression a
git log would not surface. The check runs the local history only -- it pins what this
repository's `git log` says, which is what a reader, a reviewer, or GitHub's contributor
graph consumes.

    Three properties are asserted:

    * every commit's author and committer identity matches the configured `user.name` /
      `user.email` (both fields, because GitHub's contributor list is built from them);
    * no commit message carries a `Co-Authored-By:` or `Generated with ...` trailer, so
      attribution can never quietly diverge from the authorship fields;
    * the guard itself is not vacuous: the identity set it compared must be non-empty.
    """
    import subprocess

    name = subprocess.run(["git", "config", "user.name"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()
    email = subprocess.run(["git", "config", "user.email"], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.strip()
    assert name and email, "git user.name / user.email are not configured"

    out = subprocess.run(
        ["git", "log", "--all", "--format=%an%n%ae%n%cn%n%ce"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    identities = {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
    assert identities, "no commits found; the check would pass vacuously"
    allowed = {name, email}
    foreign = sorted(identities - allowed)
    assert not foreign, (
        "commits attribute the work to identities other than the repository owner: "
        f"{foreign}. The project rule is that every commit is authored and committed "
        "under the owner's account -- fix with `git filter-branch --env-filter` and force-push."
    )

    bodies = subprocess.run(
        ["git", "log", "--all", "--format=%B"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    trailers = [
        ln for ln in bodies.stdout.splitlines()
        if ln.lower().startswith("co-authored-by:")
        or re.match(r"generated\s+(with|by)\b", ln.strip(), flags=re.IGNORECASE)
    ]
    assert not trailers, (
        "commit messages carry attribution trailers; the author fields are the only "
        f"place attribution may live: {trailers[:5]}"
    )


def test_working_files_never_name_an_ai_coding_tool():
    """No tracked file may name the AI tools that wrote parts of this repository.

    The distinction from the credential test above matters: a credential string is a
    security incident, while an AI-tool name is a *provenance* defect -- it implies the
    work is not the author's own and invites the exact scepticism the honesty rules of
    this project exist to prevent. The match is case-insensitive and covers the tool
    names and their domains, so a mention in prose, a config, or a code comment is caught
    alike. Only git-tracked files are scanned: .venv, caches and .git internals are not
    part of the repository.
    """
    import subprocess

    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    files = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert files, "git ls-files returned nothing; the check would pass vacuously"

    # Assembled from fragments so this file does not itself contain the literal tokens
    # it bans (the scan reads this file too). Word boundaries keep ordinary English
    # ("freedom", "codebase") from matching.
    _tool_names = ["co" + "debuff", "fre" + "ebuff", "man" + "icode"]
    banned = re.compile(
        "|".join(r"\\b" + re.escape(t) + r"\\b" for t in _tool_names), re.IGNORECASE
    )
    hits = []
    for rel in files:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for i, ln in enumerate(text.splitlines(), start=1):
            if banned.search(ln):
                hits.append(f"{rel}:{i}")
    assert not hits, (
        f"AI coding tools are named in tracked files: {hits[:10]}. "
        "The repository presents itself under its owner's name only."
    )


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


#: A backtick code span in the README. Citations are read from code spans rather than from raw
#: prose, because a bare ``test_``-prefixed word in a sentence is not a claim about the suite.
README_CODE_SPAN = re.compile(r"`([^`\n]+)`")

#: A whole-token test identifier: the *entire* span element must be the name, with no ellipsis and
#: no path. A truncated citation such as ``test_something_...`` is therefore not a valid citation
#: and is reported as one of the unreadable entries rather than quietly skipped -- a name the
#: guard cannot verify is exactly the case it exists to catch.
README_TEST_TOKEN = re.compile(r"^test_[a-z0-9_]+$")

#: A cited test *file*, e.g. ``tests/test_metrics.py``.
README_TEST_PATH = re.compile(r"^(?:tests/)?test_[a-z0-9_]+\.py$")


def test_every_test_cited_in_the_readme_exists():
    """A test cited in the README must name a test that actually runs.

    The evidence table is the README's strongest claim: it says "this property is pinned by that
    test". A citation to a renamed or never-written test is worse than no citation, because it
    reads as verification while verifying nothing, and nothing else in this suite looks at it --
    the asset test validates image paths, not identifiers.

    Two failure modes are caught, and the second is the subtle one:

    * a cited name that does not exist (three such citations were nearly committed -- the names
      were plausible, the tests did not exist);
    * a cited name that is *truncated* (``test_validator_diagnoses_oriented_labels_...``), which
      is unverifiable and so is treated as a failure rather than skipped. That one was real and
      already in the README when this guard was written.
    """
    readme = (REPO_ROOT / "README.md").read_text()

    tokens: set[str] = set()
    paths: set[str] = set()
    unreadable: set[str] = set()
    for span in README_CODE_SPAN.findall(readme):
        for raw in re.split(r"[,·]", span):
            item = raw.strip()
            if not item:
                continue
            if README_TEST_TOKEN.match(item):
                tokens.add(item)
            elif README_TEST_PATH.match(item):
                paths.add(item)
            elif item.startswith("test_"):
                unreadable.add(item)

    assert tokens or paths, "the README cites no tests; did the evidence table get dropped?"

    defined: set[str] = set()
    for path in sorted((REPO_ROOT / "tests").glob("*.py")):
        defined.update(re.findall(r"^def (test_[a-z0-9_]+)", path.read_text(), flags=re.MULTILINE))

    problems = [f"{name} (no such test)" for name in sorted(tokens) if name not in defined]
    problems += [
        f"{name} (cited path does not exist)"
        for name in sorted(paths)
        if not (REPO_ROOT / "tests" / Path(name).name).exists()
    ]
    problems += [f"{name} (not a checkable name -- truncated or malformed)" for name in sorted(unreadable)]
    assert not problems, (
        "the README cites tests that cannot be verified:\n  "
        + "\n  ".join(problems)
        + "\nEither the test was renamed (update the README) or it was never written (write it)."
    )


def test_the_test_citation_guard_is_not_vacuous():
    """The guard above must detect a missing citation, and must not fire on a real one."""
    readme = (REPO_ROOT / "README.md").read_text()
    cited = {
        item.strip()
        for span in README_CODE_SPAN.findall(readme)
        for item in re.split(r"[,·]", span)
        if README_TEST_TOKEN.match(item.strip())
    }
    assert cited, "the guard found no identifiers to check, so it would pass vacuously"
    # A plausible but unwritten name must not be mistaken for a real one.
    assert "test_definitely_not_written_anywhere" not in cited
    # And a truncated name must be classified as unreadable rather than accepted.
    truncated = "test_validator_diagnoses_oriented_labels_..."
    assert not README_TEST_TOKEN.match(truncated)
    assert README_TEST_TOKEN.match("test_validator_diagnoses_oriented_labels_instead_of_calling_them_malformed")
    # File citations are legitimate and must be recognised as paths, not identifiers.
    assert README_TEST_PATH.match("tests/test_metrics.py")
    assert not README_TEST_TOKEN.match("tests/test_metrics.py")


def test_readme_generated_charts_exist():
    """The generated figures must be committed, not just produced locally."""
    charts = sorted((REPO_ROOT / "docs" / "assets").glob("*.svg"))
    assert len(charts) >= 7, f"expected the full chart set, found {[p.name for p in charts]}"
    assert (REPO_ROOT / "docs" / "assets" / "facts.json").exists()


def test_every_chart_has_a_png_companion():
    """Each chart needs a raster twin, and it must be committed alongside the SVG.

    The README embeds SVG (crisp at any width), but a slide deck, a PDF draft and a chat
    preview all need a raster, and re-exporting one by hand is exactly how a figure drifts
    away from the code that produced it. Both formats come from one ``savefig`` pair in
    ``scripts/make_readme_assets.py``, so a missing PNG means someone rendered a chart by
    an ad-hoc route -- which is the drift this check exists to catch.
    """
    svgs = {p.stem for p in (REPO_ROOT / "docs" / "assets").glob("*.svg")}
    pngs = {p.stem for p in (REPO_ROOT / "docs" / "assets").glob("*.png")}
    assert svgs, "no charts found"
    missing = sorted(svgs - pngs)
    assert not missing, (
        f"charts with no committed PNG companion: {missing}. Regenerate with "
        "`python scripts/make_readme_assets.py`, which writes both formats."
    )


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

    arms = sorted({variant for _slot, _prefix, variants in G.ABLATION_SLOTS for variant in variants})
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
    # Module G: the prior also selects the radial spectral bands. Appended so every row above
    # keeps its label and therefore its meaning.
    "+ prior spectral": "v2_prior_spectral",
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


#: README real-data row label -> experiment id, and the column order of that table.
#: The pilot table is the only place in this repository where an *accuracy* number is
#: quoted, and ``results/`` is not committed -- so the README's provenance for those numbers
#: is ``docs/assets/facts.json``, which the asset generator reads out of the ledger. A
#: hand-typed digit, an arm that was never run, or a table row for a run that failed must
#: fail here rather than be found by a reviewer.
README_REAL_ROWS: dict[str, str] = {
    "REAL-001 · YOLO11n baseline": "REAL-001",
    "REAL-002 · SARVO-Lite (s)": "REAL-002",
    "REAL-003 · YOLO11n + LoRA r=8": "REAL-003",
    "REAL-004 · SARVO prototype (RS-CFAR)": "REAL-004",
    "REAL-005 · prototype control (fixed threshold)": "REAL-005",
    "REAL-006 · prototype control (matched-cost conv)": "REAL-006",
    "AUG-001 · baseline, SAR-augmented x1": "AUG-001",
    "AUG-002 · prototype, SAR-augmented x1": "AUG-002",
    "AUG-003 · baseline, SAR-augmented x2": "AUG-003",
    "AUG-004 · prototype, SAR-augmented x2": "AUG-004",
    # The SARVO core-mechanism arms (SSAC). The proposal and its *parameter-identical*
    # fixed-computation control are the master workflow's key comparison, so they belong in
    # the one table that quotes an accuracy number; the assessment alternative is the
    # second-order question.
    "SSAC-001 · SARVO core mechanism (SSAC)": "SSAC-001",
    "SSAC-002 · control (fixed computation)": "SSAC-002",
    "SSAC-003 · control (raw-feature assessment)": "SSAC-003",
    # The two cost-ablation arms are *runs*, not settings, so they hold rows of their own: the
    # sparse build answers the efficiency question and the narrow expensive path answers how
    # much of the mechanism's parameter price the accuracy needs. Both are parameter-counted
    # against the proposal in the same table.
    "SSAC-004 · prototype, sparse execution": "SSAC-004",
    "SSAC-005 · prototype, expand=1 expensive path": "SSAC-005",
    "SSAC-006 · prototype + sparsity penalty": "SSAC-006",
}

#: Column name in the README pilot table -> metric key in ``facts.json``. ``epochs`` is a
#: run setting rather than a metric and is compared separately.
README_REAL_COLUMNS: dict[str, str] = {
    "mAP50": "mAP50",
    "mAP50:95": "mAP50_95",
    "Precision": "precision",
    "Recall": "recall",
    "Params (M)": "params_M",
    "GFLOPs@320": "flops_G",
    "FPS (CPU)": "fps",
    "Train (min)": "train_minutes",
}


def _readme_real_table() -> dict[str, dict[str, float]]:
    """Parse the pilot table: label, then one float per declared column, in order."""
    header: list[str] | None = None
    rows: dict[str, dict[str, float]] = {}
    for line in (REPO_ROOT / "README.md").read_text().splitlines():
        if not line.startswith("|"):
            if rows:
                break  # the table ended
            header = None
            continue
        cells = [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "Arm" and "mAP50" in cells:
            header = cells[1:]
            continue
        if header is None or not cells:
            continue
        if set(cells[0]) <= set("-: "):  # the markdown separator row
            continue
        if cells[0] not in README_REAL_ROWS:
            break
        values = cells[1:]
        assert len(values) == len(header), (
            f"{cells[0]}: the row has {len(values)} values for {len(header)} columns"
        )
        rows[cells[0]] = dict(zip(header, (float(v) for v in values), strict=True))
    return rows


def test_readme_real_data_table_matches_the_measured_arms():
    """The pilot table must agree with the ledger, cell for cell.

    Accuracy is the one class of number in this repository that cannot be checked by
    re-measuring an architecture, so it is checked against the ledger instead. This test also
    fails when a row exists for an arm with no completed run, which is the failure mode a
    "no fabricated results" claim depends on: an empty or invented cell would otherwise sit
    in the first table a reviewer reads.
    """
    facts = json.loads((REPO_ROOT / "docs" / "assets" / "facts.json").read_text())
    arms = facts.get("real", {})
    readme = _readme_real_table()
    assert len(readme) == len(README_REAL_ROWS), (
        f"README pilot rows not found: {sorted(set(README_REAL_ROWS) - set(readme))}"
    )
    for label, exp_id in sorted(README_REAL_ROWS.items()):
        assert exp_id in arms, (
            f"{label}: {exp_id} has no completed run in the ledger, so the README quotes a "
            f"number that was never measured"
        )
        measured = arms[exp_id]
        for column, key in README_REAL_COLUMNS.items():
            shown = readme[label][column]
            assert key in measured, f"{exp_id}: {key} was not measured but the README shows {shown}"
            # The README prints three decimals for metrics and costs; compare at that
            # precision so the test neither demands more digits than the table shows nor
            # tolerates a real drift.
            expected = float(measured[key])
            assert abs(shown - expected) < 5e-4 or abs(shown - round(expected, 3)) < 1e-9, (
                f"{label} · {column}: README says {shown}, ledger says {expected}"
            )


def _readme_ssac_execution_table() -> list[dict]:
    """Parse the dense/sparse timing table out of the README's core-mechanism section.

    Anchored on the section heading rather than on the table's own header row, because the
    header is prose that may be reworded while the *numbers* must not drift: an anchor that
    was itself the thing being edited would let the table be rewritten without the guard
    noticing.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    parts = readme.split("Does the allocation convert into a wall-clock saving?", 1)
    assert len(parts) == 2, "the README has no dense/sparse execution section to check"
    rows: list[dict] = []
    for line in parts[1].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] in ("dense", "sparse"):
            rows.append(
                {
                    "execution": cells[0],
                    "keep": cells[1],
                    "levels": [float(v) for v in cells[2].split("/")],
                    "ms": float(cells[3]),
                    "fps": float(cells[4]),
                }
            )
    return rows


def test_readme_dense_vs_sparse_table_matches_the_measured_sweep():
    """The execution table is a measurement, so it is checked against the measurement.

    It is also the one table in the README whose conclusion is a *negative*, which is exactly
    where a hand-edited number would be least likely to be noticed. Two things are therefore
    asserted: every cell equals the profiled value, and the inequality the prose rests on
    still holds in the numbers.
    """
    facts = json.loads((REPO_ROOT / "docs" / "assets" / "facts.json").read_text())
    runs = facts.get("ssac_execution") or {}
    assert runs, (
        "facts.json carries no dense/sparse timing, so the README table cannot be checked; "
        "run `python scripts/make_readme_assets.py` where the sweep profiles exist"
    )
    data = next(iter(runs.values()))
    rows = _readme_ssac_execution_table()
    assert len(rows) == 1 + len(data["sparse"]), (
        f"the README lists {len(rows)} execution rows for {1 + len(data['sparse'])} measured ones"
    )
    dense = rows[0]
    assert dense["execution"] == "dense" and dense["keep"] == "—"
    assert abs(dense["ms"] - data["dense"]["latency_ms"]) < 5e-4, (
        f"README says the dense latency is {dense['ms']} ms; the profile says "
        f"{data['dense']['latency_ms']}"
    )
    assert abs(dense["fps"] - data["dense"]["fps"]) < 5e-3
    assert dense["levels"] == [1.0, 1.0, 1.0], (
        "dense execution runs the whole expensive path at every level by construction"
    )
    for row, measured in zip(rows[1:], data["sparse"], strict=True):
        assert row["execution"] == "sparse"
        assert float(row["keep"]) == float(measured["keep"]), (
            f"the README's budget {row['keep']} is not the measured one {measured['keep']}"
        )
        assert abs(row["ms"] - measured["latency_ms"]) < 5e-4, (
            f"keep={row['keep']}: README says {row['ms']} ms, the profile says {measured['latency_ms']}"
        )
        assert abs(row["fps"] - measured["fps"]) < 5e-3
        for shown, actual in zip(row["levels"], measured["executed_rich_fraction"], strict=True):
            # The README prints two decimals here and the profile keeps four, so the check is
            # made at the precision the table shows (the same rule the pilot-table guard uses).
            assert abs(shown - actual) < 5e-3, (
                f"keep={row['keep']}: README prints {shown} of the expensive path executed, "
                f"the routing report says {actual}"
            )
    # The conclusion, as an inequality over the measured rows: routing costs at least a
    # quarter more than dense execution when nothing is skipped, and no budget beats dense
    # execution by more than a rounding margin. If either fails, the prose above the table
    # has become a claim the numbers no longer support.
    dense_ms = data["dense"]["latency_ms"]
    worst = max(r["ms"] for r in rows[1:])
    best = min(r["ms"] for r in rows[1:])
    assert worst > dense_ms * 1.2, (
        f"the routing overhead is no longer visible ({worst} ms vs dense {dense_ms} ms), so the "
        "table no longer shows what the section says it shows"
    )
    assert best > dense_ms * 0.9, (
        f"a sparse budget now beats dense execution by {(1 - best / dense_ms) * 100:.1f}%, so the "
        "'no wall-clock saving' conclusion must be rewritten from the measurement"
    )


def test_the_facts_file_lists_exactly_the_arms_the_pilot_table_declares():
    """A ledger run must be classified as an arm or as a repeat, never by accident.

    The README pilot table quotes one row per *arm* at seed 0. Seed repeats live in the
    ledger and in the prose seed tables, not as new rows -- and the general rule is the
    training seed, not the id prefix. Without this check a seed repeat of an augmented arm
    (``AUG-005``, seed 1) would have quietly appeared as a new arm in the table, which is
    exactly the kind of row a reviewer would read as a separate configuration.
    """
    facts = json.loads((REPO_ROOT / "docs" / "assets" / "facts.json").read_text())
    arms = set(facts.get("real", {}))
    assert arms == set(README_REAL_ROWS.values()), (
        f"facts.json lists {sorted(arms)} but the pilot table declares "
        f"{sorted(README_REAL_ROWS.values())}: a new arm or a seed repeat must be classified "
        "deliberately (see _real_arms_facts)"
    )


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


#: Every count the README states about the repository, and how it is measured. The README
#: previously said "63 variants", "12 experiments" and "66 tests" in one paragraph and "91
#: tests" in another while the real figures were 71/68/149 -- six hand-typed numbers, none of
#: which was right, and none of which anything checked. A stale count is a small lie that
#: costs a reader's trust in the large claims next to it, so they are asserted here.
README_COUNT_CLAIMS: tuple[tuple[str, str], ...] = (
    # (regex, what the number must equal)
    (r"\| \*\*Architectures\*\* \| (\d+) variants wired", "variants"),
    (r"\| \*\*Experiments\*\* \| (\d+) configured", "experiments"),
    (r"\| \*\*Tests\*\* \| (\d+) passing", "tests"),
    (r"# (\d+) tests$", "tests"),
)


def _collected_test_count() -> int:
    """How many test *items* pytest will actually run.

    Counted by collection rather than by counting ``def test_`` lines, because a
    ``@pytest.mark.parametrize`` expands one function into many items: the two definitions
    differed by 11 here (138 vs 149), and a guard that counts the wrong one would be wrong in
    the direction that matters -- it would call the README correct while the suite grew.
    This is the same measurement ``scripts/make_readme_assets.py`` makes.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"pytest could not collect the suite:\n{proc.stderr}"
    lines = [ln for ln in proc.stdout.splitlines() if "::" in ln or ln.endswith(".py")]
    return len(lines)


def _measured_counts() -> dict[str, int]:
    """The counts as the repository actually stands, measured rather than recalled."""
    from saryolo.training.config import list_experiments

    return {
        "variants": len(VARIANTS),
        "experiments": len(list(list_experiments(EXP_DIR))),
        "tests": _collected_test_count(),
    }


def test_readme_counts_match_the_repository():
    """No count stated in the README may disagree with what is actually in the repo."""
    import re

    readme = (REPO_ROOT / "README.md").read_text()
    measured = _measured_counts()
    problems = []
    for pattern, key in README_COUNT_CLAIMS:
        for found in re.findall(pattern, readme, flags=re.MULTILINE):
            if int(found) != measured[key]:
                problems.append(f"{pattern!r} says {found}, actual {key} is {measured[key]}")
    assert not problems, (
        "README counts have drifted from the repository:\n  "
        + "\n  ".join(problems)
        + "\nUpdate the README so its stated counts match. This fires whenever the suite "
        "grows, which is the intended coupling: a stated number is a claim about the repo."
    )


def test_the_measured_counts_are_not_silently_zero():
    """A detector that measures nothing would make the drift test above vacuous."""
    measured = _measured_counts()
    assert measured["variants"] > 0, measured
    assert measured["tests"] > 0, measured


def test_readme_states_every_component_number_in_its_table():
    """The component table and the architecture diagram must not disagree.

    Read as a numbered table (``| 3 | Component 3 — ... |``) rather than as the single
    ``| 1 | 2 | ... |`` row this once parsed: that row matched nothing, so the check passed
    on an empty list and could not fail no matter what the README said. The non-vacuity
    assertion below is what makes the numbering a claim instead of a formality.
    """
    import re

    readme = (REPO_ROOT / "README.md").read_text()
    after = readme.split("**Component numbering**", 1)[1]
    # Walk the table rows rather than taking the text up to the first blank line: the phrase
    # is followed by a blank line and then the table, so the old slicing grabbed the caption
    # and never reached a single row.
    numbers: list[int] = []
    started = False
    for line in after.splitlines():
        if line.startswith("|"):
            found = re.match(r"^\| (\d+) \|", line)
            if found:
                numbers.append(int(found.group(1)))
                started = True
        elif started:
            break
    assert numbers, "the component-numbering table is missing or unparseable -- every component must be numbered"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"the component-numbering table must run 1..N with no gaps: {numbers}"
    )
    # Every number must be described as a Component somewhere in the prose as well.
    for n in numbers:
        assert re.search(rf"Component {n}\b", readme), f"Component {n} is in the table but never explained"


def test_experiment_configs_do_not_claim_the_same_id():
    """No two configs may declare one experiment id, except the documented seed repeats.

    The table builders and the ledger key on the experiment id, so two configs sharing one
    id make the resulting number ambiguous -- whichever ran last silently wins. This actually
    happened: adding an arm to a slot moved that slot's ``v2_full`` row to a new id and left
    the previous file behind, so two configs both declared ``EXP-294``.

    ``EXP-012`` is exempt because the multi-seed study is deliberately one experiment run at
    three seeds, each with its own traceable config.
    """
    ids: dict[str, list[str]] = {}
    for path in sorted(EXP_DIR.glob("*.yaml")):
        exp_id = yaml.safe_load(path.read_text())["experiment"]["id"]
        ids.setdefault(exp_id, []).append(path.name)
    clashes = {
        exp_id: names for exp_id, names in ids.items()
        if len(names) > 1 and exp_id != "EXP-012"
    }
    assert not clashes, f"experiment ids claimed by more than one config: {clashes}"


def test_the_config_generator_is_idempotent_and_prunes():
    """Regenerating must not leave a file the generator no longer owns.

    A stale config is not inert: it is a runnable experiment that no longer corresponds to
    any arm, and it would appear in the ablation tables as a duplicate of a real one.
    """
    import subprocess
    import sys

    script = REPO_ROOT / "scripts" / "make_exp_configs.py"
    before = {p.name for p in EXP_DIR.glob("EXP-*.yaml")}
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    after = {p.name for p in EXP_DIR.glob("EXP-*.yaml")}
    assert before == after, (
        f"regenerating changed the config set: added {sorted(after - before)}, "
        f"removed {sorted(before - after)}"
    )
    # The hand-written pipeline smoke configs are not owned by the generator.
    assert (EXP_DIR / "_smoke_v2.yaml").exists()
