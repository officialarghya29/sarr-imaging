"""Guards on the audit and research-planning documents.

These documents are the project's own statement of what is and is not established. They
are read by a reviewer *before* the paper, so an overclaim in one of them is more damaging
than the same sentence in a draft -- it reads as the author's considered position. The
checks below are the mechanical part of "do not claim what you have not measured":

* the documents that the master context requires actually exist;
* no audit document claims a measured accuracy result, because none exists;
* every claim-bearing document distinguishes verified from blocked material;
* the reported-baseline registry is cited by the document that discusses it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Deliverables the master context names. Each must exist, or the reconnaissance stage is
#: incomplete and the absence would be invisible.
REQUIRED_DOCS = (
    "docs/literature_audit.md",
    "docs/novelty_and_overlap.md",
    "docs/research_questions.md",
    "docs/baseline_comparison.md",
    "reports/agent_initial_audit.md",
    "reports/reproduction_status.md",
)

#: Documents whose job is to separate established fact from open questions. Each must use
#: an explicit marker, so a reader can tell which rows are conclusions and which are plans.
CLAIM_DOCS = (
    "docs/literature_audit.md",
    "docs/novelty_and_overlap.md",
    "docs/research_questions.md",
    "docs/baseline_comparison.md",
    "reports/agent_initial_audit.md",
    "reports/reproduction_status.md",
)

#: A phrase that would assert a detection accuracy result. The repository has measured no
#: accuracy, so any of these appearing outside a negation or a quoted caveat is a defect.
_ACCURACY_CLAIM = re.compile(
    r"\b(?:we|saryolo|sarvo)\s+(?:achieve|achieves|obtain|obtains|reach|reaches|"
    r"outperform|outperforms|beat|beats|surpass|surpasses)\b[^.\n]{0,40}\b(?:mAP|AP)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("rel", REQUIRED_DOCS)
def test_master_context_deliverables_exist(rel):
    path = REPO_ROOT / rel
    assert path.is_file(), f"missing master-context deliverable: {rel}"
    assert path.read_text().strip(), f"{rel} is empty"


@pytest.mark.parametrize("rel", CLAIM_DOCS)
def test_audit_documents_mark_blocked_material_explicitly(rel):
    """A planning document must say what is blocked, not merely what is planned.

    The specific failure this prevents is a document that reads as a status report of
    completed work when in fact the work is planned. ``BLOCKED-ON-RUN`` (or the equivalent
    ``OPEN`` used by the research-questions doc) is the marker the rest of the repository
    already uses, so the documents stay greppable.
    """
    text = (REPO_ROOT / rel).read_text()
    markers = ("BLOCKED-ON-RUN", "OPEN", "not implemented", "not yet", "absent")
    assert any(m in text for m in markers), (
        f"{rel} contains no blocked/open marker; a planning document that never states "
        "what is unfinished reads as a report of finished work"
    )


@pytest.mark.parametrize("rel", CLAIM_DOCS)
def test_no_audit_document_claims_an_accuracy_result(rel):
    """No accuracy claim may appear in an audit document -- none has been measured."""
    text = (REPO_ROOT / rel).read_text()
    for match in _ACCURACY_CLAIM.finditer(text):
        line = text[: match.start()].count("\n") + 1
        raise AssertionError(
            f"{rel}:{line}: claims an accuracy result ({match.group(0)!r}). No model has "
            "been trained on a real SAR dataset; rephrase as a question or an experiment."
        )


def test_the_literature_audit_labels_its_evidence_depth():
    """Every repository/paper entry must say how deeply it was inspected.

    The audit distinguishes *verified* (reached and inspected), *abstract-read*, and
    *listed* (existence confirmed, contents not read). Collapsing those into one confidence
    level is how a search snippet becomes a cited fact.
    """
    text = (REPO_ROOT / "docs/literature_audit.md").read_text().lower()
    for level in ("verified", "abstract-read", "listed"):
        assert level in text, f"the audit does not use the {level!r} evidence level"


def test_the_audit_records_the_repository_existence_check():
    """The existence check is an executed command, so its output belongs in the document."""
    text = (REPO_ROOT / "docs/literature_audit.md").read_text()
    assert "git ls-remote" in text, "the audit does not record how existence was checked"
    assert "EXISTS" in text, "the audit does not record the existence-check result"


def test_the_baseline_doc_names_the_missing_lora_comparator():
    """The master context requires a LoRA comparison; the gap must be stated, not omitted."""
    text = (REPO_ROOT / "docs/baseline_comparison.md").read_text()
    assert "LoRA" in text
    assert "not implemented" in text.lower() or "missing" in text.lower(), (
        "the baseline document must state that the LoRA comparator does not exist yet"
    )


def test_the_reported_baselines_registry_is_referenced_by_the_documents_that_use_it():
    """A cited published number must point at the registry, so the caveats travel with it."""
    for rel in ("docs/literature_audit.md", "docs/baseline_comparison.md"):
        text = (REPO_ROOT / rel).read_text()
        assert "reported_baselines" in text, (
            f"{rel} quotes published numbers without pointing at the registry that records "
            "their caveats"
        )


def test_the_reproduction_status_matches_the_actual_test_count():
    """A status document that quotes a stale suite size is a small lie about the repo.

    The number is read from the document and compared against a fresh collection, so the
    document cannot drift away from the suite it claims to summarise.
    """
    import subprocess
    import sys

    text = (REPO_ROOT / "reports/reproduction_status.md").read_text()
    claimed = {int(n) for n in re.findall(r"\*\*(\d+) passed\*\*", text)}
    assert claimed, "the reproduction report states no test count"

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    collected = len([ln for ln in proc.stdout.splitlines() if "::" in ln or ln.endswith(".py")])
    assert claimed == {collected}, (
        f"reproduction_status.md claims {sorted(claimed)} tests but the suite collects {collected}"
    )
