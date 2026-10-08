"""Every relative link and in-page anchor in the repository's markdown must resolve.

Why this file exists
--------------------
The README and the ``docs/`` tree are the deliverable a reader actually navigates, and both
are edited far more often than they are read end to end. A link that points at a renamed file,
or at a heading that has been reworded, renders as ordinary blue text and 404s only when
someone clicks it — which is exactly the kind of defect that survives an audit, because
nothing in a build or a test run touches it.

Two things are checked, and the second is the one that actually breaks:

* **Relative file links** resolve from the linking file's own directory (so a link in
  ``docs/`` is not resolved against the repository root by accident).
* **Anchors** (``#heading``, and ``path.md#heading``) match a heading in the target document,
  under GitHub's slug rules, including the ``-1``/``-2`` suffixes GitHub adds for repeated
  headings. A reworded heading breaks the anchor without touching the path, so a
  file-existence check alone would pass while the link is dead.

External URLs are deliberately *not* fetched: a network dependency in the test suite turns one
unreachable vendor page into a red build, and the repository's rule is that its own artefacts
must be checkable offline.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories that hold prose a reader navigates. Reports are included because the audit
#: documents cross-reference each other by relative path.
DOC_DIRS = ("", "docs", "paper", "reports", "configs")

#: Fenced code blocks are skipped: a link inside one is an example, not a link.
_FENCE = re.compile(r"^\s*(```|~~~)")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
#: Code spans keep their *text* in the slug (GitHub renders them as code, and keeps the
#: words); only the backticks go. Dropping the content instead would silently shorten every
#: slug of a heading that names a parameter.
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HTML_TAG = re.compile(r"<[^>]+>")


def _markdown_files() -> list[Path]:
    files = [REPO_ROOT / "README.md"]
    for name in DOC_DIRS:
        files.extend(sorted((REPO_ROOT / name).glob("*.md")))
    return [p for p in files if p.exists()]


def _slug(heading: str) -> str:
    """GitHub's heading slug: lowercase, markdown stripped, punctuation dropped.

    Each space becomes one hyphen and runs are **not** collapsed: removing an em dash from
    ``"Foo — Bar"`` leaves two spaces, so GitHub's anchor really is ``foo--bar``. Collapsing
    here would report a live anchor as dead whenever a heading contains a dash-as-punctuation,
    which several do in this repository.
    """
    text = _INLINE_CODE.sub(r"\1", heading)
    text = _HTML_TAG.sub("", text)
    text = text.replace("*", "").replace("_", "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text)


def _anchors(path: Path) -> set[str]:
    """Every anchor a document offers, including GitHub's duplicate-heading suffixes."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    in_fence = False
    for line in path.read_text().splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        found = _HEADING.match(line)
        if not found:
            continue
        slug = _slug(found.group(2))
        if not slug:
            continue
        count = seen.get(slug, 0)
        out.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return out


def _links(path: Path) -> list[str]:
    """Relative link targets in ``path``, in order, with fenced blocks skipped."""
    targets: list[str] = []
    in_fence = False
    for line in path.read_text().splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        targets.extend(_LINK.findall(line))
    return targets


def _relative_targets() -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for path in _markdown_files():
        for target in _links(path):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            rows.append((path, target))
    return rows


def test_the_document_set_is_not_empty_and_the_guard_has_targets():
    """A link check with no links would pass forever, so its inputs are asserted first."""
    files = _markdown_files()
    assert len(files) >= 5, f"expected the README plus the docs tree, found {[p.name for p in files]}"
    rows = _relative_targets()
    assert len(rows) >= 40, f"only {len(rows)} relative link(s) found; the scan is not working"
    assert any("#" in target for _path, target in rows), "no anchors found; the anchor check is vacuous"


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_relative_link_in_the_document_tree_resolves(path):
    """The file must exist, and an anchor must match a heading in the file it names."""
    problems: list[str] = []
    for target in _links(path):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file_part, _, anchor = target.partition("#")
        if file_part:
            resolved = (path.parent / file_part).resolve()
            if not resolved.exists():
                problems.append(f"{target!r} -> missing {resolved.relative_to(REPO_ROOT)}")
                continue
        else:
            resolved = path
        if anchor and resolved.suffix == ".md":
            available = _anchors(resolved)
            if anchor not in available:
                problems.append(
                    f"{target!r} -> no heading in {resolved.name} slugs to {anchor!r} "
                    f"(have: {sorted(available)[:6]}...)"
                )
    assert not problems, (
        f"{path.relative_to(REPO_ROOT)} has dead links:\n  " + "\n  ".join(problems)
    )


def test_the_anchor_slug_rule_matches_github_on_the_headings_that_exist():
    """Pin the slugger against headings whose GitHub anchor is not obvious from the text.

    Every case here was checked against the rendered anchors GitHub generates for these very
    headings, because the slugger is the only part of this module that could be wrong in the
    *permissive* direction: a bug that produced short slugs would make real links look fine.
    """
    assert _slug("The core mechanism (SSAC)") == "the-core-mechanism-ssac"
    assert _slug("The core mechanism (SSAC) — pilot") == "the-core-mechanism-ssac--pilot"
    assert _slug("2. Seed sensitivity (pilot)") == "2-seed-sensitivity-pilot"
    assert _slug("`keep` is a fraction") == "keep-is-a-fraction"
    assert _slug("Does the allocation convert into a wall-clock saving? No — and the mechanism is why") == (
        "does-the-allocation-convert-into-a-wall-clock-saving-no--and-the-mechanism-is-why"
    )
