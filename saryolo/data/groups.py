"""Cross-source grouping and leave-one-source-out (LOSO) folds.

Why this exists
---------------
The project's headline claim is that a detector can generalise to a *source it was
never trained on* -- a different satellite, or a different sensor on the same
platform. That is a stronger claim than in-domain accuracy, and it is a different
claim from the cross-*dataset* number: two datasets can share a sensor, and one
dataset can mix several (SAR-Ship-Dataset is Sentinel-1 **and** Gaofen-3; SARDet-100K
is unified from ten sources). Testing generalisation therefore needs splits grouped
by source, not by dataset name.

The machinery already exists for scenes: :func:`saryolo.data.splits.split_files`
takes a ``scene_key`` and never places two chips of one acquisition on both sides of
a split. A source key is the same mechanism with a coarser unit. What does not exist
is a way to *derive* that key, and it is the derivation where this quietly goes
wrong: a key function that matches nothing produces **one group**, and a one-group
LOSO run tests nothing at all while looking entirely healthy. Worse, it still
reports a number -- the "held-out source" is then just a random chip split wearing a
cross-sensor label.

So the design principle here is that the grouping must be *visible before it is
used*: sources are discovered and reported with their sizes, a rule that cannot
match is an error rather than a single group, and a fold whose held-out set is too
small to measure is refused instead of being reported as a real metric.

Deriving the key, without inventing a layout
--------------------------------------------
There is no safe universal rule, because the archives differ: some ship one
directory per sensor, some encode the sensor in the filename, and some carry the
information only in a metadata table. Rather than hard-coding a guess per dataset,
the caller states which of the three applies (:class:`SourceRule`), and the code
reports what it found so the assumption is falsifiable at a glance:

* ``parent`` -- the first path component(s) below a stated root, for archives that
  ship one directory per source;
* ``regex`` -- a capture group in the filename, for archives that encode it there;
* ``sidecar`` -- an explicit image -> source mapping, for archives that encode it
  nowhere, which is the honest fallback rather than a heuristic.

Refusing to guess is the point. A rule that cannot be satisfied raises with the
discovered path shape, so the failure tells you how to write the right rule.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "SourceRule",
    "SourceGroups",
    "Fold",
    "binned_rule",
    "discover_sources",
    "leave_one_out_folds",
    "write_loso_splits",
]

#: Image extensions treated as dataset images. Kept in step with the rest of the
#: data layer; a chip that is not recognised here is reported as unmatched rather
#: than silently dropped.
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SourceRule:
    """How to derive a source key for one image.

    Exactly one strategy applies, and it must be stated rather than inferred. Use the
    constructors (:meth:`parent`, :meth:`regex`, :meth:`sidecar`) so the fields that
    do not apply cannot be half-filled.

    Attributes:
        kind: ``"parent"``, ``"regex"`` or ``"sidecar"``.
        root: For ``parent``: the directory the relative path is taken below.
        depth: For ``parent``: how many components to keep. ``1`` means "the first
            directory under the root", which is the usual one-directory-per-sensor
            layout.
        pattern: For ``regex``: a pattern with exactly one capture group, applied to
            the filename stem.
        mapping: For ``sidecar``: image stem (or filename) -> source key.
    """

    kind: str
    root: Path | None = None
    depth: int = 1
    pattern: str | None = None
    mapping: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("parent", "regex", "sidecar"):
            raise ValueError(f"unknown rule kind {self.kind!r}; expected parent/regex/sidecar")
        if self.kind == "parent":
            if self.root is None:
                raise ValueError("a 'parent' rule needs a root directory")
            if self.depth < 1:
                raise ValueError(f"depth must be >= 1, got {self.depth}")
        elif self.kind == "regex":
            if not self.pattern:
                raise ValueError("a 'regex' rule needs a pattern")
            if re.compile(self.pattern).groups != 1:
                raise ValueError(
                    f"pattern {self.pattern!r} must have exactly one capture group, "
                    f"got {re.compile(self.pattern).groups}; the group is the source key"
                )
        elif self.kind == "sidecar":
            if not self.mapping:
                raise ValueError("a 'sidecar' rule needs a mapping")

    @classmethod
    def parent(cls, root: str | Path, depth: int = 1) -> SourceRule:
        """One directory per source: the key is the first ``depth`` components."""
        return cls(kind="parent", root=Path(root), depth=depth)

    @classmethod
    def regex(cls, pattern: str) -> SourceRule:
        """Filename-encoded source: the key is the single capture group."""
        return cls(kind="regex", pattern=pattern)

    @classmethod
    def sidecar(cls, mapping: dict[str, str]) -> SourceRule:
        """Explicit mapping, loaded from wherever the archive keeps it."""
        return cls(kind="sidecar", mapping=dict(mapping))

    def describe(self) -> str:
        if self.kind == "parent":
            return f"parent(depth={self.depth}) below {self.root}"
        if self.kind == "regex":
            return f"regex({self.pattern!r}) on the file stem"
        return f"sidecar({len(self.mapping or {})} entries)"

    def key_for(self, path: Path) -> str | None:
        """The source key for one path, or ``None`` if this rule cannot resolve it.

        Returning ``None`` rather than raising is deliberate: the caller collects every
        unresolved path and reports them together, which is far more useful than failing
        on the first one.
        """
        if self.kind == "parent":
            assert self.root is not None
            try:
                rel = Path(path).resolve().relative_to(Path(self.root).resolve())
            except ValueError:
                return None
            parts = rel.parts[:-1]  # drop the filename
            if len(parts) < self.depth:
                return None
            return "/".join(parts[: self.depth])
        if self.kind == "regex":
            assert self.pattern is not None
            match = re.search(self.pattern, Path(path).stem)
            return match.group(1) if match else None
        assert self.mapping is not None
        return self.mapping.get(Path(path).stem) or self.mapping.get(Path(path).name)


def _format_bound(value: float) -> str:
    """Compact, stable label for a bin boundary (``5`` rather than ``5.0``)."""
    return f"{value:g}"


def binned_rule(
    values: dict[str, float | None],
    edges: Sequence[float],
    field: str = "resolution_m",
    allow_missing: bool = False,
) -> SourceRule:
    """Group images into bins of a *continuous* acquisition field.

    This is the cross-resolution protocol (Experiment D of the brief), and it exists because
    :meth:`SourceRule.parent` and :meth:`SourceRule.regex` cannot express it: resolution is a
    number, not a directory name, and the same sensor spans a range. Grouping by equality
    (``MetadataTable.groups``) would give one group per distinct value -- on Sentinel-1 that is
    a group per metre, so almost every "source" would be a handful of chips.

    The bins are half-open and unbounded at both ends -- ``(-inf, e1]``, ``(e1, e2]``, ...,
    ``(en, +inf)`` -- so every finite value lands in exactly one bin. Open ends matter: the
    lowest and highest resolutions in a real archive are usually *not* near the edges a caller
    guesses, and a closed range would either drop them or need a special case.

    Args:
        values: Image stem -> value from the metadata table. A stem present here with a
            ``None`` value means the field was recorded as unknown for that image.
        edges: Ascending bin boundaries, at least one.
        field: Field name, used only to label the groups.
        allow_missing: Permit images whose value is unknown or absent. Off by default for the
            same reason as in :func:`discover_sources`: such an image belongs to no bin, so it
            is absent from every fold and silently shrinks the held-out set. An "unknown
            resolution" group would be worse still -- it would be held out as a fold that tests
            nothing about resolution.

    Returns:
        A ``sidecar`` rule whose keys are bin labels such as ``resolution_m<=5`` and
        ``resolution_m>20``.

    Raises:
        ValueError: If ``edges`` is empty or not strictly ascending, if any value is not
            finite, or if values are missing and ``allow_missing`` is False.
    """
    edges = [float(e) for e in edges]
    if not edges:
        raise ValueError("at least one bin edge is required")
    for previous, current in zip(edges, edges[1:], strict=False):
        if current <= previous:
            raise ValueError(
                f"bin edges must be strictly ascending, got {edges}; a repeated or reversed "
                f"edge makes an empty bin, which would be reported as a 'source' with no images"
            )

    labels = [f"{field}<={_format_bound(edges[0])}"]
    labels += [
        f"{_format_bound(lo)}<{field}<={_format_bound(hi)}"
        for lo, hi in zip(edges, edges[1:], strict=False)
    ]
    labels.append(f"{field}>{_format_bound(edges[-1])}")

    def label_for(value: float) -> str:
        for i, edge in enumerate(edges):
            if value <= edge:
                return labels[i]
        return labels[-1]

    mapping: dict[str, str] = {}
    missing: list[str] = []
    for stem, value in values.items():
        if value is None:
            missing.append(stem)
            continue
        number = float(value)
        # NaN compares false against every edge, so it would fall through to the last bin and be
        # reported as the *highest* resolution class without anything looking wrong.
        if not math.isfinite(number):
            raise ValueError(f"{stem}: {field} is {number!r}, which cannot be binned")
        mapping[stem] = label_for(number)

    if missing and not allow_missing:
        raise ValueError(
            f"{len(missing)} image(s) have no {field} recorded (e.g. {sorted(missing)[:3]}). "
            f"They belong to no bin, so they would be absent from every fold -- and the "
            f"evaluation set would shrink to a subset nobody chose. Fill the metadata table, "
            f"or pass allow_missing=True and accept that the cross-resolution number is "
            f"computed on the rest."
        )
    if not mapping:
        raise ValueError(
            f"no image has a recorded {field}, so no bin could be populated. Building folds from "
            f"this rule would produce nothing to train or test on."
        )
    if len(set(mapping.values())) < 2:
        # Say so precisely, because this is the one failure of a cross-resolution run that is
        # easy to miss: the folds still build and the run still reports a number.
        raise ValueError(
            f"every image with a recorded {field} falls into a single bin "
            f"({sorted(set(mapping.values()))}) by edges {edges}. Cross-resolution evaluation "
            f"needs at least two populated bins; widen the edges, or the data does not contain "
            f"a resolution shift to measure."
        )
    return SourceRule.sidecar(mapping)


@dataclass
class SourceGroups:
    """The discovered grouping, with everything needed to audit it."""

    groups: dict[str, list[str]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    rule: str = ""

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_unmatched(self) -> int:
        return len(self.unmatched)

    def sizes(self) -> dict[str, int]:
        return {key: len(files) for key, files in sorted(self.groups.items())}

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "n_groups": self.n_groups,
            "sizes": self.sizes(),
            "unmatched": self.unmatched[:50],
            "n_unmatched": len(self.unmatched),
        }

    def summary(self) -> str:
        lines = [f"Source grouping via {self.rule}", f"  {self.n_groups} source(s):"]
        for key, count in self.sizes().items():
            lines.append(f"    {key}: {count} images")
        if self.unmatched:
            lines.append(
                f"  {len(self.unmatched)} image(s) could not be keyed, e.g. {self.unmatched[:3]}"
            )
        return "\n".join(lines)


def discover_sources(
    paths: list[str | Path],
    rule: SourceRule,
    allow_unmatched: bool = False,
) -> SourceGroups:
    """Group image paths by source, reporting -- and by default refusing -- gaps.

    Args:
        paths: Image paths.
        rule: How to derive the key.
        allow_unmatched: Permit images the rule cannot key. Off by default: an unmatched
            image is silently absent from every fold, so the held-out evaluation would
            be measured on a subset nobody chose. Turn this on only when the unmatched
            files are known non-images.

    Raises:
        ValueError: If no image could be keyed, if fewer than two sources were found, or
            if images could not be keyed and ``allow_unmatched`` is False. Two sources is
            the floor because a single source has nothing to hold out.
    """
    groups: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for path in paths:
        path = Path(path)
        key = rule.key_for(path)
        if key is None:
            unmatched.append(str(path))
            continue
        groups.setdefault(key, []).append(str(path))

    result = SourceGroups(groups={k: sorted(v) for k, v in sorted(groups.items())},
                          unmatched=sorted(unmatched), rule=rule.describe())

    if not result.groups:
        raise ValueError(
            f"No image could be keyed by {rule.describe()}. "
            f"Found {len(paths)} candidate file(s); the rule does not match this layout."
        )
    if unmatched and not allow_unmatched:
        raise ValueError(
            f"{len(unmatched)} of {len(paths)} image(s) could not be keyed by "
            f"{rule.describe()} (e.g. {unmatched[:3]}). Either fix the rule or pass "
            f"allow_unmatched=True and confirm they are not dataset images -- images absent "
            f"from every fold silently shrink the evaluation set."
        )
    if result.n_groups < 2:
        raise ValueError(
            f"Only one source ({next(iter(result.groups))}) was found by {rule.describe()}. "
            f"A leave-one-source-out run needs at least two: with one group the held-out set "
            f"is just a random chip split, and the resulting number would be reported as "
            f"cross-source generalisation while measuring nothing of the kind."
        )

    # A rule pointed one level too high keys on the *split* layout instead of the sensor:
    # the groups come out as {train, val, test} and the run then holds out a split it created
    # itself, which is an in-domain evaluation wearing a cross-source label. This is not
    # hypothetical -- pointing `--rule parent` at a processed images directory does exactly
    # this, and every group then looks plausible (three groups, sensible sizes).
    #
    # The check looks at the *last* component of each key, not the whole key, because raising
    # the depth turns the same mistake into `images/train`, `images/val`, `images/test`, which
    # is the identical error wearing a prefix -- and it passes every other guard here.
    split_names = {"train", "training", "val", "valid", "validation", "test", "testing"}
    if {k.replace("\\", "/").strip("/").split("/")[-1].strip().lower() for k in result.groups} <= split_names:
        raise ValueError(
            f"the discovered groups {sorted(result.groups)} are split names, not sources. "
            f"The rule is keying on the train/val/test layout rather than on sensor or "
            f"acquisition. Point --root at the level that actually varies by source and "
            f"raise --depth, or use --rule regex / --rule sidecar."
        )
    return result


@dataclass(frozen=True)
class Fold:
    """One LOSO fold: the held-out source is the test set, nothing else.

    Attributes:
        name: The held-out source key.
        test: Chips of the held-out source.
        train: Training chips, drawn only from the other sources.
        val: Validation chips, also from the other sources.
        val_is_source_held_out: Always ``False`` here, and recorded so a reader knows
            the validation set shares sources with training. Validation drives early
            stopping only -- it must never be the source of a reported number, because
            it does not test generalisation.
    """

    name: str
    test: tuple[str, ...]
    train: tuple[str, ...]
    val: tuple[str, ...]
    val_is_source_held_out: bool = False

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def to_dict(self) -> dict:
        return {"name": self.name, "sizes": self.sizes,
                "val_is_source_held_out": self.val_is_source_held_out}


def leave_one_out_folds(
    groups: SourceGroups,
    val_ratio: float = 0.2,
    seed: int = 0,
    min_test_images: int = 30,
) -> tuple[Fold, ...]:
    """Build one fold per held-out source.

    The held-out source is entirely the test set. The remaining sources are split into
    train/val by chip, which is safe here: no chip of the held-out source can reach
    either, and validation is only used for early stopping.

    Args:
        groups: Output of :func:`discover_sources`.
        val_ratio: Fraction of the *remaining* chips used for validation.
        seed: Shuffle seed for the train/val split; recorded in the manifest.
        min_test_images: A held-out source smaller than this is refused. A source with a
            handful of chips cannot support a metric, and reporting mAP on it produces a
            number that looks like evidence and is noise.

    Raises:
        ValueError: If any source is below ``min_test_images``, or if either side of a
            fold would be empty.
    """
    if not groups.groups:
        raise ValueError("no sources to hold out")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    small = {k: len(v) for k, v in groups.groups.items() if len(v) < min_test_images}
    if small:
        raise ValueError(
            f"source(s) too small to evaluate: {small} (min_test_images="
            f"{min_test_images}). Held-out sources below a few dozen chips produce a "
            f"number that looks like evidence but is noise; merge them with a related "
            f"source, or lower the threshold deliberately."
        )

    rng = random.Random(seed)
    folds: list[Fold] = []
    for held_out in sorted(groups.groups):
        test = tuple(groups.groups[held_out])
        remaining = [f for k, files in groups.groups.items() if k != held_out for f in files]
        if not remaining:
            raise ValueError(
                f"holding out {held_out!r} leaves nothing to train on; a single-source "
                f"dataset cannot support a cross-source claim"
            )
        shuffled = sorted(remaining)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_ratio))
        val, train = tuple(sorted(shuffled[:n_val])), tuple(sorted(shuffled[n_val:]))
        if not train:
            raise ValueError(f"fold {held_out!r} has an empty training set")
        folds.append(Fold(name=held_out, test=test, train=train, val=val))
    return tuple(folds)


def write_loso_splits(
    folds: tuple[Fold, ...] | list[Fold],
    out_dir: str | Path,
    groups: SourceGroups | None = None,
    seed: int = 0,
    names: list[str] | None = None,
    nc: int | None = None,
    acquisition_metadata: str | Path | None = None,
) -> Path:
    """Persist one directory per fold, plus a manifest recording how it was built.

    Layout::

        <out_dir>/<fold>/train.txt | val.txt | test.txt
        <out_dir>/<fold>/data.yaml          training: val = val.txt
        <out_dir>/<fold>/eval_holdout.yaml  measuring: val = test.txt
        <out_dir>/manifest.json

    The manifest is the point of the exercise: it records the rule, every source and its
    size, the seed, and the disjointness check for each fold, so a cross-source result can be
    traced to the grouping that produced it rather than to a remembered convention.

    There are two data configs per fold, and the distinction is the whole protocol. A
    validation split is what ``evaluate_detections`` reads, so pointing it at ``val.txt``
    would report a number measured on sources the model trained on -- an in-domain score
    labelled cross-source. ``eval_holdout.yaml`` therefore points ``val`` at ``test.txt``,
    which is how the held-out source is actually measured with the existing evaluation
    command, no special case in the metric code.

    Args:
        names: Class names, written into the data configs. Omitted configs then carry no
            ``names`` and cannot be trained on, so they are only written when supplied.
        nc: Class count, defaulting to ``len(names)``.
        acquisition_metadata: Optional metadata table whose image-stem rows are filtered per fold and
            carried into both training and held-out evaluation YAMLs.

    Raises:
        ValueError: If any chip appears in more than one split of a fold, or if a held-out
            source leaks into training. Both are checked here because a leaking fold still
            trains and still reports a number.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "protocol": "leave-one-source-out",
        "seed": seed,
        "grouping": groups.to_dict() if groups is not None else None,
        "folds": [],
    }
    if acquisition_metadata is not None:
        manifest["acquisition_metadata"] = str(Path(acquisition_metadata).resolve())

    for fold in folds:
        _assert_disjoint(fold)
        fold_dir = out_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split, items in (("train", fold.train), ("val", fold.val), ("test", fold.test)):
            (fold_dir / f"{split}.txt").write_text("".join(f"{n}\n" for n in items))
        if names:
            import yaml

            header = (
                "# GENERATED by saryolo.data.groups.write_loso_splits -- do not edit.\n"
                f"# Leave-one-source-out fold {fold.name!r}: the held-out source is test.txt.\n"
            )
            base = {"path": str(fold_dir), "nc": int(nc or len(names)), "names": list(names)}
            fold_metadata = None
            if acquisition_metadata is not None:
                from .metadata import MetadataTable

                table = MetadataTable.load(acquisition_metadata)
                # Fold splits hold full paths; the table is keyed by stem.
                fold_stems = {Path(item).stem for item in (*fold.train, *fold.val, *fold.test)}
                missing = sorted(fold_stems - set(table.entries))
                if missing:
                    raise ValueError(
                        f"acquisition metadata has no row for {len(missing)} fold image(s) "
                        f"(e.g. {missing[:3]})"
                    )
                fold_table = MetadataTable(
                    entries={stem: table.entries[stem] for stem in sorted(fold_stems)},
                    source=table.source,
                )
                fold_metadata = fold_table.save(fold_dir / "acquisition_metadata.json")
                base["acquisition_metadata"] = str(fold_metadata.resolve())
            (fold_dir / "data.yaml").write_text(
                header + yaml.safe_dump(
                    {**base, "train": str(fold_dir / "train.txt"),
                     "val": str(fold_dir / "val.txt"), "test": str(fold_dir / "test.txt")},
                    sort_keys=False,
                )
            )
            (fold_dir / "eval_holdout.yaml").write_text(
                header
                + "# 'val' is the HELD-OUT source: this is the file to evaluate on.\n"
                + yaml.safe_dump({**base, "val": str(fold_dir / "test.txt")}, sort_keys=False)
            )
        manifest["folds"].append(fold.to_dict())

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def _assert_disjoint(fold: Fold) -> None:
    """Refuse a fold whose splits overlap, or whose held-out source reaches training.

    The two checks are separate on purpose. Overlapping splits is a bookkeeping error;
    the held-out source leaking into training is a *protocol* error, and the second is
    the one that turns a cross-source claim into an in-domain one while every table
    still fills in.
    """
    train, val, test = set(fold.train), set(fold.val), set(fold.test)
    if train & test:
        raise ValueError(f"fold {fold.name!r}: {len(train & test)} chip(s) in both train and test")
    if val & test:
        raise ValueError(f"fold {fold.name!r}: {len(val & test)} chip(s) in both val and test")
    if train & val:
        raise ValueError(f"fold {fold.name!r}: {len(train & val)} chip(s) in both train and val")
    if not test:
        raise ValueError(f"fold {fold.name!r}: the held-out source is empty")
