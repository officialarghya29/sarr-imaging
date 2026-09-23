"""SAR acquisition metadata: the conditioning signal, and its bookkeeping.

Why this exists
---------------
The project's primary claim is that acquisition metadata -- sensor, resolution,
polarisation, band, incidence angle -- can be used to make one detector generalise to
sensors it never trained on. That claim needs two things this repository had neither of:
an acquisition descriptor attached to every image, and an encoding of it that a network
can consume.

Three decisions shape this module, and each one exists because the obvious alternative
fails quietly.

**Missing metadata is not zero metadata.** A field that was never recorded must not encode
as ``0.0`` alongside a field that really is zero: "no incidence angle recorded" and
"0 degrees incidence" are different facts, and a detector that cannot tell them apart will
learn to read absent metadata as a physical statement. Every field therefore encodes as a
value *and* an availability flag, and the two travel together.

**Categorical vocabularies are frozen and built from training sources only.** A learned
embedding table is indexed by integer. If the vocabulary is rebuilt per run, ``sentinel1``
can be index 3 in one run and 5 in the next while the weights still carry the old meaning --
a silent, total corruption of the conditioning. And if the vocabulary is built from a split
that contains the held-out sensor, "unseen sensor" stops meaning anything. So a vocabulary
is declared once, serialised next to the run, and *has a reserved UNKNOWN index*.

**An unseen sensor must be representable, not invisible.** Leave-one-source-out means the
held-out sensor is unseen by construction, so its categorical id has no trained embedding.
Mapping it silently onto an existing sensor would make a generalisation claim from a model
that had, in fact, seen it. Unknown ids are explicit, and the continuous physical
descriptors (resolution, band/wavelength, incidence angle) are what remain informative for
a sensor the model has never met.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FIELDS",
    "CATEGORICAL_FIELDS",
    "CONTINUOUS_FIELDS",
    "BAND_WAVELENGTH_CM",
    "RESOLUTION_RANGE_M",
    "INCIDENCE_RANGE_DEG",
    "UNKNOWN_INDEX",
    "AcquisitionMetadata",
    "Vocabulary",
    "MetadataTable",
    "MetadataRule",
    "build_metadata_table",
    "encode_metadata",
    "load_metadata_sidecar",
]

#: The acquisition fields this project knows about, in a fixed order so anything derived
#: from them is reproducible.
FIELDS: tuple[str, ...] = ("sensor", "resolution_m", "polarization", "mode", "band", "incidence_deg")

#: Fields whose values are names, encoded through a frozen :class:`Vocabulary`.
CATEGORICAL_FIELDS: tuple[str, ...] = ("sensor", "polarization", "mode")

#: Fields whose values are numbers and carry over to an unseen sensor.
#:
#: ``band`` sits here rather than among the categorical fields on purpose: a radar band is a
#: physical wavelength, so a sensor nobody has trained on still contributes a meaningful
#: value. As a category it would only ever be a label for something already seen.
CONTINUOUS_FIELDS: tuple[str, ...] = ("resolution_m", "incidence_deg", "band")

#: Index reserved for "a value this vocabulary has never seen".
UNKNOWN_INDEX: int = 0

#: Radar band letter -> nominal wavelength in centimetres.
#:
#: Nominal centre wavelengths (IEEE bands), not exact satellite figures: the point is a
#: physically ordered descriptor that places an unseen sensor near related ones, and the
#: approximation is documented rather than presented as precision it does not have.
BAND_WAVELENGTH_CM: dict[str, float] = {
    "P": 75.0, "L": 23.0, "S": 10.0, "C": 5.6, "X": 3.1, "KU": 2.2, "K": 1.35, "KA": 0.85,
}

#: Declared normalisation ranges for the continuous fields. Fixed constants, not fitted to a
#: split, so a descriptor means the same thing in every run and for every dataset.
RESOLUTION_RANGE_M: tuple[float, float] = (0.1, 100.0)
INCIDENCE_RANGE_DEG: tuple[float, float] = (0.0, 90.0)


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Acquisition parameters for one image, any of which may be unknown.

    Immutable, because one of these is attached to every sample in a batch and mutated state
    shared across a dataset is a bug waiting for a busy day.
    """

    sensor: str | None = None
    resolution_m: float | None = None
    polarization: str | None = None
    mode: str | None = None
    band: str | None = None
    incidence_deg: float | None = None

    def __post_init__(self) -> None:
        for name in ("resolution_m", "incidence_deg"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.resolution_m is not None and float(self.resolution_m) <= 0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m}")

    @property
    def present(self) -> tuple[str, ...]:
        """Which fields are actually recorded."""
        return tuple(f for f in FIELDS if getattr(self, f) is not None)

    def missing(self, fields: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Which of ``fields`` this sample does not have."""
        return tuple(f for f in fields if getattr(self, f) is None)

    def key(self) -> str:
        """A short stable label, for grouping images and for fold names."""
        bits = []
        for f in FIELDS:
            value = getattr(self, f)
            if value is not None:
                bits.append(f"{f}={value}")
        return "|".join(bits) or "no-metadata"

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in FIELDS}

    @classmethod
    def from_dict(cls, payload: dict) -> AcquisitionMetadata:
        """Build from a mapping, rejecting unknown keys.

        An unknown key is a typo (``resolution`` for ``resolution_m``), and accepting it
        would leave the real field unknown while the config appears to set it -- the kind of
        error that shows up only as a puzzlingly flat ablation.
        """
        unknown = sorted(set(payload) - set(FIELDS))
        if unknown:
            raise KeyError(
                f"unknown acquisition field(s) {unknown}; known fields are {list(FIELDS)}"
            )
        cleaned = {k: (None if v == "" else v) for k, v in payload.items()}
        for name in ("resolution_m", "incidence_deg"):
            if cleaned.get(name) is not None:
                cleaned[name] = float(cleaned[name])
        for name in CATEGORICAL_FIELDS + ("band",):
            if cleaned.get(name) is not None:
                cleaned[name] = str(cleaned[name])
        return cls(**cleaned)


@dataclass
class Vocabulary:
    """A frozen, ordered set of category names with a reserved unknown index.

    Built from *training* sources and then serialised, never rebuilt implicitly: a rebuilt
    vocabulary silently renumbers the embedding table while the trained weights keep their
    old meaning.
    """

    name: str
    entries: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, name: str, values: list[str | None]) -> Vocabulary:
        """Build from observed values, sorted for determinism (not by first appearance)."""
        observed = sorted({str(v) for v in values if v is not None})
        return cls(name=name, entries=observed)

    @property
    def size(self) -> int:
        """Table size, including the reserved unknown slot."""
        return len(self.entries) + 1

    def index(self, value: str | None) -> int:
        """Index for ``value``, or :data:`UNKNOWN_INDEX` if unseen or missing."""
        if value is None:
            return UNKNOWN_INDEX
        try:
            return self.entries.index(str(value)) + 1
        except ValueError:
            return UNKNOWN_INDEX

    def is_known(self, value: str | None) -> bool:
        return value is not None and str(value) in self.entries

    def to_dict(self) -> dict:
        return {"name": self.name, "entries": list(self.entries), "unknown_index": UNKNOWN_INDEX}

    @classmethod
    def from_dict(cls, payload: dict) -> Vocabulary:
        if int(payload.get("unknown_index", UNKNOWN_INDEX)) != UNKNOWN_INDEX:
            raise ValueError(
                f"vocabulary {payload.get('name')!r} declares unknown_index="
                f"{payload.get('unknown_index')}; this encoder reserves {UNKNOWN_INDEX} for "
                f"unseen values and would mis-index a table built elsewhere"
            )
        return cls(name=payload["name"], entries=list(payload["entries"]))


@dataclass
class MetadataTable:
    """Per-image acquisition metadata, plus the encoding vocabularies built with it.

    The table is keyed by image stem so it can be joined to a dataset listing, and it records
    *how* it was produced so a cross-sensor result can be traced to the metadata that defined
    the split.
    """

    entries: dict[str, AcquisitionMetadata] = field(default_factory=dict)
    vocabularies: dict[str, Vocabulary] = field(default_factory=dict)
    source: str = ""

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, stem: str) -> AcquisitionMetadata:
        """Metadata for an image stem, or an all-unknown record.

        Returning an all-unknown record rather than raising is deliberate: at evaluation
        time an image without a metadata row must still be *classifiable* as unknown, and the
        adapter's unknown path is then exercised on real data instead of only in tests.
        """
        return self.entries.get(stem, AcquisitionMetadata())

    def coverage(self) -> dict[str, float]:
        """Fraction of entries carrying each field -- the input to any 'metadata explains it'
        argument, since a field recorded for 3% of images cannot explain anything."""
        if not self.entries:
            return {f: 0.0 for f in FIELDS}
        n = len(self.entries)
        return {f: sum(1 for m in self.entries.values() if getattr(m, f) is not None) / n for f in FIELDS}

    def groups(self, by: tuple[str, ...] | str = ("sensor",)) -> dict[str, list[str]]:
        """Image stems grouped by an acquisition key, e.g. ``by="sensor"``."""
        if isinstance(by, str):
            by = (by,)
        grouped: dict[str, list[str]] = {}
        for stem, meta in sorted(self.entries.items()):
            key = "|".join(f"{f}={getattr(meta, f)}" for f in by)
            grouped.setdefault(key, []).append(stem)
        return grouped

    def build_vocabularies(self, stems: list[str] | None = None) -> dict[str, Vocabulary]:
        """Build vocabularies from the given stems (default: all), sorted for determinism.

        Callers *must* pass the training stems when preparing a leave-one-source-out run:
        vocabularies built over the whole table would give the held-out sensor a trained
        embedding, and the cross-source claim would be measuring a sensor the model had seen.
        """
        stems = sorted(self.entries) if stems is None else list(stems)
        self.vocabularies = {
            name: Vocabulary.build(name, [getattr(self.entries.get(s, AcquisitionMetadata()), name)
                                          for s in stems])
            for name in CATEGORICAL_FIELDS
        }
        return self.vocabularies

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "entries": {stem: meta.to_dict() for stem, meta in sorted(self.entries.items())},
            "vocabularies": {k: v.to_dict() for k, v in sorted(self.vocabularies.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> MetadataTable:
        return cls(
            entries={stem: AcquisitionMetadata.from_dict(m) for stem, m in payload.get("entries", {}).items()},
            vocabularies={k: Vocabulary.from_dict(v) for k, v in payload.get("vocabularies", {}).items()},
            source=payload.get("source", ""),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> MetadataTable:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ------------------------------------------------------------------------------- rules
@dataclass(frozen=True)
class MetadataRule:
    """How each metadata field is derived for an image.

    Stated, never inferred, for the same reason the split rules are: an archive might put the
    sensor in a directory name, encode the resolution in the filename, or carry everything in
    a table. A rule that silently matches nothing would produce a table of all-unknown
    metadata, and the conditioning ablation would then compare arms that are identical --
    reporting "metadata does not help" from a run in which no metadata was ever read.

    Exactly one strategy is used per table:

    * ``directory`` -- fields from the path components below ``root`` (``depth=1`` uses the
      first directory, ``depth=2`` the first two joined by ``/``), mapped through
      ``field_map`` (e.g. ``{0: "sensor"}`` means the first component is the sensor);
    * ``filename`` -- fields from a regex with named groups, e.g. ``(?P<sensor>[A-Za-z0-9]+)_``,
      applied to the file stem;
    * ``sidecar`` -- everything from an explicit table (see :func:`load_metadata_sidecar`).
    """

    kind: str
    root: Path | None = None
    depth: int = 1
    field_map: dict[int | str, str] = field(default_factory=dict)
    pattern: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("directory", "filename", "sidecar"):
            raise ValueError(f"unknown metadata rule {self.kind!r}; expected directory/filename/sidecar")
        if self.kind == "directory":
            if self.root is None:
                raise ValueError("a 'directory' rule needs a root")
            if self.depth < 1:
                raise ValueError(f"depth must be >= 1, got {self.depth}")
            if not self.field_map:
                raise ValueError("a 'directory' rule needs a field_map, e.g. {0: 'sensor'}")
        if self.kind == "filename":
            if not self.pattern:
                raise ValueError("a 'filename' rule needs a pattern with named groups")
            names = set(re.compile(self.pattern).groupindex)
            if not names:
                raise ValueError(
                    f"pattern {self.pattern!r} has no named groups; use (?P<field>...) so the "
                    f"groups can be mapped to acquisition fields"
                )
            unknown = sorted(names - set(FIELDS))
            if unknown:
                raise ValueError(f"pattern names unknown acquisition field(s) {unknown}")

    def describe(self) -> str:
        if self.kind == "directory":
            return f"directory(depth={self.depth}, {self.field_map}) below {self.root}"
        if self.kind == "filename":
            return f"filename({self.pattern!r})"
        return "sidecar"

    def read(self, path: Path) -> dict:
        """Fields derived for one image. Missing pieces are simply absent from the mapping."""
        if self.kind == "directory":
            assert self.root is not None
            try:
                rel = Path(path).resolve().relative_to(Path(self.root).resolve())
            except ValueError:
                return {}
            parts = rel.parts[:-1]
            if len(parts) < self.depth:
                return {}
            component = "/".join(parts[: self.depth])
            return {field: component for _slot, field in self.field_map.items()}
        if self.kind == "filename":
            assert self.pattern is not None
            match = re.search(self.pattern, Path(path).stem)
            return match.groupdict() if match else {}
        return {}


def load_metadata_sidecar(path: str | Path, stem_column: str = "stem") -> dict[str, AcquisitionMetadata]:
    """Read a CSV (or JSON object) of per-image acquisition metadata.

    The CSV route is the honest fallback for archives that encode acquisition nowhere in the
    file layout: someone has to state it, and a table is better than a filename heuristic.

    Args:
        path: ``.csv`` or ``.json``. A JSON object maps stem -> field mapping.
        stem_column: Column holding the image stem.

    Returns:
        Mapping from image stem to :class:`AcquisitionMetadata`.
    """
    import csv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metadata sidecar not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain an object mapping image stem -> fields")
        return {stem: AcquisitionMetadata.from_dict(rec) for stem, rec in payload.items()}

    out: dict[str, AcquisitionMetadata] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        if stem_column not in reader.fieldnames:
            raise ValueError(
                f"{path} has no {stem_column!r} column (found {reader.fieldnames}); every row "
                f"must say which image it describes"
            )
        unknown = sorted(set(reader.fieldnames) - set(FIELDS) - {stem_column})
        if unknown:
            raise ValueError(
                f"{path} has unrecognised column(s) {unknown}; known acquisition fields are "
                f"{list(FIELDS)}. An unrecognised column is a typo, and accepting it would "
                f"leave the real field unknown while the table looks complete."
            )
        for row in reader:
            stem = (row[stem_column] or "").strip()
            if not stem:
                raise ValueError(f"{path} has a row with an empty {stem_column}")
            fields = {k: row[k] for k in FIELDS if row.get(k) not in (None, "")}
            out[Path(stem).stem] = AcquisitionMetadata.from_dict(fields)
    return out


def build_metadata_table(
    images: list[str | Path],
    rule: MetadataRule | None = None,
    sidecar: dict[str, AcquisitionMetadata] | None = None,
) -> MetadataTable:
    """Assemble a table for a list of images, refusing a table that defeats its own purpose.

    Args:
        images: Image paths to describe.
        rule: How to derive fields from a path. May be ``None`` only when ``sidecar`` is given.
        sidecar: Explicit per-stem metadata, taking precedence over ``rule``.

    Raises:
        ValueError: If no field was derived for any image, and if no rule or sidecar was
            supplied at all. An all-unknown table is not a neutral starting point: it turns a
            conditioning ablation into a comparison of identical models, so "sensor
            conditioning does not help" could be concluded from a run that never read a
            sensor. A table with only *some* fields is legitimate -- single-field arms are
            exactly the ablation the brief asks for -- so coverage is reported by
            :meth:`MetadataTable.coverage` rather than enforced here.
    """
    if rule is None and not sidecar:
        raise ValueError("provide a metadata rule or a sidecar; guessing acquisition values is not supported")

    entries: dict[str, AcquisitionMetadata] = {}
    for image in images:
        path = Path(image)
        fields: dict = {}
        if sidecar and path.stem in sidecar:
            fields = sidecar[path.stem].to_dict()
        elif rule is not None and rule.kind != "sidecar":
            fields = rule.read(path)
        entries[path.stem] = AcquisitionMetadata.from_dict(fields)

    table = MetadataTable(entries=entries, source=(rule.describe() if rule else "sidecar"))
    coverage = table.coverage()

    if not any(c > 0 for c in coverage.values()):
        raise ValueError(
            f"no acquisition field could be derived for any of the {len(entries)} image(s) "
            f"using {table.source}. A table of unknowns would make every conditioning arm "
            f"identical while the experiment still reports a result."
        )
    return table


def encode_metadata(
    meta: AcquisitionMetadata,
    vocabularies: dict[str, Vocabulary] | None = None,
) -> tuple[list[float], list[int], list[float]]:
    """Encode one sample as ``(continuous, categorical ids, availability)``.

    The three vectors are returned separately because they are consumed differently: the
    continuous values pass through an MLP, the ids index embedding tables, and the
    availability flags gate both so that *absent* never reads as *zero*.

    Scaling is fixed and documented rather than fitted: resolution is ``log10`` between the
    declared range, band is ``log10`` wavelength in cm, incidence is degrees over ninety. A
    fitted normaliser would mean the same metadata encodes differently in different runs.
    """
    vocabularies = vocabularies or {}

    continuous: list[float] = []
    for f in CONTINUOUS_FIELDS:
        value = getattr(meta, f)
        if value is None:
            continuous.append(0.0)
            continue
        if f == "resolution_m":
            lo, hi = RESOLUTION_RANGE_M
            continuous.append((math.log10(float(value)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
        elif f == "incidence_deg":
            lo, hi = INCIDENCE_RANGE_DEG
            continuous.append((float(value) - lo) / (hi - lo))
        else:  # band -> log10 nominal wavelength in cm, a physical ordering
            wavelength = BAND_WAVELENGTH_CM.get(str(value).upper())
            continuous.append(0.0 if wavelength is None else math.log10(wavelength) / math.log10(100.0))

    categorical: list[int] = []
    for f in CATEGORICAL_FIELDS:
        vocab = vocabularies.get(f)
        value = getattr(meta, f)
        categorical.append(vocab.index(value) if vocab is not None else UNKNOWN_INDEX)

    availability: list[float] = [
        1.0 if getattr(meta, f) is not None else 0.0 for f in CONTINUOUS_FIELDS + CATEGORICAL_FIELDS
    ]
    return continuous, categorical, availability
