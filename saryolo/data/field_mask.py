"""Metadata-field ablation: which acquisition fields does the adapter actually need?

The master workflow's ablation matrix (sensor-only, resolution-only, polarization-only, ...)
is served two ways. At the *graph* level an arm simply declares a smaller field set on the
adapter (``fields="sensor"`` and so on). But a second, equally important study lives at the
*data* level: how does a **fully-wired** adapter degrade as the acquisition record itself
loses fields -- the deployment question "we recorded no resolution for this archive; does the
detector still work, or does it collapse?"

That question cannot be answered by the graph-level arms, because an arm built to read only
``sensor`` never sees a resolution at all: there is nothing to degrade. The data-level study
runs one trained model and feeds it records whose fields have been deliberately withheld.

Withholding is implemented as masking at encode time, and the masked record must be
indistinguishable from a record that was never there. That is why masking zeroes *both* the
value and its availability flag: the availability flag exists precisely so that "absent"
never reads as "zero", so a mask that only zeroed the value would feed the model a plausible
lie (resolution = 0.1 m, the bottom of the declared range) instead of an honest absence.
"""

from __future__ import annotations

from typing import Any

from saryolo.data.metadata import CATEGORICAL_FIELDS, CONTINUOUS_FIELDS

__all__ = ["metadata_field_mask", "resolve_metadata_fields"]

#: Position of each field in the encoded vectors, so a mask can be applied without hard-coding
#: the order in three places.
FIELD_ORDER: tuple[str, ...] = CONTINUOUS_FIELDS + CATEGORICAL_FIELDS


def resolve_metadata_fields(spec: Any) -> tuple[str, ...] | None:
    """Validate a ``metadata_fields`` spec from a data config.

    Accepts ``None``/empty (no masking -- every field the table carries is delivered), a list
    of field names (exactly those fields are delivered), or the named sets used by the
    ablation arms. Unknown names are refused rather than ignored: a typo here would silently
    run the full-metadata arm while the ledger said "sensor only", and the resulting
    comparison would be between two identical runs.

    Returns ``None`` when nothing should be masked.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = spec.strip()
        if not spec:
            return None
        spec = [part.strip() for part in spec.split(",") if part.strip()]
    if isinstance(spec, (list, tuple)):
        if not spec:
            return None
        fields = [str(f) for f in spec]
    else:
        raise TypeError(f"metadata_fields must be a list of field names or a named set, got {spec!r}")

    unknown = sorted(set(fields) - set(FIELD_ORDER))
    if unknown:
        raise ValueError(
            f"metadata_fields names unknown acquisition field(s) {unknown}; known fields are "
            f"{list(FIELD_ORDER)}. A typo here would silently run the full-metadata arm."
        )
    return tuple(fields)


def metadata_field_mask(kept_fields: tuple[str, ...] | None) -> list[float] | None:
    """An availability-style mask: 1.0 for a delivered field, 0.0 for a withheld one.

    ``None`` means no masking (all ones) and is returned as ``None`` rather than a list so the
    hot path can skip the multiply entirely for runs that do not ablate fields.
    """
    if kept_fields is None:
        return None
    kept = set(kept_fields)
    return [1.0 if f in kept else 0.0 for f in FIELD_ORDER]


def apply_metadata_mask(
    descriptor: dict[str, Any],
    mask: list[float] | None,
) -> dict[str, Any]:
    """Mask one encoded descriptor in place-shaped tensors.

    The mask zeroes the *availability* flags and the values together. For a categorical field,
    the id is also forced to the reserved unknown index, so the embedding lookup reads the same
    "never seen this" row a genuinely unseen value would produce -- the same convention the
    adapter applies to fields outside its declared set.
    """
    if mask is None:
        return descriptor
    n_continuous = len(CONTINUOUS_FIELDS)
    for i, keep in enumerate(mask):
        if keep:
            continue
        if i < n_continuous:
            descriptor["continuous"][..., i] = 0.0
            descriptor["availability"][..., i] = 0.0
        else:
            j = i - n_continuous
            descriptor["categorical"][..., j] = 0  # UNKNOWN_INDEX: the reserved never-seen row
            descriptor["availability"][..., i] = 0.0
    return descriptor
