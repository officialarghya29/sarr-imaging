"""Tests for the missing-metadata degradation study (metadata-field masking).

The study's claim -- "the model degrades gracefully as acquisition fields are
withheld" -- is only measurable if a withheld field is *encoded exactly like a
field that was never recorded*. The tests here pin that equivalence, the mask's
refusals, and the two integration points where the mask must survive (dataset
construction and fold generation): a mask applied when the training dataset is
built but dropped when the fold configs are written would let one fold run under
a different protocol than its neighbour, and the per-fold comparison would mix
protocols without anything saying so.
"""

from __future__ import annotations

import pytest
import torch

from saryolo.data.field_mask import (
    apply_metadata_mask,
    metadata_field_mask,
    resolve_metadata_fields,
)
from saryolo.data.metadata import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    AcquisitionMetadata,
    MetadataTable,
    Vocabulary,
    encode_metadata,
)

FIELD_ORDER = CONTINUOUS_FIELDS + CATEGORICAL_FIELDS


# ---------------------------------------------------------------------- spec resolution
def test_none_and_empty_mean_no_masking():
    assert resolve_metadata_fields(None) is None
    assert resolve_metadata_fields("") is None
    assert resolve_metadata_fields([]) is None
    assert metadata_field_mask(None) is None


def test_a_field_list_resolves_to_itself():
    assert resolve_metadata_fields(["sensor", "resolution_m"]) == ("sensor", "resolution_m")
    assert resolve_metadata_fields("sensor,resolution_m") == ("sensor", "resolution_m")


def test_an_unknown_field_is_refused_not_ignored():
    """A typo must not silently run the full-metadata arm under a restricted name."""
    with pytest.raises(ValueError, match="unknown acquisition field"):
        resolve_metadata_fields(["sensor", "resoluton_m"])  # typo'd


# ---------------------------------------------------------------------- mask equivalence
def _encoded_with_field(sensor="s1", resolution=10.0, band="C"):
    meta = AcquisitionMetadata(sensor=sensor, resolution_m=resolution, band=band)
    vocab = {"sensor": Vocabulary.build("sensor", ["s1", "g3"])}
    return encode_metadata(meta, vocab), vocab


def test_masked_field_encodes_like_a_never_recorded_field():
    """The core property: withholding must be indistinguishable from absence.

    If a masked resolution read as anything other than an absent one, the degradation study
    would feed the model a fabricated value (for instance the bottom of the resolution range)
    and call the resulting degradation a deployment reality.
    """
    (cont_present, cat_present, avail_present), _ = _encoded_with_field()
    (cont_absent, cat_absent, avail_absent), _ = _encoded_with_field(resolution=None)

    mask = metadata_field_mask(tuple(f for f in FIELD_ORDER if f != "resolution_m"))
    descriptor_present = {
        "continuous": torch.tensor(cont_present),
        "categorical": torch.tensor(cat_present),
        "availability": torch.tensor(avail_present),
    }
    masked = apply_metadata_mask(
        {
            "continuous": descriptor_present["continuous"].clone(),
            "categorical": descriptor_present["categorical"].clone(),
            "availability": descriptor_present["availability"].clone(),
        },
        mask,
    )

    assert torch.equal(masked["continuous"], torch.tensor(cont_absent))
    assert torch.equal(masked["availability"], torch.tensor(avail_absent))
    # The categorical fields were untouched by this mask and must agree too.
    assert torch.equal(masked["categorical"], torch.tensor(cat_absent))


def test_a_masked_categorical_field_lands_on_the_unknown_row():
    """A withheld sensor must index the reserved never-seen row, not a trained one."""
    (cont, cat, avail), _ = _encoded_with_field()
    i = FIELD_ORDER.index("sensor")

    masked = apply_metadata_mask(
        {
            "continuous": torch.tensor(cont).clone(),
            "categorical": torch.tensor(cat).clone(),
            "availability": torch.tensor(avail).clone(),
        },
        metadata_field_mask(tuple(f for f in FIELD_ORDER if f != "sensor")),
    )

    assert int(masked["categorical"][CATEGORICAL_FIELDS.index("sensor")]) == 0
    assert float(masked["availability"][i]) == 0.0


def test_the_mask_leaves_kept_fields_untouched():
    (cont, cat, avail), _ = _encoded_with_field()
    original = (cont.copy(), cat.copy(), avail.copy())

    masked = apply_metadata_mask(
        {
            "continuous": torch.tensor(cont).clone(),
            "categorical": torch.tensor(cat).clone(),
            "availability": torch.tensor(avail).clone(),
        },
        metadata_field_mask(tuple(FIELD_ORDER)),  # keep everything
    )

    assert torch.equal(masked["continuous"], torch.tensor(original[0]))
    assert torch.equal(masked["categorical"], torch.tensor(original[1]))
    assert torch.equal(masked["availability"], torch.tensor(original[2]))


def test_batched_descriptors_mask_per_field_not_per_row():
    """The mask is per *field*: every row of a batch loses the same field."""
    (cont, cat, avail), _ = _encoded_with_field()
    descriptor = {
        "continuous": torch.tensor(cont).repeat(4, 1),
        "categorical": torch.tensor(cat).repeat(4, 1),
        "availability": torch.tensor(avail).repeat(4, 1),
    }
    i = FIELD_ORDER.index("band")

    masked = apply_metadata_mask(
        descriptor, metadata_field_mask(tuple(f for f in FIELD_ORDER if f != "band"))
    )

    assert torch.all(masked["continuous"][:, i] == 0.0)
    assert torch.all(masked["availability"][:, i] == 0.0)


# ---------------------------------------------------------------------- trainer integration
def test_data_config_field_mask_reads_and_validates_the_spec():
    from saryolo.training.trainer import data_config_field_mask

    assert data_config_field_mask({}) is None
    assert data_config_field_mask({"metadata_fields": ["sensor"]}) is not None
    with pytest.raises(ValueError, match="unknown acquisition field"):
        data_config_field_mask({"metadata_fields": ["nope"]})


def test_the_descriptor_path_applies_the_mask_end_to_end():
    """The exact function the datasets call: a masked table row reaches the batch masked."""
    from saryolo.training.trainer import _metadata_descriptor

    _meta = AcquisitionMetadata(sensor="s1", resolution_m=10.0, band="C")
    table = MetadataTable(entries={"img": _meta}, source="test:mask")
    vocab = {"sensor": Vocabulary.build("sensor", ["s1"]), "polarization": Vocabulary.build("polarization", []),
             "mode": Vocabulary.build("mode", [])}
    mask = metadata_field_mask(("sensor",))  # deliver sensor only

    full = _metadata_descriptor(table, vocab, "img")
    masked = _metadata_descriptor(table, vocab, "img", field_mask=mask)

    assert not torch.equal(full["continuous"], masked["continuous"])
    assert float(masked["availability"][FIELD_ORDER.index("resolution_m")]) == 0.0
    # The sensor itself survived.
    assert float(masked["availability"][FIELD_ORDER.index("sensor")]) == 1.0
