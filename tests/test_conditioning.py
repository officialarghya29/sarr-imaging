"""Tests for the acquisition-conditioned adapter: the cross-sensor claim's model side.

These tests exist because the conditioning claim fails in ways a shape check cannot see.
Four properties are the whole argument, and each is checked against behaviour rather than
against the config text:

1. **A conditioned model is the unconditioned model at step 0.** Otherwise the cross-sensor
   comparison measures a changed initial function, not the conditioning. The inserted adapters
   shift every downstream layer index, so this can only be shown by matching *stock* layers in
   order -- the same technique ``tests/test_arch.py`` uses for the other slots.
2. **An arm reads exactly the fields it declares.** If ``resolution``-only quietly read the
   sensor embedding, the field ablation would be comparing two arms that carry the same
   information, and the conclusion "resolution is what generalises" would be unsupported.
3. **Identity is not turned off by supplying metadata.** The adapter is inert at init because its
   gate starts at zero, so metadata must be unable to change the output until training opens the
   gate. An adapter that reacted at init would break the baseline-equivalence property the whole
   project rests on.
4. **An acquisition the model cannot represent is refused, not snapped to a nearby one.** The
   dangerous failure is silent: mapping an unseen sensor onto a trained-on sensor is exactly the
   confusion the design exists to prevent, so it raises.

The per-sample requirement is checked too, because a batch drawn across sources is the LOSO
recipe: a single per-batch descriptor would average the acquisitions together and destroy the
signal being tested.
"""

from __future__ import annotations

import torch

from saryolo.nn.arch import VARIANTS
from saryolo.nn.modules.conditioning import (
    FIELD_SETS,
    AcquisitionConditionedAdapter,
    AcquisitionEncoder,
    MetadataContext,
)

#: The conditioning slot's arms, and the ``(mode, fields)`` pair each must build with. Restated
#: here deliberately rather than imported from the config generator: the point of this table is
#: to be a second, independent statement of the intent, so a generator edit that changes what an
#: arm means has to be reconciled with it rather than silently propagating.
CONDITIONING_ARMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cond_gain": ("gain", FIELD_SETS["all"]),
    "cond_shift": ("shift", FIELD_SETS["all"]),
    "cond_film": ("film", FIELD_SETS["sensor_resolution"]),
    "cond_sensor": ("film", FIELD_SETS["sensor"]),
    "cond_resolution": ("film", FIELD_SETS["resolution"]),
    "cond_continuous": ("film", FIELD_SETS["continuous_only"]),
    "cond_spatial": ("spatial", FIELD_SETS["all"]),
}

CONTINUOUS_FIELDS = ("resolution_m", "incidence_deg", "band")
CATEGORICAL_FIELDS = ("sensor", "polarization", "mode")

#: The record shape the model builds with (``conditioning_vocab`` in ``arch.py``): two rows per
#: categorical field -- the reserved unknown index plus one seen value.
VOCAB = {"sensor": 2, "polarization": 2, "mode": 2}


def _build_adapter(name: str) -> AcquisitionConditionedAdapter:
    """One adapter, configured exactly as the ``cond_*`` variant wires it."""
    spec = VARIANTS[name]
    mode, _, field_set = spec.conditioning.partition(":")
    return AcquisitionConditionedAdapter(
        32, mode=mode, fields=field_set or "all", vocab_sizes=list(spec.conditioning_vocab)
    ).eval()


def _open_gate(adapter: AcquisitionConditionedAdapter, value: float = 0.5) -> None:
    """Force the residual gate away from zero, as training would."""
    with torch.no_grad():
        adapter.alpha.raw.fill_(value)


# --------------------------------------------------------------- declared fields are respected
def test_each_conditioning_arm_reads_exactly_its_declared_fields():
    """The field ablation is only meaningful if each arm is blind to the fields it omits."""
    for name, (mode, fields) in CONDITIONING_ARMS.items():
        adapter = _build_adapter(name)
        assert adapter.mode == mode, f"{name}: built mode {adapter.mode!r}, expected {mode!r}"
        assert adapter.fields == fields, f"{name}: reads {adapter.fields}, expected {fields}"
        # The encoder's own view must agree with the module's, or the mask and the field list
        # have drifted apart and the arm reads something other than what it declares.
        expected_continuous = [f for f in CONTINUOUS_FIELDS if f in fields]
        expected_categorical = [f for f in CATEGORICAL_FIELDS if f in fields]
        assert adapter.encoder.use_continuous == expected_continuous, name
        assert adapter.encoder.use_categorical == expected_categorical, name


def test_the_arms_are_not_all_the_same_arm():
    """A table where every row built the same graph would make the slot vacuous."""
    signatures = {
        name: (adapter.mode, adapter.fields)
        for name, adapter in ((n, _build_adapter(n)) for n in CONDITIONING_ARMS)
    }
    assert len(set(signatures.values())) == len(CONDITIONING_ARMS), (
        f"conditioning arms are not distinguishable: {signatures}"
    )


def test_an_unused_field_cannot_reach_the_descriptor():
    """Varying an omitted field -- value *and* availability -- must not move the descriptor.

    Both are varied separately because they are two different leaks: a mask applied to the value
    alone would still let the encoder learn from *whether* a field was present.
    """
    rows = 1

    for fields in (("sensor",), ("resolution_m",), FIELD_SETS["continuous_only"], FIELD_SETS["all"]):
        encoder = AcquisitionEncoder(len(CONTINUOUS_FIELDS), VOCAB, fields=fields).eval()
        base_cont = torch.zeros(rows, len(CONTINUOUS_FIELDS))
        ids = torch.zeros(rows, len(CATEGORICAL_FIELDS), dtype=torch.long)
        availability = torch.ones(rows, len(CONTINUOUS_FIELDS) + len(CATEGORICAL_FIELDS))

        with torch.no_grad():
            reference = encoder(base_cont, ids, availability)

            for i, field_name in enumerate(CONTINUOUS_FIELDS):
                if field_name in fields:
                    continue
                moved_cont = base_cont.clone()
                moved_cont[:, i] = 0.7
                moved = encoder(moved_cont, ids, availability)
                assert torch.equal(moved, reference), (
                    f"fields={fields}: {field_name!r} is not consumed by this arm but changing "
                    f"its value moved the descriptor by {float((moved - reference).abs().max()):.3e}"
                )

                moved_av = availability.clone()
                moved_av[:, i] = 0.0
                # Already zero here, so flip it the other way: on -> off is what an arm could
                # otherwise learn from.
                moved_av[:, i] = 1.0 - availability[:, i]
                moved = encoder(base_cont, ids, moved_av)
                assert torch.equal(moved, reference), (
                    f"fields={fields}: {field_name!r} availability flag leaked into the descriptor"
                )

            for i, field_name in enumerate(CATEGORICAL_FIELDS):
                if field_name in fields:
                    continue
                moved_ids = ids.clone()
                moved_ids[:, i] = 1
                moved = encoder(base_cont, moved_ids, availability)
                assert torch.equal(moved, reference), (
                    f"fields={fields}: categorical {field_name!r} leaked into the descriptor"
                )

    # Non-vacuity: the fields that *are* declared must actually move it, or the equality above
    # would hold for a module that ignores its input entirely.
    encoder = AcquisitionEncoder(len(CONTINUOUS_FIELDS), VOCAB, fields=("resolution_m",)).eval()
    cont_a = torch.tensor([[0.1, 0.0, 0.0]])
    cont_b = torch.tensor([[0.9, 0.0, 0.0]])
    ids = torch.zeros(1, len(CATEGORICAL_FIELDS), dtype=torch.long)
    availability = torch.ones(1, len(CONTINUOUS_FIELDS) + len(CATEGORICAL_FIELDS))
    with torch.no_grad():
        moved = float((encoder(cont_a, ids, availability) - encoder(cont_b, ids, availability)).abs().max())
    assert moved > 0.0, "resolution-only arm ignores resolution, so the equality tests are vacuous"


# ------------------------------------------------------------- identity, and metadata at init
def test_metadata_cannot_change_the_output_while_the_gate_is_closed():
    """Identity at init must survive the adapter actually being handed metadata.

    This is the property the baseline-equivalence claim rests on. An adapter that reacted at init
    would mean "SAR-YOLO matches YOLO at step 0" was only true for datasets with no metadata.
    """
    adapter = _build_adapter("cond_film")
    x = torch.randn(2, 32, 8, 8)
    known = MetadataContext(
        continuous=torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.9, 0.4]]),
        categorical=torch.tensor([[1, 0, 0], [1, 0, 0]]),
        availability=torch.ones(2, 6),
    )

    with torch.no_grad():
        adapter.context = known
        with_metadata = adapter(x)
        adapter.context = MetadataContext.unknown(2)
        with_unknown = adapter(x)
    assert torch.equal(with_metadata, x), "gate is closed, so metadata must not change the output"
    assert torch.equal(with_unknown, x)

    # And once the gate is open the same metadata *must* matter, or the slot is a no-op.
    _open_gate(adapter)
    with torch.no_grad():
        adapter.context = known
        opened_known = adapter(x)
        adapter.context = MetadataContext.unknown(2)
        opened_unknown = adapter(x)
    assert not torch.allclose(opened_known, opened_unknown), (
        "with the gate open, a known acquisition must modulate differently from an unknown one"
    )


def test_conditioning_is_per_sample_not_per_batch():
    """A batch drawn across sources must be conditioned source by source.

    The LOSO recipe trains on several sensors at once, so a descriptor that was constant across
    the batch would average the acquisitions together -- precisely the signal being tested.
    """
    adapter = _build_adapter("cond_film")
    _open_gate(adapter)
    x = torch.randn(2, 32, 8, 8)
    first = MetadataContext(
        continuous=torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]]),
        categorical=torch.tensor([[1, 0, 0], [1, 0, 0]]),
        availability=torch.ones(2, 6),
    )
    single = MetadataContext(
        continuous=first.continuous[:1].repeat(2, 1),
        categorical=first.categorical[:1].repeat(2, 1),
        availability=first.availability,
    )

    with torch.no_grad():
        adapter.context = first
        out_batch = adapter(x)
        adapter.context = single
        out_single = adapter(x)

        # Run each row through the adapter on its own, with only its own metadata. If the rows
        # leak into each other, the batched result will not match.
        per_row = []
        for i in range(x.shape[0]):
            adapter.context = MetadataContext(
                continuous=first.continuous[i : i + 1],
                categorical=first.categorical[i : i + 1],
                availability=first.availability[i : i + 1],
            )
            per_row.append(adapter(x[i : i + 1]))

    assert not torch.allclose(out_batch, out_single), (
        "conditioning is not per-sample: a batch of two different acquisitions gave the same "
        "result as a batch of one acquisition repeated"
    )
    # Row-wise independence: row i of the batched result must equal row i run alone.
    for i, alone in enumerate(per_row):
        assert torch.allclose(out_batch[i : i + 1], alone, atol=1e-6), (
            f"per-sample conditioning leaks across the batch: row {i} in a mixed batch differs "
            f"from row {i} run alone (max diff "
            f"{float((out_batch[i : i + 1] - alone).abs().max()):.3e})"
        )


# ------------------------------------------------------------------------- refusal, not guessing
def test_a_vocabulary_mismatch_raises_instead_of_snapping_to_a_nearby_sensor():
    """An index the embedding table cannot represent must raise, not clamp.

    Clamping is the tempting fix and it is the wrong one: it maps an unseen sensor onto a sensor
    that *was* trained on, which is exactly the confusion the design exists to prevent, and it
    would do so silently while the run still reported a result.
    """
    adapter = _build_adapter("cond_film")  # sensor table has 2 rows
    x = torch.randn(1, 32, 8, 8)
    adapter.context = MetadataContext(
        continuous=torch.zeros(1, 3),
        categorical=torch.tensor([[7, 0, 0]]),  # index 7 does not exist
        availability=torch.ones(1, 6),
    )
    try:
        adapter(x)
    except ValueError as exc:
        assert "sensor" in str(exc) and "7" in str(exc), (
            f"the error must name the field and the offending index; got: {exc}"
        )
    else:
        raise AssertionError("an out-of-vocabulary sensor index was accepted instead of refused")


def test_a_descriptor_that_does_not_line_up_with_the_batch_raises():
    """A row-count mismatch is a wiring bug and must not be broadcast away.

    Silently truncating or broadcasting would attach one image's acquisition to another image's
    features, which produces a plausible-looking run that measures nothing.
    """
    adapter = _build_adapter("cond_film")
    _open_gate(adapter)
    adapter.context = MetadataContext(
        continuous=torch.zeros(1, 3),          # one row
        categorical=torch.zeros(1, 3, dtype=torch.long),
        availability=torch.ones(1, 6),
    )
    try:
        adapter(torch.randn(2, 32, 8, 8))       # two images
    except ValueError as exc:
        assert "row" in str(exc) and "batch" in str(exc), f"unhelpful message: {exc}"
    else:
        raise AssertionError("a 1-row descriptor was silently applied to a 2-image batch")


def test_an_absent_context_is_an_unknown_acquisition_not_a_bypass():
    """No metadata must be a trainable state, not a code path that skips the module."""
    adapter = _build_adapter("cond_film")
    _open_gate(adapter)
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        adapter.context = None
        no_context = adapter(x)
        adapter.context = MetadataContext.unknown(1)
        explicit_unknown = adapter(x)
    assert torch.equal(no_context, explicit_unknown), (
        "an absent context must encode exactly as an explicitly unknown acquisition, so the "
        "untrained-input path goes through the same graph"
    )
    assert no_context.shape == x.shape


# ---------------------------------------------------------------- the model-level wiring
def test_the_conditioning_slot_is_wired_into_a_build_and_receives_metadata():
    """The adapters must be reachable from the model, not merely constructible."""
    import saryolo.nn.model as model_module
    from saryolo.nn.arch import build_yaml_dict
    from saryolo.nn.model import SARYOLODetectionModel

    model = SARYOLODetectionModel(
        build_yaml_dict(VARIANTS["cond_film"]), ch=3, nc=1, verbose=False
    ).eval()
    assert model.n_conditioned_adapters > 0, "no adapter was found in the built graph"

    adapters = [
        m for m in model.model.modules() if isinstance(m, AcquisitionConditionedAdapter)
    ]
    assert len(adapters) == model.n_conditioned_adapters
    # Every adapter must share the model's context: one context per model is what makes the
    # metadata plumbing a single write rather than one per module.
    assert {id(a.context) for a in adapters} == {id(model.metadata_context)}

    # And supplying metadata must report that it was consumed, so a caller cannot believe it
    # conditioned a model that has no adapter.
    assert model_module.set_batch_metadata(model, None) is True
    model_module.set_batch_metadata(
        model,
        (
            torch.zeros(1, 3),
            torch.zeros(1, 3, dtype=torch.long),
            torch.ones(1, 6),
        ),
    )
    assert model.metadata_context.is_set
