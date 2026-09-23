"""Acquisition-conditioned adapter: the model side of the cross-sensor claim.

The problem this addresses
--------------------------
One detector is trained across several SAR sources and then asked to work on a source it has
never seen. The failure is not that the network lacks capacity; it is that features learned on
one acquisition distribution are being reused as though acquisitions did not differ. The
hypothesis this module tests is narrow and falsifiable: *conditioning feature modulation on the
acquisition parameters recovers part of that loss, on sources that were never trained on.*

That last clause decides the design, and it rules out the obvious shortcut. The fastest way to
condition on a sensor is a learned embedding table indexed by sensor id -- and it cannot work
here, because the held-out sensor has no row in that table. Leave-one-source-out means the
categorical path is *unavailable* exactly where the claim is tested. So conditioning runs on
two channels:

* **categorical** -- learned embeddings for sensors already seen, which let the model specialise
  its response to a known sensor's statistics;
* **continuous** -- physically ordered descriptors (resolution, nominal band wavelength,
  incidence angle) that exist for a sensor nobody has ever trained on.

Both feed one encoder, and the ablation arms exist to find out which of them actually carries
the generalisation. If only the categorical channel matters, the claim fails on its own terms,
because it cannot transfer to an unseen sensor by construction.

Two properties are pinned by test rather than asserted
-----------------------------------------------------
**Identity at initialisation.** A fresh conditioned model must be numerically the unconditioned
one. The modulation is gated, the gate starts at zero, and the *rest* of the path starts
non-zero -- the failure mode this avoids is a zero-initialised modulation whose gradient
vanishes identically, which leaves a module that can never open. With the gate at zero the
gradient reaches the gate itself, so the adapter can learn to open; identity is then bought by
the gate rather than by freezing the module.

**Unused fields carry no information.** The ablation compares sensor-only against
resolution-only, so an arm that merely ignores a field at the *loss* level while still reading it
would be measuring nothing. For a field an arm does not consume, both its value and its
availability flag are zeroed, which makes the two arms genuinely information-free with respect
to that field.

Why the metadata is per-sample
------------------------------
A training batch mixes sources: the leave-one-source-out recipe trains on three sensors at once,
and a batch is drawn across them. Conditioning therefore has to vary *within* a batch -- a single
per-batch descriptor would average the acquisitions together and destroy the signal being tested.
The encoder is applied per sample and the modulation is broadcast over space, which is what keeps
the cost near zero.

How metadata reaches the modules
--------------------------------
Ultralytics resolves a custom layer with one input (``parse_model`` computes ``c2 = ch[f]``), so an
adapter cannot be handed the metadata as a second graph input. Instead every adapter shares one
:class:`MetadataContext` owned by the model, and the model fills it from ``batch["metadata"]``
before the forward pass. The wiring happens once, at model construction, so a newly added adapter
is picked up by the same walk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ._common import ZeroGate, resolve_c1

__all__ = [
    "MetadataContext",
    "AcquisitionEncoder",
    "AcquisitionConditionedAdapter",
    "MODES",
    "FIELD_SETS",
    "parse_vocab_sizes",
]

#: Adapter designs, from the brief's adapter ablation. Kept deliberately small: the point is to
#: find the *simplest* one that works, so a more elaborate design has to justify itself against
#: these and not merely against a plain detector.
#:
#: ``gain`` is deliberately not called ``scale``. Ultralytics' ``parse_model`` resolves a *string*
#: argument as ``locals()[a] if a in locals() else ast.literal_eval(a)``, and ``scale`` is one of
#: its local variables -- so a mode named ``"scale"`` arrives at the module as the model scale
#: (``"s"``) and the module raises "mode must be one of ...", which reads like a config typo rather
#: than an argument-shadowing trap. Every mode and field-set name must therefore avoid
#: ``parse_model``'s locals; ``tests/test_conditioning.py`` pins this by inspecting what the built
#: modules actually received rather than trusting the YAML text.
MODES: tuple[str, ...] = ("gain", "shift", "film", "spatial")

#: Which acquisition fields each arm consumes, for the metadata ablation matrix.
#:
#: ``unseen_continuous`` is the arm that matters most: it uses only the physically-ordered
#: fields, so it is the one that can transfer to a sensor with no embedding row at all.
FIELD_SETS: dict[str, tuple[str, ...]] = {
    "sensor": ("sensor",),
    "resolution": ("resolution_m",),
    "sensor_resolution": ("sensor", "resolution_m"),
    "continuous_only": ("resolution_m", "band", "incidence_deg"),
    "all": ("sensor", "resolution_m", "polarization", "mode", "band", "incidence_deg"),
}


@dataclass
class MetadataContext:
    """Per-sample acquisition descriptors, shared by every adapter in one model.

    Holds the three encoded vectors produced by :func:`saryolo.data.metadata.encode_metadata`
    for a batch. ``None`` means "no metadata was supplied", which is a legitimate state: with no
    metadata the adapter is an exact identity, so a model carrying the adapter still runs on a
    dataset that has none.
    """

    continuous: torch.Tensor | None = None
    categorical: torch.Tensor | None = None
    availability: torch.Tensor | None = None

    def set(
        self,
        continuous: torch.Tensor | None,
        categorical: torch.Tensor | None = None,
        availability: torch.Tensor | None = None,
    ) -> None:
        self.continuous = continuous
        self.categorical = categorical
        self.availability = availability

    @classmethod
    def unknown(cls, batch: int, continuous_dim: int = 3, categorical_dim: int = 3) -> MetadataContext:
        """An explicitly *unknown* acquisition for ``batch`` samples: zeros, availability off.

        This is what an absent context means too. "No metadata was supplied" is a legitimate
        input state -- a deployment with no acquisition table, or an image whose parameters were
        never recorded -- and it is modelled as one: all-zero values with every availability flag
        off, which is exactly the encoding :func:`saryolo.data.metadata.encode_metadata` produces
        for a record with no fields set. Making it a separate code path that bypasses the module
        would leave that state untrained and would also make the module's branch degenerate.
        """
        return cls(
            continuous=torch.zeros(batch, continuous_dim),
            categorical=torch.zeros(batch, categorical_dim, dtype=torch.long),
            availability=torch.zeros(batch, continuous_dim + categorical_dim),
        )

    def clear(self) -> None:
        self.continuous = self.categorical = self.availability = None

    @property
    def is_set(self) -> bool:
        return self.continuous is not None

    def to(self, device: torch.device | str) -> MetadataContext:
        return MetadataContext(
            continuous=None if self.continuous is None else self.continuous.to(device),
            categorical=None if self.categorical is None else self.categorical.to(device),
            availability=None if self.availability is None else self.availability.to(device),
        )


def parse_vocab_sizes(
    spec: dict[str, int] | str | list | tuple | None,
    categorical_fields: tuple[str, ...] = ("sensor", "polarization", "mode"),
) -> dict[str, int]:
    """Accept vocab sizes as a mapping, a ``"sensor=5,mode=3"`` string, or a positional list.

    The positional-list form is what a model YAML uses, and it is not a style choice. A YAML row
    reaches its module through ``parse_model``, which runs ``ast.literal_eval`` on every *string*
    argument and suppresses only ``ValueError``. A string it cannot compile raises ``SyntaxError``
    straight through and the model fails to build -- verified: ``"sensor=5,polarization=3"``
    crashes with a bare ``SyntaxError: invalid syntax`` that says nothing about vocabularies.
    Integers carry no such risk, and the field order is already fixed by
    :data:`saryolo.data.metadata.CATEGORICAL_FIELDS`, so a list of ints is unambiguous.

    Returns:
        Mapping from categorical field to table size (including the reserved unknown index).
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        return {str(k): int(v) for k, v in spec.items()}
    if isinstance(spec, (list, tuple)):
        if len(spec) != len(categorical_fields):
            raise ValueError(
                f"positional vocab sizes have {len(spec)} entries but {len(categorical_fields)} "
                f"categorical fields exist, in the order {list(categorical_fields)}"
            )
        return {name: int(size) for name, size in zip(categorical_fields, spec, strict=True)}

    sizes: dict[str, int] = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"vocab size {item!r} is not in 'field=size' form (expected e.g. "
                f"'sensor=5,polarization=3,mode=2')"
            )
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in categorical_fields:
            raise ValueError(
                f"vocab size names unknown categorical field {name!r}; known fields are "
                f"{', '.join(categorical_fields)}"
            )
        try:
            sizes[name] = int(value)
        except ValueError as exc:
            raise ValueError(f"vocab size for {name!r} is not an integer: {value!r}") from exc
    return sizes


class AcquisitionEncoder(nn.Module):
    """Encode ``(continuous, categorical, availability)`` into one descriptor per sample.

    Args:
        continuous_dim: Width of the continuous vector (one per continuous field).
        vocab_sizes: Table size per categorical field, *including* the reserved unknown index.
        fields: Which acquisition fields this arm consumes. Anything outside this set has both
            its value and its availability flag zeroed, so the arm is information-free with
            respect to it.
        embed_dim: Per-field categorical embedding width.
        hidden: Hidden width of the continuous MLP.
        out_dim: Descriptor width handed to the modulator.
    """

    def __init__(
        self,
        continuous_dim: int,
        vocab_sizes: dict[str, int] | str,
        fields: tuple[str, ...],
        continuous_fields: tuple[str, ...] = ("resolution_m", "incidence_deg", "band"),
        categorical_fields: tuple[str, ...] = ("sensor", "polarization", "mode"),
        embed_dim: int = 8,
        hidden: int = 32,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        vocab_sizes = parse_vocab_sizes(vocab_sizes)
        self.continuous_fields = tuple(continuous_fields)
        self.categorical_fields = tuple(categorical_fields)
        self.fields = tuple(fields)

        unknown = sorted(set(self.fields) - set(self.continuous_fields) - set(self.categorical_fields))
        if unknown:
            raise ValueError(f"unknown acquisition field(s) {unknown} in fields={self.fields!r}")

        self.use_continuous = [f for f in self.continuous_fields if f in self.fields]
        self.use_categorical = [f for f in self.categorical_fields if f in self.fields]

        # Boolean masks are buffers, not parameters, so they follow the module between devices and
        # cannot drift from the field lists above.
        self.register_buffer(
            "_cont_mask",
            torch.tensor([1.0 if f in self.fields else 0.0 for f in self.continuous_fields]),
            persistent=False,
        )
        self.register_buffer(
            "_cat_mask",
            torch.tensor([1.0 if f in self.fields else 0.0 for f in self.categorical_fields]),
            persistent=False,
        )

        self.embeddings = nn.ModuleDict()
        for field_name in self.use_categorical:
            size = int(vocab_sizes.get(field_name, 2))
            if size < 2:
                raise ValueError(
                    f"vocabulary for {field_name!r} has size {size}; it must be at least 2 so the "
                    f"reserved unknown index has somewhere to point"
                )
            # Index zero is the explicit UNKNOWN/missing value. A held-out sensor has no
            # training embedding row; keep this row exactly neutral instead of injecting
            # a random, never-trained acquisition vector at evaluation time.
            self.embeddings[field_name] = nn.Embedding(size, embed_dim, padding_idx=0)

        cont_in = len(self.use_continuous) * 2  # value + availability, never value alone
        cat_out = len(self.use_categorical) * embed_dim
        self.continuous_mlp = nn.Sequential(nn.Linear(max(cont_in, 1), hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.fuse = nn.Sequential(nn.Linear(max(hidden + cat_out, 1), hidden), nn.SiLU(), nn.Linear(hidden, out_dim))

    @property
    def embedding_size(self) -> int:
        total = 0
        for field_name in self.use_categorical:
            total += self.embeddings[field_name].num_embeddings
        return total

    def _continuous_input(self, continuous: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        """Selected, masked continuous fields, each paired with its availability flag."""
        value_mask = self._cont_mask.to(continuous.device)
        pieces = []
        for i, field_name in enumerate(self.continuous_fields):
            active = 1.0 if field_name in self.fields else 0.0
            if not active:
                continue
            piece = continuous[:, i] * value_mask[i]
            if availability is not None:
                # The availability flag is zeroed for an unused field too: leaving it on would let
                # the encoder learn from *whether* a field was present even while ignoring its value.
                available = availability[:, i] * value_mask[i]
                piece = torch.stack([piece, available], dim=-1)
            else:
                piece = torch.stack([piece, torch.ones_like(piece)], dim=-1)
            pieces.append(piece)
        if not pieces:
            # No continuous field is consumed by this arm (the sensor-only arm is exactly that
            # case). A single constant column is fed rather than an empty one so the first linear
            # has something to consume: with zero columns the matmul raised a shape error, and the
            # arm is perfectly well defined -- all of its information arrives through the
            # categorical embeddings, and this branch contributes only its bias.
            return continuous.new_zeros((continuous.shape[0], 1))
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor | None = None,
        availability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Descriptor of shape ``(B, out_dim)``."""
        cat_parts = []
        if categorical is not None:
            cat_mask = self._cat_mask.to(categorical.device)
            for i, field_name in enumerate(self.categorical_fields):
                if field_name not in self.fields:
                    continue
                ids = categorical[:, i]
                # An unused field is forced to the unknown index rather than merely skipped, so
                # the arm reads the same "no information" row it reads for a genuinely unseen
                # value and cannot use the difference between them.
                ids = torch.where(cat_mask[i].bool(), ids, torch.zeros_like(ids))
                table = self.embeddings[field_name].num_embeddings
                high = int(ids.max()) if ids.numel() else 0
                if high >= table:
                    # Raised here, not clamped, and named. The alternative -- clamping to a valid
                    # index -- would map an unseen sensor onto a sensor that was trained on, which
                    # is precisely the confusion this whole design exists to prevent. Torch's own
                    # error here is an opaque "index out of range in self".
                    raise ValueError(
                        f"{field_name!r} encodes index {high} but the embedding table has "
                        f"{table} rows. The vocabulary used to encode this batch does not match "
                        f"the one the model was built with; rebuild the model with vocab_sizes "
                        f"taken from the same metadata table that encoded the data."
                    )
                cat_parts.append(self.embeddings[field_name](ids))
        cat_vec = torch.cat(cat_parts, dim=-1) if cat_parts else None

        cont_vec = self.continuous_mlp(self._continuous_input(continuous, availability))

        fused_in = cont_vec if cat_vec is None else torch.cat([cont_vec, cat_vec], dim=-1)
        return self.fuse(fused_in)


class AcquisitionConditionedAdapter(nn.Module):
    """Modulate features per sample using acquisition metadata.

    Contract: **exact identity at initialisation**, like every other module here -- identity is
    bought by the zero-initialised gate and by nothing else. An *absent* context is treated as an
    explicitly unknown acquisition (:meth:`MetadataContext.unknown`) rather than as a bypass, so
    the untrained-input path is the same graph and stays trainable.

    Args:
        c1: Feature channels, or the parser's ``ch`` list.
        source: Index into ``ch`` this module consumes. Required whenever the YAML row's ``from``
            is an explicit index, because ``ch[-1]`` is then a different layer.
        mode: One of :data:`MODES`.
        fields: Acquisition fields to consume (a key or value of :data:`FIELD_SETS`).
        vocab_sizes: Categorical table sizes, including the unknown index.
        out_dim: Encoder descriptor width.
        spatial_rank: Number of spatial basis maps for ``mode="spatial"``.
    """

    #: Exposed as class attributes because every other module in this package declares its
    #: vocabulary that way, and the test-suite registries discover modes through the class.
    #: Without these, ``AcquisitionConditionedAdapter`` was the one module whose modes were
    #: invisible to the ``parse_model`` safety check and to the identity/frozen-gate sweeps --
    #: the two guards that would have caught a mode name being silently substituted by
    #: ``parse_model``'s local variables, which is the exact trap the module docstring warns
    #: about. The field sets are exposed for the same reason: they are a second string
    #: vocabulary on the same YAML row, and a mangled field-set name would collapse two
    #: ablation arms onto one model while the table still listed them as separate rows.
    MODES = MODES
    FIELD_SETS = FIELD_SETS

    def __init__(
        self,
        c1,
        source: int | None = None,
        mode: str = "film",
        fields: tuple[str, ...] | str = "all",
        vocab_sizes: dict[str, int] | str | None = None,
        continuous_fields: tuple[str, ...] = ("resolution_m", "incidence_deg", "band"),
        categorical_fields: tuple[str, ...] = ("sensor", "polarization", "mode"),
        out_dim: int = 32,
        spatial_rank: int = 4,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        vocab_sizes = parse_vocab_sizes(vocab_sizes)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        if isinstance(fields, str):
            fields = FIELD_SETS.get(fields)
            if fields is None:
                raise ValueError(f"unknown field set {fields!r}; known: {sorted(FIELD_SETS)}")
        self.c1 = resolve_c1(c1, source)
        self.mode = mode
        self.fields = tuple(fields)
        self.spatial_rank = int(spatial_rank)

        self.encoder = AcquisitionEncoder(
            continuous_dim=len(continuous_fields),
            vocab_sizes=dict(vocab_sizes or {}),
            fields=self.fields,
            continuous_fields=continuous_fields,
            categorical_fields=categorical_fields,
            out_dim=out_dim,
        )

        # Per-sample modulation heads, per channel. What makes the modes differ is *which* heads
        # exist: gain only (`gain`), offset only (`shift`), both (`film`), and both with a
        # position-dependent factor (`spatial`). The ablation is about expressivity, so a mode
        # must not carry a head its definition excludes.
        self.to_gamma = nn.Linear(out_dim, self.c1) if mode in ("gain", "film", "spatial") else None
        self.to_beta = nn.Linear(out_dim, self.c1) if mode in ("shift", "film", "spatial") else None
        if mode == "spatial":
            self.to_spatial = nn.Linear(out_dim, self.spatial_rank)
            self.spatial_basis = nn.Parameter(torch.randn(self.spatial_rank, 8, 8) * 0.02)

        # The gate is the identity mechanism -- the same ``ZeroGate`` every other module uses, so
        # the shared identity and frozen-parameter tests apply to this module unchanged.
        self.alpha = ZeroGate(alpha_init)
        self.context: MetadataContext | None = None

        # The modulation path starts NON-zero, biases included. With a zero-initialised branch on
        # top of a zero-initialised gate, ``dL/dalpha = <dL/dout, gamma*x + beta>`` is identically
        # zero and the adapter can never leave identity while looking perfectly healthy. Small
        # non-zero values keep the gate's gradient alive; identity still comes from the gate alone.
        for head in (self.to_gamma, self.to_beta):
            if head is not None:
                nn.init.normal_(head.weight, std=0.02)
                nn.init.normal_(head.bias, std=0.02)
        if mode == "spatial":
            nn.init.normal_(self.to_spatial.weight, std=0.02)
            nn.init.normal_(self.to_spatial.bias, std=0.02)

    # ------------------------------------------------------------------ forward
    def _context_for(self, batch: int, device: torch.device) -> MetadataContext:
        """The context to condition on, falling back to an explicitly unknown acquisition.

        The fallback exists so that "this image's acquisition was never recorded" is a state the
        model is trained on rather than a state that bypasses the module: an absent context is
        encoded exactly as a record with no fields set.
        """
        ctx = self.context
        if ctx is None or not ctx.is_set or ctx.continuous is None:
            return MetadataContext.unknown(
                batch,
                continuous_dim=len(self.encoder.continuous_fields),
                categorical_dim=len(self.encoder.categorical_fields),
            ).to(device)
        return ctx

    def _descriptor(self, batch: int, device: torch.device) -> torch.Tensor:
        """Encoder output for the current batch."""
        ctx = self._context_for(batch, device)
        rows = ctx.continuous.shape[0]
        if rows != batch:
            # Not a data property but a wiring bug, so it is raised rather than papered over: a
            # mismatch means the descriptor rows no longer line up with the images, and a silent
            # truncation or broadcast would attach one image's acquisition to another's features.
            # In normal operation this cannot happen, because the model sets the context from
            # ``batch["metadata"]`` -- the same batch it is about to run.
            raise ValueError(
                f"acquisition metadata has {rows} row(s) but the feature batch has {batch}; "
                f"the descriptor must be per-sample and aligned with the batch"
            )
        continuous = ctx.continuous.to(device)
        categorical = None if ctx.categorical is None else ctx.categorical.to(device)
        availability = None if ctx.availability is None else ctx.availability.to(device)
        return self.encoder(continuous, categorical, availability)

    @property
    def gate(self) -> torch.Tensor:
        """Current gate value, as a differentiable scalar (``tanh`` of the raw parameter)."""
        return self.alpha()

    def _spatial_map(self, descriptor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Low-rank position-dependent factor of shape ``(B, 1, H, W)``.

        The descriptor picks weights over a small learned basis, which is interpolated to the
        feature resolution. This is what lets the modulation vary across the image without a
        convolution, and it costs ``rank * 8 * 8`` parameters rather than a full spatial head.
        """
        weights = self.to_spatial(descriptor)                                  # (B, r)
        basis = self.spatial_basis.unsqueeze(0).expand(descriptor.shape[0], -1, -1, -1)
        basis = torch.nn.functional.interpolate(basis, size=size, mode="bilinear", align_corners=False)
        return torch.einsum("br,brhw->bhw", weights, basis).unsqueeze(1)        # (B, 1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = self._descriptor(x.shape[0], x.device)
        gate = self.alpha()
        out = x
        if self.to_gamma is not None:
            gamma = self.to_gamma(descriptor)
            while gamma.dim() < x.dim():
                gamma = gamma.unsqueeze(-1)
            out = out * (1.0 + gate * gamma)
        if self.to_beta is not None:
            beta = self.to_beta(descriptor)
            while beta.dim() < x.dim():
                beta = beta.unsqueeze(-1)
            if self.mode == "spatial":
                # A position-dependent factor on the offset: the modulation is allowed to differ
                # across the image, at the cost of ``rank * 8 * 8`` parameters rather than a full
                # spatial head.
                beta = beta * (1.0 + self._spatial_map(descriptor, x.shape[-2:]))
            out = out + gate * beta
        return out
