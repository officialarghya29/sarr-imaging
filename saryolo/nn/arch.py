"""Symbolic architecture builder for SAR-YOLO.

Why a builder instead of hand-written YAML
------------------------------------------
Every SAR-YOLO variant differs from the baseline only by *which modules are
inserted where*. Hand-maintaining ~20 YAML files with explicit, renumbered
``from`` indices is the single most likely place to introduce a silent wiring
bug (a wrong index still parses, and just silently degrades accuracy). The
builder assembles rows symbolically and computes every index, so:

* the baseline it emits is **bit-identical** to stock ``yolo11.yaml`` (asserted
  in ``tests/test_arch.py`` against the published parameter counts);
* ablations are expressed as module *subsets*, not as edited copies;
* the module-level ablation grid (SE / ECA / CBAM / ours, and
  concat / add / static / ours) is generated from the same code path.

The emitted YAML is what actually gets trained and committed under
``configs/models/``, so runs remain reproducible even if the builder changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

__all__ = ["ModelSpec", "build_yaml_dict", "build_yaml_text", "variant_filename", "SCALES", "VARIANTS"]

#: Compound scaling constants, copied verbatim from ultralytics ``cfg/models/11/yolo11.yaml``.
SCALES: dict[str, list[float]] = {
    "n": [0.50, 0.25, 1024],
    "s": [0.50, 0.50, 1024],
    "m": [0.50, 1.00, 512],
    "l": [1.00, 1.00, 512],
    "x": [1.00, 1.50, 512],
}

#: Published parameter counts for stock YOLO11, used as a regression guard in tests.
BASELINE_PARAMS: dict[str, int] = {"n": 2_624_080, "s": 9_458_752}

#: Modes of the SARVO prototype front end (``saryolo/nn/modules/cfar.py``). Duplicated as a
#: literal rather than imported so that ``arch.py`` stays free of a dependency on the module
#: package (the module package imports nothing from here, and a cycle would make the builder
#: unimportable from a checkpoint load). A test pins the two lists together.
CFAR_MODES: tuple[str, ...] = ("cfar", "conv", "fixed")

#: Modes of the SARVO core mechanism (``saryolo/nn/modules/ssac.py``). Duplicated as a
#: literal for the same reason as ``CFAR_MODES``; a test pins the two lists together.
SSAC_MODES: tuple[str, ...] = ("adaptive", "adaptive_raw", "fixed")

#: Execution modes of the same block. Orthogonal to ``SSAC_MODES``: the mode decides how the
#: allocation is computed, the execution decides how much arithmetic is spent on it. A
#: ``sparse`` arm is parameter-identical to its ``dense`` twin, so the pair differs only in
#: the amount of arithmetic executed -- which is what makes an efficiency comparison
#: attributable. Duplicated as a literal; a test pins the two lists together.
SSAC_EXECUTIONS: tuple[str, ...] = ("dense", "sparse")


@dataclass
class ModelSpec:
    """Declarative description of one SAR-YOLO architecture variant.

    Attributes:
        name: Variant name, e.g. ``"saryolo_full"``.
        scale: Compound scale key (``n``/``s``/``m``/``l``/``x``).
        nc: Number of classes.
        adapter: SIA variant (Module A), emitted as the **first** backbone row so it
            operates on the image rather than on features; ``None`` disables. Kept out of
            the v2 full configuration on purpose: the brief's rule is to keep an input
            representation only if the experiment justifies it, and v2 is already +72%
            parameters, so the adapter is a candidate arm rather than a default.
        enhancement: SFE variant; ``None`` disables the module.
        speckle: SFM variant; ``None`` disables the module.
        frequency: SFR variant, on the deepest backbone stage; ``None`` disables.
        prior: TPM variant, inserted per detection level *before* attention;
            ``None`` disables the module. Placed upstream of attention on purpose:
            attention then operates on target-modulated features, which is the
            structural form of the "target-aware attention" claim.
        attention: Attention variant inserted before the head; ``None`` disables.
        context: CAG variant, inserted per detection level after attention;
            ``None`` disables.
        refinement: TADR variant, inserted per detection level last, immediately
            pre-head; ``None`` disables. Placed after the prior on purpose: the
            deformable offsets are then predicted from an already prior-modulated
            feature, which is what makes the arm "target-aware" without duplicating
            the prior's evidence network.
        refinement_max_offset: Search radius of Component 11's deformable offsets, in
            normalised grid units. Exposed as a field because the smoke run showed the
            learned offsets saturating the bound, so it is a hyperparameter to sweep
            rather than a constant to leave implicit.
        fusion: Fusion variant inserted after each ``Concat``; ``None`` disables.
        efficiency: Marks an arm whose stated purpose is the *cost* claim rather than the
            accuracy ladder. The flag is carried into the generated YAML header and is what
            the efficiency-frontier table keys on, so an arm cannot be advertised as a
            cheaper operating point without the generator agreeing that is what it is. It
            changes no architecture row -- it is provenance, not wiring.
        levels: Detection levels, a subset of ``("p2", "p3", "p4", "p5")``.
        sar_loss: Optional Component-7 loss block, emitted as a top-level
            ``sar_loss`` key. Read by :class:`SARYOLODetectionModel` at criterion
            construction, which keeps the objective reproducible from the YAML.
        notes: Free-form provenance note carried into the YAML header.
    """

    name: str
    scale: str = "s"
    nc: int = 1
    #: SARVO prototype first representation (master Phase 5): the mode of
    #: :class:`saryolo.nn.modules.cfar.RatioSpaceCFARFrontEnd`, emitted as the **first**
    #: backbone row so it reads the image rather than a feature map. ``None`` disables.
    #: Distinct from ``adapter`` (SIA), which is also input-side but produces a new
    #: representation from learned filters; this one computes an analytic radar
    #: statistic and learns only how much of it to use.
    cfar: str | None = None
    #: SARVO core mechanism (master Phase 2, SSAC): the mode of
    #: :class:`saryolo.nn.modules.ssac.ScatterSelectiveRefinement`, emitted per detection
    #: level immediately pre-head. ``None`` disables. Distinct from every other slot in
    #: that it decides *how much computation* a location receives rather than only how the
    #: feature there is weighted.
    ssac: str | None = None
    #: Core-mechanism configuration, emitted into the module's YAML argument list. Explicit
    #: fields rather than one packed string, because the cost ablation varies them one at a
    #: time and a half-specified arm would be a wiring mistake waiting to happen. Every one of
    #: them is validated in ``__post_init__`` and pinned against the module's own constructor
    #: order by ``tests/test_ssac.py::test_the_yaml_argument_order_matches_the_module_signature``.
    ssac_scales: tuple[int, ...] = (3, 7)
    ssac_hidden: int = 16
    ssac_expand: int = 2
    ssac_execution: str = "dense"
    ssac_tile: int = 16
    ssac_keep: float = 0.25
    ssac_penalty: float = 0.0
    ssac_tau: float = 1.0
    ssac_select: float = 0.5
    adapter: str | None = None
    enhancement: str | None = None
    speckle: str | None = None
    frequency: str | None = None
    prior: str | None = None
    attention: str | None = None
    context: str | None = None
    refinement: str | None = None
    refinement_max_offset: float = 0.5
    fusion: str | None = None
    efficiency: bool = False
    #: Acquisition-conditioned adapter: ``"<mode>:<field set>"``, e.g. ``"film:all"`` or
    #: ``"scale:continuous_only"``. A single string rather than two fields because the ablation
    #: always varies them together, and a half-specified arm is a wiring mistake waiting to
    #: happen.
    conditioning: str | None = None
    #: Categorical table sizes for the conditioning adapter, in ``CATEGORICAL_FIELDS`` order
    #: (``sensor``, ``polarization``, ``mode``). Emitted into the YAML because an embedding table
    #: that disagrees with the vocabulary used to encode the data would silently mis-index every
    #: sensor. A tuple of ints rather than a string: ``parse_model`` literal-evals every string
    #: argument and a non-literal one raises ``SyntaxError``, which is not suppressed.
    conditioning_vocab: tuple[int, ...] = (2, 2, 2)
    levels: tuple[str, ...] = ("p3", "p4", "p5")
    sar_loss: dict[str, float] | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.cfar is not None and self.cfar not in CFAR_MODES:
            raise ValueError(f"cfar must be one of {CFAR_MODES}, got {self.cfar!r}")
        if self.ssac is not None and self.ssac not in SSAC_MODES:
            raise ValueError(f"ssac must be one of {SSAC_MODES}, got {self.ssac!r}")
        if self.ssac_execution not in SSAC_EXECUTIONS:
            raise ValueError(f"ssac_execution must be one of {SSAC_EXECUTIONS}, got {self.ssac_execution!r}")
        if self.ssac and not self.ssac_scales:
            raise ValueError("ssac_scales must not be empty: the assessment needs a window")
        if self.ssac and self.ssac_tile < 1:
            raise ValueError(f"ssac_tile must be >= 1, got {self.ssac_tile!r}")
        if self.ssac and not 0.0 < self.ssac_keep <= 1.0:
            raise ValueError(f"ssac_keep must be a fraction in (0, 1], got {self.ssac_keep!r}")
        if self.ssac and self.ssac_execution == "sparse" and self.ssac_keep >= 1.0:
            # Not an error in the module (keep=1 is the equivalence case the tests use), but a
            # sparse *arm* that refines every tile pays the gather/scatter for nothing, so it
            # would be an arm whose stated purpose it cannot serve.
            raise ValueError("a sparse ssac arm with ssac_keep >= 1 refines every tile and cannot measure a routing saving")
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {sorted(SCALES)}, got {self.scale!r}")
        if "p2" in self.levels and self.levels != ("p2", "p3", "p4", "p5"):
            raise ValueError(f"When enabled, the P2 level must come with P3, P4 and P5 (got {self.levels!r}).")
        if not self.levels or self.levels[-1] != "p5" or self.levels[0] not in ("p2", "p3"):
            raise ValueError(f"levels must be a contiguous ('p2'|'p3'),'p4','p5' set, got {self.levels!r}")

    @property
    def has_p2(self) -> bool:
        return "p2" in self.levels

    @property
    def module_names(self) -> list[str]:
        """Enabled module slugs, in the order they appear along the forward graph."""
        mods = []
        if self.cfar:
            mods.append("cfar")
        if self.adapter:
            mods.append("adapter")
        if self.enhancement:
            mods.append("sfe")
        if self.speckle:
            mods.append("speckle")
        if self.frequency:
            mods.append("frequency")
        if self.fusion:
            mods.append("fusion")
        if self.prior:
            mods.append("prior")
        if self.attention:
            mods.append("attention")
        if self.context:
            mods.append("context")
        if self.refinement:
            mods.append("refine")
        if self.ssac:
            mods.append("ssac")
        if self.has_p2:
            mods.append("p2_head")
        return mods


class _Builder:
    """Accumulates backbone/head rows while tracking global layer indices.

    ``parse_model`` iterates ``d["backbone"] + d["head"]`` with a single running
    index ``i``, and channel lookups (``ch[f]``) use that same global numbering,
    so the counter here must be global rather than per-section.
    """

    def __init__(self) -> None:
        self.backbone: list[list[Any]] = []
        self.head: list[list[Any]] = []
        self._i = 0

    @property
    def n_rows(self) -> int:
        return self._i

    def add(self, section: str, src: int | list[int], repeats: int, module: str, args: list[Any]) -> int:
        """Append one YAML row and return its global index."""
        idx = self._i
        (self.backbone if section == "backbone" else self.head).append([src, repeats, module, args])
        self._i += 1
        return idx


def variant_filename(spec: ModelSpec | str) -> str:
    r"""Canonical YAML file name for a variant.

    The ``yolo11<scale>`` prefix is **required**, not cosmetic. Ultralytics'
    ``yaml_model_load`` overwrites any ``scale`` key in the YAML body with
    ``guess_model_scale(path)``, which only recognises the pattern
    ``yolo(e-)?v?\d+[nslmx]`` in the *file name*. A file named ``baseline_s.yaml``
    therefore yields an empty scale, and ``parse_model`` silently falls back to
    the first entry of ``scales`` (``n``) — so a model you believe is YOLO11s is
    actually built as YOLO11n. Encoding the scale in the file name is the only
    reliable way to select it when loading from a file.

    See ``tests/test_arch.py::test_variant_filenames_encode_scale``, which pins
    this against ``guess_model_scale`` itself.
    """
    if isinstance(spec, str):
        spec = VARIANTS[spec]
    return f"yolo11{spec.scale}_{spec.name}.yaml"


def build_yaml_dict(spec: ModelSpec) -> dict[str, Any]:
    """Build the Ultralytics model dictionary for ``spec``.

    The baseline path (all modules ``None``, ``levels=("p3","p4","p5")``)
    reproduces stock ``yolo11.yaml`` exactly.
    """
    b = _Builder()

    # ------------------------------------------------------------------ backbone
    # SARVO prototype (RS-CFAR): the radar statistic as the first representation. Emitted
    # before everything else because it is the module that decides what the first
    # convolution reads; emitting it later would make it a feature module, which is the
    # slot the existing SFM/SFE components already occupy and the reason this design was
    # placed input-side in the first place (docs/architecture_proposals.md §2).
    if spec.cfar:
        b.add("backbone", -1, 1, "RatioSpaceCFARFrontEnd", ["ch", spec.cfar])

    # Module A (SIA), the only row that consumes the image itself. Everything downstream
    # sees its output, so this is the one place where "SAR-aware input representation" is
    # not a description of features but of the tensor the first convolution reads.
    if spec.adapter:
        # Args stop after the mode: `source` is meaningless at row 0 (there is no source
        # layer) and the kernel/reduction defaults are the ones the slot study holds fixed.
        b.add("backbone", -1, 1, "SARInputAdapter", ["ch", spec.adapter])
    b.add("backbone", -1, 1, "Conv", [64, 3, 2])  # P1/2
    b.add("backbone", -1, 1, "Conv", [128, 3, 2])  # P2/4
    p2 = b.add("backbone", -1, 2, "C3k2", [256, False, 0.25])

    # Component 1 (SFE): fine-detail enhancement on the highest-resolution retained feature.
    if spec.enhancement:
        p2 = b.add("backbone", -1, 1, "SARFeatureEnhancement", ["ch", spec.enhancement, 3, 8])

    b.add("backbone", -1, 1, "Conv", [256, 3, 2])  # P3/8
    p3: int = b.add("backbone", -1, 2, "C3k2", [512, False, 0.25])
    b.add("backbone", -1, 1, "Conv", [512, 3, 2])  # P4/16
    p4 = b.add("backbone", -1, 2, "C3k2", [512, True])
    b.add("backbone", -1, 1, "Conv", [1024, 3, 2])  # P5/32
    # The P5 C3k2 is consumed positionally by SPPF (via `-1`) and, unlike P2-P4, is
    # never referenced as a lateral link, so it needs no index binding here.
    b.add("backbone", -1, 2, "C3k2", [1024, True])
    b.add("backbone", -1, 1, "SPPF", [1024, 5])
    spp = b.add("backbone", -1, 2, "C2PSA", [1024])

    # Component 2 (SFM): speckle/clutter discrimination where semantics are strongest.
    if spec.speckle:
        spp = b.add("backbone", -1, 1, "SpeckleAwareFeatureModule", ["ch", spec.speckle, 8, 3])

    # Component 9 (SFR): spatial-frequency branch on the deepest backbone stage. Placed
    # here deliberately: the FFT cost scales with spatial resolution, so P5/32 is the
    # cheapest stage to attach it to, and it is also where speckle/clutter statistics are
    # already aggregated. Both modules preserve channels, so `spp` stays the P5 lateral.
    if spec.frequency:
        spp = b.add("backbone", -1, 1, "SpatialFrequencyRepresentation", ["ch", spec.frequency, 8, 8])

    def fuse(src_a: int, src_b: int, c_out: int, c3k: bool) -> int:
        """Emit ``Upsample/Conv -> Concat -> [fusion] -> C3k2`` and return the output index."""
        # The fusion block reads the Concat output via `-1`, so no index binding is needed.
        b.add("head", [src_a, src_b], 1, "Concat", [1])
        if spec.fusion:
            b.add("head", -1, 1, "AdaptiveMultiScaleFusion", ["ch", [src_a, src_b], 8, spec.fusion])
        return b.add("head", -1, 2, "C3k2", [c_out, c3k])

    # ------------------------------------------------------------------- top-down
    up = b.add("head", -1, 1, "nn.Upsample", [None, 2, "nearest"])
    n_p4_up = fuse(up, p4, 512, False)

    up = b.add("head", -1, 1, "nn.Upsample", [None, 2, "nearest"])
    n_p3 = fuse(up, p3, 256, False)

    #: Index of the P2 neck output; only materialised when the P2 head is enabled.
    n_p2: int | None = None
    if spec.has_p2:
        up = b.add("head", -1, 1, "nn.Upsample", [None, 2, "nearest"])
        n_p2 = fuse(up, p2, 128, False)

    # ------------------------------------------------------------------ bottom-up
    if spec.has_p2:
        down = b.add("head", -1, 1, "Conv", [128, 3, 2])
        out_p3 = fuse(down, n_p3, 256, False)
        down = b.add("head", -1, 1, "Conv", [256, 3, 2])
        out_p4 = fuse(down, n_p4_up, 512, False)
        down = b.add("head", -1, 1, "Conv", [512, 3, 2])
        out_p5 = fuse(down, spp, 1024, True)
    else:
        down = b.add("head", -1, 1, "Conv", [256, 3, 2])
        out_p4 = fuse(down, n_p4_up, 512, False)
        down = b.add("head", -1, 1, "Conv", [512, 3, 2])
        out_p5 = fuse(down, spp, 1024, True)
        out_p3 = n_p3

    # Per-level research slots, immediately pre-head, in the order they act on the feature:
    #   Component 8 (TPM) -> Component 3 (SAA) -> Component 10 (CAG) -> Component 11 (TADR)
    # The prior comes first so attention, context and deformation all operate on
    # target-modulated features. Every slot passes its source index explicitly: these rows
    # use an explicit `from`, so `ch[-1]` would resolve to an unrelated layer and silently
    # mis-wire.
    heads = {"p3": out_p3, "p4": out_p4, "p5": out_p5}
    if n_p2 is not None:
        heads["p2"] = n_p2
    for lvl in spec.levels:
        if spec.conditioning:
            # Placed first in the per-level chain so that every later slot -- prior, attention,
            # context, refinement -- operates on acquisition-conditioned features. Putting it
            # after them would condition the final modulation only, and the ablation would then
            # be measuring a different (weaker) intervention than the one being claimed.
            mode, _, field_set = spec.conditioning.partition(":")
            heads[lvl] = b.add(
                "head", heads[lvl], 1, "AcquisitionConditionedAdapter",
                ["ch", heads[lvl], mode, field_set or "all", list(spec.conditioning_vocab)],
            )
        if spec.prior:
            heads[lvl] = b.add(
                "head", heads[lvl], 1, "TargetPriorModulation", ["ch", heads[lvl], spec.prior, 3, 8]
            )
        if spec.attention:
            gate = "static" if spec.attention == "saa_static" else "adaptive"
            heads[lvl] = b.add(
                "head", heads[lvl], 1, "SARAdaptiveAttention", ["ch", heads[lvl], 16, 7, gate]
            )
        if spec.context:
            heads[lvl] = b.add(
                "head", heads[lvl], 1, "ContextAggregation", ["ch", heads[lvl], spec.context, 8]
            )
        if spec.refinement:
            heads[lvl] = b.add(
                "head", heads[lvl], 1, "TargetAwareRefinement",
                ["ch", heads[lvl], spec.refinement, 3, spec.refinement_max_offset],
            )
        if spec.ssac:
            # The SARVO core mechanism acts last in the per-level chain, so the detection
            # head reads the adaptively-refined representation -- which is what makes the
            # "the allocation supports detection" claim direct rather than indirect.
            #
            # The argument list is written out in full, in the module's own positional order,
            # rather than relying on defaults: the YAML is the artefact a checkpoint carries,
            # so a reader (and the arg-order test) must be able to see which mechanism was
            # actually built without reading the class. The trailing two constructor arguments
            # (``alpha_init``, ``eps``) stay at their defaults -- they are numerical details,
            # not mechanism choices.
            heads[lvl] = b.add(
                "head", heads[lvl], 1, "ScatterSelectiveRefinement",
                [
                    "ch", heads[lvl], spec.ssac, list(spec.ssac_scales), spec.ssac_hidden,
                    spec.ssac_expand, spec.ssac_execution, spec.ssac_tile, spec.ssac_keep,
                    spec.ssac_penalty, spec.ssac_tau, spec.ssac_select,
                ],
            )

    b.add("head", [heads[lvl] for lvl in spec.levels], 1, "Detect", ["nc"])

    out: dict[str, Any] = {
        "nc": spec.nc,
        # Set explicitly: ultralytics otherwise infers the scale from the *filename*
        # (guess_model_scale), which would silently fall back to 'n' for a dict.
        "scale": spec.scale,
        "scales": {k: list(v) for k, v in SCALES.items()},
        "backbone": b.backbone,
        "head": b.head,
    }
    if spec.sar_loss:
        # Emitted after the architecture rows so parse_model ignores it (it only reads
        # named task keys) while our criterion can still pick it up from model.yaml.
        out["sar_loss"] = dict(spec.sar_loss)
    return out


def build_yaml_text(spec: ModelSpec) -> str:
    """Serialise ``spec`` to YAML text, with a provenance header."""
    import yaml

    mods = ", ".join(spec.module_names) if spec.module_names else "none (stock YOLO11)"
    header = (
        f"# SAR-YOLO / {spec.name}  -- GENERATED FILE, do not edit by hand.\n"
        f"# Regenerate with:  python -m saryolo arch --variant {spec.name}\n"
        f"# scale={spec.scale}  levels={'-'.join(spec.levels)}  modules={mods}\n"
        f"# efficiency_arm={spec.efficiency}  (True = the arm exists for the cost claim)\n"
        f"# sar_loss={'off' if not spec.sar_loss else spec.sar_loss}\n"
        f"#\n"
        f"# NOTE: when this file is loaded, ultralytics derives the compound scale from the\n"
        f"# FILE NAME (guess_model_scale), overwriting the 'scale' key below. The file must\n"
        f"# therefore keep its 'yolo11{spec.scale}_' prefix or it will silently build at scale 'n'.\n"
    )
    if spec.notes:
        header += f"# {spec.notes}\n"
    header += "#\n# [from, repeats, module, args]\n"
    return header + yaml.safe_dump(build_yaml_dict(spec), sort_keys=False, default_flow_style=None)


#: Starting SAR-loss weights for the full model. These are *initial* values to be
#: tuned by the EXP-008 sweep; they are recorded here so every run is reproducible.
SAR_LOSS_FULL: dict[str, float] = {
    "w_sep": 0.2,
    "w_small": 0.5,
    "w_smooth": 0.05,
    "margin": 1.0,
    "bg_frac": 0.25,
}


# --------------------------------------------------------------------------- variants
def _v(name: str, **kw) -> ModelSpec:
    return ModelSpec(name=name, **kw)


#: Canonical experiment variants. ``EXP-00x`` ids match ``configs/exp/``.
VARIANTS: dict[str, ModelSpec] = {
    # EXP-001 — engineering baseline, must equal stock YOLO11.
    "baseline": _v("baseline", notes="EXP-001 baseline: stock YOLO11, no SAR modules."),
    # EXP-002 — Component 1.
    "sfe": _v("sfe", enhancement="sfe", notes="EXP-002 baseline + SAR Feature Enhancement."),
    # EXP-003 — Components 1+2.
    "speckle": _v("speckle", enhancement="sfe", speckle="sfm",
                  notes="EXP-003 + Speckle-Aware Feature Module."),
    # EXP-004 — Components 1-3.
    "attention": _v("attention", enhancement="sfe", speckle="sfm", attention="saa",
                    notes="EXP-004 + SAR-Adaptive Attention."),
    # EXP-005 — Components 1-4.
    "amf": _v("amf", enhancement="sfe", speckle="sfm", attention="saa", fusion="amf",
              notes="EXP-005 + Adaptive Multi-Scale Fusion."),
    # EXP-006 — + P2 small-object head.
    "p2": _v("p2", enhancement="sfe", speckle="sfm", attention="saa", fusion="amf",
             levels=("p2", "p3", "p4", "p5"), notes="EXP-006 + P2 small-object detection head."),
    # EXP-007 — full architecture. The SAR loss is switched on in the YAML's sar_loss block.
    "full": _v("full", enhancement="sfe", speckle="sfm", attention="saa", fusion="amf",
               levels=("p2", "p3", "p4", "p5"), sar_loss=dict(SAR_LOSS_FULL),
               notes="EXP-007 FULL SAR-YOLO: all modules + P2 head + SAR-aware loss."),
    # EXP-007 without the auxiliary objective: isolates the architecture from the loss.
    "full_noloss": _v("full_noloss", enhancement="sfe", speckle="sfm", attention="saa", fusion="amf",
                      levels=("p2", "p3", "p4", "p5"),
                      notes="FULL architecture with the SAR-aware loss switched off (ablation control)."),
    # --- attention module-level ablation (slot-matched) ---
    "att_none": _v("att_none", enhancement="sfe", speckle="sfm",
                   notes="Module ablation: attention slot disabled."),
    "att_se": _v("att_se", enhancement="sfe", speckle="sfm", attention="se",
                 notes="Module ablation: SE in the attention slot."),
    "att_eca": _v("att_eca", enhancement="sfe", speckle="sfm", attention="eca",
                  notes="Module ablation: ECA in the attention slot."),
    "att_cbam": _v("att_cbam", enhancement="sfe", speckle="sfm", attention="cbam",
                   notes="Module ablation: CBAM in the attention slot."),
    "att_saa_static": _v("att_saa_static", enhancement="sfe", speckle="sfm", attention="saa_static",
                         notes="Module ablation: ours with static (non-adaptive) branch weights."),
    # --- fusion module-level ablation ---
    "fus_concat": _v("fus_concat", enhancement="sfe", speckle="sfm", attention="saa", fusion="concat",
                     notes="Module ablation: stock Concat in the fusion slot."),
    "fus_add": _v("fus_add", enhancement="sfe", speckle="sfm", attention="saa", fusion="add",
                  notes="Module ablation: plain Add in the fusion slot."),
    "fus_static": _v("fus_static", enhancement="sfe", speckle="sfm", attention="saa", fusion="static",
                     notes="Module ablation: ours with static (non-adaptive) scale weights."),
    # --- preprocessing module-level ablation ---
    "pre_identity": _v("pre_identity", enhancement="identity", speckle="sfm", attention="saa", fusion="amf",
                       notes="Module ablation: no input enhancement branch."),
    "pre_log": _v("pre_log", enhancement="log", speckle="sfm", attention="saa", fusion="amf",
                  notes="Module ablation: log-compression enhancement."),
    "pre_clahe": _v("pre_clahe", enhancement="clahe", speckle="sfm", attention="saa", fusion="amf",
                    notes="Module ablation: CLAHE-style enhancement."),
    "pre_standardize": _v("pre_standardize", enhancement="standardize", speckle="sfm", attention="saa", fusion="amf",
                          notes="Module ablation: local standardisation enhancement."),
    # --- speckle module-level ablation ---
    "spk_none": _v("spk_none", enhancement="sfe", speckle="none", attention="saa", fusion="amf",
                   notes="Module ablation: no speckle handling."),
    "spk_lee": _v("spk_lee", enhancement="sfe", speckle="lee", attention="saa", fusion="amf",
                  notes="Module ablation: classical Lee filter."),
    "spk_denoise": _v("spk_denoise", enhancement="sfe", speckle="denoise", attention="saa", fusion="amf",
                      notes="Module ablation: fixed low-pass despeckling."),
}

# ------------------------------------------------------------------- v2 components
#: Configuration of the full v2 model: the v1 full model plus the target-prior,
#: spatial-frequency, context and refinement components.
V2_FULL: dict[str, Any] = {
    "enhancement": "sfe",
    "speckle": "sfm_clutter",
    "frequency": "sff",
    "prior": "learned",
    "attention": "saa",
    "context": "multi",
    "refinement": "deform",
    "fusion": "amf",
    "levels": ("p2", "p3", "p4", "p5"),
}


def _v2(name: str, scale: str = "s", notes: str = "", **overrides: Any) -> ModelSpec:
    """Build a v2 variant: the full v2 configuration with the named slots overridden.

    Passing ``slot=None`` removes the module; passing ``slot="none"`` keeps the module
    in the graph with its mechanism switched off. Both controls are used below, because
    they answer different questions -- see the note on the slot studies.
    """
    cfg = dict(V2_FULL)
    cfg.update(overrides)
    return ModelSpec(name=name, scale=scale, sar_loss=dict(SAR_LOSS_FULL), notes=notes, **cfg)


def _with_sar_loss(spec: ModelSpec, **overrides: float | str) -> ModelSpec:
    """Copy a spec with extra SAR-loss keys merged into its ``sar_loss`` block.

    Loss-only variants go through here rather than through ``_v2(**overrides)``, because
    ``_v2`` passes ``sar_loss`` positionally into ``ModelSpec`` -- a ``sar_loss`` key in
    ``overrides`` would therefore be a duplicate keyword argument rather than an override.

    The copy matters: ``VARIANTS`` is shared module state, and every measurement, YAML
    emitter and test reads from it.
    """
    merged = {**(spec.sar_loss or {}), **overrides}
    return replace(spec, sar_loss=merged)


#: EXP-013..016 -- the v2 cumulative ladder. Each row adds exactly one new component.
#: The v1 ladder (EXP-001..008) is left untouched: those configs are already committed
#: and referenced by the paper tables, and rewriting their meaning would break the
#: traceability between a published number and the run that produced it.
VARIANTS.update({
    "v2_clutter": _v2("v2_clutter", frequency=None, prior=None, context=None, refinement=None,
                      notes="EXP-013 v1 FULL + clutter-aware three-branch SFM."),
    "v2_prior": _v2("v2_prior", frequency=None, context=None, refinement=None,
                    notes="EXP-014 + target prior modulation (Component 8)."),
    "v2_freq": _v2("v2_freq", context=None, refinement=None,
                   notes="EXP-015 + spatial-frequency representation (Component 9)."),
    "v2_ctx": _v2("v2_ctx", refinement=None,
                  notes="EXP-016 + context aggregation (Component 10)."),
    "v2_full": _v2("v2_full",
                   notes="EXP-017 FULL v2: v1 + clutter + prior + frequency + context + refinement (Components 1-11)."),
    "v2_prior_spectral": _v2(
        "v2_prior_spectral", prior="spectral",
        notes=("Module G: the target prior also selects the radial frequency bands. This is "
               "where 'target-conditioned frequency selection' can actually be expressed, "
               "because the prior exists here; the backbone spectral slot runs before it."),
    ),
    # Removal arm for Module G: the reference model for it is v2_prior_spectral (EXP-018),
    # not v2_full, so "removal" here means dropping the spectral selection *and* reverting
    # the prior to the spatial-only mechanism -- one edit, the same slot.
    "v2_nopspectral": _v2(
        "v2_nopspectral",
        notes="Removal ablation: v2_prior_spectral without prior-conditioned spectral selection (Module G removed).",
    ),

    # --- SEC. 4 of the brief: the representation-consistency term, as a *loss* slot.
    #
    # A loss slot is not an architectural one, and that changes what a control has to be. The
    # term never adds a parameter, so every arm below is byte-for-byte the same size as
    # `v2_full`: `w_consistency: 0.0` is the control, and the arm differs only in the
    # objective. That is also why these arms are *not* rows in the removal slot -- a removal
    # there is verified by a strict parameter drop, and a loss removal can never satisfy it,
    # so it would either fail its own guard or have to be excluded from it.
    #
    # The perturbation is swept rather than fixed. The claim is that the model should be
    # invariant to *the physics it will meet*, and a term that only helps when the benchmark
    # corruption happens to match its training corruption is a tuned constant, not a
    # principle. `v2_cons` (speckle, 4.0) is the ladder arm; the rest test whether the choice
    # of corruption is load-bearing.
    "v2_cons": _with_sar_loss(
        _v2("v2_cons", notes="SEC. 4: full v2 trained with the representation-consistency term."),
        w_consistency=0.5, consistency_kind="speckle", consistency_severity=4.0,
    ),
    "cons_sev1": _with_sar_loss(
        _v2("cons_sev1", notes="Consistency slot: mild speckle (1 look). Is the term just denoising?"),
        w_consistency=0.5, consistency_kind="speckle", consistency_severity=1.0,
    ),
    "cons_sev16": _with_sar_loss(
        _v2("cons_sev16", notes="Consistency slot: extreme speckle (16 looks). Does the term survive heavy noise?"),
        w_consistency=0.5, consistency_kind="speckle", consistency_severity=16.0,
    ),
    "cons_lowcontrast": _with_sar_loss(
        _v2("cons_lowcontrast", notes="Consistency slot: invariance to contrast compression rather than speckle."),
        w_consistency=0.5, consistency_kind="low_contrast", consistency_severity=2.2,
    ),
    "cons_lowsnr": _with_sar_loss(
        _v2("cons_lowsnr", notes="Consistency slot: invariance to additive noise at 20% of the dynamic range."),
        w_consistency=0.5, consistency_kind="low_snr", consistency_severity=0.20,
    ),

    # --- removal ablation (SEC. 23 of the brief): drop one component from the full model.
    # These differ from the slot studies below in that the *module is absent*, so they
    # answer "does this component earn its place in the architecture?" rather than
    # "which mechanism inside this slot is better?".
    "v2_noclutter": _v2("v2_noclutter", speckle="sfm",
                        notes="Removal ablation: v2 without the clutter-aware branch."),
    "v2_noprior": _v2("v2_noprior", prior=None,
                      notes="Removal ablation: v2 without target prior modulation."),
    "v2_nofreq": _v2("v2_nofreq", frequency=None,
                     notes="Removal ablation: v2 without the spatial-frequency branch."),
    "v2_noctx": _v2("v2_noctx", context=None,
                    notes="Removal ablation: v2 without context aggregation."),
    "v2_norefine": _v2("v2_norefine", refinement=None,
                       notes="Removal ablation: v2 without target-aware deformable refinement."),

    # --- Component 5 slot study. Every arm keeps the module in the graph and holds every
    # other slot at its v2 setting, so the arms differ only in how the prior is produced.
    # `channel` is the matched-capacity control for `learned` (same network, spatially
    # pooled prior); `cfar` is the classical non-learned reference; `static` is uniform.
    "tp_none": _v2("tp_none", prior="none", notes="TPM slot: modulation disabled (inert slot)."),
    "tp_cfar": _v2("tp_cfar", prior="cfar", notes="TPM slot: classical CFAR-style prior."),
    "tp_static": _v2("tp_static", prior="static", notes="TPM slot: spatially uniform learned prior."),
    "tp_channel": _v2("tp_channel", prior="channel",
                      notes="TPM slot: capacity-matched control -- spatial variation removed."),
    "tp_spectral": _v2("tp_spectral", prior="spectral",
                       notes="TPM slot: the prior also selects the radial spectral bands "
                             "(Module G -- target-conditioned frequency selection)."),
    "tp_spectral_feat": _v2("tp_spectral_feat", prior="spectral_feat",
                            notes="TPM slot: same band head and band count, but conditioned on "
                                  "the raw feature instead of the prior (matched-capacity control)."),

    # --- Component 6 slot study.
    "fr_none": _v2("fr_none", frequency="none", notes="SFR slot: spectral branch disabled."),
    "fr_highpass": _v2("fr_highpass", frequency="highpass", notes="SFR slot: fixed high-pass filter."),
    "fr_static": _v2("fr_static", frequency="static", notes="SFR slot: learnable, input-independent."),

    # --- Component 7 slot study.
    "cx_none": _v2("cx_none", context="none", notes="CAG slot: context disabled."),
    "cx_local": _v2("cx_local", context="local", notes="CAG slot: local (dilated) context only."),
    "cx_regional": _v2("cx_regional", context="regional", notes="CAG slot: regional context only."),

    # --- Component 11 slot study. `local` is the capacity control for `deform`: it has
    # the same sub-network and no offsets, so `deform - local` isolates *deformation*
    # rather than the extra convolution. `static` isolates content-adaptive sampling from
    # learned-but-fixed sampling.
    "rf_none": _v2("rf_none", refinement="none", notes="TADR slot: refinement disabled."),
    "rf_local": _v2("rf_local", refinement="local",
                    notes="TADR slot: capacity control -- same network, no offsets."),
    "rf_static": _v2("rf_static", refinement="static",
                     notes="TADR slot: learned but input-independent offsets."),
    # The offset bound is a sweep, not a constant: in smoke training the learned offsets
    # reached |d| ~ 0.47 against s = 0.5, so the tanh was operating where its derivative is
    # smallest. These arms test whether the bound is binding.
    "rf_off25": _v2("rf_off25", refinement_max_offset=0.25,
                    notes="TADR slot: half the search radius (is the bound binding?)."),
    "rf_off100": _v2("rf_off100", refinement_max_offset=1.0,
                     notes="TADR slot: double the search radius (does the model want more?)."),

    # --- SEC. 14 of the brief: the alternative frequency study, beyond FFT vs learned.
    # Same slot, same bands, same parameter count as `v2_full`'s spectral arm, so the
    # transform is the variable rather than the capacity.
    "fr_dct": _v2("fr_dct", frequency="dct",
                  notes="Frequency study: 8x8 block DCT-II with learnable radial bands."),
    "fr_wavelet": _v2("fr_wavelet", frequency="wavelet",
                      notes="Frequency study: one-level Haar with learnable per-sub-band gains."),})


# --- Acquisition conditioning (the cross-sensor claim), applied to the full v2 model.
    #
    # Two independent dimensions are wired here because the brief asks for both, and they answer
    # different questions:
    #
    #   * *mode* -- how the metadata modulates the feature (scale / shift / film / spatial). This
    #     is the adapter-design ablation, and it is deliberately small: the point is to find the
    #     simplest design that works, so a more elaborate one has to beat these and not merely
    #     beat a plain detector.
    #   * *field set* -- which acquisition parameters are consumed. `continuous_only` is the arm
    #     that matters most for the headline claim: it uses only the physically-ordered fields, so
    #     it is the one that can transfer to a sensor with no embedding row at all. If the
    #     categorical arm is the only one that helps, the claim fails on its own terms.
VARIANTS.update({
    "cond_gain": _v2("cond_gain", conditioning="gain:all",
                      notes="Conditioning slot: metadata gain only (fewest parameters)."),
    "cond_shift": _v2("cond_shift", conditioning="shift:all",
                      notes="Conditioning slot: metadata offset only."),
    "cond_film": _v2("cond_film", conditioning="film:sensor_resolution",
                     notes="Conditioning slot: PROPOSED arm -- per-channel modulation from sensor and resolution, the two fields a deployment is most likely to know."),
    "cond_sensor": _v2("cond_sensor", conditioning="film:sensor",
                       notes="Conditioning slot: sensor embedding only. Cannot transfer to an unseen sensor by construction."),
    "cond_resolution": _v2("cond_resolution", conditioning="film:resolution",
                           notes="Conditioning slot: resolution only, a continuous field that exists for any sensor."),
    "cond_continuous": _v2("cond_continuous", conditioning="film:continuous_only",
                           notes="Conditioning slot: physical descriptors only (resolution, band, incidence) -- the arm that can reach an unseen sensor."),
    "cond_spatial": _v2("cond_spatial", conditioning="spatial:all",
                        notes="Conditioning slot: position-dependent modulation, the most expressive arm."),
})

    # --- Module A (SIA) slot study. The brief forbids assuming which input representation is
# right, so all four arms are wired and none is in the v2 default: the adapter has to earn
# its place in the full model through this comparison, not by being assumed.
VARIANTS.update({
    "in_identity": _v2("in_identity", adapter="identity",
                       notes="SIA slot: raw intensity (control; adds no parameters)."),
    "in_local": _v2("in_local", adapter="local",
                    notes="SIA slot: local-statistics representation only."),
    "in_learned": _v2("in_learned", adapter="learned",
                      notes="SIA slot: learned representation only."),
    "in_hybrid": _v2("in_hybrid", adapter="hybrid",
                     notes="SIA slot: proposed -- learned + local-statistics streams, learned fusion."),
})

#: v2 at the scales used for the main benchmark, plus `n` for the CPU pipeline smoke test
#: and for development-scale module triage (the project's phase-1 rule: debug on the
#: smallest model, promote to `s`/`m` only once a component shows signal).
for _s in ("n", "s", "m", "l"):
    VARIANTS[f"v2_full_{_s}"] = _v2(f"v2_full_{_s}", scale=_s, notes=f"v2 full model at scale {_s}.")

VARIANTS["v2_full_p35_s"] = _v2(
    "v2_full_p35_s", levels=("p3", "p4", "p5"),
    notes="v2 full model without the P2 detection level (multi-scale ablation).",
)

#: The efficiency frontier (SARVO: "beat the current SAR detectors on cost at equal
#: accuracy"). These arms are not a new mechanism; they are the *component subset* a
#: removal ablation would select if the two dominant cost drivers fail to repay
#: themselves. That framing matters, because it is the only honest one: the light arms
#: drop exactly the two slots whose measured share of the compute is largest, so the
#: paper reports a frontier rather than a model that grew a second set of modules.
#:
#: The measurement that motivates the subset, at scale ``s`` with a one-class head:
#:
#:   full v2           16.230 M   55.68 GFLOPs
#:   − context         15.066 M   52.93 GFLOPs
#:   − context − AMF   11.016 M   32.55 GFLOPs   ← the light arm's shape
#:
#: So context aggregation (+1.16 M / +2.75 G) and adaptive multi-scale fusion
#: (+4.05 M / +20.38 G) together are ~42% of full v2's compute. Removing them is the
#: single largest cost lever available without touching the backbone, and it is a lever
#: the ablation machinery already owns -- both slots have their own removal arms
#: (``v2_noctx``, and ``fus_*``), so the light arm cannot be a hidden change.
#:
#: Everything that carries a physical prior is *kept*: SFE, the clutter-aware SFM, the
#: spectral branch, the target prior, the deformable refinement, the P2 level and the SAR
#: loss. The cost claim must not be bought by deleting the science.
LITE_FULL: dict[str, Any] = {
    "enhancement": "sfe",
    "speckle": "sfm_clutter",
    "frequency": "sff",
    "prior": "learned",
    "attention": "saa",
    "context": None,
    "refinement": "deform",
    "fusion": None,
    "levels": ("p2", "p3", "p4", "p5"),
}


def _lite(name: str, scale: str = "s", notes: str = "", **overrides: Any) -> ModelSpec:
    """Build a light variant: the efficiency subset with the named slots overridden."""
    cfg = dict(LITE_FULL)
    cfg.update(overrides)
    return ModelSpec(name=name, scale=scale, sar_loss=dict(SAR_LOSS_FULL), efficiency=True,
                     notes=notes, **cfg)


#: One arm per scale, plus the conditioning and no-P2 points of the frontier.
for _s in ("n", "s", "m"):
    VARIANTS[f"v2_lite_{_s}"] = _lite(
        f"v2_lite_{_s}", scale=_s,
        notes=(f"Efficiency frontier at scale {_s}: full v2 minus context aggregation and "
               f"adaptive multi-scale fusion (the two largest compute slots)."),
    )

VARIANTS["v2_lite_cond_s"] = _lite(
    "v2_lite_cond_s", conditioning="film:sensor_resolution",
    notes=("Efficiency frontier + the acquisition-conditioned adapter. The cross-sensor "
           "claim must be affordable, so the adapter is priced on the light model rather "
           "than only on full v2."),
)

VARIANTS["v2_lite_p35_s"] = _lite(
    "v2_lite_p35_s", levels=("p3", "p4", "p5"),
    notes=("Efficiency frontier without the P2 level: the cheapest point that still keeps "
           "every physical prior. Reported so the P2 cost is visible on the frontier."),
)

#: SARVO prototype arms (master Phase 5, `docs/architecture_proposals.md` §2). These are
#: *not* component-ladder arms: no SFE, no SFM, no conditioning, no extra detection level.
#: They differ from the stock detector in exactly one place -- the representation the first
#: convolution reads -- so the comparison against `baseline_s` is attributable to the
#: statistic, and not to any of the modules the ladder would otherwise carry.
VARIANTS["cfar_s"] = _v(
    "cfar_s", cfar="cfar",
    notes=("SARVO prototype: analytic multi-scale CFAR statistic as the first representation, "
           "with a zero-initialised per-pixel gain (identity at init)."),
)
VARIANTS["cfar_fixed_s"] = _v(
    "cfar_fixed_s", cfar="fixed",
    notes=("SARVO prototype control: the same statistic stack with an *analytic* threshold and "
           "no learnable parameters. Separates 'the statistics help' from 'the learned gain "
           "helps'. Deliberately not identity at init -- a control that equals the baseline "
           "measures nothing."),
)
#: Scale ``n`` of the same two arms. The real-data pilot runs at scale ``n`` (it is the only
#: scale that fits the CPU budget), so the prototype must exist there too, against the same
#: baseline the pilot already measured -- otherwise the comparison would be across scales as
#: well as across representations.
VARIANTS["cfar_conv_s"] = _v(
    "cfar_conv_s", cfar="conv",
    notes=("SARVO prototype matched-cost control: the *same* gain network fed the raw "
           "intensity instead of the statistic stack. Exact parameter parity with cfar_s, so "
           "a difference between the two is the representation and not the capacity."),
)
VARIANTS["cfar_n"] = _v("cfar_n", scale="n", cfar="cfar", notes="cfar_s at scale n (pilot scale).")
VARIANTS["cfar_fixed_n"] = _v(
    "cfar_fixed_n", scale="n", cfar="fixed",
    notes="cfar_fixed_s at scale n (pilot scale); the fixed-threshold control.",
)
VARIANTS["cfar_conv_n"] = _v(
    "cfar_conv_n", scale="n", cfar="conv",
    notes="cfar_conv_s at scale n (pilot scale); the matched-cost control.",
)

#: SARVO core-mechanism prototype arms (master Phase 2, SSAC). Like the CFAR arms these
#: are *not* component-ladder arms: no SFE, no SFM, no conditioning, no extra detection
#: level. They differ from the stock detector in exactly one place -- the selective
#: refinement inserted per detection level -- so a measured difference is attributable to
#: the allocation mechanism and not to any module the ladder would otherwise carry.
#:
#: `adaptive` is the proposal; `adaptive_raw` is the assessment alternative (the same
#: scorer fed the raw feature) that tests whether the SAR statistic is the useful signal;
#: `fixed` is the **matched fixed-computation control** -- parameter-identical to
#: `adaptive` by construction, with the allocation made spatially constant. The master
#: workflow names the fixed-computation arm as the key control, and this is it.
VARIANTS["ssac_s"] = _v(
    "ssac_s", ssac="adaptive",
    notes=("SARVO core mechanism: Scatter-Selective Adaptive Computation -- a per-region "
           "allocation between a cheap shared path and an expensive selective refinement, "
           "driven by an analytic SAR statistic and gated by a zero-initialised residual."),
)
VARIANTS["ssac_raw_s"] = _v(
    "ssac_raw_s", ssac="adaptive_raw",
    notes=("SSAC assessment alternative: identical scorer and allocation, fed the raw "
           "feature instead of the SAR statistic. Matches the proposal => the statistic "
           "is not the useful signal."),
)
VARIANTS["ssac_fixed_s"] = _v(
    "ssac_fixed_s", ssac="fixed",
    notes=("SSAC matched fixed-computation control: exact parameter parity with ssac_s, "
           "with the allocation made spatially constant. Isolates adaptivity from capacity."),
)
#: Scale `n` of the same three arms. The real-data pilot runs at scale `n` (the only scale
#: that fits the CPU budget), so the arms must exist there against the same baseline the
#: pilot already measured -- otherwise the comparison would be across scales as well as
#: across mechanisms.
VARIANTS["ssac_n"] = _v(
    "ssac_n", scale="n", ssac="adaptive", notes="ssac_s at scale n (pilot scale).",
)
VARIANTS["ssac_raw_n"] = _v(
    "ssac_raw_n", scale="n", ssac="adaptive_raw", notes="ssac_raw_s at scale n (pilot scale).",
)
VARIANTS["ssac_fixed_n"] = _v(
    "ssac_fixed_n", scale="n", ssac="fixed",
    notes="ssac_fixed_s at scale n (pilot scale); the fixed-computation control.",
)

#: SSAC **execution** arms (master Phase 9, cost ablation). Each is parameter-identical to the
#: dense proposal; they differ only in how much arithmetic the expensive path performs, which
#: is what lets a cost difference be attributed to the execution rather than to capacity.
#:
#: * ``ssac_sparse_s``  -- refines only the selected tiles, with a halo of context. The arm the
#:   efficiency claim has to be earned on: it is the only one whose measured latency can fall.
#: * ``ssac_e1_s``      -- the expensive path's bottleneck at expand=1, i.e. the cheap end of
#:   the cost/benefit question "how much of the mechanism's +31 % parameter cost is load-bearing".
#: * ``ssac_pen_s``     -- the proposal with an explicit allocation-sparsity penalty. The
#:   penalty is deliberately absent from the proposal itself (penalising the allocation to be
#:   small is circular when sparsity is the mechanism's own claim); as an arm it measures what
#:   the mechanism does when *pushed* sparse, which is the premise a sparse build relies on.
for _arm, _kw in (
    ("ssac_sparse_s", {"ssac_execution": "sparse", "ssac_tile": 16, "ssac_keep": 0.25}),
    ("ssac_e1_s", {"ssac_expand": 1}),
    ("ssac_pen_s", {"ssac_penalty": 0.01, "sar_loss": {"w_ssac_sparsity": 0.01}}),
):
    VARIANTS[_arm] = _v(
        _arm, ssac="adaptive", **{k: v for k, v in _kw.items()},
        notes=f"SSAC cost-ablation arm ({_arm}): {sorted(_kw)}",
    )
    _scale_n = f"{_arm[:-2]}_n"  # not str.replace: the arm names contain other "_s"
    VARIANTS[_scale_n] = _v(
        _scale_n, scale="n", ssac="adaptive", **{k: v for k, v in _kw.items()},
        notes=f"{_arm} at scale n (pilot scale).",
    )

#: Baseline comparison variants (EXP-001b): scales of the stock detector.
for _s in ("n", "s", "m", "l"):
    VARIANTS[f"baseline_{_s}"] = _v(f"baseline_{_s}", scale=_s, notes=f"Baseline comparison: YOLO11{_s}.")

#: The full model at every scale.
for _s in SCALES:
    VARIANTS[f"full_{_s}"] = _v(
        f"full_{_s}", scale=_s, enhancement="sfe", speckle="sfm", attention="saa", fusion="amf",
        levels=("p2", "p3", "p4", "p5"), sar_loss=dict(SAR_LOSS_FULL),
        notes=f"FULL SAR-YOLO at scale {_s}.",
    )

#: The full architecture at the operating point chosen for the large-scale benchmark.
for _s in ("s", "m", "l"):
    VARIANTS[f"full_p35_{_s}"] = _v(
        f"full_p35_{_s}", scale=_s, enhancement="sfe", speckle="sfm", attention="saa", fusion="amf",
        sar_loss=dict(SAR_LOSS_FULL), notes=f"FULL SAR-YOLO (P3-P5, no P2 head) at scale {_s}.",
    )


def _main() -> None:
    """CLI: emit YAML for one variant, or all of them."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Emit SAR-YOLO model YAML files.")
    parser.add_argument("--variant", default="all", help="Variant name, or 'all'.")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes written into the YAML.")
    parser.add_argument("--out", default="configs/models", help="Output directory.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"Unknown variant {name!r}. Available: {', '.join(sorted(VARIANTS))}")
        spec = VARIANTS[name]
        spec.nc = args.nc
        path = out / variant_filename(spec)
        path.write_text(build_yaml_text(spec))
        print(f"wrote {path}")


if __name__ == "__main__":
    _main()
