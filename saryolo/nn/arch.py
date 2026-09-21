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
        levels: Detection levels, a subset of ``("p2", "p3", "p4", "p5")``.
        sar_loss: Optional Component-7 loss block, emitted as a top-level
            ``sar_loss`` key. Read by :class:`SARYOLODetectionModel` at criterion
            construction, which keeps the objective reproducible from the YAML.
        notes: Free-form provenance note carried into the YAML header.
    """

    name: str
    scale: str = "s"
    nc: int = 1
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
    levels: tuple[str, ...] = ("p3", "p4", "p5")
    sar_loss: dict[str, float] | None = None
    notes: str = ""

    def __post_init__(self) -> None:
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
                      notes="Frequency study: one-level Haar with learnable per-sub-band gains."),
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
