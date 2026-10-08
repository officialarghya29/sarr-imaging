"""Scatter-Selective Adaptive Computation (SSAC) — the SARVO prototype mechanism.

Motivation
----------
The detector's neck spends the *same* expensive feature processing at *every*
location of a feature map. In SAR that is wasteful in a specific way: a large
fraction of a chip is background whose local statistics are predictable (open
sea, flat field, uniform desert), while the locations that decide the detection —
a weak ship return, a small target against textured clutter — are sparse. SSAC
tests the hypothesis that a *cheap* local assessment can predict which regions
deserve the expensive processing, so the same accuracy is reached with less of
the expensive computation, without discarding weak targets.

Formulation
-----------
For a feature map ``F`` (B, C, H, W):

    # 1. cheap path, always computed, shared by every location
    F_cheap = DW3x3(F)

    # 2. region assessment -- an analytic SAR statistic, then a tiny learned scorer
    s       = [mean_c(F), r_1, c_1, ..., r_K, c_K]     # log-ratio + local CoV per scale
    a       = MLP(s)                                   # (B, 1, H, W) evidence logit
    g       = sigmoid(a / tau)                         # per-region allocation, in (0, 1)

    # 3. expensive path, wanted only where g is high
    F_rich  = [Conv1x1 -> DW5x5 -> Conv1x1](F_cheap)

    # 4. selective mix, gated by a zero-initialised residual
    out     = F + alpha * ( g * F_rich + (1 - g) * F_cheap - F )

Step 3 is where the two *execution modes* differ, and the difference is the whole
point of the mechanism:

* ``execution="dense"`` runs step 3 at **every** location. The allocation then changes
  values, not the amount of arithmetic executed: the FLOPs and the latency are the
  full expensive-path cost whether or not a region was selected.
* ``execution="sparse"`` splits the map into ``tile x tile`` regions, keeps the
  ``keep`` fraction of them with the highest mean evidence, and runs the expensive
  path **only on those tiles** — gathered into a batch dimension, run, and scattered
  back. Rejected tiles are left at the cheap path's value, which is exactly what the
  mix in step 4 would have produced there if ``g`` were 0, so nothing has to be
  masked afterwards.

Each gathered tile is taken with a ``halo`` of surrounding pixels (the receptive
radius of the expensive path, derived from its own kernels rather than hard-coded),
so a tile is refined as if it were still embedded in the full map. The condition that
makes the sparse mode *checkable* is therefore: in ``eval`` mode, with ``keep = 1``,
sparse execution reproduces dense execution to floating-point tolerance. The tiles'
BatchNorm statistics are the one thing that cannot be preserved — in ``train`` mode a
tile sees only its own pixels, so sparse *training* is an approximation of dense
training and is reported as such rather than presented as equivalent.

What this module does **not** claim
-----------------------------------
Neither mode is claimed to be faster than the other a priori. Sparse execution
removes arithmetic, but it also adds a gather/scatter and a Python-level routing
step, and on this CPU-only host the honest question is whether the removed
arithmetic is larger than the added bookkeeping. That is *measured* —
``scripts/ssac_efficiency.py`` times the same checkpoint in both modes at several
``keep`` levels — and reported whatever it says, including when it says the sparse
mode is slower. The accuracy question (Experiment 1 of the master workflow) is
answerable in the dense form alone; the efficiency question needs the sparse form
and a wall-clock number, not a FLOP ratio.

``alpha`` is a :class:`~saryolo.nn.modules._common.ZeroGate`, so ``out == F``
*bit-for-bit* at initialisation and the module is attributable in the same way as
every other module in this repository: a measured change is the learned
allocation, not added capacity that was active from step 0.

Why the assessment statistic is SAR-native
------------------------------------------
The allocation decision is the mechanism, and the *signal* it is built on is
what makes this a SAR design rather than a generic dynamic network. ``r_k`` is
the CFAR decision variable (log-ratio against a local mean) and ``c_k`` is the
local coefficient of variation, whose square is the reciprocal of the equivalent
number of looks. Both are dimensionless, so the learned scorer transfers across
scenes of different absolute brightness — the same argument the front end in
:mod:`saryolo.nn.modules.cfar` is built on. This is deliberate: the repository
already *measured* that these statistics carry usable signal on real HRSID chips
(``REAL-004``), so SSAC reuses a validated SAR signal instead of inventing an
unvalidated one. Whether that signal predicts *where* expensive processing helps
is the separate question this module exists to test.

Modes and the controls that could kill it
-----------------------------------------
``MODES`` declares the proposal and the two arms that falsify it:

* ``"adaptive"``     — the proposal: a per-region allocation from the SAR statistic.
* ``"adaptive_raw"`` — *assessment alternative*: the identical scorer fed the raw
  feature instead of the statistic. If this matches the proposal, the SAR
  statistic is not the useful signal and the design is over-claimed.
* ``"fixed"``        — the **matched fixed-computation control**: identical layer
  shapes, so an *exactly* equal parameter count, with the allocation made
  spatially constant (the scorer's logit is averaged over the map before the
  sigmoid, turning the per-region decision into one learned difficulty scalar per
  image). This is the arm the master workflow calls the key control: it isolates
  *adaptivity* from *capacity*. If the proposal does not beat it, the benefit was
  the extra parameters and the refinement, not the selective allocation.

Execution modes, and what each one is for
-----------------------------------------
``EXECUTIONS`` is orthogonal to ``MODES``: the mode decides how the allocation is
computed, the execution decides how much arithmetic is spent on it. ``"dense"``
pays the full expensive-path cost at every location; ``"sparse"`` refines only the
tiles the allocation selected. Both are implemented and both are measured — the
dense form answers the accuracy question (Experiment 1 of the master workflow),
and the sparse form is where any efficiency claim has to be earned. Sparse
execution does not *assume* a saving: it removes arithmetic and adds a
gather/scatter and a routing step, so the wall-clock difference between the two is
a measured quantity rather than a deduction from a FLOP ratio.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ._common import ConvBNAct, DWConvBNAct, ZeroGate, resolve_c1

__all__ = ["ScatterSelectiveRefinement"]


class ScatterSelectiveRefinement(nn.Module):
    """Selective refinement of the locations the model judges worth the compute.

    Channel-preserving (``C -> C``) and an exact identity at initialisation in every
    mode, so it can be inserted pre-head in a YOLO-family graph without the parser or
    the detection head noticing anything until the gate has learned to move.

    Args:
        c1: Input channels, or the parser's ``ch`` list.
        source: Index into ``ch`` for the layer actually consumed. Required whenever the
            YAML row's ``from`` is an explicit index rather than ``-1``.
        mode: ``"adaptive"`` (proposal), ``"adaptive_raw"`` (assessment alternative) or
            ``"fixed"`` (matched fixed-computation control).
        scales: Local-statistics window sizes for the assessment signal. Geometric
            ``(3, 7)`` by default: the statistic is only meaningful relative to a window
            larger than the scatterer and smaller than the scene.
        hidden: Width of the scorer MLP. Small on purpose — the assessment must be cheap
            enough that it cannot eat the gain it is meant to find.
        expand: Bottleneck expansion of the expensive path. ``2`` makes the expensive
            path's arithmetic clearly larger than the cheap path's, which is what makes
            an allocation decision meaningful at all.
        execution: ``"dense"`` (the expensive path runs everywhere) or ``"sparse"`` (it
            runs only on the selected tiles). Changes no parameter, so a dense and a
            sparse build of the same arm are parameter-identical and the difference
            between them is the *amount of arithmetic executed*.
        tile: Spatial tile size for sparse execution, in feature-map pixels. Small
            tiles route more finely and pay proportionally more halo; large tiles are
            cheaper and coarser. Ignored in dense mode.
        keep: Fraction of tiles the sparse mode refines, taken as the highest-mean
            evidence tiles per image (at least one, so a map is never left unrefined).
            Ignored in dense mode.
        penalty: Weight of the allocation-sparsity term the *model* adds to its loss from
            this block's mean allocation (see ``saryolo/nn/model.py``). ``0`` disables it,
            and it is off in every arm except the deliberate sparsity arm: penalising the
            allocation to be small is circular when the mechanism's own claim is that it
            will be small.
        tau: Temperature of the allocation sigmoid.
        select: Threshold on ``g`` used only for reporting the *selected fraction*; it
            does not change the forward pass in either execution mode.
        alpha_init: Initial residual gate (0 => exact identity at init).
        eps: Numerical floor for the log and the ratio denominators.
    """

    MODES = ("adaptive", "adaptive_raw", "fixed")
    #: Execution modes. Orthogonal to ``MODES``: the mode decides *how the allocation is
    #: computed*, the execution decides *how much arithmetic is spent on it*.
    EXECUTIONS = ("dense", "sparse")
    version = 1
    #: Every mode gates a residual with a zero-initialised ``alpha``, so every mode is an
    #: exact identity at init. Declared as a dict (not a bool) to mirror ``cfar``'s pattern
    #: and to leave room for a mode that deliberately perturbs the baseline.
    IDENTITY_AT_INIT = {"adaptive": True, "adaptive_raw": True, "fixed": True}

    #: Class-level defaults, mirroring the constructor's.
    #:
    #: These are not decoration. Ultralytics checkpoints pickle the *model instance*, so a
    #: block saved by an earlier version of this class is unpickled without ``__init__``
    #: being called: attributes added later simply do not exist on it. Without the fallbacks
    #: here, timing an existing trained checkpoint in a new execution mode would raise
    #: ``AttributeError`` -- and the alternative (re-training to get a compatible object)
    #: would silently change the weights under a claim about execution cost. With them, an
    #: old checkpoint loads as a faithfully *dense* build with the documented routing
    #: defaults, which is what it actually was.
    scales: tuple[int, ...] = (3, 7)
    hidden: int = 16
    expand: int = 2
    execution: str = "dense"
    tile: int = 16
    keep: float = 0.25
    penalty: float = 0.0
    halo: int = 2
    tau: float = 1.0
    select: float = 0.5
    last_gate_mean: torch.Tensor | None = None

    def __init__(
        self,
        c1,
        source: int | None = None,
        mode: str = "adaptive",
        scales: tuple[int, ...] = (3, 7),
        hidden: int = 16,
        expand: int = 2,
        execution: str = "dense",
        tile: int = 16,
        keep: float = 0.25,
        penalty: float = 0.0,
        tau: float = 1.0,
        select: float = 0.5,
        alpha_init: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if execution not in self.EXECUTIONS:
            raise ValueError(f"execution must be one of {self.EXECUTIONS}, got {execution!r}")
        if tile < 1:
            raise ValueError(f"tile must be >= 1, got {tile!r}")
        if not 0.0 < keep <= 1.0:
            raise ValueError(f"keep must be a fraction in (0, 1], got {keep!r}")
        if penalty < 0.0:
            raise ValueError(f"penalty must be >= 0, got {penalty!r}")
        if not scales:
            raise ValueError("scales must not be empty; the assessment needs at least one window")
        if any(k < 3 or k % 2 == 0 for k in scales):
            raise ValueError(f"scales must be odd and >= 3, got {scales!r}")
        if expand < 1:
            raise ValueError(f"expand must be >= 1, got {expand!r}")
        if not 0.0 < select < 1.0:
            raise ValueError(f"select must be a fraction in (0, 1), got {select!r}")
        self.c1 = resolve_c1(c1, source)
        self.mode = mode
        self.scales = tuple(int(k) for k in scales)
        self.hidden = int(hidden)
        self.tau = float(tau)
        self.select = float(select)
        self.expand = int(expand)
        self.execution = execution
        self.tile = int(tile)
        self.keep = float(keep)
        self.penalty = float(penalty)
        self.eps = float(eps)
        self.identity_at_init = self.IDENTITY_AT_INIT[mode]
        #: Mean allocation of the last forward pass, kept only when ``penalty > 0`` so that
        #: the sparsity term is differentiable and an unpenalised arm retains no graph.
        self.last_gate_mean: torch.Tensor | None = None

        self.pools = nn.ModuleList(
            nn.AvgPool2d(k, stride=1, padding=k // 2, count_include_pad=False) for k in self.scales
        )

        # Region assessment: an analytic statistic stack, then a learned scorer. The scorer
        # reads the statistic (proposal / fixed control) or the raw feature (the assessment
        # alternative), which is the only difference between `adaptive` and `adaptive_raw`.
        n_stats = 1 + 2 * len(self.scales)
        head_in = self.c1 if mode == "adaptive_raw" else n_stats
        self.head = nn.Sequential(
            nn.Conv2d(head_in, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

        # Cheap path: shared, always computed, and the *default* the model falls back to
        # where the allocation is low. Kept shallow on purpose -- it is the baseline cost.
        self.cheap = DWConvBNAct(self.c1, self.c1, k=3)

        # Expensive path: the computation the allocation is trying to spend only where it
        # pays. A bottleneck (expand -> depth-wise 5x5 -> contract) so its arithmetic is
        # substantially larger than the cheap path's, otherwise there is nothing to allocate.
        self.rich = nn.Sequential(
            ConvBNAct(self.c1, self.c1 * self.expand, 1),
            DWConvBNAct(self.c1 * self.expand, self.c1 * self.expand, k=5),
            ConvBNAct(self.c1 * self.expand, self.c1, 1),
        )

        self.alpha = ZeroGate(alpha_init)
        # The halo is derived from the expensive path's own kernels, not hard-coded, so that
        # changing its kernel sizes cannot silently invalidate the sparse/dense equivalence
        # (which is asserted in the tests).
        self.halo = self._receptive_halo()

    def _receptive_halo(self) -> int:
        """Receptive radius of the expensive path, summed over its spatial kernels."""
        radius = 0
        for m in self.rich.modules():
            for conv in (getattr(m, "conv", None), getattr(m, "dw", None)):
                if isinstance(conv, nn.Conv2d):
                    radius += conv.kernel_size[0] // 2
        return radius

    # ------------------------------------------------------------------ assessment
    def statistics(self, x: torch.Tensor) -> torch.Tensor:
        """The analytic SAR assessment statistic stack, shape ``(B, 1 + 2K, H, W)``.

        Computed on the channel mean (the SAR pipeline replicates one measured channel),
        exactly as the front end does, so the two mechanisms read the same kind of signal.
        Exposed separately so a test or a figure can inspect what the scorer sees rather
        than inferring it from the output.
        """
        mean_channels = x.mean(dim=1, keepdim=True).clamp_min(self.eps)
        log = torch.log(mean_channels)
        stats = [mean_channels]
        for pool in self.pools:
            mu = pool(log)
            var = (pool(log * log) - mu * mu).clamp_min(0.0)
            stats.append(log - mu)                              # log-ratio: the CFAR variable
            stats.append(torch.sqrt(var) / (mu.abs() + self.eps))  # local coefficient of variation
        return torch.cat(stats, dim=1)

    def evidence_logit(self, x: torch.Tensor) -> torch.Tensor:
        """Raw evidence logit ``a``, shape ``(B, 1, H, W)``, before the sigmoid.

        In ``fixed`` mode the logit is averaged over the spatial axes, which is what turns
        a per-region decision into one difficulty scalar per image while keeping the
        parameter count *exactly* equal to the proposal's.
        """
        a = self.head(x) if self.mode == "adaptive_raw" else self.head(self.statistics(x))
        if self.mode == "fixed":
            a = a.mean(dim=(2, 3), keepdim=True)
        return a

    def gain_map(self, x: torch.Tensor) -> torch.Tensor:
        """The per-region allocation ``g`` about to be applied, shape ``(B, 1, H, W)``.

        Factored out of :meth:`forward` so the allocation is *measurable* rather than
        inferred: the design's load-bearing claim is that the allocation is a per-region
        decision, so a gate that collapsed to one value per image would make that claim
        empty while still producing a plausible output. Whether that happened is checked
        directly (``saryolo/evaluation/gain.py`` is the front end's version of the same
        diagnostic; SSAC's is asserted in the tests and reported by the runner).
        """
        return torch.sigmoid(self.evidence_logit(x) / self.tau)

    def selected_fraction(self, x: torch.Tensor) -> float:
        """Fraction of locations above the reporting threshold; a diagnostic, not a switch.

        This is the number the efficiency story depends on: if the allocation does not
        actually concentrate on a minority of locations, a sparse implementation would
        have nothing to skip and the whole mechanism is a re-parameterisation.
        """
        with torch.no_grad():
            g = self.gain_map(x)
        return float((g > self.select).float().mean())

    # ------------------------------------------------------------------- execution
    def _tile_scores(self, g: torch.Tensor) -> torch.Tensor:
        """Per-tile mean allocation ``(B, nH*nW)``, in row-major tile order."""
        return torch.nn.functional.avg_pool2d(g, self.tile, self.tile).reshape(g.shape[0], -1)

    def _selected_tiles(self, scores: torch.Tensor) -> torch.Tensor:
        """Indices of the tiles sparse execution refines, ``(B, k)``, sorted.

        ``topk`` returns the k highest-evidence tiles per image; the sort makes the gather
        order a pure function of the scores, so the same input selects the same tiles in
        the same order on every run. At least one tile is always selected: a keep that
        rounds to zero would make the block silently identical to the cheap path while
        still reporting that it had routed.
        """
        n = scores.shape[1]
        k = max(1, min(n, int(math.ceil(self.keep * n))))
        return scores.topk(k, dim=1).indices.sort(dim=1).values

    def routing_stats(self, x: torch.Tensor) -> dict:
        """Routing diagnostics: what a sparse build *would* skip, and at what halo cost.

        ``executed_rich_fraction`` is the share of the expensive path's arithmetic that
        sparse execution performs, halo included: ``k/nTiles * ((tile + 2*halo)/tile)^2``.
        It is arithmetic, not time — the module's latency claim is measured by
        ``scripts/ssac_efficiency.py`` and only there.
        """
        with torch.no_grad():
            g = self.gain_map(x)
            h, w = g.shape[-2:]
            n_tiles = math.ceil(h / self.tile) * math.ceil(w / self.tile)
            k = max(1, min(n_tiles, int(math.ceil(self.keep * n_tiles))))
            halo = (self.tile + 2 * self.halo) / self.tile
        return {
            "tiles": int(n_tiles),
            "selected_tiles": int(k),
            "selected_tile_fraction": round(k / n_tiles, 4),
            "executed_rich_fraction": round(min(1.0, k / n_tiles * halo * halo), 4),
            "halo": int(self.halo),
            "gate_mean": round(float(g.mean()), 5),
            "gate_selected_fraction": round(float((g > self.select).float().mean()), 5),
        }

    def _refine_sparse(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """``self.rich(f)`` computed only on the selected tiles, halo-corrected.

        Rejected tiles come back as ``f`` itself, which is exactly the value the mix in
        :meth:`forward` would assign them at ``g = 0`` — so the caller needs no mask.

        Padding is to a whole number of tiles (bottom/right, with zeros, matching the
        zero padding a dense convolution applies at the map border), and each gathered
        tile carries ``halo`` pixels of context on every side so its receptive field is
        the same one it would have had in the dense pass. That is what makes
        ``keep = 1`` sparse execution reproduce dense execution in ``eval`` mode.
        """
        tile, halo = self.tile, self.halo
        b, c, h, w = f.shape
        n_h, n_w = math.ceil(h / tile), math.ceil(w / tile)
        pad_r, pad_b = n_w * tile - w, n_h * tile - h
        if pad_r or pad_b:
            f = torch.nn.functional.pad(f, (0, pad_r, 0, pad_b))
            g = torch.nn.functional.pad(g, (0, pad_r, 0, pad_b))
        selected = self._selected_tiles(self._tile_scores(g))
        k = selected.shape[1]

        padded = torch.nn.functional.pad(f, (halo, halo, halo, halo))
        # (B, H, W, C) so a tile can be gathered with two spatial index tensors.
        source = padded.permute(0, 2, 3, 1)
        span = tile + 2 * halo
        steps = torch.arange(span, device=f.device)
        rows = (selected // n_w)[:, :, None] * tile + steps
        cols = (selected % n_w)[:, :, None] * tile + steps
        batch_ix = torch.arange(b, device=f.device)[:, None, None, None]
        patches = source[batch_ix, rows[:, :, :, None], cols[:, :, None, :]]
        # -> (B, k, span, span, C): the batch index broadcasts over the tile patch.
        patches = patches.permute(0, 1, 4, 2, 3).reshape(b * k, c, span, span)
        refined = self.rich(patches)[..., halo : halo + tile, halo : halo + tile]
        refined = refined.reshape(b, k, c, tile, tile)

        # Default every tile to the cheap features, then overwrite the selected ones.
        tiles = f.reshape(b, c, n_h, tile, n_w, tile).permute(0, 2, 4, 1, 3, 5).reshape(
            b, n_h * n_w, c, tile, tile
        )
        # ``scatter`` rather than ``index_copy``: the non-inplace variant accepts only a 1-D
        # index, and this scatter is differentiable w.r.t. both operands.
        index = selected[:, :, None, None, None].expand(-1, -1, c, tile, tile)
        tiles = tiles.scatter(1, index, refined)
        out = tiles.reshape(b, n_h, n_w, c, tile, tile).permute(0, 3, 1, 4, 2, 5)
        out = out.reshape(b, c, n_h * tile, n_w * tile)
        return out[..., :h, :w].contiguous()

    # --------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gain_map(x)
        if self.penalty > 0.0:
            # Recorded with its graph intact: the model's loss adds it as a differentiable
            # term. Kept on the module (not as a buffer) because it is a training-scoped
            # activation, not part of the state dict.
            self.last_gate_mean = g.mean()
        cheap = self.cheap(x)
        rich = self._refine_sparse(cheap, g) if self.execution == "sparse" else self.rich(cheap)
        mix = g * rich + (1.0 - g) * cheap
        # Identity at init: alpha == 0, so the whole block is a no-op until it learns.
        return x + self.alpha().to(x.dtype) * (mix - x)

    def extra_repr(self) -> str:
        return (
            f"c1={self.c1}, mode={self.mode}, scales={self.scales}, expand={self.expand}, "
            f"execution={self.execution}, tile={self.tile}, keep={self.keep}"
        )
