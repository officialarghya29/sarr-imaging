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

What this module does **not** do, and says so
---------------------------------------------
It is implemented with *dense* tensor ops: the expensive path runs everywhere and
the allocation changes values, not the amount of arithmetic executed. So on this
CPU implementation the measured FLOPs and latency do **not** fall with the
selected fraction, and the module reports that rather than claiming a saving it
did not make (``docs/ssac_design.md`` §4 states the FLOP-saving variant as the
next step and why it needs a sparse kernel/GPU to be honest). The accuracy
question — Experiment 1 of the master workflow — is fully answerable in this
form, and that is what the prototype is built to test first.
"""

from __future__ import annotations

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
        tau: Temperature of the allocation sigmoid.
        select: Threshold on ``g`` used only for reporting the *selected fraction*; it
            does not change the dense forward pass.
        alpha_init: Initial residual gate (0 => exact identity at init).
        eps: Numerical floor for the log and the ratio denominators.
    """

    MODES = ("adaptive", "adaptive_raw", "fixed")
    version = 1
    #: Every mode gates a residual with a zero-initialised ``alpha``, so every mode is an
    #: exact identity at init. Declared as a dict (not a bool) to mirror ``cfar``'s pattern
    #: and to leave room for a mode that deliberately perturbs the baseline.
    IDENTITY_AT_INIT = {"adaptive": True, "adaptive_raw": True, "fixed": True}

    def __init__(
        self,
        c1,
        source: int | None = None,
        mode: str = "adaptive",
        scales: tuple[int, ...] = (3, 7),
        hidden: int = 16,
        expand: int = 2,
        tau: float = 1.0,
        select: float = 0.5,
        alpha_init: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
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
        self.tau = float(tau)
        self.select = float(select)
        self.expand = int(expand)
        self.eps = float(eps)
        self.identity_at_init = self.IDENTITY_AT_INIT[mode]

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

    # --------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gain_map(x)
        cheap = self.cheap(x)
        rich = self.rich(cheap)
        mix = g * rich + (1.0 - g) * cheap
        # Identity at init: alpha == 0, so the whole block is a no-op until it learns.
        return x + self.alpha().to(x.dtype) * (mix - x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}, scales={self.scales}, expand={self.expand}"
