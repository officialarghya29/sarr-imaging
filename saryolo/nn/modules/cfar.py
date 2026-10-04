"""Ratio-space CFAR front end — the SARVO prototype (master Phase 5).

Motivation
----------
A SAR image is a coherent measurement, and the failure that matters for small
targets is not noise but *clutter*: the local mean of a sea surface, a harbour
wall or a field is a slowly varying reflector whose amplitude can exceed a weak
target's return. The classical answer to this is not denoising, it is the
constant-false-alarm-rate (CFAR) statistic: compare each cell against the
statistics of a ring of surrounding cells and threshold the *ratio*. Because the
comparison is a ratio, the threshold transfers between scenes whose absolute
brightness differs, which is exactly what a detector operating across
acquisitions needs.

What this module is
-------------------
The detector's first representation, replacing "a convolution reads the raw
magnitude image and re-derives a local-contrast statistic over its first layers"
with "the radar's own statistic is computed analytically, and a learned gain
decides per pixel how much of it to use".

Formulation. For an input ``x`` (B, C, H, W) with ``C`` channels that share a
sensor (the SAR pipeline replicates one measured channel, so the statistic is
computed on the channel mean)::

    m     = mean_c(x)                        # the measured intensity
    L     = log(m + eps)                     # multiplicative speckle -> additive
    mu_k  = box_k(L)                         # local mean at scale k
    var_k = box_k(L^2) - mu_k^2              # local variance
    r_k   = L - mu_k                         # log-ratio (dimensionless contrast)
    c_k   = sqrt(var_k) / (|mu_k| + eps)     # local coefficient of variation
    s     = [m, r_1, c_1, ..., r_K, c_K]     # the statistic stack
    g     = tanh(MLP(s))                     # learned, per-pixel, in (-1, 1)
    out   = x * (1 + g)

The last layer of the MLP is zero-initialised and has a zero bias, so ``g == 0``
and ``out == x`` *bit-for-bit* at initialisation: the module is an exact identity
until it has learned something. That preserves the property the whole ablation
rests on — a measured gain is attributable to the statistic, not to extra
capacity that was active from step 0.

Why the log-ratio and the coefficient of variation
--------------------------------------------------
``r_k`` is the CFAR decision variable itself, so a learned gain over ``r_k`` is a
learned threshold. ``c_k`` is the classical texture statistic: for fully
developed speckle its square is the reciprocal of the equivalent number of looks
(ENL), so a region whose ``c`` is high relative to its scale is textured rather
than a point return. Both are dimensionless, which is what makes the same
learned gain meaningful across scenes of different absolute brightness.

Modulation is applied to ``x``, not to ``L``. Two reasons: inverting the log to
return to image space would make the identity approximate rather than exact, and
the backbone was designed to read an intensity image — feeding it log values
would change the input distribution by more than the statistic does.

Modes and their controls
------------------------
``MODES`` declares both the proposal and the control the proposals document names
as the experiment that could kill it:

* ``"cfar"`` — the proposed arm. Learned, zero-initialised per-pixel gain.
* ``"fixed"`` — the *fixed-threshold* control: the same statistic stack, but the
  gain is an analytic threshold with **no learnable parameters**. It separates
  "the statistics help" from "the learnable part helps". It is deliberately not
  an identity at initialisation (a control that equals the baseline measures
  nothing), and it therefore declares ``identity_at_init = False``.

The matched-cost *plain-conv-stem* control lives in the architecture, not here:
it is a different first row of the same graph, so it is built as a separate
variant rather than a mode of this module.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import resolve_c1

__all__ = ["RatioSpaceCFARFrontEnd"]


class RatioSpaceCFARFrontEnd(nn.Module):
    """Analytic multi-scale CFAR statistic as the detector's first representation.

    Channel-preserving (``C -> C``) and an exact identity at initialisation in the
    proposed mode, so it can be inserted as row 0 of a YOLO-family graph without
    the parser or the first convolution noticing.

    Args:
        c1: Input channels, or the parser's ``ch`` list.
        mode: ``"cfar"`` (proposed) or ``"fixed"`` (no-parameter control).
        scales: Local-statistics window sizes. Deliberately geometric (3, 7, 15):
            the ratio statistic is only meaningful relative to a window that is
            large compared with the target and small compared with the scene, so
            the arms span more than an octave.
        hidden: Width of the gain MLP. Small on purpose — the whole module's
            parameter cost is well under the 0.5 % budget the repository already
            holds conditioning to.
        eps: Numerical floor for the log and the ratio denominators.
    """

    MODES = ("cfar", "fixed")
    version = 1
    #: Whether a freshly built module in this mode is an exact identity. ``False``
    #: for the fixed-threshold control, which exists precisely to perturb the
    #: baseline; the identity invariant is asserted only where it is claimed.
    IDENTITY_AT_INIT = {"cfar": True, "fixed": False}

    def __init__(
        self,
        c1,
        mode: str = "cfar",
        scales: tuple[int, ...] = (3, 7, 15),
        hidden: int = 24,
        eps: float = 1e-5,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if not scales:
            raise ValueError("scales must not be empty; a CFAR statistic needs at least one window")
        if any(k < 3 or k % 2 == 0 for k in scales):
            raise ValueError(f"scales must be odd and >= 3, got {scales!r}")
        self.c1 = resolve_c1(c1)
        self.mode = mode
        self.scales = tuple(int(k) for k in scales)
        self.eps = float(eps)
        self.identity_at_init = self.IDENTITY_AT_INIT[mode]

        self.pools = nn.ModuleList(
            nn.AvgPool2d(k, stride=1, padding=k // 2, count_include_pad=False) for k in self.scales
        )
        n_stats = 1 + 2 * len(self.scales)
        if mode == "cfar":
            self.gain = nn.Sequential(
                nn.Conv2d(n_stats, hidden, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, 1),
            )
            # Exact identity at init: g == 0, so out == x bit-for-bit.
            nn.init.zeros_(self.gain[-1].weight)
            nn.init.zeros_(self.gain[-1].bias)

    def statistics(self, x: torch.Tensor) -> torch.Tensor:
        """The CFAR statistic stack, computed analytically (no learned parameters).

        Exposed separately so a test or a figure can inspect the representation the
        gain actually sees, rather than inferring it from the output.
        """
        mean_channels = x.mean(dim=1, keepdim=True).clamp_min(self.eps)
        log = torch.log(mean_channels)
        stats = [mean_channels]
        for pool in self.pools:
            mu = pool(log)
            var = (pool(log * log) - mu * mu).clamp_min(0.0)
            stats.append(log - mu)  # log-ratio: the CFAR decision variable
            stats.append(torch.sqrt(var) / (mu.abs() + self.eps))  # local coefficient of variation
        return torch.cat(stats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.statistics(x)
        if self.mode == "cfar":
            gain = torch.tanh(self.gain(stats))
        else:
            # Fixed-threshold control: threshold the mean local contrast at a
            # constant and suppress where the region looks textured rather than
            # point-like. No parameters, so nothing here can be attributed to
            # learning.
            contrast = stats[:, 2::2].mean(dim=1, keepdim=True)  # the c_k channels
            ratio = stats[:, 1::2].mean(dim=1, keepdim=True)  # the r_k channels
            gain = torch.tanh(2.0 * (ratio - contrast).clamp(-1.0, 1.0))
        return x * (1.0 + gain)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}, scales={self.scales}"
