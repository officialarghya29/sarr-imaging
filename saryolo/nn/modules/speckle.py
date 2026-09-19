"""Component 2 — Speckle-Aware Feature Module (SFM).

Motivation
----------
Speckle is signal-dependent multiplicative noise: it scales with local
reflectivity, so it cannot be removed by a fixed filter without also attenuating
the bright target returns that carry the detection signal. What a detector
actually needs is not a *denoised* feature but the ability to tell "this
high-variance response is a target" apart from "this high-variance response is
speckle/clutter".

Formulation
-----------
Following the hypothesis in the project brief, SFM estimates a per-channel
speckle-likelihood map ``N`` and a learned target-response map ``T``::

    mu, var, hp = LocalStats(F)
    N = sigma(psi([F ; hp ; var]))        # speckle / clutter likelihood in [0,1]
    T = dwconv(F)                          # structure-preserving target extractor
    F' = F + alpha * (T - beta * N * F)

with ``beta = sigmoid(...)`` (so it stays in ``(0,1)`` and cannot flip sign) and
``alpha`` a tanh-bounded zero-initialised gate. At init ``F' == F``.

The subtraction term is deliberately *multiplicative in the input*: speckle is
signal dependent, so the amount of suppression must scale with local intensity
rather than being a constant offset.

Ablation axis: ``mode`` swaps in the classical alternatives the paper compares
against, in the same architectural slot: ``"none"`` (no handling), ``"lee"``
(adaptive local-mean filter — the classical SAR despeckling baseline),
``"denoise"`` (fixed bilinear low-pass), ``"sfm"`` (the proposed speckle-aware
module), and ``"sfm_clutter"`` (SFM extended with a separate clutter branch, see
below).

Speckle is not clutter
----------------------
Speckle is a *fine-scale*, signal-dependent interference pattern; clutter is a
*coarse-scale* structure — terrain, sea state, harbour infrastructure, land/water
boundaries. A single noise estimator conflates them, which is a problem because
one should be suppressed while the other should be understood. ``"sfm_clutter"``
gives each its own estimator at its own scale (near-window statistics for speckle,
wide-window statistics for clutter) and its own suppression strength::

    N = sigma(psi_near([F ; hp_near ; var_near]))      # speckle likelihood
    C = sigma(psi_wide([F ; hp_wide ; sd_wide]))       # clutter likelihood
    F' = F + alpha * (T - beta * N * F - gamma * C * F)

``gamma`` is a ``sigmoid``-bounded learned scalar, for the same reason ``beta`` is:
a suppression weight that can change sign would turn suppression into enhancement
for the very structure it is meant to remove.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import ConvBNAct, DWConvBNAct, LocalStats, ZeroGate, resolve_c1

__all__ = ["SpeckleAwareFeatureModule"]


class SpeckleAwareFeatureModule(nn.Module):
    """Separates target structure from speckle/clutter responses and reweights them.

    Args:
        c1: Input channels, or the parser's ``ch`` list.
        mode: ``"sfm"`` (proposed), ``"none"``, ``"lee"``, ``"denoise"``.
        reduction: Hidden bottleneck ratio in the noise estimator.
        kernel: Local-statistics window size.
        beta_init: Initial ``beta`` logit; ``sigmoid(0.5) ~= 0.62``.
    """

    MODES = ("sfm", "sfm_clutter", "none", "lee", "denoise")
    version = 1

    def __init__(self, c1, mode: str = "sfm", reduction: int = 8, kernel: int = 3, beta_init: float = 0.5):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.c1 = resolve_c1(c1)
        self.mode = mode
        self.stats = LocalStats(k=kernel)

        if mode in ("sfm", "sfm_clutter"):
            hidden = max(self.c1 // max(reduction, 1), 8)
            self.noise = nn.Sequential(
                ConvBNAct(3 * self.c1, hidden, k=1),
                ConvBNAct(hidden, hidden, k=3),  # spatial context: speckle is spatially correlated
                nn.Conv2d(hidden, self.c1, 1),
            )
            # Noise head starts near "no noise", so the residual begins as a pure target extractor.
            # (This is safe against the frozen-branch failure mode because the head is not the only
            # term in the residual: `structure` is non-degenerate at init, so the gate still
            # receives a gradient. See tests/test_arch.py::test_no_module_is_frozen_at_init.)
            nn.init.zeros_(self.noise[-1].weight)
            nn.init.constant_(self.noise[-1].bias, -2.0)
            if mode == "sfm_clutter":
                # Clutter is a coarse-scale phenomenon, so its estimator looks at a wide
                # window instead of the near window used for speckle.
                self.wide = LocalStats(k=2 * kernel + 1)
                self.clutter = nn.Sequential(
                    ConvBNAct(3 * self.c1, hidden, k=1),
                    ConvBNAct(hidden, hidden, k=3),
                    nn.Conv2d(hidden, self.c1, 1),
                )
                nn.init.zeros_(self.clutter[-1].weight)
                nn.init.constant_(self.clutter[-1].bias, -2.0)
                self.gamma = nn.Parameter(torch.tensor(0.5))
            self.structure = DWConvBNAct(self.c1, self.c1, k=3)
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
            self.alpha = ZeroGate(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x

        if self.mode in ("lee", "denoise"):
            # Classical baselines: a local-mean (Lee-style) / box low-pass smoothing.
            mu = self.stats.pool(x)
            if self.mode == "lee":
                _, var, _, _ = self.stats(x)
                # Lee filter: F + k * (mu - F), k = var / (var + noise_var)
                noise_var = var.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
                k = (var / (var + noise_var)).clamp(0.0, 1.0)
                return x + k * (mu - x)
            # Fixed low-pass: blend toward the local mean.
            return 0.5 * x + 0.5 * mu

        # Proposed SFM.
        mu, var, _, hp = self.stats(x)
        noise_map = torch.sigmoid(self.noise(torch.cat((x, hp, var), dim=1)))
        structure = self.structure(x)
        beta = torch.sigmoid(self.beta)
        suppressed = structure - beta * noise_map * x
        if self.mode == "sfm_clutter":
            _, _, sd_w, hp_w = self.wide(x)
            clutter_map = torch.sigmoid(self.clutter(torch.cat((x, hp_w, sd_w), dim=1)))
            suppressed = suppressed - torch.sigmoid(self.gamma) * clutter_map * x
        return x + self.alpha() * suppressed

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}"
