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
``"denoise"`` (fixed bilinear low-pass), and the proposed ``"sfm"``.
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

    MODES = ("sfm", "none", "lee", "denoise")
    version = 1

    def __init__(self, c1, mode: str = "sfm", reduction: int = 8, kernel: int = 3, beta_init: float = 0.5):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.c1 = resolve_c1(c1)
        self.mode = mode
        self.stats = LocalStats(k=kernel)

        if mode == "sfm":
            hidden = max(self.c1 // max(reduction, 1), 8)
            self.noise = nn.Sequential(
                ConvBNAct(3 * self.c1, hidden, k=1),
                ConvBNAct(hidden, hidden, k=3),  # spatial context: speckle is spatially correlated
                nn.Conv2d(hidden, self.c1, 1),
            )
            # Noise head starts near "no noise", so the residual begins as a pure target extractor.
            nn.init.zeros_(self.noise[-1].weight)
            nn.init.constant_(self.noise[-1].bias, -2.0)
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
        return x + self.alpha() * (structure - beta * noise_map * x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}"
