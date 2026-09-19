"""Component 1 — SAR Feature Enhancement (SFE).

Motivation
----------
SAR returns are multiplicative-speckle dominated and often low contrast, so raw
intensity features under-represent target structure. Aggressive denoising is the
wrong fix: it removes the very high-frequency evidence (strong point returns,
dihedral edges) that detectors rely on. SFE instead *re-expresses* the feature
through a local contrast normalisation and mixes it back through a gated
residual, so enhancement is additive and reversible rather than destructive.

Formulation
-----------
Given a feature ``F`` with local mean ``mu`` and local std ``sd`` over a k x k
window::

    S = (F - mu) / (sd + eps)        # local-contrast-normalised structure
    Z = phi([F ; S ; sd])            # learned 1x1 mixing of raw, structure, speckle-scale
    F' = F * gamma + beta + alpha * Z

``gamma``/``beta`` are per-channel affine corrections (learnable), ``alpha`` is a
tanh-bounded zero-initialised gate. At init ``gamma = 1``, ``beta = 0``,
``alpha = 0``, hence ``F' == F`` exactly.

Ablation axis: ``mode`` selects the feature-improvement variant. ``"sfe"`` is the
proposed block; ``"identity"``, ``"log"``, ``"standardize"`` and ``"clahe"`` are
the preprocessing-style alternatives compared against it in the paper's
preprocessing table, kept in the *same* slot and under the same init contract so
the comparison isolates the mechanism rather than the architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import ConvBNAct, LocalStats, ZeroGate, resolve_c1

__all__ = ["SARFeatureEnhancement"]


class SARFeatureEnhancement(nn.Module):
    """Learnable local-contrast feature enhancement with a zero-initialised residual gate.

    Args:
        c1: Input channels, or the parser's ``ch`` list (see ``_common`` docstring).
        mode: Enhancement variant, one of
            ``"sfe"`` (proposed), ``"identity"``, ``"log"``, ``"standardize"``, ``"clahe"``.
        kernel: Local-statistics window size.
        reduction: Hidden bottleneck ratio for the mixing conv.
        alpha_init: Initial value of the residual gate (0 => exact identity at init).
    """

    MODES = ("sfe", "identity", "log", "standardize", "clahe")
    #: Bumped whenever the parameter layout changes, so checkpoints can be validated.
    version = 1

    def __init__(self, c1, mode: str = "sfe", kernel: int = 3, reduction: int = 8, alpha_init: float = 0.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.c1 = resolve_c1(c1)
        self.mode = mode
        self.stats = LocalStats(k=kernel)

        if mode == "sfe":
            hidden = max(self.c1 // max(reduction, 1), 8)
            self.reduce = ConvBNAct(3 * self.c1, hidden, k=1)
            self.expand = ConvBNAct(hidden, self.c1, k=1, act=False)
            # The residual branch is deliberately NOT zero-initialised. Exact identity at
            # init comes from the gate alone (alpha = 0 => F' = F + 0*z), and that is
            # *sufficient*. Zero-initialising z as well would make the branch permanently
            # dead rather than merely quiet: with z == 0 the gate gradient is
            # dL/dalpha = <dL/dout, z> = 0, so alpha can never leave 0, and dL/dz =
            # alpha*dL/dout = 0 freezes the branch too. Both quantities vanish together.
            # See tests/test_arch.py::test_no_module_is_frozen_at_init.
            self.gamma = nn.Parameter(torch.ones(1, self.c1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.c1, 1, 1))
            self.alpha = ZeroGate(alpha_init)
        elif mode == "clahe":
            # Learnable global gain + local contrast, i.e. a differentiable CLAHE analogue.
            self.gamma = nn.Parameter(torch.ones(1, self.c1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.c1, 1, 1))
            self.clip = ZeroGate(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return x

        mu, var, sd, hp = self.stats(x)

        if self.mode == "log":
            # Log compression: standard SAR dynamic-range reduction.
            # Guarded to stay finite for near-zero/negative activations.
            return torch.sign(x) * torch.log1p(x.abs().clamp_min(0.0))

        if self.mode == "standardize":
            return (x - mu) / sd

        if self.mode == "clahe":
            structure = (x - mu) / sd
            clip = self.clip() * 2.0
            structure = structure.clamp(-1.0 - clip, 1.0 + clip)
            refined = (x - mu) * structure + mu
            return refined * self.gamma + self.beta

        # Proposed SFE.
        structure = (x - mu) / sd
        z = self.expand(self.reduce(torch.cat((x, structure, sd), dim=1)))
        return x * self.gamma + self.beta + self.alpha() * z

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}"
