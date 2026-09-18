"""Component 4 — Adaptive Multi-Scale Fusion (AMF), plus the fusion baselines it is compared against.

Motivation
----------
SAR targets span a wide scale range within a single scene. A standard PAN-FPN
``Concat`` gives every input level the same, fixed influence, so a small,
speckle-obscured target can be drowned out by a large, high-energy neighbour.
AMF is inserted **immediately after each neck ``Concat``** and reweights the
concatenated channel groups before the (stock) projection block that follows.

Formulation
-----------
Let the concatenated input be split into ``G`` groups of sizes ``groups[g]``
(each group being one input branch of the fusion, i.e. one scale)::

    d_g   = Descriptor(F_g)                       # per-group (avg,max,std) descriptor
    w     = softmax(MLP([d_1 ; ... ; d_G]))       # per-sample per-scale weights
    F~    = [ w_1*F_1 ; ... ; w_G*F_G ]           # adaptive re-weighted aggregation
    M     = sigma(dwconv(psi(F~)))                # cross-scale channel gating
    F'    = F + alpha * (F~ * M - F)

``alpha`` is zero-initialised, so AMF is an exact identity at init.

The channel count is preserved, which is what lets the block sit between a
``Concat`` and the stock ``C3k2`` without touching the rest of YAML.

Ablation axis: ``mode`` selects the fusion operator in the same slot.

``"concat"``
    Stock behaviour: the block is present in the graph but returns its input.
``"add"``
    Projected additive fusion: each branch is 1x1-projected to a shared width and
    summed, then projected back to the concatenated width. A *plain* sum is not
    available here because PAN-FPN branches have unequal widths (the top-down
    branch is wider than the backbone branch it is concatenated with), so
    equal-width summation is not defined for this slot. This mirrors how
    residual ``Add`` fusion is implemented whenever widths differ.
``"static"``
    Our block with learnable but input-independent scale weights. Isolates the
    contribution of *adaptivity* from the contribution of having a fusion block
    at all.
``"amf"``
    The proposed input-adaptive fusion.

Note: ``config``/``static``/``amf`` are exact identities at initialisation;
``add`` is an operator replacement and is therefore not.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import ChannelDescriptor, DWConvBNAct, ZeroGate, resolve_c1

__all__ = ["AdaptiveMultiScaleFusion"]


class AdaptiveMultiScaleFusion(nn.Module):
    """Input-adaptive, channel-preserving reweighting of concatenated multi-scale features.

    Args:
        c1: Input channels, or the parser's ``ch`` list (see ``_common`` docstring).
        sources: Layer indices whose outputs were concatenated. When given, the
            group sizes are read from ``c1`` (the parser's ``ch`` list) at build
            time, so the split is correct for any width/scale multiplier.
        reduction: Hidden bottleneck ratio for the per-scale weight MLP.
        mode: ``"amf"`` (proposed), ``"concat"``, ``"add"``, ``"static"``.
        groups: Explicit group sizes, used only when ``sources`` is ``None``.
    """

    MODES = ("amf", "concat", "add", "static")
    version = 1

    def __init__(
        self,
        c1,
        sources=None,
        reduction: int = 8,
        mode: str = "amf",
        groups=None,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")

        if sources is not None:
            if not isinstance(sources, (list, tuple)) or not isinstance(c1, (list, tuple)):
                raise TypeError("With 'sources', c1 must be the parser's 'ch' list.")
            groups = [int(c1[i]) for i in sources]
        if groups is None:
            groups = [resolve_c1(c1)]
        self.groups = list(int(g) for g in groups)
        self.c1 = int(sum(self.groups))
        self.mode = mode

        # Self-check: this block must be wired with `from: -1` (directly after the Concat
        # it reweights), so `ch[-1]` is the concatenated width. A stale `sources` list
        # would otherwise silently build the block at the wrong width. Note that
        # `sources` must also be listed in the same order as the Concat inputs, since
        # only the total width can be validated here.
        if sources is not None:
            expected = resolve_c1(c1)
            if expected != self.c1:
                raise ValueError(
                    f"AdaptiveMultiScaleFusion group sizes {self.groups} sum to {self.c1}, but the "
                    f"incoming feature has {expected} channels. 'sources' must match the Concat inputs "
                    f"and the module must be wired with `from: -1`."
                )

        if mode == "add":
            # Shared width for the projected sum; kept close to the average branch width.
            hidden = max(self.c1 // max(len(self.groups), 1), 8)
            self.proj_in = nn.ModuleList(nn.Conv2d(g, hidden, 1, bias=False) for g in self.groups)
            self.proj_out = nn.Conv2d(hidden, self.c1, 1, bias=False)

        if mode == "amf":
            hidden = max(self.c1 // max(reduction, 1), 8)
            self.desc = ChannelDescriptor()
            self.weight_mlp = nn.Sequential(
                nn.Linear(3 * self.c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, len(self.groups))
            )
            # Start from uniform fusion: all scales contribute equally until training says otherwise.
            nn.init.zeros_(self.weight_mlp[-1].weight)
            nn.init.zeros_(self.weight_mlp[-1].bias)
            self.mix = DWConvBNAct(self.c1, self.c1, k=3)
            self.channel_gate = nn.Sequential(nn.Conv2d(self.c1, self.c1, 1, bias=False))
            nn.init.zeros_(self.channel_gate[0].weight)
            self.alpha = ZeroGate(0.0)
        elif mode == "static":
            # Learnable but input-independent weights: the ablation that isolates adaptivity.
            self.static_logits = nn.Parameter(torch.zeros(len(self.groups)))
            self.mix = DWConvBNAct(self.c1, self.c1, k=3)
            self.channel_gate = nn.Sequential(nn.Conv2d(self.c1, self.c1, 1, bias=False))
            nn.init.zeros_(self.channel_gate[0].weight)
            self.alpha = ZeroGate(0.0)

    def _split(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.groups, dim=1))

    def _reweight(self, x: torch.Tensor) -> torch.Tensor:
        chunks = self._split(x)
        b = x.shape[0]
        if self.mode == "amf":
            w = torch.softmax(self.weight_mlp(self.desc(x)), dim=1)
        else:
            w = torch.softmax(self.static_logits, dim=0).view(1, -1).expand(b, -1)
        return torch.cat([chunks[g] * w[:, g].view(b, 1, 1, 1) for g in range(len(chunks))], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return x

        if self.mode == "add":
            # Each branch projected to a shared width, summed, then projected back to the
            # concatenated width so the block stays channel-preserving.
            chunks = self._split(x)
            summed = sum(proj(chunk) for proj, chunk in zip(self.proj_in, chunks, strict=True))
            return self.proj_out(summed)

        fused = self._reweight(x)
        gated = fused * torch.sigmoid(self.channel_gate(self.mix(fused)))
        return x + self.alpha() * (gated - x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, groups={self.groups}, mode={self.mode}"
