"""Component 3 — SAR-Adaptive Attention (SAA) and the attention baselines it is compared against.

Motivation
----------
Existing attention blocks (SE, ECA, CBAM) fuse their branches with *fixed*
weights that are identical for every input. In SAR this is a poor fit: whether a
region should be emphasised depends on the imaging regime actually present —
a low-SNR scene needs more spatial/contrast weighting, a bright-clutter scene
needs more channel selectivity. SAA therefore computes channel, spatial and
local-contrast evidence separately and then fuses them with **per-sample**
weights predicted from the feature's own global statistics.

Formulation
-----------
::

    a = sigma(MLP(Descriptor(F)))            # channel attention       (B, C, 1, 1)
    s = sigma(dwconv([mean_c F ; max_c F]))  # spatial attention       (B, 1, H, W)
    l = sigma(psi([F ; (F-mu)/sd]))          # local-contrast evidence (B, 1, H, W)
    w = softmax(MLP(Descriptor(F)))          # adaptive branch weights (B, 3)
    M = w0*a + w1*s + w2*l                   # broadcast fused attention
    F' = F + alpha * (F * M - F)

All alternatives share the ``F + alpha*(F*M - F)`` residual form and the same
``alpha = 0`` initialisation, so every block is an exact identity at init and the
comparison isolates the attention mechanism itself.

Branch weights are *per-sample* and *content-dependent*, which is exactly the
property the ``static`` ablation removes: ``gate="static"`` replaces the
predicted weights with learned constants shared across the batch, isolating the
contribution of adaptivity from the contribution of having three branches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import ChannelDescriptor, ConvBNAct, LocalStats, ZeroGate, resolve_c1

__all__ = [
    "SARAdaptiveAttention",
    "SEAttention",
    "ECAAttention",
    "CBAMAttention",
    "IdentityAttention",
    "build_attention",
]


class _ResidualAttention(nn.Module):
    """Shared ``F + alpha * (F*M - F)`` wrapper giving every attention variant identity init."""

    def __init__(self, c1: int, alpha_init: float = 0.0):
        super().__init__()
        self.c1 = c1
        self.alpha = ZeroGate(alpha_init)

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.attention_map(x)
        return x + self.alpha() * (x * m - x)


class IdentityAttention(_ResidualAttention):
    """No-attention control: returns the feature untouched."""

    def __init__(self, c1, source: int | None = None, **kwargs):
        super().__init__(resolve_c1(c1, source))

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


class SEAttention(_ResidualAttention):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018) — channel-only baseline."""

    def __init__(self, c1, source: int | None = None, reduction: int = 16, **kwargs):
        c1 = resolve_c1(c1, source)
        super().__init__(c1)
        hidden = max(c1 // max(reduction, 1), 4)
        self.mlp = nn.Sequential(nn.Linear(2 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, c1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.desc = ChannelDescriptor()

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        flat = x.flatten(2)
        desc = torch.cat((flat.mean(-1), flat.amax(-1)), dim=1)
        w = torch.sigmoid(self.mlp(desc)).view(b, c, 1, 1)
        return w.expand_as(x)


class ECAAttention(_ResidualAttention):
    """ECA-Net (Wang et al., CVPR 2020) — 1D-conv channel baseline."""

    def __init__(self, c1, source: int | None = None, kernel: int = 3, **kwargs):
        c1 = resolve_c1(c1, source)
        super().__init__(c1)
        k = kernel if kernel % 2 else kernel + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = torch.sigmoid(self.conv(pooled.unsqueeze(1)).squeeze(1)).view(b, c, 1, 1)
        return w.expand_as(x)


class CBAMAttention(_ResidualAttention):
    """CBAM (Woo et al., ECCV 2018) — sequential channel then spatial baseline."""

    def __init__(self, c1, source: int | None = None, reduction: int = 16, kernel: int = 7, **kwargs):
        c1 = resolve_c1(c1, source)
        super().__init__(c1)
        hidden = max(c1 // max(reduction, 1), 4)
        self.mlp = nn.Sequential(nn.Linear(2 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, c1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        nn.init.zeros_(self.spatial.weight)

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        flat = x.flatten(2)
        desc = torch.cat((flat.mean(-1), flat.amax(-1)), dim=1)
        ch = torch.sigmoid(self.mlp(desc)).view(b, c, 1, 1)
        xc = x * ch
        sm = torch.cat((xc.mean(1, keepdim=True), xc.amax(1, keepdim=True)), dim=1)
        sp = torch.sigmoid(self.spatial(sm))
        return ch * sp


class SARAdaptiveAttention(_ResidualAttention):
    """Proposed SAR-Adaptive Attention: channel + spatial + local-contrast, adaptively fused.

    Args:
        c1: Input channels, or the parser's ``ch`` list.
        source: Index of the layer this block consumes. Required when the YAML row's
            ``from`` is an explicit index (which it is here — attention is applied per
            detection level, not to the previous row); see ``_common.resolve_c1``.
        reduction: Hidden bottleneck ratio for the channel/adaptive MLPs.
        kernel: Local-statistics window (local-contrast branch) and spatial conv kernel.
        gate: ``"adaptive"`` (proposed, per-sample softmax weights) or
            ``"static"`` (learned constants; the ablation that isolates adaptivity).
    """

    version = 1

    def __init__(self, c1, source: int | None = None, reduction: int = 16, kernel: int = 7,
                 gate: str = "adaptive", **kwargs):
        c1 = resolve_c1(c1, source)
        super().__init__(c1)
        if gate not in ("adaptive", "static"):
            raise ValueError(f"gate must be 'adaptive' or 'static', got {gate!r}")
        self.gate_mode = gate
        hidden = max(c1 // max(reduction, 1), 4)

        # --- channel branch: second-order descriptor (avg/max/std) -> gates ---
        self.channel_branch = nn.Sequential(
            nn.Linear(3 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, c1)
        )
        nn.init.zeros_(self.channel_branch[-1].weight)
        nn.init.zeros_(self.channel_branch[-1].bias)

        # --- spatial branch: order statistics + local contrast, large-kernel depth-wise conv ---
        self.local = LocalStats(k=3)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(4, max(c1 // max(reduction, 1), 4), kernel, padding=kernel // 2, groups=1, bias=False),
            nn.BatchNorm2d(max(c1 // max(reduction, 1), 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c1 // max(reduction, 1), 4), 1, 1, bias=False),
        )
        nn.init.zeros_(self.spatial_branch[-1].weight)

        # --- local-contrast branch: SAR-specific structure map ---
        self.contrast_branch = nn.Sequential(
            ConvBNAct(2, max(c1 // max(reduction, 1), 4), k=3),
            nn.Conv2d(max(c1 // max(reduction, 1), 4), 1, 1, bias=False),
        )
        nn.init.zeros_(self.contrast_branch[-1].weight)

        # --- adaptive fusion of the three branch maps ---
        self.desc = ChannelDescriptor()
        if gate == "adaptive":
            self.gate_mlp = nn.Sequential(nn.Linear(3 * c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, 3))
            # Neutral start: all three branches weighted equally until training disagrees.
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
        else:
            self.static_logits = nn.Parameter(torch.zeros(3))

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]

        # channel branch
        a = torch.sigmoid(self.channel_branch(self.desc(x))).view(b, c, 1, 1)

        # spatial + local-contrast branches
        mu, var, sd, hp = self.local(x)
        stats = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True),
                           var.mean(1, keepdim=True), sd.mean(1, keepdim=True)), dim=1)
        s = torch.sigmoid(self.spatial_branch(stats))
        # Local-contrast evidence is reduced to two spatial maps (mean absolute structure
        # and mean absolute high-pass energy) so the branch stays cheap and single-channel,
        # matching the shape of the channel/spatial branch maps it is fused with.
        structure = (x - mu) / sd
        contrast = torch.cat((structure.abs().mean(1, keepdim=True), hp.abs().mean(1, keepdim=True)), dim=1)
        local_map = torch.sigmoid(self.contrast_branch(contrast))

        # adaptive branch weighting
        if self.gate_mode == "adaptive":
            w = torch.softmax(self.gate_mlp(self.desc(x)), dim=1)  # (B, 3)
        else:
            w = torch.softmax(self.static_logits, dim=0).view(1, 3).expand(b, 3)
        w = w.view(b, 3, 1, 1)

        m = w[:, 0:1] * a + w[:, 1:2] * s + w[:, 2:3] * local_map
        return m

    def extra_repr(self) -> str:
        return f"c1={self.c1}, gate={self.gate_mode}"


#: Registry used by the architecture builder and the module-level ablation table.
ATTENTION_BUILDERS = {
    "none": IdentityAttention,
    "se": SEAttention,
    "eca": ECAAttention,
    "cbam": CBAMAttention,
    "saa": SARAdaptiveAttention,
    "saa_static": lambda c1, source=None, **kw: SARAdaptiveAttention(c1, source, gate="static", **kw),
}


def build_attention(name: str, c1, source: int | None = None, **kwargs) -> nn.Module:
    """Instantiate an attention variant by name (used to emit the module-level ablation YAMLs)."""
    if name not in ATTENTION_BUILDERS:
        raise KeyError(f"Unknown attention '{name}'. Known: {sorted(ATTENTION_BUILDERS)}")
    return ATTENTION_BUILDERS[name](c1, source, **kwargs)
