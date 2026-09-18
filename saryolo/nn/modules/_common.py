"""Shared building blocks for SAR-YOLO neural modules.

Design contract for every SAR-YOLO neck module
-----------------------------------------------
A module inserted into a YOLO YAML must satisfy three constraints, otherwise it
cannot be dropped into ``ultralytics.nn.tasks.parse_model`` without patching it:

1. **Single input, single output.** ``parse_model`` resolves output channels with
   ``c2 = ch[f]`` for unknown modules, which breaks if ``f`` is a list.
2. **Channel preserving.** Because (1) reports ``c2 == c1`` to the parser, the
   module *must* return the same number of channels it received. Projection back
   to a new width is left to the stock ``Conv``/``C3k2`` that follows.
3. **Identity at initialisation.** Every residual gate starts at ``alpha = 0``,
   so a freshly built SAR-YOLO is numerically *identical* to its YOLO baseline.
   Any improvement is therefore attributable to learned SAR-specific behaviour
   rather than to extra capacity at init — this is what makes the ablation table
   in the paper defensible.

Channel arguments
-----------------
``parse_model`` substitutes any string argument that names a local variable, and
``ch`` (the running channel list) is one of them. Passing ``"ch"`` therefore
hands the module the channel list *as it stands at that layer*, letting us
resolve width-scaled channels exactly at build time for any ``scale``/``width_multiple``.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "resolve_c1",
    "ConvBNAct",
    "DWConvBNAct",
    "LocalStats",
    "ChannelDescriptor",
    "ZeroGate",
]


def resolve_c1(c1: int | list | tuple, source: int | None = None) -> int:
    """Resolve the input channel count from an explicit int or the parser's ``ch`` list.

    ``parse_model`` passes ``"ch"`` through verbatim (see module docstring), so
    modules must accept a list of previously-produced channel counts.

    Args:
        c1: Explicit channel count, or the parser's ``ch`` list.
        source: Index into ``ch`` for the layer this module actually consumes.
            Required whenever the YAML row's ``from`` is an explicit index rather
            than ``-1``: ``ch[-1]`` is the most recently built layer, which is a
            different layer entirely in that case. Passing the wrong entry here
            silently builds the module at the wrong width, so callers that read
            ``ch`` must always pass ``source`` when ``from != -1``.
    """
    if isinstance(c1, (list, tuple)):
        if len(c1) == 0:
            raise ValueError("Received an empty channel list; cannot resolve input channels.")
        if source is None:
            return int(c1[-1])
        if not -len(c1) <= source < len(c1):
            raise IndexError(f"source index {source} is outside the channel list of length {len(c1)}")
        return int(c1[source])
    if isinstance(c1, bool):  # bool is an int subclass; reject it explicitly
        raise TypeError("Channel count must be an int, got bool.")
    if isinstance(c1, int):
        return c1
    raise TypeError(f"Channel count must be int or list, got {type(c1).__name__}.")


class ConvBNAct(nn.Module):
    """Standard Conv2d -> BatchNorm2d -> SiLU block (self-contained; no ultralytics coupling)."""

    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConvBNAct(nn.Module):
    """Depth-wise separable conv: cheap structure extraction that preserves spatial detail."""

    def __init__(self, c_in: int, c_out: int | None = None, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        c_out = c_in if c_out is None else c_out
        self.dw = nn.Conv2d(c_in, c_in, k, s, k // 2, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class LocalStats(nn.Module):
    """Local first/second-order statistics of a feature map.

    SAR imagery is multiplicative-speckle dominated: the local mean tracks the
    slowly varying scene reflectivity while the local standard deviation tracks
    speckle strength. ``(x - mu) / sd`` is therefore a local contrast
    normalisation that equalises background clutter while preserving the
    structure of bright target returns — the core SAR-specific primitive shared
    by the enhancement, speckle and attention modules.
    """

    def __init__(self, k: int = 3, eps: float = 1e-5):
        super().__init__()
        self.pool = nn.AvgPool2d(k, stride=1, padding=k // 2, count_include_pad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mu, var, sd, highpass)``."""
        mu = self.pool(x)
        var = (self.pool(x * x) - mu * mu).clamp_min(0.0)
        sd = torch.sqrt(var + self.eps)
        return mu, var, sd, x - mu


class ChannelDescriptor(nn.Module):
    """Global channel descriptor (avg, max, std) used for input-adaptive gating.

    ``std`` is included because speckle strength is a second-order statistic:
    a branch weighting that can see only mean/max information cannot distinguish
    a uniformly bright clutter region from a high-variance target region.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        flat = x.flatten(2)
        avg = flat.mean(-1)
        mx = flat.amax(-1)
        std = flat.std(-1, unbiased=False)
        return torch.cat((avg, mx, std), dim=1).view(b, 3 * c)


class ZeroGate(nn.Module):
    """Learnable scalar residual gate, bounded to ``(-1, 1)`` and initialised at 0.

    Identity initialisation is what keeps a SAR-YOLO build bit-for-bit
    equivalent to its baseline before training (see module docstring).
    """

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(float(init)))

    def forward(self) -> torch.Tensor:
        return torch.tanh(self.raw)

    def extra_repr(self) -> str:
        return f"gate={float(torch.tanh(self.raw)):+.4f}"
