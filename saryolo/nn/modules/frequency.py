"""Component 9 — Spatial-Frequency Representation (SFR).

Motivation
----------
Convolution is a spatial operator with a fixed, local, low-pass-biased basis. SAR
imagery is dominated by two structures that a spatial conv represents poorly:

* **speckle**, which is broadband and (for fully developed speckle) has a
  characteristic spectral signature, and
* **target returns / edges**, which are localised in space and therefore spread
  across the spectrum.

A spatial-only backbone therefore has to spend capacity approximating a frequency
selective operator. This module gives the network an explicit spectral branch and
lets *training* decide how much of the spectrum to keep, instead of assuming that
a particular classical filter (Lee, box low-pass, high-pass) is the right one.

Formulation
-----------
The spectral gain is parameterised as a small number of **radial frequency bands**
rather than one free coefficient per frequency bin::

    F_hat    = rfft2(F)                                  # (B, C, H, W//2+1), complex
    r        = |f| / |f|_nyquist                          # normalised radial frequency per bin
    g_c(b)   = log-gain of channel c in radial band b     # learnable, (C, B)
    g_c(r)   = linear interpolation of g_c over r         # smooth in r, differentiable
    F'       = irfft2(F_hat * exp(g(r)))

Band parameterisation matters for three reasons, all of which are practical:

1. **Parameter count** is ``C x B`` (a few thousand) instead of ``C x H x W/2``
   (millions). A per-bin mask would be almost all of the model's parameters and
   would overfit long before it acted like a filter.
2. **Resolution independence.** Radial bands are defined in *normalised* frequency,
   so a filter learned at 640x640 keeps its meaning at 512 or 1024. A per-bin mask
   would not transfer, and multi-resolution testing (SEC. 27 of the brief) is a
   planned experiment.
3. ``exp(g)`` is strictly positive, so the branch is interpretable as a filter
   magnitude response and cannot flip the sign of a frequency coefficient.

Ablation ladder -- each arm is an exact identity at initialisation::

    "none"     spectral branch disabled.
    "highpass" fixed high-pass response (a classical, non-learned spectral filter),
               applied through the same learned gate.
    "static"   learnable radial bands, but input-independent.
    "sff"      proposed: learnable radial bands *modulated per sample* by the
               feature's own global descriptor, so the filter adapts to the imaging
               regime actually present (low-SNR scene vs bright-clutter scene)
               instead of applying one fixed spectrum to every image.

``"sff"`` starts numerically equal to ``"static"`` (the modulation head is
zero-initialised), so ``sff - static`` measures the value of *input adaptivity*
alone -- the same decomposition the attention module uses with its own static arm.

Implementation notes
--------------------
* ``torch.fft`` has no half-precision CPU kernel, so the spectral path is computed
  in single precision and cast back. Under AMP this is required, not merely tidy:
  an FFT on a float16 activation raises at runtime.
* Where the module sits in the graph is a deliberate choice, not a convenience.
  It is inserted on the **deepest backbone stage** (P5/32), where the feature map
  is smallest: FFT cost grows with spatial resolution, and applying it at P2/4
  would multiply the cost of the cheapest-to-place module by ~64x while adding
  spectral detail the P2 head can already see spatially.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import ChannelDescriptor, ZeroGate, resolve_c1

__all__ = ["SpatialFrequencyRepresentation"]


def _radial_band_index(h: int, w: int, bands: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-band interpolation weights for each rfft2 bin.

    Returns ``(lo, hi, w)``, each flattened to ``(h * (w // 2 + 1),)``, where the gain
    for a bin is ``g[lo] * (1 - w) + g[hi] * w``. Using the two nearest bands (rather
    than hard-assigning bins to bands) keeps the gain a continuous, differentiable
    function of the band parameters, so every band receives gradient.
    """
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)  # cycles/sample in [-0.5, 0.5)
    fx = torch.fft.rfftfreq(w, device=device, dtype=dtype)  # cycles/sample in [0, 0.5]
    # Normalise by the Nyquist radius sqrt(0.5^2 + 0.5^2) so r == 1 sits at the corner.
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / (0.5 * (2.0**0.5))
    pos = (radius.clamp(0.0, 1.0) * (bands - 1)).clamp(0.0, bands - 1.0)
    lo = pos.floor()
    hi = (lo + 1).clamp(max=bands - 1)
    weight = pos - lo
    return lo.flatten().long(), hi.flatten().long(), weight.flatten()


class SpatialFrequencyRepresentation(nn.Module):
    """Learnable radial spectral filter, fused back into the spatial feature.

    Args:
        c1: Input channels, or the parser's ``ch`` list (see ``_common`` docstring).
        mode: ``"sff"`` (proposed), ``"static"``, ``"highpass"``, ``"none"``.
        bands: Number of radial frequency bands. More bands = finer spectral control.
        reduction: Bottleneck ratio for the input-adaptive modulation head.
        alpha_init: Initial residual gate (0 => exact identity at init).
    """

    MODES = ("sff", "static", "highpass", "none")
    #: Bumped whenever the parameter layout changes, so checkpoints can be validated.
    version = 1

    def __init__(self, c1, mode: str = "sff", bands: int = 8, reduction: int = 8, alpha_init: float = 0.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if bands < 2:
            raise ValueError(f"bands must be >= 2 to interpolate between bands, got {bands}")
        self.c1 = resolve_c1(c1)
        self.mode = mode
        self.bands = int(bands)

        if mode == "highpass":
            # Fixed classical response: monotonically increasing gain with radius, i.e.
            # "attenuate the slowly varying background, keep the fine structure".
            # Registered as a buffer so it is part of the state dict but never trained.
            self.register_buffer("band_gain", torch.linspace(-1.0, 1.0, self.bands).view(1, -1))
        elif mode in ("static", "sff"):
            # Small random init, *not* zeros. A zero log-gain gives gain == 1 exactly, so
            # the filtered branch would equal its input and the gate gradient
            # dL/dalpha = <dL/dout, y - x> would be identically zero: the spectral branch
            # would never switch on. Exact identity at init is provided by the gate alone.
            # See tests/test_arch.py::test_no_module_is_frozen_at_init.
            self.band_gain = nn.Parameter(torch.randn(self.c1, self.bands) * 1e-2)

        if mode == "sff":
            hidden = max(self.c1 // max(reduction, 1), 8)
            self.desc = ChannelDescriptor()
            self.modulate = nn.Sequential(
                nn.Linear(3 * self.c1, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, self.bands)
            )
            # Zero-init the modulation so `sff` begins exactly where `static` begins; the
            # only thing training can add is input-dependent spectral adaptation.
            nn.init.zeros_(self.modulate[-1].weight)
            nn.init.zeros_(self.modulate[-1].bias)

        self.alpha = ZeroGate(alpha_init)

    # ------------------------------------------------------------------- spectrum
    def spectral_gain(self, x: torch.Tensor) -> torch.Tensor:
        """Per-bin spectral gain, shaped ``(1, C, H, W//2+1)`` (or ``(B, C, H, W//2+1)``)."""
        h, w = x.shape[-2:]
        lo, hi, weight = _radial_band_index(h, w, self.bands, x.device, torch.float32)

        # highpass: (1, bands), broadcast over channels.  static: (C, bands).
        gain = self.band_gain
        if self.mode == "sff":
            # Per-sample spectral adaptation: (C, bands) + (B, 1, bands) -> (B, C, bands).
            gain = gain.unsqueeze(0) + self.modulate(self.desc(x)).unsqueeze(1)
        if gain.dim() == 2:
            gain = gain.unsqueeze(0)  # (1, S, bands)

        g_lo = gain.index_select(-1, lo)  # (..., S, H*W/2)
        g_hi = gain.index_select(-1, hi)
        g = g_lo * (1.0 - weight) + g_hi * weight
        return torch.exp(g.clamp(-6.0, 6.0)).view(*gain.shape[:-1], h, w // 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x

        # torch.fft has no half-precision CPU kernel; computing in fp32 is required for
        # AMP correctness rather than a numerical nicety.
        compute_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        xh = x.to(compute_dtype)
        spectrum = torch.fft.rfft2(xh, norm="ortho")
        filtered = torch.fft.irfft2(spectrum * self.spectral_gain(xh), s=xh.shape[-2:], norm="ortho")
        y = filtered.to(x.dtype)
        return x + self.alpha() * (y - x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}, bands={self.bands}"
