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
    "dct"      8x8 block DCT-II with learnable radial bands inside the block. A
               *local* transform: it sees 8x8 tiles rather than the whole map.
    "wavelet"  one-level Haar analysis with learnable per-sub-band gains. Localise in
               both space and frequency (the four sub-bands are octave-scale bands).

The last two exist to answer SEC. 14 of the brief, which asks for the alternative
frequency study explicitly (FFT vs DCT vs Wavelet vs learned) rather than assuming the
FFT is the right transform. They are kept in this class rather than as separate module
types because they occupy the *same* slot and the comparison is only meaningful if
nothing else changes; the transform is the variable, not the wiring.

Why the two alternatives are worth measuring rather than dismissing:

* The FFT arm is **global**: one coefficient per frequency for the entire map, so a
  filter that is right for a ship cannot differ from one for a wake elsewhere in the
  same image. A block DCT and a Haar decomposition are **local**, which is the usual
  reason SAR despeckling is done in the wavelet domain: speckle is local in space.
* The cost is the other side of that argument: a block transform assumes stationarity
  inside each block and introduces block boundaries, and the Haar basis is fixed rather
  than learned, so it can only reweight four octave bands per level.

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

#: Block edge of the DCT arm. Fixed at the JPEG-era size so the basis is a constant, which
#: is what keeps the arm valid at any input resolution (a per-resolution basis would have
#: to be re-learned whenever the input size changed).
_DCT_BLOCK = 8


def _interp_weights(radius: torch.Tensor, bands: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-band interpolation weights for each coefficient, from its normalised radius.

    Returns ``(lo, hi, w)`` flattened, where the gain for a coefficient is
    ``g[lo] * (1 - w) + g[hi] * w``. Using the two *nearest* bands (rather than
    hard-assigning coefficients to bands) keeps the gain a continuous, differentiable
    function of the band parameters, so every band receives gradient.
    """
    pos = (radius.clamp(0.0, 1.0) * (bands - 1)).clamp(0.0, bands - 1.0)
    lo = pos.floor()
    hi = (lo + 1).clamp(max=bands - 1)
    return lo.flatten().long(), hi.flatten().long(), (pos - lo).flatten()


def _radial_band_index(h: int, w: int, bands: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-band interpolation weights for each rfft2 bin."""
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)  # cycles/sample in [-0.5, 0.5)
    fx = torch.fft.rfftfreq(w, device=device, dtype=dtype)  # cycles/sample in [0, 0.5]
    # Normalise by the Nyquist radius sqrt(0.5^2 + 0.5^2) so r == 1 sits at the corner.
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / (0.5 * (2.0**0.5))
    return _interp_weights(radius, bands)


def _block_band_index(bands: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-band weights for the DCT arm, over the block's ``(u, v)`` coefficient grid.

    Normalised the same way the FFT arm is (``r == 1`` at the highest-frequency corner),
    so a band index means the same kind of thing in both arms and the parameter counts are
    directly comparable: ``C x bands`` either way.
    """
    u = torch.arange(_DCT_BLOCK, device=device, dtype=dtype)
    radius = torch.sqrt(u[:, None] ** 2 + u[None, :] ** 2) / ((2.0**0.5) * (_DCT_BLOCK - 1))
    return _interp_weights(radius, bands)


def _dct_basis(device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II basis as ``(64, 64)``: rows are coefficient, columns are pixel.

    Orthonormal (so ``B @ B.T == I``) because the arm multiplies the analysis matrix and
    its transpose: with an unnormalised basis the round trip would rescale the feature and
    the "identity at zero gain" property would hold only up to a constant factor.
    """
    n = _DCT_BLOCK
    k = torch.arange(n, device=device, dtype=dtype)
    # Orthonormal 1D DCT-II scaling: alpha(0) = sqrt(1/N), alpha(u) = sqrt(2/N). The 2D
    # basis is then the outer product of two 1D bases, i.e. alpha(u)*alpha(v), and *that*
    # product -- not a constant 1/2 -- is what makes B B^T = I. Getting this wrong scales
    # the analysis or synthesis step by a constant, which breaks the round trip while
    # still producing finite output that looks like a working filter.
    alpha = torch.full((n,), (2.0 / n) ** 0.5, device=device, dtype=dtype)
    alpha[0] = (1.0 / n) ** 0.5
    basis = torch.zeros(n, n, n, n, device=device, dtype=dtype)  # (u, v, i, j)
    for u in range(n):
        for v in range(n):
            basis[u, v] = (
                alpha[u] * alpha[v]
                * torch.cos((2 * k[:, None] + 1) * u * torch.pi / (2 * n))
                * torch.cos((2 * k[None, :] + 1) * v * torch.pi / (2 * n))
            )
    return basis.reshape(n * n, n * n)


def _haar_split(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """One-level orthonormal Haar analysis: ``LL, LH, HL, HH``, each half-resolution."""
    even_even = x[..., 0::2, 0::2]
    even_odd = x[..., 0::2, 1::2]
    odd_even = x[..., 1::2, 0::2]
    odd_odd = x[..., 1::2, 1::2]
    half = 0.5
    ll = (even_even + even_odd + odd_even + odd_odd) * half
    lh = (even_even - even_odd + odd_even - odd_odd) * half
    hl = (even_even + even_odd - odd_even - odd_odd) * half
    hh = (even_even - even_odd - odd_even + odd_odd) * half
    return ll, lh, hl, hh


def _haar_merge(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_haar_split` (the transform is orthonormal, so A^-1 == A^T)."""
    half = 0.5
    even_even = (ll + lh + hl + hh) * half
    even_odd = (ll - lh + hl - hh) * half
    odd_even = (ll + lh - hl - hh) * half
    odd_odd = (ll - lh - hl + hh) * half
    out = torch.empty(
        (*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2), device=ll.device, dtype=ll.dtype
    )
    out[..., 0::2, 0::2] = even_even
    out[..., 0::2, 1::2] = even_odd
    out[..., 1::2, 0::2] = odd_even
    out[..., 1::2, 1::2] = odd_odd
    return out


class SpatialFrequencyRepresentation(nn.Module):
    """Learnable radial spectral filter, fused back into the spatial feature.

    Args:
        c1: Input channels, or the parser's ``ch`` list (see ``_common`` docstring).
        mode: ``"sff"`` (proposed), ``"static"``, ``"highpass"``, ``"dct"``, ``"wavelet"``,
            ``"none"``.
        bands: Number of radial frequency bands. More bands = finer spectral control. For
            the ``"dct"`` arm this is the number of radial groups inside the 8x8 block; the
            ``"wavelet"`` arm has exactly four sub-bands and ignores it.
        reduction: Bottleneck ratio for the input-adaptive modulation head.
        alpha_init: Initial residual gate (0 => exact identity at init).
    """

    MODES = ("sff", "static", "highpass", "dct", "wavelet", "none")
    #: Bumped whenever the parameter layout changes, so checkpoints can be validated.
    #: v3: the "none" arm no longer carries an unreachable gate parameter.
    version = 3

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
        elif mode == "dct":
            # Same shape as the FFT arms' bands: (C, bands). Built lazily on first forward
            # because the basis is device dependent while the parameter is not.
            self.band_gain = nn.Parameter(torch.randn(self.c1, self.bands) * 1e-2)
            self.register_buffer("dct_basis", torch.zeros(0), persistent=False)
        elif mode == "wavelet":
            # Four octave sub-bands (LL, LH, HL, HH) per channel -- the Haar basis is fixed,
            # so this arm's only freedom is reweighting those four bands.
            self.band_gain = nn.Parameter(torch.randn(self.c1, 4) * 1e-2)
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

        # The disabled arm gets no gate. A gate that is created but never reached in forward
        # still appears in `parameters()`, so a "removed" arm would then differ from the full
        # model by a stray scalar and "slot disabled" would not be literally true. The
        # parameter layout changed, hence the version bump above.
        if mode != "none":
            self.alpha = ZeroGate(alpha_init)

    # ------------------------------------------------------------------- spectrum
    def _block_gain(self, x: torch.Tensor) -> torch.Tensor:
        """Per-coefficient gain for the local transforms, shaped ``(B?, C, 64)``.

        Interpolated over the coefficient radius exactly as the FFT arm interpolates over
        bin radius, so ``bands`` means the same thing in both and the parameter counts are
        comparable.
        """
        lo, hi, weight = _block_band_index(self.bands, x.device, torch.float32)
        g_lo = self.band_gain.index_select(-1, lo)
        g_hi = self.band_gain.index_select(-1, hi)
        return g_lo * (1.0 - weight) + g_hi * weight

    def _basis(self, device, dtype) -> torch.Tensor:
        """DCT-II basis, cached on first use (and re-created if the device changes)."""
        if self.dct_basis.numel() == 0 or self.dct_basis.device != device:
            self.dct_basis = _dct_basis(device, dtype)
        return self.dct_basis.to(dtype)

    def _forward_dct(self, x: torch.Tensor) -> torch.Tensor:
        """Overlapping-free 8x8 block DCT round trip with a learnable radial gain.

        Padding is replication, so the padded ring cannot introduce a hard edge that the
        transform would then spread across the block boundaries.
        """
        n = _DCT_BLOCK
        b, c, h, w = x.shape
        pad_h, pad_w = (-h) % n, (-w) % n
        padded = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else x
        ph, pw = padded.shape[-2:]

        blocks = nn.functional.unfold(padded, kernel_size=n, stride=n)  # (B, C*n*n, L)
        n_blocks = blocks.shape[-1]
        blocks = blocks.view(b, c, n * n, n_blocks)
        coeffs = self._basis(x.device, x.dtype) @ blocks  # (B, C, n*n, L)
        gain = self._block_gain(x).unsqueeze(0).unsqueeze(-1)  # (1, C, n*n, 1)
        coeffs = coeffs * torch.exp(gain.clamp(-6.0, 6.0))
        restored = self._basis(x.device, x.dtype).t() @ coeffs
        folded = nn.functional.fold(
            restored.reshape(b, c * n * n, n_blocks), (ph, pw), kernel_size=n, stride=n
        )
        return folded[..., :h, :w]

    def _forward_wavelet(self, x: torch.Tensor) -> torch.Tensor:
        """One-level Haar analysis/synthesis with learnable per-sub-band gains."""
        h, w = x.shape[-2:]
        pad = (w % 2, h % 2)
        padded = nn.functional.pad(x, (0, pad[0], 0, pad[1]), mode="replicate") if any(pad) else x
        ll, lh, hl, hh = _haar_split(padded)
        gain = torch.exp(self.band_gain.clamp(-6.0, 6.0)).view(1, self.c1, 4, 1, 1)
        restored = _haar_merge(*(band * gain[:, :, i] for i, band in enumerate((ll, lh, hl, hh))))
        return restored[..., :h, :w]

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
        if self.mode == "dct":
            y = self._forward_dct(xh)
        elif self.mode == "wavelet":
            y = self._forward_wavelet(xh)
        else:
            spectrum = torch.fft.rfft2(xh, norm="ortho")
            filtered = torch.fft.irfft2(spectrum * self.spectral_gain(xh), s=xh.shape[-2:], norm="ortho")
            y = filtered
        y = y.to(x.dtype)
        return x + self.alpha() * (y - x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}, bands={self.bands}"
