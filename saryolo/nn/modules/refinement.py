"""Component 11 — Target-Aware Deformable Refinement (TADR).

Motivation
----------
The components before this one change *what the feature contains*: enhancement
re-expresses it, decomposition suppresses interference, the prior amplifies
target-like structure, attention and context reweight it. None of them changes
*where the feature is sampled*. A detector that only reweights its features is
stuck with whatever alignment the backbone's fixed grid produced, and in SAR that
grid is frequently a poor match for the target: ship wakes, harbour edges and
speckle spikes pull the local statistics around, so the response that is strongest
at a cell is often a neighbouring clutter pixel rather than the target itself.

TADR adds a lightweight second stage that lets each cell *look somewhere else*::

    d      = Delta(F)                          offsets, (B, 2, H, W)
    d      = s * tanh(d)                        bounded, so sampling cannot run away
    F~     = grid_sample(F, base_grid + d)      differentiable bilinear resample
    F'     = mix(F~)                           depth-wise 3x3 + BN + SiLU
    out    = F + alpha * (F' - F)               alpha = 0 at init  =>  identity

This is a *refinement* in the spatial sense, which is what makes it a distinct
mechanism rather than a variation of attention (a multiplicative gate on the
feature present at a cell) or of context (an additive field aggregated around a
cell). The ablation ladder is built to isolate exactly that distinction::

    "none"     identity.
    "local"    depth-wise 3x3 mixing, **no offsets**. This is the capacity control:
               it has the same sub-network as the proposed arm, so `deform - local`
               measures the value of *deformation* rather than of an extra conv.
    "static"   a learned but input-independent offset field, averaged over the
               batch. Isolates *content-adaptive* sampling from learned sampling.
    "deform"   the proposed arm: offsets predicted per sample from the feature.

Note that the proposed arm consumes an already prior-modulated feature, because
the architecture wires it after Component 8. The "target-aware" conditioning is
therefore structural, not a second prior network -- deliberately, so that this
component does not duplicate Component 8's evidence computation.

Why this is per-level, and what that costs
-----------------------------------------
A *cross-level* routing stage (reweighting P2-P5 jointly against one shared prior)
would need a module that consumes several feature maps. That cannot be expressed
in this repository's model YAMLs, and the reason is a hard constraint rather than
a design preference -- verified empirically, not assumed:

* ``parse_model`` resolves an unknown module's output channels with ``c2 = ch[f]``.
  For a row whose ``from`` is a list this raises
  ``TypeError: list indices must be integers or slices, not list``.
* naming a ``Detect`` **subclass** as the head does not work either: the head
  branch is a ``frozenset`` *identity* membership test, so a subclass falls through
  to the generic path, never receives ``[reg_max, end2end, ch_list]``, and dies with
  ``IndexError: tuple index out of range``.

Cross-level routing would therefore require either patching ``parse_model`` (which
this project avoids, so that the stock model summary, FLOPs counter, validator and
checkpointing keep working) or a model-level mechanism declared outside the YAML.
It is left out rather than faked per-level and described as cross-level. The cost
of the omission is stated plainly in ``docs/METHOD.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import DWConvBNAct, ZeroGate, resolve_c1

__all__ = ["TargetAwareRefinement"]


class TargetAwareRefinement(nn.Module):
    """Prior-conditioned deformable refinement, channel preserving and identity at init.

    Args:
        c1: Input channels, or the parser's ``ch`` list (see ``_common`` docstring).
        source: Index into ``ch`` for the layer actually consumed. Required because
            this module is wired from an explicit layer index (not ``-1``).
        mode: ``"deform"`` (proposed), ``"static"``, ``"local"``, ``"none"``.
        kernel: Depth-wise mixing kernel applied after resampling.
        max_offset: Largest offset in normalised grid units (1.0 = half the feature
            map). Offsets are ``max_offset * tanh(.)``, so this bounds how far a cell
            may look and keeps the resampling inside a sensible neighbourhood.
        alpha_init: Initial residual gate (0 => exact identity at init).
    """

    MODES = ("deform", "static", "local", "none")
    #: Bumped whenever the parameter layout changes, so checkpoints can be validated.
    version = 1

    def __init__(
        self,
        c1,
        source: int | None = None,
        mode: str = "deform",
        kernel: int = 3,
        max_offset: float = 0.5,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.c1 = resolve_c1(c1, source)
        self.mode = mode
        self.kernel = int(kernel)
        self.max_offset = float(max_offset)

        # Every *active* mode shares this sub-network, so "local" is a genuine capacity
        # control for "deform": the arms differ only in whether the sampling grid moves.
        # It is deliberately not built for "none": a disabled slot that still allocates the
        # mixing convolution is not a disabled slot, and it would make the removal ablation
        # compare arms that differ only in whether a parameter block is *used*.
        if mode != "none":
            self.mix = DWConvBNAct(self.c1, self.c1, k=self.kernel)

        if mode == "deform":
            # Two channels of offset, shared across the feature channels (the standard
            # deformable formulation): neighbouring channels of the same level see the
            # same geometry, and per-channel offsets would multiply the field's cost.
            self.offset = nn.Conv2d(self.c1, 2, 3, padding=1, bias=True)
        elif mode == "static":
            # Learned but input-independent: one offset field, the same for every sample.
            # Small random init rather than zeros, so the gate has a gradient at init --
            # the frozen-branch failure mode documented in _common and pinned by
            # tests/test_arch.py::test_no_module_is_frozen_at_init.
            self.static_offset = nn.Parameter(torch.randn(2) * 1e-2)

        self.alpha = ZeroGate(alpha_init)

    # ------------------------------------------------------------------- sampling
    def _base_grid(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Identity sampling grid as ``(H, W, 2)`` in ``(x, y)`` order.

        ``align_corners=True`` with a ``linspace(-1, 1)`` ramp is the pairing that makes
        a zero offset an exact identity resample, which is what lets "no deformation"
        be a meaningful baseline for the other arms.
        """
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((gx, gy), dim=-1)

    def _offsets(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor | None:
        """Bounded offsets shaped ``(B, H, W, 2)``, or ``None`` for non-warping arms."""
        if self.mode == "local" or self.mode == "none":
            return None
        if self.mode == "static":
            d = self.static_offset.view(1, 2, 1, 1).expand(x.shape[0], 2, h, w)
        elif self.mode == "deform":
            d = self.offset(x)
        else:  # pragma: no cover - the two arms above are the only reachable ones here
            raise AssertionError(f"unreachable mode {self.mode!r}")
        # tanh bound: a cell may look at most `max_offset` away, so the refinement stays
        # a local adjustment and cannot sample an unrelated part of the scene.
        return (self.max_offset * torch.tanh(d)).permute(0, 2, 3, 1).contiguous()

    def offset_map(self, x: torch.Tensor) -> torch.Tensor:
        """Bounded offsets as ``(B, H, W, 2)`` in ``(x, y)`` grid units, for figures.

        The counterpart of :meth:`TargetPriorModulation.prior_map`: it makes the module's
        behaviour inspectable, so an explainability figure can show *where* refinement
        looks rather than only asserting that it does. Returned in grid units (1.0 = half
        the feature map) so the magnitude is comparable across levels and resolutions.

        Raises:
            ValueError: If called on an arm that has no offsets (``"local"``/``"none"``).
                Returning zeros would render a plot that looks like a trained model
                choosing not to move, which is a different and misleading statement.
        """
        offsets = self._offsets(x, x.shape[2], x.shape[3])
        if offsets is None:
            raise ValueError(f"mode {self.mode!r} has no offset field to visualize")
        return offsets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x

        _, _, h, w = x.shape
        # The whole body runs in at least fp32. Two reasons, and the first is a hard
        # requirement rather than a preference: ``grid_sample`` has no half-precision CPU
        # kernel, and the offset convolution would otherwise receive a float16 activation
        # against fp32 weights ("Input type (c10::Half) and bias type (float)"). The second
        # is that a sub-pixel resample is exactly the operation whose result is dominated by
        # interpolation error in low precision. The spectral branch upcasts for the same
        # first reason.
        compute_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        xf = x.to(compute_dtype)
        offsets = self._offsets(xf, h, w)

        if offsets is None:
            refined = self.mix(xf)
        else:
            # Both arms produce a per-sample grid, so the grid's batch matches the input's
            # and no broadcasting is needed: (B, H, W, 2) against (B, C, H, W).
            grid = self._base_grid(h, w, x.device, compute_dtype).unsqueeze(0) + offsets
            sampled = torch.nn.functional.grid_sample(
                xf, grid, mode="bilinear", padding_mode="border", align_corners=True,
            )
            refined = self.mix(sampled)

        refined = refined.to(x.dtype)
        return x + self.alpha().to(x.dtype) * (refined - x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}, max_offset={self.max_offset}"
