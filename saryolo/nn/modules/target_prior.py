"""Component 8 — Target Prior Modulation (TPM): the project's central hypothesis.

Motivation
----------
Speckle and clutter are not the same thing, and neither is the target. A detector
that only *denoises* throws away evidence; a detector that only attends does not
say *what* it is attending to. The hypothesis of this project is narrower and
testable:

    If a network is given an explicit, learned notion of "where target-like
    structure is likely to be", and that notion is *used to modulate the
    detection features* rather than merely plotted, then hard targets (small,
    low-contrast, speckle-obscured) become easier to detect.

This module is the mechanism that makes that hypothesis falsifiable. It produces a
**signed target prior** ``M`` and applies it multiplicatively::

    mu_n, sd_n = LocalStats_k(F)                  # near-scale local statistics
    sd_w       = LocalStats_(2k+1)(F)             # wide-scale local statistics
    e          = psi([F ; F - mu_n ; sd_n ; sd_w])   # target evidence, 1 channel
    M          = 2 * sigmoid(e) - 1                  # signed prior in (-1, 1)
    F'         = F * (1 + M)                         # modulate, do not replace
    F'         = F + alpha * (F * (1 + M) - F)       # alpha = 0 at init => identity

Following the formula in the project brief (``F_target = F x (1 + M_target)``), but
wrapped in the repository's zero-initialised residual gate so a freshly built
SAR-YOLO is numerically *identical* to its baseline. ``M`` is signed on purpose:
``M > 0`` marks target-like structure (amplify), ``M < 0`` marks speckle/clutter-like
structure (suppress). A non-negative-only prior could express the first half of
that sentence but not the second.

Why the prior is *signed* and *multiplicative*
----------------------------------------------
Speckle is signal dependent, so the correction must scale with local intensity
(multiplicative) rather than being a constant offset (additive). This is the same
reasoning that motivates the ``beta * N * F`` term in
:class:`~saryolo.nn.modules.speckle.SpeckleAwareFeatureModule`.

The ablation ladder, and why it is shaped this way
--------------------------------------------------
``mode`` varies only *how M is produced*, holding the slot, the gate and the
``F * (1 + M)`` application fixed::

    "none"     no modulation (control; the gate is the only parameter)
    "cfar"     classical, *non-learned* prior: the CFAR statistic generalised to
               features, i.e. "this response exceeds the local background by more
               than one local standard deviation". Zero learnable parameters.
    "static"   learnable but spatially *uniform* prior (per-channel logits).
    "channel"  the same evidence network as "learned", but its spatial extent is
               average-pooled away, so the resulting prior is spatially uniform.
    "learned"  the proposed spatially-varying, content-adaptive prior map.

``"channel"`` exists to remove a confound that a reviewer would otherwise be right
to raise: comparing ``"static"`` (about ``C`` parameters) against ``"learned"``
(a small CNN) confounds *spatial selectivity* with *capacity*. ``"channel"`` has
the same network as ``"learned"`` and differs only in whether the prior is allowed
to vary across space, so ``learned`` vs ``channel`` isolates spatial selectivity at
matched capacity, while ``learned`` vs ``static`` answers the coarser question.

All five arms are exact identities at initialisation (``alpha = 0``), so every arm
starts from the baseline function and the comparison is about what training does
with the extra structure -- not about who starts with a larger perturbation.

Visualisation contract
----------------------
:meth:`TargetPriorModulation.prior_map` returns the prior as ``(B, 1, H, W)`` for
*every* mode, so the figure pipeline (SEC. 29-30 of the brief: image ->
representation -> prior/attention -> prediction) can render any arm identically.
For the spatially uniform arms the returned map is constant across space *by
construction*, and that is exactly what those arms are testing.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import ConvBNAct, DWConvBNAct, LocalStats, ZeroGate, resolve_c1

__all__ = ["TargetPriorModulation"]


class TargetPriorModulation(nn.Module):
    """Signed, content-adaptive target prior applied multiplicatively to features.

    Args:
        c1: Input channels, or the parser's ``ch`` list (see ``_common`` docstring).
        source: Index into ``ch`` for the layer actually consumed. Required because
            this module is wired from an explicit layer index (not ``-1``).
        mode: Prior construction, one of
            ``"learned"`` (proposed), ``"channel"``, ``"static"``, ``"cfar"``, ``"none"``.
        kernel: Near-scale local-statistics window.
        reduction: Bottleneck ratio of the evidence network.
        cfar_threshold: CFAR arm only -- how many local standard deviations above the
            local mean a response must sit to count as target-like.
        cfar_gain: CFAR arm only -- slope of the logistic transfer.
        alpha_init: Initial residual gate (0 => exact identity at init).
    """

    MODES = ("learned", "channel", "static", "cfar", "none")
    #: Bumped whenever the parameter layout changes, so checkpoints can be validated.
    version = 1

    def __init__(
        self,
        c1,
        source: int | None = None,
        mode: str = "learned",
        kernel: int = 3,
        reduction: int = 8,
        cfar_threshold: float = 1.0,
        cfar_gain: float = 1.0,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.c1 = resolve_c1(c1, source)
        self.mode = mode
        self.cfar_threshold = float(cfar_threshold)
        self.cfar_gain = float(cfar_gain)

        # Near- and wide-scale statistics. Only the learned/channel arms consume the
        # wide window, but LocalStats has no parameters, so declaring it once keeps the
        # module's forward path uniform across arms.
        self.stats = LocalStats(k=kernel)
        self.wide = LocalStats(k=2 * kernel + 1)

        if mode in ("learned", "channel"):
            hidden = max(self.c1 // max(reduction, 1), 8)
            # [F ; high-pass ; near sd ; wide sd]: raw intensity, local contrast and two
            # scales of speckle strength. The 3x3 conv gives the evidence a spatial
            # neighbourhood, because a target is a *region*, not a single response.
            self.evidence = nn.Sequential(
                ConvBNAct(4 * self.c1, hidden, k=1),
                DWConvBNAct(hidden, hidden, k=3),
                nn.Conv2d(hidden, 1, 1),
            )
            # Small-but-nonzero init, on purpose. Exact identity at init comes from the
            # gate (alpha = 0), which is sufficient on its own; forcing M to start at
            # exactly 0 as well would be counterproductive, because then the gate
            # gradient dL/dalpha = <dL/dout, F*M> would be identically zero and the
            # prior would never switch on. A gentle init keeps the prior close to
            # uniform at the start without killing its gradient.
            # See tests/test_arch.py::test_no_module_is_frozen_at_init.
            nn.init.normal_(self.evidence[-1].weight, std=1e-2)
            nn.init.zeros_(self.evidence[-1].bias)
        elif mode == "static":
            # Spatially uniform, per-channel learnable prior. Deliberately tiny: this arm
            # is the "can a *constant* emphasis match a spatial prior?" control.
            # Small random rather than zeros for the same reason as the evidence head: an
            # all-zero prior makes the residual `F * (1 + M) - F` exactly zero, which would
            # freeze the gate rather than merely quieten it.
            self.static_logits = nn.Parameter(torch.randn(1, self.c1, 1, 1) * 1e-2)

        self.alpha = ZeroGate(alpha_init)

    # ------------------------------------------------------------------ evidence
    def _evidence(self, x: torch.Tensor) -> torch.Tensor:
        """Target evidence with shape ``(B, S, H', W')``; ``S`` is 1 or ``C``."""
        if self.mode == "cfar":
            mu, _, sd, _ = self.stats(x)
            # CFAR, generalised from intensity to features: how far above the local
            # background this response sits, measured in local standard deviations.
            excess = (x - mu) / sd
            return torch.tanh(self.cfar_gain * (excess - self.cfar_threshold))

        if self.mode == "static":
            return self.static_logits

        _, _, sd_n, hp = self.stats(x)
        _, _, sd_w, _ = self.wide(x)
        e = self.evidence(torch.cat((x, hp, sd_n, sd_w), dim=1))  # (B, 1, H, W)
        if self.mode == "channel":
            # Pool the evidence over space and keep the batch axis: the prior keeps its
            # capacity but loses the ability to vary spatially. This is the matched-capacity
            # control for `learned`.
            e = e.mean(dim=(2, 3), keepdim=True)
        return e

    def _prior(self, x: torch.Tensor) -> torch.Tensor:
        """Signed prior ``M`` in ``(-1, 1)``, broadcastable against ``x``."""
        if self.mode == "none":
            return torch.zeros_like(x)

        e = self._evidence(x)
        if self.mode in ("cfar", "learned", "channel"):
            # 2*sigmoid(z) - 1 == tanh(z/2): bounded in (-1, 1) and signed, so the prior
            # can both amplify (M > 0) and suppress (M < 0) rather than only gating on.
            return 2.0 * torch.sigmoid(e) - 1.0
        # static: per-channel logits, already the modulation, squashed to (-1, 1).
        return torch.tanh(e)

    def prior_map(self, x: torch.Tensor) -> torch.Tensor:
        """Spatially-reduced prior as ``(B, 1, H, W)``, for every mode.

        Returned in ``(-1, 1)``. Positive values mark target-like structure, negative
        values speckle/clutter-like structure. The uniform arms (``static``,
        ``channel``) return a constant map, which is what those arms assert.
        """
        m = self._prior(x)
        if m.shape[1] != 1:
            m = m.mean(dim=1, keepdim=True)
        return m.expand(x.shape[0], 1, *x.shape[-2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self._prior(x)
        return x + self.alpha() * (x * (1.0 + m) - x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}"
