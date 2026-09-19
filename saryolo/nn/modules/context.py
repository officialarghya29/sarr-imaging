"""Component 10 — Context Aggregation (CAG).

Motivation
----------
A SAR target is not identifiable from its own 3x3 neighbourhood alone. A bright
point return in the middle of open sea, a bright point return inside a harbour,
and a speckle spike on grassland can have nearly identical local appearance; what
separates them is the *surrounding* scene. Standard convolutions grow their
receptive field by going deeper, which spends parameters and resolution to
approximate context that could be gathered directly.

CAG gathers context explicitly at two extents and adds it back as a residual::

    c_local    = psi([ DWConv_d(x) for d in dilations ])   # multi-extent spatial context
    c_regional = phi(global_avg_pool(x))                    # regional/channel context
    F'         = F + alpha * (c_local + c_regional)

Dilated depth-wise convolutions widen the receptive field without downsampling, so
context is acquired at the detection resolution rather than in a coarser branch.
This matters for the same reason the P2 head exists: a downsampled context path
cannot help a target that is only a few pixels wide.

Relation to attention, stated plainly
-------------------------------------
This module overlaps in *purpose* with the local-contrast branch of
:class:`~saryolo.nn.modules.attention.SARAdaptiveAttention`, but not in mechanism:
attention produces a multiplicative *gate* on the existing feature, whereas CAG
adds a *context field* derived from a wider neighbourhood. Whether that distinction
earns its parameters is an empirical question, which is why the ablation ladder
below exists and why the leave-one-out variants in ``tests/test_arch.py`` measure
it against the full model rather than asserting it.

Component numbering used throughout this repository::

    1 SFE   2 SFM   3 SAA   4 AMF   5 P2 head   6 oriented (planned, unimplemented)
    7 SAR-aware loss   8 target prior   9 spatial-frequency   10 context aggregation

Ablation ladder -- all arms are exact identities at initialisation::

    "none"      no context (control).
    "local"     dilated multi-extent spatial context only.
    "regional"  regionally pooled channel context only.
    "multi"     proposed: local + regional.

Why the mode is called ``regional`` and not ``global``
-----------------------------------------------------
Not a stylistic choice. ``parse_model`` resolves every string argument with
``ast.literal_eval`` inside ``contextlib.suppress(ValueError)``. For an ordinary
identifier that raises ``ValueError``, which is suppressed and the string survives;
but ``global`` is a Python **keyword**, so ``literal_eval`` raises ``SyntaxError``,
which is *not* suppressed and aborts model construction with a bare
``SyntaxError: invalid syntax``. Mode names therefore have to be valid
non-keyword identifiers -- see
``tests/test_arch.py::test_module_mode_names_are_safe_for_parse_model``.
The same code path also means a mode name that collides with a local variable of
``parse_model`` (``ch``, ``f``, ``n``, ``m``, ``args``, ...) would be *silently*
substituted by that variable rather than erroring, which is worse; the test guards
against both.

Every arm keeps the ``F + alpha * (...)`` form with a zero-initialised gate, so all
arms start from the baseline function. The context sub-branches themselves use
their normal initialisation -- see the note below.

Initialisation note (this repository has already been bitten by this once)
-------------------------------------------------------------------------
Exact identity at init is established by the zero-initialised gate **alone**. The
context sub-branches must therefore *not* also be zero-initialised: if the context
field were exactly 0 at init, the gate gradient ``dL/dalpha = <dL/dout, context>``
would vanish identically and the module would stay switched off forever, while
appearing perfectly healthy to an identity-at-init test. See
``tests/test_arch.py::test_no_module_is_frozen_at_init``.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import ConvBNAct, ZeroGate, resolve_c1

__all__ = ["ContextAggregation"]


class ContextAggregation(nn.Module):
    """Additive multi-extent context, channel preserving and identity at init.

    Args:
        c1: Input channels, or the parser's ``ch`` list (see ``_common`` docstring).
        source: Index into ``ch`` for the layer actually consumed. Required because
            this module is wired from an explicit layer index (not ``-1``).
        mode: ``"multi"`` (proposed), ``"local"``, ``"regional"``, ``"none"``.
        reduction: Bottleneck ratio for the regional context branch.
        dilations: Dilations of the spatial context branches, one conv each.
        alpha_init: Initial residual gate (0 => exact identity at init).
    """

    MODES = ("multi", "local", "regional", "none")
    #: Bumped whenever the parameter layout changes, so checkpoints can be validated.
    version = 1

    def __init__(
        self,
        c1,
        source: int | None = None,
        mode: str = "multi",
        reduction: int = 8,
        dilations: tuple[int, ...] = (1, 2, 4),
        alpha_init: float = 0.0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.c1 = resolve_c1(c1, source)
        self.mode = mode
        self.dilations = tuple(int(d) for d in dilations)

        if mode in ("multi", "local"):
            # Depth-wise dilated convs: cheap, no resolution loss, receptive field grows
            # with the dilation instead of with depth.
            self.branches = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(self.c1, self.c1, 3, 1, d, dilation=d, groups=self.c1, bias=False),
                    nn.BatchNorm2d(self.c1),
                    nn.SiLU(inplace=True),
                )
                for d in self.dilations
            )
            self.mix = nn.Conv2d(self.c1 * len(self.dilations), self.c1, 1, bias=False)

        if mode in ("multi", "regional"):
            hidden = max(self.c1 // max(reduction, 1), 8)
            self.regional = nn.Sequential(
                ConvBNAct(self.c1, hidden, k=1, act=True),
                nn.Conv2d(hidden, self.c1, 1, bias=False),
            )

        self.alpha = ZeroGate(alpha_init)

    def context(self, x: torch.Tensor) -> torch.Tensor:
        """The context field added to the feature, shaped like ``x``."""
        ctx = torch.zeros_like(x)
        if self.mode in ("multi", "local"):
            ctx = ctx + self.mix(torch.cat([b(x) for b in self.branches], dim=1))
        if self.mode in ("multi", "regional"):
            # Regional context: pool over space, project, broadcast back. This is the
            # "what kind of scene am I in" signal, complementing the local field.
            pooled = x.mean(dim=(2, 3), keepdim=True)
            ctx = ctx + self.regional(pooled)
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        return x + self.alpha() * self.context(x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}, dilations={self.dilations}"
