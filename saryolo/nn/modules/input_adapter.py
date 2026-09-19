"""Module A — SAR Input Adapter (SIA).

Motivation
----------
Every other module in this package operates on *backbone features*. This one operates
on the **image**, before the first convolution sees it, because the first convolution
is where SAR's statistics are least represented: a stock detector's stem is trained on
natural images, and it consumes raw intensity as if it were an RGB photograph. SAR
intensity is multiplicative-speckle dominated, its dynamic range is enormous, and its
target evidence lives mostly in local *contrast* and *structure* rather than in
absolute level.

The brief (SEC. 3, Module A) specifies exactly this comparison, and forbids assuming
the answer: intensity, local statistics, a learned representation, and a hybrid of
all three, each measured.

Formulation
-----------
For an input image ``x`` (``c1`` channels) with local mean ``mu``, local std ``sd``
and high-pass residual ``hp = x - mu`` over a ``k x k`` window::

    stats   = psi([x ; mu ; sd ; hp])          # local-statistics representation
    learned = phi(DWConv(x))                    # learned representation
    z       = psi2([stats ; learned])           # hybrid: both streams, learned fusion
    x'      = x + alpha * (z - x)               # alpha = 0 at init  =>  identity

``alpha`` is a tanh-bounded zero-initialised gate, so every arm — including the
proposed hybrid — is an *exact* identity at initialisation, exactly like the feature
level modules. The adapter therefore cannot change the baseline function before
training; whatever it contributes has to be learned.

Arms
----
``"identity"``  raw intensity. Exactly zero parameters: the control, and a bit-exact
                no-op (measured, not assumed -- see below).
``"local"``     local-statistics representation only. ``4*c1^2 + 2*c1 + 1`` parameters.
``"learned"``   learned representation only, no explicit statistics.
``"hybrid"``    proposed: both streams fused by a learned 1x1 projection.

Parameter cost at the first-row width (``c1 = 3`` for a stock 3-channel model), measured on
the built model rather than estimated, since that is what the ablation table reports:
``identity`` +0, ``local`` +43, ``learned`` +98, ``hybrid`` +164.

The control's exactness was checked directly: dropping the adapter row from ``in_identity``
leaves a graph layer-for-layer identical to ``v2_full`` (same types, same per-layer parameter
counts), and after synchronising the shared weights the two produce **bit-identical** outputs
(max absolute difference 0.0). So ``in_identity`` measures the rest of the model and nothing
else -- which is the whole point of having that arm.

Why the projected form, and when it can help
--------------------------------------------
``parse_model`` resolves an unknown module's output width with ``c2 = ch[f]``, so the
adapter must return *the same* number of channels it received — a genuine widening of
the input tensor is impossible without patching the parser or the stem. The extra
channels are therefore a hidden representation projected back down, which is why this
is an *adapter* rather than a new input layer.

One further honesty note that belongs with the module rather than in a limitations
section. Ultralytics loads a single-channel SAR image by replicating it into three
channels, so on single-polarisation data ``x`` and ``mu``/``sd``/``hp`` are each
internally redundant: the local statistics carry no information the raw channels do
not already order differently. On such data the honest expectation is that ``local``
adds little over ``identity``, and the experiment should say so. The arm becomes
genuinely richer only for multi-channel input (dual-polarisation VV/VH, or a
pre-computed feature stack such as intensity + local statistics + gradient).

Ablation: the four arms above. The brief's ``A1..A4``, kept in one slot under one init
contract, so a difference between arms is attributable to the representation rather
than to where the module was inserted.
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import ConvBNAct, DWConvBNAct, LocalStats, ZeroGate, resolve_c1

__all__ = ["SARInputAdapter"]


class SARInputAdapter(nn.Module):
    """Input-level SAR representation adapter, channel preserving and identity at init.

    Args:
        c1: Input channels, or the parser's ``ch`` list (see ``_common`` docstring).
            At the first YAML row this is the stem's input width (``3`` for a stock
            YOLO model), because ``parse_model`` seeds the channel list with it.
        mode: ``"hybrid"`` (proposed), ``"local"``, ``"learned"``, ``"identity"``.
            Placed *before* ``source``, unlike the feature-level modules: this module sits
            at the first YAML row, where there is no source layer to index at all, so a
            positional ``source`` would force every emitted row to carry a placeholder.
        source: Index into ``ch`` for the layer actually consumed. Left as ``None`` at the
            first row, which resolves to the stem's input width.
        kernel: Local-statistics window size.
        reduction: Hidden bottleneck ratio.
        alpha_init: Initial residual gate (0 => exact identity at init).
    """

    MODES = ("hybrid", "local", "learned", "identity")
    #: Bumped whenever the parameter layout changes, so checkpoints can be validated.
    version = 1

    def __init__(
        self,
        c1,
        mode: str = "hybrid",
        source: int | None = None,
        kernel: int = 3,
        reduction: int = 8,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.c1 = resolve_c1(c1, source)
        self.mode = mode
        self.kernel = int(kernel)
        hidden = max(self.c1 // max(reduction, 1), 8)

        if mode in ("local", "hybrid"):
            self.stats = LocalStats(k=self.kernel)
            # 4 streams: raw, local mean, local std, high-pass residual.
            self.project = ConvBNAct(4 * self.c1, self.c1, k=1, act=True)

        if mode in ("learned", "hybrid"):
            # Depth-wise then point-wise: a cheap representation that cannot mix channels
            # before it has seen them, which keeps a 3-channel input from being collapsed
            # by a single 1x1 projection.
            self.learned = DWConvBNAct(self.c1, hidden, k=3)
            self.expand = ConvBNAct(hidden, self.c1, k=1, act=False)

        if mode == "hybrid":
            # Fusion of the two streams. Deliberately *not* zero-initialised: identity at
            # init comes from the gate alone. A zero-initialised fusion would make the gate
            # gradient <dL/dout, z - x> vanish identically and freeze the whole adapter,
            # which is the failure mode documented in _common and pinned by
            # tests/test_arch.py::test_no_module_is_frozen_at_init.
            self.fuse = ConvBNAct(2 * self.c1, self.c1, k=1, act=True)

        if mode != "identity":
            self.alpha = ZeroGate(alpha_init)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        """The adapted representation ``z`` (before gating), shaped like ``x``.

        Exposed so the visualization pipeline can render what the adapter produces,
        rather than only what it contributes after gating.
        """
        if self.mode == "identity":
            return x

        if self.mode == "learned":
            # No stats branch in this arm, so `self.stats` is not even constructed: the arm
            # is the learned representation on its own.
            return self.expand(self.learned(x))

        mu, _var, sd, hp = self.stats(x)
        stats = self.project(torch.cat((x, mu, sd, hp), dim=1))
        if self.mode == "local":
            return stats

        learned = self.expand(self.learned(x))
        return self.fuse(torch.cat((stats, learned), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return x
        return x + self.alpha() * (self.representation(x) - x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, mode={self.mode}"
