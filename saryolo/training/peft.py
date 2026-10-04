"""Parameter-efficient adaptation: the LoRA baseline the master context requires.

Why this module exists
----------------------
The project's central claim is that a *lightweight, metadata-conditioned* adapter
generalises across acquisition sources at negligible parameter cost. That claim has a
natural competitor which the master context (Direction B / Stage 10) names explicitly:
low-rank adaptation. LoRA reaches the same "adapt a frozen backbone cheaply" goal by a
completely different mechanism -- it adds trainable rank-``r`` updates to existing
weights and freezes everything else -- so any comparison that omits it is comparing the
proposed adapter against a *full fine-tune*, which is not a fair or interesting
comparator. A reviewer's first question about a parameter-efficient method is "how does
it compare with LoRA?", and this module is the answer.

Why it is injected rather than declared in a model YAML
-------------------------------------------------------
LoRA is not an architectural block. It does not change the graph, the tensor shapes, or
the forward pass at initialisation; it changes *which weights are trained*. Expressing it
as a YAML row would therefore be wrong in two ways: it would put a training-time decision
into the architecture registry (so a model YAML would no longer describe a model), and it
would make the identity-at-initialisation contract meaningless, because a LoRA-wrapped
layer is not a module inserted into a graph but a re-parameterisation of an existing one.
It is a property of the *run*, so it lives in the run config (``peft:`` block) and is
applied by the runner before training.

Identity at initialisation, again
---------------------------------
The ``B`` matrix is zero-initialised and ``A`` is drawn from a small normal distribution,
so ``scale * B @ A`` is exactly zero at step 0 and the wrapped layer computes bit-for-bit
what the base layer computed. This mirrors the gate convention every module in
``saryolo.nn.modules`` follows, and it is what makes "the LoRA arm differs from the
baseline only in what was learned" a true statement rather than a hope. The failure mode
this avoids is the mirror image of the gate bug documented in ``_common.py``: with *both*
factors zero, ``dL/dA = B^T dL/dy = 0`` and the adapter can never leave zero, so a
zero-zero initialisation would produce a permanently dead arm that still passes every
shape and parameter-count check.

What is frozen, precisely
-------------------------
Only the base layers that were actually wrapped are frozen, and only their weight tensors
are detached from the optimiser. BatchNorm running statistics and affine parameters inside
a wrapped convolution's *parent* block are deliberately left trainable unless the caller
asks otherwise: freezing every non-LoRA parameter is a legitimate and stricter setting
(``freeze_base=True``), but it changes what the experiment measures -- a fully frozen
backbone with a trainable detection head is a *linear probe*, not LoRA. Both settings are
supported and the choice is recorded in the run's summary, because the two are not
interchangeable and a reader has to be able to tell which one produced a number.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

__all__ = [
    "LoRAConfig",
    "LoRALayer",
    "inject_lora",
    "merge_lora",
    "adapter_merged",
    "lora_parameter_summary",
    "DEFAULT_TARGETS",
]

#: Module classes LoRA knows how to wrap. Names rather than classes because the config is
#: YAML and because matching by class *name* also catches ultralytics' own subclasses
#: (``ultralytics.nn.modules.conv.Conv`` is an ``nn.Conv2d`` subclass, so it is caught by
#: the ``"Conv2d"`` target without being named separately).
DEFAULT_TARGETS: tuple[str, ...] = ("Conv2d", "Linear")


@dataclass
class LoRAConfig:
    """A LoRA run, as declared in an experiment config's ``peft`` block.

    Attributes:
        rank: LoRA rank ``r``. The adapter cost per wrapped layer is
            ``r * (in_features + out_features)`` parameters.
        alpha: Scaling numerator; the effective scale is ``alpha / rank``, which is the
            convention the original paper uses and which keeps the update magnitude
            roughly constant as the rank changes.
        targets: Module class names to wrap (see :data:`DEFAULT_TARGETS`).
        dropout: Dropout applied to the low-rank path's input. ``0.0`` disables it.
        freeze_base: Also freeze every parameter that is not a LoRA factor. ``False``
            (the default) freezes only the wrapped layers' own weights, leaving the
            detection head, BatchNorm statistics and any unwrapped module trainable --
            which is what "LoRA fine-tuning" means in the literature. ``True`` turns the
            run into a strict adapter-only optimisation.
        exclude: Fully-qualified module names to leave unwrapped, for a targeted study
            (e.g. ``["model.23"]`` to keep the detection head un-adapted).
    """

    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = DEFAULT_TARGETS
    dropout: float = 0.0
    freeze_base: bool = False
    exclude: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {self.rank}")
        if not self.targets:
            raise ValueError("LoRA targets must not be empty; nothing would be wrapped")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> LoRAConfig:
        """Build from a config mapping, rejecting unknown keys.

        An unknown key is a typo (``r: 8`` for ``rank: 8``), and silently ignoring it would
        run the default rank while the config appears to request another -- an ablation
        whose two arms are identical, which is the single most expensive kind of config
        error because it produces a complete-looking table of zero differences.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown LoRA key(s) {unknown}; known keys are {sorted(known)}"
            )
        data = dict(payload)
        if "targets" in data:
            data["targets"] = tuple(data["targets"])
        if "exclude" in data:
            data["exclude"] = tuple(data["exclude"])
        return cls(**data)

    @property
    def scale(self) -> float:
        """Effective scaling factor applied to the low-rank product."""
        return float(self.alpha) / float(self.rank)

    def describe(self) -> dict[str, Any]:
        return {
            "method": "lora",
            "rank": self.rank,
            "alpha": float(self.alpha),
            "scale": round(self.scale, 6),
            "targets": list(self.targets),
            "dropout": float(self.dropout),
            "freeze_base": bool(self.freeze_base),
            "exclude": list(self.exclude),
        }


class LoRALayer(nn.Module):
    """A frozen base layer plus a trainable low-rank update.

    For a base linear map ``W`` with ``y = W x + b``, the adapted layer computes

    .. code-block:: text

        y = W x + b + scale * B (A x)

    with ``A`` of shape ``(r, in_features)`` and ``B`` of shape ``(out_features, r)``.
    ``B`` starts at zero, so the update is exactly zero at initialisation and the wrapped
    layer is bit-for-bit the base layer.

    A convolution is treated as a linear map by flattening its kernel: the update is a
    ``1x1`` (or grouped, stride-matched) convolution whose weight is the low-rank product.
    The base convolution's own weight is frozen; its bias, if any, is frozen too, because
    the bias is part of the map being adapted and leaving it trainable would make the
    parameter count of the arm ambiguous.

    Args:
        base: The layer to wrap. Must be an ``nn.Conv2d`` or ``nn.Linear``.
        rank: LoRA rank.
        alpha: Scaling numerator (effective scale is ``alpha / rank``).
        dropout: Dropout probability on the low-rank path's input.
    """

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, (nn.Conv2d, nn.Linear)):
            raise TypeError(
                f"LoRA wraps nn.Conv2d/nn.Linear, got {type(base).__name__}. "
                f"Extend DEFAULT_TARGETS only for layers that are linear in their input."
            )
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()

        if isinstance(base, nn.Linear):
            in_features, out_features = base.in_features, base.out_features
        else:
            # A convolution is treated as a linear map over its *flattened kernel*, not over
            # its channel pairs. The distinction matters: a low-rank factorisation of
            # (out, in/groups) can only produce a 1x1 update, which changes the layer's
            # receptive field rather than adapting the filter it already has. Adapting the
            # flattened kernel of shape (out, in/groups, kh, kw) keeps the base geometry and
            # is what "a low-rank update to an existing weight" actually means here.
            kernel_elems = base.kernel_size[0] * base.kernel_size[1] * (base.in_channels // base.groups)
            in_features, out_features = kernel_elems, base.out_channels

        # A starts small and non-zero; B starts at exactly zero. Identity at init is bought
        # by B alone -- and B being the *only* zero factor is what keeps the gradient alive:
        # dL/dA = (dL/dy) B^T is zero while B is zero, but dL/dB = (dL/dy)(A x)^T is not, so
        # B leaves zero on the first step and A follows. Zeroing both would freeze the pair.
        self.lora_a = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.normal_(self.lora_a, std=1.0 / math.sqrt(max(in_features, 1)))

        # The base layer is frozen: that is the definition of the method, and it is what
        # makes the trainable-parameter count a meaningful number to report.
        for param in self.base.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------ structure
    @property
    def in_features(self) -> int:
        return self.lora_a.shape[1]

    @property
    def out_features(self) -> int:
        return self.lora_b.shape[0]

    @property
    def n_lora_parameters(self) -> int:
        return int(self.lora_a.numel() + self.lora_b.numel())

    def delta_weight(self) -> torch.Tensor:
        """The low-rank update ``scale * B @ A``, shaped like the base weight."""
        return (self.scale * (self.lora_b @ self.lora_a)).view_as(self.base.weight)

    def merged_base(self) -> nn.Module:
        """The base layer with the low-rank update folded into its weight.

        Provided for inference: the returned layer is an ordinary ``Conv2d``/``Linear``, so
        it costs exactly what the base costs and needs no adapter code at load time.

        This returns a *new* module rather than mutating in place, because mutating is the
        tempting version and it is wrong: the adapter's own forward pass still adds
        ``B @ A``, so an in-place merge double-counts the update and the model silently
        computes something neither the adapted nor the base layer ever produced. Measured --
        an in-place merge changed the output by 0.28 against an adapter whose update was
        already applied. :func:`merge_lora` is the only supported entry point, and it
        replaces the adapter rather than modifying it.
        """
        import copy

        merged = copy.deepcopy(self.base)
        with torch.no_grad():
            merged.weight.add_(self.delta_weight())
        return merged

    # ------------------------------------------------------------------ forward
    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self._low_rank(x)

    def _low_rank(self, x: torch.Tensor) -> torch.Tensor:
        """The adapter's contribution.

        Computed unconditionally, including at initialisation when ``B`` is exactly zero.

        An earlier version short-circuited on ``not torch.any(self.lora_b)`` to make the
        identity at init bit-exact rather than exact-up-to-reassociation. It was measured
        and it was wrong: skipping the product removes ``lora_a`` and ``lora_b`` from the
        autograd graph entirely, so every adapter parameter received ``grad = None`` and the
        arm could never train. It would have reported a complete LoRA result for a model
        that was never adapted -- the same "dead branch" failure documented in
        ``_common.py``, reached by a different route.

        The exactness the short-circuit was buying is available for free: at init
        ``B @ A`` is *exactly* zero, and adding an exact zero tensor to the base output is
        bit-identical to not adding it. ``tests/test_peft.py`` pins both properties -- the
        bit-exact identity *and* the non-zero gradient -- so the trade-off cannot be made
        silently again.
        """
        x = self.dropout(x)
        if isinstance(self.base, nn.Linear):
            return (x @ self.lora_a.t()) @ self.lora_b.t()
        # Convolution: the update is a convolution with the base layer's geometry and a
        # rank-constrained kernel. `groups`, `stride`, `padding` and `dilation` are copied so
        # the adapter adds to the same map the base computes, not to a different one.
        return nn.functional.conv2d(
            x, self.delta_weight(), None,
            self.base.stride, self.base.padding, self.base.dilation, self.base.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, scale={self.scale:.3f}, "
            f"base={type(self.base).__name__}, "
            f"in={self.in_features}, out={self.out_features}"
        )


def inject_lora(model: nn.Module, config: LoRAConfig) -> dict[str, Any]:
    """Replace every targeted layer in ``model`` with a :class:`LoRALayer`.

    Args:
        model: The model to adapt, modified in place. ``config.exclude`` names are
            interpreted **relative to this module**, so passing the detection model gives
            names like ``model.0.conv`` while passing its inner ``Sequential`` gives
            ``0.conv``. The runner always passes the detection model, which is the scope the
            committed configs use; passing anything else silently changes what ``exclude``
            matches, so the scope is stated here rather than left to be inferred.
        config: The LoRA run configuration.

    Returns:
        A summary suitable for recording in the experiment ledger: how many layers were
        wrapped, the trainable/frozen parameter counts before and after, and the config
        itself. The counts are the reason this returns anything -- a LoRA arm whose
        trainable count silently equalled the full fine-tune would make the efficiency
        comparison meaningless while looking perfectly healthy.

    Raises:
        ValueError: If nothing was wrapped, or a target name matched no layer class in the
            model. Both are silent failures otherwise: an arm that wraps zero layers *is*
            the baseline, and it would be reported as a LoRA result.
    """
    if not isinstance(config, LoRAConfig):
        raise TypeError(f"inject_lora expects a LoRAConfig, got {type(config).__name__}")

    # Collect first, then replace: mutating during traversal would change the module tree
    # under the iterator, and a child of an already-wrapped layer must not be wrapped twice.
    to_wrap: list[tuple[str, nn.Module, nn.Module, str]] = []
    matched_classes: set[str] = set()
    skipped_frozen: list[str] = []
    for name, module in model.named_modules():
        if any(name == e or name.startswith(f"{e}.") for e in config.exclude):
            continue
        for cls in type(module).__mro__:
            if cls.__name__ in config.targets:
                matched_classes.add(cls.__name__)
                # A layer the *base* model already declares untrainable is not wrapped. The
                # case that motivates this is ultralytics' DFL projection, whose weight is
                # explicitly frozen upstream: adapting it would add parameters that can
                # never receive gradient, so the arm's trainable count would overstate what
                # is actually optimised and the adapter would sit permanently at zero.
                # Measured: without this rule one of 88 wrapped layers had a zero gradient.
                if not any(p.requires_grad for p in module.parameters(recurse=False)):
                    skipped_frozen.append(name)
                    break
                to_wrap.append((name, module, _parent_of(model, name), cls.__name__))
                break

    unmatched = sorted(set(config.targets) - matched_classes)
    if unmatched and tuple(config.targets) != tuple(DEFAULT_TARGETS):
        # A target the caller *chose* and that matched nothing is a no-op arm and is refused:
        # the run would report a LoRA result while adapting fewer layers than its config says.
        # The default list is exempt because it is a deliberate broad net -- a pure-conv
        # detector legitimately has no Linear layer -- and the at-least-one-layer check below
        # still covers that case.
        raise ValueError(
            f"LoRA target(s) {unmatched} matched no layer in the model. A target that "
            f"matches nothing makes the arm a no-op, so it is refused rather than ignored."
        )
    if not to_wrap:
        raise ValueError(
            "LoRA wrapped no layers; the arm would be identical to a full fine-tune while "
            "being reported as parameter-efficient"
        )

    before_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, module, parent, cls_name in to_wrap:
        if parent is None:
            raise ValueError(f"cannot wrap the root module {name!r}")
        wrapped = LoRALayer(module, config.rank, config.alpha, config.dropout)
        setattr(parent, name.rsplit(".", 1)[-1], wrapped)
        del cls_name

    if config.freeze_base:
        # The stricter setting: every non-LoRA parameter is frozen, including the detection
        # head and BatchNorm affine parameters. Recorded because this is a *linear probe*
        # with a low-rank adapter, not LoRA fine-tuning, and the two numbers are not
        # comparable.
        for module in model.modules():
            if isinstance(module, LoRALayer):
                continue
            for param in module.parameters(recurse=False):
                param.requires_grad_(False)

    summary = lora_parameter_summary(model)
    summary.update(config.describe())
    summary["n_wrapped_layers"] = len(to_wrap)
    # Recorded rather than dropped: which targets matched is part of what the arm did, and a
    # default list that silently matched only Conv2d describes a different experiment from one
    # that matched both.
    summary["matched_targets"] = sorted(matched_classes)
    summary["unmatched_targets"] = unmatched
    summary["skipped_frozen_layers"] = sorted(skipped_frozen)
    summary["trainable_before"] = int(before_trainable)
    summary["fraction_trainable"] = round(
        summary["trainable_params"] / max(summary["total_params"], 1), 6
    )
    return summary


def merge_lora(model: nn.Module) -> int:
    """Replace every :class:`LoRALayer` in ``model`` with its merged base layer.

    The supported way to collapse an adapted model for inference or for a parameter-count
    comparison: after this the model is a stock detector whose weights happen to have been
    adapted, so it can be saved and loaded by any tool that understands the base graph.

    Args:
        model: The adapted model, modified in place.

    Returns:
        How many adapters were merged.
    """
    replacements: list[tuple[str, nn.Module, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            parent = _parent_of(model, name)
            if parent is None:
                raise ValueError(f"cannot merge the root module {name!r}")
            replacements.append((name, parent, module.merged_base()))
    for name, parent, merged in replacements:
        setattr(parent, name.rsplit(".", 1)[-1], merged)
    return len(replacements)


@contextmanager
def adapter_merged(model: nn.Module) -> Iterator[int]:
    """Temporarily replace every adapter in ``model`` with its merged base layer.

    On exit the adapters are put back exactly as they were, so the caller's module tree and
    its parameters are unchanged. This is what makes an adapted checkpoint *usable*.

    Why the checkpoint needs it
    ---------------------------
    Ultralytics serialises the EMA, and every path that loads a checkpoint back fuses each
    convolution with the BatchNorm that follows it. A :class:`LoRALayer` is not a convolution
    -- it is a wrapper that *contains* one -- so fusion raises
    ``'LoRALayer' object has no attribute 'weight'``. Measured, not assumed: a complete LoRA
    run finished training, wrote ``best.pt``, and then died in Ultralytics' own end-of-run
    validation of that file, with no usable checkpoint produced. Folding the update into the
    base weight and dropping the wrapper for the duration of the save gives a checkpoint that
    is an ordinary detector whose weights happen to have been adapted -- which is also the
    artifact a reader would actually deploy.

    Why it is a context manager rather than a merge
    -----------------------------------------------
    Merging for real would end the run: the adapter's parameters are the thing being
    optimised, and the base weights must keep their pre-merge values or the next step would
    optimise an already-adapted layer. The save is the only moment that has to see the merged
    form.

    Yields:
        How many adapters were merged.
    """
    saved: list[tuple[nn.Module, str, nn.Module]] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALayer):
            parent = _parent_of(model, name)
            if parent is None:
                raise ValueError(f"cannot merge the root module {name!r}")
            saved.append((parent, name.rsplit(".", 1)[-1], module))
    for parent, attr, module in saved:
        setattr(parent, attr, module.merged_base())
    try:
        yield len(saved)
    finally:
        for parent, attr, module in saved:
            setattr(parent, attr, module)


def _parent_of(model: nn.Module, qualified_name: str) -> nn.Module | None:
    """The module that owns ``qualified_name``, or ``None`` for the root itself."""
    if not qualified_name:
        return None
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def lora_parameter_summary(model: nn.Module) -> dict[str, int]:
    """Split a model's parameters into LoRA factors, other trainable, and frozen.

    Reported as three numbers rather than two because "trainable" alone hides the thing a
    reader needs to know: how much of what is trainable is the adapter. A LoRA arm with a
    fully trainable head has a very different parameter budget from one without, and the
    ledger should say which it was.
    """
    lora = other_trainable = frozen = 0
    for module in model.modules():
        is_lora = isinstance(module, LoRALayer)
        for param in module.parameters(recurse=False):
            n = int(param.numel())
            if is_lora:
                lora += n
            elif param.requires_grad:
                other_trainable += n
            else:
                frozen += n
    return {
        "lora_params": lora,
        "other_trainable_params": other_trainable,
        "frozen_params": frozen,
        "trainable_params": lora + other_trainable,
        "total_params": lora + other_trainable + frozen,
    }
