"""SAR-YOLO model class: a ``DetectionModel`` whose criterion is the SAR-aware loss."""

from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel

from .losses import build_criterion

__all__ = ["SARYOLODetectionModel", "attach_metadata_context"]

#: Batch key carrying per-image acquisition metadata as a ``(continuous, categorical,
#: availability)`` triple of tensors.
METADATA_KEY = "metadata"


def attach_metadata_context(model) -> int:
    """Give every conditioned adapter in ``model`` one shared metadata context.

    Ultralytics resolves a custom layer with a single input, so metadata cannot be a second graph
    input. Each adapter instead reads from a context object the model owns, and this walk wires
    them up once at construction -- which is why a newly added adapter module needs no further
    plumbing: it is picked up here.

    Returns:
        How many adapters were attached. Zero is a valid answer for a model with no conditioning.
    """
    from .modules.conditioning import AcquisitionConditionedAdapter, MetadataContext

    context = MetadataContext()
    attached = 0
    for module in model.modules():
        if isinstance(module, AcquisitionConditionedAdapter):
            module.context = context
            attached += 1
    model.metadata_context = context
    return attached


def set_batch_metadata(model, metadata) -> bool:
    """Point the model's metadata context at one batch of descriptors.

    Args:
        model: A model carrying a context (see :func:`attach_metadata_context`).
        metadata: Either a ``(continuous, categorical, availability)`` triple, or ``None`` to
            mark the batch as an unknown acquisition.

    Returns:
        Whether the model actually conditions on anything. ``False`` means it has no adapter, so
        the metadata it was handed cannot affect it -- which is worth knowing, because otherwise
        a caller could believe it had supplied conditioning that nothing consumes.
    """
    context = getattr(model, "metadata_context", None)
    if context is None or not getattr(model, "n_conditioned_adapters", 0):
        return False
    if metadata is None:
        context.clear()
    else:
        continuous, categorical, availability = metadata
        context.set(continuous, categorical, availability)
    return True


class SARYOLODetectionModel(DetectionModel):
    """``DetectionModel`` that reads its SAR-loss configuration from the model YAML.

    The configuration lives in a ``sar_loss`` block *inside the model YAML* rather
    than in the trainer's argument namespace. That keeps a run reproducible from
    the committed YAML alone, and avoids depending on ultralytics' argument
    validation, which rejects unknown keys passed to ``model.train()``.

    Example ``sar_loss`` block::

        sar_loss:
          w_sep: 0.2
          w_small: 0.5
          w_smooth: 0.05
          margin: 1.0
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One shared context for every conditioned adapter in this graph. Done after the parent
        # builds the layers, because the layers are what we are walking.
        self.n_conditioned_adapters = attach_metadata_context(self)

    def init_criterion(self):
        """Return the SAR-aware criterion, using the YAML ``sar_loss`` block if present."""
        # During training the Ultralytics trainer attaches `args` (the resolved
        # hyperparameters) before the criterion is built. A bare model -- one built
        # directly for unit tests or FLOPs profiling -- has no `args`, and
        # v8DetectionLoss reads `model.args.box/cls/dfl`. Supplying the documented
        # Ultralytics defaults keeps criterion construction possible in those cases
        # instead of failing with a confusing AttributeError.
        if getattr(self, "args", None) is None:
            from types import SimpleNamespace

            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

        sar_cfg: dict[str, Any] = {}
        if isinstance(self.yaml, dict):
            sar_cfg = self.yaml.get("sar_loss") or {}
        if not isinstance(sar_cfg, dict):
            raise TypeError(f"model YAML 'sar_loss' must be a mapping, got {type(sar_cfg).__name__}")
        return build_criterion(self, sar_cfg)

    # ---------------------------------------------------- representation consistency
    def loss(self, batch, preds=None):
        """``DetectionModel.loss`` plus SEC. 4's consistency view, when it is enabled.

        With ``w_consistency = 0`` (the default) this is bit-for-bit the parent behaviour: one
        forward pass, one criterion call. Only when the weight is non-zero does a *second*
        forward run on a degraded copy of the same batch, and its feature maps are handed to
        the criterion so the representation is penalised for changing.

        The second pass deliberately does **not** update BatchNorm running statistics. A second
        forward in training mode would otherwise move every BN buffer twice per step, so
        enabling consistency would silently change the normalisation of the whole network --
        and the ablation would be measuring that, not the consistency term. The BN modules are
        switched to eval mode for the perturbed pass and restored afterwards.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        # Acquisition conditioning must be in place *before* the forward pass, because the
        # adapters read it during that pass. Set from the batch when a loader supplies it, and
        # otherwise marked unknown -- the same state the modules encode for a record with no
        # fields, so an unconditioned dataset is a trained input rather than a bypassed path.
        set_batch_metadata(self, batch.get(METADATA_KEY))
        preds = self.forward(batch["img"]) if preds is None else preds

        weight = getattr(self.criterion, "w_consistency", 0.0)
        if weight and "feats" in preds and batch.get("img") is not None:
            from torch import nn

            from saryolo.nn.losses import perturb_batch

            cfg = getattr(self.criterion, "sar_cfg", {})
            # The draw changes each step so the model is not fitted to one fixed noise
            # realisation, but the sequence is a pure function of the call order, which the
            # training seed already fixes.
            self._consistency_step = getattr(self, "_consistency_step", 0) + 1
            perturbed = perturb_batch(
                batch["img"],
                kind=str(cfg.get("consistency_kind", "speckle")),
                severity=float(cfg.get("consistency_severity", 4.0)),
                seed=self._consistency_step,
            )
            bns = [m for m in self.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            was_training = [m.training for m in bns]
            for m in bns:
                m.eval()
            try:
                preds_b = self.forward(perturbed)
            finally:
                for m, state in zip(bns, was_training, strict=True):
                    m.train(state)
            if "feats" in preds_b:
                self.criterion.set_views(preds["feats"], preds_b["feats"])
        return self.criterion(preds, batch)
