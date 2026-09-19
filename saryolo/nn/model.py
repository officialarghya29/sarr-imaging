"""SAR-YOLO model class: a ``DetectionModel`` whose criterion is the SAR-aware loss."""

from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel

from .losses import build_criterion

__all__ = ["SARYOLODetectionModel"]


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
