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
