"""Training integration with Ultralytics.

The SAR-YOLO model and trainer are exposed as a ``task_map`` override so the
whole Ultralytics training stack (dataloading, augmentation, AMP, EMA, LR
schedules, checkpointing, DDP) is reused unchanged. Only two things differ:

1. ``SARYOLODetectionModel`` builds the SAR-aware criterion (Component 7).
2. ``SARYOLOTrainer`` returns that model class instead of the stock one.

Everything else is deliberately stock, so a SAR-YOLO number is directly
comparable to its YOLO baseline run through the identical pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from saryolo.nn.model import SARYOLODetectionModel

__all__ = ["SARYOLOTrainer", "SARYOLO"]


class SARYOLOTrainer(DetectionTrainer):
    """``DetectionTrainer`` that builds the SAR-YOLO model.

    Any Ultralytics argument accepted by the stock trainer is valid here. The
    SAR-loss weights are *not* trainer arguments: they live in the model YAML's
    ``sar_loss`` block (see :class:`SARYOLODetectionModel`). That keeps a run
    reproducible from the committed YAML and avoids depending on Ultralytics'
    argument validation, which rejects unknown keys.
    """

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a :class:`SARYOLODetectionModel` instead of the stock detection model."""
        model = SARYOLODetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def set_model_names_for_load(self, model):
        """Attach dataset class names; the stock helper is reused when available."""
        setter = getattr(super(), "set_model_names_for_load", None)
        return setter(model) if callable(setter) else model


class SARYOLO(YOLO):
    """``YOLO`` facade whose ``detect`` task uses the SAR-YOLO model and trainer.

    Use this exactly like ``ultralytics.YOLO``::

        from saryolo import SARYOLO

        model = SARYOLO("configs/models/full_s.yaml")
        model.train(data="configs/datasets/ssdd.yaml", epochs=100, imgsz=640, seed=0)

    Loading a *trained* checkpoint works through plain ``YOLO`` too, because the
    checkpoint pickles the SAR-YOLO model object and importing ``saryolo``
    registers the layers it needs.
    """

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Ultralytics task registry with the ``detect`` entry replaced."""
        task_map = {key: dict(value) for key, value in super().task_map.items()}
        task_map["detect"]["model"] = SARYOLODetectionModel
        task_map["detect"]["trainer"] = SARYOLOTrainer
        return task_map


def is_saryolo_yaml(model_path: str) -> bool:
    """Whether a model YAML references at least one SAR-YOLO layer."""
    from pathlib import Path

    path = Path(model_path)
    if not path.is_file() or path.suffix not in (".yaml", ".yml"):
        return False
    text = path.read_text()
    return any(
        name in text
        for name in (
            "SARFeatureEnhancement",
            "SpeckleAwareFeatureModule",
            "SARAdaptiveAttention",
            "AdaptiveMultiScaleFusion",
            "SEAttention",
            "ECAAttention",
            "CBAMAttention",
        )
    )


def resolve_model_class(model_path: str):
    """Pick the right Ultralytics facade for a model path.

    A YAML containing SAR layers must go through :class:`SARYOLO` so the SAR-aware
    criterion is built; everything else (including stock baselines and any
    pickled checkpoint) can use plain ``YOLO``.
    """
    from pathlib import Path

    if is_saryolo_yaml(model_path):
        return SARYOLO
    if Path(model_path).suffix in (".yaml", ".yml"):
        return YOLO
    return SARYOLO  # checkpoints: unpickling restores the model class regardless


def load_model(model_path: str, verbose: bool = False):
    """Load a model, choosing the correct facade automatically."""
    return resolve_model_class(model_path)(model_path, verbose=verbose)


def clone_overrides(overrides: dict) -> dict:
    """Deep-copy overrides so Ultralytics cannot mutate a shared config dict."""
    return deepcopy(overrides)
