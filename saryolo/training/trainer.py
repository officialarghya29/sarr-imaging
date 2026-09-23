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
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import RANK

from saryolo.nn.model import SARYOLODetectionModel

__all__ = ["SARYOLOTrainer", "SARYOLO", "load_model", "apply_init"]


# ---------------------------------------------------------------------------- initialisation stage
def apply_init(model, weights: str) -> dict:
    """Return the *stated* init stage for a run, and make it real.

    Phase 3 of the master plan asks the RGB-pretraining → SAR-fine-tuning arm to be a
    named baseline whose init stage travels with the result, not a silent accident of
    Ultralytics' ``pretrained`` default. Three stages exist:

    * ``none``   — a fresh build from the model YAML (the random-init control). The
      trainer will do exactly this anyway; saying it is the point.
    * ``coco11`` — ``yolo11s.pt``, the COCO-pretrained stock baseline.
    * a checkpoint path — any stated checkpoint (MSFA-style SAR weights, another
      detector's backbone, …).

    How the weights actually reach the trainer: Ultralytics' ``DetectionTrainer.setup_model``
    rebuilds the model from a YAML *unless* the facade hands it a ready ``nn.Module``,
    and it only loads pretrained weights when it built the model itself from a
    ``.pt`` (or when ``args.pretrained`` is a path). Mutating the facade's inner
    model is therefore **not** a transfer — the trainer would silently rebuild and
    discard it. The honest mechanism is the trainer's own: serialise a fresh build of
    the graph with the intersection of the source weights loaded in, and restart the
    facade from that checkpoint. ``intersect_dicts`` (the same function Ultralytics'
    ``BaseModel.load`` uses) keeps only shape-compatible tensors, so a mismatched
    head degrades to a partially initialised backbone instead of crashing mid-run —
    and the returned summary states exactly how much transferred.

    Returns:
        A summary dict, written into the run record, so a result cannot be quoted
        later without its init stage travelling with it.

    Raises:
        FileNotFoundError: if a checkpoint path does not exist.
        ValueError: on an unstated init mode, or if fewer than one parameter tensor
            transferred from a checkpoint (a typo'd or incompatible checkpoint would
            otherwise run as an accidental random-init while claiming otherwise).
    """
    if weights == "none":
        return {"init": "none", "init_weights": None}

    import torch

    if weights in ("coco11", "coco"):
        from ultralytics.utils.downloads import attempt_download_asset

        source = Path(attempt_download_asset("yolo11s.pt"))
        if not source.is_file():
            raise FileNotFoundError(f"yolo11s.pt could not be resolved by Ultralytics (got {source})")
        stage = "coco11"
    else:
        source = Path(weights)
        if not source.is_file():
            raise FileNotFoundError(f"init.weights checkpoint does not exist: {source}")
        stage = "checkpoint"

    if not hasattr(model, "model") or not hasattr(model, "yaml"):
        raise TypeError(f"apply_init expects an Ultralytics DetectionModel, got {type(model).__name__}")

    # A fresh build of the same graph, then the intersection of the source weights.
    # Built from `model.yaml` (not copied) so the initialised model is provably the
    # same architecture the config asked for, at the config's nc/ch.
    nc = int(model.yaml.get("nc", 1))
    ch = int(model.yaml.get("ch", 3))
    from ultralytics.nn.tasks import DetectionModel, intersect_dicts

    fresh = DetectionModel(deepcopy(model.yaml), ch=ch, nc=nc, verbose=False)
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    src_model = (ckpt.get("ema") or ckpt["model"]) if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not hasattr(src_model, "state_dict"):
        raise ValueError(f"{source} does not contain model weights (no state_dict)")
    csd = src_model.float().state_dict()
    transferred = intersect_dicts(csd, fresh.state_dict())
    if not transferred:
        raise ValueError(
            f"no parameter transferred from {source}: the checkpoint shares no "
            "shape-compatible tensor with this architecture. A run initialised from it "
            "would be a random-init run wearing a pretrained label"
        )
    # Snapshot the fresh build BEFORE loading, so the summary can report what actually
    # changed. Zero-initialised BN buffers (running_mean=0, running_var=1) are
    # shape-compatible with every build and equal by construction, so counting the raw
    # intersection as "transferred" inflates the number ~5x while saying nothing.
    before_sd = {k: v.clone() for k, v in fresh.state_dict().items()}
    fresh.load_state_dict(transferred, strict=False)
    after_sd = fresh.state_dict()
    transferred_changed = sum(1 for k in transferred if not torch.equal(before_sd[k], after_sd[k]))
    del before_sd, after_sd
    if transferred_changed == 0:
        raise ValueError(
            f"every shape-compatible tensor from {source} already equals a fresh build's "
            "initialisation; loading it would change nothing while the record would claim "
            "a pretrained start"
        )

    # Hand the initialised model back through the trainer's own path: a checkpoint of
    # the same graph, which `setup_model` will load instead of rebuilding. The filename
    # is content-derived (transferred-tensor count + source stem), so repeated runs of
    # one config reuse the file instead of littering the directory.
    out = Path("results/init") / f"init_{stage}_{source.stem}_{transferred_changed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": fresh,
        "yaml": model.yaml,
        "names": model.names,
        "epoch": -1,
        "train_args": {"init": stage, "source": str(source)},
    }, out)
    model.ckpt_path = str(out)
    return {
        "init": stage,
        "init_weights": str(source),
        "init_transferred_tensors": transferred_changed,
        "init_checkpoint": str(out),
    }


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
