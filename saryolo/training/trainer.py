"""Training integration with Ultralytics, including optional acquisition metadata."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import RANK

from saryolo.nn.model import SARYOLODetectionModel, set_batch_metadata

__all__ = ["SARYOLOTrainer", "SARYOLO", "load_model", "resolve_model_class", "apply_init"]


def apply_init(model, weights: str) -> dict:
    """Apply a stated initialization stage and return an auditable transfer summary."""
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
    from ultralytics.nn.tasks import DetectionModel, intersect_dicts

    fresh = DetectionModel(
        deepcopy(model.yaml), ch=int(model.yaml.get("ch", 3)), nc=int(model.yaml.get("nc", 1)), verbose=False
    )
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    src_model = (ckpt.get("ema") or ckpt["model"]) if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not hasattr(src_model, "state_dict"):
        raise ValueError(f"{source} does not contain model weights (no state_dict)")
    transferred = intersect_dicts(src_model.float().state_dict(), fresh.state_dict())
    if not transferred:
        raise ValueError(f"no shape-compatible parameter transferred from {source}")
    before = {key: value.clone() for key, value in fresh.state_dict().items()}
    fresh.load_state_dict(transferred, strict=False)
    changed = sum(not torch.equal(before[key], fresh.state_dict()[key]) for key in transferred)
    if not changed:
        raise ValueError(f"all compatible tensors from {source} already equal fresh initialization")

    out = Path("results/init") / f"init_{stage}_{source.stem}_{changed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": fresh, "yaml": model.yaml, "names": model.names, "epoch": -1,
         "train_args": {"init": stage, "source": str(source)}},
        out,
    )
    model.ckpt_path = str(out)
    return {
        "init": stage,
        "init_weights": str(source),
        "init_transferred_tensors": changed,
        "init_checkpoint": str(out),
    }


def _resolve_metadata_path(data: dict[str, Any]) -> Path | None:
    """Resolve optional metadata path from a raw or Ultralytics-resolved dataset config."""
    value = data.get("acquisition_metadata")
    if not value:
        return None
    table_path = Path(value).expanduser()
    if table_path.is_absolute():
        return table_path.resolve()
    yaml_file = data.get("yaml_file")
    candidates = ([Path(yaml_file).resolve().parent / table_path] if yaml_file else [])
    candidates.append(table_path.resolve())
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())


def _metadata_config(data: dict[str, Any]) -> dict[str, Any]:
    """Load the acquisition table from Ultralytics' resolved dataset config."""
    from saryolo.data.metadata import MetadataTable

    table_path = _resolve_metadata_path(data)
    if table_path is None:
        return {}
    if not table_path.is_file():
        raise FileNotFoundError(f"dataset acquisition_metadata table does not exist: {table_path}")
    return {"path": table_path, "table": MetadataTable.load(table_path)}


def metadata_augmentation_overrides(data_yaml: str | Path | None) -> dict[str, float]:
    """Disable image mixing when metadata describes one acquisition per input image."""
    if not data_yaml:
        return {}
    import yaml

    path = Path(data_yaml).expanduser()
    if not path.is_file():
        return {}
    config = yaml.safe_load(path.read_text()) or {}
    if not config.get("acquisition_metadata"):
        return {}
    # Mosaic/MixUp/CutMix/CopyPaste combine acquisitions into one image and invalidate a
    # single per-image descriptor. Spatial/photometric transforms remain enabled.
    return {"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}


def _model_has_conditioner(cfg: Any) -> bool:
    """Whether a model config graph contains the acquisition adapter module."""
    import yaml

    config = yaml.safe_load(Path(cfg).read_text()) if isinstance(cfg, (str, Path)) else cfg
    return isinstance(config, dict) and any(
        len(row) > 2 and row[2] == "AcquisitionConditionedAdapter"
        for section in ("backbone", "head")
        for row in config.get(section, [])
    )


def _metadata_descriptor(table, vocabularies, stem: str, field_mask=None) -> dict[str, torch.Tensor]:
    """Convert one sourced record to CPU tensors, ready for DataLoader collation.

    ``field_mask`` implements the missing-metadata degradation study (master workflow, the
    metadata ablation): a run can declare that only some acquisition fields are *delivered*
    even though the table carries them all. Masking zeroes the value and the availability flag
    together -- and forces a categorical id to the reserved unknown index -- so a withheld
    field is encoded exactly like a field that was never recorded, which is what makes "the
    model degrades gracefully without metadata" a measurable claim instead of a hope.
    """
    from saryolo.data.field_mask import apply_metadata_mask
    from saryolo.data.metadata import encode_metadata

    continuous, categorical, availability = encode_metadata(table.get(stem), vocabularies)
    descriptor = {
        "continuous": torch.tensor(continuous, dtype=torch.float32),
        "categorical": torch.tensor(categorical, dtype=torch.long),
        "availability": torch.tensor(availability, dtype=torch.float32),
    }
    return apply_metadata_mask(descriptor, field_mask)


def data_config_field_mask(data: dict[str, Any]):
    """The field mask a dataset config requests, validated and ready to apply.

    ``metadata_fields`` in the data YAML names the acquisition fields a run *delivers* to the
    adapter. The distinction from the adapter's own ``fields`` argument matters: that selects
    which fields the *module* reads (a graph-level ablation), this selects which fields the
    *data* supplies (a deployment-level degradation). The full-model arms of the missing-
    metadata study need the second, because a graph that only ever reads ``sensor`` has no
    resolution to lose.
    """
    from saryolo.data.field_mask import metadata_field_mask, resolve_metadata_fields

    kept = resolve_metadata_fields(data.get("metadata_fields"))
    return metadata_field_mask(kept)


def encode_prediction_metadata(
    paths, table, vocabularies, allow_unknown: bool = False
) -> dict[str, torch.Tensor]:
    """Encode a predictor batch in path order, preserving one acquisition per image.

    ``allow_unknown`` chooses between the two legitimate meanings of a missing row. Prediction
    refuses by default: an image whose acquisition cannot be established would be scored on an
    assumption, and the deployment path should say so rather than guess. A *diagnosis* pass is
    the opposite case -- an image with no recorded parameters is itself one of the conditions
    being studied, so it is encoded as the explicit unknown record instead.
    """
    stems = [Path(path).stem for path in paths]
    missing = sorted(set(stems) - set(table.entries))
    if missing and not allow_unknown:
        raise ValueError(f"prediction metadata has no row for {len(missing)} image(s), e.g. {missing[:3]}")
    rows = [_metadata_descriptor(table, vocabularies, stem) for stem in stems]
    return {field: torch.stack([row[field] for row in rows], dim=0) for field in rows[0]}


class AcquisitionMetadataPredictor(DetectionPredictor):
    """Detection predictor that conditions each image batch from its acquisition table."""

    def setup_model(self, model, verbose: bool = True):
        """Keep the native SAR model accessible for setting its shared batch context."""
        super().setup_model(model, verbose=verbose)
        native = getattr(self.model, "model", None)
        if native is None or not getattr(native, "n_conditioned_adapters", 0):
            raise ValueError("metadata-conditioned inference requires a checkpoint with an acquisition adapter")

    def preprocess(self, im):
        processed = super().preprocess(im)
        paths = self.batch[0]
        metadata = encode_prediction_metadata(paths, self.metadata_table, self.metadata_vocabularies)
        native = getattr(self.model, "model", None)
        if native is None or not set_batch_metadata(native, metadata):
            raise RuntimeError("metadata was encoded but no acquisition adapter consumed it")
        return processed


class AcquisitionMetadataDetectionDataset(YOLODataset):
    """Detection dataset that adds one descriptor after normal per-image transforms."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Ultralytics Format intentionally emits only model fields, so attach metadata after all
        # transforms. Mixing transforms are disabled for conditioned training.
        sample = super().__getitem__(index)
        stem = Path(self.im_files[index]).stem
        try:
            sample["metadata"] = self._acquisition_rows[stem]
        except KeyError as exc:
            raise KeyError(f"acquisition metadata has no row for image stem {stem!r}") from exc
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Use stock detection collation, then stack the per-image metadata tensors."""
        records = [sample.pop("metadata") for sample in batch]
        result = YOLODataset.collate_fn(batch)
        result["metadata"] = {
            field: torch.stack([record[field] for record in records], dim=0)
            for field in ("continuous", "categorical", "availability")
        }
        return result


def _attach_metadata(dataset: YOLODataset, table, vocabularies, field_mask=None) -> None:
    """Join a metadata table to a constructed dataset without changing its sample order."""
    if not isinstance(dataset, YOLODataset):
        raise TypeError(f"metadata conditioning supports YOLO detection datasets, got {type(dataset).__name__}")
    stems = [Path(path).stem for path in dataset.im_files]
    if len(stems) != len(set(stems)):
        raise ValueError("image stems are not unique in this split; acquisition metadata joins would be ambiguous")
    missing = sorted(set(stems) - set(table.entries))
    if missing:
        raise ValueError(
            f"acquisition metadata has no row for {len(missing)} dataset image(s) "
            f"(e.g. {missing[:3]}); rebuild the table for this exact split"
        )
    dataset.__class__ = AcquisitionMetadataDetectionDataset
    dataset._acquisition_rows = {
        stem: _metadata_descriptor(table, vocabularies, stem, field_mask=field_mask) for stem in stems
    }


def _native_module(handle) -> Any | None:
    """The ``nn.Module`` that actually consumes metadata, from whatever a validator was handed.

    Ultralytics hands a validator two different things depending on who called it, and only one
    of them is the module the adapters live in:

    * from a trainer, ``trainer.ema.ema`` -- already the ``nn.Module``;
    * from ``Trainer.final_eval`` / a standalone ``val``, a *path* to a ``.pt`` checkpoint, which
      ``BaseValidator`` wraps in an :class:`AutoBackend`. ``AutoBackend`` is itself an
      ``nn.Module`` but it is not the model: it forwards ``.model`` to the backend it wraps.

    Reading ``.model`` off the ``AutoBackend`` is the same resolution
    :class:`AcquisitionMetadataPredictor` uses, and it matters here because a ``Path`` carries no
    ``metadata_context``: ``set_batch_metadata`` would return ``False`` and the run would report
    numbers from an unconditioned model while looking perfectly healthy.
    """
    from ultralytics.nn.autobackend import AutoBackend

    if handle is None or isinstance(handle, (str, Path)):
        return None
    if isinstance(handle, AutoBackend):
        return getattr(handle, "model", None)
    return handle


def _checkpoint_model(weights: str | Path):
    """The model object inside a saved checkpoint, without building an inference backend for it.

    Used only for reading *attributes* off a checkpoint (its frozen vocabulary, whether it is
    conditioned). The weights are deliberately not loaded onto a device here -- the inference
    path is ``AutoBackend``'s job -- so this stays cheap enough to call while a validator is
    still deciding how to build its dataloader.
    """
    path = Path(weights)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        # ``ema`` first: it is the weight set validation is scored on.
        return payload.get("ema") or payload.get("model")
    return payload


def _checkpoint_metadata_vocabularies(weights: str | Path) -> dict | None:
    """Read the training-only metadata vocabularies out of a checkpoint.

    Needed when a conditioned model is validated standalone, because the val dataloader is built
    before the inference model exists and the vocabulary must come from the checkpoint rather
    than from a live training run. Nothing is rebuilt here: a vocabulary re-derived at validation
    time would renumber the embedding table under trained weights.
    """
    return getattr(_checkpoint_model(weights), "metadata_vocabularies", None)


class AcquisitionDetectionValidator(DetectionValidator):
    """Use metadata-bearing batches during trainer and standalone validation."""

    def __call__(self, trainer=None, model=None, **kwargs):
        self._metadata_checkpoint = None
        if trainer is not None:
            from ultralytics.utils.torch_utils import unwrap_model

            ema = getattr(trainer, "ema", None)
            source = ema.ema if ema is not None else trainer.model
            self._metadata_model = _native_module(unwrap_model(source))
        else:
            # A checkpoint path is the common case here (``final_eval`` validates ``best.pt``).
            # It is remembered rather than stored as the model, so ``get_dataloader`` can take
            # the frozen vocabularies from it and ``init_metrics`` can install the real module.
            self._metadata_checkpoint = model if isinstance(model, (str, Path)) else None
            self._metadata_model = _native_module(model)
        return super().__call__(trainer=trainer, model=model, **kwargs)

    def init_metrics(self, model):
        """Capture the module the inference pass really uses, before any batch is preprocessed.

        This is the hook that makes the checkpoint path work: by the time the first batch is
        preprocessed, the model exists (in training it is the EMA, standalone it is the module
        behind the ``AutoBackend``), and this is where the validator can see it.
        """
        native = _native_module(model)
        if native is not None:
            self._metadata_model = native
        return super().init_metrics(model)

    def get_dataloader(self, dataset_path, batch_size):
        from ultralytics.data import build_dataloader

        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        table_config = _metadata_config(self.data)
        model = getattr(self, "_metadata_model", None)
        serialized = getattr(model, "metadata_vocabularies", None)
        if table_config:
            if not serialized and self._metadata_checkpoint is not None:
                serialized = _checkpoint_metadata_vocabularies(self._metadata_checkpoint)
            if not serialized:
                raise ValueError(
                    "conditioned validation needs the training-only metadata vocabulary: the "
                    "checkpoint carries none and no training run is attached. Validate through the "
                    "trainer, or pass a checkpoint trained with acquisition metadata."
                )
            from saryolo.data.metadata import Vocabulary

            vocabularies = {key: Vocabulary.from_dict(value) for key, value in serialized.items()}
            _attach_metadata(
                dataset, table_config["table"], vocabularies,
                field_mask=data_config_field_mask(self.data),
            )
        else:
            conditioned = getattr(model, "n_conditioned_adapters", 0)
            if not conditioned and self._metadata_checkpoint is not None:
                conditioned = getattr(_checkpoint_model(self._metadata_checkpoint), "n_conditioned_adapters", 0)
            if conditioned:
                raise ValueError("conditioned model validation requires acquisition_metadata in the dataset YAML")
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if self.training else self.args.workers * 2,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile and self.training,
            device=self.device,
        )

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        model = getattr(self, "_metadata_model", None)
        # A control arm carries no adapter: metadata in the batch cannot affect it, and that is the
        # intended comparison, so this is not an error. Only a *conditioned* model must be fed.
        if model is None or not getattr(model, "n_conditioned_adapters", 0):
            return batch
        if batch.get("metadata") is None:
            raise RuntimeError(
                "a conditioned model reached validation without acquisition metadata in the batch; "
                "the reported metrics would describe an unconditioned model"
            )
        if not set_batch_metadata(model, batch.get("metadata")):
            raise RuntimeError(
                "acquisition metadata was supplied but no adapter in the validation model consumed "
                "it; the model was not attached to its metadata context"
            )
        return batch


class SARYOLOTrainer(DetectionTrainer):
    """DetectionTrainer with sourced metadata conditioning when configured."""

    def __init__(self, cfg="default.yaml", overrides: dict[str, Any] | None = None, _callbacks=None):
        self._metadata_table = None
        self._metadata_vocabularies = None
        self._metadata_vocabularies_serialized = None
        overrides = dict(overrides or {})
        overrides.update(metadata_augmentation_overrides(overrides.get("data")))
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model_cfg = cfg
        table_config = _metadata_config(self.data)
        if table_config:
            from saryolo.data.conditioning import prepare_conditioned_config, training_metadata_vocabularies

            prepared = training_metadata_vocabularies(self.data)
            if prepared is None:
                raise ValueError("acquisition_metadata is set but training metadata could not be prepared")
            self._metadata_vocabularies, self._metadata_vocabularies_serialized = prepared
            self._metadata_table = table_config["table"]
            if _model_has_conditioner(cfg):
                model_cfg = prepare_conditioned_config(cfg, self._metadata_vocabularies)
        model = SARYOLODetectionModel(
            model_cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        if self._metadata_vocabularies_serialized is not None:
            model.metadata_vocabularies = self._metadata_vocabularies_serialized
        elif getattr(model, "n_conditioned_adapters", 0):
            raise ValueError("conditioned model requires acquisition_metadata in the dataset YAML")
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        table_config = _metadata_config(self.data)
        if not table_config:
            if getattr(self.model, "n_conditioned_adapters", 0):
                raise ValueError("conditioned model requires acquisition_metadata in the dataset YAML")
            return dataset

        from saryolo.data.conditioning import training_metadata_vocabularies

        if mode == "train":
            if any(float(getattr(self.args, key, 0.0) or 0.0) > 0.0
                   for key in ("mosaic", "mixup", "cutmix", "copy_paste")):
                raise ValueError(
                    "acquisition metadata cannot be paired with image-mixing augmentations; "
                    "the trainer should have disabled them before saving run arguments"
                )
            if self._metadata_vocabularies is None:
                prepared = training_metadata_vocabularies(self.data)
                if prepared is None:
                    raise ValueError("training-only metadata vocabularies must be fit before dataset construction")
                self._metadata_vocabularies, self._metadata_vocabularies_serialized = prepared
            self._metadata_table = table_config["table"]
        elif self._metadata_table is None or self._metadata_vocabularies is None:
            raise ValueError("training metadata vocabulary must be built before validation")
        _attach_metadata(
            dataset, self._metadata_table, self._metadata_vocabularies,
            field_mask=data_config_field_mask(self.data),
        )
        return dataset

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Set model context before each training forward; validation uses its own validator hook."""
        batch = super().preprocess_batch(batch)
        from ultralytics.utils.torch_utils import unwrap_model

        set_batch_metadata(unwrap_model(self.model), batch.get("metadata"))
        return batch

    def validate(self):
        """Ensure EMA carries the training-only vocabulary needed by the metadata validator."""
        if self.ema is not None and self._metadata_vocabularies_serialized is not None:
            self.ema.ema.metadata_vocabularies = self._metadata_vocabularies_serialized
        return super().validate()

    def get_validator(self):
        return AcquisitionDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks
        )

    def set_model_names_for_load(self, model):
        setter = getattr(super(), "set_model_names_for_load", None)
        return setter(model) if callable(setter) else model


def resolve_model_class(model_path: str):
    """Pick the appropriate facade for stock/custom YAMLs and checkpoints."""
    path = Path(model_path)
    if path.suffix in (".yaml", ".yml") and not is_saryolo_yaml(model_path):
        return YOLO
    return SARYOLO


def load_model(model_path: str, verbose: bool = False):
    """Load a stock or SAR-YOLO model, selecting its facade from the model path."""
    return resolve_model_class(model_path)(model_path, verbose=verbose)


def clone_overrides(overrides: dict) -> dict:
    """Deep-copy overrides before passing them to Ultralytics, which may mutate them."""
    return deepcopy(overrides)


class SARYOLO(YOLO):
    """YOLO facade whose detect task uses the SAR-YOLO model and trainer."""

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        task_map = {key: dict(value) for key, value in super().task_map.items()}
        task_map["detect"]["model"] = SARYOLODetectionModel
        task_map["detect"]["trainer"] = SARYOLOTrainer
        task_map["detect"]["validator"] = AcquisitionDetectionValidator
        return task_map


def is_saryolo_yaml(model_path: str) -> bool:
    """Whether a YAML references a custom SAR-YOLO or conditioning layer."""
    path = Path(model_path)
    if not path.is_file() or path.suffix not in (".yaml", ".yml"):
        return False
    names = (
        "SARFeatureEnhancement", "SpeckleAwareFeatureModule", "SARAdaptiveAttention",
        "AdaptiveMultiScaleFusion", "SEAttention", "ECAAttention", "CBAMAttention",
        "AcquisitionConditionedAdapter", "TargetPriorModulation", "SpatialFrequencyRepresentation",
        "ContextAggregation", "TargetAwareRefinement", "SARInputAdapter",
    )
    return any(name in path.read_text() for name in names)
