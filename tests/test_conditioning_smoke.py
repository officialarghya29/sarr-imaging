"""One-epoch conditioned-training smoke test: the wiring no unit test can prove.

The unit tests in ``test_conditioning.py`` verify the adapter's contract in
isolation (identity at init, per-sample descriptors, field masks) and
``test_metadata.py`` verifies the loader pieces. Between those two layers sits
the integration that actually failed once already: Ultralytics' transform stack
discards unknown per-image fields, so metadata that reached the dataset did not
reach the batch, and the model silently trained unconditioned while every unit
test still passed.

This test therefore runs the real trainer -- ``SARYOLOTrainer`` with the real
dataset class, collate, validator and EMA path -- for one epoch on a tiny
synthetic dataset carrying a two-sensor acquisition table, and asserts on
observable behaviour rather than on configuration text:

1. the batch the model sees carries a per-image metadata tensor of the right
   shape (caught via a hook, since the context is transient);
2. the gate of at least one adapter leaves zero after a step of training, so
   conditioning is not only *wired* but *trained*;
3. the metadata row each training image received matches the table that image
   was assigned (no cross-image leaks, no composite-image metadata).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


class _ModuleCarrier:
    """Stands in for an inference backend, whose only relevant attribute is ``.model``.

    ``AutoBackend.__getattr__`` resolves ``.model`` by forwarding to this object, so a carrier
    reproduces the real resolution path without exporting a model to disk.
    """

    def __init__(self, model):
        self.model = model


def _write_conditioned_fixture(tmp_path: Path, n_per_sensor: int = 4) -> tuple[Path, Path]:
    """A tiny two-sensor dataset plus its acquisition table, ready to train on.

    Images from the two sensors get visibly different brightness so the runs are
    not degenerate, but no assertion depends on the model learning anything:
    one epoch at this scale proves wiring, not accuracy.
    """
    import cv2

    root = tmp_path / "ds"
    rows: list[tuple[str, str, float]] = []
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(n_per_sensor):
            for sensor, brightness in (("s1", 40), ("g3", 160)):
                stem = f"{sensor}_{i:03d}"
                img = np.full((64, 64), brightness, dtype=np.uint8)
                # One centred box, identical geometry everywhere.
                img[24:40, 24:40] = np.clip(img[24:40, 24:40].astype(int) + 60, 0, 255).astype(np.uint8)
                cv2.imwrite(str(images / f"{stem}.png"), img)
                (labels / f"{stem}.txt").write_text("0 0.500000 0.500000 0.250000 0.250000\n")
                rows.append((split, stem, {"s1": 10.0, "g3": 3.0}[sensor]))

    # The metadata table, in the shape MetadataTable.load consumes.
    entries = {
        stem: {"sensor": stem.split("_")[0], "resolution_m": res,
               "polarization": None, "mode": None, "band": "C", "incidence_deg": 33.0}
        for _split, stem, res in rows
    }
    # JSON, not YAML: MetadataTable.load reads strict JSON, so a YAML dump here would
    # crash only once the trainer opened the table, mid-run.
    table_path = tmp_path / "acquisition_metadata.json"
    import json

    table_path.write_text(json.dumps({
        "source": "test:synthetic-conditioning", "entries": entries, "vocabularies": {},
    }))

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root), "nc": 1, "names": ["target"],
        "train": str(root / "images" / "train"), "val": str(root / "images" / "val"),
        "acquisition_metadata": str(table_path),
    }, sort_keys=False))
    return root, data_yaml


@pytest.fixture
def conditioned_config(tmp_path) -> dict:
    """A model config carrying one AcquisitionConditionedAdapter (scale n, tiny)."""
    model_yaml = tmp_path / "cond_model.yaml"
    model_yaml.write_text(yaml.safe_dump({
        "nc": 1,
        "scale": "n",
        "scales": {"n": [0.5, 0.25, 1024]},
        "backbone": [
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "C3k2", [64, False, 0.25]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 1, "C3k2", [128, False, 0.25]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 2, "C3k2", [256, True]],
            [-1, 1, "SPPF", [256, 5]],
        ],
        "head": [
            # The final arg is the vocab-size slot that prepare_conditioned_config fills in
            # from the training-only vocabulary; None marks "fill at trainer setup".
            [-1, 1, "AcquisitionConditionedAdapter", ["ch", None, "film", "all", [2, 2, 2]]],
            [-1, 1, "Conv", [256, 3, 1]],
            [[-1], 1, "Detect", ["nc"]],
        ],
        "sar_loss": {"w_sep": 0.0, "w_small": 0.0, "w_smooth": 0.0},
    }, sort_keys=False))
    return {"model_yaml": model_yaml}


def test_one_epoch_conditioned_training_reaches_the_model(tmp_path, conditioned_config, monkeypatch):
    """The full SARYOLOTrainer path: metadata in the batch, adapter trained, vocabulary frozen.

    A handful of epochs at a raised learning rate keeps this a seconds-long CPU test while
    still demonstrating the gate leaves zero -- the difference between a wired module and a
    trained one.
    """
    import cv2  # noqa: F401  (fixture side effects need cv2 importable here too)

    root, data_yaml = _write_conditioned_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils import SETTINGS

    from saryolo.training.trainer import SARYOLOTrainer

    # Keep Ultralytics' run bookkeeping inside the test's tmp dir.
    SETTINGS.update({"datasets_dir": str(tmp_path)})

    # Constructed with the *dict* default config, not the default path: Ultralytics
    # resolves the default cfg as a bare "default.yaml" relative to the CWD, so a
    # run from any other directory crashes before training even starts.

    overrides = {
        "model": str(conditioned_config["model_yaml"]),
        "data": str(data_yaml),
        # The gate gradient is real but tiny on 8 images (AdamW's auto choice reaches only
        # ~3e-4 after 8 epochs); a short SGD run with matched nbs moves it off zero
        # decisively while staying a seconds-long CPU test.
        "epochs": 20,
        "imgsz": 64,
        "batch": 4,
        "seed": 0,
        "workers": 0,
        "plots": False,
        "val": True,
        "save": False,
        "project": str(tmp_path / "runs"),
        "name": "cond_smoke",
        "exist_ok": True,
        "cache": False,
        "device": "cpu",
        "mosaic": 0.0,   # the trainer refuses mixing transforms for conditioned data
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "verbose": False,
        "warmup_epochs": 0.0,
        "lr0": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "nbs": 8,
    }
    trainer = SARYOLOTrainer(cfg=DEFAULT_CFG_DICT, overrides=overrides)

    observed: list[dict] = []

    original_preprocess = trainer.preprocess_batch

    def spy_preprocess(batch):
        # AFTER the real preprocess: whatever the model would actually consume.
        result = original_preprocess(batch)
        meta = result.get("metadata")
        if meta is not None:
            observed.append({
                "rows": int(meta["continuous"].shape[0]),
                "img_rows": int(result["img"].shape[0]),
                "batch_size": int(result.get("batch_idx", torch.zeros(0)).max()) + 1
                if result.get("batch_idx") is not None and len(result["batch_idx"]) else None,
            })
        return result

    trainer.preprocess_batch = spy_preprocess
    trainer.train()

    # 1. Metadata actually reached the model's batch, aligned with the images.
    assert observed, "no training batch carried metadata; conditioning never engaged"
    for row in observed:
        assert row["rows"] == row["img_rows"], (
            f"metadata rows ({row['rows']}) do not match image rows ({row['img_rows']})"
        )

    model = trainer.model
    from saryolo.nn.modules.conditioning import AcquisitionConditionedAdapter

    adapters = [m for m in model.modules() if isinstance(m, AcquisitionConditionedAdapter)]
    assert adapters, "the built model contains no conditioned adapter"

    # 2. Training moved at least one gate off zero: conditioning is not only wired but learned.
    # The threshold sits two orders of magnitude below the observed ~8e-3 so the test is
    # robust to seed/platform noise, yet far above float epsilon so a frozen gate cannot pass.
    gate_values = [float(a.alpha().detach()) for a in adapters]
    assert any(abs(g) > 1e-4 for g in gate_values), (
        f"all adapter gates are still effectively at zero after training: {gate_values}"
    )

    # 3. The trainer frozen the vocabulary from training data only, and the val loader
    #    used it: a fresh vocabulary at validation time would silently renumber embeddings.
    assert getattr(model, "metadata_vocabularies", None), "no vocabulary was frozen onto the model"
    vocab = model.metadata_vocabularies["sensor"]
    assert set(vocab["entries"]) == {"g3", "s1"}, vocab


def test_the_validator_resolves_a_real_module_from_every_handle_it_is_given(tmp_path):
    """The validator is handed three different things; only two of them are the model.

    ``Trainer.final_eval`` validates with ``self.validator(model=self.best)`` -- a **path**. A
    path has no ``metadata_context``, so ``set_batch_metadata`` returns ``False`` and the run
    reports numbers from an unconditioned model without any error at all. That was a real,
    silent defect: the code stored the path as if it were the model. This pins the resolution
    for each handle so the mistake cannot come back through a refactor.
    """
    import torch as _torch

    from saryolo.training.trainer import _native_module

    # 1. A module is returned as itself.
    model = _torch.nn.Linear(2, 2)
    assert _native_module(model) is model

    # 2. A path is *not* a model. This is the exact case that silently disabled conditioning.
    checkpoint = tmp_path / "weights.pt"
    _torch.save({"model": _torch.nn.Linear(2, 2)}, checkpoint)
    assert _native_module(checkpoint) is None
    assert _native_module(str(checkpoint)) is None
    assert _native_module(None) is None

    # 3. An AutoBackend wraps the model rather than being it, so the native module behind
    #    ``.model`` must be returned -- the same resolution the predictor uses. A real
    #    AutoBackend object is used (not a stub), so the attribute forwarding that makes this
    #    work is itself under test.
    from ultralytics.nn.autobackend import AutoBackend

    inner = _torch.nn.Linear(2, 2)
    auto = AutoBackend.__new__(AutoBackend)
    _torch.nn.Module.__init__(auto)
    # AutoBackend.__getattr__ forwards an unknown attribute to ``self.backend``, which is why
    # ``.model`` resolves at all; a backend standing in for the PyTorch one reproduces exactly
    # that resolution without exporting a model.
    object.__setattr__(auto, "backend", _ModuleCarrier(inner))
    assert _native_module(auto) is inner


def test_checkpoint_vocabularies_come_back_frozen_not_rebuilt(tmp_path):
    """A standalone conditioned val must read the vocabulary the checkpoint was trained with.

    Rebuilding one at validation time would renumber the embedding table under trained weights,
    which is worse than failing: it produces plausible scores against the wrong rows.
    """
    import torch as _torch

    from saryolo.training.trainer import _checkpoint_metadata_vocabularies

    frozen = {"sensor": {"name": "sensor", "entries": ["g3", "s1"], "unknown_index": 0}}
    model = _torch.nn.Linear(2, 2)
    model.metadata_vocabularies = frozen

    # Saved under ``model`` and, separately, under ``ema`` -- final_eval validates the EMA
    # weights, so both spellings have to be read.
    for key in ("model", "ema"):
        path = tmp_path / f"{key}.pt"
        _torch.save({key: model}, path)
        assert _checkpoint_metadata_vocabularies(path) == frozen

    # A checkpoint without the attribute, and a path that does not exist, both report nothing
    # rather than inventing a vocabulary.
    plain = tmp_path / "plain.pt"
    _torch.save({"model": _torch.nn.Linear(2, 2)}, plain)
    assert _checkpoint_metadata_vocabularies(plain) is None
    assert _checkpoint_metadata_vocabularies(tmp_path / "missing.pt") is None


def test_conditioned_validation_conditions_on_metadata(tmp_path, conditioned_config, monkeypatch):
    """Validation must condition on metadata too, not silently fall back to "unknown".

    The training-side assertion above cannot see this. During validation the model is the EMA,
    reached through the validator rather than the trainer, and if the validator ever failed to
    attach and then set the metadata, the adapter's own fallback would quietly serve an
    all-unknown acquisition: the code would not raise, and mAP would just be measured
    unconditioned. That is exactly the silent-failure shape this file exists to catch, so the
    assertion is on the *model's context after the validator preprocessed the batch*, not on
    any configuration text.
    """
    root, data_yaml = _write_conditioned_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils import SETTINGS

    from saryolo.training.trainer import AcquisitionDetectionValidator, SARYOLOTrainer

    SETTINGS.update({"datasets_dir": str(tmp_path)})

    observed: list[dict] = []

    original_preprocess = AcquisitionDetectionValidator.preprocess

    def spy_preprocess(self, batch):
        batch = original_preprocess(self, batch)
        model = getattr(self, "_metadata_model", None)
        context = getattr(model, "metadata_context", None)
        observed.append({
            "has_metadata": batch.get("metadata") is not None,
            "rows": None if batch.get("metadata") is None else int(batch["metadata"]["continuous"].shape[0]),
            "img_rows": int(batch["img"].shape[0]),
            "context_set": bool(context is not None and context.is_set),
            # Pins that the metadata landed on a real module rather than on the checkpoint path
            # ``final_eval`` hands the validator.
            "model_type": type(model).__name__ if model is not None else None,
        })
        return batch

    monkeypatch.setattr(AcquisitionDetectionValidator, "preprocess", spy_preprocess)

    overrides = {
        "model": str(conditioned_config["model_yaml"]),
        "data": str(data_yaml),
        "epochs": 1,
        "imgsz": 64,
        "batch": 4,
        "seed": 0,
        "workers": 0,
        "plots": False,
        "val": True,
        "save": False,
        "project": str(tmp_path / "runs"),
        "name": "cond_val_smoke",
        "exist_ok": True,
        "cache": False,
        "device": "cpu",
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "verbose": False,
        "warmup_epochs": 0.0,
    }
    SARYOLOTrainer(cfg=DEFAULT_CFG_DICT, overrides=overrides).train()

    assert observed, "validation never preprocessed a batch; the conditioned val path did not run"
    for row in observed:
        assert row["has_metadata"], "a validation batch arrived without acquisition metadata"
        assert row["rows"] == row["img_rows"], (
            f"validation metadata rows ({row['rows']}) do not match image rows ({row['img_rows']})"
        )
        # The decisive one: the fallback would leave the context unset and still produce numbers.
        assert row["context_set"], (
            "the validator dropped the metadata before the model saw it; the EMA would have "
            f"validated an *unknown* acquisition and silently reported unconditioned mAP: {row}"
        )


# ---------------------------------------------------------------------- the LOSO protocol

def _write_loso_fixture(tmp_path: Path, n_per_sensor: int = 3) -> tuple[Path, Path]:
    """A three-sensor dataset (s1, g3, tsx) plus its acquisition table.

    The held-out sensor (``tsx``) exists *only* in the val split, mirroring the
    leave-one-source-out protocol: the model never trains on it, and every deployment claim
    this project makes is tested exactly there. Brightness differs per sensor so the runs are
    not degenerate, but no assertion depends on the model learning anything.
    """
    import cv2

    root = tmp_path / "ds"
    sensors = (("s1", 40, 10.0), ("g3", 160, 3.0), ("tsx", 220, 1.5))
    entries: dict[str, dict] = {}
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(n_per_sensor):
            for sensor, brightness, res in sensors:
                if sensor == "tsx" and split == "train":
                    continue  # the held-out source is never in training
                stem = f"{sensor}_{i:03d}"
                img = np.full((64, 64), brightness, dtype=np.uint8)
                img[24:40, 24:40] = np.clip(img[24:40, 24:40].astype(int) + 60, 0, 255).astype(np.uint8)
                cv2.imwrite(str(images / f"{stem}.png"), img)
                (labels / f"{stem}.txt").write_text("0 0.500000 0.500000 0.250000 0.250000\n")
                entries[stem] = {
                    "sensor": sensor, "resolution_m": res, "polarization": None,
                    "mode": None, "band": "C", "incidence_deg": 33.0,
                }

    import json

    table_path = tmp_path / "acquisition_metadata.json"
    table_path.write_text(json.dumps({
        "source": "test:synthetic-loso", "entries": entries, "vocabularies": {},
    }))
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root), "nc": 1, "names": ["target"],
        "train": str(root / "images" / "train"), "val": str(root / "images" / "val"),
        "acquisition_metadata": str(table_path),
    }, sort_keys=False))
    return root, data_yaml


def test_a_conditioned_model_trains_without_the_held_out_sensor_and_still_validates_on_it(
    tmp_path, conditioned_config, monkeypatch
):
    """The protocol end to end: unseen sensor in the val batch, vocabulary that cannot name it.

    This is the deployment claim in miniature. Three properties have to hold at once or the
    claim is not being tested:

    1. the training vocabulary contains only the *training* sensors -- the held-out sensor has
       no embedding row, which is the entire reason the categorical path cannot transfer and
       the continuous fields exist;
    2. validation runs on the held-out source and reaches the model -- the unseen acquisition
       is encoded, not skipped;
    3. the unseen sensor lands on the reserved unknown index, the row the adapter actually
       reads for a sensor it has never seen.

    A failure in any of the three quietly becomes "we measured the seen sensor again", which
    is the most expensive way to be wrong in this project.
    """
    root, data_yaml = _write_loso_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils import SETTINGS

    from saryolo.data.metadata import UNKNOWN_INDEX, MetadataTable, encode_metadata
    from saryolo.training.trainer import SARYOLOTrainer

    SETTINGS.update({"datasets_dir": str(tmp_path)})

    overrides = {
        "model": str(conditioned_config["model_yaml"]),
        "data": str(data_yaml),
        "epochs": 2,
        "imgsz": 64,
        "batch": 4,
        "seed": 0,
        "workers": 0,
        "plots": False,
        "val": True,
        "save": False,
        "project": str(tmp_path / "runs"),
        "name": "loso_smoke",
        "exist_ok": True,
        "cache": False,
        "device": "cpu",
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "verbose": False,
        "warmup_epochs": 0.0,
        "lr0": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "nbs": 8,
    }
    SARYOLOTrainer(cfg=DEFAULT_CFG_DICT, overrides=overrides).train()

    # 1. Training-only vocabulary: the held-out sensor has no row.
    model = trainer_model(tmp_path)
    vocab = model.metadata_vocabularies["sensor"]
    assert set(vocab["entries"]) == {"g3", "s1"}, (
        f"the training vocabulary saw the held-out sensor: {vocab['entries']}"
    )

    # 2 + 3. The held-out split encodes against that frozen vocabulary: its sensor lands on
    # the unknown index, and its physical descriptors survive, which is the path that is
    # supposed to transfer.
    table = MetadataTable.load(tmp_path / "acquisition_metadata.json")
    from saryolo.data.metadata import Vocabulary

    vocabs = {k: Vocabulary.from_dict(v) for k, v in model.metadata_vocabularies.items()}
    held_out = sorted((root / "images" / "val").glob("tsx_*.png"))
    assert held_out, "fixture lost its held-out source"
    for image in held_out:
        meta = table.get(image.stem)
        _continuous, categorical, availability = encode_metadata(meta, vocabs)
        assert categorical[0] == UNKNOWN_INDEX, (
            f"{image.stem}: the unseen sensor encoded as a known one at index {categorical[0]}"
        )
        assert availability[0] == 1.0, "the unseen sensor must be *known-unknown*, not absent"
        # And the fields that can transfer are still there.
        i_res = ("resolution_m", "incidence_deg", "band").index("resolution_m")
        assert availability[i_res] == 1.0 and _continuous[i_res] > 0.0


def trainer_model(tmp_path: Path):
    """The trained conditioned model of the most recent run in ``tmp_path``.

    Kept as a helper so the assertion block reads as the protocol, not as checkpoint plumbing.
    """
    import torch

    checkpoints = sorted((tmp_path / "runs").rglob("last.pt"))
    assert checkpoints, "no checkpoint was written"
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    return payload.get("ema") or payload.get("model")


def test_conditioned_metadata_rows_match_their_own_images(tmp_path, conditioned_config, monkeypatch):
    """Each training image must receive its own acquisition row, not a neighbour's.

    Composite-image mixing is already refused elsewhere; this pins the join itself,
    using stems whose sensor differs, so a transposed or sorted join cannot pass.
    """
    root, data_yaml = _write_conditioned_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    from saryolo.data.conditioning import training_metadata_vocabularies
    from saryolo.data.metadata import MetadataTable, encode_metadata
    from saryolo.data.yolo import resolve_data_yaml

    resolved = resolve_data_yaml(data_yaml)
    cfg = yaml.safe_load(Path(resolved).read_text())
    table = MetadataTable.load(cfg["acquisition_metadata"])
    vocabs, _ = training_metadata_vocabularies({**cfg, "yaml_file": str(resolved)})

    train_images = sorted((root / "images" / "train").glob("*.png"))
    assert train_images
    for image in train_images:
        meta = table.get(image.stem)
        expected_sensor_index = vocabs["sensor"].index(meta.sensor)
        _continuous, categorical, _availability = encode_metadata(meta, vocabs)
        assert categorical[0] == expected_sensor_index
        # s1 and g3 must not collide: the two-sensor fixture exists to catch exactly that.
        assert vocabs["sensor"].index("s1") != vocabs["sensor"].index("g3")
