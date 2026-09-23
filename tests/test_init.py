"""Tests for the Phase-3 init stage: stated initialisation, real transfer, honest counts.

The failure being guarded against is the one the ultralytics source reading revealed:
a YAML-built facade's inner model is silently *rebuilt* by ``train()``, so a transfer
that only mutates the facade would vanish while the record still claimed a pretrained
start. These tests pin the checkpoint route, the honest transfer count, and the
config spellings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from saryolo.nn.arch import VARIANTS, build_yaml_dict
from saryolo.nn.model import SARYOLODetectionModel
from saryolo.training.config import load_experiment
from saryolo.training.trainer import apply_init


@pytest.fixture(scope="module")
def source_checkpoint(tmp_path_factory) -> Path:
    """A stand-in 'pretrained' checkpoint: a stock yolo11s graph, COCO-class head."""
    root = tmp_path_factory.mktemp("init_src")
    src = DetectionModel("yolo11s.yaml", ch=3, nc=80, verbose=False)
    path = root / "source_pretrained.pt"
    torch.save({"model": src, "epoch": -1}, path)
    return path


@pytest.fixture(scope="module")
def baseline_model():
    return SARYOLODetectionModel(build_yaml_dict(VARIANTS["baseline"]), ch=3, nc=1, verbose=False)


# ---------------------------------------------------------------------- apply_init
def test_none_init_states_that_nothing_happens(baseline_model):
    summary = apply_init(baseline_model, "none")
    assert summary == {"init": "none", "init_weights": None}


def test_checkpoint_init_transfers_and_reports_a_count(baseline_model, source_checkpoint, tmp_path, monkeypatch):
    """The summary count must equal the tensors the reloaded checkpoint actually
    changed relative to a fresh build -- a transfer that loads nothing is worse than
    a crash, because it would run as random-init while claiming a pretrained start."""
    monkeypatch.chdir(tmp_path)
    summary = apply_init(baseline_model, str(source_checkpoint))
    assert summary["init"] == "checkpoint"
    assert summary["init_transferred_tensors"] > 0
    assert Path(summary["init_checkpoint"]).is_file()

    # Reload the checkpoint exactly like the runner does and count real differences.
    from saryolo.training.trainer import load_model

    m2 = load_model(summary["init_checkpoint"])
    fresh = DetectionModel(build_yaml_dict(VARIANTS["baseline"]), ch=3, nc=1, verbose=False)
    diff = sum(
        1 for k in m2.model.state_dict() if not torch.equal(m2.model.state_dict()[k], fresh.state_dict()[k])
    )
    # Allow a small delta: the fresh reference build consumes RNG, so layers *outside*
    # the intersection can diverge slightly across builds. The transferred count must
    # dominate, and must never exceed the real difference.
    assert summary["init_transferred_tensors"] <= diff
    assert diff - summary["init_transferred_tensors"] <= 0.05 * len(fresh.state_dict())


def test_checkpoint_init_writes_a_content_stable_filename(baseline_model, source_checkpoint, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s1 = apply_init(baseline_model, str(source_checkpoint))
    s2 = apply_init(baseline_model, str(source_checkpoint))
    assert s1["init_checkpoint"] == s2["init_checkpoint"]


def test_init_refuses_a_checkpoint_sharing_no_compatible_tensors(baseline_model, tmp_path, monkeypatch):
    """A 1x1-conv-only 'checkpoint' shares nothing with the graph; loading it would
    be a random-init run wearing a pretrained label."""
    monkeypatch.chdir(tmp_path)
    bogus = tmp_path / "bogus.pt"
    torch.save({"model": torch.nn.Conv2d(3, 3, 1)}, bogus)
    with pytest.raises(ValueError, match="no shape-compatible parameter transferred"):
        apply_init(baseline_model, str(bogus))


def test_init_refuses_a_missing_checkpoint(baseline_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        apply_init(baseline_model, str(tmp_path / "nope.pt"))


def test_init_refuses_a_source_that_is_not_model_weights(baseline_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    junk = tmp_path / "junk.pt"
    torch.save({"not_a_model": [1, 2, 3]}, junk)
    with pytest.raises(ValueError, match="no state_dict|does not contain model weights"):
        apply_init(baseline_model, str(junk))


# ---------------------------------------------------------------------- config spellings
def _write_exp(path: Path, init_value) -> Path:
    """Write an experiment config in a tmp directory laid out like configs/, because
    load_experiment resolves -- and demands the existence of -- the model/dataset paths."""
    root = path.parent.parent
    (root / "models").mkdir(exist_ok=True)
    (root / "datasets").mkdir(exist_ok=True)
    for src, dst in (("configs/models/yolo11s_baseline.yaml", "models/yolo11s_baseline.yaml"),
                     ("configs/datasets/ssdd.yaml", "datasets/ssdd.yaml")):
        if not (root / dst).exists():
            (root / dst).write_text(Path(src).read_text())
    path.write_text(
        f"""
experiment:
  id: EXP-TEST-INIT
  name: init test
model: ../models/yolo11s_baseline.yaml
dataset: ../datasets/ssdd.yaml
init: {init_value}
train:
  epochs: 1
  seed: 0
"""
    )
    return path


def test_config_accepts_the_documented_init_spellings(tmp_path):
    cfg = load_experiment(_write_exp(tmp_path / "a.yaml", "coco11"))
    assert cfg.extra["init"] == "coco11"
    cfg = load_experiment(_write_exp(tmp_path / "b.yaml", "{weights: some.pt}"))
    assert cfg.extra["init"] == {"weights": "some.pt"}


def test_config_refuses_an_unstated_init_mode(tmp_path):
    """`init: coco110` would silently fall through to random-init without this guard."""
    with pytest.raises(ValueError, match="'init' must be"):
        load_experiment(_write_exp(tmp_path / "c.yaml", "coco110"))


# ---------------------------------------------------------------------- generator arms
def test_generated_init_arms_exist_and_share_the_model():
    """The three Phase-3 arms must exist, share one model YAML, and differ only in
    their init stage -- that is what makes the later comparison attributable."""
    configs = {}
    for name in ("EXP-401_init_baseline.yaml", "EXP-402_init_baseline.yaml", "EXP-403_init_baseline.yaml"):
        path = Path("configs/exp") / name
        assert path.is_file(), f"missing generated arm {name}"
        configs[name] = load_experiment(path)
    models = {cfg.model_path.name for cfg in configs.values()}
    assert len(models) == 1, f"init arms must share one model YAML, got {models}"
    seeds = {cfg.seed for cfg in configs.values()}
    assert len(seeds) == 1, "init arms must share one seed"
    inits = sorted(str(cfg.extra.get("init", "none")) for cfg in configs.values())
    assert inits == sorted(["none", "coco11", "{'weights': 'PATH/TO/msfa_or_sar_checkpoint.pt'}"])
