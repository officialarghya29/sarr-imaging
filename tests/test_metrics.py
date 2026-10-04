"""Unit tests for the COCO-style AP implementation.

These matter more than they look. In the smoke run every metric came out as
``0.0`` — which is the expected value for two epochs at 128px on synthetic data,
but it is *indistinguishable* from a metric function that is simply broken and
always returns zero. These tests establish, on hand-constructed inputs where the
answer is known analytically, that the metric responds correctly, including the
scale-wise routing the paper's central claim depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from saryolo.evaluation.metrics import AREA_RANGES, Detection, compute_ap

REPO_ROOT = Path(__file__).resolve().parents[1]


def _det(image: str, cls: int, xyxy, score: float = 0.9) -> Detection:
    return Detection(image, cls, tuple(float(v) for v in xyxy), score=score)


def test_perfect_predictions_give_ap_one():
    """Ground truth exactly reproduced must score AP = 1.0 at every IoU threshold."""
    gts = [_det("a", 0, (10, 10, 50, 50)) for _ in [0]]
    dets = [_det("a", 0, (10, 10, 50, 50), score=0.99)]
    stats = compute_ap(dets, gts, nc=1, area_range=AREA_RANGES["all"])
    assert stats["AP"] == pytest.approx(1.0)
    assert stats["AP50"] == pytest.approx(1.0)
    assert stats["n_positives"] == 1.0


def test_no_predictions_give_ap_zero():
    gts = [_det("a", 0, (10, 10, 50, 50))]
    stats = compute_ap([], gts, nc=1, area_range=AREA_RANGES["all"])
    assert stats["AP"] == pytest.approx(0.0)


def test_no_ground_truth_gives_none_not_zero():
    """An area range with no ground truth is *not measurable*.

    Returning 0.0 here would be actively misleading: a dataset with no large
    objects would appear to score AP_large = 0, which a reader would interpret as
    the model failing on large objects.
    """
    gts = [_det("a", 0, (0, 0, 10, 10))]  # area 100 -> small
    stats = compute_ap([_det("a", 0, (0, 0, 10, 10))], gts, nc=1, area_range=AREA_RANGES["large"])
    assert stats["n_positives"] == 0.0
    assert stats["AP"] is None


def test_missed_detection_is_a_false_negative():
    gts = [_det("a", 0, (10, 10, 50, 50))]
    dets = [_det("a", 0, (200, 200, 240, 240), score=0.9)]
    stats = compute_ap(dets, gts, nc=1, area_range=AREA_RANGES["all"])
    assert stats["AP"] == pytest.approx(0.0)


def test_half_iou_does_not_count_as_ap75():
    """A box overlapping at ~0.5 IoU passes AP50 but must fail the higher thresholds."""
    gts = [_det("a", 0, (0, 0, 100, 100))]
    dets = [_det("a", 0, (0, 0, 100, 50), score=0.9)]  # IoU = 0.5 exactly
    stats = compute_ap(dets, gts, nc=1, area_range=AREA_RANGES["all"])
    assert stats["AP50"] == pytest.approx(1.0)
    assert stats["AP"] < 1.0


def test_scale_ranges_route_by_area():
    """Small/medium/large bins must partition ground truth by pixel area.

    The thresholds are 32^2 and 96^2. Binning *linear* sizes against 32/96 (an
    easy mistake) would put a 40x40 object in 'large' instead of 'medium'.
    """
    small = _det("a", 0, (0, 0, 20, 20))       # area 400
    medium = _det("b", 0, (0, 0, 50, 50))      # area 2500
    large = _det("c", 0, (0, 0, 150, 150))     # area 22500
    gts = [small, medium, large]
    dets = [
        _det("a", 0, (0, 0, 20, 20)),
        _det("b", 0, (0, 0, 50, 50)),
        _det("c", 0, (0, 0, 150, 150)),
    ]
    assert compute_ap(dets, gts, 1, AREA_RANGES["small"])["AP"] == pytest.approx(1.0)
    assert compute_ap(dets, gts, 1, AREA_RANGES["medium"])["AP"] == pytest.approx(1.0)
    assert compute_ap(dets, gts, 1, AREA_RANGES["large"])["AP"] == pytest.approx(1.0)
    assert compute_ap(dets, gts, 1, AREA_RANGES["small"])["n_positives"] == 1.0


def test_scale_range_boundary_is_exclusive_at_top():
    """A 32x32 object (area exactly 1024) belongs to 'medium', per the COCO convention."""
    exact = _det("a", 0, (0, 0, 32, 32))
    assert compute_ap([], [exact], 1, AREA_RANGES["small"])["n_positives"] == 0.0
    assert compute_ap([], [exact], 1, AREA_RANGES["medium"])["n_positives"] == 1.0


def test_confidence_ordering_affects_precision_curve():
    """A higher-scoring true positive must beat a lower-scoring false positive."""
    gts = [_det("a", 0, (0, 0, 100, 100))]
    good = [_det("a", 0, (0, 0, 100, 100), score=0.9), _det("a", 0, (500, 500, 520, 520), score=0.1)]
    bad = [_det("a", 0, (0, 0, 100, 100), score=0.1), _det("a", 0, (500, 500, 520, 520), score=0.9)]
    assert compute_ap(good, gts, 1, AREA_RANGES["all"])["AP"] > compute_ap(bad, gts, 1, AREA_RANGES["all"])["AP"]


def test_class_mismatch_is_not_a_true_positive():
    gts = [_det("a", 0, (0, 0, 100, 100))]
    dets = [_det("a", 1, (0, 0, 100, 100), score=0.9)]
    stats = compute_ap(dets, gts, nc=2, area_range=AREA_RANGES["all"])
    assert stats["AP"] == pytest.approx(0.0)


# --------------------------------------------------------------- prediction output path
def test_predict_to_labels_resolves_a_relative_out_dir(tmp_path, monkeypatch):
    """Regression: ultralytics re-roots a *relative* ``project`` under its own runs dir.

    Inference for ``out_dir="results/eval"`` was written to ``runs/detect/results/eval`` while
    the function then looked in ``results/eval`` and raised ``FileNotFoundError``. Every caller
    that passed a relative path failed -- including this function's own default, and therefore
    ``saryolo eval``, the robustness sweep and the cross-dataset evaluation. The contract pinned
    here is the one that makes them work: the directory handed to ultralytics must be absolute,
    and the directory returned must be the one actually written.
    """
    from pathlib import Path

    from saryolo.evaluation import metrics

    seen: dict = {}

    class _FakeModel:
        def predict(self, **kwargs):
            project = Path(kwargs["project"])
            seen["project"] = project
            # This is exactly what ultralytics does: re-root a relative project under runs/.
            root = project if project.is_absolute() else Path("runs") / "detect" / project
            labels = root / "labels"
            labels.mkdir(parents=True, exist_ok=True)
            (labels / "a.txt").write_text("0 0.500000 0.500000 0.250000 0.250000 0.9\n")

    monkeypatch.setattr("saryolo.training.trainer.load_model", lambda *_a, **_k: _FakeModel())
    monkeypatch.chdir(tmp_path)

    labels = metrics.predict_to_labels("w.pt", tmp_path / "imgs", out_dir="results/preds")

    assert seen["project"].is_absolute(), "a relative project is what caused the bug"
    assert Path(labels).is_absolute()
    assert Path(labels).exists(), f"returned {labels}, which was never written"
    assert Path(labels).parent == (tmp_path / "results" / "preds")
    assert not (tmp_path / "runs").exists(), "output must not be re-rooted under runs/detect"


def _tiny_dataset(root, counts=("a", "b")):
    """A minimal images/ + labels/ tree with non-empty labels, sized 100x50."""
    import cv2
    import numpy as np

    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for stem in counts:
        cv2.imwrite(str(root / "images" / f"{stem}.png"), np.zeros((50, 100, 3), dtype=np.uint8))
        (root / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n")
    return root


def test_ground_truth_is_read_from_a_txt_list_split(tmp_path):
    """A split given as an image *list* must still resolve to its ground truth.

    Regression, and a silent one: the labels were derived by string-replacing ``images`` ->
    ``labels`` on the split entry. For a list entry that produced the list file itself as the
    "labels directory", so ``glob("*.txt")`` found the list, looked for an image named after
    it, found none, and returned zero boxes -- while the evaluation still reported success with
    ``mAP50 = None``. The list form is what the leave-one-source-out folds emit, so without
    this the whole cross-source protocol would have measured nothing.
    """
    from saryolo.evaluation.metrics import load_yolo_ground_truth

    data_root = _tiny_dataset(tmp_path)
    listing = tmp_path / "val_list.txt"
    listing.write_text("".join(f"{data_root / 'images' / f'{s}.png'}\n" for s in ("a", "b")))
    cfg = tmp_path / "data.yaml"
    cfg.write_text(f"path: {data_root}\nnc: 1\nnames:\n- target\nval: {listing}\n")

    gts = load_yolo_ground_truth(cfg)
    assert len(gts) == 4, f"expected 2 images x 2 boxes, got {len(gts)}"
    assert {g.image for g in gts} == {"a", "b"}
    # 100x50 image, box 0.2x0.2 -> 20x10 px, centred.
    first = next(g for g in gts if g.image == "a" and g.xyxy[0] == pytest.approx(40.0))
    assert first.xyxy == pytest.approx((40.0, 20.0, 60.0, 30.0))


def test_ground_truth_from_a_directory_split_is_unchanged(tmp_path):
    """The list handling must not disturb the ordinary directory split."""
    from saryolo.evaluation.metrics import load_yolo_ground_truth

    data_root = _tiny_dataset(tmp_path / "ds")
    cfg = tmp_path / "data.yaml"
    cfg.write_text(f"path: {data_root}\nnc: 1\nnames:\n- target\nval: images\n")
    assert len(load_yolo_ground_truth(cfg)) == 4


def test_evaluation_refuses_a_split_with_no_ground_truth(tmp_path):
    """An unmeasurable split must raise, not return a dict of Nones as if it finished.

    The failure this prevents is not a crash but a *quiet* one: an empty metrics dict looks
    like a completed evaluation, so a broken split reference would be written into a results
    table as a number nobody could tell was missing.
    """
    from saryolo.evaluation.metrics import evaluate_detections

    data_root = tmp_path / "ds"
    (data_root / "images").mkdir(parents=True)
    (data_root / "labels").mkdir(parents=True)
    import cv2
    import numpy as np

    cv2.imwrite(str(data_root / "images" / "a.png"), np.zeros((32, 32, 3), dtype=np.uint8))
    (data_root / "labels" / "a.txt").write_text("")  # image present, no boxes anywhere
    cfg = tmp_path / "data.yaml"
    cfg.write_text(f"path: {data_root}\nnc: 1\nnames:\n- target\nval: images\n")

    # No weights are needed: ground truth is read before inference precisely so this is
    # caught without paying for a prediction pass.
    with pytest.raises(ValueError, match="no ground-truth boxes"):
        evaluate_detections("does-not-matter.pt", cfg)


def test_label_lookup_prefers_an_existing_file(tmp_path):
    """A wrong guess between the two images->labels conventions must not erase ground truth."""
    from saryolo.evaluation.metrics import _label_path_for

    data_root = _tiny_dataset(tmp_path)
    image = data_root / "images" / "a.png"
    assert _label_path_for(image) == data_root / "labels" / "a.txt"

    # A parent directory that also contains "images" is where the first-replacement rule
    # produces a path that does not exist; the component rule is the one that is right.
    odd = tmp_path / "images_export" / "images" / "a.png"
    odd.parent.mkdir(parents=True)
    odd.write_bytes(b"x")
    (tmp_path / "images_export" / "labels").mkdir()
    (tmp_path / "images_export" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert _label_path_for(odd) == tmp_path / "images_export" / "labels" / "a.txt"


def test_predict_to_labels_returns_the_directory_it_wrote(tmp_path, monkeypatch):
    """The returned path must contain the labels, so callers never guess where they landed."""
    from pathlib import Path

    from saryolo.evaluation import metrics

    class _FakeModel:
        def predict(self, **kwargs):
            labels = Path(kwargs["project"]) / "labels"
            labels.mkdir(parents=True, exist_ok=True)
            (labels / "img.txt").write_text("0 0.5 0.5 0.1 0.1 0.8\n")

    monkeypatch.setattr("saryolo.training.trainer.load_model", lambda *_a, **_k: _FakeModel())
    out = tmp_path / "nested" / "deep"
    labels = metrics.predict_to_labels("w.pt", tmp_path / "imgs", out_dir=out)
    assert Path(labels) == (out / "labels")
    assert sorted(p.name for p in Path(labels).glob("*.txt")) == ["img.txt"]


def test_ground_truth_can_be_read_from_any_split_not_only_validation(tmp_path):
    """The split is a parameter, because a caller that scored another split was silently wrong.

    ``mine-hard`` predicts on the *train* images and then scored them against the *val* ground
    truth: with the split hard-coded to ``val``, every predicted image had no labels and every
    labelled image had no predictions, so the miner ranked images by raw spurious-detection
    count and reported success. This pins the resolution rule that made it wrong.
    """
    import cv2
    import numpy as np

    from saryolo.evaluation.metrics import load_yolo_ground_truth

    root = tmp_path / "ds"
    for split, n_boxes in (("train", 2), ("val", 1)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        img = (np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(root / "images" / split / "a.png"), img)
        (root / "labels" / split / "a.txt").write_text(
            "".join("0 0.5 0.5 0.1 0.1\n" for _ in range(n_boxes))
        )
    (tmp_path / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: ship\n"
    )
    cfg = tmp_path / "data.yaml"
    # The default must stay the validation split: every evaluation caller depends on it.
    assert len(load_yolo_ground_truth(cfg)) == 1
    assert len(load_yolo_ground_truth(cfg, split="train")) == 2
    assert len(load_yolo_ground_truth(cfg, split="val")) == 1


# ------------------------------------------------------- agreement with the COCO reference
def _coco_reference(dets, gts, nc: int, images: dict[str, tuple[int, int]]):
    """mAP50:95 for the same boxes, computed by pycocotools itself.

    This is the only external referee available for the repository's metric, and it is the
    right one: COCO AP is what the module claims to implement, and pycocotools is the
    definition. Returns ``None`` when pycocotools is not installed so the caller can skip.
    """
    pytest.importorskip("pycocotools.cocoeval")
    import json
    import tempfile
    from pathlib import Path

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    stems = sorted(images)
    ids = {s: i + 1 for i, s in enumerate(stems)}
    gt_json = {
        "images": [
            {"id": ids[s], "width": images[s][0], "height": images[s][1], "file_name": s}
            for s in stems
        ],
        "annotations": [],
        "categories": [{"id": c, "name": str(c)} for c in range(1, nc + 1)],
    }
    for i, g in enumerate(gts, start=1):
        x1, y1, x2, y2 = g.xyxy
        gt_json["annotations"].append(
            {
                "id": i,
                "image_id": ids[g.image],
                "category_id": g.cls + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": max((x2 - x1) * (y2 - y1), 0.0),
                "iscrowd": 0,
            }
        )
    dt_json = [
        {
            "image_id": ids[d.image],
            "category_id": d.cls + 1,
            "bbox": [d.xyxy[0], d.xyxy[1], d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]],
            "score": d.score,
        }
        for d in dets
    ]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "gt.json").write_text(json.dumps(gt_json))
    (tmp / "dt.json").write_text(json.dumps(dt_json))
    coco = COCO(str(tmp / "gt.json"))
    ev = COCOeval(coco, coco.loadRes(str(tmp / "dt.json")), "bbox")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    # `stats` is only populated by ``summarize`` -- reading it after ``accumulate`` alone
    # raises IndexError on an empty list, which would look like an evaluator failure.
    ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def test_the_ap_implementation_reproduces_pycocotools():
    """Our COCO AP must equal the reference implementation on boxes it did not choose.

    The metric is the only thing standing between a trained model and every claim in the
    paper, and it is implemented here rather than imported. Ultralytics' validator reports a
    *different* mAP50:95 for the same checkpoint (a detection-path tail difference at
    conf=0.001, not an AP difference), so "which evaluator" has to be settled by an outside
    referee. This is that referee.
    """
    images = {"a": (100, 100), "b": (80, 120), "c": (64, 64)}
    # Overlapping boxes, a miss, a duplicate, a false positive, and a multi-class spread --
    # the cases where a subtly wrong matching rule shows up and a single perfect box does not.
    gts = [
        _det("a", 0, (10, 10, 40, 40)),
        _det("a", 1, (50, 50, 80, 90)),
        _det("b", 0, (5, 5, 25, 25)),
        _det("c", 0, (30, 30, 35, 35)),
    ]
    dets = [
        _det("a", 0, (11, 11, 41, 41), 0.95),
        _det("a", 0, (12, 9, 39, 42), 0.80),
        _det("a", 1, (48, 52, 82, 88), 0.70),
        _det("b", 0, (6, 6, 26, 26), 0.60),
        _det("b", 1, (1, 1, 9, 9), 0.50),
        _det("c", 0, (28, 28, 36, 36), 0.40),
    ]
    ours = compute_ap(dets, gts, nc=2, area_range=AREA_RANGES["all"])
    ref = _coco_reference(dets, gts, 2, images)
    assert ref is not None
    assert ours["AP"] == pytest.approx(ref[0], abs=1e-3), (
        f"COCO AP disagrees with pycocotools: ours {ours['AP']:.4f} vs reference {ref[0]:.4f}"
    )
    assert ours["AP50"] == pytest.approx(ref[1], abs=1e-3)


def test_the_reported_protocol_field_says_how_the_number_was_measured():
    """A metrics dict must name its protocol, so two evaluators cannot be silently mixed."""
    cfg = REPO_ROOT / "configs" / "datasets" / "hrsid_real.yaml"
    ckpt = REPO_ROOT / "results" / "runs" / "REAL-001" / "weights" / "best.pt"
    if not ckpt.exists() or not (REPO_ROOT / "datasets/processed/hrsid_real/images/val").is_dir():
        pytest.skip("HRSID subset or checkpoint not present; protocol field not exercised here")
    from saryolo.evaluation.metrics import evaluate_detections

    metrics = evaluate_detections(ckpt,
    cfg,
    imgsz=320,
    out_dir="/tmp/m_protocol",
    )
    assert metrics["eval_protocol"] == "coco"
    assert metrics["eval_max_det"] == 100
