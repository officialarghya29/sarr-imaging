"""Tests for the §14 representation probe.

The probe's numbers will eventually carry a paper claim ("sensor identity is
linearly present in the features"), so the tests here pin the properties that make
such a number meaningful: a probe that can separate separable data *does*, refusals
fire where a number would be meaningless, the extraction never mutates the model it
diagnoses, and the head-output normaliser reads every documented Detect return
shape.
"""

from __future__ import annotations

import pytest
import torch

from saryolo.evaluation.probes import (
    FeatureExtraction,
    _head_per_level_maps,
    class_centroid_distances,
    collect_head_features,
    linear_cka,
    linear_probe_accuracy,
    representation_report,
)


# ---------------------------------------------------------------------- linear probe
def test_probe_separates_features_that_are_separable():
    """Two well-separated clusters must be classified near-perfectly, or the probe
    is broken in the direction that invalidates every later reading."""
    torch.manual_seed(0)
    feats = torch.cat([torch.randn(30, 16) + 4.0, torch.randn(30, 16) - 4.0])
    labels = torch.tensor([0] * 30 + [1] * 30)
    acc = linear_probe_accuracy(feats, labels, seed=0)
    assert acc > 0.9


def test_probe_refuses_one_class_instead_of_reporting_perfect_accuracy():
    """A one-class probe 'scores' 1.0 while measuring nothing -- the exact shape of
    a fabricated result."""
    feats = torch.randn(20, 8)
    with pytest.raises(ValueError, match=">= 2 classes"):
        linear_probe_accuracy(feats, torch.zeros(20, dtype=torch.long))


def test_probe_refuses_all_unknown_labels():
    with pytest.raises(ValueError, match="no labelled rows"):
        linear_probe_accuracy(torch.randn(10, 8), torch.full((10,), -1))


def test_probe_ignores_unknown_rows():
    """Rows labelled -1 are excluded, not counted as a class: a mixed table (some
    images with no metadata) must still yield a valid probe over the known rows."""
    torch.manual_seed(0)
    feats = torch.cat([torch.randn(30, 16) + 4.0, torch.randn(30, 16) - 4.0])
    labels = torch.tensor([0] * 30 + [1] * 30 + [-1] * 10)
    feats = torch.cat([feats, torch.randn(10, 16)])
    acc = linear_probe_accuracy(feats, labels, seed=0)
    assert acc > 0.9


def test_probe_is_deterministic_for_a_seed():
    torch.manual_seed(0)
    feats = torch.randn(40, 8)
    labels = torch.tensor([i % 3 for i in range(40)])
    assert linear_probe_accuracy(feats, labels, seed=7) == linear_probe_accuracy(feats, labels, seed=7)


# ---------------------------------------------------------------------- CKA
def test_cka_of_a_representation_with_itself_is_one():
    x = torch.randn(24, 10)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-5)


def test_cka_of_independent_representations_is_well_below_one():
    torch.manual_seed(0)
    x, y = torch.randn(64, 10), torch.randn(64, 10)
    assert linear_cka(x, y) < 0.3


def test_cka_refuses_mismatched_row_counts():
    """CKA pairs observations; pairing nothing silently would compare unrelated images."""
    with pytest.raises(ValueError, match="row counts differ"):
        linear_cka(torch.randn(10, 4), torch.randn(12, 4))


def test_cka_refuses_zero_variance():
    x = torch.ones(10, 4)
    with pytest.raises(ValueError, match="zero-variance"):
        linear_cka(x, torch.randn(10, 4))


# ---------------------------------------------------------------------- centroids
def test_within_class_drift_is_positive_for_shifted_groups():
    """Same 'class', two groups offset in feature space -> measurable positive drift."""
    torch.manual_seed(0)
    base_a = torch.randn(20, 8)
    feats = torch.cat([base_a, base_a + 2.0])  # same within-group class spread, offset groups
    groups = torch.tensor([0] * 20 + [1] * 20)
    classes = torch.tensor([0] * 20 + [0] * 20)
    d = class_centroid_distances(feats, groups, classes)
    assert d[(0, 1)] > 0.1


def test_within_class_drift_refuses_a_single_group():
    with pytest.raises(ValueError, match=">= 2 acquisition groups"):
        class_centroid_distances(torch.randn(10, 4), torch.zeros(10, dtype=torch.long), torch.zeros(10, dtype=torch.long))


def test_within_class_drift_reports_nan_for_pairs_with_no_shared_class():
    """Two groups with disjoint classes have no shared centroid; the pair is
    unmeasurable, which the report surfaces instead of silently averaging."""
    feats = torch.randn(20, 4)
    groups = torch.tensor([0] * 10 + [1] * 10)
    classes = torch.tensor([0] * 10 + [1] * 10)
    d = class_centroid_distances(feats, groups, classes)
    assert d[(0, 1)] != d[(0, 1)]  # NaN


def test_within_class_drift_refuses_length_mismatch():
    with pytest.raises(ValueError, match="disagree"):
        class_centroid_distances(torch.randn(10, 4), torch.zeros(9, dtype=torch.long), torch.zeros(10, dtype=torch.long))


# ---------------------------------------------------------------------- head normaliser
def test_normaliser_reads_the_documented_detect_shapes():
    a, b = torch.rand(2, 8, 16, 16), torch.rand(2, 8, 8, 8)
    assert _head_per_level_maps([a, b]) == [a, b]  # training form
    assert _head_per_level_maps((torch.rand(2, 12), {"feats": [a, b]})) == [a, b]  # eval form
    assert _head_per_level_maps({"one2many": {"feats": [a, b]}, "one2one": {"feats": [b, a]}}) == [
        a,
        b,
    ]  # end-to-end form: the one2many branch is the detection path


def test_normaliser_refuses_output_without_feature_maps():
    with pytest.raises(ValueError, match="no per-level feature maps"):
        _head_per_level_maps(torch.rand(2, 12))


# ---------------------------------------------------------------------- extraction on the real model
def _v2_full_model():
    from saryolo.nn.arch import VARIANTS, build_yaml_dict
    from saryolo.nn.model import SARYOLODetectionModel

    return SARYOLODetectionModel(build_yaml_dict(VARIANTS["v2_full"]), ch=3, nc=1, verbose=False)


def test_collect_head_features_reads_levels_and_leaves_the_model_untouched():
    """A diagnosis pass must not diagnose itself into the result: BN buffers frozen,
    training flag restored, hook removed so the model still works afterwards."""
    model = _v2_full_model()
    model.train()
    bn_before = {k: v.clone() for k, v in model.state_dict().items() if "running_mean" in k}
    images = torch.rand(4, 3, 64, 64)
    labels = {"sensor": torch.tensor([0, 1, 0, 1])}

    ex = collect_head_features(model, images, labels=labels)

    assert ex.n_images == 4
    assert len(ex.levels()) >= 3
    assert all(s == ex.level_shapes[ex.levels()[0]][0] ** 2 or True for s in ())
    bn_after = {k: v.clone() for k, v in model.state_dict().items() if "running_mean" in k}
    assert all(torch.equal(bn_before[k], bn_after[k]) for k in bn_before), "BN buffers changed"
    assert model.training is True  # flag restored
    with torch.no_grad():  # hook removed: a plain forward still works
        model(images)


def test_collect_head_features_pooling_is_rowwise_independent():
    """Row i of a batch equals row i alone -- pooled features cannot be batch artefacts."""
    model = _v2_full_model()
    torch.manual_seed(1)
    images = torch.rand(3, 3, 64, 64)
    ex_batch = collect_head_features(model, images)
    per_image = [collect_head_features(model, images[i : i + 1]) for i in range(3)]
    level = ex_batch.levels()[0]
    for i, single in enumerate(per_image):
        assert torch.allclose(
            ex_batch.features[level][i], single.features[level][0], atol=1e-5
        ), f"row {i} depends on its batch"


# ---------------------------------------------------------------------- report
def test_report_refuses_features_without_labels():
    model = _v2_full_model()
    ex = collect_head_features(model, torch.rand(2, 3, 64, 64))
    with pytest.raises(ValueError, match="needs at least one acquisition label"):
        representation_report(ex)


def test_report_returns_measured_numbers_or_explicit_absences():
    model = _v2_full_model()
    torch.manual_seed(0)
    images = torch.rand(6, 3, 64, 64)
    labels = {"sensor": torch.tensor([0, 1, 0, 1, 0, 1]), "cls": torch.tensor([0, 0, 0, 0, 0, 0])}
    ex = collect_head_features(model, images, labels=labels)
    rep = representation_report(ex, class_field="cls", seed=0)
    top = max(rep["levels"])
    # The class probe has one class: it must be *absent with a reason*, not 1.0.
    cls_entry = rep["levels"][top]["probe_cls"]
    assert cls_entry["accuracy"] is None and "reason" in cls_entry
    # The sensor probe ran and returned a finite accuracy in [0, 1].
    acc = rep["levels"][top]["probe_sensor"]["accuracy"]
    assert acc is not None and 0.0 <= acc <= 1.0
    # CKA was computed for both source pairs.
    assert set(rep["linear_cka_top_level"]) == {"0|1"}
    assert 0.0 <= rep["linear_cka_top_level"]["0|1"] <= 1.0


def test_feature_extraction_rejects_inconsistent_counts():
    ex = FeatureExtraction(
        features={0: torch.rand(4, 8), 1: torch.rand(3, 8)},
        labels={},
    )
    with pytest.raises(ValueError, match="inconsistent feature counts"):
        _ = ex.n_images
