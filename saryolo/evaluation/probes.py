"""Representation diagnosis (§14 of the master plan): is the sensor *in* the features?

The question this module exists to answer
-----------------------------------------
The central hypothesis of the cross-sensor claim is that a SAR detector trained on
several sources learns *object + acquisition appearance*, so that a large share of
its feature variation is explained by which sensor produced the image rather than
by what is in it. That is a claim about representations, and it can be tested
without any further training of the detector: freeze the model, hook the detection
head's per-level feature maps, pool each image to a vector, and measure how well a
*small logistic probe* can predict the acquisition field from that vector alone.

Readings the probe supports, and what each would mean
----------------------------------------------------
1. **Probe accuracy** per feature level. A sensor probe far above the chance rate
   implied by the source count says sensor identity is linearly present in the
   features. If `sensor` is predictable but `class` is not, the representation is
   dominated by appearance — the failure the conditioning adapter addresses.
2. **Within-class source separation.** For each pair of acquisition groups and
   each object class, the cosine distance between group centroids of pooled
   features. Large within-class / cross-source distance, relative to the
   between-class distance within one source, is the quantitative version of "the
   detector sees the sensor before it sees the object".
3. **Linear CKA** between the pooled feature matrices of two acquisition groups.
   1.0 means the two representations are linearly interchangeable; low CKA under
   a *fixed class distribution* is drift the adapter has to pay for at the head.

The probe is deliberately a *linear* model on a *frozen* detector. A probe that
could train features itself would measure the probe, not the detector.

Every function here works on whatever the model's detection head emits, read
through a forward hook — nothing about the head's internal structure is assumed
beyond ultralytics' contract that training/eval forwards expose per-level maps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "FeatureExtraction",
    "collect_head_features",
    "linear_probe_accuracy",
    "class_centroid_distances",
    "linear_cka",
    "representation_report",
]

#: Logistic-probe hyperparameters. Small and fixed on purpose: the probe must not be
#: able to fit the task by being powerful, only by the features being informative.
PROBE_EPOCHS = 60
PROBE_LR = 0.05
PROBE_WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------- feature collection
@dataclass
class FeatureExtraction:
    """Per-level pooled features from one model forward pass, over many images.

    Attributes:
        features: ``{level: (n_images, n_channels)}`` mean-pooled per-level maps.
        labels: ``{field: (n_images,)}`` integer group labels, ``-1`` = unknown.
        level_shapes: spatial size of each level, recorded because a probe that
            compares levels must know it compared different resolutions.
    """

    features: dict[int, Tensor] = field(default_factory=dict)
    labels: dict[str, Tensor] = field(default_factory=dict)
    level_shapes: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def n_images(self) -> int:
        sizes = {v.shape[0] for v in self.features.values()}
        if len(sizes) > 1:
            raise ValueError(f"inconsistent feature counts across levels: {sorted(sizes)}")
        return sizes.pop()

    def levels(self) -> tuple[int, ...]:
        return tuple(sorted(self.features))


def _head_per_level_maps(head_output) -> list[Tensor]:
    """Per-level ``(n, C, h, w)`` maps from whatever the detection head returned.

    Ultralytics' ``Detect`` contract, as read from the installed source: training
    returns the per-head dict directly; eval returns ``(prediction, per_head)``. The
    per-head dict carries the raw per-level maps under ``"feats"``, and an end-to-end
    model wraps two such dicts under ``one2many`` / ``one2one``. The one2one branch is
    an auxiliary training path over the *same* maps detached -- the one2many branch is
    the detection path, so that is what a representation diagnosis reads.
    """
    if (
        isinstance(head_output, (tuple, list))
        and len(head_output) == 2
        and isinstance(head_output[1], (tuple, list, dict))
    ):
        # (prediction, per_level) eval form -- the first element is the decoded
        # inference tensor, the second carries the per-level maps.
        head_output = head_output[1]
    if isinstance(head_output, dict) and "one2many" in head_output:
        head_output = head_output["one2many"]
    if isinstance(head_output, dict) and "feats" in head_output:
        maps = list(head_output["feats"])
    else:
        maps = []
        def _collect(obj) -> None:
            if isinstance(obj, Tensor):
                if obj.dim() == 4:
                    maps.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect(v)
            elif isinstance(obj, (tuple, list)):
                for v in obj:
                    _collect(v)

        _collect(head_output)
    if not maps:
        raise ValueError("the head emitted no per-level feature maps")
    if not all(isinstance(m, Tensor) and m.dim() == 4 for m in maps):
        raise TypeError("per-level maps must each be a 4-D (n, C, h, w) tensor")
    return maps


def collect_head_features(
    model: nn.Module,
    images: Tensor,
    labels: dict[str, Tensor] | None = None,
) -> FeatureExtraction:
    """Hook the detection head, forward ``images`` in eval mode, pool per level.

    Args:
        model: any ``nn.Module`` whose *last submodule* is the detection head (the
            ultralytics ``DetectionModel`` contract). The hook is removed in a
            ``finally`` block, so a failed forward cannot leave the model wired.
        images: ``(n, C, H, W)`` batch, normalised however the model expects.
        labels: optional ``{field: (n,)}`` integer group labels to carry alongside.

    The model is put in eval mode for the extraction, so batch-norm running
    statistics are untouched — a diagnosis pass must not alter the thing diagnosed.
    """
    from saryolo.nn.model import SARYOLODetectionModel  # noqa: F401  (import validates the wiring)

    head = model.model[-1]
    collected: list = []
    handle = head.register_forward_hook(lambda _m, _i, o: collected.append(o))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(images)
    finally:
        handle.remove()
        if was_training:
            model.train()
    if not collected:
        raise RuntimeError("the head hook never fired; is model.model[-1] the detection head?")

    extraction = FeatureExtraction(labels=labels or {})
    maps = _head_per_level_maps(collected[0])
    for level, m in enumerate(maps):
        if m.shape[0] != images.shape[0]:
            raise ValueError(f"level {level} batch {m.shape[0]} != input batch {images.shape[0]}")
        pooled = m.mean(dim=(-2, -1))
        extraction.features[level] = pooled.detach().cpu()
        extraction.level_shapes[level] = (int(m.shape[-2]), int(m.shape[-1]))
    return extraction


# ---------------------------------------------------------------------------- linear probe
def linear_probe_accuracy(
    features: Tensor,
    labels: Tensor,
    seed: int = 0,
    epochs: int = PROBE_EPOCHS,
    known_mask: Tensor | None = None,
) -> float:
    """Accuracy of a logistic probe predicting ``labels`` from frozen ``features``.

    A single-probe split would make the number an artefact of one split; five folds
    with disjoint test partitions are averaged instead. All folds are measured, so
    the number is a mean over the data, not over a fortunate partition.

    Args:
        features: ``(n, d)`` pooled feature matrix.
        labels: ``(n,)`` integer labels; rows with ``-1`` are excluded.
        seed: probe initialisation and shuffling seed.
        epochs: fixed small training length.
        known_mask: optional precomputed mask of rows to use.

    Raises:
        ValueError: on fewer than two classes or an empty/unknown-only input — a
            probe over one class would "score" 1.0 while measuring nothing.
    """
    torch.manual_seed(seed)

    if known_mask is None:
        known_mask = labels >= 0
    known_mask = known_mask.to(torch.bool)
    if known_mask.sum() == 0:
        raise ValueError("probe has no labelled rows; every label is unknown (-1)")

    feats = features[known_mask].float()
    y = labels[known_mask].long()
    classes = torch.unique(y)
    if classes.numel() < 2:
        raise ValueError(
            f"probe needs >= 2 classes, found {classes.numel()}; a one-class probe would "
            "report perfect accuracy while measuring nothing"
        )

    # Remap to contiguous targets for the probe head.
    remap = {int(c): i for i, c in enumerate(classes)}
    targets = torch.tensor([remap[int(v)] for v in y])

    n = feats.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    folds = 5
    fold_size = math.ceil(n / folds)
    accuracies = []
    for f in range(folds):
        test_idx = idx[f * fold_size : (f + 1) * fold_size]
        train_idx = torch.tensor([i for i in idx.tolist() if i not in set(test_idx.tolist())])
        if train_idx.numel() == 0 or test_idx.numel() == 0:
            continue
        probe = nn.Linear(feats.shape[1], classes.numel())
        torch.manual_seed(seed + f)  # probe init varies per fold, data split does not
        opt = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
        loss_fn = nn.CrossEntropyLoss()
        probe.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(probe(feats[train_idx]), targets[train_idx])
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(feats[test_idx]).argmax(dim=1)
        accuracies.append(float((pred == targets[test_idx]).float().mean()))
    if not accuracies:
        raise ValueError("every probe fold was empty; the labelled set is too small")
    return float(np.mean(accuracies))


# ---------------------------------------------------------------------------- geometry
def _l2_normalise(x: Tensor) -> Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def class_centroid_distances(
    features: Tensor,
    group_labels: Tensor,
    class_labels: Tensor,
) -> dict[tuple[int, int], float]:
    """Mean within-class cosine distance between acquisition-group centroids.

    For every (group A, group B) pair and every class present in both, compute the
    centroid of that class's features inside each group, and the cosine distance
    between the two. Averaged per pair, this is the "same object, different sensor"
    drift — the quantity the conditioning adapter is hypothesised to reduce.

    Raises:
        ValueError: if the two label vectors disagree in length or carry no shared
            (group, class) pair — the distance of nothing is not zero, it is undefined.
    """
    lengths = {int(features.shape[0]), int(group_labels.shape[0]), int(class_labels.shape[0])}
    if len(lengths) > 1:
        # Note: ``a != b != c`` would NOT do here -- it chains as ``(a != b) and (b != c)``,
        # so a mismatch between features and either label vector can slip through when the
        # two label vectors happen to agree with each other but not with the features.
        raise ValueError(f"feature and label lengths disagree: {sorted(lengths)}")
    feats = _l2_normalise(features.float())
    groups = torch.unique(group_labels)
    groups = groups[groups >= 0]
    if groups.numel() < 2:
        raise ValueError("need >= 2 acquisition groups to measure cross-group drift")

    centroids: dict[tuple[int, int], Tensor] = {}
    for g in groups.tolist():
        for c in torch.unique(class_labels).tolist():
            mask = (group_labels == g) & (class_labels == c)
            if mask.sum() > 0:
                centroids[(g, c)] = feats[mask].mean(dim=0)

    distances: dict[tuple[int, int], float] = {}
    for ga, gb in combinations(sorted({g for g, _ in centroids}), 2):
        shared = sorted({c for (g, c) in centroids if g == ga} & {c for (g, c) in centroids if g == gb})
        if not shared:
            distances[(ga, gb)] = float("nan")
            continue
        d = [
            1.0 - float(torch.nn.functional.cosine_similarity(centroids[(ga, c)], centroids[(gb, c)], dim=0))
            for c in shared
        ]
        distances[(ga, gb)] = float(np.mean(d))
    return distances


def linear_cka(x: Tensor, y: Tensor) -> float:
    """Linear CKA between two ``(n, d)`` representations of the *same* images.

    Follows Kornblith et al. (2019): ``||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)``
    on column-centred matrices. Requires the same number of rows — CKA is defined
    over paired observations, and pairing images across acquisition groups is
    exactly what makes the comparison meaningful (same scene content, different
    acquisition). A mismatched pair count is refused rather than silently compared.

    Raises:
        ValueError: on row-count mismatch or a degenerate (zero-variance) matrix.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA pairs observations; row counts differ ({x.shape[0]} vs {y.shape[0]})")
    if x.shape[0] < 2:
        raise ValueError("CKA needs >= 2 paired observations")
    xc = x.float() - x.float().mean(dim=0, keepdim=True)
    yc = y.float() - y.float().mean(dim=0, keepdim=True)
    if xc.var(dim=0).sum() == 0 or yc.var(dim=0).sum() == 0:
        raise ValueError("CKA is undefined for a zero-variance representation")
    cross = (yc.T @ xc).norm()
    return float(cross**2 / (xc.T @ xc).norm() / (yc.T @ yc).norm())


# ---------------------------------------------------------------------------- report
def representation_report(
    extraction: FeatureExtraction,
    class_field: str | None = None,
    seed: int = 0,
) -> dict:
    """Assemble the §14 diagnosis: probe accuracies, drift, CKA, per level.

    Args:
        extraction: from :func:`collect_head_features`, carrying group labels (the
            acquisition field, e.g. ``sensor``) and optionally ``class_field`` labels.
        class_field: label key holding object-class labels; when given, the
            class probe and the within-class drift are computed too.

    Returns a dict whose every number is either measured or absent — a field the
    inputs cannot support (e.g. a class probe with one class) is reported as
    ``None`` with a reason, never as a defaulted value.
    """
    report: dict = {"n_images": extraction.n_images, "levels": {}}
    if not extraction.labels:
        raise ValueError(
            "representation_report needs at least one acquisition label field; features "
            "without labels cannot answer the sensor-entanglement question"
        )
    field_names = list(extraction.labels)

    for level in extraction.levels():
        feats = extraction.features[level]
        level_report: dict = {"spatial": extraction.level_shapes[level]}
        for field_name in field_names:
            labels = extraction.labels[field_name]
            try:
                acc = linear_probe_accuracy(feats, labels, seed=seed)
            except ValueError as exc:
                level_report[f"probe_{field_name}"] = {"accuracy": None, "reason": str(exc)}
                continue
            level_report[f"probe_{field_name}"] = {"accuracy": acc}
        report["levels"][level] = level_report

    group_field = field_names[0]
    group_labels = extraction.labels[group_field]
    top_level = max(extraction.levels())
    feats_top = extraction.features[top_level]

    if class_field is not None and class_field in extraction.labels:
        class_labels = extraction.labels[class_field]
        try:
            drift = class_centroid_distances(feats_top, group_labels, class_labels)
            report["within_class_group_distance"] = {
                f"{a}|{b}": v for (a, b), v in drift.items() if not math.isnan(v)
            }
            report["unmeasurable_pairs"] = [
                f"{a}|{b}" for (a, b), v in drift.items() if math.isnan(v)
            ]
        except ValueError as exc:
            report["within_class_group_distance"] = {"reason": str(exc)}
        if group_field in extraction.labels:
            report["group_field"] = group_field
            report["class_field"] = class_field

    groups = torch.unique(group_labels)
    groups = groups[groups >= 0]
    if groups.numel() >= 2:
        cka = {}
        for ga, gb in combinations(groups.tolist(), 2):
            try:
                cka[f"{ga}|{gb}"] = linear_cka(feats_top[group_labels == ga], feats_top[group_labels == gb])
            except ValueError as exc:
                cka[f"{ga}|{gb}"] = None
                report.setdefault("cka_refusals", {})[f"{ga}|{gb}"] = str(exc)
        report["linear_cka_top_level"] = cka
    return report
