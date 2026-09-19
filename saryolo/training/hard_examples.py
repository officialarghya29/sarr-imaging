"""Hard-example mining (SEC. 6 of the brief) as an offline training strategy.

Why this is *not* a new module
------------------------------
The brief asks for hard-example mining after baseline training: predict, score the errors,
prioritise the difficult images, retrain. That is a change to the *sampling*, not to the
architecture, and it does not need a custom dataset class or a patched trainer -- ultralytics
accepts a ``.txt`` file listing image paths as a split, and an image listed twice is sampled
twice. So the whole strategy is:

1. score every image by how badly the model did on it (this module),
2. emit a training list containing all training images **plus** the hardest ones repeated,
3. emit a data config whose ``train`` points at that list.

Nothing is approximated and no bookkeeping lives inside the trainer, which is what makes the
result attributable: the two runs differ only in the list of images they saw, and both lists
are committed artifacts.

Why it reuses the failure taxonomy rather than matching boxes itself
-------------------------------------------------------------------
The first version of this module carried its own greedy matcher. That was wrong for two
reasons, and both are the kind the project's own rules forbid:

* it made a **third** IoU matcher (alongside ``metrics._match_image`` and
  ``error_analysis._iou``), so three places could disagree about what a "match" is;
* worse, it meant the hard-example count and the paper's failure-analysis table were computed
  by *different* code, so the two could silently contradict each other about the same model.

So difficulty is derived from :func:`saryolo.visualization.error_analysis.analyse_failures`,
the same taxonomy the failure table is built from. Because the taxonomy assigns each
prediction and each ground truth **exactly one** outcome, the per-image difficulty is a plain
weighted histogram over those outcomes, and the totals reconcile with the failure table by
construction rather than by convention.

What the weights express
------------------------
A hard example is not simply "low mAP". The failures this paper exists to fix are weighted
highest, and the weights are explicit arguments because they are a choice the paper has to
justify and an ablation has to vary:

* ``small_object_miss`` -- the headline claim is small-object recall, so a dropped small
  target is the most valuable thing to train on again;
* ``false_negative`` -- a target the model did not fire on at all;
* ``false_positive`` / ``clutter_false_positive`` -- clutter and speckle responding as a
  target, the failure Component 2 exists to suppress;
* ``localization_error`` -- found but poorly placed;
* ``classification_error`` -- found in the right place with the wrong label;
* ``clutter_confusion`` -- partial or duplicate firings on one object, the failure the
  deformable refinement addresses.

The score is normalised by the ground-truth count so that a dense scene is not automatically
"harder" than a sparse one where the model failed on the single object present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DIFFICULTY_WEIGHTS",
    "ImageDifficulty",
    "score_examples",
    "rank_examples",
    "write_oversampled_list",
    "write_hard_data_config",
]

#: Weight of each taxonomy outcome in the difficulty score. Every key of the taxonomy has an
#: entry, and :func:`score_examples` raises if the taxonomy grows a kind this dict does not
#: cover -- an unweighted failure mode silently scoring zero is exactly how a miner stops
#: mining the thing it was added for.
DIFFICULTY_WEIGHTS: dict[str, float] = {
    "small_object_miss": 3.0,
    "false_negative": 1.0,
    "clutter_false_positive": 1.5,
    "false_positive": 1.0,
    "localization_error": 1.0,
    "classification_error": 1.5,
    "clutter_confusion": 0.5,
    # A correct detection contributes nothing to difficulty, but it must still be *known*:
    # a taxonomy that failed to report hits would make every image look hard.
    "true_positive": 0.0,
}


@dataclass
class ImageDifficulty:
    """Difficulty of one image, with the per-outcome counts the score was built from.

    The counts are kept because the score alone is not interpretable: two images can score the
    same for opposite reasons (one dropped four tiny targets, the other hallucinated four
    blobs), and a report that hides which is which cannot support a claim about *why*
    retraining helped.
    """

    image: str
    n_gt: int = 0
    n_det: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    score: float = 0.0

    @property
    def missed(self) -> int:
        return self.counts.get("false_negative", 0) + self.counts.get("small_object_miss", 0)

    @property
    def spurious(self) -> int:
        return (
            self.counts.get("false_positive", 0)
            + self.counts.get("clutter_false_positive", 0)
        )

    def as_row(self) -> dict:
        return {
            "image": self.image,
            "n_gt": self.n_gt,
            "n_det": self.n_det,
            "missed": self.missed,
            "spurious": self.spurious,
            "score": round(self.score, 4),
            "counts": dict(self.counts),
        }


def score_examples(
    preds: list,
    gts: list,
    weights: dict[str, float] | None = None,
    contrast_by_image: dict[str, float] | None = None,
    small_area: float = 32.0**2,
    iou_threshold: float = 0.5,
) -> list[ImageDifficulty]:
    """Score every image by how badly the model failed on it, via the failure taxonomy.

    Args:
        preds: Predicted :class:`~saryolo.evaluation.metrics.Detection` objects (all images).
        gts: Ground-truth ``Detection`` objects (all images).
        weights: Outcome weights; defaults to :data:`DIFFICULTY_WEIGHTS` merged over the
            defaults, so a caller may tune one weight without restating the rest.
        contrast_by_image: Optional per-image local contrast, forwarded to the taxonomy so that
            false positives in the low-contrast tail are counted as ``clutter_false_positive``.
        small_area: Area in pixels below which a ground truth counts as small. The COCO
            convention (``32**2``) is the default so the miner emphasises exactly the objects
            ``AP_small`` reports on.
        iou_threshold: IoU at which a detection counts as having found an object.

    Returns:
        One :class:`ImageDifficulty` per image that appears in either list, ordered by
        descending score (ties broken by image name, so the order is deterministic).
    """
    from saryolo.visualization.error_analysis import analyse_failures

    effective = dict(DIFFICULTY_WEIGHTS)
    if weights:
        unknown = set(weights) - set(effective)
        if unknown:
            raise KeyError(f"Unknown difficulty weight(s) {sorted(unknown)}; known: {sorted(effective)}")
        effective.update(weights)

    summary = analyse_failures(
        preds,
        gts,
        iou_threshold=iou_threshold,
        small_area=small_area,
        contrast_by_image=contrast_by_image,
    )
    # The taxonomy raises no error for a kind it does not know, so the coverage check happens
    # here, against what it actually produced.
    missing = set(summary.counts) - set(effective)
    if missing:
        raise KeyError(
            f"Failure taxonomy produced outcome(s) {sorted(missing)} with no difficulty weight. "
            "Add them to DIFFICULTY_WEIGHTS: an unweighted outcome would silently score zero."
        )

    rows: dict[str, ImageDifficulty] = {
        image: ImageDifficulty(image=image, counts=dict(counts))
        for image, counts in summary.per_image().items()
    }

    # Image sizes come from the ground truth, and an image with no ground truth (pure clutter)
    # still needs a denominator: `max(n_gt, 1)` keeps its false positives *visible* instead of
    # dividing its score to zero, which is the one case where the clutter response is the
    # entire signal.
    for gt in gts:
        rows.setdefault(gt.image, ImageDifficulty(image=gt.image)).n_gt += 1
    for det in preds:
        rows.setdefault(det.image, ImageDifficulty(image=det.image)).n_det += 1

    for row in rows.values():
        numerator = sum(effective[kind] * n for kind, n in row.counts.items())
        row.score = numerator / max(row.n_gt, 1)

    return sorted(rows.values(), key=lambda r: (-r.score, r.image))


def rank_examples(rows: list[ImageDifficulty], top_frac: float = 0.2, min_count: int = 0):
    """Split a scored list into ``(hard, easy)``, hardest first.

    Args:
        rows: Output of :func:`score_examples`.
        top_frac: Fraction of images treated as hard.
        min_count: Lower bound on the size of the hard set, so a small validation split does
            not produce an empty or single-image set.

    Returns:
        ``(hard, easy)``; ``hard`` is empty only when ``top_frac`` and ``min_count`` are both
        zero, in which case there is nothing to oversample and that is worth seeing.
    """
    ordered = sorted(rows, key=lambda r: (-r.score, r.image))
    n_hard = max(int(round(len(ordered) * max(top_frac, 0.0))), int(min_count))
    if n_hard <= 0:
        return [], ordered
    return ordered[:n_hard], ordered[n_hard:]


def write_oversampled_list(
    all_images: list[str | Path],
    hard_images: list[str | Path],
    out_path: str | Path,
    repeats: int = 2,
) -> Path:
    """Write a YOLO image-list file with the hard images repeated.

    Every image in ``all_images`` is written exactly once first, so the hard set is *added*
    empirically rather than substituted for the rest of the training data. A list that dropped
    the other images would be a smaller, different dataset, and the comparison against the
    baseline run would no longer be controlled.

    ``repeats=2`` means a hard image appears twice per epoch. The file is the only artifact
    that differs between the two runs, which is what makes the ablation attributable.

    Raises:
        ValueError: If a requested hard image is not in ``all_images``. This was previously
            filtered out silently, which made the whole strategy a no-op that still reported
            success: mining the *validation* split and passing those paths here wrote a list
            containing none of them, because a val image is never in the train list. Silently
            dropping is the one thing this cannot do -- the run would look like hard-example
            training and be identical to the baseline.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    base = [str(Path(p)) for p in all_images]
    present = set(base)
    hard = {str(Path(p)) for p in hard_images}
    missing = sorted(hard - present)
    if missing:
        shown = ", ".join(missing[:3])
        raise ValueError(
            f"{len(missing)} hard image(s) are not in the training list, so they cannot be "
            f"oversampled: {shown}{', ...' if len(missing) > 3 else ''}. "
            "Hard examples must be mined from the split that is being trained on; a hard "
            "image from another split is either ignored (making this a no-op) or leaks that "
            "split into training."
        )

    lines = list(base)
    # `repeats - 1` extra copies: the first is already in `lines`. `repeats=1` therefore means
    # "no oversampling" and the file is exactly the input list, which is the honest control.
    for _ in range(max(repeats - 1, 0)):
        lines.extend(p for p in base if p in hard)
    out.write_text("\n".join(lines) + "\n")
    return out


def write_hard_data_config(
    source_data_yaml: str | Path,
    train_list: str | Path,
    out_path: str | Path,
    note: str = "",
) -> Path:
    """Write a data config identical to ``source_data_yaml`` but with ``train`` on a list file.

    ``val``/``test`` and the class names are copied unchanged: the evaluation protocol must not
    move with the training strategy, or the two runs would not be comparable.
    """
    import yaml

    from saryolo.data.yolo import load_data_config

    _root, cfg = load_data_config(source_data_yaml)
    out_cfg = dict(cfg)
    # `path` is already absolute in `cfg` (load_data_config resolves it), so it is copied
    # verbatim. Only `train` moves -- onto an absolute image list.
    out_cfg["train"] = str(Path(train_list).resolve())
    header = (
        "# GENERATED by saryolo.training.hard_examples -- do not edit by hand.\n"
        "# Same splits and classes as the source config; only `train` is an image list,\n"
        "# so the hard-example run and the baseline differ solely in what was sampled.\n"
    )
    if note:
        header += f"# {note}\n"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump(out_cfg, sort_keys=False))
    return out
