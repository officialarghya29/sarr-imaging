"""SAR-specific augmentation (SEC. 5 of the brief).

Why this reuses :mod:`saryolo.evaluation.robustness` instead of reimplementing
-----------------------------------------------------------------------------
The brief asks for SAR-specific training augmentation -- speckle simulation, SNR variation,
low contrast, blur, resolution degradation, clutter. That list is *exactly* the corruption
model the robustness benchmark already implements and tests. Writing a second one would mean
the augmentation the model trains on and the degradation it is later evaluated under could
drift apart, and a robustness result would then be measuring an inconsistency between two
pieces of code rather than a real property of the model.

Sharing the implementation makes the claim much stronger: the model is trained under the very
same physical model it is tested under, and the severity grid is literally the same tuple.

Appearance-only, which is what makes it safe without label maths
---------------------------------------------------------------
Every corruption in :data:`~saryolo.evaluation.robustness.CORRUPTIONS` is *appearance-only*:
it changes pixel values and never moves, adds or removes a target. That is the property that
makes augmentation possible here without transforming bounding boxes at all -- unusual for
augmentation, and worth stating rather than leaving implicit.

It is enforced, not assumed: :func:`build_augmented_train_split` compares the output shape to
the input shape for every image and refuses to write a mismatched pair, because the moment a
future corruption resizes an image the labels silently become wrong for it.

Offline on purpose, and the trade-off
-------------------------------------
This writes an augmented *copy* of the training split rather than augmenting online once per
epoch. That is a deliberate choice with a real cost, so it is stated plainly:

* **Gained** -- the training set is a committed, inspectable artifact; every run over it is
  byte-identical; each output file's corruption and severity is recorded in a manifest; and
  the multi-view groups it produces are what SEC. 4's consistency loss needs as pairs.
* **Lost** -- the augmentation is fixed rather than resampled per epoch, so the model sees
  ``views`` variants of an image instead of a fresh draw each time. Disk cost is ``views``
  times the training split.

The alternative (an online transform hook) would avoid the disk cost but would put the
augmentation pipeline outside the reproducible, inspectable path the rest of the project uses,
and could not be diffed when a number changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "SAR_AUGMENTATIONS",
    "AugmentationPlan",
    "build_augmented_train_split",
    "default_plan",
]

#: Which corruptions are used for training augmentation, by default, and why.
#:
#: ``clutter`` is deliberately **absent** from the default set. It is a useful robustness
#: probe, but as *training* augmentation it injects bright blobs that are not labelled, which
#: teaches the model to suppress bright compact regions -- the exact appearance of the small
#: targets the paper is trying to improve. It stays available for the ablation that tests that
#: concern; it is not in the default because the default should not gamble the headline claim.
SAR_AUGMENTATIONS: tuple[str, ...] = (
    "speckle",
    "low_contrast",
    "blur",
    "low_resolution",
    "low_snr",
)


@dataclass
class AugmentationPlan:
    """Which degradations to draw from, how many views, and how strongly.

    Attributes:
        kinds: Corruption names, each a key of ``robustness.CORRUPTIONS``.
        views: Number of augmented copies emitted per training image.
        severities: Optional per-kind restriction of the severity grid, e.g.
            ``{"speckle": (8.0, 4.0)}`` to train only on the milder half. Restricting to a
            subset of the published grid is allowed; inventing severities outside it is not,
            because the robustness figure's x-axis has to stay comparable.
        include_clean: Whether to emit one undeformed copy alongside the views. Kept on by
            default so the training set does not become a purely degraded distribution.
        seed: Seed for the per-image draw, so the plan is reproducible.
    """

    kinds: tuple[str, ...] = SAR_AUGMENTATIONS
    views: int = 2
    severities: dict[str, tuple[float, ...]] = field(default_factory=dict)
    include_clean: bool = True
    seed: int = 0

    def grid(self, kind: str) -> tuple[float, ...]:
        """Severity grid actually used for ``kind``, validated against the published one."""
        from saryolo.evaluation.robustness import CORRUPTIONS

        if kind not in CORRUPTIONS:
            raise KeyError(f"Unknown corruption {kind!r}. Known: {sorted(CORRUPTIONS)}")
        published = CORRUPTIONS[kind].severities
        chosen = tuple(self.severities.get(kind, published))
        if not chosen:
            raise ValueError(f"Severity grid for {kind!r} is empty")
        stray = [s for s in chosen if s not in published]
        if stray:
            raise ValueError(
                f"Severity {stray} for {kind!r} is outside the published grid {published}. "
                "Training severities must come from the robustness grid, or the augmented run "
                "and the robustness benchmark would no longer refer to the same degradation."
            )
        return chosen


def default_plan(views: int = 2, seed: int = 0) -> AugmentationPlan:
    """The default training plan: every SAR augmentation except clutter. See the module docstring."""
    return AugmentationPlan(kinds=SAR_AUGMENTATIONS, views=views, include_clean=True, seed=seed)


def build_augmented_train_split(
    images_dir: str | Path,
    out_root: str | Path,
    plan: AugmentationPlan | None = None,
    labels_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Write a multi-view augmented copy of a training split, plus a manifest.

    Labels are copied unchanged. That is correct only because every corruption is
    appearance-only, which is verified per image rather than assumed.

    Args:
        images_dir: Source images, read only. The raw dataset is never modified.
        out_root: Destination root; images land in ``images/train``, labels in ``labels/train``.
        plan: The augmentation plan; defaults to :func:`default_plan`.
        labels_dir: Label directory; defaults to the ``images``->``labels`` sibling of
            ``images_dir``.
        limit: Optional cap on the number of source images, for smoke runs.

    Returns:
        The manifest dict, also written to ``manifest.json`` under ``out_root``. Each entry
        records the source image, the emitted file, the corruption and the severity applied
        (``null`` for the clean copy), and the view group, so any training degradation can be
        traced back to the exact parameter that produced it.
    """
    import cv2

    from saryolo.evaluation.robustness import apply_corruption

    plan = plan or default_plan()
    images_dir, out_root = Path(images_dir), Path(out_root)
    img_dest = out_root / "images" / "train"
    lbl_dest = out_root / "labels" / "train"
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(plan.seed)
    images = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    )
    if limit:
        images = images[:limit]

    labels_src = Path(labels_dir) if labels_dir else Path(str(images_dir).replace("images", "labels", 1))

    entries: list[dict] = []
    skipped: list[dict] = []
    for path in images:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append({"image": path.name, "reason": "unreadable"})
            continue
        label = labels_src / f"{path.stem}.txt"
        if not label.exists():
            # No label file means no boxes, which is legal (a pure-background scene) but must
            # not silently become a training image whose "ground truth" is unknown.
            skipped.append({"image": path.name, "reason": "no_label_file"})
            continue
        label_text = label.read_text()

        emissions: list[tuple[str, float | None]] = []
        if plan.include_clean:
            emissions.append(("identity", None))
        for _ in range(plan.views):
            kind = str(rng.choice(list(plan.kinds)))
            grid = plan.grid(kind)
            severity = float(grid[int(rng.integers(0, len(grid)))] if len(grid) > 1 else grid[0])
            emissions.append((kind, severity))

        for view, (kind, severity) in enumerate(emissions):
            tag = "clean" if kind == "identity" else f"{kind}{severity:g}"
            name = f"{path.stem}__v{view}_{tag}{path.suffix}"
            out_img = apply_corruption(img, kind, severity if severity is not None else 0.0, rng)
            if out_img.shape[:2] != img.shape[:2]:
                # Appearance-only is the entire justification for not transforming labels.
                # If this ever fires, the labels for this file would be wrong, so refuse.
                raise ValueError(
                    f"Corruption {kind!r} changed image shape {img.shape[:2]} -> "
                    f"{out_img.shape[:2]}; labels would no longer correspond to the image."
                )
            cv2.imwrite(str(img_dest / name), out_img)
            (lbl_dest / f"{Path(name).stem}.txt").write_text(label_text)
            entries.append({
                "source": str(path),
                "image": name,
                "group": path.stem,
                "view": view,
                "corruption": None if kind == "identity" else kind,
                "severity": severity,
                "param": None if kind == "identity" else _param_name(kind),
            })

    manifest = {
        "plan": {
            "kinds": list(plan.kinds),
            "views": plan.views,
            "include_clean": plan.include_clean,
            "seed": plan.seed,
            "severities": {k: list(plan.grid(k)) for k in plan.kinds},
        },
        "images_source": str(images_dir),
        "n_source": len(images),
        "n_emitted": len(entries),
        "entries": entries,
        "skipped": skipped,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _param_name(kind: str) -> str:
    """The physical parameter name of a corruption (``looks``, ``sigma``, ...)."""
    from saryolo.evaluation.robustness import CORRUPTIONS

    return CORRUPTIONS[kind].param_name
