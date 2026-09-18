"""Synthetic SAR-like dataset generator — for PIPELINE TESTING ONLY.

**These images are not real SAR data and no result computed on them may be
reported as a research finding.** Their only purpose is to make the whole
train -> validate -> evaluate -> plot pipeline executable and testable without a
GPU, a dataset download, or network access, so that a wiring bug is caught in
seconds on a laptop instead of after an hour of Colab GPU time.

The generator does model the qualitative properties the project cares about, so
that the modules are at least exercised on the right kind of signal:

* multiplicative speckle (Gamma-distributed, ``L`` looks) — the defining SAR noise
* a low-contrast background with slowly varying reflectivity
* bright, compact target responses with a dihedral-like intensity profile
* optional clutter (bright blobs that are NOT targets) to create hard negatives
* optional strong speckle / low SNR to exercise the robustness suite
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["make_synthetic_dataset"]


def _speckle(shape: tuple[int, int], looks: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative Gamma speckle. Larger ``looks`` -> less severe speckle."""
    if looks <= 0:
        return np.ones(shape, dtype=np.float32)
    return rng.gamma(shape=looks, scale=1.0 / looks, size=shape).astype(np.float32)


def _background(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Smoothly varying background reflectivity in roughly [0.25, 0.55]."""
    h, w = shape
    coarse = rng.random((max(h // 16, 2), max(w // 16, 2))).astype(np.float32)
    import cv2

    bg = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return 0.25 + 0.30 * (bg - bg.min()) / max(bg.max() - bg.min(), 1e-6)


def make_synthetic_dataset(
    out_root: str | Path,
    n_train: int = 64,
    n_val: int = 16,
    n_test: int = 16,
    imgsz: int = 256,
    class_names: tuple[str, ...] = ("target",),
    looks: float = 6.0,
    clutter_rate: float = 0.5,
    max_objects: int = 6,
    min_object_px: float = 6.0,
    max_object_px: float = 26.0,
    seed: int = 0,
) -> dict:
    """Write a synthetic YOLO-format dataset.

    Args:
        out_root: Output root (``images/<split>`` and ``labels/<split>`` created).
        n_train, n_val, n_test: Image counts per split.
        imgsz: Square image size.
        class_names: Class names; objects are assigned round-robin.
        looks: Speckle severity (number of looks); lower is noisier.
        clutter_rate: Expected number of non-target bright blobs per image.
        max_objects: Maximum real targets per image.
        min_object_px, max_object_px: Target size range in pixels.
        seed: RNG seed.

    Returns:
        Summary dict including the paths written.
    """
    import cv2

    out_root = Path(out_root)
    rng = np.random.default_rng(seed)
    totals = {"train": n_train, "val": n_val, "test": n_test}
    counts = {split: {"images": 0, "boxes": 0} for split in totals}

    for split, n in totals.items():
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(n):
            img = _background((imgsz, imgsz), rng) * _speckle((imgsz, imgsz), looks, rng)
            rows: list[str] = []

            # Real targets: bright, compact, with a small specular peak.
            n_obj = int(rng.integers(1, max_objects + 1))
            for k in range(n_obj):
                bw = float(rng.uniform(min_object_px, max_object_px))
                bh = float(rng.uniform(min_object_px, max_object_px))
                cx = float(rng.uniform(bw / 2 + 2, imgsz - bw / 2 - 2))
                cy = float(rng.uniform(bh / 2 + 2, imgsz - bh / 2 - 2))
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
                img[y1:y2, x1:x2] += float(rng.uniform(0.35, 0.75))
                # Bright point return, as a corner reflector would produce.
                img[min(y1 + 1, imgsz - 1):min(y1 + 3, imgsz), min(x1 + 1, imgsz - 1):min(x1 + 3, imgsz)] += 0.9
                cid = k % len(class_names)
                rows.append(f"{cid} {cx / imgsz:.6f} {cy / imgsz:.6f} {bw / imgsz:.6f} {bh / imgsz:.6f}")

            # Hard negatives: bright blobs that are clutter, not targets.
            for _ in range(int(rng.poisson(clutter_rate))):
                bw = float(rng.uniform(min_object_px, max_object_px * 1.5))
                bh = float(rng.uniform(min_object_px, max_object_px * 1.5))
                cx = float(rng.uniform(2, imgsz - 2))
                cy = float(rng.uniform(2, imgsz - 2))
                x1, y1 = int(max(cx - bw / 2, 0)), int(max(cy - bh / 2, 0))
                x2, y2 = int(min(cx + bw / 2, imgsz)), int(min(cy + bh / 2, imgsz))
                if x2 > x1 and y2 > y1:
                    img[y1:y2, x1:x2] += float(rng.uniform(0.25, 0.5))

            img = np.clip(img, 0.0, None)
            # Scale to 8-bit with a mild gamma so the low-contrast regime is preserved.
            img = np.clip(img / max(img.max(), 1e-6), 0, 1) ** 0.9
            img8 = (img * 255.0).astype(np.uint8)

            name = f"{split}_{idx:05d}"
            cv2.imwrite(str(img_dir / f"{name}.png"), img8)
            (lbl_dir / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            counts[split]["images"] += 1
            counts[split]["boxes"] += len(rows)

    readme = out_root / "README_SYNTHETIC.md"
    readme.write_text(
        "# SYNTHETIC pipeline-test data — NOT a research dataset\n\n"
        "Generated by `saryolo.data.synth.make_synthetic_dataset`.\n\n"
        "These images are procedurally generated Gamma-speckle textures with\n"
        "synthetic bright targets. They exist so the train/validate/evaluate/plot\n"
        "pipeline can be exercised end to end on a CPU, without network access.\n\n"
        "**Do not report any metric computed on this data as a research result.**\n"
        "All reported numbers must come from a real benchmark (SSDD, HRSID,\n"
        "SARDet-100K), downloaded and cited per `docs/DATASETS.md`.\n"
    )

    return {
        "root": str(out_root),
        "class_names": list(class_names),
        "nc": len(class_names),
        "splits": counts,
        "synthetic": True,
        "warning": "Synthetic data for pipeline testing only; not reportable.",
    }
