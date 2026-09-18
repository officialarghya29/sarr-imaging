"""Train/val/test splitting with explicit leakage control.

Two leakage risks matter for SAR benchmarks:

1. **Random splits of scene-derived chips.** HRSID and SAR-Ship-Dataset are cut
   into chips from a small number of large acquisitions. A random chip split puts
   near-identical patches in train and test and inflates mAP substantially. The
   splitter therefore supports grouping by scene/acquisition key.
2. **Official split overrides.** Where a dataset ships an official split, that
   split is authoritative — results are only comparable to the literature if it
   is used. This module reads those files and, importantly, *verifies* them
   instead of re-splitting.

``leakage_report`` quantifies what remains: exact duplicates and near-duplicate
pairs that straddle a split boundary.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["split_files", "leakage_report", "read_official_split", "LeakageReport"]


def read_official_split(path: str | Path) -> dict[str, list[str]]:
    """Read an official split file, inferring which split each entry belongs to.

    Accepts either a directory containing ``train.txt``/``val.txt``/``test.txt``,
    or a single file (in which case the split name is taken from its file name).
    """
    path = Path(path)
    out: dict[str, list[str]] = {}
    if path.is_dir():
        for candidate in sorted(path.glob("*.txt")):
            name = candidate.stem.lower()
            name = {"valid": "val", "validation": "val"}.get(name, name)
            if name in ("train", "val", "test"):
                out[name] = [line.strip() for line in candidate.read_text().split() if line.strip()]
    else:
        name = path.stem.lower()
        name = {"valid": "val", "validation": "val"}.get(name, name)
        out[name] = [line.strip() for line in path.read_text().split() if line.strip()]
    if not out:
        raise FileNotFoundError(f"No usable split lists found at {path}")
    return out


def split_files(
    files: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 0,
    scene_key=None,
) -> dict[str, list[str]]:
    """Deterministically split file names into train/val/test.

    Args:
        files: Identifiers to split (typically image stems).
        ratios: ``(train, val, test)`` fractions; must sum to ~1.
        seed: Random seed; recorded by the experiment ledger for reproducibility.
        scene_key: Optional callable mapping a file to its scene/acquisition id.
            When given, splitting is performed over *scenes* so that all chips
            from one acquisition stay in a single split.

    Returns:
        Mapping ``{"train": [...], "val": [...], "test": [...]}``.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    rng = random.Random(seed)
    units: dict[str, list[str]] = defaultdict(list)
    if scene_key is None:
        for f in files:
            units[f].append(f)
    else:
        for f in files:
            units[str(scene_key(f))].append(f)
    keys = sorted(units)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    if n and n_train == 0:
        n_train = 1
    groups = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    return {split: sorted(f for k in ks for f in units[k]) for split, ks in groups.items()}


@dataclass
class LeakageReport:
    """Leakage findings across split boundaries."""

    n_files: int = 0
    exact_duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)
    near_duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)
    split_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.exact_duplicate_pairs and not self.near_duplicate_pairs

    def summary(self) -> str:
        lines = [
            "Leakage report",
            f"  files: {self.n_files}   splits: {self.split_sizes}",
            f"  exact duplicate pairs across splits: {len(self.exact_duplicate_pairs)}",
            f"  near-duplicate pairs across splits:  {len(self.near_duplicate_pairs)}",
        ]
        if not self.clean:
            lines.append("  ACTION REQUIRED: re-split by scene, or move duplicates into one split.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_files": self.n_files,
            "split_sizes": self.split_sizes,
            "exact_duplicate_pairs": self.exact_duplicate_pairs,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "clean": self.clean,
        }


def leakage_report(
    splits: dict[str, list[str]],
    image_dir: str | Path,
    near_dup_threshold: int = 4,
) -> LeakageReport:
    """Check for exact and near-duplicate images appearing in different splits.

    Args:
        splits: ``{"train": [stem, ...], "val": [...], "test": [...]}``.
        image_dir: Directory containing the images.
        near_dup_threshold: dHash Hamming distance treated as near-duplicate.
    """
    import cv2
    import numpy as np

    from .validate import _dhash, _md5  # reuse the same hashing so reports agree

    image_dir = Path(image_dir)
    index = {}
    for path in image_dir.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            index.setdefault(path.stem, path)
            index.setdefault(path.name, path)

    report = LeakageReport()
    hashes: dict[str, tuple[str, int]] = {}
    for split, names in splits.items():
        report.split_sizes[split] = len(names)
        for name in names:
            path = index.get(Path(name).stem) or index.get(name)
            if path is None:
                continue
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            hashes[f"{split}:{name}"] = (_md5(path), _dhash(img))
    report.n_files = len(hashes)

    items = list(hashes.items())
    for i in range(len(items)):
        (key_i, (md5_i, hash_i)) = items[i]
        split_i = key_i.split(":", 1)[0]
        for j in range(i + 1, len(items)):
            (key_j, (md5_j, hash_j)) = items[j]
            split_j = key_j.split(":", 1)[0]
            if split_i == split_j:
                continue
            if md5_i == md5_j:
                report.exact_duplicate_pairs.append((key_i, key_j))
            elif bin(hash_i ^ hash_j).count("1") <= near_dup_threshold:
                report.near_duplicate_pairs.append((key_i, key_j))
    return report


def write_splits(splits: dict[str, list[str]], out_dir: str | Path) -> Path:
    """Persist split lists to ``datasets/splits/<name>/{train,val,test}.txt``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, names in splits.items():
        (out_dir / f"{split}.txt").write_text("\n".join(names) + ("\n" if names else ""))
    (out_dir / "split_summary.json").write_text(
        json.dumps({"sizes": {k: len(v) for k, v in splits.items()},
                    "class_balance": _class_balance(Path(out_dir).parent, splits)}, indent=2)
    )
    return out_dir


def _class_balance(root: Path, splits: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Per-split box counts by class, to expose split-induced class imbalance."""
    labels_root = root.parent / "processed"
    balance: dict[str, Counter] = {}
    for split, names in splits.items():
        counter: Counter = Counter()
        for label_dir in labels_root.glob(f"*/labels/{split}"):
            for name in names:
                label = label_dir / f"{Path(name).stem}.txt"
                if not label.exists():
                    continue
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if parts:
                        counter[parts[0]] += 1
        balance[split] = counter
    return {k: dict(v) for k, v in balance.items()}
