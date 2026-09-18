"""Emit Ultralytics ``data.yaml`` files for a processed dataset.

Ultralytics derives label paths from image paths by replacing ``/images/`` with
``/labels/``, so the processed layout produced by :mod:`saryolo.data.convert`
(``images/<split>`` alongside ``labels/<split>``) is what the trainer expects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

__all__ = [
    "write_data_yaml",
    "dataset_root",
    "project_root",
    "resolve_data_yaml",
    "load_data_config",
    "split_dirs",
    "label_row_kind",
    "DETECTION_FIELDS",
    "OBB_FIELDS",
]

#: A YOLO detection label row is ``class cx cy w h``.
DETECTION_FIELDS = 5

#: A YOLO-OBB label row is ``class x1 y1 x2 y2 x3 y3 x4 y4``.
OBB_FIELDS = 9


def label_row_kind(row: Sequence[str]) -> str:
    """Classify one YOLO label row.

    Both five-field detection rows and nine-field oriented rows are *valid* YOLO
    formats, so they must not be conflated. Treating a nine-field row as generic
    input is actively harmful in two ways, and both are silent:

    * a validator reports thousands of meaningless "malformed row" errors instead
      of the one actionable fact ("this dataset is oriented");
    * a profiler drops every row and reports a well-formed dataset containing
      **zero objects**, which then makes every downstream architecture decision
      ("is the P2 head justified?") meaningless.

    Args:
        row: Whitespace-split fields of one label line.

    Returns:
        ``"detection"``, ``"oriented"`` or ``"malformed"``.
    """
    if len(row) == DETECTION_FIELDS:
        return "detection"
    if len(row) == OBB_FIELDS:
        return "oriented"
    return "malformed"

#: Files that mark the project root when walking upwards.
_ROOT_MARKERS = ("pyproject.toml", ".git", "CITATION.cff")


def project_root(start: str | Path | None = None) -> Path:
    """Locate the project root by walking up from ``start`` for a marker file."""
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return Path.cwd().resolve()


def resolve_data_yaml(
    data_yaml: str | Path,
    root: str | Path | None = None,
    out_dir: str | Path = "results/resolved_data",
) -> Path:
    """Return a data config with ``path`` made absolute and portable.

    Ultralytics resolves a *relative* ``path`` against its global ``datasets_dir``
    setting (``~/datasets`` by default), **not** against the YAML's own directory.
    Committed configs therefore cannot use a relative path if they are to work on
    both a laptop and a Colab VM. This helper rewrites ``path`` to an absolute
    path resolved against the project root and writes the result to ``out_dir``,
    which is what the trainer and every evaluation entry point consume.

    Args:
        data_yaml: A YAML whose ``path`` may be relative to the project root.
        root: Explicit project root, or inferred by walking upwards.
        out_dir: Where the resolved copy is written.

    Returns:
        Path to the resolved YAML (the original file if nothing needed changing).
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")
    cfg = yaml.safe_load(data_yaml.read_text()) or {}
    raw_path = cfg.get("path")
    if raw_path is None:
        raise ValueError(f"{data_yaml} is missing the required 'path' key")
    path = Path(raw_path)
    if path.is_absolute():
        return data_yaml
    resolved = _resolve_relative_root(path, data_yaml, root)
    if resolved is None:
        return data_yaml
    cfg["path"] = str(resolved)
    cfg.setdefault("train", "images/train")
    cfg.setdefault("val", "images/val")
    out = Path(out_dir).resolve() / data_yaml.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def _resolve_relative_root(path: Path, data_yaml: Path, root: str | Path | None) -> Path | None:
    """Resolve a relative ``path`` against the YAML's directory, then the project root.

    Both conventions are accepted so a config can be written either way; the
    first candidate that exists on disk wins. If neither exists, the YAML-relative
    interpretation is returned so the downstream error message points at the
    location the config author most likely meant.
    """
    candidates: list[Path] = []
    candidates.append((data_yaml.parent / path).resolve())
    base = Path(root).resolve() if root else project_root(data_yaml)
    candidates.append((base / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_data_config(data_yaml: str | Path, root: str | Path | None = None) -> tuple[Path, dict]:
    """Load a data config and return ``(root, cfg)`` with the path already absolute.

    Shared by every evaluation path so that train and eval can never disagree
    about which directory a split lives in.
    """
    resolved = resolve_data_yaml(data_yaml, root=root)
    cfg = yaml.safe_load(Path(resolved).read_text()) or {}
    base = Path(cfg.get("path", "."))
    if not base.is_absolute():
        base = (Path(resolved).parent / base).resolve()
    return base, cfg


def dataset_root(processed_root: str | Path, name: str) -> Path:
    """Absolute path to a processed dataset."""
    return (Path(processed_root) / name).resolve()


def split_dirs(data_yaml: str | Path, split: str = "val") -> tuple[Path, Path]:
    """Resolve ``(images_dir, labels_dir)`` for a split of a data config."""
    root, cfg = load_data_config(data_yaml)
    rel = cfg.get(split) or f"images/{split}"
    images_dir = Path(rel) if Path(rel).is_absolute() else root / rel
    labels_dir = Path(str(images_dir).replace("images", "labels", 1))
    return images_dir, labels_dir


def write_data_yaml(
    out_path: str | Path,
    root: str | Path,
    class_names: list[str] | tuple[str, ...],
    splits: tuple[str, ...] = ("train", "val"),
    test: str | None = None,
) -> Path:
    """Write an Ultralytics-compatible ``data.yaml``.

    Args:
        out_path: Where to write the YAML.
        root: Dataset root containing ``images/`` and ``labels/``.
        class_names: Ordered class names (index == class id).
        splits: Which splits exist; ``train`` and ``val`` are always required.
        test: Optional test split directory name.

    Returns:
        The path written.
    """
    root = Path(root).resolve()
    payload: dict = {"path": str(root), "nc": len(class_names), "names": list(class_names)}
    for split in splits:
        payload[split] = f"images/{split}"
    if test:
        payload["test"] = f"images/{test}"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out
