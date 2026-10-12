"""Cross-dataset (domain-shift) evaluation.

A SAR detector that only works on the dataset it was trained on has not
demonstrated generalization. This module evaluates an existing checkpoint on a
*different* dataset's split and reports the domain gap.

Guard rails
-----------
Cross-dataset numbers are only meaningful when the label spaces are compatible.
Comparing a ship-only detector against a 6-class dataset produces a number that
looks like a result but measures nothing — so class-name overlap is checked and
an incompatible pairing raises instead of quietly reporting a low mAP. Any class
present in the target set but absent from the source set is reported explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

__all__ = ["class_compatibility", "cross_dataset_eval", "domain_gap_table"]


def _names(data_yaml: str | Path) -> list[str]:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    names = cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n).lower() for n in names]


def class_compatibility(source_yaml: str | Path, target_yaml: str | Path) -> dict:
    """Report how the two label spaces line up.

    Returns:
        Dict with ``compatible``, ``shared``, ``source_only`` and ``target_only``
        class lists. ``compatible`` means the two label spaces **share at least one class**,
        which is the minimum for a score to mean anything. It is deliberately *not* a subset
        test: evaluation is scored on the target ground truth, so a source class that does not
        appear in the target simply never enters the score. The stricter risk -- a target class
        the model cannot possibly predict -- is the separate ``target_only`` condition that
        :func:`cross_dataset_eval` enforces.
    """
    src, tgt = _names(source_yaml), _names(target_yaml)
    shared = sorted(set(src) & set(tgt))
    return {
        "source_classes": src,
        "target_classes": tgt,
        "shared": shared,
        "source_only": sorted(set(src) - set(tgt)),
        "target_only": sorted(set(tgt) - set(src)),
        "compatible": len(shared) > 0,
    }


def cross_dataset_eval(
    weights: str | Path,
    target_data_yaml: str | Path,
    source_data_yaml: str | Path | None = None,
    imgsz: int = 640,
    device: str | None = None,
    out_dir: str | Path = "results/generalization",
    allow_partial_overlap: bool = False,
) -> dict:
    """Evaluate a checkpoint trained on one dataset against another.

    Args:
        weights: Checkpoint to evaluate.
        target_data_yaml: Data config of the *evaluation* dataset.
        source_data_yaml: Data config of the *training* dataset, if known.
            Supplying it enables the class-compatibility guard.
        imgsz: Evaluation input size.
        device: Torch device string.
        out_dir: Where prediction artefacts are written.
        allow_partial_overlap: Permit evaluation when only some classes overlap.
            Without this, a partial overlap raises, because the resulting mAP
            would silently average over classes the model cannot possibly predict.

    Returns:
        Metrics dict augmented with ``compatibility`` and a ``domain_shift`` flag.
    """
    from .metrics import evaluate_detections

    out_dir = Path(out_dir)
    compatibility = None
    same_dataset = None
    if source_data_yaml is not None:
        compatibility = class_compatibility(source_data_yaml, target_data_yaml)
        # ``domain_shift`` used to be the literal ``True`` on every call, so pointing ``--source``
        # and ``--target`` at one config produced an in-domain evaluation carrying a domain-shift
        # label. Measured on this repository: ``cross-dataset --source
        # configs/datasets/hrsid_real.yaml --target configs/datasets/hrsid_real.yaml`` reported
        # ``domain_shift: true`` for the dataset the checkpoint was trained on. The verdict is
        # now computed from the two resolved data roots, falling back to the config paths when a
        # root cannot be resolved, so no new key is required of a caller's data config.
        same_dataset = _same_dataset(source_data_yaml, target_data_yaml)
        if not compatibility["compatible"]:
            raise ValueError(
                f"No class overlap between {source_data_yaml} and {target_data_yaml}: "
                f"{compatibility['source_only']} vs {compatibility['target_only']}. "
                "Cross-dataset evaluation would be meaningless."
            )
        if compatibility["target_only"] and not allow_partial_overlap:
            raise ValueError(
                f"Target dataset has classes the source model cannot predict: {compatibility['target_only']}. "
                "Pass allow_partial_overlap=True only if you will report this limitation explicitly."
            )

    metrics = evaluate_detections(weights, target_data_yaml, imgsz=imgsz, device=device, out_dir=out_dir)
    metrics["compatibility"] = compatibility
    # ``None`` means "no source config was supplied, so this pairing was not checked" -- the same
    # convention ``compatibility`` uses, rather than claiming a shift nobody established.
    metrics["domain_shift"] = None if same_dataset is None else not same_dataset
    metrics["same_dataset_as_source"] = same_dataset
    # Create the directory here rather than inherit it from the evaluator. In the current path
    # ``evaluate_detections`` happens to create it (Ultralytics' ``project=`` does), so this was
    # never reached; but every other writer in this repository creates its own output directory,
    # and relying on a callee's side effect makes this function fail at its very last step --
    # after a full evaluation -- the moment that callee changes. A no-op when the directory exists.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_dataset.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def _data_root(data_yaml) -> Path | None:
    """The dataset directory a config names, or ``None`` when it cannot be established.

    A config that does not resolve is not an error here -- this only decides whether two configs
    describe one dataset, and refusing to answer is the honest outcome when one of them cannot
    be resolved.
    """
    from saryolo.data.yolo import load_data_config

    try:
        root, _ = load_data_config(data_yaml)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return Path(root).resolve()


def _same_dataset(source_data_yaml, target_data_yaml) -> bool:
    """Whether two data configs describe the same dataset.

    Compares resolved data roots, which is what makes two differently-named configs for one
    directory compare equal; falls back to the two config paths when either root is unavailable,
    which can only be wrong in the conservative direction (reporting a shift that may not be
    there, never hiding one).
    """
    source_root, target_root = _data_root(source_data_yaml), _data_root(target_data_yaml)
    if source_root is not None and target_root is not None:
        return source_root == target_root
    return Path(source_data_yaml).resolve() == Path(target_data_yaml).resolve()


def domain_gap_table(in_domain: dict, cross_domain: dict) -> list[dict]:
    """Build the domain-gap rows (in-domain vs cross-domain) for the paper table.

    Missing measurements are carried through as ``None`` so the table generator
    renders ``TBD`` rather than a fabricated drop.
    """
    rows = []
    for label, key in (("mAP50", "mAP50"), ("mAP50:95", "mAP50_95")):
        a = in_domain.get(key)
        b = cross_domain.get(key)
        drop = (a - b) if (a is not None and b is not None) else None
        rows.append({
            "metric": label,
            "in_domain": a,
            "cross_domain": b,
            "absolute_drop": drop,
            "relative_drop_pct": (100.0 * drop / a) if (drop is not None and a) else None,
        })
    return rows
