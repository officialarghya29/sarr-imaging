#!/usr/bin/env python
"""Run the experiment matrix in order, recording every run in the ledger.

Training only the *training* experiments matters here: EXP-009/010/011 reuse the
EXP-007 checkpoint (they are evaluation studies, not separate training runs), so
running them as training would waste GPU hours and produce a redundant row.

Usage
-----
    python scripts/train_all_experiments.py --exp-dir configs/exp --only EXP-001 EXP-002
    python scripts/train_all_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saryolo.training.config import list_experiments, load_experiment  # noqa: E402
from saryolo.training.runner import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", default="configs/exp")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these experiment ids")
    parser.add_argument("--ledger", default="results")
    parser.add_argument("--project", default="results/runs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="List what would run and exit")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed run")
    args = parser.parse_args()

    configs = []
    for path in list_experiments(args.exp_dir):
        try:
            cfg = load_experiment(path)
        except Exception as exc:
            print(f"skip {path.name}: {exc}")
            continue
        raw = yaml.safe_load(path.read_text()) or {}
        if raw.get("evaluation_only"):
            continue
        if args.only and cfg.experiment_id not in args.only:
            continue
        configs.append((path, cfg))

    if not configs:
        print("nothing to run")
        return 0

    print(f"{len(configs)} training run(s):")
    for path, cfg in configs:
        print(f"  {cfg.experiment_id:<10} {path.name:<34} seed={cfg.seed} epochs={cfg.train.get('epochs')}")
    if args.dry_run:
        return 0

    failures = 0
    for path, cfg in configs:
        print(f"\n=== {cfg.experiment_id}: {cfg.name} ===", flush=True)
        try:
            record = run_experiment(path, ledger_root=args.ledger, project=args.project, device=args.device)
        except Exception:
            failures += 1
            traceback.print_exc()
            if not args.keep_going:
                return 1
            continue
        if record.status != "completed":
            failures += 1
            print(f"  FAILED: {record.notes}")
            if not args.keep_going:
                return 1
        else:
            print(f"  mAP50={record.metrics.get('mAP50')} mAP50-95={record.metrics.get('mAP50_95')}")

    print(f"\ndone: {len(configs) - failures} succeeded, {failures} failed")
    print("Generated tables must only use rows with status='completed'.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
