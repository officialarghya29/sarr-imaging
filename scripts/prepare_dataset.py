#!/usr/bin/env python
"""Convert a raw SAR dataset download into the project's processed YOLO layout.

Example
-------
    python scripts/prepare_dataset.py --dataset ssdd --raw /content/raw/SSDD

Then validate and profile it:

    python -m saryolo check-data --dataset datasets/processed/ssdd --classes ship
    python -m saryolo stats --dataset datasets/processed/ssdd --classes ship --name ssdd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saryolo.data import get_dataset, prepare_dataset, write_data_yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Registry key: ssdd, hrsid, sardet100k, sar_ship")
    parser.add_argument("--raw", required=True, help="Directory holding the raw download")
    parser.add_argument("--out-root", default="datasets/processed")
    parser.add_argument("--mode", default="symlink", choices=["symlink", "copy"],
                        help="symlink saves disk; copy survives moving/zipping the tree")
    parser.add_argument("--config-out", default="configs/datasets",
                        help="Where to write the data.yaml (use 'configs/datasets' to overwrite the committed config)")
    args = parser.parse_args()

    spec = get_dataset(args.dataset)
    result = prepare_dataset(args.dataset, args.raw, args.out_root, mode=args.mode)
    root = Path(args.out_root) / spec.name

    splits = tuple(sorted(result.get("splits", {}).keys())) or ("train", "val")
    yaml_path = write_data_yaml(
        Path(args.config_out) / f"{spec.name}.yaml",
        root,
        result.get("classes") or spec.classes,
        splits=splits,
        test="test" if "test" in splits else None,
    )

    print(json.dumps(result, indent=2))
    print(f"\ndata config -> {yaml_path}")
    print("\nNext: validate the conversion before training on it.")
    print(f"  python -m saryolo check-data --dataset {root} --classes {' '.join(result.get('classes') or spec.classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
