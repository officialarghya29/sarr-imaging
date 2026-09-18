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
from saryolo.data.registry import DATASETS  # noqa: E402

#: Where a freshly prepared data config is written.
#:
#: Deliberately NOT ``configs/datasets``: the committed configs there use *relative*
#: paths so they stay portable, whereas the generated one records the absolute path
#: of the machine that ran the conversion. Writing over the committed files would
#: turn a Colab path like ``/content/datasets/...`` into a tracked change, and a
#: commit would then carry a machine-specific path into the repository.
DEFAULT_CONFIG_OUT = "configs/datasets/generated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help=f"Registry key: {', '.join(sorted(DATASETS))}")
    parser.add_argument("--raw", required=True, help="Directory holding the raw download")
    parser.add_argument("--out-root", default="datasets/processed")
    parser.add_argument("--mode", default="symlink", choices=["symlink", "copy"],
                        help="symlink saves disk; copy survives moving/zipping the tree")
    parser.add_argument("--config-out", default=DEFAULT_CONFIG_OUT,
                        help=f"Where to write the data.yaml (default: {DEFAULT_CONFIG_OUT})")
    args = parser.parse_args()

    spec = get_dataset(args.dataset)
    result = prepare_dataset(args.dataset, args.raw, args.out_root, mode=args.mode)
    root = Path(args.out_root) / spec.name
    classes = result.get("classes") or list(spec.classes)

    splits = tuple(sorted(result.get("splits", {}).keys())) or ("train", "val")
    yaml_path = write_data_yaml(
        Path(args.config_out) / f"{spec.name}.yaml",
        root,
        classes,
        splits=splits,
        test="test" if "test" in splits else None,
    )

    print(json.dumps(result, indent=2))
    print(f"\ndata config -> {yaml_path}")

    if spec.annotation_format == "dota":
        # Fail loudly here rather than let a GPU run discover it. Oriented labels
        # do not match the five-column format the detection models consume.
        print(
            f"\nWARNING: {spec.title} is an ORIENTED (DOTA-format) dataset, so the labels "
            "written above use the YOLO-OBB form (class + 8 corner coordinates).\n"
            "         The detection models in configs/models/ expect 'class cx cy w h' and "
            "will NOT train correctly on them.\n"
            "         Run `python -m saryolo check-data` to confirm, and see "
            f"configs/datasets/{spec.name}.yaml for the options."
        )
    else:
        print("\nNext: validate the conversion before training on it.")
        print(f"  python -m saryolo check-data --dataset {root} --classes {' '.join(classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
