#!/usr/bin/env python
"""Generate the RT-DETR conditioning feasibility variant YAML.

Why a generator
---------------
A YOLO YAML's ``from`` column is positional: inserting one row silently invalidates
every later index. That is why the YOLO variants are produced by the symbolic
builder (`saryolo.nn.arch`); the same hazard applies to the RT-DETR graph, so this
file is *generated from the installed ultralytics stock YAML* — never hand-edited.

What it produces
----------------
The stock ``yolov8-rtdetr`` graph (scale ``s``) with exactly one
``AcquisitionConditionedAdapter`` inserted on the deepest neck level (P5) before
the decoder, every downstream ``from`` index shifted, and the decoder's source
list repointed at the adapter's output for P5.

Usage::

    python scripts/make_rtdetr_variant.py           # writes configs/models/rtdetr/
    python scripts/make_rtdetr_variant.py --check   # exit 1 if the committed file drifted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "configs" / "models" / "rtdetr"

#: Insertion point: the deepest neck ``C2f`` (P5) — conditioning the coarsest level
#: first keeps the decoder-side indices stable for the two finer levels.
FIELDS = "sensor_resolution"
ADAPTER_MODE = "film"

HEADER = """\
# SAR-YOLO / RT-DETR conditioning feasibility variant -- GENERATED FILE, do not edit by hand.
# Regenerate with:  python scripts/make_rtdetr_variant.py
# Graph: stock ultralytics yolov8-rtdetr (scale s) + one AcquisitionConditionedAdapter
# inserted on the deepest neck level (P5) before the decoder. See SARYOLORTDetectionModel
# in saryolo/nn/model.py for the scope of this arm (feasibility, not a measured result).
#
# [from, repeats, module, args]
"""


def build_variant() -> dict:
    """Load the stock RT-DETR YAML, insert the adapter row, fix every index."""
    import ultralytics.cfg as cfg_pkg

    stock_path = Path(cfg_pkg.__file__).parent / "models" / "v8" / "yolov8-rtdetr.yaml"
    if not stock_path.exists():  # pragma: no cover - guards ultralytics layout changes
        raise FileNotFoundError(f"stock RT-DETR YAML not found at {stock_path}")
    config = yaml.safe_load(stock_path.read_text())
    config["nc"] = 1
    config["scales"] = {"s": list(config["scales"]["s"])}

    backbone, head = config["backbone"], config["head"]
    n_backbone = len(backbone)
    c2f_positions = [i for i, row in enumerate(head) if row[2] == "C2f"]
    if not c2f_positions:
        raise ValueError("no C2f rows found in the stock RT-DETR head; ultralytics layout changed?")
    deepest_list_pos = c2f_positions[-1]
    deepest_layer_idx = n_backbone + deepest_list_pos

    adapter_row = [
        -1,
        1,
        "AcquisitionConditionedAdapter",
        ["ch", deepest_layer_idx, ADAPTER_MODE, FIELDS, [2, 2, 2]],
    ]
    head.insert(deepest_list_pos + 1, adapter_row)

    def shift(reference):
        """Shift every downstream ``from`` by +1 past the insertion point."""
        if isinstance(reference, int):
            return reference + 1 if reference >= deepest_layer_idx else reference
        if isinstance(reference, list):
            return [shift(r) for r in reference]
        return reference

    for row in head[deepest_list_pos + 2 :]:
        row[0] = shift(row[0])

    # The decoder consumed [P3, P4, P5]; P5 is now the adapter's output instead of the C2f.
    decoder = head[-1]
    if decoder[2] != "RTDETRDecoder":
        raise ValueError(f"expected the last head row to be the RTDETRDecoder, got {decoder[2]!r}")
    sources = decoder[0]
    sources[-1] = deepest_layer_idx + 1  # adapter sits directly after the P5 C2f
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed file matches, do not write")
    args = parser.parse_args()

    rendered = HEADER + yaml.safe_dump(build_variant(), sort_keys=False, default_flow_style=None)
    out_path = OUT_DIR / "rtdetr_s_cond_film.yaml"

    if args.check:
        if not out_path.exists():
            print(f"MISSING: {out_path.relative_to(REPO_ROOT)} — run scripts/make_rtdetr_variant.py")
            return 1
        if out_path.read_text() != rendered:
            print(f"DRIFTED: {out_path.relative_to(REPO_ROOT)} differs from the generator output")
            return 1
        print(f"OK: {out_path.relative_to(REPO_ROOT)} matches the generator")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
