#!/usr/bin/env python
"""Fetch a real HRSID subset from the Hugging Face mirror into the repo layout.

Why this exists
---------------
The project's pilot numbers must come from *real* SAR imagery, not the synthetic
smoke fixture. The official HRSID release is distributed through Google Drive and
Baidu, which are not scriptable here; the ``dronefreak/HRSID`` mirror on the Hub
carries the same release already converted to YOLO boxes with the official
train/valid/test split (5,604 images: train 3,096 / valid 546 / test 1,962).

This script pulls a bounded subset with retries, because the network here is
intermittent, and writes it into the repository's expected layout::

    datasets/processed/hrsid_real/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt

Nothing about the images or labels is modified: they are copied byte-for-byte.

Usage
-----
    python scripts/fetch_hrsid_subset.py --train 200 --valid 100 --test 100

The subset is deliberately bounded. A full 5,604-image run needs a GPU and a real
schedule; this exists to prove the pipeline learns on real SAR data and to make
that proof reproducible.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

REPO_ID = "dronefreak/HRSID"
#: Local split name -> repository split name. The mirror calls the middle split
#: ``valid``; every data config here and every Ultralytics convention calls it ``val``,
#: so the local layout uses ``val`` and the mapping is explicit rather than implicit.
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def _list_files(retries: int = 10) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    for attempt in range(retries):
        try:
            return api.list_repo_files(REPO_ID, repo_type="dataset")
        except Exception as exc:  # network is intermittent here
            # `attempt` is used for the human-readable progress, so the loop variable is not
            # dead: a retry loop that reports nothing looks like a hang.
            print(f"  list attempt {attempt + 1}/{retries} failed: {type(exc).__name__}", flush=True)
            time.sleep(5)
    raise SystemExit(f"could not list {REPO_ID} after {retries} attempts")


def _fetch(remote: str, dest: Path, retries: int = 8) -> bool:
    """Download one file into ``dest``, retrying on transport failures."""
    from huggingface_hub import hf_hub_download

    if dest.exists():
        return True
    for _ in range(retries):
        try:
            cached = hf_hub_download(REPO_ID, remote, repo_type="dataset")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy(cached, dest)
            return True
        except Exception:
            time.sleep(2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="datasets/processed/hrsid_real")
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", "--valid", dest="val", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.out)
    targets = {"train": args.train, "val": args.val, "test": args.test}

    # Keep an existing ``valid`` directory (written by an earlier version of this
    # script, which used the mirror's own split name) rather than silently leaving a
    # stale duplicate split on disk next to the ``val`` one.
    stale = out / "images" / "valid"
    if stale.is_dir():
        print(f"NOTE: found a stale '{stale}' from an older layout; rename it to 'val' by hand if it holds more data")

    files = _list_files()
    images = sorted(f for f in files if f.startswith("data/images/") and f.endswith(".jpg"))
    by_split: dict[str, list[str]] = {}
    for f in images:
        by_split.setdefault(f.split("/")[2], []).append(f)
    print(f"mirror holds {len(images)} images: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_split.items())))

    copied = failed = 0
    for split, want in targets.items():
        repo_split = SPLITS[split]
        have = len(list((out / "images" / split).glob("*.jpg"))) if (out / "images" / split).exists() else 0
        print(f"{split}: have {have}, target {want}")
        # Walk the mirror in order but skip anything already on disk, so a bounded
        # top-up continues where the last interrupted run stopped instead of
        # re-downloading the same prefix and leaving the tail permanently empty.
        for remote_img in by_split.get(repo_split, []):
            if len(list((out / "images" / split).glob("*.jpg"))) >= want:
                break
            stem = Path(remote_img).stem
            img_dest = out / "images" / split / f"{stem}.jpg"
            lab_dest = out / "labels" / split / f"{stem}.txt"
            ok_img = _fetch(remote_img, img_dest)
            ok_lab = _fetch(remote_img.replace("/images/", "/labels/").replace(".jpg", ".txt"), lab_dest)
            if ok_img and ok_lab:
                copied += 1
            else:
                failed += 1
        now = len(list((out / "images" / split).glob("*.jpg")))
        print(f"  {split}: now {now} (of {len(by_split.get(repo_split, []))} available)")

    print(f"\nassembled {copied} image/label pairs ({failed} incomplete) into {out}")
    return 1 if failed and not copied else 0


if __name__ == "__main__":
    raise SystemExit(main())
