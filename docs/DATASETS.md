# Datasets

> **None of these datasets are redistributed by this repository.** Only the code
> is MIT licensed. Obtain each dataset from its official source under its own
> terms, and cite the original paper in any publication.

There is no widely used public dataset named "SARR". The pipeline is therefore
dataset-agnostic and is developed across the public SAR benchmarks below,
cheapest first, so that the pipeline and modules are trustworthy before spending
GPU hours on the large benchmark.

| Order | Key | Dataset | Images | Classes | Annotations | Tier |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `ssdd` | SSDD | ~1,160 | 1 (ship) | PASCAL-VOC XML | pilot |
| 2 | `hrsid` | HRSID | ~5,604 | 1 (ship) | COCO JSON | pilot |
| 3 | `srsdd` | SRSDD-v1.0 | ~1,022 crops | 6 (fine-grained) | DOTA (oriented) | benchmark |
| 4 | `sar_ship` | SAR-Ship-Dataset | ~43,819 | 1 (ship) | DOTA (oriented) | benchmark |
| 5 | `sardet100k` | SARDet-100K | ~116,598 | 6 | COCO JSON | benchmark |

---

## 1. SSDD — SAR Ship Detection Dataset

* **Source:** <https://github.com/TianwenZhang0825/Official-SSDD>
* **Cite:** Zhang, T. et al., *Multi-Scale Context Aggregation for SAR Ship
  Detection*, 2019.
* **Format:** PASCAL-VOC XML, one file per image.
* **Why first:** ~1.2k images and ~2.5k ships. The full ablation grid
  (EXP-001…008) plus the module-level ablations finishes on a free Colab T4 in
  hours, which is what makes it the right place to debug.

**Use the official split.** SSDD ships split list files (e.g.
`ImageSets/Main/train.txt`). Pass them rather than inventing a split, so results
stay comparable with the published baselines:

```python
from saryolo.data import prepare_dataset

prepare_dataset(
    "ssdd",
    raw_dir="/content/raw/SSDD",
    out_root="datasets/processed",
    official_split_files={
        "train": "/content/raw/SSDD/ImageSets/Main/train.txt",
        "val":   "/content/raw/SSDD/ImageSets/Main/val.txt",
        "test":  "/content/raw/SSDD/ImageSets/Main/test.txt",
    },
)
```

Images listed in no split file are **skipped and counted** (reported as
`unlisted_images`), never silently dumped into `train`.

---

## 2. HRSID — High-Resolution SAR Images Dataset

* **Source:** <https://github.com/chaozhong2010/HRSID>
* **Cite:** Wei, S. et al., *HRSID: A High-Resolution SAR Images Dataset for Ship
  Detection and Instance Segmentation*, IEEE Access, 2020.
* **Format:** COCO JSON.
* **Note:** 5,604 images from **99 parent scenes** across Sentinel-1, TerraSAR-X
  and Gaofen-3. Recommended input size 800.
* **The 400 background images are a free robustness probe.** The authors filtered
  out 400 crops of pure background and ship them separately. They contain no
  positives, so they are useless for the main table — but they are exactly what
  the false-positive and clutter arm of the robustness sweep (EXP-009) needs,
  since every detection on them is by definition a false alarm. Keep them out of
  training and out of the main test split.

> ⚠️ **Leakage warning — read this before reporting any HRSID number.**
> The chips are crops of a small number of large acquisitions, so a random
> chip-level split places near-identical patches in train and test and inflates
> mAP substantially. Split by scene, and verify:

```python
from saryolo.data.splits import leakage_report

report = leakage_report(splits, image_dir="datasets/processed/hrsid/images/train")
print(report.summary())   # lists exact and near-duplicate pairs that straddle splits
```

---

## 3. SRSDD-v1.0 — SAR Rotation Ship Detection Dataset

* **Source:** <https://github.com/HeuristicLU/SRSDD-V1.0>
* **Cite:** Lei, S. et al., *SRSDD-v1.0: A High-Resolution SAR Rotation Ship
  Detection Dataset*, Remote Sensing 13(24), 2021.
* **Format:** DOTA-style oriented boxes, 1024×1024 crops recommended.
* **Contents:** **2,884 instances across six fine-grained ship categories** —
  ore-oil, bulk-cargo, fishing, law-enforcement, dredger, container — cut from
  **30 panoramic Gaofen-3 port tiles** at 1 m resolution.
* **Use for:** the oriented-detection study (Component 6), and fine-grained
  recognition. It is the only supported dataset carrying *both* orientation and a
  fine-grained taxonomy, so it answers two questions that single-class ship
  benchmarks cannot: "does orientation prediction help?" and "does the attention
  slot help distinguish *which kind* of ship?".

> **Two corrections to commonly repeated numbers.** The dataset is frequently
> cited as having **seven** categories; the paper states **six**. And the official
> release is **30 panoramic tiles**, while the widely circulated processed form is
> ~1,022 square crops — record which one you actually used, because they are not
> the same benchmark. Verify both counts against the official release before
> quoting them in a paper.

---

## 4. SAR-Ship-Dataset

* **Source:** <https://github.com/CAESAR-Radi/SAR-Ship-Dataset>
* **Cite:** Wang, Y. et al., *SAR-Ship-Dataset: A Large-Scale SAR Ship Dataset
  for Deep-Learning-Based Ship Detection*, Remote Sensing, 2019.
* **Format:** DOTA-style oriented boxes; ~44k chips from Sentinel-1 and Gaofen-3.
* **Use for:** scale. It carries orientation too, but with a single class and
  chips with no scene grouping, so it is the volume/robustness benchmark rather
  than the oriented-detection one — prefer SRSDD-v1.0 for Component 6, where the
  orientation study is paired with a fine-grained label space.
* **Leakage warning:** like HRSID, the chips derive from a limited number of
  acquisitions. Run the leakage check before trusting a random split.

```python
from saryolo.data import prepare_dataset
prepare_dataset("sar_ship", raw_dir="/content/raw/SAR-Ship-Dataset")
# writes YOLO-OBB labels (8 normalised corner coordinates)
```

---

## 5. SARDet-100K — the headline benchmark

* **Source:** <https://github.com/zcablii/SARDet_100K>
* **Cite:** Li, Y. et al., *SARDet-100K: Towards Open-Source Benchmark and
  Toolkit for Large-Scale SAR Object Detection*, NeurIPS 2024.
* **Format:** COCO JSON. ~116k images, ~245k instances, 6 classes
  (`ship`, `aircraft`, `car`, `tank`, `bridge`, `harbor`), unified from 10 source
  datasets.
* **Use for:** the paper's main comparison table.

**Compute reality check.** This is roughly 100× SSDD in image count. A free Colab
session will not complete a 100-epoch run. Plan for either multi-session training
with checkpoint resume (supported by Ultralytics via `resume=True`) or Colab Pro.
Do the entire module-development cycle on SSDD/HRSID first, then run only the
final chosen configurations at scale.

The six classes are severely imbalanced (cars dominate), so report per-class AP
as well as mAP. `saryolo.data.statistics` reports the imbalance ratio explicitly.

### Acquisition profiles: the conditioning bridge (verified 2026-09-23)

SARDet-100K publishes a per-source table of target, resolution, band, polarization
and satellites — the exact signal the acquisition-conditioning arms and the
leave-one-source-out protocol need.
`saryolo.data.convert.SARDet_SOURCE_PROFILES` records it (AIR_SARShip, HRSID, MSAR,
SADD, SAR-AIRcraft, ShipDataset, SSDD, OGSOD, SIVED; the paper's release notes list
`SAR-Ship-Dataset` as `ShipDataset`), and
`saryolo.data.convert.write_acquisition_metadata(images_dir, out, dataset="sardet100k")`
turns it into a `MetadataTable` JSON keyed by image stem, ready for
`loso --rule resolution --metadata` and for the conditioning trainer.

Two things the profiles deliberately do *not* do: they do not invent values
(every entry is sourced from the official table; ranges are carried as mid-point
plus the published range), and they do not guess silently — a stem matching no
source is counted and reported, a dataset without a verified profile is refused,
and all-zero matching raises. HRSID standalone gets the same treatment with its
three stated resolutions (0.5 / 1 / 3 m) and its Sentinel-1B / TerraSAR-X /
TanDEM-X scene list; per-image per-polarization labels are not published, so
`polarization` stays unknown rather than guessed.

The cs231n mirror (<https://github.com/DonnieRaymond3/cs231n_ship_detection>)
vendors both HRSID (`HRSID/HRSID_JPG/JPEGImages/` + COCO JSON: 5,604 images,
train 3,642 / test 1,962) and SSDD (1,160 images; four label variants — use
`BBox_SSDD` for horizontal-box detectors) with on-disk verification counts, which
makes it the fastest licensed route to both pilot datasets. Its HRSID copy is the
same official release, so all HRSID handling above applies unchanged; SSDD's
inshore/offshore test subsets (46/186) are the intended cross-scenery probe and
map onto our scene-grouped splits.

---

## Preparation workflow

For any dataset:

```bash
# 1. Convert to YOLO format (symlinks by default, so 100k+ images cost no extra disk)
python scripts/prepare_dataset.py --dataset ssdd --raw /content/raw/SSDD

# 2. Validate: corrupt files, bad class ids, out-of-range and zero-area boxes,
#    orphan labels, exact and near-duplicates
python -m saryolo check-data --dataset datasets/processed/ssdd --classes ship

# 3. Profile: class distribution, object size histogram, aspect ratios, objects
#    per image, density heat map, and SAR difficulty proxies (local contrast, SNR)
python -m saryolo stats --dataset datasets/processed/ssdd --classes ship --name ssdd
```

Outputs land in `validation_reports/` and `dataset_statistics/` (JSON + figures).

### Why validate before training

The statistics report is what tells you whether the architecture's assumptions
hold. Two examples that change the plan:

* **If `size_distribution` is dominated by `small`**, the P2 detection head
  (Component 5) is justified, and `AP_small` becomes the metric that matters.
* **If `target_background_ratio` is near 1.0**, targets barely differ from
  background in intensity, which is the regime where the speckle-aware module
  should show the clearest benefit — and where a contrast-only baseline will
  fail.

Running `stats` first is therefore not bookkeeping; it is what makes the module
decisions evidence-driven rather than assumed.

---

## Layout after preparation

```
datasets/
├── raw/                       # untouched downloads (gitignored)
├── processed/<dataset>/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/   # YOLO: class cx cy w h (normalised)
│   └── README_SYNTHETIC.md        # only for generated smoke data
└── splits/<dataset>/{train,val,test}.txt
```

Converters never write into `raw/`, so a conversion bug is always recoverable.
