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
| 3 | `sar_ship` | SAR-Ship-Dataset | ~43,819 | 1 (ship) | DOTA (oriented) | benchmark |
| 4 | `sardet100k` | SARDet-100K | ~116,598 | 6 | COCO JSON | benchmark |

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

## 3. SAR-Ship-Dataset

* **Source:** <https://github.com/CAESAR-Radi/SAR-Ship-Dataset>
* **Cite:** Wang, Y. et al., *SAR-Ship-Dataset: A Large-Scale SAR Ship Dataset
  for Deep-Learning-Based Ship Detection*, Remote Sensing, 2019.
* **Format:** DOTA-style oriented boxes; ~44k chips from Sentinel-1 and Gaofen-3.
* **Use for:** the optional oriented-detection study (Component 6). These are the
  only supported annotations that carry meaningful orientation, so they are the
  only dataset on which "does orientation prediction help?" is a real question.

```python
from saryolo.data import prepare_dataset
prepare_dataset("sar_ship", raw_dir="/content/raw/SAR-Ship-Dataset")
# writes YOLO-OBB labels (8 normalised corner coordinates)
```

---

## 4. SARDet-100K — the headline benchmark

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
