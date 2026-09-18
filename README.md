# SAR-YOLO — Speckle-Aware Object Detection for SAR Imagery

Research code for **SAR-YOLO**: a SAR-specific object detector built by adding
*scientifically motivated, individually ablatable* modules to a YOLO baseline,
together with a full reproducible pipeline from dataset validation to a
CVPR-style paper.

> **Status: infrastructure complete and verified; no real-dataset results yet.**
> Every result table in this repository is empty and renders as `TBD` on purpose.
> The pipeline has been proven end to end on **synthetic** data and on CPU; the
> real numbers require running the notebooks on a GPU against a downloaded
> benchmark. Nothing here is a published or claimed result.

---

## Why this repository is shaped the way it is

The brief that motivated this project was explicit about the failure mode to
avoid: `YOLO + CBAM + SE + Transformer + BiFPN = "new model"`. So the design
rules here are:

1. **Add one module at a time, and prove each earns its place.** The experiment
   matrix (`EXP-002` … `EXP-007`) adds exactly one component per run.
2. **Every module is an exact identity at initialisation.** A freshly built
   SAR-YOLO produces *numerically identical* predictions to its YOLO baseline
   (verified by `tests/test_arch.py::test_models_output_identically_to_baseline_at_init`).
   Any improvement is therefore learned behaviour, not extra capacity at init.
3. **Slot-matched baselines.** The proposed attention and fusion blocks are
   compared against SE / ECA / CBAM and concat / projected-add / static-weighted
   fusion **in the same architectural slot**, so the comparison isolates the
   mechanism rather than the extra parameters.
4. **Adaptivity is isolated explicitly.** `att_saa_static` and `fus_static` are
   the proposed blocks with their input-dependent weighting replaced by learned
   constants. Those two runs are what justify calling the blocks *adaptive*.
5. **No fabricated numbers, anywhere.** Missing measurements stay `None` and
   render as `TBD`; there is no code path that invents a metric.

---

## Architecture

```
SAR image
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│ YOLO11 backbone                                                  │
│   P2/4 ──► Component 1: SAR Feature Enhancement (SFE)            │
│   P3/8, P4/16, P5/32, SPPF/C2PSA                                 │
│   P5/32 ──► Component 2: Speckle-Aware Feature Module (SFM)      │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│ PAN-FPN neck                                                     │
│   after every Concat ──► Component 4: Adaptive Multi-Scale       │
│                          Fusion (AMF)                            │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│ per-level ──► Component 3: SAR-Adaptive Attention (SAA)          │
│               → Detect(P2, P3, P4, P5)   Component 5: P2 head    │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
Component 7: SAR-aware loss (target/background separation,
             small-object emphasis, background speckle regularisation)
```

Component 6 (oriented detection) is deliberately **not** implemented: it is only
justified if the annotations carry meaningful orientation, which is true for
`SAR-Ship-Dataset` (DOTA-format) but not for SSDD/HRSID/SARDet-100K. The
converter for it exists; the head does not, and will only be added if that
experiment is actually run.

The mathematics of each component, and the reasoning behind each design choice,
are in [`docs/METHOD.md`](docs/METHOD.md).

---

## Verified in this environment

These are checks that were actually run, not aspirations:

| What | Result |
| --- | --- |
| Baseline reproduces stock YOLO11 **exactly** | `2,624,080` params (n) and `9,458,752` (s) — matches Ultralytics' published counts |
| All 26 architecture variants build and run a forward pass | pass |
| Every SAR module is an **exact** identity at init | pass (`torch.equal` on the output) |
| SAR-YOLO predicts identically to baseline with baseline weights | max abs diff `0.0` |
| SAR-aware loss is wired into training | `train/sar_loss` and `val/sar_loss` columns present with non-zero values |
| Baseline + full SAR-YOLO train end to end on CPU | both complete; checkpoints and ledger rows written |
| Test suite | **46 passed** |

Reproduce all of it locally, with no dataset download and no GPU:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python ultralytics pytest
source .venv/bin/activate

pytest tests/ -q                                          # 46 tests
python -m saryolo arch --variant all --nc 1               # emit model YAMLs
python -m saryolo synth-data --out datasets/processed/synthetic_smoke
python -m saryolo train --exp configs/exp/_smoke_baseline.yaml
python -m saryolo train --exp configs/exp/_smoke_full.yaml
python -m saryolo ledger
```

> The `_smoke_*.yaml` configs use **synthetic** Gamma-speckle data. They exist to
> catch wiring bugs in seconds instead of after an hour of GPU time. Any number
> they produce is meaningless as a research result.

---

## Running the real thing (GPU / Colab)

```bash
# 1. Get a dataset onto the machine (see docs/DATASETS.md for licensed routes)
python scripts/prepare_dataset.py --dataset ssdd --raw /content/raw/SSDD

# 2. Validate and profile it before spending GPU time
python -m saryolo check-data --dataset datasets/processed/ssdd --classes ship
python -m saryolo stats      --dataset datasets/processed/ssdd --classes ship --name ssdd

# 3. Baseline first, then one module at a time
python -m saryolo train --exp configs/exp/EXP-001_baseline.yaml
python -m saryolo train --exp configs/exp/EXP-002_sfe.yaml

# 4. Or walk the whole matrix
python scripts/train_all_experiments.py --keep-going
```

On Colab use the notebooks, which handle the dataset download, GPU selection and
checkpoint resume:

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_dataset_prep.ipynb` | Download, convert, validate, profile and leakage-check a dataset |
| `notebooks/02_train_and_ablate.ipynb` | Train the chain EXP-001…EXP-008 and generate the ablation tables |
| `notebooks/03_benchmark_and_paper.ipynb` | Robustness (EXP-009), efficiency (EXP-010), cross-dataset (EXP-011), multi-seed (EXP-012), figures |

---

## Dataset plan (cheapest first)

Following the project rule *understand the data before building the model*, the
datasets are worked through in increasing cost:

| Order | Dataset | Images | Classes | Tier | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | **SSDD** | ~1.2k | 1 | pilot | Full ablation grid fits in a free Colab session |
| 2 | **HRSID** | ~5.6k | 1 | pilot | Harder scenes; tests the small-object claims |
| 3 | **SAR-Ship-Dataset** | ~44k | 1 | benchmark | Oriented annotations, if Component 6 is pursued |
| 4 | **SARDet-100K** | ~117k | 6 | benchmark | The current large-scale SAR benchmark for final numbers |

Details, licences, citations, leakage warnings and download routes are in
[`docs/DATASETS.md`](docs/DATASETS.md). **No dataset is redistributed here** —
the MIT licence covers the code only.

---

## Repository map

```
saryolo/
├── nn/
│   ├── arch.py            symbolic architecture builder (all variants + ablations)
│   ├── modules/
│   │   ├── enhancement.py SFE   (Component 1) + log/standardize/CLAHE baselines
│   │   ├── speckle.py     SFM   (Component 2) + Lee/low-pass baselines
│   │   ├── attention.py   SAA   (Component 3) + SE/ECA/CBAM baselines
│   │   └── fusion.py      AMF   (Component 4) + concat/add/static baselines
│   ├── losses.py          SAR-aware loss (Component 7)
│   ├── model.py           DetectionModel with the SAR criterion
│   └── register.py        publishes custom layers to ultralytics
├── data/                  registry, converters, validator, statistics, splits
├── training/              trainer, experiment runner, config loader
├── evaluation/            COCO AP + scale-wise AP, robustness, efficiency, domain shift
├── visualization/         detections, Grad-CAM, feature maps, failure taxonomy
├── tracking/              append-only experiment ledger + environment capture
├── paper/                 LaTeX + table/figure generators
└── cli.py                 `python -m saryolo <command>`

configs/
├── datasets/    ssdd.yaml, hrsid.yaml, sardet100k.yaml, synthetic_smoke.yaml
├── models/      35 generated YAMLs (baseline, EXP-00x chain, module ablations, all scales)
└── exp/         EXP-001…012 + `_smoke_*` pipeline tests

scripts/         prepare_dataset, make_exp_configs, train_all_experiments
notebooks/       Colab notebooks
tests/           arch parity, identity-at-init, metrics, losses, data
docs/            DATASETS.md, METHOD.md
```

Model YAMLs and experiment configs are **generated**, not hand-maintained:

```bash
python -m saryolo arch --variant all --nc 1        # configs/models/
python scripts/make_exp_configs.py --dataset ssdd  # configs/exp/
```

---

## Experiment matrix

| ID | Run | Isolates |
| --- | --- | --- |
| EXP-001 | YOLO baseline | reference point |
| EXP-002 | + SFE | Component 1 |
| EXP-003 | + SFM | Component 2 |
| EXP-004 | + SAA | Component 3 |
| EXP-005 | + AMF | Component 4 |
| EXP-006 | + P2 head | Component 5 |
| EXP-007 | **FULL SAR-YOLO** | all + Component 7 |
| EXP-008 | FULL without SAR loss | isolates the loss from the architecture |
| EXP-009 | Robustness sweep | reuses EXP-007 checkpoint |
| EXP-010 | Efficiency benchmark | reuses EXP-007 checkpoint |
| EXP-011 | Cross-dataset | reuses EXP-007 checkpoint |
| EXP-012 | Multi-seed (0, 1, 2) | mean ± std |

Plus module-level ablations (`att_se`, `att_eca`, `att_cbam`, `att_saa_static`,
`fus_concat`, `fus_add`, `fus_static`, `spk_lee`, `spk_denoise`, `pre_log`,
`pre_clahe`, `pre_standardize`) so each proposed block is compared against the
standard component it replaces, in the same slot.

### Results

Empty by design — fill by running the matrix. The generator renders unmeasured
cells as `TBD` and **will refuse to emit a number that was never measured**.

| Model | mAP50 | mAP50:95 | Params (M) | GFLOPs | FPS |
| --- | --- | --- | --- | --- | --- |
| YOLO11 baseline | TBD | TBD | TBD | TBD | TBD |
| SAR-YOLO | TBD | TBD | TBD | TBD | TBD |

---

## Reproducibility

Every run writes a row to `results/experiments.jsonl` (and a derived
`experiments.csv`) recording the experiment id, config hash, seed, epochs,
batch, image size, optimizer, learning rate, weights path, git commit, Python /
PyTorch / CUDA / Ultralytics versions, GPU name, and the measured metrics.

```bash
python -m saryolo ledger
```

The ledger is append-only: a rerun never overwrites an earlier result, and
failed runs are retained (with their error) so a table can never silently
include a run that did not finish.

---

## Known limitations, stated up front

* **The adaptive-vs-static comparisons are the load-bearing experiments.** If
  `att_saa_static` matches `att_saa`, the adaptivity claim is not supported and
  the paper must say so.
* **HRSID and SAR-Ship-Dataset leak under random chip splits** (chips are cut
  from a few large scenes). `saryolo.data.splits.leakage_report` exists to catch
  this; a scene-grouped split is required before trusting mAP on those datasets.
* **Scale-wise AP here is our own implementation**, with the deviations from
  pycocotools documented in the `saryolo.evaluation.metrics` module docstring.
* **Cross-dataset evaluation refuses to run on incompatible label spaces**,
  rather than reporting a meaningless low mAP.
* **Oriented detection (Component 6) is unimplemented.** The DOTA converter
  exists; the head does not.

## License

MIT for the code (see `LICENSE`). Datasets are **not** covered and must be
obtained from their official sources under their own terms.

## Citation

See `CITATION.cff`. Please cite the original dataset papers as well — the
required citations are listed per dataset in `docs/DATASETS.md`.
