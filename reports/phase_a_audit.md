# Phase A audit — repository state, data flow, and the verified blocker

**Date:** 2026-10-12
**Commit audited:** `dc623dd5592e503a41c50c11b2d1087c3488ebd1` (`main` == `origin/main`, working tree clean, no tags)
**Method:** read-only inspection. No model, data, configuration, checkpoint or test file was
modified while producing this document. Every statement below is either a command whose output
is quoted, or a property of a named file. Where a number appears without a command, the command
that produced it is the one immediately above it.

**Scope of the review.** This document answers the master directive's Phase A (repository audit)
together with the "FIRST TASK" in section 15: map the model and data pipeline, determine the most
important verified blocker, and return a prioritized implementation sequence. It also states, in
§7, what "establish a reproducible baseline" concretely means for this repository, because the
override forbids touching model code before that plan is written down.

---

## 1. Repository state

```console
$ git log --oneline -1 && git status --short && git branch -vv
dc623dd Time the dense/sparse sweep in one process, and stop the arch command editing the variant registry
                                          # (no output: nothing modified, nothing untracked-but-changed)
* main dc623dd [origin/main] Time the dense/sparse sweep ...
```

```console
$ git log --format='%an <%ae> | %cn <%ce>' | sort | uniq -c
     76 Arghya Bose <officialarghya29@gmail.com> | Arghya Bose <officialarghya29@gmail.com>

$ git grep -lIiE 'codebuff|freebuff|manicode|Co-Authored-By: (Claude|GPT|Copilot)' -- .
                                          # (no output: no tracked file names an AI tool)
$ git grep -nE 'TODO|FIXME|XXX|HACK' -- '*.py'
                                          # (no output: no unresolved markers)
```

Every commit in the history is authored **and** committed under the repository owner's identity;
there is no co-author trailer and no tracked file that names an AI coding tool. That property is
itself guarded (`tests/test_repo.py::test_working_files_never_name_an_ai_coding_tool`).

**Tracked content.** 481 files: 62 under `saryolo/`, 56 test modules under `tests/`, 258 under
`configs/`, 42 under `docs/`, 25 under `paper/`, 14 under `reports/`, 9 under `scripts/`.

**Environment.** `.venv`, CPython 3.12.3, torch 2.14.0+cpu, ultralytics 8.4.155, numpy 2.5.2,
streamlit 1.65.0. CPU-only host: AMD Ryzen 7 8840HS (16 threads), 30 GB RAM, **no CUDA device**.

**Suite, re-run at the audited commit** (not quoted from a document):

```console
$ .venv/bin/python -m pytest -q
977 passed in 164.35s (0:02:44)          # exit status 0
```

This matches the local count the README badge and `reports/reproduction_status.md` claim. The
count is guarded in four README places plus the status documents, so it cannot drift silently.

---

## 2. Inventory: what exists

| Layer | Count | Where |
| --- | ---: | --- |
| Registered architecture variants | **110** | `saryolo/nn/arch.py` (`VARIANTS`) |
| Generated model YAMLs (1:1 with the registry) | **110** | `configs/models/*.yaml` |
| Experiment configs | 138 | `configs/exp/*.yaml` |
| Dataset configs | 9 | `configs/datasets/*.yaml` |
| CLI subcommands | 20 | `saryolo/cli.py` |
| Test modules | 56 | `tests/` |
| Defects found and fixed, with regression tests | 29 (D-01…D-29) | `defect_log.md` |
| Committed checkpoints | **1** | `weights/sarvo_ssac001.pt` |

```console
$ sha256sum weights/sarvo_ssac001.pt results/runs/SSAC-001/weights/best.pt
c81287334beba49c27b13682fe1120ab5c025743dc1a8fe41c5c7f13f45a98a5  weights/sarvo_ssac001.pt
c81287334beba49c27b13682fe1120ab5c025743dc1a8fe41c5c7f13f45a98a5  results/runs/SSAC-001/weights/best.pt
```

The committed demo checkpoint is byte-identical to the pilot run it claims to be, 7,096,137 bytes,
3,396,329 parameters.

**Module usage.** No dead module was found: each of the 62 source modules is imported from outside
itself by at least one of `saryolo/`, `tests/`, `scripts/` or `app.py` (checked by grepping each
module's dotted path). There is therefore no "unused module" cleanup to do, and none should be
attempted — the override forbids restructuring for cleanliness alone.

---

## 3. The data flow, traced through the code

```
datasets/processed/hrsid_real/images/{train,val,test}/*.jpg     (800x800 chips, 1 class: ship)
        │
        │  saryolo/data/validate.py::validate_yolo_dataset   (readability, malformed rows,
        │                                                      empty labels, md5 + dHash)
        │  saryolo/data/splits.py::{split_files, leakage_report, read_official_split}
        │  saryolo/data/yolo.py::{load_data_config, resolve_data_yaml, split_dirs}
        ▼
Ultralytics Datasets → augmentation (saryolo/augmentation/sar.py; saryolo/data/synth.py for the
synthetic fixture) → Trainer (saryolo/training/trainer.py, subclass of DetectionTrainer)
        │  metadata join, when a conditioned variant is trained:
        │  SARYOLOTrainer.build_dataset → _attach_metadata → AcquisitionMetadataDetectionDataset
        ▼
model: saryolo/nn/model.py::SARYOLODetectionModel  (chosen by is_saryolo_yaml from the layer
       names in saryolo/nn/modules/__init__.py::CUSTOM_MODULES)   ← D-01's fix
        │  forward → criterion (saryolo/nn/losses.py::build_criterion, `sar_loss` block in the
        │  model YAML) → loss() adds the consistency view and the allocation-sparsity penalty
        ▼
checkpoint (results/runs/<ID>/weights/best.pt)  +  append-only row in results/experiments.jsonl
        │
        │  saryolo/inference.py::{validate_and_decode, load_detector, predict_image, annotate_image}
        │  saryolo/evaluation/metrics.py::{load_yolo_ground_truth, compute_ap, evaluate_detections}
        ▼
NMS/post-processing inside Ultralytics' Detect head → COCO-style AP at IoU 0.5 and 0.5:0.95,
per-class AP, scale-wise AP (AP_small where supported) → results/experiments.jsonl `metrics`
        │
        ▼
docs/assets/facts.json (generated by scripts/make_readme_assets.py) → README tables, charts,
paper/tables/*.{md,tex} (via `python -m saryolo assets`)
```

Two properties of this pipeline are worth stating explicitly because they are *verified*, not
assumed:

* **The evaluation code never reads the test split.** Grepping the training path for the string
  `test` finds it only in `data/convert.py` and `data/splits.py`, i.e. in dataset assembly, never
  in the trainer, the criterion or the model selection. Test-set usage for training, early
  stopping or threshold tuning would have to be added deliberately.
* **The loss refuses a non-finite value** rather than continuing in an invalid numerical state
  (`SARYOLODetectionModel.loss`, `tests/test_reliability.py`), and the inference path refuses a
  non-finite detection (`tests/test_inference.py`). Both are D-08 / D-11.

---

## 4. Implemented versus proposed

| Item | State | Evidence |
| --- | --- | --- |
| Stock-YOLO11 baseline parity | **Implemented + asserted** | `tests/test_arch.py` pins the published parameter counts (n: 2,624,080; s: 9,458,752) |
| v1 module ladder (SFE, SFM, SAA, AMF, P2) | **Built, never measured on real data** | 110 variants build and forward; only `v2_lite_s` has a real-data run (REAL-002) |
| v2 ladder (clutter SFM, target prior, spectral branch, context, deformable refinement) | **Built, never measured** | registered as EXP-013…017; no ledger row |
| Acquisition conditioning (7 adapter/field-set arms) | **Built, never measured** | metadata join exists and is tested; no conditioned run |
| LoRA parameter-efficient arm | **Built, measured once** | REAL-003, rank 8, one seed |
| Ratio-space CFAR front end | **Implemented + measured at pilot scale** | REAL-004/005/006 + two extra seeds |
| SAR appearance augmentation | **Implemented + measured, 3 seeds** | AUG-001…006; the largest accuracy movement in the record |
| **SSAC (the core mechanism)** | **Implemented, measured, accuracy claim withdrawn** | SSAC-001…010; sparse execution measured at 320 px and 640 px |
| RT-DETR arm | **Feasibility only** | builds and shape-tests; no trained result |
| Cross-sensor / cross-resolution (LOSO, resolution bins) | **Tooling implemented, never run on real data** | `saryolo loso`; HRSID ships no verified sensor sidecar |
| Visualisation, probes, gain diagnostic | **Implemented, used on the pilot** | `saryolo/{visualization,evaluation/probes,evaluation/gain}` |

---

## 5. Verified findings

Each row names the artefact and the command. Severity follows `defect_log.md`'s scale: **Critical**
silently invalidates results, **High** produces a wrong number, **Medium** hides a failure mode or
a guard that does not guard, **Low** is documentation/usability drift.

### F-01 · the pilot's split is scene-contaminated — **High** (methodology)

HRSID chips carry their scene in the filename (`P0001_0_800_8400_9200` = scene `P0001`, window
rows 0–800, columns 8400–9200). Counting the prefix per split:

```console
train scenes: P0001 P0002 P0003 P0004 P0005 P0006
val   scenes: P0001 P0002 P0003 P0004 P0005 P0006 P0007 P0008 P0009 P0010
test  scenes: P0001 P0002 P0003 P0004
val : 37/60 chips come from a scene that is also in train (62%)
test: 60/60 chips come from a scene that is also in train (100%)
total distinct scenes in the subset: 10
```

**Every val chip and every test chip comes from a scene the model has also trained on**, because
`scripts/fetch_hrsid_subset.py` copies the *first N files in sorted filename order* per split
(`sorted(...)`, then `break` when the count reaches `--train/--val/--test`), not a random sample
and not a scene-grouped one. The pilot's numbers are therefore **within-scene** numbers. They are
not comparable with a generalisation number, and the held-out test split — the one reserved for
final evaluation — shares all 4 of its scenes with training, so evaluating it would not change
that.

Also measured: the duplicate audit that `docs/RUNBOOK_HRSID.md` calls mandatory had never been
run for this dataset, and no split manifest is stored (`datasets/splits/` did not exist).

```console
$ python -c "from saryolo.data.splits import leakage_report; ..."
Leakage report
  files: 320   splits: {'train': 200, 'val': 60, 'test': 60}
  exact duplicate pairs across splits: 0
  near-duplicate pairs across splits:  1
near: [('train:P0003_0_800_7200_8000', 'val:P0001_0_800_7200_8000')]
```

So the pixel-level duplicate risk is **one** near-duplicate pair — small — while the *scene*-level
contamination is 62 %/100 %. The two are different problems and the second is the serious one.

**Fixable on CPU in minutes**, and the machinery already exists and works. Running the repository's
own leave-one-source-out generator over the six scenes that are large enough:

```console
$ python -m saryolo loso --images datasets/processed/hrsid_real/images/train \
    --data configs/datasets/hrsid_real.yaml --rule regex --pattern '^(P[0-9]+)' \
    --val-ratio 0.25 --min-test-images 20 --out datasets/splits/loso_hrsid_scene
  6 source(s): P0001 45, P0002 29, P0003 38, P0004 27, P0005 30, P0006 31
  held-out source                train     val    test
  P0001                            116      39      45
  P0002                            128      43      29
  P0003                            122      40      38
  P0004                            130      43      27
  P0005                            128      42      30
  P0006                            127      42      31
```

Each fold's test chips are scene-disjoint from its training chips, so a number from
`test.txt` is a cross-scene number. Two honest limits, both measured: the protocol reaches
**6 scenes and ≤45 test chips per fold**, and pointing the same command at all 10 scenes is
refused by its own guard because four scenes hold only 2–9 chips
(`P0007 6, P0008 2, P0009 9, P0010 6`).

### F-02 · the published pilot metrics are validation metrics, and the documents do not say so — **Medium** (reporting)

```console
$ grep -E '^(val|split):' results/runs/SSAC-001/args.yaml
val: true
split: val
```

The pilot's mAP comes from the trainer's end-of-run validation on the 60-image val split. No arm
has ever been evaluated on the test split. Reporting the number as "on a real HRSID subset" is not
false, but it does not identify the split, and the two readings differ by the whole of F-01 (62 %
vs 100 % scene overlap). Every table in `README.md`, `paper/RESULTS.md` §3 and
`reproduction_status.md` §3 needs the split named.

### F-03 · the manuscript's Method and the measured evidence describe different models — **High** (paper)

```console
$ wc -l paper/manuscript/main.tex            # 436
$ grep -c -i 'ssac' paper/manuscript/main.tex # 0
$ grep -c -i 'cfar' paper/manuscript/main.tex # 2   (both in Related Work / the abstract teaser)
```

The manuscript's §3 Method is the **v1/v2 module ladder** (SFE, SFM, SAA, AMF, P2 head, target
prior, spectral branch, context, refinement, SAR loss) — the half of the project with **no
real-data measurement at all** — while the entire measured record, the committed checkpoint, the
demo and the efficiency study are the **CFAR front end and SSAC**, which the manuscript never
names. `app.py` and `README.md` call SSAC "the core mechanism". A reviewer comparing the paper to
the repository would find the paper describing one model and the artifacts another.

Related and equally visible: **99 benchmark cells across `paper/tables/*.md` are still `TBD`**
(`module_ablation` 51, `main_ablation` 13, `efficiency_frontier` 13, `removal_ablation` 8, …).
That is the correct state for unmeasured arms, but it means the paper currently has no main table.

### F-04 · the efficiency sweep was measured at a resolution the checkpoint was never trained at — **Medium** (claim scope)

`SSAC-001` was trained at `imgsz: 320` (132 px… see F-06) and the saving appears only at 640 px
(`README.md`, `paper/RESULTS.md` §8.5: dense 196.216 ms, `keep 0.1` 179.628 ms, **−8.5 %**, ranges
non-overlapping). The documents say "one checkpoint, CPU only" but do not say that the setting the
saving depends on is also a setting no arm has ever been trained at — which matters because the
model's own accuracy at 640 px is unmeasured, so the cheaper operating point has no measured
accuracy attached to it.

### F-05 · `saryolo/nn/arch.py::_main()` still writes the requested class count into the shared registry — **Low** (latent, same class as D-29)

```python
# saryolo/nn/arch.py, _main()
spec = VARIANTS[name]
spec.nc = args.nc                     # mutates the module-level spec
```

`_cmd_arch` in `saryolo/cli.py` was fixed for exactly this in D-29
(`dataclasses.replace(arch.VARIANTS[name], nc=args.nc)`), but the module-level entry point keeps
the old line, so `python -m saryolo.nn.arch --variant all --nc 2` reproduces D-29 from a different
door. It is not reachable through the `saryolo` CLI and no test drives it, which is why the guard
that caught D-29 does not catch this.

### F-06 · the object regime is small-object, and the pilot trains at 0.4× native scale — **High** (design input)

Measured from the 431 training boxes of the real subset (native chips are 800×800):

```console
native 800 px box max-side:  p10 = 16.0   p50 = 43.0   p90 = 65.0
at imgsz 320 (x0.4):         p10 =  6.4   p50 = 17.2   p90 = 26.0
median box area at imgsz 320: 171 px^2   (~13x13 px)
sampled image shapes: {(800, 800): 40}
```

The pilot trains at 320 px on 800 px chips, so the median ship becomes a **17 px** object and the
committed checkpoint is a small-object detector operating at a scale where the object is a handful
of pixels. Three consequences, each actionable: (i) the achieved AP_small of 0.0737 is what this
regime produces, not a tuning accident; (ii) **resolution is the highest-leverage measurable
axis** on this host — 640 px doubles the median object to ~34 px and is exactly the setting where
the one positive efficiency result was measured, so one experiment addresses F-04 and F-06
together; (iii) any claim about "small-object detection" must state the training resolution
alongside the native one.

### F-07 · the evidence base is one dataset, one scale, one resolution, one seed pair — **High** (power)

From `results/experiments.jsonl` (77 rows, **30 completed runs**):

```console
$ python -c "…print completed runs by id/model/epochs/imgsz…"
SMOKE-001…004   synthetic_smoke (2 epochs, 128 px; pipeline proof, not results)
REAL-001…006    40 epochs, 320 px, hrsid_real      (stock 11n, v2_lite_s, LoRA r8, CFAR, fixed, conv)
MSEED-B1/C1/B2/C2, SSAC-001…010, AUG-001…006 — all 40 epochs, 320 px, hrsid_real
```

26 real runs, all on the same 200/60/60 subset, all at 320 px, one architecture scale except
REAL-002 (`s`). The effects under test are of the same order as the seed spread the pilot itself
measured (SSAC mean +0.0220 with **±0.0373** across three seeds), which is why the SSAC accuracy
claim was withdrawn — correctly. A wider claim cannot be earned by re-analysing these runs; it
needs more data, more seeds, or a different resolution.

Where the compute actually goes, measured from the run logs:

```console
$ python -c "…median per-epoch seconds from results/runs/*/results.csv…"
REAL-001 (11n, 200 train + 60 val, 320 px): 9.9 s/epoch   40 epochs = 0.11 h
SSAC-001 (11n+SSAC, same):                 10.8 s/epoch   40 epochs = 0.12 h
REAL-004 (11n+CFAR, same):                 18.3 s/epoch   40 epochs = 0.21 h
```

### F-08 · the "full-release training needs a GPU" framing is only partly true — **Medium** (feasibility)

The documents repeat that the full benchmark is `BLOCKED-ON-RUN` / needs a GPU. That is right for
the *whole* suite, but the measured rate above says a **single** scale-`n` arm over the mirror's
full split (3,096 train / 546 valid) is `14.0 ×` the images per epoch ⇒ ≈139 s/epoch ⇒
**≈1.5 h for 40 epochs at 320 px** on this CPU host. Full-HRSID scale — 15× the training data than
every current row — is reachable tonight without a GPU. What genuinely needs a GPU is the ladder
(110 variants × scales × 3 seeds), not one more arm.

### F-09 · count and environment drift not covered by a guard — **Low** (documentation)

`docs/DEPLOYMENT.md` (`960 passed, 17 skipped`) and the clean-clone row of
`release_readiness.md` (`960 passed, 17 skipped`) are maintained by hand; every other count in the
repository is pinned by `tests/test_repo.py` or `tests/test_audit_docs.py`. The values are correct
today (local 977 − 17 dataset/GPU-gated skips = 960), but nothing fails if they stop being correct.
Separately, `requirements.txt` declares every dependency with an unbounded `>=` (e.g.
`torch>=2.2.0`, `numpy>=1.24.0`), so a fresh install is not the measured environment; only the
observed versions are recorded, in the ledger's `environment` block.

---

## 6. The most important verified blocker

> **The repository's measurements are not yet a measurement *of the thing it claims to measure.**
>
> Three separately verified gaps compose into one blocker. (1) The evaluation substrate is
> scene-contaminated: 10 scenes total, and **62 % of val and 100 % of test chips come from a scene
> present in training**, so no number in the record is a generalisation number (F-01). (2) The
> paper's Method describes the v1/v2 module ladder — which has **zero** real-data runs — while
> every measured artefact, the committed checkpoint and the demo are the CFAR front end and SSAC,
> which the manuscript never names (F-03). (3) The effects under test are smaller than the noise
> the pilot itself measured (±0.0373 across three seeds), on 60 validation chips (F-07), at a
> resolution where the median object is 17 px (F-06).
>
> The first and second are **removable on this host, from this commit, without new hardware**: the
> split can be regenerated scene-disjointly (the generator exists and has been run — §5/F-01), the
> numbers can be labelled with the split and scene composition they were measured on, and 15×
> more of the same real dataset is reachable in ~1.5 h per arm (F-08). The third is what a GPU
> would change fastest, and it stays `BLOCKED-ON-RUN`.

The single most important **action** that follows is therefore not a model change. It is: build the
honest substrate (scene-grouped folds + a full-release run), re-measure the arms that already
exist on it, and label every number with what it was measured on — then decide, from those
numbers, which architectural component has earned a place.

**Deadline context (verified, external).** CVPR 2027 registration is **10 Nov 2026** and the paper
deadline is **16 Nov 2026** (Anywhere-on-Earth), per the CVF call for papers. From 2026-10-12 that
is **29 days to registration and 35 days to submission**. The remaining time is sufficient for the
CPU-only work above and is **not** sufficient for the ladder at the paper's scale.

---

## 7. The baseline plan (what "reproducible baseline" means here)

The override requires a reproducible baseline before any improvement. In this repository the
baseline is already committed, and the audit established exactly what it is, so the plan is to
**re-derive it rather than rebuild it**:

| Step | Command | Acceptance criterion |
| --- | --- | --- |
| Environment + suite | `.venv/bin/python -m pytest -q` | 977 passed, exit 0 — **done in this audit** |
| Architecture parity | `pytest tests/test_arch.py -k baseline` | published parameter counts reproduced |
| Checkpoint integrity | `sha256sum weights/sarvo_ssac001.pt` | `c81287…f45a98a5` — **done in this audit** |
| Single-image inference | `python -m saryolo predict --weights weights/sarvo_ssac001.pt --source <chip> --imgsz 640` | boxes inside the image, JSON written |
| Val metrics re-derived | `python -m saryolo eval --weights … --split val` | reproduces the ledger's REAL-001 / SSAC-001 numbers within the documented protocol |
| **Substrate repaired** | scene-grouped folds (F-01) + a manifest written to disk | every reported number identifies its split *and* its scene-disjointness |
| **Baseline re-measured on the repaired substrate** | REAL-001 and SSAC-001 on the scene folds | first cross-scene numbers for both arms, with per-fold spread |

Only after that row does an architectural change become admissible, and any such change must
keep the committed checkpoint loadable and write its own checkpoint under a new path
(`results/runs/<NEW-ID>/`, never over `results/runs/SSAC-001/` or `weights/`).

---

## 8. Prioritized implementation sequence

Ordered by expected scientific value per unit of the wall-clock actually available (35 days),
respecting the override: preservation first, one verified problem per change, no broad rewrite.

**P0 — make the record true (CPU, minutes–hours; no model change)**

1. **D-30**: replace `spec.nc = args.nc` with `dataclasses.replace(...)` in `saryolo/nn/arch.py::_main`
   (F-05), and extend the existing D-29 regression test to drive the module entry point as well as
   the CLI. One line plus a test; closes D-29 from its second door.
2. **Persist a scene-grouped fold manifest** for `hrsid_real` and record the leakage result (F-01),
   so the substrate a number came from is an artefact rather than a command someone once ran.
   Re-splitting the *processed* dataset on disk is a change to existing data and is **not** done
   without approval; the fold lists and the manifest are additive.
3. **Label every published pilot number with its split and scene composition** — `README.md`,
   `paper/RESULTS.md` §3, `reports/reproduction_status.md` §3, `docs/assets/facts.json` +
   `scripts/make_readme_assets.py`, and the honest-scope text in `app.py` (F-02). No number changes;
   what changes is that a reader can no longer mistake a within-scene val number for a held-out one.
4. **State the resolution caveat** where the 640 px saving is claimed: the saving is measured at an
   input scale no arm has been trained at (F-04).

**P1 — the smallest experiment that changes the conclusion (CPU, ~1.5–2 h per arm)**

5. **Full-release run of the two arms that already exist** (stock 11n and 11n+SSAC, scale `n`,
   320 px, 40 epochs, same schedule, seeds 0/1/2) over the mirror's full split — ≈1.5 h per arm by
   the measured rate, ≈15× the training data of every current row (F-08). This is the single
   largest increase in statistical power available without new hardware.
6. **Resolution arm**: the same two arms at 640 px. Addresses F-04 and F-06 together and attaches a
   measured accuracy to the operating point where the efficiency result lives.

**P2 — held-out evaluation, once**

7. Evaluate the frozen arms **once** on scene-disjoint folds (F-01) and report per-fold numbers
   with spread, separately from the validation selection numbers (F-02). The test split is not
   used for any tuning, before or after.

**P3 — paper alignment (after P1/P2 have numbers)**

8. Restructure `paper/manuscript/main.tex` around what is measured (CFAR front end, SSAC, SAR
   appearance augmentation), move the v1/v2 ladder to a clearly labelled "designed, not yet
   measured" section, fill the table cells the new runs can fill, and leave the rest `TBD`.
   Verifiable acceptance: zero `TBD` in the tables whose runs P1/P2 produced, and the Method
   listing only measured components (F-03, F-07).
9. Extend the count guards to `docs/DEPLOYMENT.md` and the clean-clone row of
   `release_readiness.md`, and add an upper bound to `requirements.txt` for the packages whose
   version changes behaviour (F-09).

**Not in this sequence, deliberately.** No new architectural module, no replacement of SSAC or the
CFAR front end, no rename, no refactor, and no change to `weights/sarvo_ssac001.pt`. The v1/v2
ladder stays exactly as committed as a selectable baseline. If P1/P2 show that a fundamental
architectural change is required, this document is where the evidence for it starts and the change
stops for approval.

---

## 9. `BLOCKED-ON-RUN` (cannot be removed on this host)

| Item | Why | What would unblock it |
| --- | --- | --- |
| The v1/v2 ladder at the paper's scale (110 variants × scales × 3 seeds) | 30-minute-per-run CPU arithmetic; the suite is ~4 orders of magnitude larger than any single run measured here | a GPU run per `paper/OUTLINE.md` §5 |
| Cross-sensor / cross-resolution LOSO on real data | HRSID ships no verified scene→sensor/resolution sidecar; a second dataset is not on disk | a second sourced dataset + verified acquisition metadata |
| Peak GPU memory, AMP, forced OOM, CUDA kernels | no CUDA device on this host | a CUDA host |
| RT-DETR trained end to end | training-loop integration unfinished, then GPU-bound | integration + a run |
| Hosted public demo | requires interactive account authorisation against the owner's GitHub account | the owner publishing it; everything else is prepared and verified locally |
| Sign/tolerance claims at paper scale | the effects are at or below the measured seed spread on 60 val chips | more data (P1.5) + more seeds |

---

## 10. What this audit did not do

* No model, configuration, checkpoint, test or dataset file was modified. The files inspected are
  named inline above; the commands run are the ones quoted.
* Nothing was re-trained and no published metric was recomputed except the suite itself; the
  numbers cited from `results/experiments.jsonl` are the ledger's, labelled as such, with the
  commands that re-read them quoted.
* The leakage and scene-grouping results in F-01 are new measurements made *during* this audit;
  they are reported here for the first time. The fold lists they produced live under
  `datasets/splits/`, which is git-ignored, so the manifest is reproduced by the command quoted in
  §5 rather than committed.
* No architectural conclusion is drawn. P1/P2 are named as the experiments that would license one.
