# SARVO / SAR-YOLO — master-directive status report

**Date:** 2026-10-11
**Scope:** the whole repository, consolidated from the working records rather than restated
from memory. This is the single report the master directive asks for: repository findings,
compute findings, closest prior work, the revised contribution, and the first executable
task — each tied to the artefact that produced it.

**Method.** Every statement about this repository is a measurement or a test named in-line,
and every statement about prior work keeps the evidence depth recorded in
[`../docs/ssac_assessment.md`](../docs/ssac_assessment.md) (`verified` = page read,
`abstract-read`, `snippet-read`). Nothing here declares the mechanism novel. Where this
report and a measurement disagree, the measurement wins and this file is wrong.

The detailed records this consolidates:

| Record | What it holds |
| --- | --- |
| [`../paper/RESULTS.md`](../paper/RESULTS.md) §8 | the pilot's measured numbers and their protocol |
| [`../paper/OUTLINE.md`](../paper/OUTLINE.md) | the claim labels (C1–C5) and the blocked-on-run list |
| [`../docs/ssac_design.md`](../docs/ssac_design.md) | the mechanism specification |
| [`../docs/ssac_assessment.md`](../docs/ssac_assessment.md) | the novelty and prior-art audit |
| [`../docs/claim_evidence_audit.md`](../docs/claim_evidence_audit.md) | each claim with its label and evidence |
| [`reproduction_status.md`](reproduction_status.md) | the commands and their measured outputs |
| [`defect_log.md`](defect_log.md) | defects found and fixed, each with its regression test |
| [`release_readiness.md`](release_readiness.md) | requirement-by-requirement readiness and gaps |

---

## 1. What the directive asked, and what this answers

The master directive is a ten-stage workflow — *inspect → baselines → novelty → mechanism →
implement → test → benchmark → analyse → refine → document* — with an explicit no-fabrication
rule and a required first status report. This document is that report. It is deliberately a
*consolidation*: it introduces no new number, and it links to the record that measured each
one.

---

## 2. Repository findings

**What the repository is.** A SAR object-detection research codebase (working name
**SARVO** — *SAR Acquisition-Robust Visual Optimization*) built on a forked Ultralytics YOLO
graph, with one candidate central mechanism (**SSAC**, Scatter-Selective Adaptive
Computation) and a ladder of supporting ablations. Current size, measured by the suite and
the registry:

| Quantity | Value | Measured by |
| --- | --- | --- |
| Architecture variants wired | 110 | `saryolo.nn.arch.VARIANTS` (`tests/test_repo.py`) |
| Experiment configs | 134 | `saryolo.training.config.list_experiments` |
| Tests | **979 tests** | `pytest --collect-only -q` (`tests/test_repo.py`) |
| Datasets on disk | HRSID subset, 200 / 60 / 60, one class (`ship`) | `docs/assets/facts.json` → `real_subset` |

**The architecture ladder.** Every module is an **exact identity at initialisation** and is
inserted so the stock summary, FLOPs counter, validator and checkpointing keep working
(`tests/test_arch.py`). The mechanisms, in the order the record established them:

* **CFAR front end** (multi-scale log-ratio + local coefficient of variation), built and
  measured on real data — `REAL-004` against the matched-cost control `REAL-006`.
* **Conditioning** (acquisition metadata adapters, with frozen vocabularies), tested and
  wired, but its generalisation claim is **unmeasured**.
* **SSAC** (the candidate mechanism): cheap shared path → analytic SAR statistic → tiny
  learned scorer → expensive bottleneck path → gated residual mix, with a matched
  fixed-computation control of *exactly equal parameter count*.
* **PEFT / LoRA**, hard-example mining, SAR augmentation, group/LOSO splitting — all
  implemented and tested.

**What the pilot measured, and what it did not.** On the HRSID subset at 320 px, CPU only,
one class:

* `SSAC-001` (the proposal) scored **mAP50 / mAP50:95 = 0.5626 / 0.3089**; the baseline
  `REAL-001` scored **0.5706 / 0.3012** (`docs/assets/facts.json`, ledger
  `results/experiments.jsonl`).
* The *accuracy* claim for adaptive allocation — that it beats the parameter-identical fixed
  control — **looked** supported at seed 0 and **does not survive three seeds**: paired
  differences +0.0063 / +0.0645 / −0.0050, mean **+0.0220 ± 0.0373** (paper/RESULTS.md §8.7).
  It is **withdrawn**. This is the single most important honest negative in the record.
* Small-object preservation **did not** trip its pre-registered withdrawal condition
  (§8.2); the allocation is genuinely spatial (§8.3).

**Engineering posture.** Every "ran" claim is tied to a command or a test; unmeasured cells
render as `TBD`; the ledger is append-only; the demo never shows a prediction it did not
compute; defects are recorded with the regression test that pins them (`defect_log.md`,
D-01…D-30). Four deepscans of the previously untested surfaces (environment capture, the
qualitative panels, the corruption sweep, the gain diagnostic, the shared module primitives,
the YAML generator, the dataset validator, checkpoint registration, and — in the third pass —
the paper table/figure builders, the runner and statistics writers, the LoRA layer, the
conditioning data join, the failure-analysis writers and the metadata/registry/split helpers)
and of the efficiency/scope corners exposed and fixed five defects (D-19 a mislabelled FP16
provenance field on a CUDA-less host; D-20 a silently-degraded gain input; D-21 an omitted
routing report; D-22 a checkpoint safe-load allowlist that was insufficient and unreachable,
making the shipped checkpoint unloadable under `ULTRALYTICS_SAFE_LOAD`; D-23 a contrast-map
reader that resolved a list-split's relative entries against the process cwd, so the map
failed or emptied and the clutter proxy silently stopped working; and — in the fourth pass,
which re-ran the public-API inventory and a full-suite coverage map before reading anything —
D-24 the labels-directory rule used by six consumers, which replaced the *first* `images`
substring in a path rather than the `images` path component, so any dataset whose root
contains the word resolved its labels into a directory that does not exist and every metric
came back `None` over an empty ground truth on a run that reported success; and D-25 an
RT-DETR criterion fallback that supplied the v8 argument namespace to a criterion that reads
`nc`, so a bare model still failed — the error its docstring said it prevented). A fifth pass
closed the *documented-helper* gap and re-took the efficiency measurement: D-26 the per-level
routing report attached to a timing, which described a *dense* pass with the sparse allocation's
fractions (a run that executed 1.00 printed 0.44, i.e. a saving it never took, and the guard that
pins the dense row to 1.00 would have caught it only after the fact); D-27 the cost sweep itself,
whose dense baseline was timed in a *different process* from its sparse rows on a host whose
between-process drift (179.7 ms vs 203.5 ms for the same checkpoint at 640 px) exceeds the effect —
re-measured with every row in one process, the 640 px saving is **8.5 %** and not the 13.5 % the
split-process pairing reported, while the direction (a saving at 640 px, none at 320 px, the
tightest budget's range entirely below the dense one) is unchanged; and D-28 two docstrings that
pointed the reader at a measurement script that does not exist, now a guard over every
`scripts/*.py` named anywhere in the tree; and — found by the CLI test that commands are written
to prove — D-29 `saryolo arch` writing its `--nc` flag **into the shared variant registry** rather
than into a copy, so one `--nc 2` invocation left all 28 specs at two classes for the rest of the
process and made the committed-YAML byte-identity guard report a clean tree as hand-edited. A sixth pass was a read-only Phase A audit of the whole repository against the master context (its own report: [`phase_a_audit.md`](phase_a_audit.md)), which found the same class one door further on — D-30, the identical registry-mutating line surviving in `python -m saryolo.nn.arch`, the entry point no test drove — and, more consequentially, three findings about the *evidence* rather than the code: the pilot's split is scene-contaminated (10 scenes, 62 % of val and 100 % of test chips drawn from a scene also in training, because the subset is the first N chips in filename order), the published pilot metrics are validation metrics that the documents do not label as such, and the manuscript's Method describes the v1/v2 ladder — which has no real-data run at all — while never naming SSAC or the CFAR front end, which is where every measured artefact lives. Those are recorded with their measurements in `phase_a_audit.md` and carry the prioritized sequence; they are findings about scope and labelling, not defects with a regression test, so they are not rows in the table above. The suite runs on the development host with **no GPU and no dataset download**
(the dataset-gated and GPU-gated checks skip with a *named* reason).

---

## 3. Compute findings

The host decides what can be claimed, so it is stated up front.

| Quantity | Value | Consequence |
| --- | --- | --- |
| GPU | **none** (`torch.cuda.is_available() == False`) | large-scale training, multi-seed-at-scale, LOSO, AMP, CUDA peak memory, forced OOM are **`blocked-on-run`** |
| Framework | PyTorch **2.14.0+cpu** | the CUDA-wheel memory envelope is a stated requirement, enforced by CI, not measured here |
| CPU / RAM / disk | 16-core AMD Ryzen 7 8840HS · 30 GB · 112 GB free | CPU-feasible training and all efficiency measurement |
| Demo peak RSS (CPU wheel) | **451 MB** measured (fresh interpreter) | inside the **1024 MB** CPU budget; the CUDA budget is **3072 MB** |
| Checkpoint | `weights/sarvo_ssac001.pt`, 3,396,329 params, 6.77 MB | committed, so a hosted deploy needs no configuration |

The compute finding that shapes the paper is not the host, however — it is the
**input-scale ceiling** on the efficiency claim:

* At **320 px** (the training scale) the deepest governed level of the mechanism is a
  **single 16 px tile** (tiles/level `[9, 4, 1]`), so no routing budget can spare it.
  Sparse execution is **+51 %** latency at `keep = 1.0` (the gather's halo recompute) and no
  budget reaches the dense latency — the closest is +5 % — so **no saving is claimed at
  320 px**.
* At **640 px** the same level splits into four tiles (`[25, 9, 4]`), the overhead at
  `keep = 1.0` falls to **+25 %**, and sparse execution lands below the dense range at
  `keep ≤ 0.25`: **−6.4 / −8.5 %** at 0.25 / 0.1, its range (177.1–184.2 ms) entirely below
  the dense row's (191.7–211.5 ms) (paper/RESULTS.md §8.5).
* Every row of that sweep is timed **in one process** (dense baseline and budgets round-robin,
  three cycles of three blocks), because this host's between-process drift is larger than the
  effect: an earlier pairing of separately-timed runs reported 13.5 % instead of 8.5 %.
* The FLOP counters **cannot see the sparsity** — a sparse build reports *more* GFLOPs
  (2.293 vs 1.998 G) because they trace kernels regardless of which execute. The efficiency
  claim therefore rests on measured wall-clock, never on a FLOP ratio.

---

## 4. Closest prior work (condensed from the novelty audit)

The principle is **not novel** and is not claimed as such. The audit's conclusion is that
"allocate compute per region from image evidence" is the dynamic-network programme:

| Closest work | Evidence depth | What it already owns |
| --- | --- | --- |
| Spatially Adaptive Computation Time (SACT), CVPR 2017 | `abstract-read` | the exact core idea: per-region evidence → per-region compute |
| SplatNet, ECCV 2018 | `snippet-read` | region-varying *resolution* from a learned gate |
| Dynamic Neural Networks survey, TPAMI 2022 | `snippet-read` | the whole design space, including an explicit "region selection" category |
| Adaptive/routing detection (AdaDets-class), 2024–2026 | `snippet-read` | adaptive computation *inside a detector* |
| CFARNet, arXiv 2022 | `abstract-read` | a learned detector built on the CFAR statistic (as an output constraint) |
| SARFormer, CVPRW 2025 | `verified` | the closest *SAR-conditioning* work (no detection, no compute allocation) |
| SARDet-100K / MSFA, NeurIPS 2024 | `abstract-read` | the benchmark and the RGB→SAR motivation |

**Where a contribution could live.** Not in P1 (the principle — taken). Only in
**P2** (*what* evidence drives the decision) and **P3** (how weak targets are protected) —
and the only honest P2 signal is the SAR-native statistic the repository already measured,
reused rather than reinvented. That gap is narrow, and the report says so.

---

## 5. Revised contribution

After the novelty audit **and** the three-seed negative, the contribution is the *mechanism
and its measurement protocol*, not a headline accuracy gain:

| # | Contribution | Label today |
| --- | --- | --- |
| C1 | SSAC: per-region allocation of expensive feature processing driven by a SAR-native analytic statistic, with a matched fixed-computation control of exactly equal parameter count | mechanism **final**; its accuracy effect **withdrawn at pilot scale** |
| C2 | A sparse execution of the same mechanism (tile gather/scatter, kernel-derived halo), parameter-identical to dense and reproducing it **bit-for-bit at `keep = 1.0`** in eval | **controlled** |
| C3 | A wall-clock measurement of selective computation showing where the saving appears and why the input scale decides it | **preliminary at 640 px**, not separable at 320 px |
| C4 | A cost ablation isolating how much of the parameter price the accuracy needs (`expand 2 → 1` loses the benefit) | **controlled** (seed 0) |
| C5 | The sparsity premise: a penalty can push the learned allocation sparse at no measurable accuracy cost | **preliminary** (seed 0) |

The abstract must state the withdrawal and the scale-bound efficiency result. If the
mechanism never clears its controls at the paper's scale, the paper is about the
**measurement** — what selective computation does and does not buy on SAR — and that fallback
is pre-registered in [`../paper/OUTLINE.md`](../paper/OUTLINE.md) so it cannot be dropped
quietly.

---

## 6. First executable task, and what is blocked on a run

**First executable task (no GPU required).** Train an arm **at 640 px** and re-run the
dense/sparse sweep on it, so an efficiency number and an accuracy number exist at the *same*
named scale — the 640 px saving is currently one checkpoint measured at evaluation time.
Concretely: `python -m saryolo train --exp configs/exp/SSAC-001_hrsid_ssac.yaml` with
`imgsz=640` (a small-epoch CPU run is feasible), then
`python -m saryolo efficiency --weights … --data configs/datasets/hrsid_real.yaml --real
--ssac-execution sparse --ssac-keep 1.0 0.5 0.25 0.1 --ssac-tile 16 --runs 5`, and fold the
result in with `python scripts/make_readme_assets.py`. This converts C3 from "one checkpoint"
toward a claim, and it is the step the paper's writing order (§7 of the outline) waits on.

**Deployment (owner action, one click).** The Streamlit demo, the CLI `predict` entry point,
the committed checkpoint and the deploy-contract tests are all in place; the hosted instance
is **not yet created**, because creating it needs an interactive sign-in only the owner can
perform. Then `python scripts/verify_deploy.py --url <url>` reports the HTTP verdict and the
browser table in [`../docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) §5 covers the in-app
inference that HTTP cannot see.

**Blocked on a run (no GPU on this host).** Full-HRSID training, more seeds/longer schedules,
cross-sensor / cross-resolution LOSO folds, compute-matched region-selective baselines, the
CUDA-wheel envelope, and any forced-OOM behaviour. Each is recorded in
[`../paper/OUTLINE.md`](../paper/OUTLINE.md) §5 and [`release_readiness.md`](release_readiness.md)
as an explicit gap rather than omitted.

---

## 7. Where each fact lives (traceability)

| Fact class | Source of truth |
| --- | --- |
| Accuracy / efficiency numbers | ledger `results/experiments.jsonl` → `docs/assets/facts.json` → `paper/RESULTS.md` §8 |
| Claim labels | `docs/claim_evidence_audit.md` rows 17–21 |
| Mechanism | `docs/ssac_design.md`; `tests/test_ssac.py` |
| Novelty / overlap | `docs/ssac_assessment.md` |
| Defects and their guards | `reports/defect_log.md` |
| Readiness and gaps | `reports/release_readiness.md` |
| Deployment | `docs/DEPLOYMENT.md`; `tests/test_deploy.py` |
| Counts (variants / experiments / tests) | `tests/test_repo.py` guards; `docs/assets/facts.json` |

This report adds no number of its own; the one count it states — **979 tests** — is the
same one the suite's own count guards check.
