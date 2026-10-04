# Repository and resource audit — what exists, what works, what is missing

**Date:** 2026-10-04 (refreshed; first written 2026-10-03)
**Repository:** `/home/arghya/Projects/Sarr image processing` (branch `main`)
**Method:** repository inspection plus executed commands only. No claim below is made
without a command that produced it. Where a fact could not be established in this
environment it is marked **BLOCKED-ON-RUN** rather than assumed.

This is the master workflow's Phase-1 deliverable. It is refreshed rather than rewritten
because most of it was still true on 2026-10-04; the cells that moved are marked **[now]**
so a reader can see what changed since the first pass. The two changes that matter: a real
SAR dataset is on disk and has been measured, and the LoRA / data-efficiency items that were
listed as missing are built.

---

## 1. Environment (measured)

| Property | Value | How it was established |
| --- | --- | --- |
| Python | 3.12.3 | `.venv/bin/python --version` |
| Ultralytics | 8.4.155 | `requirements.txt` / installed env |
| PyTorch | 2.x (CPU build) | `torch.__version__` |
| **CUDA available** | **False** | `torch.cuda.is_available()` |
| **GPU device count** | **0** | `torch.cuda.device_count()` |
| Real datasets on disk | **[now] HRSID subset, 200 train / 60 val / 60 test** | `docs/assets/facts.json` → `real_subset`; counted on disk by `_dataset_split_counts` |
| Checkpoints on disk | **[now] including three real-data arms** | `results/runs/REAL-00{1,2,3}/weights/best.pt` |
| Test suite | **[now] 426 passing / 21 files** | `pytest -q` |
| Full SAR benchmark on disk | **none** (subset only) | the assembled release is a fraction of HRSID's 5,604 images |

**The single most important fact in this audit:** this machine has no GPU. Every accuracy
number that depends on the *full* release, multiple sources, or multiple seeds is therefore
**not obtainable here**, and none has been produced. What *is* obtainable here — and has been
measured — is architecture cost, identity-at-initialisation, protocol correctness, metric
correctness, and (since 2026-10-04) a three-arm pilot on a real HRSID subset. The repository
is built around that split, and the pilot is labelled a pilot everywhere it appears.

---

## 2. What exists

### 2.1 Model and architecture

| Area | State | Evidence |
| --- | --- | --- |
| Symbolic architecture builder | **Works** | `saryolo/nn/arch.py`; baseline reproduces stock YOLO11 exactly (2,624,080 n / 9,458,752 s at 80 classes) |
| Variant registry | **92 variants** | `len(VARIANTS)`; every one builds and forwards |
| Custom modules (Components 1–12, 33) | **Work** | registered into `ultralytics.nn.tasks`; each is an exact identity at init (`max\|f(x)−x\| = 0.0e+00`) |
| Acquisition-conditioned adapter (Comp 33) | **Works, wired end to end** | gate-closed identity; per-sample; vocabulary mismatch raises rather than clamping |
| RT-DETR feasibility arm | **Builds and forwards** | `SARYOLORTDetectionModel` + generated `rtdetr_s_cond_film.yaml`; bit-identical to stock at init |
| **Efficiency frontier (SARVO-Lite)** | **Cost measured** | 11.016 M / 32.55 GFLOPs vs reference 16.230 M / 55.68 GFLOPs at scale `s` |
| **[now] Parameter-efficient arm (LoRA)** | **Built and measured on the pilot** | `saryolo/training/peft.py`; `REAL-003` recorded 273,896 adapter params, 87 wrapped layers, 10.1 % trainable; the checkpoint is an ordinary detector with the adapter folded in |
| **[now] Data-efficiency sweep** | **Built, unmeasured** | `EXP-701…705` declare native `fraction` values 1/5/10/25/50 %; nesting is pinned by test |
| **[now] Architecture direction** | **Three proposals written, one recommended; nothing implemented** | `docs/architecture_proposals.md` — the current v2 is a module ladder on a YOLO11 skeleton, which the brief rules out as a final answer |

### 2.2 Data

| Area | State | Evidence |
| --- | --- | --- |
| Dataset registry + configs | **Works** | 5 datasets (SSDD, HRSID, SRSDD, SAR-Ship, SARDet-100K) each with a matching data config |
| Converters (VOC / COCO / DOTA) | **Work** | oriented-label trap is diagnosed, not silently mis-parsed |
| Validator + statistics | **Work** | refuses to validate oriented labels as clean |
| Source grouping + LOSO folds | **Work** | refuses random splits, single groups, unkeyed images, zero-GT folds |
| Acquisition metadata table | **Works for SARDet-100K (published source table)** | HRSID has **no per-chip resolution mapping** → its resolution axis is inert until a sourced sidecar exists |
| Real data on disk | **[now] Present (subset)** | HRSID, 200/60/60, one class, YOLO boxes, via `scripts/fetch_hrsid_subset.py` |
| Per-chip resolution metadata | **Absent** | HRSID ships no scene-to-chip mapping, so the cross-resolution axis stays inert on this dataset |

### 2.3 Training and evaluation

| Area | State | Evidence |
| --- | --- | --- |
| Trainer + runner + ledger | **Work** | append-only JSONL; failed runs retained with their error |
| SAR-aware loss | **Works** | reaches the optimiser; `train/sar_loss` non-zero in smoke logs |
| COCO AP + scale-wise AP | **Works** | matcher agrees with hand-computed cases |
| Robustness sweep | **Built, unmeasured** | deterministic corruptions, GT untouched |
| Efficiency profiling | **Works and now deeply tested** | params exact; FLOPs scale with area (3–4.5× for 4× pixels); latency consistent and side-effect-free |
| Representation probes | **Built, unmeasured** | requires a trained checkpoint |
| Cross-dataset / domain-gap | **Built, unmeasured** | refuses incompatible label spaces |

### 2.4 Paper and reproducibility

| Area | State |
| --- | --- |
| Paper table generators | **Work**; unmeasured cells render `TBD`; no code path invents a number |
| Efficiency frontier table | **New**; measured cost + separately-sectioned *cited* published baselines |
| Submission gate | `python -m saryolo assets --require-complete` fails while any cell is unmeasured |
| Figure generators | **Work**; refuse to draw with missing data (the cost-frontier figure returns `None` rather than plotting at a guessed height) |
| README charts | **Generated and committed**; byte-reproducible; layout-checked |

---

## 3. What is broken or incomplete

| Item | Severity | Detail |
| --- | --- | --- |
| **No benchmark accuracy anywhere** | Blocking for the paper | A pilot exists on a subset; the full release, multiple sources and multiple seeds still need a GPU. Every ladder accuracy cell is `TBD`. |
| HRSID per-chip resolution | Blocking for the cross-resolution axis | The release ships no scene→chip mapping; fields stay null rather than being invented. |
| RT-DETR training-loop integration | Medium | The facade and graph work; the task-map entry that routes a conditioned predictor into the RT-DETR trainer is not implemented, so no cross-architecture *number* can exist yet. |
| RGB-pretraining baseline arm | Built but unmeasured | `init:` key + EXP-401…403 exist; the transfer counts are verified, the accuracy is not. |
| **[now] The architecture is a module ladder on a YOLO11 skeleton** | **Highest** | The brief rules this out as a final answer: a collection of modules on someone else's graph is not an independently designed architecture. `docs/architecture_proposals.md` answers it with three directions and one recommendation; **none is implemented yet**. |
| **[now] LoRA / parameter-efficient adaptation baseline** | **Resolved** | `saryolo/training/peft.py` + `EXP-601…605`; one pilot arm measured (`REAL-003`). Two real bugs were found by measurement and fixed (adapter discarded during model rebuild; checkpoint unloadable with wrappers). The tuned rank/scale sweep is still unmeasured. |
| **[now] Data-efficiency sweep (1–100 %)** | **Built, unmeasured** | `EXP-701…705` via the library's native `fraction`; nesting pinned by test. |
| **[now] Master-context documents** | **Resolved** | `docs/literature_audit.md`, `novelty_and_overlap.md`, `research_questions.md`, `baseline_comparison.md`, `related_work.md`, `architecture_proposals.md` all exist and are guarded by `tests/test_audit_docs.py`. |

---

## 4. What has been experimentally verified (vs merely built)

Verified **here, now**:

* the baseline is bit-identical to stock YOLO11 in parameter count;
* every module is an exact identity at initialisation, and none is frozen (non-zero gate
  gradient);
* a fresh conditioned model equals the unconditioned one; metadata cannot change the
  output while the gate is closed;
* the LOSO protocol runs end to end on a conditioned model, and a held-out sensor lands on
  the unknown row while its physical descriptors survive;
* a withheld metadata field encodes exactly like a never-recorded one;
* the frontier arms are strictly cheaper than their scale-matched reference, dropping only
  the two declared slots;
* efficiency profiling is internally consistent (see §2.3);
* **[now]** three arms on a real HRSID subset trained, validated and profiled under one
  schedule, with every number written to the ledger by the run that measured it;
* **[now]** a subset pilot cannot fill a benchmark cell — enforced structurally, not by
  convention (`_is_subset_pilot` + `test_a_subset_pilot_cannot_fill_a_benchmark_cell`);
* **[now]** the LoRA arm's adapter genuinely influences the checkpoint (asserted from the
  saved file: 409/499 tensors differ from a fresh build, zero `lora_` keys).

**Not** verified anywhere: any *cross-sensor*, *cross-resolution* or *multi-seed* claim, or
any claim that a component improves detection on a full benchmark. Those require a GPU run
on the full release.

---

## 5. What must be preserved

* The **no-fabrication contract**: the ledger, the `TBD` rendering, and the
  `--require-complete` gate.
* The **provenance rule**: every commit authored and committed under the owner's identity,
  with no attribution trailers. Guarded mechanically by `tests/test_repo.py`.
* The **generated-artifact rule**: model YAMLs and experiment configs come from their
  builders, never hand-edited.
* **Component identity-at-init**, which is what makes any future gain attributable to
  learning rather than to added capacity.
* The **existing 426 tests** (`pytest -q`). A change that requires deleting a test is a
  change to the scientific claim and must be justified as such.

---

## 6. Prioritised next actions

Ordered by what unblocks the most, not by what is most interesting.

| # | Action | Objective | Output | Depends on | Validation | Why this priority |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **[now] Implement the recommended architecture direction as a prototype** | Turn a module ladder into an architecture claim | the recommended arm + forward/gradient/inference verification | none (CPU-only) | `pytest`; a measured cost profile; the identity/behaviour checks | This is the highest-severity gap. The brief's central requirement is an independently designed architecture, and only a prototype can decide whether the recommendation survives |
| 2 | **[now] Run the recommended arm against the matched-cost control on the existing subset** | Falsify or support the organizing principle | two ledger rows + a table row | ~30–60 min CPU | the ledger; `assets --require-complete` stays red on everything else | It is the first experiment in the whole project whose result can *withdraw* a proposal, which is what makes it worth doing first |
| 3 | **[now] Run the SSDD pilot end to end** (`docs/RUNBOOK_SSDD.md`) when a GPU is available | Produce a full-benchmark number | ledger rows + filled tables | GPU + SSDD on disk | `python -m saryolo ledger`; `assets --require-complete` | Nothing in the paper is defensible without a full-release run; it is the gate on every accuracy claim |
| 4 | **[now] LoRA rank/scale sweep and the data-efficiency sweep** | Close the two built-but-unmeasured axes | `EXP-601…605`, `EXP-701…705` rows | CPU, hours | `pytest`; the ledger | Both are already built, so the marginal cost is compute, not design |
| 5 | Finish RT-DETR training-loop integration | Make architecture generality measurable | task-map wiring + tests | action 3 | `tests/test_rtdetr_arm.py` extended | A cross-architecture number is a strong contribution, but it needs a GPU run to mean anything |
| 6 | Master-context documents | Close the novelty and planning audit | `docs/related_work.md`, `docs/architecture_proposals.md` — **done** | none | `tests/test_audit_docs.py` | Completed 2026-10-04 |

---

## 7. Honest summary

The repository in 2026-10-03 was a **verified instrument with no measurements**. On
2026-10-04 it is a **verified instrument with one honest pilot**: three arms measured on a
real HRSID subset, every number written by the run that produced it, and the one bug that
would have made the most quotable of those numbers a lie (a parameter-efficient arm whose
adapter was silently discarded) caught because a designer would not have looked for it and a
*measurement* did.

What it still is not: an independently designed architecture. Its core graph is a YOLO11
skeleton carrying SAR modules, which is the specific outcome the brief forbids, and as of
2026-10-04 there are three written candidate directions and **no implementation of any of
them**. That is the first thing to fix, and action 1 is the smallest change that moves it.
The accuracy of anything remains unknown at benchmark scale, because that needs a GPU and a
full release; the audit says so rather than filling the gap.
