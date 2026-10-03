# Initial audit — what exists, what works, what is missing

**Date:** 2026-10-03
**Repository:** `/home/arghya/Projects/Sarr image processing` (branch `main`)
**Method:** repository inspection plus executed commands only. No claim below is made
without a command that produced it. Where a fact could not be established in this
environment it is marked **BLOCKED-ON-RUN** rather than assumed.

---

## 1. Environment (measured)

| Property | Value | How it was established |
| --- | --- | --- |
| Python | 3.12.3 | `.venv/bin/python --version` |
| Ultralytics | 8.4.155 | `requirements.txt` / installed env |
| PyTorch | 2.x (CPU build) | `torch.__version__` |
| **CUDA available** | **False** | `torch.cuda.is_available()` |
| **GPU device count** | **0** | `torch.cuda.device_count()` |
| Real datasets on disk | **none** | `find datasets -maxdepth 2 -type d` → only `processed/synthetic_smoke` |
| Checkpoints on disk | 11 (all smoke) | `find runs results -name '*.pt' \| wc -l` |
| Test suite | **366 passing** | `pytest -q` |

**The single most important fact in this audit:** this machine has no GPU and no real
SAR dataset. Every accuracy number that depends on training on SSDD/HRSID/SARDet-100K is
therefore **not obtainable here**, and none has been produced. What *is* obtainable here —
and has been measured — is architecture cost, identity-at-initialisation, protocol
correctness, and metric correctness. The repository is built around that split.

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

### 2.2 Data

| Area | State | Evidence |
| --- | --- | --- |
| Dataset registry + configs | **Works** | 5 datasets (SSDD, HRSID, SRSDD, SAR-Ship, SARDet-100K) each with a matching data config |
| Converters (VOC / COCO / DOTA) | **Work** | oriented-label trap is diagnosed, not silently mis-parsed |
| Validator + statistics | **Work** | refuses to validate oriented labels as clean |
| Source grouping + LOSO folds | **Work** | refuses random splits, single groups, unkeyed images, zero-GT folds |
| Acquisition metadata table | **Works for SARDet-100K (published source table)** | HRSID has **no per-chip resolution mapping** → its resolution axis is inert until a sourced sidecar exists |
| Real data on disk | **Absent** | **BLOCKED-ON-RUN** |

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
| **No measured accuracy anywhere** | Blocking for the paper | Needs GPU + a real dataset. Every accuracy cell is `TBD`. |
| HRSID per-chip resolution | Blocking for the cross-resolution axis | The release ships no scene→chip mapping; fields stay null rather than being invented. |
| RT-DETR training-loop integration | Medium | The facade and graph work; the task-map entry that routes a conditioned predictor into the RT-DETR trainer is not implemented, so no cross-architecture *number* can exist yet. |
| RGB-pretraining baseline arm | Built but unmeasured | `init:` key + EXP-401…403 exist; the transfer counts are verified, the accuracy is not. |
| LoRA / parameter-efficient adaptation baseline | **Not implemented** | The master context (Direction B / Stage 10) requires it for a fair adaptation comparison. Absent. |
| Data-efficiency sweep (1–100%) | **Not implemented** | Master context Direction E. Absent. |
| `docs/literature_audit.md`, `docs/novelty_and_overlap.md`, `docs/research_questions.md`, `docs/baseline_comparison.md` | Missing | Master-context deliverables. `docs/research_gap.md` covers part of this. |

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
* efficiency profiling is internally consistent (see §2.3).

**Not** verified anywhere: any accuracy claim, any generalisation claim, any claim that a
component improves detection. Those require a GPU run.

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
* The **existing 366 tests**. A change that requires deleting a test is a change to the
  scientific claim and must be justified as such.

---

## 6. Prioritised next actions

Ordered by what unblocks the most, not by what is most interesting.

| # | Action | Objective | Output | Depends on | Validation | Why this priority |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Run the SSDD pilot end to end** (`docs/RUNBOOK_SSDD.md`) | Produce the first real measured number | ledger rows + filled tables | GPU + SSDD on disk | `python -m saryolo ledger`; `assets --require-complete` | Nothing in the paper is defensible without it. It is the single gate on everything else. |
| 2 | **Add a LoRA / adapter baseline** (master Direction B) | Make the adaptation comparison fair | new variant + config + tests | none (buildable now) | `pytest`; a smoke run | The master context requires it; without it SARVO's efficiency claim has no parameter-efficient comparator. Buildable without a GPU. |
| 3 | **Data-efficiency sweep** (Direction E) | Test the low-label regime | config generator + table | action 1 | smoke run on a fraction | Cheap to add, and it is where a conditioning method should show its value. |
| 4 | **Finish RT-DETR training-loop integration** | Make architecture generality measurable | task-map wiring + tests | action 1 | `tests/test_rtdetr_arm.py` extended | A cross-architecture number is a strong contribution, but it needs a GPU run to mean anything. |
| 5 | **Write the master-context literature deliverables** | Close the novelty audit | `docs/literature_audit.md`, `novelty_and_overlap.md`, `research_questions.md` | none | review | Required by the master context; no GPU needed. |

---

## 7. Honest summary

The repository is a **verified instrument with no measurements**. Its architecture,
protocol, metrics, metadata handling and efficiency profiling are all tested and working;
its accuracy is entirely unknown because it has never been trained on a real SAR dataset.
The correct next step is not another module — it is the first real run (action 1), followed
by the missing baselines (actions 2–3) that the master context requires for a fair
comparison.
