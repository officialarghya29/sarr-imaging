# Progress: master workflow → repository status

Mapped on 2026-09-23 against the current master objective (acquisition-conditioned
invariant detection, CVPR 2027, deadline 2026-11-16). The split that matters is
**built** vs **measured**: infrastructure is verifiable without a GPU, results are
not — every accuracy cell in every *ladder* table is `TBD` until a real training run
happens. Three real-data pilot arms have now been measured on a subset of HRSID (item 8
below and §3 of `reports/reproduction_status.md`); they are a beginning, not the benchmark.

## Phase map

| Master phase | Status in this repo | Evidence |
| --- | --- | --- |
| 0. Research landscape | **Started, not closed** | `docs/research_gap.md` + `docs/related_work.md` (the consolidated Phase-2 table) — verified entries + flagged full-text reads; the two closest works (SARFormer, Zhang et al. 2026) read and differentiated; novelty claims frozen until the remaining abstract reads |
| 1. Dataset audit | Built (tooling) | `check-data` / `stats` commands; EXP-001…003 configured; refuses oriented-label and empty-dataset traps |
| 1b. Leakage check | Built | Per-fold leakage mode in `loso --leakage`; duplicate detection |
| 1c. Acquisition metadata for the primary datasets | Built | Verified per-source profiles (SARDet-100K's official 10-source table; HRSID's stated resolutions/sensors) → `write_acquisition_metadata` → MetadataTable → LOSO resolution folds + conditioning arms. The cs231n mirror vendors HRSID+SSDD with on-disk counts, the fastest licensed route to the pilot datasets |
| 2. YOLO11 baseline | Built, **not measured** | Baseline reproduces stock YOLO11 exactly (identity test); EXP-004 configured |
| 3. RGB-pretraining baseline | **Missing** | Nothing in the repo establishes the RGB→SAR fine-tuning arm or the random-init control. Required before the input-adapter claim |
| 4. SAR input adapter | Built | Component 12 (SIA), slot EXP-311…314, identity-at-init verified |
| 5. Prove the failure (cross-sensor) | Protocol built, **not measured** | LOSO folds + refusals (`saryolo.data.groups`); degradation itself still unmeasured — the gating fact for the whole paper |
| 5b. Cross-resolution | Protocol built, **not measured** | `loso --rule resolution` + `metadata` command (Experiment D), documented in README and METHOD |
| 5c. Failure analysis | Partially built | Failure taxonomy + hard-example mining exist; domain-gap table generator is untested against real folds |
| 5d. First-GPU runbook | **Built** | `docs/RUNBOOK_SSDD.md` — prepare → audit → LOSO folds → metadata → baseline → conditioning → probe, with the go/no-go decision gate; Stages 1–2 verified live against a synthetic VOC fixture |
| 5e. Explainability figures | **Fixed and pinned** | Grad-CAM was dead (the model object is not indexable; a forward over the raw layer list raises at the first Concat) and its signal read the eval head's decoded tensor with the training layout's offset, so the "attribution" was the decoded *box coordinates*. Now: layout chosen by the head's declared width, a zero signal refuses instead of publishing a black square, the adapter is a valid target, and both figure paths condition on the image's acquisition and clear it afterwards (`tests/test_visualization.py`) |
| 6. Representation diagnosis | **Built, not measured** | `saryolo.evaluation.probes` (probe accuracy, within-class drift, linear CKA) + the `probe` CLI command; side-effect-free extraction pinned by test. A conditioned checkpoint is now probed *conditioned* — its acquisition descriptors are encoded from its own frozen vocabularies — because an unconditioned probe would report the baseline's representation and make the comparison meaningless. Awaiting a trained checkpoint |
| 7. Acquisition encoder | Built, **wired end to end** | Component 33 (CND) + `saryolo.data.metadata`; <0.5% overhead, measured; per-sample property pinned. The real `SARYOLOTrainer` now runs over a metadata-bearing dataset in `tests/test_conditioning_smoke.py`, and both the training batch and the *validation* forward are asserted to consume metadata — validation is where a `Path` handle used to be stored as if it were the model, which silently scored an unconditioned network. The LOSO protocol itself is now smoke-tested: a three-sensor fixture trains on two sources, and the held-out sensor's encoding is asserted to hit the reserved unknown row while its physical descriptors survive (`test_a_conditioned_model_trains_without_the_held_out_sensor_and_still_validates_on_it`) |
| 7b. Missing-metadata degradation (master workflow) | **Built, not measured** | `metadata_fields` in a data config (or `loso --metadata-fields`) withholds acquisition fields at encode time; a withheld field is encoded exactly like a never-recorded one (value, availability and categorical id all masked — `tests/test_field_mask.py`). The restriction is propagated into every generated fold config so all folds share one protocol. Answers the deployment question "does the detector collapse without metadata?" once a GPU run exists |
| 8. Invariant branch + decoupling loss (ACID-SAR §15–19) | **Missing** | Deliberately: the spec says build it only after the failure and the representation diagnosis exist |
| 9–12. Multi-seed, RT-DETR transfer, ablations | Not started | Correctly ordered after the above |
| Pilot (real data, CPU) | **Measured, 3 arms** | The official HRSID release assembled into a 200/60/60 subset; stock YOLO11n baseline, the efficiency-frontier model, and a rank-8 LoRA adapter under one schedule. Pilot numbers only — the release is 5,604 images |

## Experiment matrix

Configured and generated (never hand-edited): EXP-001…019 (ladder), EXP-012
(multi-seed), EXP-020…028 (robustness / generalisation), EXP-211…338 (slot studies:
attention, fusion, speckle, enhancement, target prior, frequency, input adapter,
consistency, conditioning). All runnable definitions; zero completed runs.

## The critical path (what unblocks what)

```text
research_gap full-text reads        (no GPU; 1–2 days)
        ↓
dataset acquired + audited          (SARDet-100K or compatible; Week 1 per spec)
        ↓
EXP-004 baseline + LOSO folds       ← FIRST MEASURED NUMBER (proves the failure)
        ↓
representation probe                ← proves the hypothesis (sensor entanglement)
        ↓
conditioning arms EXP-331…338       ← the headline comparison
        ↓
invariant branch (only if probe supports it)
        ↓
ablations / RT-DETR transfer / multi-seed
```

The two **missing** items on the critical path that do not need a GPU:
1. ~~RGB-pretraining baseline arm~~ — **built**: `init:` config key + EXP-401…403 arms (verified transfer counts).
2. ~~Representation-diagnosis tooling~~ — **built**: `saryolo.evaluation.probes` + the `probe` CLI.

Latest additions (this session):
3. ~~Multi-seed aggregation~~ — **built**: `build_multi_seed` now deduplicates restarted seed
   runs (one row per `train_seed`, latest wins), reports sample std **and best/worst**
   (SARVO Phase-40 contract), and keeps per-seed values visible even when only one seed has
   finished (mean/std still TBD — one run is not validation). 12 new tests in
   `tests/test_tables.py` pin the aggregation; the paper-tables module had no tests before.
4. ~~Red-team review~~ — **built**: `reports/red_team_review.md` attacks the design while
   every measurement-dependent verdict is marked `BLOCKED-ON-RUN` (no numbers exist yet).
5. ~~Paper-structure skeleton (SARVO Phase 55)~~ — **built**: `reports/paper/` holds the
   section skeletons (abstract → limitations), each with its evidence source and its blocked
   status; the README gained the Part X paper-assembly section.
6. ~~Efficiency frontier (SARVO: beat the field on cost)~~ — **built, cost measured**:
   `SARVO-Lite` is full v2 minus its two dominant compute slots (adaptive multi-scale
   fusion and context aggregation, ~42% of the compute), keeping every physical prior.
   Measured at scale `s`: 11.016 M / 32.55 GFLOPs versus full v2's 16.230 M / 55.68
   GFLOPs, with the conditioning adapter adding +0.7% params and no measurable compute.
   `saryolo/evaluation/reported_baselines.py` holds the *published* comparison points
   (AC-YOLO, RLE-YOLO, Edge-optimized lightweight YOLO, SARLite) with venue, URL and
   explicit caveats, in a namespace that can never satisfy a measured cell; the new
   `efficiency_frontier` paper table keeps the two kinds of number in separate sections.
   `EXP-501…505` are the runnable configs; accuracy stays `TBD`. 13 tests in
   `tests/test_efficiency_frontier.py` pin the arms and the honesty rules.

7. ~~Parameter-efficient adaptation baseline (Direction B / Stage 10)~~ — **built and
   measured**: `saryolo/training/peft.py` (rank-`r` updates to the wrapped convolutions,
   `B` zero-initialised so the wrapped layer is bit-identical to the base at step 0, base
   weights frozen, merge/absorb for inference) wired into the runner through the trainer that
   actually trains. Two failures were found by measurement rather than by reasoning: an adapter
   applied to the *facade* is discarded when Ultralytics rebuilds the model from its config
   (the ledger then described a parameter-efficient arm that was a full fine-tune — caught
   because its mAP was bit-identical to the baseline's), and an adapter left in the checkpoint
   as a wrapper cannot be loaded at all, because the loader fuses convolutions with the
   following BatchNorm (`'LoRALayer' object has no attribute 'weight'`). Both are fixed and
   pinned by `tests/test_peft.py`, including an arm trained end to end on the synthetic smoke
   data that asserts, from the checkpoint file, that the adapter influenced it and that the
   file loads as an ordinary detector. EXP-601…605 are the runnable arms (reference + ranks
   4/8/16 + frozen-base).

8. ~~Data-efficiency sweep (Direction E)~~ — **built**; `configs/exp/EXP-701…705` declare a
   training fraction of 1/5/10/25/50 %. It began as a private `data_fraction` override handled
   by this repository's trainer, which could not run the *baseline* arm at all: a stock model
   YAML resolves to the plain `YOLO` facade, so Ultralytics' trainer received an argument it
   does not know and refused it. It is now the library's own `fraction`, which sorts the file
   list, takes a prefix (so the arms are nested) and applies to the training split only — the
   nesting is pinned by test, since a library change would silently turn the sweep into a
   comparison of independent draws.

9. ~~Real-data pilot on HRSID~~ — **measured** (`REAL-001…003`): a subset of the official
   release (200/60/60), 40 epochs at 320 px, CPU only, one class, three arms sharing the
   schedule — stock YOLO11n, SARVO-Lite (s), and LoRA r=8. The subset was assembled by
   `scripts/fetch_hrsid_subset.py` (resumable, retrying, and explicit about the mirror's
   `valid` vs this repository's `val` split name). Every number is in the ledger, the README
   table is guarded against `docs/assets/facts.json`, and the qualitative panel comes from
   `scripts/make_real_figures.py`. What this is *not*: the full release, a multi-source test, or
   anything about cross-sensor generalisation.

10. ~~Architecture decision and prototype (master Phases 3 + 5)~~ — **decided and implemented**:
   `docs/architecture_proposals.md` offers three genuinely distinct directions — a
   radar-statistic input representation, an anisotropic weight-tied recursive scale-space
   path, and a self-estimated acquisition state — each with its nine required fields
   including a stated failure mode, then recommends the **first** and gives the reason that
   is not "it is the most novel". It is that the first is the only one of the three whose
   central claim can be measured *and falsified* on the hardware that exists: the second
   needs resolution-keyed real data (HRSID has none) and the third needs leave-one-source-out
   over multiple sources. `docs/related_work.md` is the companion Phase-2 comparison table.
   The uncomfortable finding this exercise produced is recorded in both documents: the
   current v2 is a module ladder on a YOLO11 skeleton, which is the outcome the brief
   explicitly rules out.
   Phase 5 followed: `saryolo/nn/modules/cfar.py` implements the recommended first
   representation (analytic multi-scale `log`-ratio + coefficient-of-variation statistics,
   zero-initialised per-pixel gain, exact identity at init), with arms `cfar_n`/`cfar_s` and
   the fixed-threshold control `cfar_fixed_n`/`cfar_fixed_s` and the matched-cost control
   `cfar_conv_n`/`cfar_conv_s`. `tests/test_cfar_frontend.py` (16 tests) pins forward, the
   exact identity, the non-zero gain gradient (including the subtler point that the first
   layer legitimately starts at zero gradient and must recover it), the real loss, an
   optimiser step, the mode/window validation, builder↔module mode parity, and the
   **measured** cost: +217 parameters (+0.008 % at scale `n`), the fixed control **+0**, the
   conv control parameter-identical by construction, and +0.15 GFLOPs at 320 px — a real
   compute price, stated rather than hidden. The honesty note that had to be written into
   the proposal: the repository already had a CFAR-style arm (`tp_cfar`, a non-learned prior
   on *features*), so the contribution is placement and scope, not the statistic.
   Phases 6–8 then ran the falsification tests on the same HRSID subset. The prototype
   (REAL-004) beats its fixed-threshold control (REAL-005) clearly, and that control is
   *worse than the baseline* (0.495 vs 0.571 mAP50) — so the learned gain matters and the
   statistic alone harms. The seed-0 primary comparison against the baseline is a near-wash
   on mAP50 (0.5691 vs 0.5706) with mAP50:95 up 0.3012 → 0.3151, inside the noise of 60 test
   images. So two more tests were run: the **matched-cost conv control** (REAL-006,
   parameter-identical to REAL-004) reaches mAP50:95 0.3021, *below* the prototype's 0.3151
   and only ~0.001 above the baseline — so the gain is not the 217 parameters. And a
   **three-seed paired repeat** (MSEED-B1/B2 vs MSEED-C1/C2) puts the prototype ahead of the
   baseline on mAP50:95 at **all three seeds** (+0.0139, +0.0232, +0.0165; mean +0.0179) with
   a *smaller* seed spread (0.0066 vs 0.0106). A five-corruption sweep (`saryolo robustness`)
   is directionally consistent: under speckle and low contrast the prototype's relative
   mAP50:95 loss is roughly half the baseline's (−10.7 % vs −15.6 %, −13.1 % vs −32.0 %). The
   front end still costs ~40 % of CPU throughput. Recorded as: **survives both falsifiers it
   named, on a 260-image single-machine pilot, with a small absolute effect** — not a result.
   The gain-collapse falsifier was then built and run (`saryolo/evaluation/gain.py`, CLI
   `saryolo gain`, tests in `tests/test_cfar_scope.py`): the trained gain is a genuine
   per-pixel decision (within-image std 0.0331 vs between-image std 0.0105, 98 % of pixels
   active), so it is **cleared**. A synthetic acquisition-shift pilot on a global radiometric
   gain (a new `brightness` corruption) is a **null/mixed result** and is reported as one: at
   severe darkening both arms hit the floor, and at gain 0.3 the prototype is worse. Phase 9
   is finalised at pilot scope in §9 of the proposals document, which freezes the interface,
   labels every claim final/preliminary/not-supported, and keeps the repository from being
   renamed into a detector the evidence does not support.

11. ~~RT-DETR feasibility arm (red-team W6)~~ — **built, not measured**:
   `SARYOLORTDetectionModel` + generator-derived `configs/models/rtdetr/rtdetr_s_cond_film.yaml`
   (`scripts/make_rtdetr_variant.py`, `--check` guards drift). Proven end to end through the
   real vocabulary path: adapter in graph, decoder consumes conditioned features, held-out
   sensor → unknown row, bit-identical to stock at init. The training-loop integration and
   any LOSO number remain open — feasibility ≠ result.

What remains on the critical path needs a GPU: run `docs/RUNBOOK_SSDD.md` end to end. Every
CPU-side prerequisite is now in place.

## Non-negotiables carried from the master spec

Never fabricate results; never claim SOTA without fair comparison; never random-split
for the generalisation claim (the LOSO guard refuses it); never hide negative results;
parameter-matched controls for every claim (the slot machinery enforces this
structurally — capacity-matched arms are verified bit-identical or strictly-smaller
by test).
