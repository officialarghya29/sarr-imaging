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
   active), so it is **cleared** — but only at seed 0 does it modulate hard (mean abs. gain
   0.113); seeds 1 and 2 learn a much weaker gain (0.021, 0.025), so the decision is per-pixel
   at every seed yet seed-sensitive in strength. Two synthetic acquisition-shift pilots are
   **null results** and are reported as such: a global radiometric gain (a new `brightness`
   corruption) separates nothing and is *worse* than the baseline at gain 0.3, and an
   anisotropic along-track-resolution axis (a new `anisotropic` corruption) improves both arms
   equally. The prototype's small advantage therefore lives on the texture corruptions, not on
   pure acquisition transfer. A paper-style methods draft for the front end is in
   `docs/methods_rs_cfar.md`. Phase 9
   is finalised at pilot scope in §9 of the proposals document, which freezes the interface,
   labels every claim final/preliminary/not-supported, and keeps the repository from being
   renamed into a detector the evidence does not support.

11. ~~RT-DETR feasibility arm (red-team W6)~~ — **built, not measured**:
   `SARYOLORTDetectionModel` + generator-derived `configs/models/rtdetr/rtdetr_s_cond_film.yaml`
   (`scripts/make_rtdetr_variant.py`, `--check` guards drift). Proven end to end through the
   real vocabulary path: adapter in graph, decoder consumes conditioned features, held-out
   sensor → unknown row, bit-identical to stock at init. The training-loop integration and
   any LOSO number remain open — feasibility ≠ result.

12. ~~SAR-appearance augmentation pilot (SEC. 5)~~ — **measured** (`AUG-001`, `AUG-002`):
   the augmented HRSID subset (`python -m saryolo augment`, one or two corruption draws per training
   image, clean val/test) trained as a baseline and as the CFAR arm, at one view
   (`AUG-001`/`AUG-002`) and two views (`AUG-003`/`AUG-004`, 600 train images). Augmentation is
   the largest accuracy movement in the repository and keeps helping as it strengthens — the
   baseline goes 0.3012 → 0.3571 → 0.3898 mAP50:95 (+29.4 % relative) — but the hypothesis it
   was run to test (that it *widens* the front end's lead) is **not supported**: the lead
   shrinks monotonically, +0.0139 → +0.0067 → +0.0016, and mAP50 is slightly worse for the arm
   at two views. Recorded honestly as a win for the detector and a negative for the front end;
   see `paper/RESULTS.md` §3. The augmentation gain itself was then seed-checked (`AUG-005`,
   `AUG-006`, the two-view baseline at seeds 1 and 2): the paired gain over the clean-split
   baseline is positive at **all three seeds** (+0.0886, +0.1089, +0.0857; mean **+0.0944 ±
   0.0126**, ~9× the seed spread of either arm), so the recommendation rests on a robust number
   even though the *reason* for it (the gap shrink) is still seed-0 only.

13. ~~Real-input efficiency, and the FLOPs-counter provenance~~ — **measured**. Every cost
   number used to be single-image on `torch.randn`, which times the *degenerate* branch of a
   data-dependent first layer, so the profile now supports `saryolo efficiency --real --data …`
   (timing a real HRSID val batch, recording `latency_source = "real"` and the input range) and
   `--runs N` (median over independent blocks with the min/max spread kept, because one block
   cannot see drift — the same checkpoint timed 37 % faster in a quiet session than a loaded
   one). On real inputs the front end costs **3.0×** the baseline's per-image latency
   (113.1 → 37.2 FPS at batch 8) against only **+9 %** FLOPs, so it is a *latency* cost, not a
   compute cost, and `SARVO-Lite (s)` is 8.2×. `measure_flops` now records *which* counter
   produced each number: Ultralytics' `get_flops` returns 0 for the CFAR arms (not an
   exception), so those rows are `thop`; the two agree to ~0.2 % on a model both can measure,
   which is now a test, so a mixed cost column is labelled rather than silent.

14. ~~Full-repository audit (every component, every subcommand, every config)~~ — **done**. All
   59 package modules import, all 99 model YAMLs parse, all 124 experiment configs and 9 dataset
   configs load, all 19 CLI subcommands respond and the ones that can run locally were executed
   (train/eval/bench/robustness/gain/efficiency/mine-hard/augment/assets/check-data/stats/
   ledger/loso). Four genuine defects were found and fixed:

   * **The hard-example miner scored the wrong split.** `load_yolo_ground_truth` was hard-coded
     to the validation split while `mine-hard` predicts on the train split, so every predicted
     image had no labels and every labelled image had no predictions: the miner ranked images
     by raw spurious-detection count against an empty reference and reported success. The split
     is now a parameter, the miner reads ground truth from the split it predicted on, and a
     prediction with no ground truth raises instead of counting every correct detection as
     spurious.
   * **Two evaluators were being mixed without saying so.** The ledger records Ultralytics'
     training-time validator; `saryolo eval`, `robustness` and `cross-dataset` use this
     repository's own COCO implementation. On the same REAL-001 checkpoint they report 0.5706 /
     **0.3012** and 0.5709 / **0.2901** mAP50:95. That is not an AP bug: the repository's
     evaluator reproduces **pycocotools exactly** on the identical detections (now a test), and
     the gap comes from the two *inference paths* differing in the low-confidence tail (3331 vs
     3502 boxes at conf=0.001). Metrics now carry an `eval_protocol` field so a quoted number
     names its evaluator.
   * **The component-numbering guard was vacuous.** It parsed a table row that never matched, so
     it passed on an empty list and could not fail whatever the README said; it now parses the
     numbered table and refuses an empty one.
   * **Stale counts** in `reports/reproduction_status.md` (92 variants, 18 table files) were
     corrected against the repository (98, 20).

   The README was rebuilt around the project banner, cut from 1,008 lines to a focused document
   that keeps the measured tables, the two guarantees, the component numbering and the
   limitations, and drops the rest to `docs/`.

15. ~~The SARVO core mechanism (SSAC) — literature review, specification, prototype and the
   fixed-computation control~~ — **built and measured**. The master context's initial hypothesis
   was Scatter-Selective Adaptive Computation: allocate expensive feature processing per region
   according to image evidence. The review (`docs/ssac_assessment.md`) found the *principle* is
   old — Spatially Adaptive Computation Time (CVPR 2017), SplatNet, the Dynamic-Network survey's
   "region selection" category, and two-stage detectors all own it — so SSAC was **revised, not
   adopted as stated**: its only defensible novelty is a SAR-native assessment signal plus a
   structural target-preservation guarantee. The specification and the three candidate
   integration points (early adaptive / intermediate refinement / adaptive resolution) are in
   `docs/ssac_design.md`, with intermediate refinement recommended on *falsifiability* grounds.

   Prototype: `saryolo/nn/modules/ssac.py::ScatterSelectiveRefinement` (modes `adaptive`,
   `adaptive_raw`, `fixed`), six arms, and `tests/test_ssac.py` (22 tests) covering the ten
   functional checks the workflow lists plus the two falsifiers. The **matched fixed-computation
   control is parameter-identical by construction** (3,396,329 parameters and 1.998 GFLOPs in
   `ssac_n` and `ssac_fixed_n`; 12,588,713 / 6.87 G in the `s` arms). Three pilot arms ran on the
   same HRSID subset and schedule as `REAL-001`, all recorded in the ledger:

   | Id | Arm | mAP50 | mAP50:95 | AP_small |
   | --- | --- | ---: | ---: | ---: |
   | SSAC-001 | proposal (SAR-statistic allocation) | 0.5626 | **0.3089** | **0.0737** |
   | SSAC-002 | matched fixed-computation control | 0.5655 | 0.3026 | 0.0647 |
   | SSAC-003 | assessment alternative (raw-feature scorer) | 0.5653 | 0.2863 | — |

   The proposal beats its parameter-identical control (**+0.0063** mAP50:95; +0.0077 over the
   `REAL-001` baseline), and the raw-feature alternative lands *below the baseline* — so the SAR
   statistic is load-bearing and the assessment is not decoration. The allocation is spatial and
   concentrated: at P3 only 4.4 % of locations exceed 0.5, with within-image spread above
   between-image spread at every level, while P4/P5 raise the allocation nearly uniformly and
   therefore save nothing. **No efficiency claim is made**: the implementation is dense, the
   sparse variant is specified but not yet built, and on this CPU the measured cost is *higher*
   (1.998 vs 1.613 GFLOPs; 46.1 vs 61.8 FPS). The withdrawal conditions were written before the
   runs and none fired; the effect is small and single-seed (60 test images, one CPU), and it is
   labelled a pilot everywhere.

16. ~~Cost ablation of the core mechanism, and the defect it exposed~~ — **measured**. The three
   questions left open by item 15 were: does the allocation convert into a wall-clock saving,
   how much of the mechanism's +31.1 % parameter price is load-bearing, and does the mechanism
   survive being pushed sparse. All three now have measured answers, and finding them turned up a
   defect that the accuracy runs could never have shown.

   **A defect, found because an arm produced a duplicate.** The sparsity-penalty arm came out
   **bit-for-bit identical** to the unpenalised proposal — every weight equal, all four metrics
   equal. A penalty changes the objective and cannot leave the weights untouched, so the symptom
   pointed at the wiring: `saryolo/training/trainer.py::is_saryolo_yaml` decided between this
   repository's model/trainer and ultralytics' stock ones from a *hand-written tuple of layer
   names*, and that tuple omitted `RatioSpaceCFARFrontEnd` and `ScatterSelectiveRefinement` — the
   repository's own two prototype mechanisms. Every SSAC and CFAR arm therefore trained on
   ultralytics' `DetectionModel` with ultralytics' own loss, so `SARYOLODetectionModel.loss` (the
   SAR-aware criterion, and the new allocation penalty) never ran. The runs converged and reported
   plausible numbers; nothing failed. The list is now derived from
   `saryolo.nn.modules.CUSTOM_MODULES`, the registry that defines what may appear in a YAML, with
   two guards in `tests/test_arch.py` — one pinning the derivation, one checking every one of the
   110 variants against its own architecture dict.

   **Blast radius, measured rather than reasoned.** Of the 68 variants declaring a `sar_loss`
   block, the enumeration finds **two** that the old list missed — the two new penalty arms — so no
   previously recorded number was affected; that check is now a test. The loss identity is pinned
   too: `SARAwareDetectionLoss` with all weights at zero is bit-identical to ultralytics'
   `v8DetectionLoss` on an identical model and batch. And the end-to-end case is measured: the
   seed-0 control was re-trained through the fixed path with `--no-ledger` (a verification may not
   touch a result row), and its weights reproduce the earlier run **bit-for-bit**, with the same
   validator metrics (0.566 / 0.303) — which licenses the SSAC-001/002/003 rows, and the arms
   that were in flight when the fix landed. The penalty arm was re-run, because there the defect
   was not harmless: it was the arm.

   | Id | Arm | mAP50 | mAP50:95 | AP_small | Params | GFLOPs@320 |
   | --- | --- | ---: | ---: | ---: | ---: | ---: |
   | REAL-001 | YOLO11n baseline | 0.5706 | 0.3012 | — | 2,590,035 | 1.613 |
   | SSAC-001 | proposal | 0.5626 | **0.3089** | **0.0737** | 3,396,329 | 1.998 |
   | SSAC-002 | matched fixed-computation control | 0.5655 | 0.3026 | 0.0647 | 3,396,329 | 1.998 |
   | SSAC-004 | proposal, **sparse execution** (keep 0.25) | 0.5570 | 0.2995 | 0.0697 | 3,396,329 | 2.293¹ |
   | SSAC-005 | proposal with `expand=1` | 0.5574 | 0.2871 | 0.0849 | 2,953,257 | 1.789 |

   ¹ The FLOP counter reads the *graph*, so it cannot see the sparsity: it reports **more** work for
   the sparse build than for the dense one (2.293 against 1.998 G) even though 48–83 % of the
   expensive path is skipped. That is why the efficiency question is answered with a wall-clock
   measurement and not a FLOP ratio.

   **The wall-clock answer is no.** One checkpoint, identical weights, timed both ways at 320 px,
   batch 4, median of 5 blocks: dense 51.251 ms; sparse 63.890 ms at `keep = 1.0` (**+29.5 %**, the
   routing's own overhead), 50.731 at 0.5, 49.165 at 0.25, 48.547 at 0.1 — where 83 % of the P3
   expensive path is skipped and the latency has returned to the dense level and stopped. The
   per-level breakdown says why: at this input size the P5 map is 10×10, a **single** 16-pixel
   tile, so no budget can skip anything at the level that carries most of the expensive path. This
   is the outcome `docs/ssac_design.md` §4 pre-registered as expected on CPU, and it is recorded as
   a limit of the pilot's input scale, not as a verdict on region-selective computation.

   **Correctness of the sparse form is proven, not assumed**: at `keep = 1.0` it reproduces dense
   execution bit-for-bit in eval mode, the halo is derived from the expensive path's own kernels,
   and the sparse build is parameter-identical to its dense twin, so a cost difference cannot be
   capacity. The loop, with the rejected rows kept, is `docs/ssac_design.md` §5.

   **Can the allocation be pushed sparse? Yes, and it costs nothing measurable.** `SSAC-006` adds
   a `w_ssac_sparsity` penalty to the identical graph and seed (the penalty is deliberately *not* a
   default, because penalising the allocation to be small is circular when sparsity is the
   mechanism's own claim). Measured on 16 real val chips from the two trained checkpoints, the
   learned allocation falls at every level — mean gate at P4 from **0.670 to 0.236**, at P3 from
   0.483 to 0.382 — and the fraction of locations above 0.5 falls from 0.044 / 1.000 / 1.000 to
   0.000 / 0.000 / 0.162 (P3/P4/P5). Accuracy 0.3126 against the proposal's 0.3089 with precision
   0.938 against 0.863 — **not** reported as a gain, because the seed check below cannot resolve a
   difference that size. It is reported as the premise the sparse build needed: the routing budget
   has headroom, so the mechanism's sparsity is a property the model can be *held* to.

   **And then the seed check withdrew the accuracy claim.** The pair that decides the mechanism —
   the proposal against its parameter-identical control, +0.0063 mAP50:95 at seed 0 — was repeated
   at seeds 1 and 2. The paired differences are **+0.0063 / +0.0645 / −0.0050** (mean **+0.0220 ±
   0.0373**). Its own seed spread (0.2801–0.3089) is larger than the effect, the control mis-trains
   at seed 1 (0.2156), and the sign is not stable. Withdrawal condition 1 of
   `docs/ssac_design.md` §3.2 fires at seed 2, so the accuracy claim is **withdrawn as unsupported
   at this scale** and the same applies to the assessment alternative (a 0.0226 gap inside a 0.0290
   spread, and not re-run). This is the outcome the master workflow's discipline exists to produce:
   a small positive result that did not survive being asked twice. What survives is the
   mechanism's design — spatial allocation, faithful sparse execution, pushable sparsity — and none
   of the positive accuracy numbers.

What remains on the critical path needs a GPU: run `docs/RUNBOOK_SSDD.md` end to end. Every
CPU-side prerequisite is now in place.

## Non-negotiables carried from the master spec

Never fabricate results; never claim SOTA without fair comparison; never random-split
for the generalisation claim (the LOSO guard refuses it); never hide negative results;
parameter-matched controls for every claim (the slot machinery enforces this
structurally — capacity-matched arms are verified bit-identical or strictly-smaller
by test).
