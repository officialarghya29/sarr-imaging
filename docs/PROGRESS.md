# Progress: master workflow → repository status

Mapped on 2026-09-23 against the current master objective (acquisition-conditioned
invariant detection, CVPR 2027, deadline 2026-11-16). The split that matters is
**built** vs **measured**: infrastructure is verifiable without a GPU, results are
not — every accuracy cell in every table is `TBD` until a real training run happens.

## Phase map

| Master phase | Status in this repo | Evidence |
| --- | --- | --- |
| 0. Research landscape | **Started, not closed** | `docs/research_gap.md` — verified entries + flagged full-text reads; the two closest works (SARFormer, Zhang et al. 2026) read and differentiated; novelty claims frozen until the remaining abstract reads |
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
   Largest open engineering gap it records: **RT-DETR transfer is unimplemented** (W6), so
   architecture-generality claims stay out of scope until it lands.

What remains on the critical path needs a GPU: run `docs/RUNBOOK_SSDD.md` end to end. Every
CPU-side prerequisite is now in place.

## Non-negotiables carried from the master spec

Never fabricate results; never claim SOTA without fair comparison; never random-split
for the generalisation claim (the LOSO guard refuses it); never hide negative results;
parameter-matched controls for every claim (the slot machinery enforces this
structurally — capacity-matched arms are verified bit-identical or strictly-smaller
by test).
