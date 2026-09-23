# Progress: master workflow → repository status

Mapped on 2026-09-23 against the current master objective (acquisition-conditioned
invariant detection, CVPR 2027, deadline 2026-11-16). The split that matters is
**built** vs **measured**: infrastructure is verifiable without a GPU, results are
not — every accuracy cell in every table is `TBD` until a real training run happens.

## Phase map

| Master phase | Status in this repo | Evidence |
| --- | --- | --- |
| 0. Research landscape | **Started, not closed** | `docs/research_gap.md` — verified entries + flagged full-text reads; novelty claims are frozen until those are done |
| 1. Dataset audit | Built (tooling) | `check-data` / `stats` commands; EXP-001…003 configured; refuses oriented-label and empty-dataset traps |
| 1b. Leakage check | Built | Per-fold leakage mode in `loso --leakage`; duplicate detection |
| 2. YOLO11 baseline | Built, **not measured** | Baseline reproduces stock YOLO11 exactly (identity test); EXP-004 configured |
| 3. RGB-pretraining baseline | **Missing** | Nothing in the repo establishes the RGB→SAR fine-tuning arm or the random-init control. Required before the input-adapter claim |
| 4. SAR input adapter | Built | Component 12 (SIA), slot EXP-311…314, identity-at-init verified |
| 5. Prove the failure (cross-sensor) | Protocol built, **not measured** | LOSO folds + refusals (`saryolo.data.groups`); degradation itself still unmeasured — the gating fact for the whole paper |
| 5b. Cross-resolution | Protocol built, **not measured** | `loso --rule resolution` + `metadata` command (Experiment D), documented in README and METHOD |
| 5c. Failure analysis | Partially built | Failure taxonomy + hard-example mining exist; domain-gap table generator is untested against real folds |
| 6. Representation diagnosis | **Missing** | No sensor-probe / CKA / embedding tooling. This is the evidence that the *central hypothesis* (sensor entanglement) is true — needed before the invariant branch is justified |
| 7. Acquisition encoder | Built | Component 33 (CND) + `saryolo.data.metadata`; <0.5% overhead, measured; per-sample property pinned |
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
1. RGB-pretraining baseline arm (Phase 3) — a config + runner concern.
2. Representation-diagnosis tooling (probe classifier on frozen features, CKA) — CPU-verifiable.

## Non-negotiables carried from the master spec

Never fabricate results; never claim SOTA without fair comparison; never random-split
for the generalisation claim (the LOSO guard refuses it); never hide negative results;
parameter-matched controls for every claim (the slot machinery enforces this
structurally — capacity-matched arms are verified bit-identical or strictly-smaller
by test).
