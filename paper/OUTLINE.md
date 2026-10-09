# SARVO — paper outline (CVPR 2027 draft)

**Working title:** *Scatter-Selective Adaptive Computation for SAR Object Detection*

This is an outline, not a manuscript. It fixes the paper's single mechanism, states the
claim the pilot has already **withdrawn**, and lists every cell that has to be measured
before the story can be written. It exists so the writing describes what was measured
rather than what was hoped for — the same rule the code and the ledger already follow
(`../docs/claim_evidence_audit.md`).

The full pilot record is in [`RESULTS.md`](RESULTS.md) §8, the mechanism specification in
[`../docs/ssac_design.md`](../docs/ssac_design.md), and the novelty/overlap audit in
[`../docs/ssac_assessment.md`](../docs/ssac_assessment.md). Nothing here supersedes those;
where this outline and a measurement disagree, the measurement wins and this file is wrong.

---

## 1. The one-sentence thesis

> In SAR imagery the locations that decide a detection are a small, predictable minority,
> so a detector can spend its expensive feature processing **there** and leave the rest on a
> cheap path — a mechanism whose value is *not* assumed to be accuracy, and whose efficiency
> is reported against the input scale it was measured at.

Everything else in the paper is either the mechanism, a control for it, or a measurement of
what it did. If the mechanism does not clear its controls at the paper's scale, the paper is
about the *measurement*: what selective computation does and does not buy on SAR, honestly
labelled. That fallback is stated here on purpose so it cannot be quietly dropped later.

## 2. Contributions, each with its current label

Labels follow the repository's vocabulary: **final**, **preliminary**, **controlled**,
**withdrawn**, **not supported**, **blocked**.

| # | Contribution | Label today | Decided by |
| ---: | --- | --- | --- |
| C1 | **SSAC**: a per-region allocation of expensive feature processing driven by a *SAR-native* analytic statistic (multi-scale CFAR log-ratio + local coefficient of variation), with a **matched fixed-computation control** of exactly equal parameter count | mechanism: **final**; its *accuracy* effect: **withdrawn at pilot scale** | `SSAC-001` vs `SSAC-002` at three seeds (§8.7) |
| C2 | A **sparse execution** of the same mechanism (tile gather/scatter, kernel-derived halo) that is parameter-identical to the dense form and reproduces it **bit-for-bit at `keep = 1.0`** in eval | **controlled** | `keep = 1.0` equality; `tests/test_ssac.py` |
| C3 | A **wall-clock measurement** of selective computation that shows where the saving appears and why the pilot's input scale is what decides it | **preliminary at 640 px** (measured positive, −13.5 % at `keep = 0.1`), **not separable at 320 px**; the ceiling is **measured** as input-scale-bound | CLI routing sweep at two scales (§8.5) |
| C4 | A **cost ablation** isolating how much of the mechanism's parameter price the accuracy needs (`expand = 2 → 1` loses the whole benefit) | **controlled** (seed 0) | `SSAC-005` (§8.6) |
| C5 | The **sparsity premise**: a penalty can push the learned allocation sparse at no measurable accuracy cost, so a sparse build has something to skip | **preliminary** (seed 0) | `SSAC-006` allocation diagnostic (§8.6) |

**What this paper must *not* claim.** The seed-0 comparison *looked* like evidence that
adaptive allocation beats a parameter-identical fixed control. It does not survive three
seeds and is **withdrawn** (C1, §8.7). The paper's contribution is therefore the *mechanism
and its measurement protocol*, not a headline accuracy gain — and the abstract must say so.

## 3. Section-by-section plan

### 1. Introduction
- SAR detection spends uniform compute on scenes that are mostly predictable background
  (open sea, flat field) around sparse, weak returns.
- State the hypothesis and, up front, that the pilot **withdraws** the accuracy claim and
  finds a wall-clock saving only at the larger input scale, not at its training scale.
  Honesty here is the paper's spine.
- List C1–C5 with their labels.
- **Figure 1 (manual):** where the compute goes vs where the targets are.

### 2. Related work
- Dynamic / selective computation: SACT, SplatNet, region-proposal-gated compute, and
  test-time adaptive networks — the *principle is not novel* and is not claimed as such
  (`../docs/ssac_assessment.md`).
- SAR-specific detectors and efficiency work (AC-YOLO and peers, `../docs/related_work.md`).
- The gap this paper occupies: a selective-computation mechanism whose **assessment signal
  is a SAR statistic** and whose **efficiency is measured on real SAR inputs at a named
  scale**, with a parameter-identical control.

### 3. Method — SSAC
- Formulation as in `../docs/ssac_design.md`: cheap shared path → analytic statistic →
  tiny learned scorer → expensive bottleneck path → gated residual mix (`alpha = 0` at
  init ⇒ exact identity, so any measured change is the learned allocation).
- **Dense vs sparse execution** as two readings of one mechanism: dense changes *values*,
  sparse changes *arithmetic*. Sparse is parameter-identical (no new parameters when the
  execution switches).
- **Halo** derived from the expensive path's own kernels, not hard-coded — the property that
  makes `keep = 1.0` sparse == dense checkable.
- **Figure 2 (manual):** block diagram; **Figure 3:** dense vs sparse data flow with tiles
  and halo.

### 4. Controls and protocol
- The three arms: proposal (`adaptive`), **matched fixed-computation control** (`fixed`,
  equal parameters, allocation made spatially constant), and **assessment alternative**
  (`adaptive_raw`, raw feature instead of the statistic).
- Pre-registered withdrawal conditions (`../docs/ssac_design.md` §3.2): "does not beat the
  control", "small-object penalty", "noise-dominated". State which fired.
- Seed protocol: three seeds for every accuracy comparison; a *pilot* is not validation.
- No-fabrication protocol: `TBD` for unmeasured cells; the ledger is append-only.

### 5. Experiments
- **5.1 Accuracy vs the controls (three seeds).** Report the withdrawn claim as withdrawn:
  paired differences +0.0063 / +0.0645 / −0.0050, mean **+0.0220 ± 0.0373** over three seeds
  (Table: §8.7). *The seed-0 table stays in the paper only to show what the withdrawal is
  about.*
- **5.2 Small-object preservation.** AP_small / AP_medium, own COCO evaluator; the third
  withdrawal condition did **not** fire (§8.2). Not seed-checked ⇒ preliminary.
- **5.3 Is the allocation spatial?** Within-image vs between-image spread of the gate; the
  control's within-image spread is exactly zero (§8.3). This is what separates the
  mechanism from a per-image difficulty scalar.
- **5.4 The sparsity premise.** `SSAC-006` allocation diagnostic (§8.6).
- **5.5 Efficiency on real inputs.** Dense vs sparse on one checkpoint, real val images, at
  **two named scales** (§8.5, `docs/assets/ssac_execution.svg`). At 320 px: routing overhead
  at `keep = 1.0` (**+29.5 %**) and no saving outside the dense run's spread. At 640 px:
  routing overhead falls to **+10.1 %**, and sparse execution clears the dense range at
  `keep ≤ 0.5` (**−8.7 / −12.2 / −13.5 %**). The **scale ceiling** is the finding — P5 is a
  single 16 px tile at 320 px and four tiles at 640 px.
- **5.6 Cost ablation.** `expand = 2 → 1` (§8.6).

### 6. Failure analysis
- The two ways the mechanism did not deliver: no accuracy edge that survives seeds; no
  wall-clock saving at the pilot's *training* scale (320 px) — the saving appears only at
  640 px, reported as the pre-registered test rather than hidden.
- What the FLOP counters **cannot** see (sparse reports *more* GFLOPs) and why the claim
  rests on measured wall-clock.
- Measured boundaries: allocation saturates at P4/P5 (fraction > 0.5 is 1.0 there), so the
  deep levels save nothing at 320 px; at 640 px they become routable and the saving appears.

### 7. Limitations (stated, not implied)
- One dataset (HRSID), one subset (200/60/60), one class, one machine, **no GPU**.
- Three seeds is a noise check, not validation.
- No cross-sensor / cross-resolution result.
- The wall-clock conclusion is a fact about **320 px and 640 px**, this tiling, and
  Python-level routing. The 640 px saving is one checkpoint, CPU-only, batch 4, with
  *evaluation-time* budgets — not evidence about selective computation in general.

### 8. Conclusion
- What was built, what was controlled, what was withdrawn, and the one measurement that
  would change the verdict (see §5 below).

## 4. The claim ledger this paper must reproduce

The abstract, the contribution list and the results tables must agree with
`paper/RESULTS.md` §8.8 and `../docs/claim_evidence_audit.md` rows 17–21 at submission time.
If any row there changes, this outline and the abstract are stale.

## 5. What has to be measured before this is a CVPR paper (all `blocked-on-run`)

| Need | Why it is not optional | Status |
| --- | --- | --- |
| **Larger input scale** (e.g. `imgsz ≈ 640`, where the deepest governed level is > 1 tile) | The efficiency verdict is *input-scale-bound*; testing the ceiling is the contribution | **done** — the 640 px sweep is measured (§8.5); still one checkpoint, CPU-only |
| **More training data** (full HRSID release, not the 200-image subset) | A 0.006 mAP difference is not resolvable on 60 test images | **blocked** |
| **More seeds / longer schedule** | Current three seeds cannot resolve the effect the pilot looked for | **blocked** |
| **Cross-sensor / cross-resolution folds (LOSO)** | Generalisation is the paper's second axis | **blocked** |
| **Fair baselines** including a fixed-computation SARVO *without* SSAC at the paper's scale | The key control at the paper's budget, not the pilot's | **blocked** |
| **Compute-matched comparison** against a region-selective baseline of the same parameter count | Otherwise adaptivity is confounded with capacity | **blocked** |

Until those exist, the honest artifact is the pilot: a mechanism, its controls, and a
measurement that says what did *not* work. That is what `RESULTS.md` §8 records and what
this outline is built around.

## 6. Repo ⇄ paper mapping (so every number is traceable)

| Paper element | Source of truth |
| --- | --- |
| Sections, equations, data flow | `../docs/ssac_design.md` |
| Novelty / prior-art overlap | `../docs/ssac_assessment.md` |
| Pilot numbers, tables, prose | [`RESULTS.md`](RESULTS.md) §8 |
| Claim labels | `../docs/claim_evidence_audit.md` rows 17–21; [`RESULTS.md`](RESULTS.md) §8.8 |
| Efficiency figures/measurements | `docs/assets/facts.json` (`ssac_execution`); `../docs/assets/ssac_execution.svg` |
| Mechanism tests | `../tests/test_ssac.py`, `../tests/test_efficiency.py` |
| Generated tables | [`tables/`](tables/) via `python -m saryolo assets` (unmeasured cells stay `TBD`) |
| LaTeX skeleton | [`manuscript/main.tex`](manuscript/main.tex) |

## 7. Writing order

1. Freeze §5's scale and re-run the three-seed pair there — the result decides whether C1 is
   a gain or stays a measurement paper.
2. **The efficiency table at 640 px exists (§8.5)**: the ceiling lifts, the saving appears,
   and the deepest level has four tiles. Re-run it on more checkpoints and a GPU host to turn
   the one-checkpoint positive into a claim the paper can lean on.
3. Re-run the cost ablation at the new scale, and train an arm **at** 640 px so an accuracy
   comparison exists at the scale the efficiency number is quoted from.
4. Only then write the abstract and introduction around numbers that exist.
5. Add the limitations paragraph from what the controls actually showed — not from this
   outline.
