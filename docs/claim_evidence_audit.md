# Claim → evidence audit

**Date:** 2026-10-04
**Purpose:** every statement this repository makes should be traceable to a measurement
or to a test. This document lists each claim, states honestly what stands behind it, and
labels it **final**, **preliminary**, **not supported**, or **blocked**. It is the
document a reviewer should be able to read in place of the whole repository and know
exactly how far to trust it.

The rules it enforces are the project's non-negotiables: no fabricated numbers; a claim
without a measured source is a `TBD`, not a guess; negative results stay in; and a pilot
on a subset is never presented as the benchmark result.

## 1. What stands behind each claim

| # | Claim | Label | Evidence (artifact / test) |
| --- | --- | --- | --- |
| 1 | The test suite passes and needs neither a GPU nor a download | **final** | `pytest -q` → 637 passed; `reports/reproduction_status.md` |
| 2 | The architecture vocabulary is wired and self-consistent | **final** | 110 variants; `tests/test_arch.py` (64 tests) |
| 3 | Every custom module is an **exact** identity at initialisation | **final** | `docs/assets/facts.json` → `identity`, all `max|f(x)−x| = 0.0e+00`; `tests/test_arch.py` |
| 4 | No module is frozen at init (identity comes from the gate, not a dead branch) | **final** | `tests/test_arch.py::test_no_module_is_frozen_at_init` |
| 5 | The CFAR front end is an exact identity at init and its gain trains | **final** | `tests/test_cfar_frontend.py` (16 tests) |
| 6 | The CFAR front end costs +217 parameters (< 0.5 %) and is scale-independent | **final** | `tests/test_cfar_scope.py`; `saryolo/evaluation/efficiency.py` |
| 7 | The log-ratio statistic is invariant to a global radiometric gain | **final** | `tests/test_cfar_scope.py::test_the_log_ratio_channel_is_invariant_to_a_global_radiometric_gain` |
| 8 | The matched-cost control is parameter-identical to the prototype | **final** | `tests/test_cfar_scope.py`; efficiency profile |
| 9 | The pipeline learns on **real** SAR imagery (HRSID subset) | **measured, pilot** | ledger `REAL-001`; `reports/reproduction_status.md` §3 |
| 10 | The prototype beats its fixed-threshold and matched-cost controls | **preliminary** | ledger `REAL-004` vs `REAL-005`, `REAL-006` |
| 11 | The prototype leads the baseline on mAP50:95 at three seeds | **preliminary** | `REAL-001`/`REAL-004`, `MSEED-B1/B2`, `MSEED-C1/C2`; mean +0.0179 |
| 12 | The trained gain is a per-pixel decision, not collapsed | **final, but seed-sensitive** | `results/gain/*/gain.json`; magnitude 0.113 / 0.021 / 0.025 across seeds |
| 13 | Training under the SAR degradation model widens the prototype's lead | **not supported** | `AUG-001…004`: the gap shrinks monotonically (+0.0139 → +0.0067 → +0.0016 mAP50:95) |
| 13b | SAR-appearance augmentation improves the detector, and more helps | **measured, pilot** | `AUG-001`/`AUG-003` baseline mAP50:95 0.3012 → 0.3571 → 0.3898 (+29.4 % relative) |
| 14 | The prototype improves robustness under an acquisition shift | **not supported** | radiometric-gain and anisotropic-resolution pilots are both null (`docs/architecture_proposals.md` §8) |
| 15 | Cross-sensor / cross-resolution generalisation | **blocked** | needs the full release + a second source; no GPU here |
| 16 | Any number for the full-release ladder, LOSO folds, tuned LoRA sweep | **blocked / `TBD`** | unmeasured; the ledger renders `TBD` |
| 17 | The adaptive SSAC allocation beats a parameter-identical fixed-computation control | **withdrawn — not supported at this scale** | `SSAC-001` vs `SSAC-002` at three seeds: +0.0063 / +0.0645 / −0.0050 mAP50:95, mean +0.0220 ± 0.0373, smaller than the proposal's own spread; `paper/RESULTS.md` §8.7 |
| 18 | The multi-scale SAR statistic is a better assessment signal than the raw feature | **withdrawn — not resolvable at this scale** | `SSAC-003`'s 0.0226 gap sits inside the proposal's 0.0290 seed spread and was not re-run; `paper/RESULTS.md` §8.8 |
| 19 | Sparse execution turns the allocation into a wall-clock saving | **preliminary — measured positive at 640 px, still scale-bound** | CLI routing sweep: at 640 px the sparse range clears the dense range at `keep ≤ 0.5` (**−8.7 / −12.2 / −13.5 %** at 0.5 / 0.25 / 0.1; `keep = 1.0` costs **+10.1 %**). At 320 px the ~5 % gap at `keep = 0.1` sits inside the dense run's own spread, so **no saving is claimed there**. One checkpoint, CPU, batch 4; `paper/RESULTS.md` §8.5 |
| 20 | The sparsity penalty pushes the allocation sparse (the premise sparse execution needs) | **preliminary** | `SSAC-006` allocation diagnostic: P4 mean gate 0.670 → 0.236; its accuracy difference is explicitly **not** claimed as a gain |
| 21 | The SSAC block is an exact identity at init and its sparse execution matches dense at `keep = 1.0` | **final** | `tests/test_ssac.py` (46 tests); bit-for-bit equality at `keep = 1.0`, no extra parameters |

## 2. The augmentation pilot

To test whether the front end's (small) edge grows when the model is trained under the
same SAR degradation model the robustness benchmark uses, two arms were trained on the
SAR-augmented HRSID subset and evaluated on the **clean** test split:

* `AUG-001` — baseline (stock YOLO11n) on the augmented train split.
* `AUG-002` — the CFAR front end on the same augmented train split.

The augmented split adds one corruption draw per training image (speckle, low contrast,
blur, low resolution, low SNR; `datasets/processed/hrsid_real_aug`, draw recorded in its
`manifest.json`). The comparison that isolates augmentation is **within** each context —
`AUG-002 − AUG-001` versus `REAL-004 − REAL-001` — because both arms in a context share
the same data.

**Measured outcome.** Augmentation is a large win for the detector, and it keeps helping as it
strengthens: the baseline rises **0.3012 → 0.3571 → 0.3898** mAP50:95 for zero, one and two
augmented views (**+29.4 % relative**), the largest movement measured in the repository. But it
is a *negative* for the hypothesis: the prototype's lead over the baseline **shrinks
monotonically**, **+0.0139 → +0.0067 → +0.0016**. Training under the SAR degradation model gives
both arms much of the robustness the analytic statistic provided, so the front end's *relative*
contribution falls to nothing at two views. Full numbers in `paper/RESULTS.md` §3.

## 3. What is deliberately *not* claimed

* **Not SOTA.** No comparison to a published SAR detector is run; `docs/baseline_comparison.md`
  marks every such row "reported-only, cited, caveated".
* **Not a new detector.** The direction is a described architecture thesis on a YOLO
  backbone, kept under its honest name (`docs/architecture_proposals.md` §9.4). The repo
  is not renamed into a detector the evidence does not yet support.
* **Not validated.** Every accuracy number is a 200/60/60 subset, one CPU, and the
  primary comparison is three seeds — a noise check, not validation.
* **Not free.** The front end costs ~40 % of CPU throughput at this resolution; that is
  stated with the numbers, not hidden.
