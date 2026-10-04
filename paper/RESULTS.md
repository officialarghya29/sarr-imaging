# Results (draft)

**Status:** draft of the paper's *Results* section, written from `results/experiments.jsonl`
and the JSON artefacts the evaluation runs wrote. Every number here is a **measured** value;
none is estimated, interpolated or carried over from a paper. The label on each block says
what it is worth: **pilot** (a 200/60/60 HRSID subset on a CPU) vs **benchmark** (none yet).
The generated tables in `paper/tables/` are produced by `python -m saryolo assets` and render
an unmeasured cell as `TBD`; this section is the prose around them.

## 0. Protocol

All accuracy numbers below are on a **subset of the official HRSID release** (ship,
horizontal boxes) assembled as **200 / 60 / 60** train/val/test, 40 epochs at 320 px, batch 4,
one class, seed 0 unless stated, **CPU only**. The full release is 5,604 images, so every
number is a **pilot**. Arms that are compared share the dataset, schedule, resolution and seed,
so a row differs from its neighbours only in what is trained. Cost is measured by
`saryolo/evaluation/efficiency.py`; accuracy by the Ultralytics validator through this
repository's `SaryoloTrainer`.

## 1. The architecture comparison (pilot)

| Id | Arm | mAP50 | mAP50:95 | Precision | Recall | Params (M) | GFLOPs@320 | FPS (CPU) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| REAL-001 | stock YOLO11n (baseline) | 0.5706 | 0.3012 | 0.9240 | 0.5380 | 2.590 | 1.613 | 61.8 |
| REAL-002 | SARVO-Lite (s), efficiency frontier | 0.5724 | 0.3287 | 0.8014 | 0.5263 | 11.016 | 8.106 | 12.4 |
| REAL-003 | YOLO11n + rank-8 LoRA | 0.5781 | 0.2984 | 0.8682 | 0.5556 | 2.590 | 1.613 | 65.6 |
| REAL-004 | **ratio-space CFAR front end** (proposed) | 0.5691 | 0.3151 | 0.9104 | 0.5346 | 2.590 | 1.764 | 36.7 |
| REAL-005 | control: same statistic, fixed threshold | 0.4952 | 0.2465 | 0.7931 | 0.4737 | 2.590 | 1.728 | 44.1 |
| REAL-006 | control: matched-cost conv stem | 0.5654 | 0.3021 | 0.9244 | 0.5146 | 2.590 | 1.649 | 44.9 |

**Reading.** The prototype (REAL-004) beats both of its own controls on mAP50:95 — the
matched-cost stem (REAL-006, parameter-*identical* to it) by **+0.0130**, and the
no-parameter fixed-threshold statistic (REAL-005) by **+0.0686**; the fixed-threshold control
is *worse than the plain baseline* (0.2465 vs 0.3012), so the learned gain is load-bearing and
the analytic statistic alone harms. Against the baseline the mAP50:95 gain is **+0.0139** while
mAP50 is flat (−0.0014), and the front end costs ~40 % of CPU throughput. **Pilot; not a
benchmark result.**

## 2. Seed sensitivity (pilot)

The baseline-versus-prototype comparison repeated at three training seeds:

| Seed | Baseline mAP50 / mAP50:95 | CFAR mAP50 / mAP50:95 | Paired Δ mAP50:95 |
| ---: | ---: | ---: | ---: |
| 0 | 0.5706 / 0.3012 | 0.5691 / 0.3151 | +0.0139 |
| 1 | 0.5375 / 0.2804 | 0.5571 / 0.3036 | +0.0232 |
| 2 | 0.5792 / 0.2871 | 0.5838 / 0.3036 | +0.0165 |
| mean ± std | 0.2895 ± 0.0106 | **0.3074 ± 0.0066** | **+0.0179** |

The prototype leads at **all three seeds** and with a smaller seed-to-seed spread. A paired
sign test on three samples is not a significance test, and the magnitudes are small; this says
"consistent", not "significant".

## 3. Training under the SAR degradation model (pilot)

Hypothesis: if the front end's edge comes from the SAR statistic, then training the model under
the same SAR degradation model the robustness benchmark uses should *widen* the edge, because
the model is no longer seeing clean imagery only. The augmented split adds one corruption draw
per training image (speckle, low contrast, blur, low resolution, low SNR) and both arms are
evaluated on the **clean** test split.

| Id | Arm | Train images | mAP50 | mAP50:95 | Precision | Recall | Train (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AUG-001 | baseline, SAR-augmented | 400 | 0.6060 | 0.3571 | 0.9548 | 0.5556 | 12.71 |
| AUG-002 | **CFAR front end, SAR-augmented** | 400 | 0.6049 | 0.3638 | 0.9493 | 0.5478 | 24.36 |

**What this measures.** Augmentation is a large, real win for the detector: the baseline rises
**0.3012 → 0.3571** mAP50:95 (**+0.0559**, +18.6 % relative) and 0.5706 → 0.6060 mAP50. This is
the biggest accuracy movement measured in the repository.

**And the hypothesis is not supported.** Under augmentation the prototype's lead over the
baseline **halves** — from **+0.0139** mAP50:95 without augmentation to **+0.0067** with it.
The honest reading is that augmenting with the same corruption model gives both arms much of
the robustness the analytic statistic was providing, so the front end's *relative* contribution
shrinks even as the absolute detector improves. The augmentation is worth keeping for the
detector; it is not evidence for the front end.

## 4. Robustness and acquisition shifts (pilot)

Five-corruption sweep, mean relative mAP50:95 change from each arm's own clean score:

| Corruption | Baseline | CFAR front end |
| --- | ---: | ---: |
| speckle | −15.6 % | **−10.7 %** |
| low contrast | −32.0 % | **−13.1 %** |
| clutter | −24.6 % | **−20.9 %** |
| blur | +7.4 % | +6.4 % |
| low resolution | +6.3 % | +5.3 % |

The prototype degrades more slowly on the **texture** corruptions. The two pure *acquisition*
axes, however, are **null**:

* **Global radiometric gain.** At severe darkening both arms hit the detection floor; at
  moderate gain the prototype keeps its small lead; at gain 0.3 it is *worse* than the baseline
  (0.109 vs 0.137 mAP50:95). Not discriminating.
* **Anisotropic along-track resolution.** Both arms *improve* (smoothing helps this subset) and
  keep their ~0.02 offset. Not discriminating.

So the measured advantage lives on clutter/texture, **not** on acquisition transfer — which is
the motivation the method is told with, and the gap is reported rather than smoothed over.

## 5. The gain diagnostic (final for the given checkpoints)

The proposal named gain collapse as the way its own claim could be empty. Reading the trained
per-pixel gain directly on the held-out chips (`saryolo gain`):

| Checkpoint | mean abs. gain | within-image std | between-image std | active fraction | collapsed |
| --- | ---: | ---: | ---: | ---: | :---: |
| seed 0 | 0.113 | 0.0331 | 0.0105 | 98.2 % | no |
| seed 1 | 0.021 | 0.0217 | 0.0121 | 6.0 % | no |
| seed 2 | 0.025 | 0.0160 | 0.0131 | 3.7 % | no |

The gain is a per-pixel map at every seed (within-image spread exceeds between-image spread), so
it never collapses — but its magnitude is strongly seed-dependent, which is one reason the
accuracy effect is small.

## 6. Limitations (stated, not implied)

* One dataset (HRSID), one subset (200/60/60), one machine, **no GPU**.
* The primary comparison is **three seeds** — a noise check, not validation.
* No cross-sensor or cross-resolution result; the LOSO machinery is untested on real data.
* The ladder ablations, the tuned LoRA sweep and the full-release numbers remain `TBD`.
* The front end costs ~40 % of CPU throughput at this resolution.
* Negative results (the two acquisition axes, the augmentation hypothesis) are kept here rather
  than moved to an appendix.
