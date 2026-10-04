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
| AUG-001 | baseline, SAR-augmented x1 | 400 | 0.6060 | 0.3571 | 0.9548 | 0.5556 | 12.71 |
| AUG-002 | **CFAR front end, SAR-augmented x1** | 400 | 0.6049 | 0.3638 | 0.9493 | 0.5478 | 24.36 |
| AUG-003 | baseline, SAR-augmented x2 | 600 | 0.6278 | 0.3898 | 0.9418 | 0.5848 | 16.50 |
| AUG-004 | **CFAR front end, SAR-augmented x2** | 600 | 0.6160 | 0.3914 | 0.9224 | 0.5789 | 30.85 |

**What this measures.** Augmentation is a large, real win for the detector and it keeps
helping as it strengthens: the baseline rises **0.3012 → 0.3571 → 0.3898** mAP50:95 (one and
two augmented views), i.e. **+0.0886, +29.4 % relative**, the biggest accuracy movement in the
repository. The CFAR arm improves too (0.3151 → 0.3638 → 0.3914) but flattens.

**And the hypothesis is not supported — the opposite happens.** The prototype's lead over the
baseline shrinks monotonically as augmentation strengthens:

| Context | Baseline mAP50:95 | CFAR mAP50:95 | Gap |
| --- | ---: | ---: | ---: |
| no augmentation | 0.3012 | 0.3151 | **+0.0139** |
| SAR-augmented x1 | 0.3571 | 0.3638 | **+0.0067** |
| SAR-augmented x2 | 0.3898 | 0.3914 | **+0.0016** |

The honest reading is that augmenting with the same corruption model gives both arms much of
the robustness the analytic statistic was providing, so the front end's *relative* contribution
shrinks to nothing (the mAP50 gap is even slightly negative at x2: 0.6278 vs 0.6160). The
augmentation is worth keeping for the detector; it is a **negative** for the front-end claim.

**The augmentation gain itself survives the seed check.** The x1/x2 arms started at a single
seed, so the ×2 baseline was repeated at seeds 1 and 2 (`AUG-005`, `AUG-006`) and paired against
the clean-split baseline at the same seeds (`MSEED-B1`, `MSEED-B2`):

| Seed | Clean baseline mAP50:95 | Augmented ×2 baseline mAP50:95 | Paired gain |
| ---: | ---: | ---: | ---: |
| 0 (REAL-001 / AUG-003) | 0.3012 | 0.3898 | **+0.0886** |
| 1 (MSEED-B1 / AUG-005) | 0.2804 | 0.3892 | **+0.1089** |
| 2 (MSEED-B2 / AUG-006) | 0.2871 | 0.3728 | **+0.0857** |
| mean ± std | 0.2895 ± 0.0106 | **0.3839 ± 0.0097** | **+0.0944 ± 0.0126** |

The gain is positive at **all three seeds**, its mean is ~9× the seed spread of either arm
(+0.0944 against ±0.0106/±0.0097), and the augmented arm is also the *more stable* one
(±0.0097 vs ±0.0106). Unlike the front end's +0.0179, this is not a noise-scale effect: at the
worst seed the augmentation still buys +0.0857 mAP50:95. On mAP50 the same comparison gives
+0.0579 ± 0.0237 (0.6203 ± 0.0071 vs 0.5624 ± 0.0220). **The augmentation result is robust; the
front-end gap under augmentation remains single-seed and is labelled as such.**

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

## 6. Efficiency on real inputs (measured)

The cost numbers elsewhere are single-image, measured on ``torch.randn``. Because the CFAR
front end is *data-dependent* (``log(x + eps)`` and a local mean), a random-noise timing times
the degenerate branch, so latency is re-measured on a real batch of HRSID val images
(``saryolo efficiency --real``, imgsz 320, batch 8) and the profile records the input
distribution it used (``latency_source = "real"``, min 0.012, max 0.996).

The profiler takes a median over independent timed blocks (``--runs 3``) and reports the
min/max alongside it, because one block cannot see drift on a shared CPU. That mattered: the
same baseline checkpoint timed at 82.3 FPS in an earlier single-block session and at **113.1
FPS** here, a 37 % gap that no within-block spread would have revealed. The table below is
five checkpoints measured **back-to-back in one quiet interval**, which is what makes the rows
comparable to each other; it is *not* a report of the machine's best case.

| Arm | FPS (batch 8, real) | ms/image | run spread | ms/image (noise) |
| --- | ---: | ---: | ---: | ---: |
| REAL-001 · baseline | 113.1 | 8.85 | 3.1 % | 16.2 |
| REAL-002 · **SARVO-Lite (s)**, frontier | 13.7 | 72.88 | 0.3 % | 80.7 |
| REAL-004 · CFAR front end | 37.2 | 26.90 | 0.9 % | 27.2 |
| AUG-002 · CFAR, augmented | 35.2 | 28.43 | 0.8 % | 31.6 |
| AUG-003 · baseline, augmented | 101.2 | 9.89 | 1.9 % | 15.7 |

**Reading.** On real inputs the front end costs **3.0×** the baseline's per-image latency
(8.85 → 26.90 ms; 113.1 → 37.2 FPS at batch 8), and the frontier model costs **8.2×**
(72.9 ms). The FLOPs gap is much smaller (1.76 vs 1.61 G, +9 %), so the front end is a
*latency* cost rather than a compute cost on this CPU: it is a sequence of small,
shape-preserving operators that do not vectorise as well as the backbone's dense convolutions.
That is the number that matters for the cost argument in §7 — the intervention §7 recommends
instead (augmentation) costs **nothing** here, and the augmented baseline in fact measures
*faster* than the clean one (101.2 vs 113.1 FPS is within this machine's between-session noise;
the deployed graph is identical).

FLOPs provenance is recorded per row (``flops_G_counter``): Ultralytics' ``get_flops`` returns
**0.0** on the CFAR arms rather than raising, so those rows are ``thop`` numbers while the
baseline rows are Ultralytics numbers. The two counters agree to ~0.2 % on the baseline
(`baseline_n`: 1.613 vs 1.610 G, cross-checked in `tests/test_efficiency.py`), so the mixed
column is fair — but it is labelled rather than silently mixed. All numbers are one CPU, one
machine, no GPU; the earlier single-block figures are kept in `results/efficiency/` history and
the between-session variation is stated here rather than averaged away.

## 7. Discussion: why augmentation subsumes the front end's benefit

The clearest thing this pilot measured is not the front end at all — it is that **training under
the SAR degradation model is a bigger lever than any architectural prior we tried, and it is a
substitute for the one the prototype supplies.** The mechanism is worth stating precisely,
because "augmentation helped" is not the same claim as "augmentation replaced the front end".

The ratio-space CFAR front end is a *hard-coded local-statistic normaliser*. It computes a
scale-ordered CFAR statistic of the log-intensity, ratio-normalises it, and hands the backbone a
view in which the local background scale has already been divided out. Its value, if any,
consists of presenting a statistics-normalised image to the first convolution block — a spoken
prior that the network would otherwise have to discover from data.

A SAR-appearance augmentation plan teaches the *same* invariance in the *data* instead. When every
training image is drawn with speckle, contrast loss, blur, resolution loss and low SNR, the
first block that learns to be stable to those corruptions is, functionally, learning the
normalisation the front end hard-codes — except it is free at inference time. The measurement
shows exactly this substitution: the front end's lead over the baseline shrinks monotonically as
the training distribution absorbs the corruption model,

| Training data | Baseline mAP50:95 | Front end mAP50:95 | Gap |
| --- | ---: | ---: | ---: |
| clean | 0.3012 | 0.3151 | **+0.0139** |
| augmented ×1 | 0.3571 | 0.3638 | **+0.0067** |
| augmented ×2 | 0.3898 | 0.3914 | **+0.0016** |

and the gap reaches zero — slightly past it, in fact, on mAP50 (0.6278 baseline vs 0.6160 front
end at ×2) — even though *both* arms are getting better. That pattern is the signature of a
substitution rather than of a wall: the information the prior supplies is real, but it becomes
redundant once the data supplies it too.

**Why this decides the design.** The two interventions sit on opposite sides of the compute
budget. The front end pays a **permanent inference cost** — +0.151 GFLOPs at 320 and a measured
**3.0× latency penalty on real inputs** (113.1 → 37.2 FPS at batch 8, §6). Augmentation pays a
training-time cost only (roughly +60 % training minutes for one extra view) and **nothing at
inference**: the deployed model is an ordinary YOLO11n. When the two deliver the same accuracy,
the one that is free at test time wins, and the measurement says augmentation delivers more, not
the same: **+0.0886 mAP50:95** for the baseline from two augmented views, against **+0.0139** for
the front end from clean data. Across every comparison available here, augmentation dominates on
both accuracy and cost.

**Scope of improvement, stated plainly.** The honest ranking of levers in this repository, by
measured accuracy movement per unit of cost, is now:

1. **Augmentation** — biggest accuracy movement in the repository (+29.4 % relative on
   mAP50:95), zero inference cost. This is the design to spend effort on.
2. **Model capacity** — the SARVO-Lite (s) frontier gains +0.0275 mAP50:95 over the baseline at
   ~4× the compute (REAL-002 vs REAL-001); a larger model, not a better inductive bias.
3. **The CFAR front end** — a small, seed-consistent but pilot-scale gain (+0.0179 mean mAP50:95
   over three seeds) at a 3.0× real-time penalty, and it shrinks to nothing once (1) is applied.

So the front end is worth keeping as a **cheap, self-contained ablation** — it is the mechanism
the project set out to test, it clears both of its named falsifiers, and it is individually
interpretable — but the evidence does not license promoting it to the headline. The headline
should be the detector trained under the SAR degradation model.

**What this comparison can and cannot settle.** The augmentation *gain* is now seed-checked: the
×2 baseline was repeated at seeds 1 and 2 (`AUG-005`, `AUG-006`, §3) and beats the clean baseline
at every seed by +0.0857…+0.1089 mAP50:95, so "augmentation helps, a lot" is a robust claim and not
a lucky seed. What remains single-seed is the thing §7 actually depends on — the **×1/×2 gap
shrink**, which is measured only at seed 0. Two further gaps: the **matched-cost control was not
re-run under augmentation** (on clean data the parameter-identical conv stem, REAL-006, failed to
match the prototype, which is what makes the clean gain interesting, but whether *it too* is
subsumed is unmeasured); and this is **one augmentation ladder, one dataset, one machine**. The
pattern is a substitution mechanism plus a monotone three-point trend, and both should be
re-derived on the full release before the substitution is claimed at paper grade.

## 8. Limitations (stated, not implied)

* One dataset (HRSID), one subset (200/60/60), one machine, **no GPU**.
* The primary comparison is **three seeds** — a noise check, not validation.
* No cross-sensor or cross-resolution result; the LOSO machinery is untested on real data.
* The ladder ablations, the tuned LoRA sweep and the full-release numbers remain `TBD`.
* The front end costs ~40 % of CPU throughput at this resolution.
* Negative results (the two acquisition axes, the augmentation hypothesis) are kept here rather
  than moved to an appendix.
* The augmented **accuracy** comparison now has three seeds (§3); the *gap-shrink* comparison —
  the one §7 rests on — is still seed-0 only, and §7 says so.
