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

## 8. The core mechanism (SSAC) — pilot

SARVO's core idea is **Scatter-Selective Adaptive Computation**: allocate expensive
feature processing per region according to image evidence. This section reports the first
controlled test of it. The mechanism, its specification and the three candidate integration
points are in `docs/ssac_design.md`; the literature review that constrained the claim is in
`docs/ssac_assessment.md`. **Every number here is a pilot** — the 200/60/60 HRSID subset, 40
epochs, 320 px, batch 4, seed 0, CPU only, one class.

### 8.1 The comparison, and the control that decides it

The master workflow names the *fixed-computation variant of the model without the adaptive
mechanism* as the key control. `SSAC-002` is exactly that, and it is **parameter-identical**
to the proposal by construction — the same block shapes, with the allocation made spatially
constant. So a difference between `SSAC-001` and `SSAC-002` is the adaptive allocation and
not capacity. `SSAC-003` is Experiment 2: the same allocation fed the raw feature instead of
the multi-scale SAR statistic.

| Id | Arm | mAP50 | mAP50:95 | Precision | Recall | Params | GFLOPs@320 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| REAL-001 | YOLO11n baseline | 0.5706 | 0.3012 | 0.9240 | 0.5380 | 2,590,035 | 1.613 |
| SSAC-001 | proposal: SAR-statistic allocation | 0.5626 | **0.3089** | 0.8628 | 0.5380 | 3,396,329 | 1.998 |
| SSAC-002 | matched fixed-computation control | 0.5655 | 0.3026 | 0.9186 | 0.5280 | 3,396,329 | 1.998 |
| SSAC-003 | assessment alternative: raw-feature scorer | 0.5653 | 0.2863 | 0.8932 | 0.5382 | 3,403,257 | 2.003 |

**What the seed-0 run appeared to settle — and what §8.7 then took back.** At seed 0 the
proposal beats its parameter-identical control by **+0.0063** mAP50:95 and the baseline by
+0.0077, and the control itself is essentially the baseline (0.3026 vs 0.3012), so the refinement
block applied uniformly buys nothing. The raw-feature assessment alternative scores **0.2863**,
*below the baseline* and 0.0226 under the proposal. Read alone, that pair of comparisons says the
adaptivity is doing something the capacity is not, and that the SAR statistic — not the raw
feature — is where the signal is.

**It does not survive the seed check.** Section 8.7 repeats the proposal/control pair at seeds 1
and 2 and finds the paired difference is **+0.0063 / +0.0645 / −0.0050** — positive at two seeds,
negative at one, mean +0.0220 with a standard deviation of 0.0373, and a proposal seed spread
(0.0290) larger than the seed-0 effect. The 0.0226 gap to the raw-feature alternative is inside
that same spread, so the assessment claim is not distinguishable from seed noise either. **Both
claims are withdrawn as unsupported at this scale.** They remain in this section because a
negative that was measured belongs on the record, and because the seed-0 table above is what the
withdrawal is *about*. (mAP50 and precision go the other way — the proposal trades a little
precision for localisation-weighted AP; that is reported, not spun.)

### 8.2 Small-object preservation (Experiment 4)

The mechanism's motivation is that weak returns must not be traded away for compute. The
scale-wise AP from the repository's own COCO evaluator, on the same 60 test images (171 GT
boxes: 100 small, 71 medium, 0 large):

| Arm | AP_small | AP_medium |
| --- | ---: | ---: |
| SSAC-001 proposal | **0.0737** | **0.5262** |
| SSAC-002 matched control | 0.0647 | 0.5096 |

The proposal is ahead at both sizes, so there is **no small-object penalty** — the third
withdrawal condition does not fire. This is consistent with the design: the cheap path runs
for every region and the allocation only *adds* expensive processing, so the failure mode is
wasted compute, never an erased target.

### 8.3 The allocation is spatial, and concentrated where small objects live

Read from the trained checkpoints by running each block's `gain_map` on 16 real val chips:

| Block | level | mean g | within-image std | between-image std | fraction > 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| SSAC-001 · block 0 | P3 (C=64) | 0.483 | **0.0269** | 0.0067 | **0.044** |
| SSAC-001 · block 1 | P4 (C=128) | 0.670 | 0.0383 | 0.0242 | 1.000 |
| SSAC-001 · block 2 | P5 (C=256) | 0.762 | 0.0438 | 0.0950 | 1.000 |
| SSAC-002 · blocks 0–2 | — | 0.478 / 0.697 / 0.760 | **0.0** (constant) | 0.0068 / 0.0752 / 0.0972 | — |

The proposal's within-image spread exceeds its between-image spread at every level, so the
allocation is a genuine *per-region* decision and never collapses to one scalar per scene;
the control's within-image spread is exactly zero, which is the property the control exists to
remove. At P3 only **4.4 %** of locations exceed 0.5 — the allocation is concentrated at the
level that carries small objects, which is where §8.2's small-object gain appears. At P4/P5
the model raised the allocation nearly uniformly (fraction 1.0), so **those levels save no
compute at all**; that is a negative for the efficiency story and is stated as one.

### 8.4 Computational overhead of the dense form (Experiment 5)

The dense implementation runs the expensive path everywhere, so the measured cost *rises*:

| Arm | Params | Δ | GFLOPs@320 | Δ | FPS (CPU) |
| --- | ---: | ---: | ---: | ---: | ---: |
| REAL-001 baseline | 2,590,035 | — | 1.613 | — | 61.8 |
| SSAC-001 / SSAC-002 | 3,396,329 | +31.1 % | 1.998 | +23.9 % | 46.1 / 47.6 |

This is the price of the dense form and nothing more: it says nothing about the sparse form,
which is measured in §8.5.

### 8.5 Does the allocation convert into a wall-clock saving? (measured: no)

Sparse execution was built for exactly this question: 16-pixel tiles, the top `keep` fraction
by mean evidence refined per image, each gathered tile carrying a 2-pixel halo (the receptive
radius of the expensive path). Two properties make the measurement admissible rather than a
timing experiment: the sparse build is **parameter-identical** to the dense one (3,396,329
parameters either way, so the difference is arithmetic and not capacity), and at `keep = 1.0`
sparse execution reproduces dense execution **bit-for-bit** in eval mode, which is what proves
the halo and the gather are right. One trained checkpoint (`SSAC-001`), identical weights,
timed both ways on real val images at 320 px, batch 4, median of 5 blocks per row:

| Execution | `keep` | Expensive path executed (P3 / P4 / P5) | ms / batch of 4 | FPS |
| --- | ---: | --- | ---: | ---: |
| dense | — | 1.00 / 1.00 / 1.00 | 51.251 | 78.05 |
| sparse | 1.0 | 1.00 / 1.00 / 1.00 | 63.890 | 62.61 |
| sparse | 0.5 | 0.87 / 0.78 / 1.00 | 50.731 | 78.85 |
| sparse | 0.25 | 0.52 / 0.39 / 1.00 | 49.165 | 81.36 |
| sparse | 0.1 | 0.17 / 0.39 / 1.00 | 48.547 | 82.39 |

Three findings, in the order they matter.

1. **Routing has a real overhead.** At `keep = 1.0` every tile is still gathered and scattered,
   so the mechanism does strictly more work than the dense form: **+29.5 %** latency for no
   saving at all. Any claim that the routing is free is false at this tile size.
2. **Skipping pays the overhead back and then stops.** Latency falls monotonically with the
   budget and lands *at* the dense level: at `keep = 0.1`, 83 % of the P3 expensive path is
   skipped and the result is 48.5 ms against the dense 51.3 ms — a ~5 % difference whose
   magnitude is inside the dense measurement's own block-to-block spread (50.6–61.1 ms). **No
   wall-clock saving is claimed.**
3. **The ceiling is the input scale, not the budget.** At 320 px the P5 feature map is 10×10,
   which is a *single* tile of 16: no budget can skip anything there, and P4 has four tiles. The
   per-level column shows the executed fraction pinned at 1.00 for P5 at every budget. The
   mechanism's expensive path is concentrated in exactly the levels a 16-pixel tile cannot
   route at this resolution.

A methodological point that the cost columns alone would hide: **FLOP counters cannot see the
sparsity.** They trace dense kernels regardless of which ones execute, so the sparse build
reports **2.293 GFLOPs** against the dense 1.998 G. The efficiency claim therefore rests on
the measured wall-clock, not on a FLOP ratio — and the ledger's `flops_G` column for the sparse
arm must be read as the counter's value for the graph, not as the arithmetic performed.

This is the outcome `docs/ssac_design.md` §4 pre-registered as the expected one on a CPU-only
host, and it matches the latency-aware-dynamics literature. It is a statement about this pilot's
input scale and this implementation (Python-level routing, 16-pixel tiles, `imgsz = 320`), not
evidence that region-selective computation cannot be efficient.

### 8.6 The cost ablation: how much of the +31 % parameter price is load-bearing

The mechanism costs **+31.1 %** parameters over the stock detector (3,396,329 against 2,590,035).
The first ablation narrows the expensive path's bottleneck from `expand=2` to `expand=1` and
leaves everything else — assessment, allocation, placement, schedule, seed — exactly as proposed
(`SSAC-005`, 2,953,257 parameters, +14.0 % over stock, 1.789 GFLOPs).

| Arm | Expensive path | mAP50:95 | AP_small (own evaluator) | Params | Δ vs stock |
| --- | --- | ---: | ---: | ---: | ---: |
| REAL-001 baseline | none | 0.3012 | — | 2,590,035 | — |
| SSAC-005 | **expand=1** | 0.2871 | 0.0849 | 2,953,257 | +14.0 % |
| SSAC-001 proposal | expand=2 | **0.3089** | 0.0737 | 3,396,329 | +31.1 % |

The narrow expensive path does **not** keep half the benefit; it loses all of it, landing
0.0141 *below* the stock baseline while the full-width proposal is 0.0077 above. That is the
answer this ablation was built to get: the extra 443,072 parameters are where the mechanism's
effect lives, not overhead around it. The one place the direction reverses is worth reporting —
`SSAC-005`'s AP_small (0.0849) is *higher* than the proposal's (0.0737), so the narrowest
expensive path favours the smallest objects while losing overall localisation. The cost story
here is not monotone, and the non-monotone part is stated rather than dropped.

**The second ablation asks the opposite question: what happens if the mechanism is *pushed*
sparse?** `SSAC-006` adds a penalty on the mean allocation (`w_ssac_sparsity = 0.01`) to the same
graph, seed and schedule — parameter-identical to the proposal (3,396,329), so the only difference
is the objective. The penalty is deliberately *not* part of the proposal: penalising the
allocation to be small would be circular when sparsity is the mechanism's own claim. As an arm it
measures whether the premise the sparse build relies on actually holds.

| Level | Proposal mean `g` | Penalised mean `g` | Proposal fraction > 0.5 | Penalised fraction > 0.5 |
| --- | ---: | ---: | ---: | ---: |
| P3 (C=64) | 0.483 | **0.382** | 0.044 | **0.000** |
| P4 (C=128) | 0.670 | **0.236** | 1.000 | **0.000** |
| P5 (C=256) | 0.762 | **0.491** | 1.000 | **0.162** |

Measured from the two trained checkpoints on 16 real val chips, the same protocol as §8.3.

The penalty does what it is for: the learned allocation falls at every level, and at the two
shallow levels it stops crossing 0.5 at all, which is what a sparse build needs in order to skip
anything. Its accuracy is **0.3126** against the proposal's 0.3089 and its precision 0.938 against
0.863 — but §8.7 shows this pilot cannot resolve differences of that size, so that is **not**
reported as a gain. It is reported as: the mechanism can be pushed to allocate a third to a half
as much, at no measurable cost in accuracy, which is the premise the sparse build needed and did
not have before this arm. The one other visible change is the *between-image* spread at P5
(0.095 → 0.167), i.e. the penalised model differentiates more between scenes — consistent with
allocating by difficulty rather than uniformly.

### 8.7 The seed check: the pilot's accuracy claim does not survive it

The seed-0 difference between the proposal and its parameter-identical control was +0.0063
mAP50:95, and this repository had already measured that the *baseline's* own seed spread is
0.021 (0.3012 / 0.2804 / 0.2871). Repeating the pair at seeds 1 and 2 is therefore the test that
decides whether the seed-0 comparison means anything — the same standard applied to the
augmentation study in §3. The pairs use the same dataset, schedule, resolution and batch as the
seed-0 pair; the only difference is the training seed.

| Seed | Proposal | Matched control | Paired difference |
| ---: | ---: | ---: | ---: |
| 0 | 0.3089 | 0.3026 | +0.0063 |
| 1 | 0.2801 | 0.2156 | **+0.0645** |
| 2 | 0.2815 | 0.2865 | **−0.0050** |
| mean | 0.2902 | 0.2682 | +0.0220 ± 0.0373 |

Three things are true at once and none of them should be dropped.

1. **The seed-0 gain is not reproduced.** The paired difference changes sign (seed 2 is 0.0050
   *behind* the control), and its standard deviation across three seeds (0.0373) is larger than
   its mean. Withdrawal condition 1 of `docs/ssac_design.md` §3.2 — "does not beat the control"
   — fires at seed 2 and is not decisively avoided on the mean. **The accuracy claim is
   withdrawn as unsupported at this scale.**
2. **The proposal is the more stable arm.** Its seed spread is 0.2801–0.3089 (0.0290) against the
   control's 0.2156–0.3026 (0.0870), because the control *mis-trains* at seed 1 (0.2156, below
   every seed-0 arm in this section). A fixed-computation variant of a mechanism does not
   obviously have to be less reliable than the mechanism; one seed that collapsed is not proof
   that it is, and it is reported as an observation with n = 1 behind it.
3. **The mean is positive and the sign is mostly positive.** +0.0220 with two of three seeds in
   the proposal's favour is not evidence of harm either. What the pilot supports is the honest
   null: **at 200 training images and 60 test images, one seed is not enough to see a 0.006
   difference, and three seeds are not enough to resolve it.** The mechanism is neither confirmed
   nor withdrawn; the experiment that would decide it is §9's scale, not this one.

**The assessment claim goes the same way, without being re-run.** `SSAC-003` (raw-feature
scorer) is 0.0226 below the proposal at seed 0, and the proposal's own seed spread is 0.0290.
A difference smaller than the noise of the arm it is measured on is not evidence that the SAR
statistic is load-bearing; it is a single-seed observation that the seed check has now shown
this pilot cannot resolve. It stays in the table, labelled as what it is.

### 8.8 Claim labels

| Claim | Label | Evidence |
| --- | --- | --- |
| The adaptive allocation beats a parameter-identical fixed-computation control | **withdrawn — not supported at this scale** | three seeds: +0.0063 / +0.0645 / −0.0050, mean +0.0220 ± 0.0373; the seed-0 gain is not reproduced (§8.7) |
| The SAR statistic is a better assessment signal than the raw feature | **withdrawn — not resolvable at this scale** | the seed-0 gap (0.0226) is smaller than the proposal's own seed spread (0.0290); `SSAC-003` was not re-run |
| The mechanism does not harm small-object performance | **preliminary** | AP_small 0.0737 vs 0.0647 at seed 0; not seed-checked |
| How much of the +31 % parameter price the accuracy needs | **controlled** (seed 0) | `SSAC-005` (`expand=1`) loses the whole benefit: 0.2871, below the stock baseline |
| A sparsity penalty can push the allocation sparse at no measurable accuracy cost | **preliminary** (seed 0) | `SSAC-006`: mean `g` 0.670 → 0.236 at P4, fraction > 0.5 → 0.000 at P3/P4, accuracy 0.3126 vs 0.3089 — a difference this pilot cannot resolve, so it is not claimed as a gain (§8.6) |
| SSAC is computationally efficient | **not supported — measured negative** | sparse execution (same weights, parameter-identical) is +29.5 % at keep=1.0 and returns only to the dense latency at keep=0.1; FLOP counters cannot see the sparsity |
| Sparse execution is a faithful implementation of the dense mechanism | **controlled** | keep=1.0 sparse == dense bit-for-bit in eval; halo derived from the expensive path; asserted in `tests/test_ssac.py` |
| SSAC is a novel mechanism | **not claimed** | the principle is SACT/SplatNet/region-selection; see `docs/ssac_assessment.md` |

## 9. Limitations (stated, not implied)

* One dataset (HRSID), one subset (200/60/60), one machine, **no GPU**.
* The primary comparison is **three seeds** — a noise check, not validation.
* No cross-sensor or cross-resolution result; the LOSO machinery is untested on real data.
* The ladder ablations, the tuned LoRA sweep and the full-release numbers remain `TBD`.
* The front end costs ~40 % of CPU throughput at this resolution.
* Negative results (the two acquisition axes, the augmentation hypothesis) are kept here rather
  than moved to an appendix.
* The augmented **accuracy** comparison now has three seeds (§3); the *gap-shrink* comparison —
  the one §7 rests on — is still seed-0 only, and §7 says so.
