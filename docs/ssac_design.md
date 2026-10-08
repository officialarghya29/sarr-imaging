# SSAC — mechanism specification and the three candidate architecture directions

**Date:** 2026-10-04
**Status:** design + implemented prototype. Phases 2 and 3 of the SARVO master workflow.
**Companion documents:** `docs/ssac_assessment.md` (literature review, novelty table,
recommendation to proceed with SSAC *revised*); `docs/architecture_proposals.md` (the
earlier input-representation direction); `docs/methods_rs_cfar.md` (the sibling mechanism).

Every cost number below is a measurement from `saryolo/evaluation/efficiency.py` on the
generated YAML, not an estimate. Every measurement is labelled with the resolution and
parameter count that produced it.

---

## 1. SSAC stated independently of the detector

SSAC is a **channel-preserving, per-region allocation block**. It is specified here without
reference to a backbone, so it can be read as a mechanism rather than as a layer in a
YOLO graph.

### 1.1 Inputs

| Element | Specification |
| --- | --- |
| Input tensor | `F ∈ ℝ^{B×C×H×W}`, a pre-head feature map for one detection level |
| Channels | any `C`; the block is `C → C` |
| Spatial sizes | any `H, W`; odd sizes are handled by padded pooling |
| Sensor metadata | **none required** (deliberate: the mechanism must work where acquisition fields are absent, e.g. HRSID) |
| Extra state | none; single input, single output (the constraint `parse_model` imposes) |

### 1.2 Region assessment

The assessment is deliberately *analytic-then-learned*, in two parts.

**Analytic statistic** (no parameters, `O(K · HW)` with `K` box filters):

```
m        = mean_c(F)                        # channel mean (SAR pipeline is one measured channel)
L        = log(m + ε)
μ_k      = box_k(L)                         # local mean at scale k ∈ {3, 7} (pixels)
ν_k      = box_k(L²) − μ_k²                 # local variance
r_k      = L − μ_k                           # log-ratio: the CFAR decision variable
c_k      = √ν_k / (|μ_k| + ε)                 # local coefficient of variation (∝ 1/√ENL)
s        = [m, r_1, c_1, …, r_K, c_K]        # dimension 1 + 2K
```

**Learned scorer** (small MLP, the only added assessment parameters):

```
a        = W₂ · SiLU(W₁ s)                   # evidence logit, ℝ^{B×1×H×W},  news = hidden
g        = σ(a / τ)                          # allocation map, ℝ^{B×1×H×W}, in (0, 1)
```

* Learnable: `W₁ (1+2K × hidden)`, `W₂ (hidden × 1)`.
* Non-learnable: the statistic, the window sizes, `ε`, `τ`.
* Spatial granularity: **one decision per pixel** at the feature resolution of the level.
  This is the finest granularity that a dense implementation can express; coarser
  granularity (patches, superpixels) is a candidate ablation.
* Complexity: `O(K·C_window·HW)` for the statistic plus `O(hidden·(1+2K)·HW)` for the
  scorer. Both are *linear in the feature area*, and the statistic is cheaper than one 3×3
  convolution at the same width.

The statistic is the same family the repository already measured on real HRSID chips
(`REAL-004`), which is why the assessment signal is not invented here: it is the one SAR
signal this project has evidence for.

### 1.3 Adaptive computation

```
F_cheap  = DW3×3(F)                          # always computed, shared by every region
F_rich   = [Conv1×1 → DW5×5 → Conv1×1](F_cheap)   # the expensive path, expand factor e
mix      = g ⊙ F_rich + (1 − g) ⊙ F_cheap
out      = F + α · (mix − F)                 # α = tanh(ZeroGate), α = 0 at init
```

* **Allocation rule:** a convex per-region mix between the cheap and the expensive path.
* **Hard, soft or learned:** *soft and learned* in this implementation (a differentiable
  mix), which keeps the graph dense and trainable. A hard top-`p` variant is the sparse
  path (§4) and is not yet built.
* **Shared compute:** `F_cheap` is computed once for all regions — this is the "shared
  computation with selective refinement" requirement, and it is why the block is not merely
  a routing layer.
* **Uncertain regions:** nothing is discarded. Where `g` is low the region keeps its cheap
  representation; where `g` is high it also receives the expensive one. There is no
  threshold that can delete a region.
* **Behaviour when most regions are difficult:** `g → 1` everywhere degenerates to the
  parameter-identical *fixed-computation control*, which is exactly what the control arm
  measures. The mechanism cannot silently become worse than its control; it can only fail
  to beat it.
* **Overhead:** bounded and **measured** (§3), not inferred.

### 1.4 Target preservation

This is the design decision that makes the small-object experiment (Experiment 4) a
property of the architecture rather than a hope:

1. **No region is ever skipped.** The cheap path runs for every location unconditionally;
   the allocation only *adds* expensive processing. The worst case is wasted compute, never
   an erased target.
2. **The rescue channel is structural.** Even at `g = 0`, the region passes through
   `F_cheap`, a learned 3×3 depth-wise transform — it is not the identity and not a
   downsample, so a small return is still transformed, not passed through a bottleneck.
3. **False suppression is detectable.** The allocation map `g` is inspectable
   (`gain_map`), and the diagnostic to run is the *mean allocation inside
   ground-truth boxes versus outside*. An allocation that does not concentrate on targets
   is the failure mode, and it is measured directly.
4. **The experiment that verifies preservation:** small/weak-target recall from the
   scale-wise AP the repo already computes, compared against the fixed-computation control.
   A drop is a withdrawal condition.

### 1.5 Output

A `C → C` feature map at the same spatial resolution, immediately consumed by the detection
head. No change to the head, the loss, the data path, or the trainer — the block is a drop-in
slot, like every other module in `saryolo/nn/modules/`.

### 1.6 Optimization

| Term | Purpose | Behaviour expected |
| --- | --- | --- |
| Detection loss (existing) | the only objective actually used | the allocation is trained *only* by detection; no auxiliary router loss |
| `α` residual gate, zero-init | identity at init, attributable gain | the standard repository contract |

Deliberately **not** added: a routing-sparsity penalty, an allocation-entropy term, a
load-balancing loss. The master workflow warns against adding every possible loss, and each
of these would *cause* the behaviour it is meant to encourage (a sparsity penalty makes `g`
sparse by construction, which would make the "the assessment is informative" claim circular).
If the allocation is informative, detection alone should create it; if it is not, a penalty
would only hide that. A sparsity regulariser is therefore a *later* arm, and only to test
whether an uninformative allocation can be rescued — not part of the proposal.

### 1.7 Tensor-shape and computational flow

```
 F            (B, C,  H,  W)
 ├─ mean_c    (B, 1,  H,  W) ─ log ─┐
 │        box_k                    │  statistics: (B, 1+2K, H, W)
 │        r_k, c_k ────────────────┤
 │                                  ↓
 │                          head  → a  (B, 1, H, W)
 │                                  ↓ σ(a/τ)
 │                                  g  (B, 1, H, W)
 ├─ DW3×3  → F_cheap (B, C, H, W)
 │            └─ 1×1 → DW5×5 → 1×1 → F_rich (B, C, H, W)
 └──────────────────────────────────────────────►
              out = F + α · ( g⊙F_rich + (1−g)⊙F_cheap − F )   (B, C, H, W)
```

### 1.8 Complexity

| Quantity | Expression | Dominated by |
| --- | --- | --- |
| Assessment | `O((K + hidden·(1+2K)) · H W)` | the scorer's 1×1 convolutions |
| Cheap path | `O(9 C H W)` | depth-wise 3×3 |
| Expensive path | `O((C·e·9 + C·e + C·e·C) H W) ≈ O(C²e·HW)` | the 1×1 expand/contract |
| **Dense total** | the sum — **no saving** | the expensive path |
| **Sparse total** (built — the `ssac_sparse` execution; see §5 S1) | `O((9C + ρ·C²e)·HW)` for selected fraction `ρ` | the expensive path, scaled by `ρ` |

The important honest statement: **the dense implementation runs the expensive path
everywhere.** The measured FLOPs rise. The mechanism's efficiency case exists only if a
sparse implementation can actually skip the expensive path — so one was built (§5 S1). It
does skip the expensive path, but at the pilot's 320 px input scale that skip does not turn
into a wall-clock saving, because the levels whose expensive path can be routed are single
tiles at that resolution (§5 S1, `paper/RESULTS.md` §8.5).

---

## 2. Three candidate architecture directions (master Phase 3)

All three integrate SSAC at a different point in the graph. Each is given its flow,
formulation, cost, complexity, advantages, weaknesses, novelty concern and required
experiment.

### Candidate A — Early adaptive processing

* **Flow:** assessment runs on the input image (or the first feature map, P1/2); only
  assessed-important regions are promoted to the full-resolution stem, others take a cheap
  path.
* **Formulation:** `g` computed from image statistics at full input resolution; the
  expensive stem applied on a per-region basis.
* **Cost:** decisions at the largest spatial size — the assessment itself is the most
  expensive of the three.
* **Implementation complexity:** high. A stem that branches per region cannot be expressed
  in the current symbolically-built YAML; it needs a hand-written model or a patched
  parser, which this repository deliberately avoids.
* **Advantages:** largest possible saving (the stem is the most expensive stage per pixel).
* **Weaknesses:** **decides before the model has any semantic context.** SSAC's own
  literature review warns against exactly this (SACT's halting is a late-stage signal).
  It is also the hardest to keep target-preserving, because a small target is least
  visible at the input.
* **Novelty concern:** highest — "input-side adaptive resolution" is SplatNet, and the
  repository's own measured RS-CFAR result (`+0.0139 → +0.0067 → +0.0016` under
  augmentation) already suggests input-side interventions are where augmentation competes
  most directly.
* **Required experiment:** A-vs-baseline recall on small objects at fixed accuracy.

### Candidate B — Intermediate selective refinement  ← **recommended, implemented**

* **Flow:** the cheap path is the neck's normal output; the expensive refinement is added
  per region, immediately pre-head, at each detection level.
* **Formulation:** as §1.3.
* **Cost (measured, one class, 320 px):**

  | Arm | Params | Δ vs baseline | GFLOPs | Δ vs baseline |
  | --- | ---: | ---: | ---: | ---: |
  | `baseline_n` | 2,590,035 | — | 1.613 | — |
  | `ssac_n` (proposal) | 3,396,329 | +31.1 % | 1.998 | +23.9 % |
  | `ssac_fixed_n` (control) | 3,396,329 | **identical** | 1.998 | **identical** |
  | `baseline_s` | 9,428,179 | — | 5.394 | — |
  | `ssac_s` (proposal) | 12,588,713 | +33.5 % | 6.870 | +27.4 % |
  | `ssac_fixed_s` (control) | 12,588,713 | **identical** | 6.870 | **identical** |

  The proposal and the control are **exactly equal** in parameters and dense FLOPs. That is
  the point: the comparison isolates *adaptivity*, so any accuracy difference is the
  allocation and not capacity. The price of the mechanism is real (+31 % params, +24 %
  dense compute at pilot scale) and is stated rather than buried.
* **Implementation complexity:** low. A channel-preserving module in
  `saryolo/nn/modules/ssac.py`, registered like every other; one `ModelSpec` field; six
  generated arms. It does not touch the trainer, loss, or data path.
* **Advantages:** the decision is made where features carry semantics (the neck output),
  which is the regime SACT's own results support; it reuses the per-level slot machinery
  the conditioned/prior/refinement arms already use; the target-preservation guarantee is
  structural (§1.4); and — decisively — **it is cheaply falsifiable on this machine.**
* **Weaknesses:** the dense cost rises, so the efficiency claim is unavailable until a
  sparse implementation exists; the `+31 %` parameter price is heavy and an `expand=1`
  (no bottleneck) arm is needed to show how much of it is load-bearing.
* **Novelty concern:** medium — region selection is old, so the *claim* must be the SAR
  signal (`ssac` vs `ssac_raw`) and the preservation guarantee, never the principle.
* **Required experiment:** the master workflow's Experiment 1 — `ssac` vs `ssac_fixed` —
  plus Experiment 2 (`ssac` vs `ssac_raw`) and Experiment 4 (small-object recall).

### Candidate C — Adaptive feature resolution

* **Flow:** different regions are processed at different spatial resolutions; low-`g`
  regions are downsampled, high-`g` regions kept at full resolution, then recombined.
* **Formulation:** `F_out = combine( upsample(F_low↓), gate ⊙ F_high )`.
* **Cost:** potentially the largest dense saving (sub-linear in area if genuinely sparse).
* **Implementation complexity:** high. Recombination needs resampling per region
  (`grid_sample`/`interpolate`), which on CPU is slow and numerically delicate, and
  position-dependent rescaling introduces sub-pixel localisation error — the worst possible
  error for a detection task whose boxes are a few pixels wide.
* **Advantages:** the only candidate whose *dense* form can reduce arithmetic.
* **Weaknesses:** **artefacting and localisation inconsistency.** The repository already
  measured that a single `grid_sample`-based module (Component 11) is the slowest and most
  numerically careful block it contains; a per-region-resolution scheme multiplies that
  risk.
* **Novelty concern:** highest — this is SplatNet, directly.
* **Required experiment:** localisation error (box-centre shift) and small-object recall
  versus Candidate B.

### Recommendation

**Candidate B, on falsifiability.** It is the only direction whose central claim can be
tested *in this environment*: it is expressible in the existing builder, its control is
parameter-identical by construction, and the decisive accuracy comparison runs in under an
hour per arm on CPU. Candidate A decides before context exists and is the direction the
measured augmentation result already competes against. Candidate C is the most interesting
on paper and the most likely to break localisation on a CPU, and its dense form offers no
clean control. The choice is a bet on a mechanism that can be *withdrawn by evidence*, which
the master workflow explicitly prefers over an unfalsifiable idea.

**Not merged.** The three are alternatives, not a stack. Building B and then A "to look
more complete" would produce exactly the module-ladder outcome the brief forbids.

---

## 3. Prototype status and the minimum validating experiment

### 3.0 The arms this specification has been built out into

| Arm | What it is | Why it exists |
| --- | --- | --- |
| `ssac_n` | the proposal: adaptive allocation, dense execution | Experiment 1 |
| `ssac_fixed_n` | parameter-identical control, allocation spatially constant | the key falsifier |
| `ssac_raw_n` | identical allocation fed the raw feature | Experiment 2 (is the statistic load-bearing?) |
| `ssac_sparse_n` | the proposal with `execution="sparse"` | the efficiency question (§4, §5) |
| `ssac_e1_n` | the expensive path's bottleneck at `expand=1` | how much of the +31.1 % parameter price is load-bearing |
| `ssac_pen_n` | the proposal plus a `w_ssac_sparsity` penalty on the mean allocation | what the mechanism does when *pushed* sparse — the premise a sparse build relies on |

Each arm is a `ModelSpec` in `saryolo/nn/arch.py`, generated into `configs/models/`, and has a
runnable experiment config; the sparse build differs from its dense twin in **no parameter**,
which is what makes the pair a cost comparison rather than a capacity comparison.

### 3.1 What is implemented and verified

`saryolo/nn/modules/ssac.py::ScatterSelectiveRefinement`, modes
`("adaptive", "adaptive_raw", "fixed")` and executions `("dense", "sparse")`, arms
`ssac_{s,n}`, `ssac_raw_{s,n}`, `ssac_fixed_{s,n}`, `ssac_sparse_{s,n}`, `ssac_e1_{s,n}`,
`ssac_pen_{s,n}`. Verified by `tests/test_ssac.py` (46 tests) plus the shared
identity/gradient guards in `tests/test_arch.py`:

| Check (master Phase 4 requirement) | Status |
| --- | --- |
| Model initialisation | pass — all modes build, modes match the builder's literal |
| Forward pass, synthetic input | pass |
| Expected output shape | pass — `C → C`, spatial shape preserved; end-to-end `(B, 4+nc, S)` |
| Gradient flow through the core mechanism | pass — gate non-zero at init; scorer non-zero once `α` moves |
| Loss calculation | pass — real `SARAwareDetectionLoss` on the real graph, finite and positive |
| Optimiser update | pass — the gate moves after a step |
| Numerical stability | pass — flat and all-zero maps stay finite |
| Batch-size and input-size handling | pass — `(1,32)`, `(4,64)`, `(2,40)` |
| Checkpoint save + restore | pass — module and end-to-end model reproduce their output |
| Basic inference | pass — finite `(1, 5, S)` tensor at 320 px |
| Control parity | pass — `adaptive` and `fixed` are parameter-identical |
| Allocation is spatial (proposal) and constant (control) | pass |
| Cost measured, not asserted | pass — profiling on the generated YAML |

### 3.2 The minimum experiment that can validate or withdraw the mechanism

Train three arms on the existing 200/60/60 HRSID subset, one schedule, one seed, CPU:

| Arm | Role |
| --- | --- |
| `REAL-001` (YOLO11n) | reference, already measured |
| `ssac_n` | **the proposal** |
| `ssac_fixed_n` | **the key control** — identical parameters, no adaptivity |

Then, if the proposal survives: `ssac_raw_n` (assessment alternative).

**Withdrawal conditions, stated in advance:**

1. `ssac_n` does not beat `ssac_fixed_n` on mAP50:95 → the benefit was capacity, not
   adaptivity; withdraw the mechanism.
2. `ssac_n` beats `ssac_fixed_n` but not `ssac_raw_n` → the SAR statistic is not the
   useful signal; the mechanism is a generic dynamic network and must be described as one.
3. Small-object recall falls relative to `ssac_fixed_n` → the mechanism contradicts its own
   motivation; withdraw.
4. The allocation does not concentrate inside ground-truth boxes → the assessment is
   uninformative; report the negative.

Any of these outcomes is a reportable result. None of them is a reason to skip the run.

---

## 4. Failure modes, and the honest efficiency position

| Failure mode | How it would show | Mitigation / detection |
| --- | --- | --- |
| **Allocation collapse** | `g` becomes constant across a scene | `gain_map` diagnostic; the `fixed` control *is* the collapsed model, so a collapse makes the proposal identical to the control by construction |
| **Assessment too expensive** | assessment FLOPs approach the expensive path's | measure the assessment share in the efficiency profile |
| **No wall-clock saving on CPU** | dense FLOPs fall (in a sparse build) but latency does not | **measured, and it happened**: sparse execution is +29.5 % latency at `keep = 1.0` (routing overhead) and returns only to the dense latency at `keep = 0.1`, where 83 % of the P3 expensive path is skipped — §5. Identical to the 2022 latency-aware-dynamics caveat and to this repo's own CFAR measurement (3.0× latency for +9 % FLOPs) |
| **Routing cannot bite at small map sizes** | the executed fraction stays 1.0 at the deepest level whatever the budget | **measured**: at 320 px the P5 map is 10×10 — a single 16-pixel tile — so no budget can skip anything there. The limit is the input resolution, not the budget; it is stated as such |
| **The FLOP counter reports the wrong thing** | a sparse build's GFLOPs are not below the dense build's | they are **above** it (2.293 vs 1.998 G), because the counter traces kernels regardless of which execute. This is why the efficiency claim rests on the wall-clock measurement and the ledger's `flops_G` for a sparse arm is read as a counter value, not as arithmetic performed |
| **Uniform difficulty** | the dataset's regions are all equally hard, so there is nothing to allocate | `selected_fraction` near 1 at convergence; the HRSID subset is small and homogeneous, so this is a live risk |
| **Small-object suppression** | scale-wise AP drops | structural (§1.4) plus Experiment 4 |
| **Wall-clock **overhead** even when dense | Python-level routing grows latency | not applicable to this dense build; would apply to the sparse one |

**The efficiency position, stated plainly (now measured).** The dense implementation pays the
full expensive-path cost, which is why the mechanism's first measurement was explicitly an
accuracy measurement. Sparse execution now exists (`execution="sparse"`: 16-pixel tiles, the
top `keep` fraction by mean evidence refined per image, each gathered tile carrying a 2-pixel
halo = the expensive path's own receptive radius), and it is validated rather than asserted:
at `keep = 1.0` it reproduces dense execution **bit-for-bit** in eval mode, and it is
parameter-identical to the dense build (3,396,329 parameters either way, so a cost difference
cannot be capacity).

The measured answer, on one trained checkpoint at 320 px, batch 4, median of 5 timed blocks:

dense 51.251 ms · sparse 63.890 ms at `keep = 1.0` · 50.731 at 0.5 · 49.165 at 0.25 · 48.547 at 0.1.

So the routing costs **+29.5 %** when nothing is skipped, the skipping buys that back as the
budget falls, and it stops at the dense latency — **no wall-clock saving is claimed**, and the
per-level executed fraction (0.17 / 0.39 / 1.00 at `keep = 0.1`) shows why: the P5 map is a
single tile at this resolution. Recorded here so the prototype cannot be quoted as an efficient
method on the strength of its arithmetic alone.

---

## 5. The optimization loop (master Phase 9): problem → change → evidence → verdict

Each row is one deliberate change to the mechanism, with what was predicted *before* the run and
what was measured. Nothing here is a tuning log: a change is kept only when it is measured, and
the rows that were rejected are kept too, because a rejected change is the evidence for the
shape the mechanism has.

All runs: HRSID 200/60/60, 40 epochs, 320 px, batch 4, seed 0, CPU, one class.

### S0 — the arm that produced a duplicate, and the defect behind it

**Problem.** The sparsity-penalty arm (`SSAC-006`) came out **bit-for-bit identical** to the
unpenalised proposal (`SSAC-001`): every weight of the two checkpoints equal to the last bit,
and all four reported metrics identical. A penalty that changes the objective cannot leave the
weights untouched.

**Cause, found by following that symptom.** `saryolo/training/trainer.py::is_saryolo_yaml` chose
between the repository's own model/trainer and ultralytics' stock ones from a *hand-written tuple
of layer names* — and the tuple omitted `RatioSpaceCFARFrontEnd` and `ScatterSelectiveRefinement`,
the repository's two prototype mechanisms. Every SSAC and CFAR arm therefore trained through the
**stock** `YOLO` facade, on ultralytics' `DetectionModel` with ultralytics' own loss, so
`SARYOLODetectionModel.loss` — and with it the SAR-aware criterion and the new penalty — never ran.
The runs converged and reported plausible numbers; nothing failed.

**Change.** The list is now derived from `saryolo.nn.modules.CUSTOM_MODULES`, the single registry
that defines what may appear in a model YAML, with two guards:
`tests/test_arch.py::test_every_custom_layer_name_forces_the_custom_facade` and
`::test_every_variant_that_needs_the_custom_model_gets_it` (checked against the architecture
dict, so it cannot agree with a wrong lookup by construction).

**Blast radius, measured four ways rather than reasoned once.** The defect changed *which* loss
ran, so it could only matter for an arm that declares a `sar_loss` block.

1. **Enumeration.** Of the 68 variants that declare a `sar_loss` block, **two** were not covered
   by the old hand-written list — and both are the new penalty arms. No previously recorded
   result was affected. This is now a test
   (`tests/test_arch.py::test_every_variant_that_declares_a_sar_loss_block_gets_the_sar_loss`),
   so a new variant cannot reintroduce it either.
2. **Loss identity.** `SARAwareDetectionLoss` with every weight at zero is *bit-identical* to
   ultralytics' `v8DetectionLoss` on an identical model and batch
   (`tests/test_losses.py::test_the_sar_criterion_with_every_weight_off_is_the_stock_detection_loss`),
   so an arm without a `sar_loss` block trained the same objective either way.
3. **End-to-end re-run.** The seed-0 control was re-trained through the fixed path with
   `--no-ledger` (a verification may not touch a result row): its weights reproduce the earlier
   run **bit-for-bit** and its validator reports the same metrics (0.566 / 0.303). That is what
   licenses reusing the SSAC-001/002/003 rows, and the arms that were in flight when the fix
   landed.
4. **The arm that mattered.** The penalty arm was re-run, because there the defect was not
   harmless — it was the whole point of the arm.

One more failure mode belongs in §4's table and is recorded here because it is the same kind of
bug: a **halo that is too small** makes the sparse pass refine a tile with the wrong context,
which shows up as an unexplained accuracy gap rather than as an error. The halo is therefore
derived from the expensive path's own kernels, and the `keep = 1.0` equivalence test fails loudly
if it is ever wrong.

### S1 — sparse tile execution (the cost question)

**Change.** `execution="sparse"`: 16-pixel tiles, the top `keep` fraction by mean evidence
refined per image, each gathered tile carrying a halo of the expensive path's receptive radius.

**Predicted before the run.** On a CPU, with Python-level routing, the arithmetic saving would
probably not appear as a wall-clock saving — the 2022 latency-aware-dynamics result and this
repository's own CFAR measurement (3.0× latency for +9 % FLOPs) both say so.

**Measured.** Correctness: at `keep = 1.0` sparse execution reproduces the dense output
**bit-for-bit** in eval mode, and the sparse build is parameter-identical (3,396,329) to the dense
one. Cost: 51.251 ms dense against 63.890 ms at `keep = 1.0` (**+29.5 %** — the routing itself),
50.731 at 0.5, 49.165 at 0.25, 48.547 at 0.1, where 83 % of the P3 expensive path is skipped.
Accuracy: the sparse arm (`SSAC-004`) scores mAP50:95 **0.2995** against the dense proposal's
0.3089 — the accuracy cost of not refining the unselected tiles is about the size of the
mechanism's own gain over the baseline, and the sparse arm's AP_small (0.0697) stays above the
control's (0.0647).

**Verdict — rejected as an efficiency claim, kept as an implementation.** The routing does not
pay for itself at 320 px, for a reason the per-level breakdown makes concrete rather than
rhetorical: the P5 map is 10×10, a single 16-pixel tile, so no budget can skip anything at the
level where most of the expensive path's cost sits. This is a statement about this input scale and
this implementation, not about region-selective computation in general; the honest next step is a
larger input size (P3 = 160×160, P5 = 40×40, 6 tiles there), which needs the GPU budget this host
does not have.

### S2 — narrowing the expensive path (`expand = 1`)

**Problem.** The mechanism costs **+31.1 %** parameters over the stock detector. How much of that
price does the accuracy actually need?

**Change.** `ssac_e1_n`: the expensive path's bottleneck at `expand=1` instead of 2, everything
else identical — 2,953,257 parameters (+14.0 % over stock) and 1.789 GFLOPs.

**Measured.** mAP50:95 **0.2871** — below the stock baseline (0.3012) and well below the
proposal (0.3089). So halving the expensive path's width does not halve the mechanism's benefit;
it removes it. The extra 443,072 parameters are the load-bearing part of the cost, which is the
answer the ablation was built to get. One nuance in the other direction, reported because it is
in the numbers: this arm's AP_small is **0.0849**, *higher* than the proposal's 0.0737 — the
narrowest expensive path favours the smallest objects while losing overall localisation. The cost
ablation does not have a single monotone story, and this is the part that is not.

**Verdict — kept as an ablation.** The narrow arm is not a cheaper operating point to adopt; it is
evidence that the mechanism's cost is where its effect is.

### S3 — pushing the allocation sparse (the premise the sparse build depends on)

**Problem.** S1 shows a sparse build only pays off if the allocation is *concentrated*. The
proposal's own allocation is concentrated at P3 (4.4 % of locations above 0.5) and **not** at
P4/P5 (fraction 1.0), so the routing has nothing to skip where the expensive path is largest.
The premise was unmeasured.

**Change.** `ssac_pen_n`: the identical graph and seed, plus a `w_ssac_sparsity = 0.01` penalty on
the mean allocation. Deliberately *not* a default — penalising the allocation to be small is
circular when sparsity is the mechanism's own claim — and as an arm it also has a structural
side effect worth knowing: the penalty is differentiable through the scorer while the block is
still an exact identity, so this arm trains its allocation from step 0.

**Measured.** The allocation falls at every level — mean `g` at P4 from **0.670 to 0.236**, at P3
from 0.483 to 0.382, at P5 from 0.762 to 0.491 — and the fraction above 0.5 goes from 0.044 / 1.0 /
1.0 to 0.000 / 0.000 / 0.162. Accuracy: 0.3126 mAP50:95 against the proposal's 0.3089, precision
0.938 against 0.863. The between-image spread at P5 rises from 0.095 to 0.167, i.e. the penalised
model allocates by scene rather than uniformly.

**Verdict — kept, with its claim capped.** The mechanism can be pushed to allocate a third to a
half as much at no measurable accuracy cost, which is the premise S1 needed. The small accuracy
difference is **not** reported as a gain: S4 shows this pilot cannot resolve differences of that
size, and a measurement that cannot be resolved is not a result.

### S4 — the seed check, and the claim it withdrew

**Problem.** The pair that decides the mechanism's accuracy claim is `SSAC-001` vs `SSAC-002`, one
seed apart, +0.0063 mAP50:95. This repository had already measured the *baseline's* seed spread as
0.021, so the size of the effect relative to the noise was an open question that only more seeds
could answer.

**Change.** Repeat the pair at seeds 1 and 2 (`SSAC-007`/`SSAC-009`, `SSAC-008`/`SSAC-010`), same
dataset, schedule, resolution and batch.

**Measured.** Paired differences **+0.0063 / +0.0645 / −0.0050** (mean +0.0220, sd 0.0373). The
proposal's own spread is 0.2801–0.3089 against the control's 0.2156–0.3026: the control mis-trains
at seed 1, and the proposal is the more stable of the two, but the *effect* is smaller than its own
spread.

**Verdict — the accuracy claim is withdrawn as unsupported at this scale.** Withdrawal condition 1
of §3.2 fires at seed 2 and is not decisively avoided on the mean, so the honest record is: at 200
training images and 60 test images the mechanism is neither confirmed nor withdrawn. The same
applies to condition 2 (`SSAC-003`'s 0.0226 gap is inside the proposal's own 0.0290 spread, and it
was not re-run). The mechanism's *design* survives — the allocation is real and spatial, sparse
execution is faithful, and the mechanism can be pushed sparse — but its accuracy advantage does
not, and no later section is built on it. Conditions 3 and 4 (small-object loss, uninformative
allocation) did not fire at any seed.
