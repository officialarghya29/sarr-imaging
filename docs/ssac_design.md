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
| **Sparse total** (unbuilt) | `O((9C + ρ·C²e)·HW)` for selected fraction `ρ` | the expensive path, scaled by `ρ` |

The important honest statement: **the dense implementation runs the expensive path
everywhere.** The measured cost rises, and the mechanism's efficiency case exists only if a
sparse implementation can actually skip the expensive path — which §4 states as the next
engineering step and identifies as the risk.

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

### 3.1 What is implemented and verified

`saryolo/nn/modules/ssac.py::ScatterSelectiveRefinement`, modes
`("adaptive", "adaptive_raw", "fixed")`, arms `ssac_{s,n}`, `ssac_raw_{s,n}`,
`ssac_fixed_{s,n}`. Verified by `tests/test_ssac.py` (22 tests) plus the shared
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
| **No wall-clock saving on CPU** | dense FLOPs fall (in a sparse build) but latency does not | this is the *expected* outcome on CPU; report it rather than claim a saving. Identical to the 2022 latency-aware-dynamics caveat and to this repo's own CFAR measurement (3.0× latency for +9 % FLOPs) |
| **Uniform difficulty** | the dataset's regions are all equally hard, so there is nothing to allocate | `selected_fraction` near 1 at convergence; the HRSID subset is small and homogeneous, so this is a live risk |
| **Small-object suppression** | scale-wise AP drops | structural (§1.4) plus Experiment 4 |
| **Wall-clock **overhead** even when dense | Python-level routing grows latency | not applicable to this dense build; would apply to the sparse one |

**The efficiency position, stated plainly.** SSAC's *accuracy* claim is testable now. Its
*efficiency* claim is not: the dense implementation pays the full expensive-path cost, and a
sparse implementation that actually skips regions is the required next engineering step —
one that needs a sparse convolution/gather kernel to be honest, which on this CPU-only
machine is unlikely to convert into a wall-clock saving. That is recorded here so the
prototype cannot be quoted as an efficient method until it has measured one.
