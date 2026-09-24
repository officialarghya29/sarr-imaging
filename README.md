# <div align="center">◆ SAR-YOLO</div>

<div align="center">

**Speckle-aware object detection for synthetic aperture radar**

*Four modules, each derived from a failure mode of SAR imagery — and each one ablatable.*

![status](https://img.shields.io/badge/tests-316_passing-22c55e) ![honesty](https://img.shields.io/badge/fabricated_results-0-black) ![arch](https://img.shields.io/badge/architectures-87_wired-3b82f6) ![exps](https://img.shields.io/badge/experiments-91_configured-8b5cf6) ![license](https://img.shields.io/badge/license-MIT-94a3b8)

</div>

---

> **The instrument comes before the measurement.** A detector paper is only as strong as its
> ablations — and ablations produced by unverified machinery are unverifiable numbers. This
> repository is the instrument: audit → baseline → eleven documented components → slot-matched
> ablations → removal tests → robustness → LOSO generalisation → paper, with honesty enforced
> in code rather than promised in prose.

| Status | |
| --- | --- |
| **Licence** | MIT for the code — datasets are never redistributed |
| **Stack** | Python 3.10+ · Ultralytics 8.4.155 · PyTorch 2.x |
| **Architectures** | 87 variants wired; every one builds and runs a forward pass |
| **Experiments** | 91 configured; each reproducible from a committed YAML |
| **Tests** | 316 passing — no dataset download and no GPU needed |
| **Accuracy results** | none yet — not one number in this repository is fabricated |

---

## ◆ Mission

```text
╔══════════════════════════════════════════════════════════════════════╗
║  THE QUESTION: can one SAR detector generalise to a sensor it has     ║
║  never seen — by being told *how* the image was acquired, not which?  ║
╚══════════════════════════════════════════════════════════════════════╝
```

| | |
| --- | --- |
| **The failure** | SAR detectors learn `object + acquisition appearance`. Change the satellite, the resolution, the polarization — and the representation changes out from under the head. |
| **The protocol** | [Leave-one-source-out](docs/METHOD.md): train on every source, test on one held out. The fold machinery *refuses* to report a number it cannot back. |
| **The mechanism** | Component 33 — a metadata-conditioned adapter, <0.5% parameter overhead, gated so a fresh model is bit-identical to the baseline. The arm that reads only physical descriptors (resolution, band, incidence) can reach a sensor with no embedding row. |
| **The evidence** | [`probe`](#part-vii--quickstart): linear probes, within-class drift and CKA on frozen features, before any new training. If the hypothesis is wrong, the invariant branch does not get built. |
| **The rule** | No fabricated numbers — ever. Unmeasured cells render `TBD`, and the generators refuse to do otherwise. |

```text
SAR image ──► SIA ──► backbone ──► components 1/2/9 ──► PAN-FPN + AMF
              │
              └─ metadata (sensor · resolution · polarization · band · incidence)
                        └─► Component 33 ──► per-level conditioning ──► TPM/SAA/CAG/TADR ──► Detect
```

---

## Read this first

| | |
| --- | --- |
| **What this is** | A complete, reproducible research pipeline for SAR object detection: dataset audit → baseline → ten documented components → ablations → removal tests → robustness → efficiency → cross-dataset → paper. |
| **What is proven** | The infrastructure. 316 tests pass; the baseline reproduces stock YOLO11 exactly; all eleven modules are measurably identity functions at initialisation *and* demonstrably not frozen; every model trains end to end. |
| **What is *not* proven** | Accuracy. **No model has been trained on a real SAR dataset in this repository.** There is no result table here with numbers in it, and the table generators refuse to print one. |
| **Why that's the point** | A detector paper is only as strong as its ablations. If the machinery that produces those ablations cannot be trusted, every number downstream is unverifiable. Build the instrument first. |

> **Honesty is enforced in code, not promised in prose.** Metrics live in an append-only ledger; a value that was never measured is stored as `None` and rendered as `TBD`. There is no code path in this repository that invents a number. See [`saryolo/paper/tables.py`](saryolo/paper/tables.py).

---

## Part I · The physics

You cannot design a SAR detector by reading about RGB detectors. Radar images are formed by coherent illumination, and that single fact cascades into every design decision below.

### 1. Speckle is *multiplicative* noise

A SAR pixel is not a photograph of a surface — it is the coherent sum of returns from many scatterers inside one resolution cell. Those returns add as complex numbers, so random phase differences cause **constructive and destructive interference**. The observed intensity is a *product* of signal and noise:

```text
I = R · S                        R = backscatter reflectivity (what we want)
                                 S = speckle field (what we get instead)
```

For fully developed speckle the single-look intensity follows a negative-exponential law:

```text
p(I) = (1 / ⟨I⟩) · exp( −I / ⟨I⟩ )
```

so its standard deviation **equals its mean**. The standard measure of how much speckle remains is the *equivalent number of looks*:

```text
ENL = ⟨I⟩² / Var(I) = L          L = number of independent looks
```

Multi-looking over *L* independent looks reduces the relative fluctuation by 1/√L — and never eliminates it.

**Why this breaks a CNN.** Nearly every convention in a modern detector implicitly assumes *additive, signal-independent* noise: batch normalisation, L2 losses, and the very idea that a fixed threshold separates object from background. Speckle is multiplicative, so the noise magnitude scales with the signal. A bright target is *noisier* than the dark sea around it. This is the opposite of the low-light intuition, and it is the reason a natural-image pretrained backbone arrives with the wrong prior.

**The classical fix, and its cost.** A log transform turns the product into a sum:

```text
log I = log R + log S            Var(log S) = ψ′(L)     (trigamma function)
```

The variance no longer depends on the mean, which is exactly what a convolutional stack would like. But the log transform also *compresses dynamic range* — precisely the high-reflectance contrast that separates a ship from its wake. **So there is a real trade-off: variance stabilisation versus target contrast.** That trade-off is the scientific content of Component 1 and Component 2, and it is why both are ablated rather than assumed.

### 2. The objects are tiny, and "tiny" is a *resolution* problem

Detection difficulty is not about pixel count — it is about how many *feature cells* an object occupies. For stride *s* and object width *w* pixels:

```text
cells = ⌈ w / s ⌉                a 16px object at stride 32 spans half a cell
```

| Object width | stride 8 (P3) | stride 16 (P4) | stride 32 (P5) |
| ---: | ---: | ---: | ---: |
| 16 px | 2 cells | 1 cell | **1 cell or vanished** |
| 32 px | 4 cells | 2 cells | 1 cell |
| 64 px | 8 cells | 4 cells | 2 cells |

The COCO convention calls anything with area ≤ 32² px² "small". A 16×16 px ship therefore lands on **one or fewer** P4/P5 cells. There is nothing for a deep head to classify. This is the entire justification for Component 5 — and it is why the P2 level is a *hypothesis to test* (EXP-006) rather than a default.

### 3. Clutter wins the gradient argument

Loss is summed over every spatial location. In a cluttered SAR scene, background cells outnumber target cells by orders of magnitude, so the gradient is dominated by the majority class. A plain detector does not "fail to see" the ship — it has been *optimised, on average, to be right about the sea*.

Hence attention is not a decorative add-on here: re-weighting locations changes the effective loss landscape. But that argument applies equally to SE, ECA and CBAM, so it cannot justify our block. The only defensible claim is *comparative* — hence the slot-matched design in Part IV.

### 4. How the score is computed

```text
AP  = ∫₀¹ p(r) dr                        area under the precision–recall curve
mAP = (1 / |T|) · Σ over τ ∈ T of AP_τ   T = { 0.50, 0.55, …, 0.95 }
```

Averaged over ten IoU thresholds — and reported separately for small, medium and large objects, because a single mAP can hide the entire effect being claimed. That scale-wise decomposition is implemented in [`saryolo/evaluation/metrics.py`](saryolo/evaluation/metrics.py).

---

## Part II · From physics to architecture

Each component exists to answer one row of this table. No component exists because it is popular.

| # | Observed failure | Mechanism | Component | Ablation that could kill it |
| --- | --- | --- | --- | --- |
| 1 | Low contrast, weak boundaries | Learned fine-detail enhancement, residual-gated | **SFE** | loses to log / CLAHE / local-std |
| 2 | Speckle mistaken for structure | Separate target vs. speckle streams, recombine | **SFM** | loses to Lee filter / low-pass |
| 3 | Clutter dominates attention | Adaptive channel+spatial gating with local contrast | **SAA** | loses to SE / ECA / CBAM, or to *static* gate |
| 4 | Scale variation across scenes | Learned per-level weighting instead of concat | **AMF** | loses to concat / projected-add |
| 5 | Objects vanish below stride | P2 high-resolution detection level | **P2 head** | AP_small does not move |
| 7 | Background gradient dominates | Target/background separation + small-object term | **SAR loss** | EXP-008 (architecture held fixed) |

### v2: four failures v1 leaves unaddressed

| # | Observed failure | Mechanism | Component | Ablation that could kill it |
| --- | --- | --- | --- | --- |
| 8 | Target information is mixed with clutter | A signed, content-adaptive target prior that **modulates** the feature | **TPM** | loses to the CFAR statistic, to a uniform prior, or to its capacity-matched control |
| 9 | Speckle is broadband; convolution is low-pass-biased | Learnable *radial* spectral filter, adapted per sample | **SFR** | loses to a fixed high-pass, or to non-adaptive bands |
| 10 | Local appearance cannot separate look-alikes | Multi-extent dilated + regional context residual | **CAG** | loses to local-only / regional-only, or does not survive removal |
| 11 | The response peak sits on a clutter pixel, not the target | Prior-conditioned **deformable** resampling: each cell looks up to a bounded distance away | **TADR** | loses to its capacity control (same network, offsets removed), to fixed offsets, or does not survive removal |
| 12 | A natural-image stem consumes raw intensity as if it were a photograph | Input-level SAR representation: local statistics and/or a learned stream, fused | **SIA** | loses to raw intensity, to local-statistics-only, or to learned-only |

**Component numbering**, used consistently across the code, the docs and the paper:

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFE | SFM | SAA | AMF | P2 head | oriented *(planned)* | SAR loss | TPM | SFR | CAG | TADR | SIA |

Components 1-7 are the **v1** model (EXP-001…007). Components 8-11, plus the clutter-aware mode of Component 2, are the **v2 extension** (EXP-013…017). Component 12 sits at the *input* rather than in the neck, so it is numbered last while executing first — the four arms it compares (`raw`, `local`, `learned`, `hybrid`) are `EXP-311…314`. They are not assumed to help: each has a removal ablation (`v2_noprior`, `v2_nofreq`, `v2_noctx`, `v2_norefine`) *and* a slot study, and a component that fails to earn its place gets deleted rather than reported. A component that only works when added in a particular order is not a component — hence the removal table is treated as the stronger evidence of the two.

Component 6 (oriented boxes) is deliberately **not implemented**. Orientation only helps if annotations carry meaningful rotation — true for `SRSDD-v1.0` (six fine-grained ship classes) and `SAR-Ship-Dataset`, but not for SSDD/HRSID. The DOTA converter exists; the head does not, and will only be added if that experiment is actually run.

```text
SAR image
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Component 12: SAR Input Adapter                             (SIA)    │
│   intensity · local statistics · learned stream · learned fusion      │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ YOLO11 backbone                                                      │
│   P2/4  ──► Component 1: SAR Feature Enhancement            (SFE)    │
│   P3/8, P4/16, P5/32, SPPF/C2PSA                                     │
│   P5/32 ──► Component 2: Speckle-Aware Feature Module       (SFM)    │
│   P5/32 ──► Component 9: Spatial-Frequency Representation   (SFR)    │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ PAN-FPN neck                                                         │
│   after every Concat ──► Component 4: Adaptive Multi-Scale  (AMF)    │
│                          Fusion                                      │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ per detection level, applied in this order:                          │
│   Component 8:  target prior modulation                     (TPM)    │
│   Component 3:  SAR-adaptive attention                      (SAA)    │
│   Component 10: context aggregation                         (CAG)    │
│   Component 11: target-aware deformable refinement          (TADR)   │
│               → Detect(P2, P3, P4, P5)  Component 5: P2 head         │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
Component 7: SAR-aware loss
             target/background separation + small-object term +
             background speckle regularisation + optional
             representation consistency (SEC. 4)
   │
   ▼
Predictions
```

The ordering of Components 8 / 3 / 10 / 11 is deliberate and load-bearing for the *target-aware* claim: the prior modulates the feature **first**, so attention, context and the deformable offsets all operate on target-modulated features rather than on raw ones. That is what makes "target-aware attention" and "target-aware refinement" structural properties of the graph here rather than descriptions of intent. Component 11 goes **last** because it is the only stage that moves where the feature is *sampled*; everything before it changes what the feature *contains*, and re-sampling an already-decided feature would undo that work.

Full mathematics, derivations and pseudocode: [`docs/METHOD.md`](docs/METHOD.md).

---

## Part III · The instrument

![Measured identity-at-initialisation property for all eleven modules](docs/assets/identity_property.svg)

### The two guarantees the whole paper rests on

**Guarantee 1 — exact identity at initialisation.** Every module returns `f(x) = x` **exactly**, not approximately, because its residual gate is zero-initialised (`out = x + 0 · branch`). A freshly built SAR-YOLO is therefore *numerically identical* to its baseline, and the measured deviation above is `0.0e+00` for all ten. Without this, a "module helps" result is confounded with "the extra layers happened to change the initial function". With it, a measured difference has exactly one available explanation: the module **learned** something.

**Guarantee 2 — the gate can actually open.** Identity comes from the gate *alone*, so the residual branch must **not** also be zero-initialised. That combination looks harmless and is fatal: with `branch = 0`, the gate gradient `dL/dα = ⟨dL/dout, branch⟩` is identically zero, so `α` never leaves 0 — and `dL/d(branch) = α · dL/dout` is zero for the same reason. Both vanish together, and the module stays a permanent no-op that passes every identity test.

This is not hypothetical. It was a real bug in this repository: Component 1 zero-initialised **both** its gate and its residual conv, so its learned enhancement branch never trained at all. `+SFE` would have measured nothing but its two affine scalars, and nothing in a training log would have shown it. `test_no_module_is_frozen_at_init` now asserts a non-zero gate gradient for every learnable mode, and the pipeline smoke run confirms it dynamically — after two epochs, all **25 gate instances across all eight module types** had left zero.

Both values above are measured live by [`scripts/make_readme_assets.py`](scripts/make_readme_assets.py) and pinned by `tests/test_arch.py`.

![Measured parameter cost of every module in the ablation ladder](docs/assets/ladder_params.svg)

Reading the chart:

- **The baseline is reproduced exactly.** 2,624,080 params (n) and 9,458,752 (s) at 80 classes — Ultralytics' published counts, matched to the unit.
- **+P2 head and FULL have identical bars.** The SAR-aware loss is an *objective*, not a layer. Component 7 costs **zero** parameters. That is precisely what makes it a clean ablation (EXP-008) — the architecture is frozen and only the training signal changes.
- **AMF and the P2 head dominate the cost.** Together they account for most of the added compute, which is why each must earn its place in EXP-005 and EXP-006 before the FULL model is ever trained. The four v2 components cost +1.99M parameters and +5.11 GFLOPs between them, and more than half of that is context aggregation alone.
- **The spectral branch costs parameters but almost no compute.** Component 9 adds +0.103M and essentially **0 GFLOPs**, because it is placed on the deepest backbone stage (P5/32) where the FFT operates on the smallest feature map. The same module at P2/4 would cost roughly 64× more — the placement is a design decision, not an implementation detail.
- **The v2 model is ~72% larger than the baseline and ~2.57× its compute.** That is a real cost, and the honest framing is that Components 8-11 have to pay for it in accuracy, AP_small and robustness, or be removed.

![Parameters versus GFLOPs for each ladder step](docs/assets/accuracy_cost.svg)

### Measured cost of every variant (scale `s`, one-class head)

| Step | Added by | Params (M) | Δ step | GFLOPs @640² | Δ step | vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11-s | — | **9.428** | — | **21.67** | — | — |
| + SFE | Component 1 | 9.437 | +0.009 | 22.25 | +0.58 | +0.1% params · +2.7% compute |
| + SFM | Component 2 | 9.873 | +0.436 | 22.60 | +0.36 | +4.7% params · +4.3% compute |
| + SAA | Component 3 | 10.037 | +0.164 | 22.70 | +0.10 | +6.5% params · +4.8% compute |
| + AMF | Component 4 | 13.840 | +3.802 | 35.21 | +12.52 | +47% params · +62% compute |
| + P2 head | Component 5 | 14.237 | +0.398 | 50.57 | **+15.36** | +51% params · **+133% compute** |
| FULL v1 | + SAR loss | 14.237 | +0.000 | 50.57 | +0.00 | +51.0% params / +133.4% compute — the loss is parameter-free |
| + clutter | Component 2 ext. | 14.406 | +0.169 | 50.74 | +0.17 | +52.8% / +134.2% |
| + prior | **Component 8** | 14.587 | +0.181 | 51.90 | +1.15 | +54.7% / +139.5% |
| + freq | **Component 9** | 14.690 | +0.103 | 51.90 | **+0.00** | +55.8% / +139.5% |
| + context | **Component 10** | 15.854 | +1.164 | 54.65 | +2.75 | +68.2% / +152.2% |
| **FULL v2** | **Component 11** | **16.230** | **+0.376** | **55.68** | **+1.03** | **+72.1% / +157.0%** |
| + prior spectral | **Module G** | 16.586 | +0.356 | 55.68 | **+0.00** | +75.9% / +157.0% |

The story the table tells is uncomfortable and useful: **the two cheapest modules carry the physical insight, and the two most expensive carry the resolution.** If EXP-006 shows the P2 head does not move AP_small, the model drops back to 35 GFLOPs and a much stronger efficiency claim — for free.

![Slot-matched module ablations: attention, fusion, speckle and enhancement](docs/assets/slot_ablations.svg)

### Slot-matched comparison — the only fair way to claim novelty

The claim is never *"attention helps"*. It is *"our block beats SE, ECA and CBAM when each sits in the identical slot on the identical backbone."* Each group above holds every other component fixed; only the named slot varies. A comparison where the "no attention" arm also lacked fusion would credit the fusion block's gains to attention — an easy and serious error.

| Slot | Arm | Params (M) | Note |
| --- | --- | ---: | --- |
| **Attention** | none → SE / ECA / CBAM | 9.873 → 10.037 | all three standard blocks are parameter-identical here |
| | ours, static gate | 9.973 | isolates *adaptivity* from capacity |
| | ours, adaptive gate | 10.037 | same budget as SE/ECA/CBAM — the decisive comparison |
| **Fusion** | Concat → Add (projected) | 10.037 → 11.627 | |
| | ours, static weights | 13.243 | |
| | ours, adaptive weights | 13.840 | |
| **Speckle** | none / Lee / low-pass | 13.403 | classical filters are **parameter-free** |
| | ours, SFM | 13.840 | +0.436M must be repaid in accuracy |
| **Enhancement** | identity / log / CLAHE / local-std | 13.831 | all four cost nothing |
| | ours, SFE | 13.840 | +0.009M — nearly free |

Two facts worth stating plainly. **The classical arms are parameter-free**, so the proposed SFM cannot justify itself on elegance — only on measured accuracy. And the adaptive vs. static rows are the load-bearing experiments: if `att_saa_static` matches `att_saa`, the adaptivity claim is unsupported and the paper must say so.

![Slot-matched module ablations for the v2 components](docs/assets/slot_ablations_v2.svg)

### The v2 slots, and the control that makes the central claim testable

The prior is the paper's central hypothesis, so it gets the most careful ablation of anything here. The question is not "does a prior help?" but "does *learning a spatially varying* prior help, beyond what cheaper alternatives achieve?" Three arms exist to make that answerable:

| Slot | Arm | Params (M) | What it isolates |
| --- | --- | ---: | --- |
| **Target prior** (8) | no modulation | 16.048 | the gate alone |
| | CFAR statistic, no learning | 16.048 | is *learning* needed, or is local statistics enough? |
| | learned, spatially uniform | 16.049 | +0.001M — one logit per channel, per level |
| | learned, capacity-matched | 16.230 | **the same network as ours, pooled over space** |
| | ours, spatial prior | 16.230 | *identical size to the row above* |
| **Frequency** (9) | no branch / fixed high-pass | 16.127 | the fixed filter is a **buffer**: zero parameters |
| | learned bands, input-independent | 16.131 | +0.004M: the bands are only `C × B` |
| | ours, input-adaptive | 16.230 | +0.099M buys per-sample adaptation |
| **Context** (10) | no context | 15.066 | |
| | local (dilated) | 16.142 | |
| | regional only | 15.153 | |
| | ours, both extents | 16.230 | |
| **Refinement** (11) | no refinement | 15.854 | no mixing network and **no offset head** — 4 parameters heavier than deleting the module outright, and those 4 are only the gate scalars |
| | local, no offsets | 16.212 | **the capacity control**: identical mixing network, grid never moves |
| | learned offsets, input-independent | 16.212 | +**8** parameters over the row above, all of them the shared offset field |
| | ours, input-adaptive offsets | 16.230 | +0.017M (17,288 params at scale `s`): the offset head is `C → 2` channels per level |
| **Target prior, spectral** (8+9) | feat-conditioned spectral | 16.586 | **the Module G control**: byte-for-byte the same size as the row below — only the *conditioning signal* differs (raw feature, not prior) |
| | ours, prior-conditioned spectral | 16.586 | the prior chooses the radial band gains: `EXP-018`, the ladder row for Module G |
| **Consistency** (SEC. 4) | term off — the control | 16.230 | `v2_full` itself: same graph, `w_consistency: 0` |
| | speckle, 1 look | 16.230 | is the term doing anything beyond mild denoising? |
| | speckle, 16 looks | 16.230 | does it survive heavy speckle? |
| | low contrast | 16.230 | a *different* degradation family: does the principle generalise past speckle? |
| | low SNR | 16.230 | additive noise rather than multiplicative |
| | ours, speckle 4 looks | 16.230 | **`EXP-019`**: every row is size-identical to the control, so the variable is the objective |
| **Conditioning** (33) | no adapter — the control | 16.230 | `v2_full` itself: the cross-sensor claim has to beat this |
| | metadata gain | 16.278 | +0.049M: fewest parameters of any design |
| | metadata shift | 16.278 | +0.049M: the *other* half of FiLM, alone |
| | film, sensor only | 16.307 | can it transfer to an unseen sensor? **No — by construction.** |
| | film, resolution only | 16.306 | a continuous field that exists for any sensor |
| | film, physical descriptors only | 16.307 | **the arm the headline claim stands on**: usable when no sensor embedding row exists |
| | ours, film sensor+resolution | 16.307 | the two fields a deployment is most likely to know |
| | ours, spatial modulation | 16.311 | the most expressive arm — has to beat the cheaper ones to be kept |
| **Removal** | − clutter / − prior / − freq / − context / − refinement / − prior spectral | 16.061 / 16.048 / 16.127 / 15.066 / 15.854 / 16.230 | each against its own reference: full v2 at 16.230, except − prior spectral which removes Module G against `v2_prior_spectral` at 16.586 — a strict removal of 356,320 params, bit-identical to full v2 |

The **capacity-matched** row is the methodological point. Comparing "learned spatial prior" against "uniform learned prior" would confound *spatial selectivity* with *parameter count* — the larger arm could win for reasons that have nothing to do with the hypothesis. `tp_channel` therefore uses the proposed arm's exact evidence network and averages its output over space, so the two arms are byte-for-byte the same size (asserted in `test_target_prior_arms_are_capacity_matched_where_claimed`) and differ in one respect only: whether the prior is allowed to vary across the image. If `tp_channel` matches `v2_full`, the spatial-prior claim is dead — and the paper must say so.

The same discipline applies to Component 9. A zero-initialised spectral gain would mean `gain = 1`, so the filtered branch would equal its input and the gate gradient would vanish identically — the frozen-module failure again. The bands therefore start small-but-non-zero, and identity comes from the gate. The `sff` arm also starts numerically equal to `static`, so the difference between those two rows measures *input adaptivity* alone.

**The removal table names a reference for each arm rather than assuming one for the slot.** A removal is only meaningful relative to a model, and one arm's reference is not `v2_full`: `v2_nopspectral` removes Module G, which is the `EXP-018` ladder step rather than part of `v2_full`. Measured against `v2_full` it differs by **zero** parameters — so a slot-wide "smaller than full" rule would either miss it or push it into being a no-op that still reports as a clean removal. The references are therefore declared next to the slot, the generator refuses to emit a removal arm without one, and the test reads the table instead of restating it (a hand-copied five-arm list is exactly how `v2_nopspectral` stayed outside the check while the slot had six arms).

**Module G lives in the prior slot, not the spectral slot — and that is forced, not stylistic.** The brief asks for the target prior to drive frequency-band selection, but the backbone spectral slot runs at P5/32 *before* any prior exists, and an Ultralytics graph cannot feed a custom module two inputs (`parse_model` resolves `c2 = ch[f]`; only hardcoded names receive a channel list — verified against the installed source). Faking the wiring with a backward connection would break the stock summary, FLOPs counter and validator. So the conditioning is implemented where the prior actually is: the prior module itself produces the band gains. `tp_spectral_feat` is the control that keeps the claim honest — same head, same bands, same descriptor, fed raw-feature statistics instead of prior evidence — so `spectral − spectral_feat` isolates *prior* conditioning from mere input adaptivity.

**SEC. 4 of the brief — representation consistency — is implemented as an opt-in loss term.** The model runs a second forward on a degraded copy of the same batch (the *same* corruption physics the robustness benchmark uses, so training and evaluation cannot drift apart) and the drift between the two views' feature maps is penalised. Three properties are pinned by test rather than asserted: the term is exactly zero when the views agree; positions where either view is silent are *excluded* rather than penalised (`cosine_similarity` returns 0 for a zero vector, so a dead position would otherwise contribute the maximum penalty and the term would spend its gradient reviving dead channels); and the perturbed pass runs BatchNorm in eval mode so the buffers see each batch exactly once — without which enabling the term would silently change the normalisation of the whole network and every ablation would measure that instead. The weight defaults to 0, so stock behaviour is bit-identical; enabling it costs one extra forward pass per step, which is stated rather than hidden.

**It is a loss slot, not an architectural one, and the difference decides where its arms live.** The term was implemented and unit-tested before any variant used it, which left it unreachable from the experiment matrix — nothing in `configs/` switched it on. It now has a runnable ladder row (`EXP-019`) and its own slot study (`EXP-321…326`), and every enabled arm states its degradation and severity **explicitly** rather than inheriting them, so the perturbation a run trains under can be read off its own config. The arms deliberately do *not* join the removal slot: a removal there is verified by a strict parameter drop, and a loss term can never produce one — all six consistency arms are byte-for-byte the same size as the control, and that equality is the point. It is also what makes the comparison clean: with the graph fixed, an accuracy difference is the objective and nothing else. The sweep over speckle ×1 / ×16 / low-contrast / low-SNR exists because a term that only helps when the benchmark corruption happens to match its training corruption is a tuned constant rather than a principle, and the paper has to be able to tell those apart.

**Acquisition conditioning is the cross-sensor claim, and its arms are chosen so the claim can fail.** The whole reason this slot exists is the leave-one-source-out protocol: a detector trained on several sensors and tested on one it has never seen. That rules out the obvious shortcut — a learned embedding table indexed by sensor id — because the held-out sensor has no row in that table. The categorical path is *unavailable exactly where the claim is tested*. So the field-set arms are the load-bearing ones, and they are ordered by what they can transfer: `film:sensor` cannot reach an unseen sensor by construction, `film:resolution` can (a continuous field every acquisition has), and `film:continuous_only` (resolution, band, incidence) is the arm the headline result has to come from. If the categorical arm is the only one that helps, the method does not generalise and the paper must say so. Two further properties are pinned by test rather than asserted: a fresh conditioned model is numerically identical to the unconditioned one (the modulation is gated and the gate starts at zero, while the rest of the path starts *non-zero* — a zero-initialised modulation would have a vanishing gate gradient and could never open), and a field an arm does not consume has both its value and its availability flag zeroed, so `sensor`-only and `resolution`-only are genuinely information-free with respect to each other rather than merely ignoring the value at the end. The cost is stated plainly: under 0.5% of parameters for every arm, and the metadata is applied per sample and broadcast over space, because a batch drawn across sources would otherwise average the acquisitions together and destroy the very signal being tested.

**The field-set arms are near-matched in capacity, but not byte-identical, and that is a real caveat rather than a rounding detail.** `cond_sensor`, `cond_resolution` and `cond_continuous` differ by at most ~1.1k parameters (0.007% of the model) because each consumes a different number of continuous descriptors. That is tight enough to attribute a difference to *which fields* are read, but it is not the byte-for-byte equality the target-prior and consistency slots achieve, so the paper should report the parameter count per arm rather than claiming exact capacity matching.

Component 11 needs a control that is uncommon in detection papers, because "deformable convolution" comparisons usually give the proposed arm *both* a new network and a new operation. `rf_local` therefore keeps the identical mixing network and removes only the offsets, so `ours − local` is attributable to **deformation** rather than to the extra convolution. Two further tests make the mechanism falsifiable at the unit level: a zero offset field must reproduce `local` exactly up to float32 round-off (measured: `7e-7`), while a **0.04-cell** displacement moves the output by `0.4` — six orders of magnitude larger, so the tolerance is demonstrably not hiding a real shift. And the base grid's corners are asserted at exactly `(-1, -1)` and `(1, 1)`, which is what pins `align_corners` to the sampling convention rather than leaving it to chance.

![Test-suite composition across the repository's modules](docs/assets/tests.svg)

---

## Part IV · Results

### What is verified right now

| Check | Result | Evidence |
| --- | --- | --- |
| Baseline reproduces stock YOLO11 exactly | `2,624,080` (n), `9,458,752` (s) | `test_baseline_matches_stock_yolo11_parameter_count` |
| **All 87 architectures construct and forward** | pass, at even and odd input sizes | `test_every_variant_builds_and_forwards` |
| Declared scales build at the right stride count | pass | `test_declared_scale_variants_build` |
| Every SAR module is an **exact** identity at init | `max\|f(x)−x\| = 0.0e+00` ×10 | measured live + `test_each_module_is_exactly_identity_at_init` |
| **No module is silently frozen at init** | every learnable mode has a non-zero gate gradient | `test_no_module_is_frozen_at_init` |
| **All gates leave zero during real training** | `25/25` non-zero after 2 epochs | `SMOKE-003` checkpoint (checked dynamically, not just statically) |
| **The consistency term reaches the loss through the real path** | weighted `sar_loss` > unweighted, second view run, BN counters advance by exactly 1 | `test_consistency_term_reaches_the_total_loss_through_the_model`, `test_consistency_pass_does_not_disturb_batchnorm_running_stats` |
| **A fresh conditioned model equals the unconditioned one** | `max\|v2_full − cond_*\| = 0.0` for all seven arms; the inserted adapters shift layer indices, so the stock-layer copy is what makes this exact | `test_new_arms_are_neutral_at_init_relative_to_v2`, `test_each_module_is_exactly_identity_at_init` |
| **Metadata cannot change the output while the gate is closed** | supplying a *known* acquisition to an untrained adapter is still a bit-exact identity | `test_metadata_cannot_change_the_output_while_the_gate_is_closed` |
| **An unused metadata field carries no information** | value *and* availability flag are both masked, so sensor-only vs. resolution-only is a real comparison | `test_an_unused_field_cannot_reach_the_descriptor` |
| **Each conditioning arm reads exactly the fields it declares** | `gain/shift/spatial` read all six; `film:sensor_resolution` reads two; `continuous_only` reads the three physical descriptors | `test_each_conditioning_arm_reads_exactly_its_declared_fields` |
| **Conditioning is per-sample, not per-batch** | row *i* of a mixed-source batch equals row *i* run alone — the LOSO recipe depends on it | `test_conditioning_is_per_sample_not_per_batch` |
| **An out-of-vocabulary sensor is refused, not clamped** | clamping would map an unseen sensor onto a trained-on one, silently | `test_a_vocabulary_mismatch_raises_instead_of_snapping_to_a_nearby_sensor` |
| **Conditioning survives a real training run, not just a unit test** | the actual `SARYOLOTrainer` runs over a metadata-bearing dataset: batches carry aligned descriptors and an adapter gate leaves zero | `tests/test_conditioning_smoke.py` |
| **Validation conditions too — the model is not silently fed an "unknown" acquisition** | the validator is handed a *path* by `final_eval`, and a path carries no context, so the metrics would describe an unconditioned model; resolution is pinned for every handle | `test_conditioned_validation_conditions_on_metadata`, `test_the_validator_resolves_a_real_module_from_every_handle_it_is_given`, `test_checkpoint_vocabularies_come_back_frozen_not_rebuilt` |
| **A conditioned checkpoint is probed *conditioned*** | feeding it no acquisition would report the *unconditioned* representation, invalidating the baseline-vs-conditioned comparison; vocabularies are read from the checkpoint, and the extraction leaves no acquisition behind | `test_probe_conditions_a_conditioned_checkpoint`, `test_collect_head_features_applies_the_acquisition_it_is_given`, `test_collect_head_features_leaves_no_acquisition_behind`, `test_probe_refuses_a_conditioned_checkpoint_without_acquisition` |
| **A withheld acquisition field encodes exactly like an absent one** | value *and* availability zeroed, categorical id forced to the reserved unknown row — the missing-metadata study measures real degradation, not a fabricated `resolution = 0.1 m` | `test_masked_field_encodes_like_a_never_recorded_field`, `test_a_masked_categorical_field_lands_on_the_unknown_row`, `tests/test_field_mask.py` |
| **The LOSO protocol runs end to end on a conditioned model** | three sensors in the fixture, one held out of training only: the vocabulary cannot name the held-out sensor, validation still conditions on it, and the unseen sensor lands on the unknown row while its physical descriptors survive | `test_a_conditioned_model_trains_without_the_held_out_sensor_and_still_validates_on_it` |
| **A metadata-field restriction survives fold generation** | `loso --metadata-fields sensor` writes the restriction into every fold config, so no fold's number is measured under a different protocol than its neighbour | `test_a_metadata_field_restriction_is_written_into_every_fold_config` |
| **LOSO refuses to report a number it cannot back** | one group, or a zero-ground-truth fold, raises instead of yielding a score | `tests/test_groups.py`, `tests/test_metrics.py` |
| **No README-cited test can be missing** | every test name in a code span must exist, and a truncated name counts as unverifiable rather than being skipped | `test_every_test_cited_in_the_readme_exists` |
| Deformable refinement's grid identity is pinned | zero offset → `7e-7` dev; 0.04-cell shift → `0.4` | `test_refinement_resampling_is_an_identity_at_zero_offset` |
| Offsets move the grid, and only in the adaptive arm | pass | `test_refinement_offsets_actually_move_the_sampling_grid`, `..._are_feature_adaptive_only_in_deform_mode` |
| SAR-YOLO predicts identically to baseline at init | max abs diff `0.0` (v1 **and** v2) | `test_models_output_identically_to_baseline_at_init` |
| Filenames cannot silently downgrade the scale | pass (87 variants) | `test_variant_filenames_encode_scale` |
| Mode names cannot break the parser | pass (no keyword or `parse_model`-local collision) | `test_module_mode_names_are_safe_for_parse_model` |
| Spectral branch is resolution- and AMP-safe | odd, non-square and fp16 inputs pass | `test_frequency_module_is_resolution_independent` |
| Every ablation arm has a runnable config | pass | `test_every_ablation_arm_has_a_runnable_experiment_config` |
| No table row names a non-existent model | pass | `test_paper_table_rows_reference_real_models` |
| SAR-aware loss reaches the optimiser | `train/sar_loss`, `val/sar_loss` non-zero | smoke run logs |
| Baseline, FULL v1 and FULL v2 train end to end on CPU | all complete, ledger written | `_smoke_*` configs |
| COCO matcher agrees with hand-computed cases | pass | `tests/test_metrics.py` |
| No table can emit an unmeasured number | pass | `tests/test_repo.py` |
| The README cost table matches the measured models | pass | `test_readme_cost_table_matches_the_measured_models` |
| Oriented (9-field) labels are diagnosed, not dropped | reported as `oriented_labels` | `test_validator_diagnoses_oriented_labels_instead_of_calling_them_malformed` |
| A dataset cannot validate clean while its boxes are unreadable | pass | same test |
| Malformed VOC XML cannot abort a batch conversion | counted as `skipped_annotations` | `test_voc_converter_survives_malformed_annotations` |
| Every registry dataset ships a matching data config | pass | `test_every_registry_dataset_has_a_matching_data_config` |
| **The input-adapter control is bit-exact** | `max\|v2_full − in_identity\| = 0.0`, layer-for-layer identical graph, `+0` parameters | measured live; `test_frequency_slot_arms_differ_as_documented` pins the gate-only cost |
| **A hard image cannot be silently dropped** | mining a foreign split raises instead of writing a no-op list | `test_hard_image_from_another_split_is_an_error_not_a_silent_no_op` |
| **Augmentation cannot invalidate labels** | every corruption preserves shape (all 6 checked); clean view is byte-exact | `tests/test_augmentation.py` |
| **The miner and the failure table cannot disagree** | both derive from one taxonomy; counts reconcile exactly | `test_score_totals_equal_the_failure_taxonomy_counts` |
| Every taxonomy outcome must have a difficulty weight | an unmapped outcome raises rather than scoring zero | `test_every_taxonomy_outcome_has_a_weight` |

### What is not yet measured

![Experiment coverage: wired and reproducible, but not yet measured](docs/assets/coverage.svg)

Every experiment below is **code-complete and reproducible from a committed config**. None has produced an accuracy number, because none has been run on a GPU against a real dataset.

| Model | mAP50 | mAP50:95 | AP_small | Params (M) | GFLOPs | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO11 baseline | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + SFE | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + SFM | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + SAA | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + AMF | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + P2 head | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **SAR-YOLO v1 (FULL)** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + clutter-aware | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + target prior | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + spatial-frequency | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| + context | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **SAR-YOLO v2 (FULL)** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| − refinement (removal) | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

`TBD` is rendered by the generator, not typed by hand. Fill these by running the notebooks on a GPU — the grid above fills itself in from the ledger.

---

## Part V · Data

![The public SAR dataset landscape with pilot and benchmark tiers](docs/assets/datasets.svg)

| Order | Dataset | Images | Classes | Format | Input | Tier | Role |
| ---: | --- | ---: | ---: | --- | ---: | --- | --- |
| 1 | **SSDD** | 1,160 | 1 | VOC | 512 | pilot | Full ablation grid inside one free Colab session |
| 2 | **HRSID** | 5,604 | 1 | COCO | 800 | pilot | Harder scenes; stresses the small-object claim; **400 background-only images** for false-positive and clutter experiments |
| 3 | **SRSDD-v1.0** | ~1,022 | 6 | DOTA | 1024 | benchmark | Rotated, fine-grained ships — where Component 6 becomes answerable |
| 4 | **SAR-Ship-Dataset** | ~43,819 | 1 | DOTA | 512 | benchmark | Scale; oriented boxes |
| 5 | **SARDet-100K** | ~116,598 | 6 | COCO | 800 | benchmark | Largest SAR detection benchmark — final numbers and cross-dataset |

**Why cheapest-first.** Development order is decided by cost, so a wiring bug surfaces on a 1,160-image dataset in minutes rather than on a 116k-image one after hours. The rule that matters more than speed: *understand the data before building the model.*

**Two notes before quoting these numbers.** SRSDD-v1.0 is widely cited as having **seven** ship categories; the dataset paper states **six** (ore-oil, bulk-cargo, fishing, law-enforcement, dredger, container) over 2,884 instances cut from 30 panoramic Gaofen-3 tiles. And image counts differ between the official release and the common cropped distributions — `docs/DATASETS.md` records exactly what must be verified per release.

The oriented-label trap is caught automatically rather than discovered after a wasted GPU run. Because both SRSDD-v1.0 and SAR-Ship-Dataset convert to **9-field** YOLO-OBB rows (class + 8 corners) while the detection models consume **5-field** rows, `check-data` reports an `oriented_labels` error and refuses to validate such a dataset clean — instead of calling every row "malformed", or profiling the dataset as a perfectly good one containing zero objects.

Details, licences, citations and download routes: [`docs/DATASETS.md`](docs/DATASETS.md). **No dataset is redistributed here** — MIT covers the code only.

### Cross-source splits: leave-one-source-out

The headline claim is that the detector generalises to a **source it was never trained on** — a different satellite, or a different sensor on the same platform. That is a stronger claim than in-domain accuracy and a different one from cross-*dataset* accuracy, because two datasets can share a sensor and one dataset can mix several: SAR-Ship-Dataset is Sentinel-1 **and** Gaofen-3, SARDet-100K is unified from **ten** sources. So the evaluation unit has to be the source, not the dataset name.

```bash
python -m saryolo.cli loso --images <dir-of-images> --rule parent --data configs/datasets/ssdd.yaml \
    --min-test-images 30 --out datasets/splits/loso
```

That writes one directory per held-out source containing `train.txt` / `val.txt` / `test.txt`, a manifest recording the rule and every source size, and — given `--data` — two runnable data configs per fold:

| Config | `val` points at | Used for |
| --- | --- | --- |
| `data.yaml` | `val.txt` | training (validation drives early stopping) |
| `eval_holdout.yaml` | `test.txt` | **the reported number** |

The two exist because `evaluate_detections` reads the config's `val` entry. A single config with `val: val.txt` would report a score measured on sources the model trained on — an in-domain number wearing a cross-source label — and nothing about the output would look wrong.

The rule is **stated, never inferred**, because no universal one is safe: some archives ship a directory per sensor, some encode it in the filename, some only in a metadata table. `--rule parent|regex|sidecar|resolution` covers those, and the fallback is an explicit mapping rather than a heuristic. The **`resolution` rule is the cross-resolution protocol (Experiment D)**: it takes `--metadata <table.json>` — built with `python -m saryolo.cli metadata --images <dir> --sidecar <csv>` when the archive states acquisition in a CSV — plus `--edges 5,10,20` (no default: the bin width decides what "cross-resolution" even means), bins each image's `resolution_m` into half-open groups labelled like `resolution_m>10`, and holds out one bin per fold exactly as LOSO holds out a sensor. The bins are physical quantities, not cluster IDs, so a fold trained on `resolution_m<=10` and evaluated on `resolution_m>10` answers the question the experiment asks — can the detector bridge a resolution gap it never saw — rather than a random split. `saryolo.data.binned_rule` refuses degenerate inputs the same way the source rules do: a single bin, a non-ascending edge list, and an image with no metadata row (unless `--allow-unmatched`, which the printed summary then discloses).

Every guard below exists because its failure mode is *silent* — the run completes and reports a number:

| Guard | What it prevents |
| --- | --- |
| A rule matching nothing is an error | One group happens to be a *valid-looking* answer: the folds build, files are written, and the “held-out source” is a random chip split |
| Fewer than two sources is an error | There is nothing to hold out, so the number measures nothing of the kind claimed |
| Groups whose last component is `train`/`val`/`test` are refused | Keying one level too high holds out a split the pipeline created itself — three plausible groups, plausible sizes, in-domain result |
| Images the rule cannot key are refused by default | An unkeyed image is absent from every fold, shrinking the test set to a subset nobody chose |
| A source below `--min-test-images` is refused | mAP over a handful of chips is noise presented as evidence |
| Splits are asserted disjoint before writing | A leaking fold still trains and still reports a number |
| **An evaluation split with zero ground truth raises** | A metrics dict of `None`s is indistinguishable from a finished run |

That last one was live: a split given as a `.txt` image list had its labels derived by string-replacing `images` → `labels` on the *list path*, so no ground truth was ever read and `mAP50` came back `None` while the command reported success. The list form is exactly what a fold emits, so without the fix the whole protocol would have measured nothing. Ground truth is now read **before** inference, so an unresolvable split fails immediately instead of after a full prediction pass.

**Both halves are now built; neither is measured.** The protocol is one half and the model side is the other: a lightweight adapter conditioned on acquisition metadata, placed **first in the per-level chain** at P3/P4/P5 so every later slot operates on acquisition-conditioned features (see the conditioning slot above and Component 33 in `docs/METHOD.md`). What does not exist is a cross-source *number* — no fold has been trained, so every cell stays `TBD`.

Two things must be established, and only the first is currently true:

1. that the baseline **degrades measurably** across sources — if it does not, there is no problem and the adapter is unmotivated;
2. that `cond_continuous`, the arm that can reach an unseen source because it reads only physically-ordered fields, **recovers part of that degradation** against the `v2_full` control.

Note the placement is a deliberate departure from the original plan, which put the adapter at the stem and neck. It sits in the neck because that is the only place `parse_model` can consume a second signal at all — a custom layer receives one tensor, and the metadata arrives through a context object the model fills before the forward pass rather than as a graph edge. The stem variant is the separate `in_*` slot study. If `cond_continuous` fails while `cond_sensor` succeeds, the honest conclusion is that the method specialises to known sensors rather than generalising to new ones — a negative result the slot is instrumented to detect, not to hide.

---

## Part VI · Design rules

The brief that motivated this project named the failure mode to avoid: `YOLO + CBAM + SE + Transformer + BiFPN = "new model"`. So:

1. **One module at a time.** EXP-002…EXP-007 add exactly one component per run.
2. **Identity at initialisation.** A fresh SAR-YOLO is numerically the baseline — so gains are learned, not structural.
3. **Slot-matched baselines.** Proposed blocks are compared against standard components *in the same slot*.
4. **Adaptivity isolated explicitly.** Static-weight variants exist purely to justify the word "adaptive".
5. **No fabricated numbers.** Unmeasured cells stay `TBD`, and the generator refuses otherwise.
6. **The final model is evidence-driven.** A component that fails its ablation is removed, and the removal is reported.

---

## Part VII · Quickstart

Reproduce every verified claim locally — **no dataset download, no GPU** (`pytest tests/ -q` → all green):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python ultralytics pytest
source .venv/bin/activate

pytest tests/ -q                                          # 316 tests
python -m saryolo arch --variant all --nc 1               # emit 87 model YAMLs
python -m saryolo synth-data --out datasets/processed/synthetic_smoke
python -m saryolo train --exp configs/exp/_smoke_baseline.yaml
python -m saryolo train --exp configs/exp/_smoke_full.yaml
python -m saryolo ledger
python scripts/make_readme_assets.py                      # regenerate this page's charts
python scripts/check_chart_layout.py                      # lint those charts for text collisions
```

> The `_smoke_*.yaml` configs use **synthetic Gamma-speckle** data. They exist to catch wiring bugs in seconds instead of after an hour of GPU time. Any number they produce is meaningless as a research result.

### Running the real thing

The complete first-GPU recipes — prepare → audit → LOSO folds → metadata → baseline →
conditioning arms → representation probe, with a go/no-go gate at each branch — are
[`docs/RUNBOOK_SSDD.md`](docs/RUNBOOK_SSDD.md) (the first measured number) and
[`docs/RUNBOOK_HRSID.md`](docs/RUNBOOK_HRSID.md) (the cross-resolution pilot),
verified stage-by-stage against fixtures.

When a trained checkpoint exists, the §14 diagnosis is one command:

```bash
python -m saryolo probe --weights runs/EXP-001/weights/best.pt \
    --data configs/datasets/ssdd.yaml --field sensor --class-field \
    --out results/probes/ssdd      # probes next to the chance rate, not next to 1.0
```

```bash
# 1. Get a dataset onto the machine (licensed routes in docs/DATASETS.md)
python scripts/prepare_dataset.py --dataset ssdd --raw /content/raw/SSDD

# 2. Audit and profile before spending GPU time
python -m saryolo check-data --dataset datasets/processed/ssdd --classes ship
python -m saryolo stats      --dataset datasets/processed/ssdd --classes ship --name ssdd

# 3. Baseline first, then one module at a time
python -m saryolo train --exp configs/exp/EXP-001_baseline.yaml
python -m saryolo train --exp configs/exp/EXP-002_sfe.yaml

# 4. Or walk the whole matrix
python scripts/train_all_experiments.py --keep-going
```

| Notebook | Purpose |
| --- | --- |
| [`01_dataset_prep.ipynb`](notebooks/01_dataset_prep.ipynb) | Download, convert, validate, profile, leakage-check |
| [`02_train_and_ablate.ipynb`](notebooks/02_train_and_ablate.ipynb) | Train EXP-001…EXP-008, generate the ablation tables |
| [`03_benchmark_and_paper.ipynb`](notebooks/03_benchmark_and_paper.ipynb) | Robustness, efficiency, cross-dataset, multi-seed, figures |

---

## Part VIII · Experiment matrix

| ID | Run | Isolates |
| --- | --- | --- |
| EXP-001 | YOLO baseline | reference point |
| EXP-002 | + SFE | Component 1 |
| EXP-003 | + SFM | Component 2 |
| EXP-004 | + SAA | Component 3 |
| EXP-005 | + AMF | Component 4 |
| EXP-006 | + P2 head | Component 5 |
| EXP-007 | **FULL SAR-YOLO** | all + Component 7 |
| EXP-008 | FULL without SAR loss | the loss, architecture frozen |
| EXP-009 | Robustness sweep | speckle, contrast, blur, resolution, clutter |
| EXP-010 | Efficiency benchmark | params, FLOPs, FPS, latency, memory |
| EXP-011 | Cross-dataset | domain shift |
| EXP-012 | Multi-seed (0, 1, 2) | mean ± std |
| EXP-013 | + clutter-aware SFM | Component 2 extension: clutter modelled separately from speckle |
| EXP-014 | + target prior | **Component 8**, the central hypothesis |
| EXP-015 | + spatial-frequency | **Component 9** |
| EXP-016 | + context | **Component 10** |
| EXP-017 | **FULL SAR-YOLO v2** | **Component 11** — the v2 reference model |
| EXP-018 | + prior spectral | **Module G** — the target prior selects the radial frequency bands |
| EXP-019 | + representation consistency | **SEC. 4** — identical graph to EXP-017, different objective |

### Module-level ablations (`EXP-2xx`)

Every arm below sits in the *same slot* with every other component held fixed, and every arm has a runnable config — not just a model YAML:

| Range | Slot | Arms |
| --- | --- | --- |
| `EXP-211…216` | attention (3) | none · SE · ECA · CBAM · ours-static · ours-adaptive |
| `EXP-221…224` | fusion (4) | Concat · projected-Add · ours-static · ours-adaptive |
| `EXP-231…234` | speckle (2) | none · Lee · low-pass · ours-SFM |
| `EXP-241…245` | enhancement (1) | identity · log · CLAHE · local-std · ours-SFE |
| `EXP-251…258` | target prior (8) | none · CFAR · uniform · capacity-matched-spatial · ours-spatial · **feat-conditioned spectral** · **ours prior-conditioned spectral (Module G)** · ladder reference |
| `EXP-261…266` | frequency (9) | none · fixed high-pass · learned bands · **DCT** · **wavelet** · ours-adaptive |
| `EXP-271…274` | context (10) | none · local · regional · ours-both |
| `EXP-281…286` | removal | −clutter · −prior · −freq · −context · −refinement · −prior-spectral (Module G) |
| `EXP-291…296` | refinement (11) | none · **local (capacity control)** · fixed offsets · offset sweep (25% · 100%) · ours-adaptive |
| `EXP-311…314` | input adapter (12) | raw (`identity`) · local-statistics · learned · ours-hybrid |
| `EXP-321…326` | consistency (SEC. 4) | off (control) · speckle ×1 · speckle ×16 · low-contrast · low-SNR · **ours, speckle ×4** |
| `EXP-331…338` | conditioning (33) | none (control) · gain · shift · film-sensor · film-resolution · film-physical-only · **ours film sensor+resolution** · spatial |
| `EXP-401…403` | init stage (Phase 3) | random init (control) · COCO `yolo11s.pt` fine-tuned · SAR-pretrained checkpoint fine-tuned |

The `EXP-26x` range answers SEC. 14 of the brief directly: the transform is the *variable*, so FFT, block-DCT and Haar wavelet are compared in one slot with everything else held fixed, rather than assuming the FFT is right. `EXP-294/295` sweep the refinement's `max_offset` bound; it is logged as an **open sweep, not a tuned constant**, because the learned offsets sit at ~94% of the bound where `tanh`'s gradient is smallest — so the bound may well be too tight.

**The init slot (Phase 3) is a config-level comparison, not an architecture one.** The three arms share one model YAML, one seed, and every train argument; the only stated difference is `init:` — `none` (a fresh build), `coco11` (COCO-pretrained `yolo11s.pt`), or a checkpoint path (MSFA-style SAR weights, when available). How much of the RGB-to-SAR performance problem is just input-initialisation is then readable off the ledger, because the runner writes the init stage, the source checkpoint, and the exact number of tensors whose values actually changed into every run record — an honest count, excluding the zero-initialised buffers that the raw intersection over-counts ~5×. A transfer that would change nothing raises instead of running as a random-init wearing a pretrained label.

### Representation diagnosis (master-plan §14)

Before any accuracy number, the central hypothesis — *sensor appearance is entangled with object semantics* — is directly testable on a frozen detector: hook the detection head's per-level maps, mean-pool each image, and ask what a **linear probe** can recover (`saryolo.evaluation.probes`). If a `sensor` probe far exceeds the chance rate implied by the source count while a `class` probe does not, the representation is appearance-dominated — the failure the conditioning adapter addresses, and the evidence that justifies the invariant branch (or, if the probe disagrees, the reason not to build it). Three complementary readings are implemented, each with its meaning stated up front: probe accuracy per level; within-class centroid distance across acquisition groups ("same object, different sensor" drift); and linear CKA between the pooled feature matrices of two acquisition groups over the same images. The extraction is side-effect-free by test — BN buffers are untouched and batch rows are independent — because a diagnosis pass must not alter the thing diagnosed. **Not yet run on a trained checkpoint**: on an untrained model the numbers are placeholders, and none of this constitutes evidence until a real checkpoint exists.

Configs are generated, not written by hand:

```bash
python scripts/make_exp_configs.py --dataset ssdd   # writes configs/exp/EXP-0xx_* and EXP-2xx_*
```

### Training strategies that are not modules (`SEC. 5` and `SEC. 6`)

Two parts of the brief change *what the model sees* rather than *what the model is*. They live outside the architecture on purpose, because a sampling strategy that needed its own layer would no longer be attributable — you could not tell whether a gain came from the extra capacity or from the extra examples.

**SAR-specific augmentation** reuses the *same* corruption model the robustness benchmark evaluates under (`saryolo/evaluation/robustness.py`), so a model is trained under the degradation it is later tested under, and the severity grid is literally the same tuple. It is offline: the augmented training split is a committed artifact with a per-file manifest recording the corruption and severity applied to each image.

```bash
python -m saryolo augment --data configs/datasets/ssdd.yaml \
       --views 2 --kinds speckle low_contrast blur low_resolution low_snr \
       --out datasets/augmented/ssdd
```

Three properties are enforced rather than assumed:

- **Labels are copied verbatim.** Every corruption is *appearance-only* — it changes pixel values and never moves a target — which is the entire justification for not transforming boxes. The builder compares output and input shapes per image and refuses a mismatch, because the moment a corruption resizes an image the labels become wrong for it.
- **`clutter` is opt-in, not default.** It injects bright blobs that are not labelled, which teaches suppression of bright compact regions — the exact appearance of the small targets this paper is trying to improve. It stays available for the ablation that tests that concern.
- **Severities must come from the published robustness grid.** A value outside it is rejected, so the augmented run and the robustness figure cannot stop referring to the same degradation.

**Hard-example mining** is an offline sampling change: score every image by how badly the model failed on it, then emit a training list containing all training images *plus* the hardest ones repeated.

```bash
python -m saryolo mine-hard --weights runs/detect/exp/weights/best.pt \
       --data configs/datasets/ssdd.yaml --split train --out results/hard_examples
```

Two details carry the weight of the claim:

- **It mines the split that will be trained on, and refuses otherwise.** Mining `val` and then oversampling those images would train on the evaluation data. `--split` defaults to `train`, and a non-`train` value prints what it contaminates. A hard image that is not in the training list is an **error**, not a filter: the first version silently dropped them, so mining `val` wrote a list containing none of the mined images while still reporting a repeat count and exiting 0 — a run indistinguishable from the baseline.
- **It reuses the failure taxonomy** (`visualization/error_analysis.py`) rather than matching boxes itself, so the difficulty ranking and the paper's failure-analysis table cannot disagree about the same model. The first version carried its own IoU matcher, which made it a *third* matcher and let the two reports contradict each other.

---

## Part IX · Repository map

```
saryolo/
├── nn/
│   ├── arch.py            symbolic builder: 87 variants, all indices computed
│   ├── modules/
│   │   ├── input_adapter.py SIA (Comp 12) + raw / local / learned arms
│   │   ├── enhancement.py SFE (Comp 1) + log / standardize / CLAHE baselines
│   │   ├── speckle.py     SFM (Comp 2) + Lee / low-pass + clutter-aware mode
│   │   ├── attention.py   SAA (Comp 3) + SE / ECA / CBAM baselines
│   │   ├── fusion.py      AMF (Comp 4) + concat / projected-add / static baselines
│   │   ├── target_prior.py TPM (Comp 8) + cfar / uniform / capacity-matched arms
│   │   ├── frequency.py   SFR (Comp 9) + high-pass / DCT / wavelet / non-adaptive arms
│   │   ├── context.py     CAG (Comp 10) + local-only / regional-only arms
│   │   ├── refinement.py  TADR (Comp 11) + local / fixed-offset control arms
│   │   └── _common.py     shared primitives + the module contract (identity, gradients)
│   ├── losses.py          SAR-aware loss (Comp 7)
│   ├── model.py           DetectionModel carrying the SAR criterion
│   └── register.py        publishes custom layers to ultralytics
├── augmentation/          SAR-specific augmentation (SEC. 5), reusing the corruption model
├── data/                  registry · converters · validator · statistics · leakage
│                          groups.py: source grouping + leave-one-source-out folds
├── training/              trainer · runner · config loader · hard_examples (SEC. 6)
├── evaluation/            COCO AP + scale-wise AP · robustness · efficiency · domain shift
├── visualization/         detections · Grad-CAM (all 8 modules) · feature maps · failure taxonomy
├── tracking/              append-only ledger + environment capture
├── paper/                 LaTeX + table/figure generators (cannot fabricate)
└── cli.py                 python -m saryolo <command>

configs/    datasets/ · models/ (80 generated) · exp/ (EXP-001…019 + EXP-2xx ablations)
scripts/    prepare_dataset · make_exp_configs · train_all_experiments
            make_readme_assets (builds this page's charts) · check_chart_layout
notebooks/  Colab: dataset prep · train + ablate · benchmark + paper
tests/      tests across arch parity, identity, gradient flow, metrics, losses, data, repo
            cross-source grouping and folds
docs/       DATASETS.md · METHOD.md · research_gap.md · PROGRESS.md
            RUNBOOK_SSDD.md · RUNBOOK_HRSID.md · assets/ (generated charts)
```

Model YAMLs and experiment configs are **generated**, never hand-edited:

```bash
python -m saryolo arch --variant all --nc 1        # configs/models/
python scripts/make_exp_configs.py --dataset ssdd  # configs/exp/
```

Because indices in a YOLO YAML are positional, hand-editing an ablation is the single most likely place to introduce a *silent* bug — a wrong index still parses and merely degrades accuracy. The builder computes every index symbolically and asserts the baseline against published counts.

---

## Part X · Reproducibility

Every run appends a row to `results/experiments.jsonl` (plus a derived `.csv`) recording: experiment id, config hash, seed, epochs, batch, image size, optimizer, LR, weights path, git commit, Python / PyTorch / CUDA / Ultralytics versions, GPU name, and measured metrics.

```bash
python -m saryolo ledger
```

The ledger is **append-only**. A rerun never overwrites an earlier result, and failed runs are retained *with their error* — so a table cannot silently include a run that did not finish.

---

## Known limitations, stated up front

- **No real-dataset accuracy exists yet.** Everything in Part IV is infrastructure validation.
- **The adaptive-vs-static rows are load-bearing.** If `att_saa_static` matches `att_saa`, the adaptivity claim is unsupported and the paper must say so. The same applies to `tp_channel` vs `v2_full` for the target prior, `fr_static` vs `v2_full` for the spectral branch, and `rf_local` vs `v2_full` for the deformable refinement.
- **Components 8-11 are unvalidated and may not survive.** They exist because v1 leaves four failures unaddressed, not because they are expected to help. v2 is ~72% larger and ~2.57× the compute of the baseline, so the removal ablation (`EXP-281…286`) can and should delete any component that does not pay for itself. A shorter, cheaper model is a *better* result, not a failure.
- **The deformable offset bound is being saturated and needs a sweep.** In the 2-epoch smoke run the learned offsets already reached `|d| ≈ 0.47` against the `max_offset = 0.5` bound, which means the tanh is operating where its gradient is smallest (`1 − tanh² ≈ 0.11`). That is either the model asking for a larger search radius or a bound set too tight, and the two have opposite fixes — so `max_offset` is a hyperparameter to sweep (a committed variant per value), not a constant to leave untuned.
- **The v2 clutter mode is a mode change, not a new slot**, so EXP-013 adds no module: it changes Component 2's speckle estimator into a three-branch target/speckle/clutter form. Its ablation is the `- clutter` row, not a slot study.
- **HRSID and SAR-Ship-Dataset leak under random chip splits** — chips are cut from a few large scenes. `saryolo.data.splits.leakage_report` exists to catch this; a scene-grouped split is required before trusting mAP on those datasets.
- **Scale-wise AP here is our own implementation**, with deviations from pycocotools documented in `saryolo/evaluation/metrics.py`.
- **Cross-dataset evaluation refuses to run on incompatible label spaces**, rather than reporting a meaningless low mAP.
- **Oriented detection (Component 6) is unimplemented.** The DOTA converter exists; the head does not.
- **SAR augmentation is offline, so it is fixed rather than resampled per epoch.** A training run sees `views` variants of each image instead of a fresh draw every epoch, and the split costs `views`× the disk. That is the price of the augmented set being a committed, byte-reproducible, per-file-attributable artifact; an online transform hook would avoid it and would put the augmentation outside the inspectable path. Stated here because it is a real trade-off, not a detail.
- **Hard-example mining is untested as a *strategy*.** The tooling is verified — determinism, leakage refusal, agreement with the failure taxonomy — but whether oversampling hard images actually improves mAP is an experiment, not a result. Mining adds no architecture, so if it does not help it costs only a run.
- **The difficulty weights are a judgement call, not a measurement.** They weight small-object misses highest because that is the paper's claim. They are explicit function arguments precisely so an ablation can vary them, and the default should not be read as a tuned result.
- **Cross-level scale routing is not implemented, and could not be without patching the parser.** A stage that reweights P2-P5 jointly against one shared prior needs a module consumed by several feature maps. `parse_model` resolves an unknown module's output channels with `c2 = ch[f]`, which raises `TypeError` for a list `from`, and naming a `Detect` subclass as the head fails too because the head branch is a `frozenset` *identity* test. Both were verified empirically rather than assumed. Component 11 is therefore *per-level* refinement, and the plan's "target-aware dynamic scale routing" is reported as unimplemented rather than faked per-level and described as cross-level.

## License

MIT for the code (see [`LICENSE`](LICENSE)). Datasets are **not** covered and must be obtained from their official sources under their own terms.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the original dataset papers too — per-dataset citations are listed in [`docs/DATASETS.md`](docs/DATASETS.md).
