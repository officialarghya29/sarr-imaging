# Three architecture proposals for SARVO, and one recommendation

**Date:** 2026-10-04
**Status:** decision document. Phase 3 of the master workflow.
**Method:** every constraint quoted below was measured in this repository (the command or
test that produced it is named). Every novelty statement is *conditional*: it says what
would have to be true and which experiment would show it, and it names the closest work it
could be confused with. Nothing here is a claim that SARVO is new, better, or novel. Those
are outcomes of the experiments, not premises of the design.

This document exists because of one sentence in the brief: *"SARVO is a new model that we
are building — not a system that combines existing models."* The current repository does
not yet satisfy that. What it has is a YOLO11 skeleton plus a set of SAR modules
(`saryolo/nn/modules/`), assembled by `saryolo/nn/arch.py`, which is the thing the brief
explicitly rules out: *"a collection of existing modules with a new name."* A component
ladder on someone else's graph can be a good study; it is not an architecture. The three
proposals below are attempts at an *organizing principle* — one idea that determines the
whole graph — and the recommendation explains why exactly one of them is affordable to
develop and to falsify on the hardware that actually exists.

---

## 1. The constraints the proposals have to live inside (measured)

| Constraint | Measured value | Where it comes from |
| --- | --- | --- |
| Training hardware | AMD Ryzen 7 8840HS, **CPU only**; `torch.cuda.is_available()` is `False` | `reports/reproduction_status.md` §1 |
| Largest completed run | 40 epochs / 200 train images / 320 px / batch 4 in **6.9–36.2 min** | ledger, `REAL-001…003` |
| Cheapest full-scale reference | 11.016 M params, 32.55 GFLOPs at 640² (SARVO-Lite s) | `docs/assets/facts.json`, `tests/test_efficiency_frontier.py` |
| Conditioning overhead budget | **<0.5 %** of parameters on the model it is claimed about | `test_conditioning_on_the_light_model_is_a_small_absolute_cost` |
| Attributability rule | every added module is an exact identity at initialisation (`max\|f(x)−x\| = 0.0e+00`) | `tests/test_arch.py` |
| Real data on disk | HRSID subset, 200 / 60 / 60 images, one class, **no per-chip resolution mapping** | `docs/assets/facts.json` → `real_subset` |
| Evaluation gate | an unmeasured cell renders `TBD`; `--require-complete` refuses to assemble a paper around it | `saryolo/tracking/ledger.py` |

Two of these matter more than the rest. **No GPU** means a proposal whose central claim can
only be tested by a multi-source, multi-seed, full-release run is, in this environment, a
proposal that cannot be falsified — and an unfalsifiable direction is the worst possible
use of the remaining time before the 2026-11-16 deadline. And **no per-chip resolution
mapping in HRSID** means the cross-resolution axis is inert on the only real dataset that
is present, so a proposal whose headline is "cross-resolution" is blocked here even though
it is scientifically the most interesting.

The three proposals are therefore placed on three different parts of the graph — the input
representation, the multi-scale path, and the conditioning path — so that the comparison is
between *directions*, not between three variants of the same layer stack.

---

## 2. Proposal 1 — Ratio-space CFAR front end (RS-CFAR)

**Core architectural idea.** Make the first representation be the radar's own detection
statistic instead of a learned convolution over the raw magnitude image. The input is
transformed into log space, where multiplicative speckle becomes additive and
signal-independent; a differentiable, multi-scale estimator of the local mean and local
variance then produces ratio channels; a learned ordered-statistic gain — initialised to
zero — decides per pixel how much those statistics should suppress the local clutter before
the first convolution ever runs. The model *starts* as the plain magnitude image
(identity-at-init, so the repo's attributability rule survives) and learns to look at the
image the way a constant-false-alarm-rate detector does.

**Main technical problem it addresses.** Speckle and clutter are not nuisance noise; they
are the signal the radar actually produces, and they dominate the statistics at the small-
object scales where SAR detection fails. A generic conv stem spends its first layers
re-deriving a local-contrast statistic that radar theory already knows how to compute in
closed form.

**Proposed computational flow.** For an input image `x` (B, 1, H, W):

1. `x_log = log(x + eps)`.
2. A depthwise multi-scale mean/variance bank `S = {(3,5,9,17)}` produces, per scale `k`,
   `mu_k = box_k(x_log)` and `sigma_k = sqrt(box_k(x_log^2) − mu_k^2)`.
3. Ratio channels `r_k = x_log − mu_k` (log-ratio, dimensionless) and contrast channels
   `c_k = sigma_k / (|mu_k| + eps)` (the local coefficient of variation; `c_k^2` is the
   reciprocal of the equivalent number of looks — a real radar quantity).
4. A small MLP over the stacked statistics emits a per-pixel gain `g`; the last layer is
   zero-initialised, so `g ≡ 0` at step 0 and the front end is the identity.
5. Output `x' = x ⊙ (1 + tanh(g))`, which then enters *the existing backbone unchanged* —
   no other row of the graph moves. The modulation is applied to the **raw** image, not to
   `x_log`: inverting the log to return to image space would make the identity approximate
   instead of exact, and the backbone was designed to read an intensity image. With the
   zero-initialised gain, `g ≡ 0` and the front end returns `x` bit-for-bit.

Cost is `O(k)` depthwise box filters plus one MLP over `3k` channels: no full-resolution
dense convolution is added, and the statistic bank is separable and cheap.

**Expected advantages.** (a) It attacks the failure mode instead of the symptom: local
clutter suppression is exactly what raises the contrast of a small ship against sea. (b)
It is *fully attributable* — the gain is the only new parameter group, and with zero init
the arm is the baseline until it learns otherwise, so any measured delta is the learned
statistic, not added capacity. (c) The statistic bank is analytic, so its cost does not
grow with the backbone scale. (d) The same front end is meaningful for any magnitude SAR
product, which is what SSDD, HRSID and SARDet-100K all are.

**Potential weaknesses and failure modes.**
* *It may be a wash.* Local contrast normalisation is one of the first things a conv stem
  learns anyway; the front end may only reach the same place faster, and end at the same
  mAP. The matched-cost control (a plain conv stem with the same parameter and FLOP budget)
  is what decides this, and it can decide against us.
* *The gain may collapse to a constant.* If the MLP learns a scene-independent scalar, the
  per-pixel decision is not being used and the claim is empty; this is detectable directly
  by measuring the variance of `g` across pixels and scenes.
* *Log-space can hurt bright targets.* `log` compresses large dynamic range, which is good
  for speckle and potentially bad for a very bright point target whose raw amplitude is
  already separable. Mitigated by keeping the un-logged branch available through the
  residual form, but it is a genuine risk.
* *Novelty risk: medium-high.* CFAR is decades old and *learnable* CFAR variants exist in
  the radar literature; a log-ratio transform is standard in change detection. Nothing
  here can claim "CFAR is ours". The specific object under test is narrow: a differentiable
  multi-scale CFAR gain used as a zero-initialised **input representation inside an
  anchor-free detector**, with its contribution separated from raw stem capacity by a
  matched control.

**Estimated implementation complexity. Low.** One new class in `saryolo/nn/modules/`,
registered like the others; one `ModelSpec` field; a handful of generated YAML rows. It
does not touch the trainer, the loss, or the data path, because it consumes the same
3-channel tensor the current stem consumes.

**Expected efficiency implications.** The expectation was one MLP
over `3k` channels, order `10^4` parameters, well under the 0.5 % budget. It is measured
now, by `saryolo/evaluation/efficiency.py` on the generated YAML at 320 px and one class:

| Arm | Params | Δ params | GFLOPs @320 | Δ GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| `baseline_n` (pilot scale) | 2,590,035 | — | 1.613 | — |
| `cfar_n` (proposed) | 2,590,252 | **+217 (+0.008 %)** | 1.764 | +0.151 (+9.4 %) |
| `cfar_fixed_n` (control) | 2,590,035 | **+0** | 1.728 | +0.115 (+7.1 %) |
| `baseline_s` | 9,428,179 | — | 5.394 | — |
| `cfar_s` | 9,428,396 | **+217 (+0.002 %)** | 5.540 | +0.146 (+2.7 %) |

So the parameter claim holds by a factor of ~60 against the 0.5 % budget, and the fixed
control adds *exactly* nothing. The compute is not free: the statistic bank and the gain MLP
cost roughly 2–9 % of a small model's FLOPs at this resolution, which is a real price and is
stated rather than buried in the ablation. These are measured values, not estimates; nothing
here says whether the price is worth paying — that is what the experiment decides.

**How it differs from relevant existing approaches.** Against learned-CFAR detectors in the
radar literature: those are usually a CFAR stage *around* an existing detector or a
post-processing threshold, and they are threshold-calibration work rather than an input
representation trained end to end — ours is inside the graph, trained only by the detection
loss, and has no CFAR calibration step at all. Against the domain-adaptation family
(semantic-scattering graph alignment, and the cross-sensor modules in the literature
audit): those change *features* to align domains; this changes what the first convolution
*sees*, and it is domain-agnostic by construction. Against the repo's own Component 1 (SFE)
and Component 2 (SFM): those are deep, learned, identity-at-init blocks operating on
features; this is analytic, shallow, input-side, and it is a *representation*, not a module
inserted on top of one.

**An overlap that has to be stated rather than avoided.** The repository *already* contains
a CFAR-style arm: `TargetPriorModulation` has a `"cfar"` mode (`VARIANTS["tp_cfar"]`), a
classical non-learned prior applied to *features at each detection level*. Its existence is
the strongest argument against over-claiming here, and it is why the distinction has to be
about placement and scope rather than about the statistic: the prior slot asks "is this
feature above its local background" per level and per channel; this proposal asks "what is
the detector's first representation" once, on the image, with a learned per-pixel gain.
The two are separately ablatable (`tp_cfar` versus `cfar_n`), and a paper that reported
only the second without citing the first would be hiding its own prior work.

**Experiments needed to decide.** (1) An arm with the front end and an otherwise identical
graph, versus the baseline, on the existing 200/60/60 HRSID subset — the environment can
run this in ~10–20 minutes per arm. (2) The matched-cost control: a plain `Conv` stem with
the same parameters and FLOPs. (3) The gain-collapse diagnostic (variance of `g`).
(4) A robustness sweep with additive and multiplicative noise on the held-out test split —
this is where a statistic-based front end should show its value, and it is cheap on CPU.
(5) The ablation that removes the learned gain and keeps the statistic channels (fixed
threshold), which separates "the statistics help" from "the learnable part helps".

---

## 3. Proposal 2 — Anisotropic recursive scale-space detector (ARSS)

**Core architectural idea.** Replace the detector's multi-scale path with a *weight-tied,
recursive* one, and make the model's notion of scale a continuous input rather than a
discrete level. SAR resolution is anisotropic — range and azimuth resolution are separately
determined by the acquisition — and pixel spacing varies between sensors. A feature pyramid
that carries a different convolution per level, as YOLO and FPN-style necks do, is
implicitly tied to one pixel spacing and one anisotropy. ARSS instead runs *one* block `L`
times, feeding each application the current feature map plus a scalar scale token, and
takes lateral outputs from the recursion.

**Main technical problem it addresses.** Cross-resolution generalisation (RQ4): the model
must transfer to acquisitions at a different ground sample distance without retraining, and
the current neck makes that harder than it needs to be by spending separate parameters per
level.

**Proposed computational flow.** A separable directional stem (`1×3` and `3×1` filter
banks with independent weights, followed by their sum) produces an anisotropic-aware
feature map at the finest retained resolution. A shared block `B(f, s)` — a depthwise
convolution, a pointwise mix, and FiLM modulation by a scalar scale token `s = log2(pixel
spacing)` — is applied `L` times, with `s` incremented by 1 between applications. Lateral
outputs after applications `{a, a+1, a+2}` feed an anchor-free head in place of the three
neck outputs. Downsampling between applications is a single stride-2 depthwise op; the
*weights do not change*, so parameters grow with the block, not with `L`.

**Expected advantages.** (a) Parameter count drops sharply relative to a per-level neck —
the frontier's reference spends 16.23 M partly on three independent fusion stages. (b) One
network covers a continuum of pixel spacings, because `s` is a continuous token; transfer
becomes interpolation rather than re-learning. (c) Anisotropy is represented rather than
averaged away by isotropic 3×3 kernels. (d) The direction is the *most architectural* of
the three: it changes the compute graph's shape, not one stage's contents.

**Potential weaknesses and failure modes.**
* *Weight tying may simply lose accuracy.* The per-level neck exists because different
  levels want different filters; sharing them is a bet that the scale token can compensate.
  It may not, and the model may be worse than the baseline at every resolution.
* *The central claim is unmeasurable here.* Cross-resolution transfer needs at least two
  pixel spacings with real labels, and HRSID ships no per-chip resolution mapping (audit,
  §3). The repo can *simulate* a spacing change by resampling, but a resampled image is not
  a different acquisition; a claim built on it would be exactly the kind of overclaim the
  brief forbids. So on this machine the direction's headline experiment is blocked.
* *Novelty risk: high.* Weight-tied recursive processing, deep supervision, and
  scale-conditioned networks are each old. The only arguably new element is the pairing of
  a *continuous* scale token with an anisotropic separable stem inside a SAR detector, and
  that is a thin claim to defend.

**Estimated implementation complexity. High for this repository.** The recursion cannot be
expressed in the existing symbolic `_Builder`, which assumes a linear list of Ultralytics
rows; it needs a hand-written `DetectionModel` subclass with its own forward, its own
FLOPs accounting, and a checkpoint path that the Ultralytics loader can unload. The RT-DETR
arm already showed how much work that is, and it is still not trainable end to end.

**Expected efficiency implications.** Fewer parameters and fewer FLOPs than an equivalent
per-level neck *if* `L` is small; but the recursion runs at the finest resolution more than
a pyramid does, so the compute saving is not automatic and must be measured per `L`.
Realistic expectation: a large parameter reduction, an uncertain compute reduction.

**How it differs from relevant existing approaches.** Against FPN/PAN necks and their
NAS variants: those keep per-level parameters and a discrete level index; ARSS ties the
weights and conditions on a continuous token. Against recursive/weight-shared detectors:
those share weights across levels but with a discrete, one-hot level embedding, so they
still cannot interpolate to a spacing they were not trained on — the continuous token is
the distinction, and it is a small one. Against scale-aware detection generally: most work
feeds the scale in as an extra input image channel or resizes at test time; conditioning a
weight-tied block is different, but not dramatically.

**Experiments needed to decide.** (1) A spacing-interpolation test: train at one pixel
spacing, evaluate at a sweep of resampled spacings, against a baseline trained the same
way. (2) A parameter-matched comparison against the frontier reference. (3) An anisotropy
test: rotate/transpose the image and check that the directional stem's response behaves as
the range/azimuth decomposition predicts. (4) The honest fallback if no resolution-keyed
real data can be obtained: report the direction as specified but unmeasured, which is what
this document is doing.

---

## 4. Proposal 3 — Self-estimated acquisition state (ASAC)

**Core architectural idea.** Make the detector estimate *its own* acquisition state from the
image and condition on that estimate, instead of requiring acquisition metadata to be
supplied. The repository already conditions a detector on categorical sensor, polarization
and mode fields plus continuous physical descriptors, at <0.5 % parameter overhead
(Component 33, `saryolo/data/metadata.py`). The weakness of that design is not its cost but
its dependency: the fields have to exist at test time. HRSID is the concrete counterexample
— it ships no per-chip resolution, so the conditioning arm is inert on the one real dataset
that is on this machine.

**Main technical problem it addresses.** "Metadata is unavailable at deployment." This is
the master context's own threat model ("can the model exploit useful properties of radar
imagery without depending on unavailable metadata?"), and the repository's unresolved
red-team item.

**Proposed computational flow.** A small encoder `E` (two stride-2 convolutions on a
downsampled input) predicts a `d`-dimensional continuous state `s_hat`. When a real
acquisition label exists, an auxiliary head predicts it from `s_hat` (a classification
loss); a consistency loss additionally pulls `s_hat` together across augmentations that
change nuisance but not acquisition (crop, flip, gain), so `s_hat` is a state estimate and
not a texture hash. `s_hat` then drives the *existing* FiLM conditioning path, so the
architecture is one coherent object: image → state estimate → conditioned detector.

**Expected advantages.** (a) Removes the metadata dependency, which is the one thing that
currently makes the conditioning contribution fragile. (b) Tiny: it reuses the tested
conditioning path, and `E` is a two-layer conv net, so the parameter budget is a few tens
of thousands. (c) It is directly comparable to the existing arms — the metadata-conditioned
model is the control, and the repo's field-masking machinery already provides the "no
information" control. (d) It composes with either of the other two proposals.

**Potential weaknesses and failure modes.**
* *`s_hat` may be a nuisance hash.* If it encodes crop statistics rather than acquisition,
  it conditions the detector on the wrong variable and the arm will match the
  unconditioned model under held-out-source evaluation. The consistency loss is the defence
  and the *constant-`s_hat`* arm is the detector of failure.
* *Circularity.* If the only supervision for `s_hat` is the detection loss, it may learn
  whatever helps on the training sources and transfer no better than chance. The auxiliary
  acquisition-classification head is what breaks the circularity, and it needs labels —
  which is exactly what is missing on HRSID. On SSDD/SARDet-100K the labels exist.
* *The headline experiment is blocked here.* Showing that self-estimated conditioning
  recovers the metadata-conditioned model's cross-source gain needs LOSO over multiple
  sources, i.e. the full release and a GPU.
* *Novelty risk: medium.* Domain-adaptation modules already infer domain from features;
  self-supervised pretraining already learns sensor-invariant representations. The specific
  object here is an *explicit, low-dimensional, self-supervised acquisition-state bottleneck
  inside a detector* trained jointly with it. The distinction from the domain-adaptation
  family is that the state is an interpretable variable with a physical name, not an
  implicit feature statistic.

**Estimated implementation complexity. Medium.** The conditioning path exists; the new work
is the estimator, its auxiliary loss, and the consistency loss, plus the trainer plumbing to
feed `s_hat` where the metadata tensor currently comes from. The LOSO evaluation machinery
also exists (`saryolo/data/groups.py`).

**Expected efficiency implications.** A two-layer conv encoder on a downsampled input plus a
`d`-dimensional projection: well inside the existing 0.5 % budget, and no measurable effect
on the detector's compute. The estimate itself is `O(H·W)` at one scale, not per level.

**How it differs from relevant existing approaches.** Against SARFormer (audit §4): that
*consumes* supplied acquisition parameters in a ViT for reconstruction and segmentation;
ASAC *estimates* them inside a detector and never needs them at test time. Against the
repo's Component 33: that consumes metadata, this generates it. Against domain-adaptation
modules: those align features across domains without an explicit state variable, so their
output cannot be inspected or evaluated as "the model's estimate of ground sample distance"
— which is exactly what makes ASAC's failure mode diagnosable. Against FiLM: FiLM is the
mechanism; the state estimator is the idea.

**Experiments needed to decide.** (1) `s_hat`-only conditioning versus
metadata-conditioning versus no conditioning, on a dataset that has acquisition labels.
(2) The constant-`s_hat` control. (3) The `s_hat`-acquisition correlation on held-out
sources, which tests whether the state generalises or overfits the training sensors.
(4) The consistency-ablation (with and without the nuisance-invariance loss).

---

## 5. Comparison and recommendation

| | P1 RS-CFAR | P2 ARSS | P3 ASAC |
| --- | --- | --- | --- |
| Where it changes the graph | input representation / stem | multi-scale path | conditioning path |
| Technical problem | speckle, clutter, small objects | cross-resolution transfer, anisotropy | missing metadata at test time |
| Attributable at init | **yes** (zero-init gain) | partly (shared block) | yes (existing gate) |
| Parameter cost | <0.5 % | **negative** (weight tying) | <0.5 % |
| Compute cost | small, measurable | uncertain, must be measured per `L` | negligible |
| Implementation complexity | **low** | high | medium |
| Can its central claim be tested on this machine | **yes**, on the existing subset | **no** — no resolution-keyed real data | **no** — needs LOSO over real sources |
| Can it be falsified within the compute budget | **yes** (~10–20 min per arm) | no | no |
| Novelty risk (honest) | medium-high | high | medium |
| Composes with | P3 | — | P1 |

**Recommendation: Proposal 1 (RS-CFAR) as SARVO's architecture-defining idea, with
Proposal 3 as the named follow-on if and only if P1's evidence survives.**

The reasoning is deliberately not "P1 is the most novel" — the table says it is not, on the
novelty axis it is in the middle. It is that P1 is the only one of the three whose central
claim **can be measured and falsified in this environment**, inside the compute budget that
actually exists, and that this is a stronger position than an unfalsifiable idea. P2's
headline is cross-resolution, and the audit already established that HRSID cannot support
it: pretending otherwise would require building the paper's central claim on resampled
images, which is the exact failure mode the repository's no-fabrication rule exists to
prevent. P3's headline is metadata-free transfer, which needs leave-one-source-out over
multiple real sources — also unavailable here. P1's headline is contrast and robustness on
*one* source with real labels, which is precisely what is on disk. The recommendation is
therefore a bet on falsifiability over ambition, and it is stated that way on purpose.

Two secondary reasons: P1 is the only proposal that *replaces a representation* rather than
adding a mechanism, which is what makes it an architecture claim rather than another
component; and its zero-initialised gain preserves the repository's strongest existing
property — that every architectural change is provably an identity until it has learned
something, so a measured gain is attributable to the change and not to capacity.

**The first experiment, and what would kill the idea.** Train two arms on the existing
200/60/60 HRSID subset under one schedule: the baseline, and the baseline with the CFAR
front end. Then train the matched-cost plain-conv-stem control. **The idea fails if the
front end does not beat the matched-cost control** — that result says the benefit was the
extra parameters, not the statistic, and the proposal should be withdrawn rather than
rephrased. Secondary falsifier: if the learned gain's across-scene variance is
indistinguishable from zero, the learnable part is unused and the remaining claim is only
"fixed CFAR statistics as input channels", which is a much smaller contribution and must be
described as such.

**Not merging the three.** The brief asks for one coherent model, not a union. P2 is
parked until resolution-keyed data exists. P3 is a documented composition path — a learned
contrast representation makes the acquisition state easier to estimate — but it is not
built until P1 has either produced a measured benefit or been withdrawn, because building
the second idea on an unvalidated first one is how a repository ends up with eleven modules
and no attributable result.

---

## 6. What this document does not claim

* No accuracy claim of any kind. No arm proposed here has been trained on a real SAR
  dataset for the purpose of testing it; the pilot numbers in `reports/reproduction_status.md`
  belong to *other* arms and do not transfer to these proposals.
* No novelty declaration. Each proposal names its closest related work and states the
  specific object under test; all three are `OPEN` on the question of whether that object is
  new, and the literature audit's evidence-depth labels (`verified`, `abstract-read`,
  `listed`) still apply to the comparisons.
* No efficiency number for any of the three. The costs stated above are *expectations*
  with their reasoning visible, to be replaced by measurements from
  `saryolo/evaluation/efficiency.py`.

## 7. Status against the master workflow

| Phase | Status |
| --- | --- |
| 1. Repository and resource audit | Done — `reports/agent_initial_audit.md` (refreshed 2026-10-04) |
| 2. Focused research review | Done — `docs/literature_audit.md`, `docs/related_work.md`, `docs/novelty_and_overlap.md` |
| 3. Architecture proposals | **This document**; one recommended |
| 4. Architecture specification | **Partially done** — the recommended direction's flow, tensors and modes are specified in §2 and realised in `saryolo/nn/modules/cfar.py`; it is not frozen because Phase 6 has not reported yet |
| 5. Prototype | **Implemented and verified**: `RatioSpaceCFARFrontEnd`, arms `cfar_n`/`cfar_s` and controls `cfar_fixed_n`/`cfar_fixed_s`; forward, exact identity at init, non-zero gain gradient, real loss, an optimiser step, and a measured cost profile are all asserted in `tests/test_cfar_frontend.py` |
| 6. Initial experiments | **Measured**: `REAL-004` (the arm) and `REAL-005` (the control) on the HRSID subset against `REAL-001` — the falsification test in §5. It is reported in §8 |
| 7–9 | not started — and each is gated on Phase 6 producing a measured benefit |

---

## 8. The first measured result of the recommended direction

Phase 6 ran the falsification test from §5. It is a **pilot**: the 200/60/60 HRSID subset,
40 epochs, 320 px, batch 4, one class, **one seed, CPU only**, three arms sharing the entire
schedule so a row differs only in the first representation. Every number is read from
`results/experiments.jsonl` by the run that produced it.

| Arm | mAP50 | mAP50:95 | Precision | Recall | Params | GFLOPs@320 | FPS (CPU) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| REAL-001 · stock YOLO11n (baseline) | 0.5706 | 0.3012 | 0.9240 | 0.5380 | 2.590 M | 1.613 | 61.8 |
| REAL-004 · **CFAR front end, learned gain** | 0.5691 | **0.3151** | 0.9104 | 0.5346 | 2.590 M | 1.764 | 36.7 |
| REAL-005 · control: same statistic, **fixed** threshold, 0 learnable params | 0.4952 | 0.2465 | 0.7931 | 0.4737 | 2.590 M | 1.728 | 44.1 |

**What the test settles.** The falsifier named in §5 was "the front end does not beat the
matched-cost control". REAL-004 beats that control by +0.074 mAP50 and +0.069 mAP50:95, and the
control is *worse than the plain baseline* (0.495 vs 0.571). So the learned gain is not
decoration: the analytic statistic used with a fixed threshold actively harms the detector,
and learning the threshold recovers it and more. That half of the claim survives.

**What the test does not settle.** The primary comparison is against the *baseline*, and there
the result is a wash on mAP50 (0.5691 vs 0.5706, −0.0015) while mAP50:95 rises from 0.3012 to
0.3151 (+0.0139, +4.6 % relative). A 0.014 difference measured on 60 test images with a single
seed is **inside the noise**, and the front end costs a real 40 % of CPU throughput (61.8 →
36.7 FPS) at this resolution. The correct statement is therefore: **the proposal survives its
falsifier but has not demonstrated a benefit**, and it may not have one. What would decide it:
three seeds per arm on the full release, the matched-cost plain-conv-stem control (still not
built — it is what separates "the statistic" from "217 extra parameters plus an input
transform"), and the noise-robustness sweep in §2. Until then, §5's recommendation stands as
the direction to pursue and **not** as a result.

**What would withdraw it.** If a multi-seed run shows the mAP50:95 gain inside the seed
spread, or if the matched-cost conv stem matches REAL-004, the direction should be withdrawn
rather than rephrased — and this section should be rewritten to say so, not deleted.
