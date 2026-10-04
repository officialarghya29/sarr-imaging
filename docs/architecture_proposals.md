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
| `cfar_conv_n` (matched-cost control) | 2,590,252 | **+217 (identical)** | 1.649 | +0.036 (+2.2 %) |
| `cfar_fixed_n` (fixed-threshold control) | 2,590,035 | **+0** | 1.728 | +0.115 (+7.1 %) |
| `baseline_s` | 9,428,179 | — | 5.394 | — |
| `cfar_s` | 9,428,396 | **+217 (+0.002 %)** | 5.540 | +0.146 (+2.7 %) |

So the parameter claim holds by a factor of ~60 against the 0.5 % budget, the matched-cost
control is parameter-*identical* to the proposed arm by construction, and the fixed-threshold
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
| 6. Initial experiments | **Measured**: `REAL-004` (the arm), `REAL-005` (fixed-threshold control) and `REAL-006` (matched-cost control) on the HRSID subset against `REAL-001`, plus a three-seed paired repeat (`MSEED-B1/B2` vs `MSEED-C1/C2`) — the falsification tests in §5. Reported in §8 |
| 7. Optimization by measured evidence | **Partially done**: the three-seed repeat and the matched-cost control were the first optimizations of the design; both are reported in §8. No other knob has been tuned |
| 8. Benchmarking / ablations / failure analysis | **Partially done**: the five-corruption sweep, the gain-collapse diagnostic and the synthetic acquisition-shift pilot for `REAL-001` / `REAL-004` are all in §8; a second-sensor benchmark is not run (no GPU) |
| 9. Finalization | **Done at pilot scope** — §9 freezes the interface, labels every claim final/preliminary/not-supported, and states the remaining (GPU-gated) critical path. It does not name a detector |

---

## 8. The measured result of the recommended direction (pilot)

Phase 6–8 ran the falsification test from §5. It is a **pilot**: the 200/60/60 HRSID subset,
40 epochs, 320 px, batch 4, one class, **CPU only**, every arm sharing the entire schedule so
a row differs only in the first representation. Six things were added past the first
seed-0 result reported earlier: the **matched-cost control** (`cfar_conv`, REAL-006), a
**three-seed paired repeat** of the primary comparison (MSEED-B1/B2 and MSEED-C1/C2), a
**corruption sweep** under the five built-in `saryolo robustness` corruptions, the
**gain-collapse diagnostic** the proposal names as its own falsifier (`saryolo gain`), and a
**synthetic acquisition-shift pilot** on a global radiometric gain. Every number below is
read from `results/experiments.jsonl`, `results/robustness/*/robustness.json`,
`results/robustness_brightness/*/robustness.json` or `results/gain/*/gain.json` by the run
that produced it.

Seed 0, all four arms on the same schedule (REAL-001 baseline, REAL-004 proposed, REAL-005
fixed-threshold control, REAL-006 matched-cost conv control):

| Arm | mAP50 | mAP50:95 | Precision | Recall | Params | GFLOPs@320 | FPS (CPU) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| REAL-001 · stock YOLO11n (baseline) | 0.5706 | 0.3012 | 0.9240 | 0.5380 | 2.590 M | 1.613 | 61.8 |
| REAL-004 · **CFAR front end, learned gain** | 0.5691 | **0.3151** | 0.9104 | 0.5346 | 2.590 M | 1.764 | 36.7 |
| REAL-005 · same statistic, **fixed** threshold, 0 learnable params | 0.4952 | 0.2465 | 0.7931 | 0.4737 | 2.590 M | 1.728 | 44.1 |
| REAL-006 · matched-cost control: same params as REAL-004, no CFAR statistic | 0.5654 | 0.3021 | 0.9244 | 0.5146 | 2.590 M | 1.649 | 44.9 |

The primary comparison then repeated at three training seeds (baseline arm, then the proposed
arm, both trained and evaluated identically):

| Seed | Baseline mAP50 / mAP50:95 | CFAR mAP50 / mAP50:95 | Paired Δ mAP50:95 |
| ---: | ---: | ---: | ---: |
| 0 (REAL-001 / REAL-004) | 0.5706 / 0.3012 | 0.5691 / 0.3151 | **+0.0139** |
| 1 (MSEED-B1 / MSEED-C1) | 0.5375 / 0.2804 | 0.5571 / 0.3036 | **+0.0232** |
| 2 (MSEED-B2 / MSEED-C2) | 0.5792 / 0.2871 | 0.5838 / 0.3036 | **+0.0165** |
| mean ± std | 0.2895 ± 0.0106 | **0.3074 ± 0.0066** | **+0.0179** |

**What the test settles.** Two falsifiers named in §5 were run. The first was "the front end
does not beat the matched-cost control". REAL-006 is parameter-*identical* to REAL-004 by
construction (a plain `Conv` stem sized to the same count) and reaches mAP50:95 **0.3021** —
below the prototype's 0.3151 and only 0.0009 above the baseline's 0.3012. So the seed-0 gain
is not simply "217 extra parameters and an extra transform"; the analytic statistic is doing
something the plain stem does not. The second falsifier was the seed-noise one, and it is the
stronger result: on mAP50:95 the proposed arm is ahead of the baseline at **all three seeds**
(+0.0139, +0.0232, +0.0165), the per-seed mean gap is **+0.0179**, and the proposed arm's
seed-to-seed spread is smaller than the baseline's (±0.0066 vs ±0.0106). The fixed-threshold
control (REAL-005) is *worse than the plain baseline* (0.4952 vs 0.5706 mAP50), so the
learned gain is load-bearing rather than decoration.

**The corruption sweep points the same way.** The built-in five-corruption sweep
(`saryolo robustness`, 60 test images) was run for REAL-001 and REAL-004. The prototype's
relative mAP50:95 loss under the SAR-relevant corruptions is roughly half the baseline's:

| Corruption | Baseline mean rel. Δ | CFAR mean rel. Δ |
| --- | ---: | ---: |
| speckle | −15.6 % | **−10.7 %** |
| low contrast | −32.0 % | **−13.1 %** |
| clutter | −24.6 % | **−20.9 %** |
| blur | +7.4 % | +6.4 % |
| low resolution | +6.3 % | +5.3 % |

(The blur and low-resolution rows *improve* both arms — an artefact of evaluating 320-px-trained
weights on downscaled then re-upscaled inputs — so they are reported for completeness but carry
no claim.) The worst single operating point, speckle at the lowest looks, is 0.1855 for the
baseline against 0.2291 for the prototype. This is consistent with the mechanism the proposal
claims: a statistic-based front end degrades more slowly as the local statistics get noisier.
It is a **pilot** observation on one subset, not a robustness benchmark.

**The gain is a per-pixel decision, not a collapsed scalar.** The proposal named gain
collapse as the way its own claim could be empty, so the diagnostic was built and run
(`saryolo gain --weights results/runs/REAL-004/weights/best.pt … --limit 60`) over the 60
held-out chips. The trained gain is **not** constant and is **not** a per-scene scalar:

| Quantity (clean, 60 images × 102 400 px) | Measured |
| --- | ---: |
| mean abs. gain | 0.113 |
| within-image std (per-pixel decision) | 0.0331 |
| between-image std (per-scene scalar) | 0.0105 |
| active fraction (\|g\| > 0.05) | 98.2 % |
| min / max gain | −0.431 / +0.152 |

The within-image spread is three times the between-image spread, so the modulation varies
*pixel to pixel inside a scene* rather than being one number per image — which is what the
design claims and what a collapsed gain would contradict. Across synthetic shifts the gain
reacts rather than freezing: under 2-look speckle its within-image std rises to 0.119 (more
modulation where the statistic is noisier), while under heavy contrast compression it
smooths to 0.005 — it learns to apply a nearly uniform darkening there, which is a real
property of the trained map and is reported rather than hidden. This falsifier is **cleared**.

**But the gain is seed-sensitive, and that is a caveat worth its own paragraph.** Re-running the
diagnostic on the seed-1 and seed-2 prototype checkpoints (`MSEED-C1`, `MSEED-C2`) gives a much
weaker modulation than seed 0:

| Checkpoint | mean abs. gain | within-image std | between-image std | active fraction | collapsed |
| --- | ---: | ---: | ---: | ---: | :---: |
| REAL-004 (seed 0) | 0.113 | 0.0331 | 0.0105 | 98.2 % | no |
| MSEED-C1 (seed 1) | 0.021 | 0.0217 | 0.0121 | 6.0 % | no |
| MSEED-C2 (seed 2) | 0.025 | 0.0160 | 0.0131 | 3.7 % | no |

At all three seeds the within-image spread still exceeds the between-image spread, so the gain is
a per-pixel map and never collapses — the falsifier is cleared at every seed. But its *magnitude* is
strongly seed-dependent (mean abs. gain 0.113 → 0.021 → 0.025), which says the detector does not
reliably commit to the sharp per-pixel modulation seed 0 learned. Combined with the modest paired
accuracy gap (§ above), the honest reading is that the mechanism is present but that the training
signal pushing the gain away from the identity is weak on this subset — a plausible part of why the
accuracy effect is small.

**Two synthetic acquisition-shift axes were run, and both are null results, reported as such.**
The first is a global radiometric gain — the acquisition variable the log-ratio channel is exactly
invariant to (a property now pinned by `tests/test_cfar_scope.py`). The same 60 chips were
darkened by a global gain and re-evaluated. It does **not** separate the arms:

| Global gain | Baseline mAP50:95 | CFAR mAP50:95 |
| ---: | ---: | ---: |
| 1.0 (clean) | 0.2889 | 0.3091 |
| 0.7 | 0.3005 | 0.3224 |
| 0.5 | 0.2750 | 0.2991 |
| 0.3 | 0.1365 | **0.1094** |
| 0.15 | 0.0007 | 0.0002 |

At severe darkening both models are on the floor (≈0 mAP as the target contrast vanishes), so
that regime is not a test; at moderate gain (0.5, 0.7) the prototype keeps its small lead; and
at gain 0.3 it is **worse** than the baseline. The honest reading is that the radiometric-gain
axis is dominated by an absolute detection floor and does not discriminate the two arms — the
mechanism's *statistic* is gain-invariant, but that does not translate into a measured
robustness advantage here. A null result is a result, and it is not dropped.

The second axis is **anisotropy** — along-track (azimuth) resolution loss only, the x axis
downsampled and restored while the range axis is untouched, which is the geometry an isotropic
`low_resolution` corruption cannot represent. It is also non-discriminating, and for the same
reason `blur` and `low_resolution` were: both arms *improve*, because smoothing helps this small
subset, and the arms keep their ~0.02 offset throughout.

| Azimuth scale | Baseline mAP50:95 | CFAR mAP50:95 |
| ---: | ---: | ---: |
| 1.0 (clean) | 0.2889 | 0.3091 |
| 0.75 | 0.2930 | 0.3180 |
| 0.50 | 0.2981 | 0.3213 |
| 0.35 | 0.2990 | 0.3218 |
| 0.25 | 0.3059 | 0.3273 |

So neither pure acquisition axis — radiometric or geometric — is where the prototype's (already
small) advantage comes from. The advantage it does show lives on the *texture* corruptions
(speckle, low contrast, clutter), which is consistent with the mechanism but is not the
acquisition-transfer story the motivation tells. That gap is the most useful thing this pilot
produced.

**A third test — training under the SAR degradation model — is a large win for the detector and
another negative for the front end.** The natural follow-up is whether augmenting training with
the same SAR corruption model widens the front end's edge, since the model would no longer see
clean imagery only (`AUG-001`/`AUG-002`, augmented train split, clean test split):

| Context | Baseline mAP50:95 | CFAR mAP50:95 | Gap |
| --- | ---: | ---: | ---: |
| no augmentation (REAL-001 / REAL-004) | 0.3012 | 0.3151 | **+0.0139** |
| SAR-augmented train (AUG-001 / AUG-002) | 0.3571 | 0.3638 | **+0.0067** |

Augmentation lifts the *baseline* from 0.3012 to 0.3571 mAP50:95 (+18.6 % relative) — the
largest accuracy movement anywhere in this repository — and it is worth keeping on that ground
alone. But it does **not** widen the front end's lead; the lead **halves**. The honest reading is
that the corruption model gives both arms much of the robustness the analytic statistic was
providing, so the front end's relative contribution shrinks even as the detector improves. The
hypothesis that augmentation would make the front end look better is **not supported**, and it is
recorded as a negative.

**What the test does not settle.** The absolute effect is small. A mean mAP50:95 gap of
+0.018 measured on **60 test images, one 200-image training subset, one CPU** is above the
seed spread on this sample and below what anyone should call a result; the primary metric
mAP50 is a wash or slightly negative at seed 0 (0.5691 vs 0.5706) and only ahead at seeds 1–2.
The front end also costs real CPU throughput (61.8 → 36.7 FPS, −40 %) at this resolution, so
the efficiency frontier gets *worse* even as accuracy improves. The correct statement is:
**the proposal survives both falsifiers it named — its own matched-cost control and the
seed-noise test — but on a 260-image, single-machine pilot, and the effect is small.** What
would turn this into a claim: three seeds on the full HRSID release and a second sensor
(SSDD or SARDet-100K) to show the statistic is not HRSID-specific. (The gain-collapse
diagnostic is done, above; the acquisition-shift pilot is done and negative; the two that
remain are the ones that need a GPU, and they are the first things a machine with one would
run.)

**What would withdraw it.** If the matched-cost conv stem had matched REAL-004, or if the
three-seed paired Δ had averaged to zero, the direction should be withdrawn rather than
rephrased. Neither happened, so the direction is retained — as the recommended direction to
develop, and still not as a paper result.

---

## 9. Finalization — the frozen spec, and what the pilot does and does not license

Phase 9 is where a name, a spec and a set of claims are made final. This section states what
is frozen, what is only preliminary, and — the part that keeps the earlier sections honest —
what the pilot does **not** license. Nothing here upgrades a pilot measurement into a paper
result.

### 9.1 The frozen interface (final)

The recommended direction is frozen at the following contract, which every test in
`tests/test_cfar_frontend.py` and `tests/test_cfar_scope.py` pins:

| Element | Frozen value |
| --- | --- |
| Class | `saryolo.nn.modules.cfar.RatioSpaceCFARFrontEnd` (registered in `CUSTOM_MODULES`) |
| Placement | row 0 of the backbone, via `ModelSpec.cfar`; a channel-preserving `C -> C` map |
| Modes | `("cfar", "conv", "fixed")` — proposal and the two controls; `arch.CFAR_MODES` must equal the module's list |
| Windows | `(3, 7, 15)`, odd and ≥ 3, validated where the mistake is made |
| Gain width | `hidden = 24`; last layer zero-init **and** zero bias, so `g ≡ 0` at step 0 |
| Identity contract | `cfar` and `conv` are exact identities at init (`max|f(x)−x| = 0.0e+00`); `fixed` declares `identity_at_init = False` on purpose |
| Public surface | `statistics(x)`, `gain_map(x)`, `forward(x)`; `forward ≡ x * (1 + gain_map(x))` |
| Arms | `cfar_n/s`, `cfar_conv_n/s`, `cfar_fixed_n/s` — six arms, one insertion point each |

A paper-style write-up of this interface is drafted in `docs/methods_rs_cfar.md`.

### 9.2 The measured record (every number is in the ledger)

| Id | Arm | Role | mAP50 | mAP50:95 |
| --- | --- | --- | ---: | ---: |
| REAL-001 | YOLO11n baseline, seed 0 | reference | 0.5706 | 0.3012 |
| REAL-002 | SARVO-Lite (s) | efficiency frontier | 0.5724 | 0.3287 |
| REAL-003 | LoRA r=8 | parameter-efficient | 0.5781 | 0.2984 |
| REAL-004 | CFAR front end, learned gain | proposal | 0.5691 | 0.3151 |
| REAL-005 | CFAR statistic, fixed threshold | control (no learning) | 0.4952 | 0.2465 |
| REAL-006 | matched-cost conv stem | control (no statistic) | 0.5654 | 0.3021 |
| MSEED-B1/B2 | baseline seeds 1, 2 | seed spread | 0.5375 / 0.5792 | 0.2804 / 0.2871 |
| MSEED-C1/C2 | front end seeds 1, 2 | pair | 0.5571 / 0.5838 | 0.3036 / 0.3036 |
| AUG-001 | baseline, SAR-augmented train | augmentation control | 0.6060 | 0.3571 |
| AUG-002 | front end, SAR-augmented train | augmentation arm | 0.6049 | 0.3638 |

Reproduce with the four training configs (`REAL-001…006`) followed by
`python -m saryolo robustness …`, `python -m saryolo gain …`, and
`python scripts/make_readme_assets.py`; the exact commands are in
`reports/reproduction_status.md` and the ledger refuses to render an unmeasured cell.

### 9.3 Claim labels (what is final, preliminary, or blocked)

| Claim | Label | Basis |
| --- | --- | --- |
| The module is an exact identity at init and its gain trains | **final** | test-pinned on the real graph, including the first-layer-recovery check |
| The parameter overhead is +217 (< 0.5 %) and scale-independent | **final** | measured on the generated YAML at `n` and `s` |
| The front end beats its own matched-cost and fixed-threshold controls | **preliminary** | one 200/60/60 subset, one machine |
| The front end leads the baseline on mAP50:95 at three seeds | **preliminary** | +0.0179 mean, inside a small-subset envelope |
| The trained gain is a per-pixel decision, not collapsed | **final, but seed-sensitive** | `results/gain/*/gain.json`; magnitude 0.113 / 0.021 / 0.025 across seeds (§8) |
| The front end improves robustness under an acquisition shift | **not supported** | two acquisition axes (radiometric gain, anisotropy) are both null (§8) |
| SAR-appearance augmentation improves the detector | **measured, pilot** | `AUG-001`: baseline mAP50:95 0.3012 → 0.3571 |
| Augmentation widens the front end's lead | **not supported** | gap halves, +0.0139 → +0.0067 (§8) |
| Cross-sensor / cross-resolution generalisation | **blocked on GPU** | needs the full release and a second source |

### 9.4 The naming decision

The repository keeps its name and the direction keeps **RS-CFAR** as a described
*architecture direction*, not as a renamed detector. This is deliberate. The brief's
prohibition is on presenting a module ladder under a new name as a new architecture, and the
pilot is far too thin to justify naming a detector: it must first replicate on a full release
and a second sensor. The frozen spec (§9.1) is what a collaborator would implement; the
measured record (§9.2) is what they would have to beat; and §9.3 is the list of claims that a
draft may and may not make from this repository as it stands.

### 9.5 The remaining critical path

1. The full HRSID release at three seeds per arm — a GPU job.
2. A second sensor (SSDD or SARDet-100K), whose cross-source machinery exists (`saryolo loso`)
   but has never been run on real data.
3. The gain diagnostic re-run on those checkpoints — the tool exists, so this is a re-run,
   not new code.
4. Only then a draft, and §8's negative results (the acquisition-shift pilot) go in the draft
   rather than an appendix.
