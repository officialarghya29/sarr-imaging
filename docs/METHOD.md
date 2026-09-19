# Method

This document states what each component computes, why it is shaped that way,
and — for each one — what the ablation that could falsify it looks like.

The design rule throughout: a module is added only if it addresses a *measured*
SAR problem, and it must be removable so its contribution can be measured alone.

---

## 0. Design constraints that shaped everything

**Constraint 1 — identity at initialisation.** Every module is written as

```
F' = F + alpha * (g(F) - F),      alpha = tanh(raw),  raw initialised to 0
```

so `F' == F` exactly, in finite precision, before any training. Consequences:

* a freshly built SAR-YOLO is *numerically identical* to its YOLO baseline
  (asserted in `tests/test_arch.py` with `max abs diff == 0.0`);
* any accuracy gain is attributable to learned behaviour rather than to the
  extra parameters changing the initial function;
* a module that does not train well degrades gracefully toward the baseline
  instead of corrupting it.

**Constraint 1b — identity comes from the gate *alone*.** The residual branch
must **not** also be zero-initialised. The two together look harmless and are
fatal:

```
F' = F + alpha * branch(F)

branch = 0 at init  =>  dL/dalpha   = <dL/dF', branch>      = 0
                       dL/dbranch  = alpha * dL/dF'          = 0   (alpha = 0)
```

Both gradients vanish together, so `alpha` never leaves 0 and the branch never
learns: the module is a permanent no-op that passes every identity test. This is
not a hypothetical failure mode — it was a real bug in Component 1, where both
the gate and the residual conv were zero-initialised, so the learned enhancement
never trained and `+SFE` would have measured only its two affine scalars.

The rule enforced by `tests/test_arch.py::test_no_module_is_frozen_at_init`:
**the gate provides the identity; the branch provides the gradient.** Branches
therefore start small-but-non-zero, and the invariant asserted is a non-zero gate
gradient for every learnable mode.

**Constraint 2 — channel preserving and single-input.** Each module returns the
same channel count it received and consumes one tensor. This is what allows them
to be dropped into a YOLO YAML without patching Ultralytics' `parse_model`
(which resolves an unknown module's output channels as `c2 = ch[f]` and fails
for multi-input rows). The practical benefit is that the stock parser, model
summary, FLOPs counter, validator and checkpointing all keep working.

**Constraint 3 — no kernel may see the raw intensity only.** SAR is a
*statistical* imaging modality. A 3×3 convolution over raw intensities cannot
distinguish "bright uniform clutter" from "bright structured target", because
both look like large positive activations. Every module therefore has access to
local first- and second-order statistics.

---

## Shared primitive: local statistics

For a feature map `F`, with a k×k average pool `A`:

```
mu  = A(F)                          # local mean     -> slowly varying reflectivity
var = max(A(F^2) - mu^2, 0)         # local variance -> speckle strength
sd  = sqrt(var + eps)
hp  = F - mu                        # high-pass     -> fine structure / speckle
S   = (F - mu) / sd                 # local contrast normalisation
```

`S` is the shared SAR-specific primitive. Local contrast normalisation is what
equalises clutter while preserving compact bright target returns, in the same
spirit as classical SAR despeckling but as a differentiable feature operator
rather than a pre-processing filter.

---

## Component 1 — SAR Feature Enhancement (SFE)

**Problem.** SAR returns are speckle-dominated with compressed dynamic range, so
raw intensity features under-represent target structure. But aggressive
denoising is the wrong fix: it removes the high-frequency evidence (point
returns, dihedral edges) that detection depends on.

**Formulation.**

```
S = (F - mu) / sd
Z = phi([F ; S ; sd])
F' = F * gamma + beta + alpha * Z
```

with per-channel learnable `gamma`, `beta` and the zero-initialised gate `alpha`.

**Why additive rather than filtered.** Enhancement is expressed as a correction
to the feature, never as a replacement. At init `gamma = 1`, `beta = 0`,
`alpha = 0`, so SFE is the identity; training can only add information.

**Baselines in the same slot** (`pre_*`): `identity`, `log` (log compression),
`standardize` (pure local contrast normalisation), `clahe` (a differentiable
CLAHE analogue with a learnable clip limit). Comparing against these is what
distinguishes "we do contrast enhancement" from "our particular parameterisation
of contrast enhancement helps".

---

## Component 2 — Speckle-Aware Feature Module (SFM)

**Problem.** Speckle is *multiplicative* and signal-dependent: it scales with
local reflectivity. It therefore cannot be removed by a fixed filter without
also attenuating bright target returns. What a detector actually needs is not a
denoised feature but the ability to tell a high-variance *target* response from a
high-variance *speckle* response.

**Formulation.**

```
N = sigma(psi([F ; hp ; var]))     # per-channel speckle/clutter likelihood in [0,1]
T = dwconv(F)                      # structure-preserving target extractor
F' = F + alpha * (T - beta * N * F),      beta = sigmoid(.)
```

Three deliberate choices:

1. **The suppression term is multiplicative in `F`** (`N * F`, not `N`). Because
   speckle intensity scales with reflectivity, a constant offset would suppress
   weak and strong regions by the same absolute amount — which is exactly wrong.
2. **`beta` is a sigmoid, not free.** It cannot change sign, so the module cannot
   learn to *amplify* speckle. This keeps the operator interpretable.
3. **The noise head is initialised to "no noise"** (output bias `-2`), so the
   residual starts as a pure target extractor.

**Baselines in the same slot** (`spk_*`): `none`, `lee` (the classical adaptive
Lee filter, `k = var/(var + noise_var)`), `denoise` (fixed low-pass blend).

---

## Component 3 — SAR-Adaptive Attention (SAA)

**Problem.** SE, ECA and CBAM fuse their branches with *fixed* weights, identical
for every input. In SAR the right emphasis varies with the imaging regime: a
low-SNR scene needs more spatial/contrast weighting, a bright-clutter scene needs
more channel selectivity. A fixed fusion cannot express that.

**Formulation.**

```
d = Descriptor(F) = [avg_c F ; max_c F ; std_c F]
a = sigma(MLP_c(d))                        # channel attention        (B, C, 1, 1)
s = sigma(conv([mean_c F ; max_c F ; var_c ; sd_c]))   # spatial attention (B, 1, H, W)
l = sigma(psi([ |S|_c ; |hp|_c ]))         # local-contrast evidence  (B, 1, H, W)
w = softmax(MLP_w(d))                      # per-sample branch weights (B, 3)
M = w0*a + w1*s + w2*l
F' = F + alpha * (F * M - F)
```

Two properties distinguish this from "CBAM with extra steps":

* **`std_c` in the descriptor.** Speckle strength is a second-order statistic; a
  descriptor of only mean and max cannot separate uniform clutter from a
  high-variance target region.
* **Input-dependent branch weighting.** `w` is predicted per sample, so the
  network can switch regime per image.

**The falsifying ablation.** `att_saa_static` is the *identical* block with `w`
replaced by learned constants shared across the batch. If it matches `att_saa`,
the adaptivity claim is unsupported and must be dropped. `att_se`, `att_eca` and
`att_cbam` provide the standard-component comparison in the same slot.

---

## Component 4 — Adaptive Multi-Scale Fusion (AMF)

**Problem.** SAR targets span a wide scale range within one scene. A standard
PAN-FPN `Concat` gives every input level the same fixed influence, so a small,
speckle-obscured target can be swamped by a large high-energy neighbour.

**Formulation.** AMF is inserted immediately after each neck `Concat`. Let the
concatenated channels split into groups `F_1..F_G`, one per fused scale:

```
d_g = Descriptor(F_g)
w   = softmax(MLP([d_1 ; ... ; d_G]))          # per-sample, per-scale weights
F~  = [ w_1 F_1 ; ... ; w_G F_G ]
M   = sigma(dwconv(psi(F~)))                    # cross-scale channel gating
F'  = F + alpha * (F~ * M - F)
```

**Group sizes are resolved at build time** from the parser's channel list, so
the split is correct for any width/scale multiplier — and the module *validates*
that the groups sum to its input width, so a stale configuration fails loudly
instead of silently mis-weighting.

**Baselines in the same slot** (`fus_*`):

* `concat` — stock behaviour (the block returns its input);
* `add` — projected additive fusion. A plain sum is **not defined** here because
  PAN-FPN branches have unequal widths (the top-down branch is wider than the
  backbone branch), so each branch is 1×1-projected to a shared width, summed,
  and projected back. This is how residual `Add` fusion is implemented whenever
  widths differ, and it is documented as such rather than presented as a plain sum;
* `static` — the proposed block with input-independent learned scale weights.
  **This is the falsifying run for the adaptivity claim.**

---

## Component 5 — P2 small-object detection head

Adds a stride-4 detection level, extending the neck downward:

```
P3_out --upsample--> Concat(P2) --> C3k2 --> P2_out          (P2/4)
P2_out --stride-2--> Concat(P3_out) --> C3k2 --> P3'         (P3/8)
```

**Only justified by evidence.** `saryolo.data.statistics` reports the COCO
size distribution. If `small` does not dominate, this head should be dropped —
it costs the most compute of any component (roughly 2.5× the baseline's GFLOPs
at 640px) for the least certain benefit. `tests/test_data.py` pins the size-bin
thresholds to `32^2` / `96^2` in **area**, because binning linear sizes against
32/96 misclassifies a 40×40 object as "large" and would wrongly justify this head.

---

## Component 7 — SAR-aware loss

Kept auxiliary and switchable. With every weight at 0 the objective is
numerically identical to stock `v8DetectionLoss` (the loss vector simply carries
a fourth zero entry), which is what makes the `+SAR loss` ablation row a
measurement of the loss rather than of a changed training regime.

```
L_total = L_box + L_cls + L_dfl
        + w_sep    * L_sep        # target/background separation
        + w_small  * L_small      # small-object confidence emphasis
        + w_smooth * L_smooth     # background speckle regularisation
```

**`L_sep` — target/background separation.**
SAR scenes are overwhelmingly background. A plain BCE can reduce its loss by
shrinking *all* foreground confidence, which costs recall on weak, low-contrast
targets. So instead:

```
L_sep = relu(margin - ( mean_conf(foreground) - mean_conf(hardest background anchors) ))
```

using the top `bg_frac` background anchors by score, so the model is rewarded for
*separating* target from clutter rather than for being uniformly unconfident.

**`L_small`.** Small targets occupy few anchors, so their gradient contribution
is diluted across the batch. This adds an extra confidence objective restricted
to anchors assigned to COCO-small ground-truth boxes — the loss-level counterpart
of the P2 head.

**`L_smooth`.** Speckle creates isolated high-variance background responses. This
penalises total variation of the *background-masked* mean score map at every
level, suppressing scattered activations while leaving foreground untouched.

All three terms are computed from quantities the baseline criterion already
produces (`fg_mask`, `target_bboxes`, `target_gt_idx`, `preds["scores"]`), so
they cost no extra assignment pass. The weights live in the model YAML's
`sar_loss` block, so a run is reproducible from the committed YAML, and an
unknown key raises rather than being silently ignored.

---

## Component 8 — Target Prior Modulation (TPM)

The central hypothesis of the work. Components 1-4 improve *how features are
formed*; this one asks whether the network can be given an explicit, learned
representation of **where target-like structure is**, and whether that
representation is useful when it is actually applied to the detection features
rather than merely visualised.

```
mu_n, sd_n = LocalStats_k(F)                 near-scale local statistics
sd_w       = LocalStats_(2k+1)(F)            wide-scale local statistics
e          = psi([F ; F - mu_n ; sd_n ; sd_w])   target evidence  (1 channel)
M          = 2*sigmoid(e) - 1                signed prior, in (-1, 1)
F'         = F * (1 + M)                     modulate
F'         = F + alpha * (F * (1 + M) - F)   alpha = 0 at init  =>  identity
```

**Why `M` is signed.** `M > 0` marks target-like structure (amplify), `M < 0`
marks speckle/clutter-like structure (suppress). A non-negative-only prior can
express the first half of that sentence and not the second.

**Why the modulation is multiplicative.** Speckle is signal-dependent, so the
correction has to scale with local intensity rather than being a constant offset
— the same reasoning behind the `beta*N*F` term in Component 2.

**Where it sits.** Per detection level, immediately *before* attention, so that
Components 3 and 10 operate on target-modulated features. That is what makes
"target-aware" a structural property of the graph here rather than a claim about
intent.

**The ablation ladder, and the confound it removes:**

| Arm | Prior | Learnable params |
| --- | --- | --- |
| `tp_none` | none | gate only |
| `tp_cfar` | CFAR statistic generalised to features, `tanh(g*((F-mu)/sd - t))` | **none** |
| `tp_static` | spatially uniform, per-channel logits | `C` per level |
| `tp_channel` | the proposed evidence network, pooled over space | same as `tp_full` |
| `tp_full` (learned) | the proposed spatially-varying map | same as `tp_channel` |

Comparing `learned` against `static` alone would confound *spatial selectivity*
with *capacity*: the larger arm could win for reasons unrelated to the
hypothesis. `tp_channel` therefore reuses the proposed arm's exact network and
averages its output over space, so the two are identical in size — asserted, not
assumed, in `test_target_prior_arms_are_capacity_matched_where_claimed`.
**This is the falsifying run for the central claim**: if `tp_channel` matches
`v2_full`, the spatial prior is not what is doing the work, and the paper must
report that.

---

## Component 9 — Spatial-Frequency Representation (SFR)

Convolution is a spatial operator with a fixed, local, low-pass-biased basis.
SAR contains two structures it represents poorly: broadband speckle, and target
returns that are localised in space and therefore spread across the spectrum.
This component gives the network an explicit spectral branch so that *training*
decides how much of the spectrum to keep, instead of a classical filter deciding:

```
F_hat  = rfft2(F)                                (B, C, H, W//2+1) complex
r      = |f| / |f|_nyquist                       normalised radial frequency per bin
g_c    = log-gain of channel c in radial band b  learnable, (C, B)
g_c(r) = linear interpolation of g_c over r      smooth and differentiable
F'     = irfft2(F_hat * exp(g(r)))
F'     = F + alpha * (F' - F)                    alpha = 0 at init  =>  identity
```

**Why radial bands and not a per-bin mask.** Three practical reasons. (1) A
per-bin mask costs `C x H x W/2` parameters — millions, which would dominate the
model and overfit long before acting like a filter; bands cost `C x B`. (2) Bands
are defined in *normalised* frequency, so a filter learned at 640 keeps its
meaning at 512 or 1024, which the planned multi-resolution test requires. (3)
`exp(g)` is strictly positive, so the branch is interpretable as a filter
magnitude response and cannot flip the sign of a coefficient.

**Why it sits on P5/32.** FFT cost scales with spatial resolution. On the deepest
default stage the transform operates on the smallest feature map; the same module
at P2/4 would cost roughly 64× more. This is a design decision, and the measured
consequence is visible in the README: Component 9 adds parameters but essentially
no FLOPs.

**Ablation:** `fr_none`, `fr_highpass` (fixed classical response, zero learnable
parameters — it is a buffer), `fr_static` (learnable bands, input-independent),
and the proposed `fr_sff` (bands modulated per sample by the feature's own global
descriptor). `sff` starts numerically equal to `static`, so `sff - static`
measures the value of input adaptivity alone.

---

## Component 10 — Context Aggregation (CAG)

A SAR target is often not identifiable from its own neighbourhood. A bright point
return on open sea, the same return in a harbour, and a speckle spike on grass can
look nearly identical locally; what separates them is the *surrounding scene*.
Standard convolution grows its receptive field by going deeper, spending
parameters and resolution to approximate context it could gather directly:

```
c_local    = psi([ DWConv_d(F) for d in (1, 2, 4) ])   multi-extent spatial context
c_regional = phi(global_avg_pool(F))                   regional channel context
F'         = F + alpha * (c_local + c_regional)        alpha = 0 at init => identity
```

Dilated depth-wise convolutions widen the receptive field *without downsampling*,
so context is acquired at the detection resolution — the same reason the P2 head
exists. A downsampled context path cannot help a target a few pixels wide.

**Relation to attention, stated plainly.** CAG overlaps in *purpose* with the
local-contrast branch of Component 3 but not in mechanism: attention produces a
multiplicative **gate** on the existing feature, while CAG adds a **context field**
derived from a wider neighbourhood. Whether that distinction earns its parameters
is empirical, which is why the removal ablation exists and is treated as the
stronger evidence than the cumulative row.

**Ablation:** `cx_none`, `cx_local`, `cx_regional`, and the proposed `cx_multi`.
The mode is named `regional` rather than `global` for a concrete reason: mode
strings pass through `parse_model`'s `ast.literal_eval`, which suppresses
`ValueError` but **not** `SyntaxError`, and `global` is a Python keyword. Naming it
`global` aborted model construction with a bare `SyntaxError` from inside
ultralytics; `test_module_mode_names_are_safe_for_parse_model` now guards the whole
mode vocabulary against that and against silent collision with a `parse_model`
local.

---

## Component 11 — Target-Aware Deformable Refinement (TADR)

Components 1-10 change *what a feature contains*. None of them changes *where it
is sampled*. A detector that only reweights its features is still bound to the
alignment the backbone's fixed grid produced, and in SAR that alignment is
frequently poor at the target: a ship's wake, a harbour wall or a speckle spike
pulls the local statistics around, so the strongest response at a cell often
belongs to a neighbouring clutter pixel rather than to the target itself.

TADR gives every cell a bounded, learned opportunity to look somewhere else:

```
d      = Delta(F)                            offsets, (B, 2, H, W)
d      = s * tanh(d)                         bounded to |d| <= s
F~     = grid_sample(F, G_base + d)          differentiable bilinear resample
F'     = mix(F~)                             depth-wise 3x3 + BN + SiLU, 1x1 mix
out    = F + alpha * (F' - F)                alpha = 0 at init => identity
```

with `G_base` the identity grid (`linspace(-1, 1)` per axis, `align_corners=True`),
so `d = 0` is exactly the original sampling position. `s` is `max_offset`, the
search radius in normalised grid units (1.0 = half the feature map).

**Why this is a distinct mechanism, not more attention.** The three candidates are
separable on paper and in the ablation:

| Stage | Acts on the feature by | Isolated by |
| --- | --- | --- |
| Attention (3) | multiplicative gate at each cell | `att_saa_static` |
| Context (10) | additive field aggregated *around* each cell | `cx_local` / `cx_regional` |
| Refinement (11) | changing *where* the value is read from | `rf_local` |

The distinction matters because a deformable arm is usually credited for two
effects at once: the new operation *and* the extra network that predicts it. The
`rf_local` control therefore keeps the identical mixing sub-network and removes
only the offsets, so `ours − local` measures deformation alone -- the same
discipline applied to the prior's capacity-matched control.

**Wiring.** TADR is the last module before each level's head, and it is placed
*after* Component 8 by construction, so its offsets are predicted from an
already prior-modulated feature. The "target-aware" conditioning is thus
structural (the feature it deforms already carries the prior) rather than a second
evidence network, which would duplicate Component 8's computation and confound the
two ablations.

**Ablation:** `rf_none` (no refinement, and *no allocated network*), `rf_local`
(capacity control), `rf_static` (learned but input-independent offsets, which
isolates content-adaptive sampling from learned sampling), `deform` (proposed), and
the removal row `v2_norefine`.

**Open item that the smoke run surfaced.** After two epochs of smoke training the
learned offsets reached `|d| ~ 0.47` against a bound of `s = 0.5`, i.e. the tanh is
saturated where its derivative is smallest (`1 - tanh^2 ~ 0.11`). Either the model
wants a larger search radius or the bound was set too tight, and the two have
opposite fixes; `max_offset` is therefore a hyperparameter to sweep rather than a
constant, and no claim in the paper should rest on the value currently in the YAML.

**Not implemented: cross-level scale routing.** The plan's "target-aware dynamic
scale routing" would reweight P2-P5 jointly against one shared prior. That needs a
module consuming several feature maps, which this repository's model YAMLs cannot
express -- verified, not assumed. `parse_model` resolves an unknown module's output
channels with `c2 = ch[f]`, so a list `from` raises `TypeError: list indices must be
integers or slices, not list`; and naming a `Detect` subclass as the head fails
because the head branch is a `frozenset` *identity* test, so the subclass never
receives `[reg_max, end2end, ch_list]` and dies with `IndexError: tuple index out of
range`. Supporting it would mean patching `parse_model`, which this project avoids
so that the stock summary, FLOPs counter, validator and checkpointing keep working.
Component 11 is therefore *per-level* refinement, and the routing idea is reported
as unimplemented.

---

## Component 12 — SAR Input Adapter (SIA)

Every other component operates on *backbone features*. This one operates on the **image**,
before the first convolution sees it, because that first convolution is where SAR statistics
are least represented: a stock detector's stem was trained on natural images and consumes raw
intensity as if it were an RGB photograph.

```text
stats   = psi([x ; mu ; sd ; hp])      mu, sd, hp = local mean / std / high-pass on a k x k window
learned = phi(DWConv(x))
z       = psi2([stats ; learned])      hybrid: both streams, learned fusion
x'      = x + alpha * (z - x)          alpha = 0 at init  =>  exact identity
```

### Why the projected form, and when it can help

`parse_model` resolves an unknown module's output width with `c2 = ch[f]`, so the adapter must
return *the same* number of channels it received. A genuine widening of the input tensor is
impossible without patching the parser or the stem, which is why the extra channels are a
hidden representation projected back down — an *adapter*, not a new input layer.

One honesty note belongs with the module. Ultralytics loads a single-channel SAR image by
replicating it into three channels, so on single-polarisation data `x` and `mu`/`sd`/`hp` are
internally redundant: the local statistics carry no information the raw channels do not
already order differently. On such data the honest expectation is that `local` adds little
over `identity`, and **the experiment should say so**. The arm becomes genuinely richer only
for multi-channel input (dual-polarisation VV/VH, or a pre-computed feature stack such as
intensity + local statistics + gradient).

### Measured cost, and the control

| arm | parameters added (c1 = 3) |
| --- | --- |
| `identity` | +0 |
| `local` | +43 |
| `learned` | +98 |
| `hybrid` | +164 |

`identity` is the control, and its exactness is measured rather than asserted. Dropping the
adapter row from `in_identity` leaves a graph layer-for-layer identical to `v2_full` (same
types, same per-layer parameter counts, 55 layers matching), and after synchronising the
shared weights the two produce **bit-identical** outputs (max absolute difference `0.0`). So
`in_identity` measures the rest of the model and nothing else.

This is the arm that also carries an important negative result in advance: because the FFT/DCT
slot already sits on the deepest backbone stage, and because a single-channel image is
replicated into three, an input adapter has less to work with than it appears. If `in_hybrid`
fails to beat `in_identity`, the correct conclusion is that the stem is not where SAR's missing
prior lives — not that the adapter was implemented badly.

## Training strategies that are not modules

Two parts of the brief change *what the model sees* rather than *what the model is*. Both live
outside the architecture deliberately: a sampling strategy that needed its own layer would no
longer be attributable, because a gain could come from the extra capacity rather than from the
extra examples.

### SAR-specific augmentation (SEC. 5)

Reuses the **same** corruption model the robustness benchmark evaluates under
(`saryolo/evaluation/robustness.py`). This is the point, not a convenience: a second
implementation would let the degradation a model trains on and the degradation it is tested
under drift apart, and a robustness result would then measure an inconsistency between two
pieces of code rather than a property of the model. Because they share one implementation, the
severity grid is literally the same tuple.

| property | how it is guaranteed |
| --- | --- |
| labels stay valid | every corruption is appearance-only, verified by comparing output and input shapes per image; a mismatch raises |
| the clean view is the original | `identity` round-trips byte-exactly (measured: max abs diff `0`) |
| reproducibility | one seed → byte-identical images, and the draw is recorded per file in a manifest |
| comparability with the robustness figure | a severity outside the published grid is rejected |
| the headline claim is not gambled | `clutter` is opt-in, not default (unlabelled bright blobs look like the small targets being improved) |

It is **offline**, with the trade-off stated: the augmented split is a committed, inspectable
artifact, at the cost of being fixed rather than resampled per epoch, and of `views`× the disk.

### Hard-example mining (SEC. 6)

An offline sampling change: score every image by how badly the model failed on it, then emit a
training list containing all training images *plus* the hardest ones repeated. Nothing is
approximated and no bookkeeping lives in the trainer, so the two runs differ only in the list
of images they saw.

```text
score(image) = sum_k  w_k * count_k(image)  /  max(n_gt, 1)
```

The count comes from the failure taxonomy in `saryolo/visualization/error_analysis.py`, not
from a matcher of its own. That matters twice over: an earlier version carried its own IoU
matcher — making a *third* one — which meant the difficulty ranking and the paper's
failure-analysis table could contradict each other about the same model.

Weights, and why: `small_object_miss` 3.0 (the headline claim), `clutter_false_positive` 1.5
and `classification_error` 1.5, `false_negative` 1.0, `false_positive` 1.0,
`localization_error` 1.0, `clutter_confusion` 0.5, `true_positive` 0.0. An unmapped taxonomy
outcome **raises** rather than scoring zero, because a failure mode silently worth nothing is
how a miner stops mining the thing it was added for.

Normalisation is by ground-truth count, so the score is a per-image failure *rate*: one miss in
a five-target scene ranks above one miss in a one-target scene. A total miss is a rate of `1.0`
at any density — which is the correct behaviour, and is why the normalisation is stated as a
rate rather than as "denser scenes score lower".

Two rules keep it honest:

- **Mine the split you will train on.** `--split` defaults to `train`; a non-`train` value
  prints what it contaminates. Mining `val` and oversampling those images trains on the
  evaluation data.
- **A hard image outside the training list is an error, not a filter.** The first version
  filtered them out silently, so mining `val` wrote a list containing none of the mined images
  while still reporting a repeat count and exiting 0 — a run indistinguishable from the
  baseline. `write_oversampled_list` now raises.

## Evaluation protocol

* **mAP50 / mAP50:95** via a self-contained COCO-protocol implementation
  (`saryolo.evaluation.metrics`), with deviations from pycocotools documented in
  the module docstring.
* **Scale-wise AP** (`AP_small` / `AP_medium` / `AP_large`) reported separately,
  because a single mAP can hide exactly the effect being claimed. An area range
  with no ground truth returns `None`, never `0.0` — otherwise a dataset with no
  large objects would appear to show the model failing on large objects.
* **Robustness** under controlled, deterministic degradations (multiplicative
  Gamma speckle, contrast compression, blur, resolution loss, injected clutter),
  with the baseline and proposed model facing byte-identical inputs. Only images
  are degraded; ground truth is untouched, because a physical degradation changes
  the sensor signal, not where the targets are.
* **Efficiency** with a stated protocol: latency measured after warm-up with CUDA
  synchronisation, FLOPs at a recorded input size, peak memory around a
  forward+backward step.
* **Cross-dataset** evaluation that *refuses to run* on incompatible label
  spaces instead of reporting a meaningless low mAP.
* **Multi-seed** runs (EXP-012) reported as mean ± std, because a single-seed
  difference of a few tenths of a point is not evidence.
