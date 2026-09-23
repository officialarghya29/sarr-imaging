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

### SEC. 4 — representation consistency (opt-in, off by default)

The fourth term compares the representation of the same batch under two draws
of the imaging process:

```
F_a = backbone(batch)                  clean view
F_b = backbone(perturb(batch, k, s))   degraded view, same corruption physics as SEC. 15
L_consistency = (1/L) * sum_l  (1 - cos(F_a^l, F_b^l))     per detection level l
L_total      += w_consistency * L_consistency
```

The hypothesis it encodes is that a SAR detector should not change its mind
about *what is where* because the speckle draw changed: the drift that matters
is directional, so the term is cosine (scale drift is a calibration difference,
not a structure change) rather than MSE.

**Dead positions are excluded, not penalised.** `cosine_similarity` returns 0
when a vector has zero norm, so a feature position whose channels are all zero
would contribute `1 - 0 = 1` — the largest possible value — and the term would
spend its gradient reviving dead positions instead of measuring drift.
Positions where either view is (near-)silent (norm <= 1e-6) are masked out of
the average. This was a real bug in an earlier version: with BatchNorm biases
and post-activation sparsity, silent positions are routine, and the term would
have been dominated by them.

**The perturbed pass runs BatchNorm in eval mode.** A second forward in train
mode would move every BN running buffer twice per step, so enabling the term
would silently change the normalisation of the whole network — and the ablation
would measure that, not the loss. The model switches BN to eval for the second
pass and restores it afterwards; the guard asserts each BN's
`num_batches_tracked` advances by exactly 1 per step with the weight on, not 2.

**Cost, stated rather than hidden:** one extra forward pass per step. The
weight defaults to 0, so stock behaviour is bit-identical when off. The severity
is rejected unless it comes from the published grid, for the same reason as the
augmentation: training and evaluation must refer to the same degradation.

#### Where the arms live, and why not in the removal table

A loss term is not a module, and that decides which control is valid for it.

The removal slot verifies an arm by a **strict parameter drop** against a named
reference: `v2_nofreq` is a removal because it is smaller than `v2_full`. No
loss arm can ever satisfy that — switching `w_consistency` off does not delete a
single weight — so a loss arm placed there would either fail its own guard or
have to be excluded from it, and the second is how a no-op gets reported as a
clean removal. The consistency arms therefore live in their own slot
(`EXP-321…326`), where the control is `v2_full` itself at `w_consistency: 0`:
same graph, same size to the byte, different objective.

That equality is not an accident of implementation, it is the experiment. With
the graph fixed, any accuracy difference is attributable to the objective alone,
which is exactly what a loss ablation is supposed to isolate. The arms are:

```
v2_full          w = 0.0                          control
cons_sev1        speckle,       1 look             mild speckle
cons_sev16       speckle,      16 looks            heavy speckle
cons_lowcontrast low_contrast,  gamma 2.2          a different family
cons_lowsnr      low_snr,       0.20               additive, not multiplicative
v2_cons          speckle,       4 looks            the ladder arm (EXP-019)
```

The sweep exists because the claim is a *principle* — the representation should
not depend on the noise realisation — and a principle must not be conflated with
a constant tuned to the test corruption. If only the arm matching the robustness
benchmark's own setting improves, then the honest reading is that the term is a
tuned denoiser, and the paper must say so.

Every enabled arm states `consistency_kind` and `consistency_severity`
**explicitly** in its YAML rather than inheriting the defaults, so the
decomposition a run trained against can be read off its own config. A test walks
the whole zoo and fails if any variant enables the term while naming a
decomposition that is not on the published grid.

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

### Module G — prior-conditioned frequency selection (the `spectral` arms)

The brief's Module G asks for the target prior to drive *frequency-band
selection*, not merely feature modulation. The wiring one would first reach for
— prior feeds the backbone spectral slot — is not expressible in an Ultralytics
graph, for two verified reasons: a custom module cannot take two inputs
(`parse_model` resolves `c2 = ch[f]`; only hardcoded names such as `CBFuse`
receive a channel *list*), and the backbone spectral slot runs at P5/32 *before*
any prior exists, so conditioning it would need a backward edge that breaks the
stock summary, FLOPs counter and validator. Rather than fake the wiring, the
conditioning lives where the prior actually is — this module produces the prior
**and** lets it choose the spectrum:

```
e      = psi([F ; F - mu_n ; sd_n ; sd_w])       prior evidence (as above)
d      = [mean(e), std(e), max(e)]               3-vector descriptor, pooled
g      = band_head(d)                            (B, C, B) per-channel log-gain per band
F_filt = radial_band_filter(F, g)                shared interpolation with Component 9
F'     = F + alpha * (F_filt * (1 + M) - F)      prior-filtered AND prior-modulated
```

The band head reuses the same radial-band interpolation as Component 9, so the
two spectral mechanisms share one implementation.

**The matched-capacity control (`spectral_feat`).** Same head, same band count,
same descriptor width, same parameters — only the head's *input* differs: pooled
raw-feature statistics instead of pooled prior evidence. Without it, a reviewer
could correctly object that any input-adaptive filter would do just as well;
with it, `spectral - spectral_feat` isolates *prior* conditioning from input
adaptivity, at byte-identical size (measured: both 16.586 M at scale s).

**Ablation rows:** `tp_spectral_feat` (control), `tp_spectral` (proposed),
`v2_prior_spectral` (the same mechanism in the ladder context — `EXP-018`).

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

**Scope note on "target-conditioned" frequency selection.** This slot's `sff`
arm adapts to the *feature* — it cannot be conditioned on the target prior,
because it runs in the backbone before any prior exists. The prior-conditioned
mechanism (Module G) therefore lives in the prior slot instead; see the
`tp_spectral` arms under Component 8 and their `spectral_feat` control.

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

## Component 33 — Acquisition-Conditioned Adapter (CND)

The cross-sensor claim. One detector is trained across several SAR sources and then asked to work
on a source it has never seen; the failure is not a lack of capacity but the reuse of features
learned under one acquisition distribution as though acquisitions did not differ. The hypothesis
is narrow and falsifiable: *conditioning the feature modulation on the acquisition parameters
recovers part of that loss, on sources that were never trained on.*

```text
SAR image ──▶ backbone ──▶ P3 / P4 / P5
                                │
                                ├── CND(metadata) ── first in the per-level chain
                                │        ├── prior
                                │        ├── attention
                                │        ├── context
                                │        └── refinement
                                ▼
                              Detect
x' = x * (1 + alpha * gamma(m)) + alpha * beta(m)      alpha = 0 at init  =>  exact identity
```

### The held-out sensor has no embedding row

That one fact decides the design and rules out the obvious shortcut. The fastest way to condition
on a sensor is a learned embedding table indexed by sensor id — and it is *unavailable exactly
where the claim is tested*, because the held-out sensor was never in the table. Conditioning
therefore runs on two channels:

| channel | fields | can reach an unseen source? |
| --- | --- | --- |
| categorical | learned embeddings for sensors already seen | no — by construction |
| continuous | resolution, nominal band wavelength, incidence angle | yes — these exist for any sensor |

Both feed one encoder, and the arms exist to find which of them carries the generalisation. **If
only the categorical channel matters, the claim fails on its own terms**, and the paper must say
so. `cond_continuous` is therefore the arm the headline result has to come from; `cond_sensor` is
the arm that tests the shortcut and is expected to be the one that cannot transfer.

### Two properties that are pinned by test

**Identity at initialisation survives being handed metadata.** A fresh conditioned model is
numerically the unconditioned one, and the modulation path starts *non-zero* behind a zero gate.
Both halves matter: with a zero-initialised branch on top of a zero gate, `dL/dalpha = <dL/dout,
gamma*x + beta>` is identically zero and the adapter can never leave identity while looking
perfectly healthy — the same frozen-module failure that Component 1 originally had. Identity is
bought by the gate and by nothing else. Measured: `max |v2_full − cond_*| = 0.0` for all seven
arms, after matching the *stock* layers in order (the inserted adapters shift every downstream
index, so a positional state-dict load is not possible). Supplying a *known* acquisition to an
untrained adapter still changes nothing, which is what makes "SAR-YOLO matches YOLO at step 0"
true for datasets that carry metadata as well as for those that do not.

**An unused field is genuinely information-free.** The field ablation is only meaningful if the
`sensor`-only and `resolution`-only arms cannot see each other's field. For a field an arm does
not consume, both the value *and* the availability flag are masked — masking the value alone
would still let the encoder learn from *whether* a field was present. The test varies each
omitted field's value and its availability flag separately and requires the descriptor to be
bit-identical, and then checks the converse (a consumed field must move it) so the equality is
not vacuous.

### Refusal rather than guessing, and per-sample by necessity

An index the embedding table cannot represent **raises**, naming the field and the offending
index. Clamping is the tempting fix and the wrong one: it maps an unseen sensor onto a sensor
that *was* trained on — precisely the confusion the design exists to prevent — and it does so
silently while the run still reports a result. The failure mode this creates is real and was hit
while writing these tests: a model built with one vocabulary and fed batches encoded with
another. The error message says which two vocabularies disagree rather than leaving a bare
"index out of range".

Conditioning is **per sample, not per batch**, and that is forced by the protocol rather than
chosen: the LOSO recipe trains on three sources at once, so a single per-batch descriptor would
average the acquisitions together and destroy the signal being tested. The encoder is applied per
sample and the modulation is broadcast over space, which is what keeps the cost near zero. The
row count must match the batch — a mismatch is a wiring bug, and broadcasting it away would
attach one image's acquisition to another image's features and still produce a plausible run.

An *absent* context is encoded as an explicitly unknown acquisition (all-zero values, every
availability flag off) rather than bypassing the module, so "this image's parameters were never
recorded" is a state the model is trained on rather than a hole in the graph.

### Measured cost

| arm | mode | fields | parameters | Δ vs `v2_full` |
| --- | --- | --- | ---: | ---: |
| `v2_full` | — | — | 16,229,583 | control |
| `cond_gain` | gain | all | 16,278,099 | +0.049M |
| `cond_shift` | shift | all | 16,278,099 | +0.049M |
| `cond_resolution` | film | resolution | 16,306,003 | +0.076M |
| `cond_continuous` | film | resolution, band, incidence | 16,306,515 | +0.077M |
| `cond_sensor` | film | sensor | 16,306,963 | +0.077M |
| `cond_film` | film | sensor + resolution | 16,307,091 | +0.078M |
| `cond_spatial` | spatial | all | 16,311,331 | +0.082M |

Every arm is under 0.5% of the model. One honest caveat: the four `film` field-set arms differ by
at most ~1.1k parameters because each consumes a different number of continuous descriptors.
That is tight enough to attribute a difference to *which fields* are read, but it is **not** the
byte-for-byte equality the target-prior slot achieves, so the parameter count should be reported
per arm rather than claimed as exactly capacity-matched.

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

### The init stage (Phase 3: RGB-pretraining vs random-init vs SAR-pretrained)

The RGB-pretraining → SAR-fine-tuning gap that SARDet-100K diagnosed is one of the
motivating observations for the whole project, so it gets a named experiment rather
than a checkbox: EXP-401/402/403 share one model YAML, one seed and every training
argument, and differ *only* in the stated `init:` stage — `none` (fresh build, the
control), `coco11` (COCO-pretrained `yolo11s.pt`), or an explicit checkpoint path
(MSFA-style SAR weights when available).

The mechanism matters because of how Ultralytics builds models: `train()` calls
`get_model(weights=self.model if self.ckpt else None, ...)`, so a YAML-built facade's
inner model is silently rebuilt — any transfer done by mutating the facade would
vanish while the ledger still claimed a pretrained start. `apply_init` therefore uses
the trainer's own route: a fresh build of the same graph, the shape-compatible
intersection of the source weights loaded in, serialised to a checkpoint, and the
facade reloaded from it. The run record carries the init stage, the source, and the
count of tensors whose **values actually changed** — the raw intersection over-counts
~5× because zero-initialised BN buffers are shape-compatible and equal by
construction. Two refusals keep the comparison honest: a checkpoint sharing no
shape-compatible tensor raises (that run would be random-init wearing a pretrained
label), and so does a transfer that would change nothing.

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
* **Cross-source** evaluation as leave-one-source-out folds
  (`saryolo.data.groups`), which is the headline protocol: see below.
* **Multi-seed** runs (EXP-012) reported as mean ± std, because a single-seed
  difference of a few tenths of a point is not evidence.

### Representation diagnosis (master-plan §14), before any accuracy number

The central hypothesis — that a SAR detector learns *object + acquisition
appearance* rather than object semantics alone — is a claim about representations,
and it can be tested on a frozen detector before any new training
(`saryolo.evaluation.probes`):

1. **Linear probe accuracy.** A logistic probe over mean-pooled per-level head
   features, five-fold. `sensor` accuracy far above the chance rate implied by the
   source count says sensor identity is linearly present; `sensor` high while
   `class` is low is the appearance-dominated failure the conditioning adapter
   addresses.
2. **Within-class centroid drift.** For each pair of acquisition groups and each
   class present in both, the cosine distance between group centroids of pooled
   features — "same object, different sensor" drift, the quantity the adapter is
   hypothesised to reduce.
3. **Linear CKA** (Kornblith et al., 2019) between the pooled feature matrices of
   two acquisition groups over the same images; low CKA under a fixed class
   distribution is drift the head has to pay for.

The probe is deliberately linear on a frozen model: a probe that could train
features would measure the probe. Extraction is side-effect-free by test — BN
buffers untouched, batch rows independent, the hook removed in a `finally` block —
and every reported number is either measured or absent with a stated reason (a
class probe over one class is `None` plus a reason, never `1.0`).

The command-line route is `python -m saryolo.cli probe --weights <ckpt> --data
<data.yaml> --field sensor`, with the acquisition labels coming either from a
metadata table (`--metadata`) or, for archives that encode the source in the
filename, `--stem-pattern '^(sentinel1|gaofen3)_'` — stated, never inferred. The
report JSON records the field vocabulary, the chance rate, and a caveat that on
an untrained checkpoint these numbers are placeholders; the printed summary puts
every accuracy next to that chance rate rather than next to 1.0.

The planned reading: if the baseline's sensor probe is high and the conditioned
model's drops while class information is retained, the mechanism does what the
paper claims at the representation level, which is the §14 evidence the failure
analysis alone cannot provide. Until a trained checkpoint exists, these are
placeholders and no claim is made.

### Leave-one-source-out: why the unit is the source

The strongest claim this project makes is generalisation to a source it has never
trained on. "Source" is not the dataset, because the two do not coincide: one
dataset can mix several sensors (SAR-Ship-Dataset is Sentinel-1 **and** Gaofen-3)
and several datasets can share one. Splitting by dataset name therefore leaves the
question open, while splitting by source answers it directly.

The mechanism already existed for scenes: `split_files(..., scene_key=...)` never
places two chips of one acquisition on both sides of a split, and a source key is
the same mechanism at a coarser unit. What did not exist was a way to *derive* the
key — and that is where the protocol quietly fails, because a key function matching
nothing does not error out; it yields **one group**. The folds then build, the files
are written, and the held-out "source" is a random chip split reported as
cross-source generalisation.

So the derivation is explicit:

| `--rule` | Key taken from |
| --- | --- |
| `parent` | the first `--depth` path components below `--root` (one directory per source) |
| `regex` | a single capture group in the filename |
| `sidecar` | an explicit image → source mapping |
| `resolution` | a *binned continuous field* from a metadata table — the cross-resolution protocol |

Four strategies rather than one because no universal rule is safe — but the
fallback is a stated mapping, never a heuristic silently guessing a sensor.

The fourth rule is Experiment D of the master plan. The unit of the LOSO fold is a
*group*, and nothing in the machinery cares whether a group is a sensor or a range
of a physical quantity: binning each image's `resolution_m` at stated edges
(`--edges 5,10,20`; no default, because the bin width decides what "cross-resolution"
even means) yields groups labelled `resolution_m<=10`, `resolution_m>10`, and so on —
half-open, unbounded at both ends, with a stable label format stripped of float
noise. Holding out one bin is then exactly the cross-resolution experiment: a model
trained on `resolution_m<=10` and evaluated on `resolution_m>10` is tested on a
resolution regime it never saw. The bins are physical quantities rather than
cluster IDs, which matters for the claim: a cluster found in feature space could be
an artefact of the embedding, whereas a bin edge is a modelling choice that is
stated in the fold manifest and can be defended or attacked directly.

The table itself comes from `python -m saryolo.cli metadata --images <dir>
--sidecar <csv>`, for archives that state acquisition in a table rather than in the
file layout. Both ends refuse the failure mode that matters: the *builder* rejects a
sidecar matching no stems (an all-unknown table would make every conditioning arm
identical while the ablation still reported a result), and the *rule* rejects an
image with no metadata row, which would otherwise vanish from every fold and shrink
the test set without a word.

**Every guard exists because its failure is silent.** A rule that matches nothing,
fewer than two sources, unkeyed images, groups whose last component is
`train`/`val`/`test`, a source below `--min-test-images`, and overlapping splits are
all refusals rather than warnings: each produces a run that *completes* and reports
a number that means something other than what it says. The split-name check earned
its place immediately — pointing `parent` at a processed `images/` directory, or
raising `--depth` above it, both key on the train/val/test layout, and the second
passes every other check while looking entirely healthy.

Two configs are written per fold, and the difference is the protocol:

```
data.yaml          train: train.txt   val: val.txt    -> training (early stopping)
eval_holdout.yaml                     val: test.txt   -> the reported number
```

`evaluate_detections` reads a config's `val` entry, so a single config validating on
`val.txt` would report a score from sources the model trained on. Validation is
never source-held-out here, and each fold records that fact rather than leaving it to
be inferred.

The list form also exposed a real evaluation bug, which the zero-ground-truth guard
now catches: labels for a `.txt` split were derived by string-replacing
`images` → `labels` on the list path, so no ground truth was read, every metric came
back `None`, and the command still exited successfully. Ground truth is read before
inference, so this fails immediately rather than after a prediction pass.

**Both halves are now built; neither is measured.** The protocol above is one half of the
claim and the model side is the other (Component 33). What does not exist is a cross-source
*number*: the folds and the adapter are wired and tested, but no GPU run against real data has
produced a result, so every cross-source cell stays `TBD`. Two things have to be established
before the claim can be made, and only the first is currently verified:

1. that the baseline degrades measurably across sources at all — if it does not, there is no
   problem to solve and the adapter is unmotivated;
2. that `cond_continuous` (the arm that can reach an unseen source) recovers part of that
   degradation against the `v2_full` control of identical size.

If (2) fails while `cond_sensor` succeeds, the honest conclusion is that the method specialises
to known sensors rather than generalising to new ones — a negative result the slot is
instrumented to detect, not to hide.
