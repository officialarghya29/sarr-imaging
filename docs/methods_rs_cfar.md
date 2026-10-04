# Methods draft — the ratio-space CFAR front end (RS-CFAR)

**Status:** draft of a paper's *Method* subsection, written from the frozen
interface in `docs/architecture_proposals.md` §9.1. It describes what the model
computes; it deliberately does **not** state a result. Every number that appears is
a *measured* value with the run that produced it, and the claims are labelled by the
evidence behind them (see §8). This file exists so the method can be reviewed and
reused independently of the pilot's outcome.

---

## 1. Motivation

A single-look or multi-look SAR magnitude image is a coherent measurement: its
dominant error term is multiplicative speckle, and its dominant failure mode for a
small target is *clutter*, not noise. A weak ship return can be exceeded by the
local mean of the sea that surrounds it, and that local mean is itself a slowly
varying quantity that differs between acquisitions. The classical detector does not
denoise the image first; it computes a statistic of the *ratio* between a cell and a
ring of its neighbours, and thresholds it — the constant-false-alarm-rate (CFAR)
decision rule. The ratio form is the point: because the decision variable is
dimensionless, a threshold calibrated on one scene transfers to a scene of a
different absolute brightness.

A generic convolutional stem over the raw intensity spends its first layers
re-deriving a local-contrast statistic that radar theory already knows how to
compute in closed form, and it does so from a representation (raw intensity) that
changes with sensor gain. The method below makes the detector's first representation
be the radar statistic itself.

## 2. The statistic

Let ``x ∈ R^{B×C×H×W}`` be the input batch. Because a SAR magnitude product is one
measured channel replicated to ``C``, the statistic is computed on the channel mean

```
m = mean_c(x) ,            L = log(m + eps).
```

For each scale ``k`` in a window set ``K = {3, 7, 15}`` (deliberately spanning more
than an octave, so the ratio is measured against backgrounds that differ in size),
let ``box_k`` be a centred average pool of size ``k``:

```
mu_k  = box_k(L)
var_k = box_k(L ⊙ L) − mu_k ⊙ mu_k            (variance of the log intensity)
r_k   = L − mu_k                               (log-ratio — the CFAR decision variable)
c_k   = sqrt(var_k) / (|mu_k| + eps)           (local coefficient of variation)
s     = [ m, r_1, c_1, …, r_K, c_K ]           ∈ R^{B×(1+2|K|)×H×W}
```

Two properties matter and are pinned by test:

* ``r_k`` is **exactly invariant to a global radiometric gain**. Replacing ``x`` by
  ``αx`` adds the constant ``log α`` to ``L``, which cancels identically in
  ``L − box_k(L)``. This is why a learned threshold over ``r_k`` can transfer across
  acquisitions (`tests/test_cfar_scope.py::test_the_log_ratio_channel_is_invariant_to_a_global_radiometric_gain`).
* For fully developed speckle, ``c_k²`` is the reciprocal of the equivalent number of
  looks, so ``c_k`` separates a textured region from a point return at the same
  brightness.

The stack is finite on a constant image (``var_k → 0`` and the ratio is floored), so
a locally flat sea or desert region cannot inject a `NaN` into a training batch.

## 3. The learned gain, and why the model starts as the baseline

A small convolutional MLP over the ``1+2|K|`` channels emits a per-pixel gain

```
g = tanh( MLP(s) ) ∈ (−1, 1)^{B×1×H×W},     x' = x ⊙ (1 + g).
```

The last MLP layer is **zero-initialised with zero bias**, so ``g ≡ 0`` and
``x' ≡ x`` *bit-for-bit* at initialisation. The front end is therefore an exact
identity until it has learned something, which preserves the repository-wide
attributability contract (`docs/METHOD.md` §0): a measured difference between an arm
and its baseline cannot be extra capacity that was active at step 0. Unlike the
residual-gate pattern of the other components, the identity here comes from the last
layer alone, and the *first* layer legitimately starts with zero gradient until the
last layer moves — the test asserts both the initial zero and that the first layer
recovers a non-zero gradient once the last layer has moved
(`test_the_gain_parameters_receive_gradient_at_initialisation`).

The modulation is applied to the raw intensity ``x`` rather than to ``L``: inverting
the log would make the identity approximate rather than exact, and the backbone was
designed to read an intensity image.

## 4. The two controls (this is the part that makes the claim testable)

A front end that adds parameters can improve a model for reasons unrelated to the
statistic. Two ablations isolate that:

| Mode | Reads | Learnable | Answers | Identity at init |
| --- | --- | --- | --- | --- |
| `cfar` | the statistic stack ``s`` | yes (zero-init gain) | the proposal | exact |
| `conv` | the raw intensity, replicated to the stack's width | **exactly the same layers** | "is it just 217 parameters and a learned transform?" | exact |
| `fixed` | the statistic stack ``s`` | **none** — an analytic threshold | "do the statistics help without learning?" | no (by design) |

The `conv` arm has *byte-for-byte the same layer shapes* as `cfar`, so its parameter
count is identical and a difference between the two is attributable to the
representation rather than to capacity. The `fixed` arm adds **exactly zero**
parameters and is deliberately **not** an identity at init: a control that equals
the baseline measures nothing. Neither control substitutes for the other; together
they turn "the statistic helps" into two separately falsifiable statements.

## 5. Integration into the detector

The front end is channel-preserving (``C → C``) and single-input, so it is emitted
as **row 0** of the backbone and the stock parser, model summary, FLOPs counter,
validator and checkpointing continue to work unchanged. It is the *only* difference
between an arm and the stock detector: the arm's module list is `["cfar"]` and
nothing else moves (`test_every_front_end_arm_diffs_from_the_stock_detector_in_one_place`).
Six arms are generated — ``{cfar, cfar_conv, cfar_fixed} × {n, s}`` — so the same
insertion point is exercised at two backbone scales.

## 6. Cost (measured, not estimated)

Profile of the generated YAML at 320 px, one class (`saryolo/evaluation/efficiency.py`):

| Arm | Params | Δ params | GFLOPs@320 | Δ GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| `baseline_n` | 2,590,035 | — | 1.613 | — |
| `cfar_n` | 2,590,252 | **+217 (+0.008 %)** | 1.764 | +0.151 |
| `cfar_conv_n` | 2,590,252 | **+217 (identical)** | 1.649 | +0.036 |
| `cfar_fixed_n` | 2,590,035 | **+0** | 1.728 | +0.115 |
| `baseline_s` | 9,428,179 | — | 5.394 | — |
| `cfar_s` | 9,428,396 | **+217 (+0.002 %)** | 5.540 | +0.146 |

The overhead is a property of the front end alone, so the absolute +217 is identical
at `n` and `s` and stays under the 0.5 % budget at both. The compute is **not** free:
the statistic bank and the gain MLP add ~2–9 % of a small model's FLOPs at this
resolution, and that price is stated rather than buried. (Numbers read from the
profiler at build time; see `saryolo/evaluation/efficiency.py`.)

## 7. Experimental design

The direction is evaluated so that each of its claims has a named falsifier:

1. **Attribution** — arm vs stock baseline, identical schedule, one insertion point.
   Falsifier: the `conv` matched-cost control matches the arm.
2. **The statistic alone** — the `fixed` no-parameter control. Falsifier: it matches
   the arm, so the learning is decoration.
3. **Noise** — the primary comparison repeated at ≥ 3 training seeds. Falsifier: the
   paired improvement is inside the seed spread.
4. **Mechanism** — the gain-collapse diagnostic (`saryolo gain`), which reads the
   trained per-pixel gain directly. Falsifier: the gain is constant (collapsed) or
   varies only between scenes (a per-scene scalar), either of which makes the
   "per-pixel decision" claim empty.
5. **Robustness** — corruption sweeps that include both SAR-relevant degradations
   (speckle, clutter, contrast) and pure acquisition shifts (radiometric gain, and
   along-track-only resolution loss, i.e. anisotropy). Falsifier: the arm degrades
   no more slowly than the baseline.

## 8. Claim status (kept with the draft, not moved to an appendix)

| Claim | Status | Evidence |
| --- | --- | --- |
| Exact identity at init; the gain trains | **final** | test-pinned on the real graph |
| +217 params, < 0.5 %, identical at `n` and `s` | **final** | profiler on the generated YAMLs |
| Beats the matched-cost and fixed-threshold controls | **preliminary** | 200/60/60 HRSID subset, one machine |
| Leads the baseline on mAP50:95 at three seeds | **preliminary** | mean +0.0179, small subset |
| Gain is a per-pixel decision, not collapsed | **final, but seed-sensitive** | `results/gain/*/gain.json`; magnitude falls 0.113 → 0.021 → 0.025 across seeds |
| Improves robustness under an acquisition shift | **not supported** | both the radiometric-gain and anisotropic-resolution pilots are null |
| Cross-sensor / cross-resolution generalisation | **blocked on GPU** | needs full release + second source |

Negative results are part of the method record and are reported in
`docs/architecture_proposals.md` §8, not dropped.

## 9. Reproducibility

```
python -m saryolo train --exp configs/exp/REAL-004_hrsid_cfar.yaml
python -m saryolo robustness --weights results/runs/REAL-004/weights/best.pt \
    --data configs/datasets/hrsid_real.yaml --imgsz 320 --limit 60
python -m saryolo gain --weights results/runs/REAL-004/weights/best.pt \
    --data configs/datasets/hrsid_real.yaml --imgsz 320 --limit 60
```

Every value quoted above is written by the run that measured it; the ledger
(`saryolo/tracking/ledger.py`) renders an unmeasured cell as `TBD` rather than a
guess, and `python -m saryolo assets --require-complete` refuses to assemble a paper
around an unmeasured number.
