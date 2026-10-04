# SSAC — focused literature review and mechanism assessment

**Date:** 2026-10-04
**Status:** decision document. Phases 1 and 3 of the SARVO master workflow.
**Method:** every claim about this repository is a measurement (the command or test that
produced it is named). Every claim about prior work is labelled with its evidence depth —
`verified` (page read), `abstract-read`, or `snippet-read` — and no prior-work fact is
stated beyond that depth. Nothing here declares SSAC novel. The assessment reaches a
*recommendation*, and it says plainly which part of SSAC is already taken and which part
could be a contribution.

This document answers Task 2 and Task 3 of the master workflow: conduct a focused
literature review, identify the closest methods to SSAC, and decide whether SSAC is
distinct enough to develop, revise or replace.

---

## 1. What SSAC is, stated as a testable proposition

**Scatter-Selective Adaptive Computation:** a detector should spend expensive feature
processing only where image evidence says it is useful. Formally, given a feature map
`F`, learn a cheap per-region assessment `g(F) ∈ (0,1)^{H×W}` and let it decide, per
region, how much of the expensive path is applied, without discarding weak targets or
mistaking clutter for objects.

The proposition has three separable parts, and they have very different novelty:

| Part | Content | Novelty risk |
| --- | --- | --- |
| **P1. The principle** | allocate compute per region according to evidence | **High risk. Well-established.** |
| **P2. The assessment signal** | *what* evidence drives the decision | the only place a SAR contribution can live |
| **P3. The preservation rule** | how weak/small targets are protected from being skipped | can be a structural contribution |

The review below is organised to show that P1 is taken, and that the defensible design
must put its weight on P2 and P3.

---

## 2. The closest prior work, and what it already owns

### 2.1 Adaptive computation (the mechanism family)

| Work | Venue | What it does | Evidence | What it owns that SSAC cannot claim |
| --- | --- | --- | --- | --- |
| **Spatially Adaptive Computation Time (SACT)** | CVPR 2017 | predicts a **halting score per spatial position** and executes a variable number of residual blocks per region | `abstract-read` ([arXiv:1612.02297](https://arxiv.org/abs/1612.02297)) | *The exact core idea*: per-region evidence → per-region compute. SSAC's "region assessment → allocation" is SACT's formulation. |
| **SplatNet** | ECCV 2018 | learns per-region spatial resolution from a gating/uncertainty signal | `snippet-read` | Region-varying *resolution*. Candidate C of the master workflow is this, generalised. |
| **Dynamic Neural Networks: A Survey** | IEEE TPAMI 2022 | taxonomy of sample-wise, spatial-wise and temporal-wise dynamic networks, incl. explicit **"region selection"** | `snippet-read` ([survey](https://www.computer.org/csdl/journal/tp/2022/11/09560049/1xtOo9Elzbi)) | The whole design space is already mapped and named. "Region selection network" is a survey category. |
| **Survey: dynamic NN for vision → multi-modal fusion (2025)** | arXiv 2025 | consolidates adaptive/dynamic/conditional vision networks | `snippet-read` ([arXiv:2501.07451](https://arxiv.org/abs/2501.07451)) | Confirms the family is active and crowded in 2025. |
| **AdaDets / input-adaptive detection** | 2024–2026 | early exits and adaptive routing **for object detection specifically** | `snippet-read` ([MDPI Electronics 2026](https://www.mdpi.com/2079-9292/15/11/2310)) | Adaptive computation *inside a detector* is done. Efficiency-by-adaptivity for detection is not an open direction. |
| **Spatially adaptive inference at coarse granularity** | arXiv 2022 | latency-aware spatial dynamic networks; notes pixel-level adaptivity often fails to convert to wall-clock | `snippet-read` ([arXiv:2210.06223](https://arxiv.org/html/2210.06223v1)) | **The exact caveat that threatens SSAC** (see §5). |
| **Two-stage detectors (R-CNN family), cascades** | 2014–2018 | spend expensive features only on evidence-selected *regions* | `abstract-read` (prior knowledge) | Region-selective expensive processing is 12 years old; a proposal mechanism selects regions before the expensive network. |

**Conclusion for P1:** "learn to allocate computation per region from image evidence" is
not a novel principle. It is SACT + the dynamic-network programme + the two-stage-detector
tradition. Any SARVO claim built on P1 alone would be a rename, which the master workflow
explicitly forbids ("Do not claim novelty based on a new name").

### 2.2 SAR-specific feature processing (where P2 has to live)

| Work | Venue | What it does | Evidence | Relationship to SSAC |
| --- | --- | --- | --- | --- |
| **CFARNet** — deep learning with a CFAR constraint | arXiv 2022 | learns a detector whose output satisfies a **differentiable CA-CFAR** constraint; CFAR decides the statistic | `abstract-read` ([arXiv:2208.02474](https://arxiv.org/html/2208.02474v3)) | The closest thing to "a learned detector built on the CFAR statistic". It uses CFAR as a *constraint on the output*, not as a signal that allocates computation. |
| **Neural-network CFAR (NN-CFAR)** | DTIC 2023 | augments CA-CFAR with a learned component | `snippet-read` | Learned CFAR is an established idea; "CFAR" cannot be the novelty. |
| **SARFormer** | CVPRW 2025 | acquisition-parameter-conditioned ViT for reconstruction/segmentation | `verified` (repo audit `docs/literature_audit.md`) | Closest *SAR-conditioning* work; no detection, no compute allocation. |
| **SARDet-100K / MSFA** | NeurIPS 2024 | 10-source SAR detection benchmark; diagnoses the RGB→SAR gap | `abstract-read` | Supplies the benchmark and the motivation, not a competing mechanism. |
| **(this repository) multi-scale log-ratio / coefficient-of-variation statistics** | measured here | a differentiable CFAR statistic used as the detector's first representation | **`verified`, measured** — `REAL-004` vs the matched-cost control `REAL-006` | Already built, already measured. This is the SAR signal SSAC should reuse rather than reinvent. |
| **(this repository) "input-adaptive" components** | measured here | SAA, AMF: content-adaptive *weights*, not compute allocation | `verified` — `tests/test_arch.py` | These are *not* adaptive computation; they reweight, they do not skip. Worth stating so SSAC is not confused with the existing ladder. |

**Conclusion for P2:** the SAR-native signal that could make SSAC distinctive already
exists in this repository and has been measured on real data: the multi-scale log-ratio /
coefficient-of-variation statistic. A new, unvalidated "scatter descriptor" invented for
SSAC would be worse, not more novel.

### 2.3 The novelty comparison table (master workflow Part 4)

| Item | SACT / SplatNet | Dynamic-NN survey category | AdaDets / input-adaptive detection | CFARNet | **SSAC (proposed)** |
| --- | --- | --- | --- | --- | --- |
| **Core principle** | per-region halting → variable depth/resolution | region selection / dynamic routing | adaptive routing for detection efficiency | CFAR-constrained output | per-region allocation of an expensive refinement, from a SAR statistic |
| **Mathematical formulation** | halting score + cumulative halting threshold | varies by method | early-exit confidence | differentiable CA-CFAR constraint | `g = σ(MLP(statistic(F))/τ)`, `out = F + α(g·F_rich + (1−g)·F_cheap − F)` |
| **Architecture** | inside a residual stage, per position | varies | backbone exits | output/decision stage | per detection level, immediately pre-head |
| **Computation** | variable per region; reported as *saved depth* | varies; survey flags wall-clock gap | early exit at confidence | no adaptive compute | **dense in this implementation** (§5); sparse variant specified, not yet built |
| **SAR relevance** | none (natural images) | none / general vision | none (general detection) | radar/SAR decision statistic | **SAR-native**: log-ratio + local CoV, dimensionless, transfer across brightness |
| **Results** | ImageNet efficiency | — | embedded detection efficiency | radar detection, radar datasets | **none yet** — pilot is running; no number is claimed here |
| **Limitations** | needs sparse execution to pay off; halting is unstable to train | broad, not a contribution | detection-specific but general-vision | a constraint, not allocation; no efficiency claim | **unproven**; the same wall-clock caveat applies; the accuracy question is the only cheaply testable one |
| **Relationship to SSAC** | **direct ancestor — same principle** | SSAC is an instance of "region selection" | **same goal, different domain** | different mechanism, same statistic | — |

The two rows that decide the assessment are the ones about **principle** and **SAR
relevance**. The principle is shared with SACT; the SAR relevance is currently *empty in
the literature for the allocation mechanism* (CFARNet uses CFAR but not to allocate
compute). That gap is narrow, and it is the only honest place for SSAC's novelty.

---

## 3. The research gap SSAC can actually claim

Not "adaptive computation for SAR" (the principle is old, and the survey already names
region selection). The candidate gap is narrower:

> **A region-selective computation mechanism whose allocation is driven by an analytic,
> dimensionless SAR detection statistic, with a structural guarantee that no region —
> including weak and small targets — loses its cheap-path representation.**

Three parts, and each has a control that could kill it:

1. **The signal matters.** `ssac` (statistic-driven) must beat `ssac_raw` (the identical
   scorer fed raw features). If it does not, the SAR signal is decoration and the
   contribution is a generic dynamic network.
2. **The allocation matters.** `ssac` must beat `ssac_fixed` — the parameter-identical
   fixed-computation control. If it does not, the benefit was the extra parameters and the
   refinement, not the adaptivity. *This is the master workflow's key control.*
3. **Preservation is real.** Small/weak-target recall must not drop relative to the
   fixed-computation control. If the model buys efficiency by missing weak returns, the
   mechanism contradicts its own motivation and must be withdrawn.

---

## 4. Assessment: proceed, revise, or replace?

**Recommendation: PROCEED WITH SSAC, REVISED.** Do not replace it; do not present it as
stated.

### Why not replace

The master workflow offers "revise or replace" only if SSAC substantially duplicates
existing work. It partially does — P1 is duplicated by SACT and the dynamic-network
programme. But the repository's *recommended* architecture direction (RS-CFAR, in
`docs/architecture_proposals.md`) is an **input representation**, and its measured benefit
was **subsumed by augmentation** (`+0.0139 → +0.0067 → +0.0016` as augmented views grow)
at a real latency cost (3.0× per-image on CPU). Retaining a mechanism whose measured value
collapses is not "preserving completed work"; it is preserving a negative. SSAC addresses
a different axis — *how much computation* a region receives rather than *what the first
convolution sees* — and that axis is not yet occupied in this repository. Both are SARVO
candidates; SSAC is the more distinctive one because P1's crowding is compensated by P2's
SAR signal, whereas RS-CFAR's statistic was already shown to be learnable from data.

### Why revise, not adopt as stated

Three revisions are forced by the review:

1. **Do not claim the principle.** State SACT, the dynamic-network survey, two-stage
   detectors, and AdaDets as prior work and position SSAC as a *SAR-specific instance* of
   region-selective computation. Any other framing invites a one-line rejection.
2. **Make the assessment signal explicitly SAR-native and reuse the measured statistic.**
   The design's assessment stack is the same log-ratio / local-CoV statistic already built
   and measured (`REAL-004`). This is not recycling for convenience: it is the only signal
   in this repository with measured SAR evidence behind it, and it converts P2 from an
   assertion into a comparison (`ssac` vs `ssac_raw`).
3. **Keep the target-preservation guarantee structural, not learned.** "Do not discard
   weak targets" must not rest on the assessment being correct. So the cheap path always
   runs for every region and the allocation only *adds* expensive processing; the failure
   mode becomes "wasted compute", never "erased target". This is the design decision that
   makes Experiment 4 (small objects) a property of the architecture rather than a hope.

### The two risks that could still withdraw it

* **The wall-clock risk (P1's known trap).** The 2022 latency-aware spatial-dynamic
  literature and this repository's own CFAR measurement both say the same thing: adaptive
  computation that saves *theoretical* FLOPs frequently does **not** save *measured*
  latency, because sparse region execution needs a sparse kernel and dense tensor
  implementations pay the full cost. On this CPU-only machine SSAC is implemented densely,
  so its measured cost is expected to *increase*, not fall, and the efficiency claim cannot
  be made from this prototype. The accuracy claim (Experiments 1–4) *is* answerable and is
  the reason the prototype is built. If SSAC fails to beat the fixed-computation control on
  accuracy, the mechanism is withdrawn and the negative is recorded.
* **The novelty-narrowness risk.** "SAR statistic drives region selection" is a thin claim.
  It is stated here as thin, and the design document names the controls that would let a
  reviewer reject it.

---

## 5. What this document does not claim

* No accuracy number. The pilot runs after this document; nothing here is a result.
* No novelty declaration. The candidate gap in §3 is `OPEN` until `ssac` beats both
  `ssac_raw` and `ssac_fixed` under a controlled protocol.
* No efficiency claim. The mechanism is dense in this implementation; the sparse variant
  is specified (`docs/ssac_design.md` §4) but unbuilt, and no saving is claimed for it.
* No claim that the repository's existing "adaptive" components (SAA, AMF) are adaptive
  computation. They are not — they reweight features; they do not allocate compute. The
  distinction is stated so the SSAC claim cannot be inflated by them.

## 6. Sources, with evidence depth

* Figurnov et al., *Spatially Adaptive Computation Time for Residual Networks*, CVPR 2017 —
  `abstract-read`.
* Wang et al., *Dynamic Neural Networks: A Survey*, IEEE TPAMI 2022 — `snippet-read`.
* Montello et al., *A survey on dynamic neural networks: from computer vision to
  multi-modal sensor fusion*, arXiv 2025 — `snippet-read`.
* *Latency-aware Spatial-wise Dynamic Networks*, arXiv 2022 — `snippet-read`.
* Lee et al., *Input-Adaptive Dynamic Neural Network for Efficient Object Detection*,
  MDPI Electronics 2026 — `snippet-read`.
* Diskin et al., *CFARNet: deep learning for target detection with constant false alarm
  rate*, arXiv 2022 — `abstract-read`.
* Wagner et al., *Neural Network CFAR Detection*, DTIC 2023 — `snippet-read`.
* Prexl et al., *SARFormer*, CVPRW 2025; Li et al., *SARDet-100K*, NeurIPS 2024 — `verified`
  and `abstract-read` respectively (see `docs/literature_audit.md`).
* This repository's own measurements: `docs/architecture_proposals.md` §8 (`REAL-004`,
  `REAL-006`, the augmentation subsumption), `tests/test_cfar_frontend.py`.
