# Related work and novelty comparison (Phase 2)

**Date:** 2026-10-04
**Status:** open. This is the consolidated comparison the master context asks for; it builds on
`docs/literature_audit.md` (which records *how* each entry was inspected) rather than replacing
it.

**The rule this table follows.** Each row states the work's own idea, its setup, and its
reported numbers, and then the one thing that matters for SARVO: what the *specific* object
under test here is, and how it differs. A similarity is named as plainly as a difference. A
row is not allowed to say "novel", "better", or "state of the art" — those are outcomes of
experiments that have not been run. Evidence depth is carried over from the audit as
**verified** (reached and inspected), **abstract-read**, or **listed** (existence confirmed,
contents not read), so a search snippet can never be read as a fact.

**How to read a row.** `Compute` quotes the authors' reported figures where they exist and
says `not captured` where they do not; reported numbers are the authors', never measured
here. The `Closest overlap` column names the thing a reviewer would challenge; the
`Distinction` column is what SARVO would have to demonstrate, not merely assert.

---

## 1. SAR object detectors

| Work (evidence) | Central idea | Architecture + training | Data + metrics | Compute | Strengths / limitations | Closest overlap with SARVO | Distinction to demonstrate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SARDet-100K / MSFA** ([repo](https://github.com/zcablii/SARDet_100K), [paper](https://arxiv.org/abs/2403.06534)) — abstract-read | A 10-source detection benchmark, plus a multi-stage pretraining recipe that closes the RGB-pretraining to SAR-finetuning gap | Detector-agnostic benchmark; MSFA is a pretraining schedule (large-scale RGB then SAR) | 116k images, 10 sources, 6 classes; standard detection AP | Pretraining-heavy | Strengths: the reference benchmark and the honest statement of the domain gap. Limitation: the fix is compute the brief excludes | Supplies the benchmark and the domain-gap motivation | Our fix is architecture-side and pretraining-free, and must be shown to be worth anything against a *pretrained* baseline, not just an untrained one |
| **AC-YOLO** — abstract-read | Lightweight ship detector with a +1.2/+1.5 AP claim against *their own* YOLO11 baseline | YOLO11 variants, ship-focused | SSDD / HRSID; mAP | reported **−30.0 % params, −15.6 % compute** (relative only) | Strengths: direct efficiency competitor on our pilot datasets. Limitation: reports no absolute cost and no cross-source protocol | Efficiency axis, same datasets | Absolute params/FLOPs/latency measured here under a documented convention, and a cross-source arm AC-YOLO does not have |
| **RLE-YOLO** — abstract-read | Ship detection with a residual-lightweight backbone | YOLOv8-based | SSDD / HRSID; 93.9 / 98.4 mAP50, **−43.9 % params, −34.5 % compute** (relative only) | not captured absolutely | Strengths: the strongest published mAP50 anchor on the pilot datasets. Limitation: in-domain only, relative cost | Accuracy reference point on our pilot datasets | We may not reach their published mAP50 on a 200-image subset; the honest statement is the difference in protocol, and any claim is against our own reproduced baseline |
| **Edge-optimized lightweight YOLO** — abstract-read | A detection model tuned for embedded SAR inference | YOLO-based | SARDet-100K; 87.7 mAP at **1.9 M params** | 1.9 M params | Strength: the absolute size anchor — smaller than SARVO-Lite (n) at 3.04 M Params. Limitation: no robustness axis | The "be small" axis, which the audit already rules out as a contribution by itself | Not size. The claim can only be cost *at a stated operating point under a cross-sensor protocol*, which no entry in this table provides |
| **SARLite** — abstract-read | A slimmed detector reporting a better accuracy/efficiency point than its baseline | YOLO-based | SARDet-100K; **+3.4 % mAP@50:95, −17 % params, −~1 GFLOP** | relative only | Strength: same efficiency axis with a positive accuracy delta. Limitation: relative-only numbers cannot be compared across papers | Same efficiency axis | Absolute measurement plus a protocol the comparison can be re-run under |

---

## 2. General detectors and pretrained backbones

| Work (evidence) | Central idea | Architecture + training | Data + metrics | Compute | Strengths / limitations | Overlap with SARVO | Distinction to demonstrate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **YOLO11** (Ultralytics) — verified | A single-stage anchor-free detector family with a CSP backbone and a PAN-FPN neck | CNN, anchor-free head, BCE/DFL loss | COCO; AP | 2,624,080 n / 9,458,752 s params at 80 classes, reproduced bit-exactly here | Strength: fast, well-tested, the training infrastructure every run in this repo uses. Limitation: it is an RGB natural-image design, and its neck carries a separate parameter set per level | **This is SARVO's current skeleton**, which is exactly why the architecture proposal exists | SARVO must replace an organizing principle of this graph (representation, scaling, or conditioning) rather than bolt modules onto it |
| **RT-DETR** — abstract-read | A real-time DETR with an efficient hybrid encoder and a decoder without NMS | Transformer detector | COCO; AP | not captured here | Strength: removes NMS and the anchor priors. Limitation: heavier at small scales and a different training loop | Feasibility arm in this repo: the graph and conditioning path are verified, the training loop is not | A cross-architecture number (same conditioning mechanism, YOLO and DETR) — currently `BLOCKED-ON-RUN` |
| **DINOv2 / DINOv3** — listed | Self-supervised visual backbones | ViT, no labels | Web-scale pretraining; frozen or fine-tuned features | Large pretraining compute | Strength: strong transferable features. Limitation: pretraining compute the brief's budget excludes, and an RGB-pretrained inductive bias that SARDet-100K shows does not transfer for free | The backbone a published SAR adaptation (SAR-LoRA-DINO) fine-tunes | SARVO argues it does not need pretraining; that argument is only credible against a pretrained control arm, which is built (`EXP-401…403`) and unmeasured |

---

## 3. Adaptation, conditioning, and robustness

| Work (evidence) | Central idea | Architecture + training | Data + metrics | Compute | Strengths / limitations | Closest overlap with SARVO | Distinction to demonstrate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SARFormer** ([paper](https://arxiv.org/abs/2504.08441)) — abstract-read | Condition a ViT on **acquisition parameters** to guide SAR representation learning | ViT, multi-view; acquisition-parameter conditioning | Height reconstruction and segmentation; not detection | not captured | Strength: the closest published use of acquisition metadata, and a strong precedent that the signal is real. Limitation: no detection, no held-out-source protocol | **High** on the metadata-conditioning idea | The object here is conditioning for *detection generalisation*, evaluated leave-one-source-out with the categorical fields withheld by protocol — a different task and a different protocol |
| **SAR-LoRA-DINO** ([repo](https://github.com/shatianming5/dino_sar)) — listed | Adapt a pretrained DINO backbone to SAR with LoRA | LoRA on a ViT; detection head | not inspected | LoRA over a large ViT | Strength: the adaptation comparator the brief requires. Limitation: different backbone and data pipeline, so no head-to-head is possible without rebuilding it | The adaptation axis; the repo has its own LoRA implementation (`saryolo/training/peft.py`) and one measured pilot arm (`REAL-003`) | That the adapter here is a *comparator*, not the architecture — the brief rules out presenting LoRA adaptation as a new detector |
| **ExPLoRA** — abstract-read | Extended pretraining plus LoRA for PEFT under domain shift | LoRA + continued pretraining | Domain-shift benchmarks | Pretraining + adapters | Strength: shows PEFT under shift is an active area. Limitation: not SAR, not detection | PEFT under domain shift | The shift is *acquisition*, and the model must be a detector trained without pretraining |
| **Semantic scattering graph alignment** ([paper](https://www.sciopen.com/article/10.1016/j.cja.2026.104066)) — abstract-read | Build a scattering-point graph and align node and structure statistics **across sensors** | CNN + GCN, hierarchical alignment | Cross-sensor detection; reports a 5–40 % mAP/F1 spread | not captured | Strength: citable evidence that the cross-sensor degradation is real and unsolved, and a same-problem baseline family. Limitation: image-derived alignment, no acquisition fields, no guarded unseen-source protocol | **High at the problem level** | An architecture-side representation change (or a metadata/self-estimated conditioning path) versus a domain-adaptation module — and a guarded LOSO protocol the work does not claim |
| **FiLM** ([paper](https://arxiv.org/abs/1709.07871)) — verified | Feature-wise linear modulation: condition a network on side information via per-channel scale and shift | Generic layer | Any task | negligible | Strength: a clean, cheap conditioning mechanism. Limitation: it is a mechanism, not a method | The mechanism inside this repo's Component 33 | Novelty cannot rest on FiLM. It rests on what is conditioned on, and why |
| **LoRA** ([paper](https://arxiv.org/abs/2106.09685)) — verified | Low-rank weight updates that leave the base frozen | Generic PEFT | Large models | small | Strength: cheap, mergeable, exact at init. Limitation: an adaptation method, not an architecture | The mechanism used by `saryolo/training/peft.py` | Same as FiLM: the mechanism is cited, the contribution is elsewhere |

---

## 4. Verified findings versus working hypotheses

The master context asks for this split explicitly, so it is made in one place.

**Established by someone else, and relied on here (not claims of ours).**

| Finding | Source | How it is used |
| --- | --- | --- |
| Multi-source SAR detection is materially harder than in-domain detection | SARDet-100K (abstract-read) | The motivation for the whole project |
| Acquisition parameters carry information a SAR model can use | SARFormer (abstract-read) | The precedent for conditioning |
| The cross-sensor degradation is large (5–40 % mAP/F1) | scattering-graph work (abstract-read) | Evidence the failure is real, citable in the introduction |
| Parameter-efficient adaptation works under domain shift | ExPLoRA, SAR-LoRA-DINO (abstract-read / listed) | The adaptation comparator's justification |

**Working hypotheses of this repository, each with the experiment that would settle it.**

| Hypothesis | Status | Experiment that settles it |
| --- | --- | --- |
| A radar-statistic input representation improves small-object detection in clutter | `OPEN`, untested | The P1 arm against a matched-cost stem, on the real subset — `docs/architecture_proposals.md` §2 |
| Acquisition state can be estimated from the image well enough to replace metadata | `OPEN`, untested | `s_hat`-only versus metadata vs none, on acquisition-labelled data |
| Weight-tied recursive scale processing transfers across pixel spacings | `OPEN`, unmeasurable here | Needs resolution-keyed real data; HRSID has none |
| Conditioning improves cross-sensor detection under LOSO | `OPEN`, `BLOCKED-ON-RUN` | `EXP-331…338` on the full release |
| The repository does not need SAR-specific pretraining | `OPEN`, `BLOCKED-ON-RUN` | The pretrained-init arms `EXP-401…403` against the no-init arms |

**Not claimed anywhere.** FiLM or LoRA as mechanisms; the domain gap as a discovery;
acquisition-metadata conditioning as a new idea; being "small" as a contribution;
parameter count as a proxy for speed; and any accuracy result whatsoever.

---

## 5. The gap, stated as a question rather than a claim

Reading the three tables together, one thing is missing from the field and it is not
"efficiency" — §1 shows several lightweight SAR detectors, one at 1.9 M parameters. Nor is it
"conditioning" — SARFormer conditioned a SAR model on acquisition parameters in 2025. Nor is
it "cross-sensor detection" — there is a dedicated literature with published deltas.

What none of the entries above does is **make an architectural decision and then show, with
a matched control, that it was the decision and not the extra capacity that helped**. The
published SAR detectors report relative improvements against their own baselines; the
adaptation work reports parameter savings; the conditioning work conditions representation
learning. The reproducible, attributable comparison is the thing this repository is
structurally built to produce (the identity-at-init rule, the ledger, the `TBD` gate), and
that is a defensible position even if every individual number is modest.

The concrete gap SARVO can occupy, pending evidence, is therefore:

> An efficient SAR detector whose *organizing principle* is drawn from radar physics rather
> than from natural-image detection, with its contribution separated from raw capacity by a
> matched-cost control, and reported with absolute cost measured under a documented
> convention alongside a guarded cross-sensor protocol.

Which organizing principle is a separate decision, made in
`docs/architecture_proposals.md` and answered there with one recommendation and the reason
the other two are deferred.
