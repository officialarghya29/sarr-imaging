# Novelty and overlap

**Purpose.** State exactly what SARVO could claim, what it must not claim, and the
evidence each claim needs. Nothing here is a novelty declaration; every row is
*conditional* on an experiment that has not been run. If an experiment fails, the
corresponding claim is deleted rather than rephrased.

---

## 1. The claim under test

> A lightweight acquisition-metadata-conditioned adapter improves **cross-sensor and
> cross-resolution** SAR object detection under a guarded **leave-one-source-out**
> protocol, at **<0.5 % parameter overhead**, without SAR-specific pretraining.

Every clause of that sentence is falsifiable, and each has a control that could kill it:

| Clause | Control that could kill it | Where it lives |
| --- | --- | --- |
| "conditioning improves cross-sensor" | `cond_continuous` fails to beat `v2_full` under LOSO | `EXP-331…338` |
| "acquisition metadata, not just any adaptivity" | `cond_sensor` succeeds while `cond_continuous` fails → the method specialises to known sensors | the field-set arms |
| "<0.5 % parameter overhead" | measured on the light model, not asserted | `tests/test_efficiency_frontier.py` |
| "without SAR-specific pretraining" | a pretrained-init arm (EXP-402/403) beats it by more than the conditioning gain | `EXP-401…403` |
| "guarded LOSO" | a fold that leaks, or a rule that silently random-splits | `saryolo/data/groups.py` refusals |

---

## 2. What is **not** claimed

| Thing | Why not |
| --- | --- |
| FiLM / feature-wise modulation as a mechanism | Perez et al., 2018. We use it; we do not claim it. |
| The domain gap itself | SARDet-100K diagnoses it. |
| Acquisition-metadata conditioning as a new idea | SARFormer conditions a ViT on acquisition parameters for reconstruction/segmentation. |
| SAR-specific pretraining | SARMAE et al. have taken that direction; we argue we do not need it, which is a different statement. |
| Being "small" | The field already reports a 1.9 M-parameter detector (see `docs/literature_audit.md` §2). |
| Parameter count as a proxy for speed | A cheaper parameter count is not a faster model; latency is measured separately (`saryolo/evaluation/efficiency.py`). |
| State-of-the-art accuracy | No accuracy has been measured. Not claimable in any form. |

---

## 3. Overlap map

| Work | Task | Backbone | Adaptation | Compute | Protocol | Overlap | Differentiation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SARFormer | Height reconstruction, segmentation | ViT | Acquisition-parameter conditioning | Not captured | Multi-view | **High** on the metadata idea | Different task (no detection), no cross-sensor held-out protocol, not a detector adapter |
| SARDet-100K / MSFA | Detection | Various | Multi-stage pretraining | High (pretraining-heavy) | Multi-source benchmark | Medium | We use their benchmark; our fix is architecture-side and pretraining-free |
| Zhang et al. 2026 (scattering graph) | Cross-sensor detection | CNN + GCN | Image-derived structure alignment | Not captured | Cross-sensor | Medium (problem level) | Architecture-side alignment vs our metadata-side conditioning |
| AC-YOLO / RLE-YOLO / SARLite | Ship detection | YOLO | Architectural, lightweight | Reported relative-only | In-domain | Efficiency axis | In-domain single/two-dataset; no cross-source protocol |
| SAR-LoRA-DINO | Detection | DINO | LoRA | Listed, not inspected | Not inspected | Adaptation axis | Different backbone; we cannot claim a head-to-head without a compatible setup |
| ExPLoRA | PEFT under domain shift | ViT | LoRA + extended pretraining | Not captured | Domain shift | Adaptation under shift | Not SAR detection; not a detector adapter |

---

## 4. The two claims the field leaves open

1. **Metadata-conditioned detection with a continuous-field transfer path**, evaluated
   leave-one-source-out with the categorical fields withheld by protocol.
2. **The same conditioning mechanism across detector families** (YOLO and DETR) on this
   problem.

Both are *candidate* contributions. Claim 1 needs the LOSO run; claim 2 needs the RT-DETR
training-loop integration and a LOSO number. Neither exists yet — both are `OPEN` and
**BLOCKED-ON-RUN**.

---

## 5. Threats a reviewer will raise, and the answer

| Threat | Current answer | Honest status |
| --- | --- | --- |
| "Conditioning is just extra capacity" | Adapter is <0.5 % params, gated, and identity at init; the field-set arms are near capacity-matched | Built and tested; needs the LOSO number |
| "Metadata is a proxy for geography" | Red-team W3 in `reports/red_team_review.md` | **Unresolved** — needs a fold that varies sensor while holding geography |
| "Only one architecture" | RT-DETR feasibility arm exists | Feasibility only; no number |
| "Insufficient statistical power" | Multi-seed aggregation built (`build_multi_seed`) | Needs ≥3 seeds per arm on a real dataset |
| "The efficiency comparison is unfair" | Reported baselines are cited with their baselines and caveats, never mixed with our measurements | Structurally enforced; the measured comparison awaits a run |

---

## 6. The rule this document exists to enforce

A claim may move from "candidate" to "contribution" **only** by a control experiment in
this repository producing a number. Absent that, the claim stays conditional, and the
paper says so. This is the same rule the ledger enforces mechanically: an unmeasured cell
renders `TBD`, and `python -m saryolo assets --require-complete` refuses to assemble a
paper around it.
