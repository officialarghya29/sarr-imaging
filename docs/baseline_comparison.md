# Baseline comparison plan

**Purpose.** Define what SARVO is compared against, on what task, under what protocol, and
— critically — which comparisons are *fair* and which are not. A number from one paper
placed next to a number from another paper is not a comparison unless the datasets, splits,
metrics and protocols are compatible. This document states the compatibility for each row.

---

## 1. The three comparison families (master context)

They are **not** interchangeable, and each needs its own task definition.

| Family | Task | What a fair comparison means here |
| --- | --- | --- |
| **A — SAR detectors** | complete detection systems | same dataset, same split, same input size, same metric, same training budget |
| **B — general detectors / pretrained backbones** | detection, different init | same detector, differing only in initialisation (`init:` key, `EXP-401…403`) |
| **C — parameter-efficient adaptation** | adapting a shared backbone | same backbone, same detector, matched data and budget; trainable vs total params reported separately |

Comparing across families (e.g. our detector vs a LoRA-adapted DINO) is only valid when the
backbone and detector are shared. Where they are not, the comparison is reported as
**incompatible** rather than as a head-to-head result.

---

## 2. Family A — detector baselines

| Baseline | Implementation status | Data | Protocol | Comparable to us? |
| --- | --- | --- | --- | --- |
| **YOLO11 (stock)** | ✅ reproduced bit-exactly | SSDD/HRSID | our LOSO + in-domain | **Yes** — the reference |
| AC-YOLO | not implemented | SSDD/HRSID | their protocol | **Partially** — different baseline detector version; only cost and reported mAP are comparable, and only with their caveats stated |
| RLE-YOLO | not implemented | SSDD/HRSID | their protocol | **Partially** — YOLOv8 family; mAP50 comparable in kind, not in protocol |
| SARDet-100K / MSFA | benchmark + reported baselines | SARDet-100K | their benchmark | **Yes for the benchmark**, no for the pretraining comparison (pretraining-heavy, excluded by compute) |

**Rule applied:** a reported number enters our tables only through
`saryolo/evaluation/reported_baselines.py`, which records the venue, the URL, and what was
and was not captured. It is rendered in a separate section and can never satisfy a measured
cell (`tests/test_efficiency_frontier.py`).

---

## 3. Family B — initialisation baselines

Config-level, not architecture-level. `EXP-401…403` share one model YAML, one seed and
every training argument; the only stated difference is `init:`.

| Arm | Init | Purpose |
| --- | --- | --- |
| `EXP-401` | `none` (fresh build) | the control |
| `EXP-402` | `coco11` (`yolo11s.pt`) | the standard RGB→SAR fine-tuning route |
| `EXP-403` | a stated SAR checkpoint path | the pretraining route (MSFA-style), when weights are available |

The runner records the init stage, the source checkpoint, and the exact number of tensors
whose values changed — excluding zero-initialised buffers, which a raw intersection
over-counts by ~5×. A transfer that would change nothing raises rather than running as a
random-init wearing a pretrained label.

---

## 4. Family C — parameter-efficient adaptation (**not yet implemented**)

The master context requires this comparison and it does **not** exist in the repository.

| Method | Status | What it would need |
| --- | --- | --- |
| Full fine-tuning | ✅ (the default) | — |
| Frozen backbone + trainable head | ❌ | a config-level freeze switch |
| **LoRA** | ❌ **missing** | a LoRA adapter on the backbone's conv/attention blocks, a config switch, and a fair-comparison harness |
| Adapters (ours) | ✅ Component 33 | — |

**Why this matters.** Without a LoRA arm, SARVO's "efficient adaptation" claim has no
parameter-efficient comparator, and the master context (Direction B, Stage 10) is not
satisfied. This is the highest-value *buildable-without-a-GPU* gap in the repository —
see `reports/agent_initial_audit.md` §6, action 2.

**Fair-comparison checklist for the LoRA arm (Stage 10):** shared backbone and detector;
identical training/validation data; matched init; matched input resolution and
augmentation; comparable training budget; tuned on validation only; evaluated on the same
held-out data; trainable and total parameters reported separately; actual training and
inference cost measured, not inferred from parameter count.

---

## 5. Metrics reported for every comparison

| Metric | Source | Notes |
| --- | --- | --- |
| mAP50, mAP50:95 | ledger | COCO protocol, our implementation (deviations documented) |
| precision, recall | ledger | |
| AP_small / medium / large | ledger | the paper's core claim is scale-wise |
| trainable / total params | efficiency module | reported **separately** |
| GFLOPs @ stated imgsz | efficiency module | the size travels with the number |
| latency, FPS | efficiency module | measured after warm-up; CUDA synchronised when on GPU |
| peak GPU memory | efficiency module | forward+backward, not forward |
| training time | ledger | |

---

## 6. Current state of every comparison

| Comparison | Cost | Accuracy |
| --- | --- | --- |
| SARVO vs YOLO11 (ours) | ✅ measured | ❌ `TBD` |
| SARVO-Lite vs full v2 (ours) | ✅ measured | ❌ `TBD` |
| SARVO vs AC-YOLO / RLE-YOLO / SARLite | ⚠️ reported-only, cited, caveated | ❌ not comparable without a run |
| SARVO vs RGB-pretrained init | — | ❌ `TBD` (`EXP-401…403`) |
| SARVO vs LoRA | ❌ not implemented | ❌ `TBD` |

**Summary.** Every *cost* comparison that can be made without a GPU has been made and is
measured. Every *accuracy* comparison is unmeasured, and the LoRA comparator does not yet
exist.
