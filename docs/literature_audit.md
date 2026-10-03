# Literature and implementation audit

**Status: open, evidence-first.** Every entry below is one of three things, labelled
explicitly:

* **verified** — the repository or paper was reached and inspected on 2026-10-03;
* **abstract-read** — the abstract (or the page's own summary) was read in full;
* **listed** — a source the master context names, whose existence was confirmed by
  `git ls-remote` but whose contents were not inspected.

No entry is described as "state of the art", "better than", or "novel" without a control
that would establish it. Where a number is quoted it is the author's reported number and
is marked as such; see also `saryolo/evaluation/reported_baselines.py`, which holds the
machine-readable version of the efficiency-relevant entries.

---

## 1. Repository existence check (executed)

```bash
git ls-remote --heads https://github.com/<repo>.git
```

| Repository | Result | Inspection depth |
| --- | --- | --- |
| `zcablii/SARDet_100K` | EXISTS | listed (not cloned) |
| `shatianming5/dino_sar` (SAR-LoRA-DINO) | EXISTS | listed (not cloned) |
| `yyl404/yolo-sar` | EXISTS | listed (not cloned) |
| `Jordan-Liao/SAR-LoRA-DINO` | EXISTS | listed (not cloned) |

> Two of these timed out on a first attempt and succeeded on retry. Network to GitHub is
> intermittent in this environment; that is a transport fact, not a statement about the
> repositories.

---

## 2. Family A — SAR object detectors

| Method | Venue / year | Task | Backbone | Data | Reported cost/accuracy | Overlap with SARVO |
| --- | --- | --- | --- | --- | --- | --- |
| **SARDet-100K + MSFA** ([repo](https://github.com/zcablii/SARDet_100K), [paper](https://arxiv.org/abs/2403.06534)) | NeurIPS 2024 (spotlight) | Detection, multi-source benchmark | Various | 116k imgs, 10 sources, 6 classes | — | Supplies our primary benchmark and the domain-gap motivation. Complementary, not competing. |
| **AC-YOLO** ([paper](https://doi.org/10.1371/journal.pone.0327362), [code](https://github.com/He-ship-sar/ACYOLO)) | PLOS ONE 2025 | Ship detection | YOLO11 | SSDD / HRSID | −30.0% params, −15.6% compute, +1.2 / +1.5 AP vs *their* YOLO11 baseline (**abstract-read**) | Direct efficiency competitor on our pilot datasets. Reports relative-only. |
| **RLE-YOLO** ([paper](https://ieeexplore.ieee.org/document/10924247)) | IEEE JSTARS 2025 | Ship detection | YOLOv8 | SSDD / HRSID | 93.9 / 98.4 mAP50; −43.9% params, −34.5% compute vs their baseline (**abstract-read**) | Strongest published mAP50 anchor on the pilot datasets. Different detector family. |
| **Edge-optimized lightweight YOLO** ([paper](https://www.mdpi.com/2072-4292/17/13/2168)) | Remote Sensing 2025 | Detection | YOLO | SARDet-100K | 87.7 mAP at **1.9 M params** (**search-snippet-read**) | The absolute size anchor. SARVO-Lite (n) at 3.04 M is *not* smaller — see §5. |
| **SARLite** ([paper](https://www.nature.com/articles/s41598-026-49143-5)) | Scientific Reports 2026 | Detection | YOLO | SARDet-100K | +3.4% mAP@50:95, −17% params, −1 GFLOP vs baseline (**abstract-read**) | Relative-only; same efficiency axis. |
| **SAR-NAS** ([paper](https://arxiv.org/html/2509.01279v1)) | arXiv 2025 | Detection via NAS | YOLOv10 | SARDet-100K | Claims best accuracy/efficiency trade-off (**abstract-read**) | A different route to efficiency (architecture search vs component removal). |

## 3. Family B — general detectors and pretrained backbones

| Method | Venue | Task | Relevance to SARVO |
| --- | --- | --- | --- |
| **YOLO11** (Ultralytics) | — | Detection | Our baseline, reproduced bit-exactly (2,624,080 n / 9,458,752 s at 80 classes). |
| **RT-DETR** | CVPR 2024 | Detection | Our second-architecture feasibility arm; graph integration verified, training loop not. |
| **DINOv2 / DINOv3** | 2023 / 2025 | Self-supervised backbone | The backbone the SAR-LoRA work adapts; a candidate pretrained init for a future arm. |

## 4. Family C — parameter-efficient adaptation and robust representation

| Method | Venue / year | Task | Adaptation | Overlap with SARVO |
| --- | --- | --- | --- | --- |
| **SARFormer** ([paper](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Prexl_SARFormer_-_An_Acquisition_Parameter_Aware_Vision_Transformer_for_Synthetic_CVPRW_2025_paper.html)) | CVPR 2025 Workshop | Height reconstruction, segmentation | Acquisition-parameter-aware ViT | **Closest published use of acquisition metadata.** Detection is not touched; no cross-sensor protocol. Must be cited and differentiated. |
| **SAR-LoRA-DINO** (`shatianming5/dino_sar`) | listed | Detection (adapter) | LoRA on DINO | The adaptation comparator the master context requires. **Not yet implemented here** — see `docs/baseline_comparison.md`. |
| **SARMAE** ([repo](https://github.com/MiliLab/SARMAE)) | CVPR 2026 | SAR pretraining | Masked autoencoder | Confirms SAR-specific pretraining is taken; strengthens the "no pretraining" angle rather than competing. |
| **LoRA** ([paper](https://arxiv.org/abs/2106.09685)) | ICLR 2022 | PEFT mechanism | Low-rank adapters | We would use the mechanism, not claim it. |
| **FiLM** ([paper](https://arxiv.org/abs/1709.07871)) | AAAI 2018 | Feature-wise modulation | Conditioning layer | The mechanism inside Component 33. Novelty cannot rest on FiLM itself. |
| Semantic scattering graph alignment ([paper](https://www.sciopen.com/article/10.1016/j.cja.2026.104066)) | CJA 2026 | Cross-sensor detection | Graph alignment (domain adaptation) | Same *problem*; different family (image-derived structure vs metadata conditioning). Its reported 5–40% cross-sensor spread is citable evidence the failure is real. |
| ExPLoRA | 2024 | PEFT under domain shift | Extended pretraining + LoRA | Relevant to Direction B: shows PEFT under shift is an active area. |

---

## 5. Three plausible research gaps (not yet claimed as novel)

Stated as *candidate* gaps. Each carries the experiment that would establish or kill it.

1. **Metadata-conditioned *detection* with a continuous-field transfer path.**
   Acquisition conditioning exists (SARFormer) but for representation learning, not
   detection; cross-sensor *detection* exists but via domain-adaptation modules, not
   acquisition fields. The untested combination: per-sample conditioning inside a
   YOLO-family detector whose categorical fields are *unavailable at test time by
   protocol*, evaluated leave-one-source-out.
   *Test that would establish it:* `EXP-331…338` under LOSO — specifically that
   `cond_continuous` (physical descriptors only) recovers part of the baseline's
   cross-source degradation, and that `cond_sensor` does not transfer by construction.

2. **A guarded cross-resolution protocol derived from physical bins.**
   Cross-resolution detection has dedicated work, but this audit found no SAR detection
   protocol that bins a *stated metadata field* and holds out a bin with explicit refusals
   for the silent-failure modes (single bin, unkeyed images, zero-GT splits).
   *Test that would establish it:* `loso --rule resolution --edges ...` on a dataset with a
   sourced per-image resolution field. **Blocked** on HRSID's missing per-chip mapping.

3. **Architecture generality of one conditioning mechanism.**
   Nothing verified claims the *same* conditioning mechanism transferring across YOLO and
   DETR families on this problem.
   *Test that would establish it:* a LOSO number from the RT-DETR arm. **Blocked** on the
   unfinished training-loop integration.

**A gap this audit explicitly does *not* claim:** efficiency alone. §2 shows the field is
crowded with lightweight SAR detectors, and at least one reports 1.9 M parameters. A
"smaller model" is not a contribution here; the defensible efficiency claim is *cost at a
stated operating point on the cross-sensor problem*, which none of the above addresses.

---

## 6. What must be cited regardless of outcome

SARDet-100K (benchmark + domain gap), SARFormer (acquisition parameters, representation
learning), FiLM (the modulation mechanism), LoRA (the adaptation baseline), and the
specific SAR detectors we compare cost against (AC-YOLO, RLE-YOLO, edge-optimized YOLO,
SARLite). Each dataset paper is cited in `docs/DATASETS.md`.
