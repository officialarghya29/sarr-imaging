# Research gap audit (Phase 0)

**Status: first pass, not closed.** This document records what a literature sweep on
2026-09-23 actually surfaced, with a link for every entry, and refuses to declare the
gap closed until the flagged full-text reads are done (RULE 12: no novelty claim
before the check). Entries are limited to work verified in search results during this
session; anything not verified is listed as *open*, not invented.

## The claim whose novelty is at stake

> A lightweight acquisition-metadata-conditioned adapter (FiLM-style, per-sample,
> continuous-field path) improves **cross-sensor and cross-resolution** SAR object
> detection under a guarded **leave-one-source-out** protocol, at <0.5% parameter
> overhead, without SAR-specific pretraining.

## Verified landscape

| Work | Venue (verified) | What it does | What it does **not** do | Overlap with our claim |
| --- | --- | --- | --- | --- |
| [SARDet-100K + MSFA](https://arxiv.org/abs/2403.06534) | NeurIPS 2024 (spotlight) | 10-source SAR detection benchmark; diagnoses the RGB-pretraining → SAR-finetuning gap; fixes it with multi-stage pretraining | Detection-side conditioning; cross-source protocol; low-compute route (MSFA is pretraining-heavy — excluded by our compute constraint) | Supplies our primary dataset and the domain-gap motivation; complementary, not competing |
| [SARFormer](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Prexl_SARFormer_-_An_Acquisition_Parameter_Aware_Vision_Transformer_for_Synthetic_CVPRW_2025_paper.html) ([arXiv](https://arxiv.org/abs/2504.08441)) | CVPR 2025 Workshop (EarthVision) | Encodes acquisition parameters to guide SAR representation learning (ViT, multi-view) | Object detection; unseen-sensor evaluation; lightweight detector adapters | **Read (abstract, 2026-09-23):** downstream tasks are height reconstruction and segmentation — detection is not touched and no cross-sensor protocol exists. Closest published use of acquisition metadata; must be cited and differentiated: conditioning for *detection generalisation*, not representation learning |
| [SARMAE](https://github.com/MiliLab/SARMAE) ([CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/37977)) | CVPR 2026 | Noise-aware masked-autoencoder SAR pretraining (SAR-1M) | Detection conditioning; metadata use | Confirms SAR-specific pretraining is already taken as a direction — strengthening our "no pretraining" angle |
| [Semantic scattering graph alignment for cross-sensor SAR detection](https://www.sciopen.com/article/10.1016/j.cja.2026.104066) | Chinese Journal of Aeronautics 39(6), 2026 (open access; abstract read in full) | Builds a semantic scattering graph from sampled scattering points; GCN context; hierarchical node+structure alignment across domains; +5–40% mAP/F1 on cross-sensor tasks | Metadata conditioning (no acquisition fields anywhere in the method); continuous-field transfer to *unseen* sources; guarded LOSO protocol; architecture generality | **Read (2026-09-23):** mechanism is image-derived structure alignment (domain-adaptation family); ours is metadata-side conditioning. The overlap is at the *problem* level — and its 5–40% cross-sensor spread is citable evidence that the failure we measure is real and unsolved. Position against it as an architecture-side fix vs. our conditioning-side fix |
| [Generalizing SAR object detection: unified cross-scenery DG framework](https://ui.adsabs.harvard.edu/abs/2025IGRSL..22L5173Z/abstract) | IEEE GRSL 2025 | Domain-generalisation framework for heterogeneous SAR detection | Acquisition-metadata conditioning | Direct neighbour; abstract-level verification only |
| LGM-Det (lightweight geometry-aware cross-sensor detector) and [dynamic feature discrimination + center-aware calibration](https://www.researchgate.net/publication/390686512_Cross-Sensor_SAR_Image_Target_Detection_Based_on_Dynamic_Feature_Discrimination_and_Center-Aware_Calibration) | 2025–2026 (journal, verified via search) | Cross-sensor SAR detection via architectural / calibration changes | Metadata conditioning; architecture-generality claim | The crowded "another cross-sensor module" neighbourhood our brief warns about. Abstract-level verification only |
| [SAR ship detection across spaceborne platforms](https://www.sciencedirect.com/science/article/abs/pii/S0924271625002795) | ISPRS Journal, 2025 | Cross-platform ship detection | (abstract not yet read) | Related; read pending |
| [FiLM](https://arxiv.org/abs/1709.07871) (+ [Distill overview](https://distill.pub/2018/feature-wise-transformations)) | AAAI 2018 | Feature-wise linear modulation as a general conditioning layer | SAR; detection; unseen-domain evaluation | We use the mechanism, not claim it. Novelty cannot rest on FiLM itself |
| [Few-shot SAR object detection survey](https://www.mdpi.com/2072-4292/18/15/2580) | Remote Sensing, 2026 | Surveys support–query conditioning in SAR FSOD | Cross-sensor shift as the axis | Context for the low-shot experiment (Experiment F) |

## What the sweep says is open

Updated after the two highest-priority reads (2026-09-23):
1. **Metadata-conditioned *detection* with a continuous-field transfer path.**
   Conditioning exists (SARFormer, representation learning); cross-sensor detection
   exists (alignment/DG modules). The combination — per-sample conditioning inside a
   YOLO-family detector whose categorical fields are *unavailable at test time by
   protocol*, evaluated leave-one-source-out — is not claimed by anything verified above.
2. **A guarded cross-resolution protocol derived from physical bins.** Cross-resolution
   detection has dedicated work generally, but the sweep found no SAR detection
   protocol that bins a stated metadata field and holds out a bin, with refusals for
   the silent-failure modes (single bin, unkeyed images, zero-ground-truth splits).
3. **Architecture generality.** Nothing verified claims the *same* conditioning
   mechanism transferring across YOLO and DETR-family detectors on this problem.

## Threats that must be resolved before the paper

- ~~The 2025–2026 cross-sensor SAR papers flagged above~~ **Resolved for the two
  closest works (SARFormer, Zhang et al. 2026):** neither conditions detection on
  acquisition metadata, neither claims a guarded unseen-source protocol. Remaining:
  the GRSL 2025 DG framework and the ISPRS 2025 cross-platform paper (abstract
  reads pending — lower risk, different mechanisms).
- SARDet-100K's own paper may already report per-source breakdowns; if so, our
  "measure the degradation" contribution shrinks to "measure it under controlled,
  guarded folds" — still publishable, but the framing must change.

## Explicitly not claimed

- FiLM conditioning as a mechanism (Perez et al.).
- The domain-gap observation itself (SARDet-100K).
- SAR-specific pretraining (SARMAE et al.) — we argue we don't need it; we don't claim it.
