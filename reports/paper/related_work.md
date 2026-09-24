# 2 Related work — skeleton

**Source of truth:** `docs/research_gap.md` (the living literature matrix).
This section must never assert a novelty claim that document does not support.

**Required coverage** (master workflow Phase 55):

| Subsection | Must cover | Differentiation needed |
| --- | --- | --- |
| 2.1 SAR object detection | SARDet-100K, HRSID/SSDD-era YOLO detectors, transformer detectors | in-domain accuracy vs our generalisation focus |
| 2.2 Acquisition-aware representation | SARFormer (acquisition-parameter encoding), sensor-aware pretraining | they condition *representation learning*; we diagnose *detection* failure under LOSO |
| 2.3 Domain generalisation / adaptation | classification-side DG; detection-side DA | no LOSO detection protocol in the SAR literature we found — verify before claiming |
| 2.4 Conditional feature modulation | FiLM, conditional norm, adapters | mechanism is borrowed and credited; contribution is the diagnosis that motivates it |
| 2.5 Crowded-module literature | wavelet/frequency/speckle YOLO plug-ins (WSF-YOLO etc.) | cited to explain what this paper deliberately is *not* |

**Rule:** if a 2.3 search later finds an existing LOSO SAR detection benchmark,
the contribution list shrinks to the mechanism + analysis — state that
honestly in `docs/research_gap.md` first.
