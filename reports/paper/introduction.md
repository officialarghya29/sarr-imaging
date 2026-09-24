# 1 Introduction — skeleton

**Sources:** `docs/METHOD.md` (motivation), EXP-005/006 (the failure numbers,
currently unmeasured), `docs/research_gap.md` (the gap).

**Narrative arc** (from the master workflow §53 — the story to build toward):

1. SAR detection is deployed where optical imagery is unavailable; detectors are
   strong in-domain.
2. The acquisition (sensor, resolution, polarization, band, incidence) changes
   the image statistics; a detector trained on one acquisition degrades on
   another. → **BLOCKED-ON-RUN: quantify with the cross-source table.**
3. Existing responses are module insertions; the literature is crowded with
   frequency/wavelet/attention blocks (cite `docs/research_gap.md` rows).
4. We ask the diagnostic question first: *do detector features encode the
   acquisition more strongly than the object?* → **BLOCKED-ON-RUN: probe
   result.**
5. We answer with a protocol (LOSO, metadata withholding, capacity-matched
   controls) and a minimal mechanism (metadata-conditioned adapter).
6. Contributions — *candidates only until freeze (Phase 58)*:
   - C1 the failure analysis + protocol,
   - C2 the diagnosis (probe) methodology,
   - C3 the conditioning mechanism + ablation,
   - C4 multi-seed, multi-arm evaluation harness with enforced traceability.
   Drop any candidate the experiments do not support.
