# 6 Limitations — skeleton

Seeded from `reports/red_team_review.md`. Limitations are stated in the paper
in the same order the review ranked them — the most damaging first, because a
reviewer reads them anyway.

1. **Metadata availability at inference (W2).** The adapter needs acquisition
   metadata. On an unseen sensor the honest input is "unknown" — the model
   encodes to a reserved unknown row and degrades. *BLOCKED-ON-RUN: the
   withholding curve; if collapse, this limitation moves into the abstract.*
2. **One detector family (W6).** RT-DETR transfer is not yet implemented;
   architecture-generality is *not claimed* in the current draft.
3. **Geography vs sensor (W3).** Sources are not scene-independent; the LOSO
   gap must survive a scene-blocked control before it is called a sensor gap.
4. **Dataset scope (W8).** HRSID ships no per-chip resolution mapping; the
   cross-resolution axis is inert until a sourced sidecar exists. SARDet-100K
   scale-up is pending.
5. **Statistical power (W5).** Three seeds bound what differences are
   detectable; effects within one std are reported as undetectable.
6. **Controlled corruptions are not acquisitions (Phase 37).** Robustness
   numbers describe synthetic corruption only — never equated with real
   sensor shift.
