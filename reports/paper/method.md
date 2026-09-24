# 3 Method — skeleton

**Source of truth:** `docs/METHOD.md` and the module docstrings. Every
subsection below maps to code that exists and is tested — the method section
may not describe a component the repository does not contain.

| Subsection | Content | Code |
| --- | --- | --- |
| 3.1 Problem setup | acquisition metadata formalisation; LOSO protocol | `saryolo/data/metadata.py`, `saryolo/data/groups.py` |
| 3.2 Baseline graph | YOLO11-family graph, 11 components, identity-at-init contract | `saryolo/nn/arch.py` |
| 3.3 Metadata-conditioned adapter | embeddings + physical descriptors; gate at zero; unknown-row encoding | `saryolo/nn/modules/conditioning.py` |
| 3.4 Metadata withholding | withheld field ≡ never-recorded field (the equivalence test) | `saryolo/data/field_mask.py` |
| 3.5 Probes | sensor-vs-class separability, within-class drift, CKA | `saryolo/evaluation/probes.py` |
| 3.6 Training objective | detection loss only for the main arms; loss ablations elsewhere | `saryolo/nn/losses.py` |

**Figure plan:** Fig. 1 architecture (metadata path highlighted); Fig. 2 the
conditioning module; Fig. 3 the *key figure* (same class, different sensor,
baseline vs conditioned representation).

**Honesty note:** the adapter is FiLM-style conditional modulation — say so
plainly, cite it, and locate the novelty in the protocol + diagnosis (see
`reports/red_team_review.md` W1).
