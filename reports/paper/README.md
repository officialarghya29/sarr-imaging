# Paper structure (SARVO Phase 55)

This directory holds the paper's section skeletons as markdown. They are the
writing plan, not the paper: every claims-bearing paragraph is marked
**BLOCKED-ON-RUN** until the experiment it depends on has a ledger row. The
rendered manuscript lives in [`paper/manuscript/main.tex`](../../paper/manuscript/)
(CVPR skeleton), and every numeric table is *generated* into
[`paper/tables/`](../../paper/tables/) by `python -m saryolo assets` — a `TBD`
cell there means the experiment has not been run, and no generator will fill it.

| Section | File | Fed by |
| --- | --- | --- |
| Abstract | [`abstract.md`](abstract.md) | written last; nothing but measured results |
| 1 Introduction | [`introduction.md`](introduction.md) | failure analysis (EXP-005/006) |
| 2 Related work | [`related_work.md`](related_work.md) | `docs/research_gap.md` (living) |
| 3 Method | [`method.md`](method.md) | `docs/METHOD.md`, `saryolo/nn/modules/` |
| 4 Experiments | [`experiments.md`](experiments.md) | protocol + implementation details (already fixed) |
| 5 Results & ablations | [`ablations.md`](ablations.md) | `paper/tables/*` (TBD until runs) |
| 6 Limitations | [`limitations.md`](limitations.md) | `reports/red_team_review.md` |

## Rules carried from the master workflow

1. No sentence in any section may state a number that is not in the ledger.
2. Negative results are reportable content, not failures to hide (Phase 56 / Rule 5).
3. The contribution claim list is rewritten from evidence at freeze time (Phase 58);
   until then the candidate contributions are labelled *candidates*.
4. Novelty statements defer to `docs/research_gap.md`; if a competing method
   overlaps, it is cited and differentiated, never omitted.
