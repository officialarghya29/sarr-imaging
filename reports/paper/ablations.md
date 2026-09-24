# 5 Results & ablations — skeleton

**Every number comes from `paper/tables/` (generated). No table may be
hand-typed.** Until the runs exist, this section is a table plan.

| Table | Generator output | Claim it supports | Status |
| --- | --- | --- | --- |
| T2 baselines | `baseline_comparison` | the baseline is competitive and stock-equivalent | TBD |
| T5 main ablation (cumulative) | `main_ablation` | each component's marginal value | TBD |
| T6 module ablation (alternatives) | `module_ablation` | "adaptive" is earned, not asserted | TBD |
| T7 removal ablation | `removal_ablation` | no passenger components | TBD |
| T8 scale-wise | `scale_analysis` | small-object behaviour | TBD |
| T8b robustness | `robustness` | controlled-corruption degradation | TBD |
| T9 multi-seed | `multi_seed` | mean ± std, best/worst per seed | TBD |
| T10 efficiency | `efficiency` | overhead is <0.5% params | TBD |
| LOSO matrix | *(to add)* | **the headline claim** | generator pending first run |

**Writing rules:**
1. If the LOSO gain is within one seed-std, the sentence is "no detectable
   difference at this budget" — not a win (red-team W5).
2. If the metadata-withholding curve collapses at "none", that is a *reported
   limitation with the deployment implication*, not a hidden row (W2).
3. Figures: cross-source degradation plot, withholding curve, probe scatter
   (sensor vs class separability), failure-case strip — all generated, none
   cherry-picked (include representative failures per Phase 45).
