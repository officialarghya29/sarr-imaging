"""Paper asset generation: tables and figures built only from measured results."""

from __future__ import annotations

from .figures import (
    FIGURE_PLAN,
    plot_ablation_bars,
    plot_accuracy_efficiency,
    plot_cost_frontier,
    plot_robustness_curve,
    plot_scale_ap,
)
from .tables import (
    TBD,
    Table,
    build_ablation,
    build_baseline_comparison,
    build_efficiency,
    build_efficiency_frontier,
    build_module_ablation,
    build_multi_seed,
    build_removal_ablation,
    build_robustness,
    build_scale_analysis,
    write_tables,
)

__all__ = [
    "TBD",
    "Table",
    "build_baseline_comparison",
    "build_ablation",
    "build_module_ablation",
    "build_removal_ablation",
    "build_scale_analysis",
    "build_robustness",
    "build_efficiency",
    "build_efficiency_frontier",
    "build_multi_seed",
    "write_tables",
    "plot_robustness_curve",
    "plot_accuracy_efficiency",
    "plot_cost_frontier",
    "plot_ablation_bars",
    "plot_scale_ap",
    "FIGURE_PLAN",
]
