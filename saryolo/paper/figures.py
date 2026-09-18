"""Paper figure generation.

Only figures whose underlying data exists are produced; a missing input raises or
is skipped with an explicit message rather than being drawn with invented values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["plot_robustness_curve", "plot_accuracy_efficiency", "plot_ablation_bars", "plot_scale_ap", "FIGURE_PLAN"]

#: The figure plan from the project brief, with the data each figure needs.
FIGURE_PLAN: dict[str, str] = {
    "fig1_motivation": "Dataset statistics: speckle/clutter examples (dataset_statistics/*.png)",
    "fig2_architecture": "SAR-YOLO architecture diagram (drawn manually / TikZ)",
    "fig3_speckle_module": "SFM diagram (drawn manually)",
    "fig4_attention": "SAA diagram (drawn manually)",
    "fig5_fusion": "AMF diagram (drawn manually)",
    "fig6_small_object": "P2 head diagram (drawn manually)",
    "fig7_qualitative": "GT vs baseline vs SAR-YOLO panels (saryolo.visualization.detections)",
    "fig8_attention_maps": "Grad-CAM / feature maps (saryolo.visualization.attention_maps)",
    "fig9_robustness": "robustness.json from both models",
    "fig10_accuracy_efficiency": "ledger efficiency + mAP for both models",
    "fig11_failures": "failure taxonomy counts (saryolo.visualization.error_analysis)",
}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_robustness_curve(
    results: dict[str, str | Path],
    out_path: str | Path,
    metric: str = "mAP50",
    dpi: int = 200,
) -> Path | None:
    """FIGURE 9 — mAP against corruption severity, one line per model.

    Args:
        results: ``{model label: path to robustness.json}``.
        out_path: Where to write the figure.
        metric: Which metric to plot.
    """
    plt = _mpl()
    loaded = {}
    for label, path in results.items():
        path = Path(path)
        if path.exists():
            loaded[label] = json.loads(path.read_text())
    if not loaded:
        return None

    corruptions = sorted({c for data in loaded.values() for c in data.get("corruptions", {})})
    if not corruptions:
        return None

    fig, axes = plt.subplots(1, len(corruptions), figsize=(4 * len(corruptions), 3.4), squeeze=False)
    for idx, name in enumerate(corruptions):
        ax = axes[0][idx]
        for label, data in loaded.items():
            per_sev = data.get("corruptions", {}).get(name, {})
            if not per_sev:
                continue
            severities, values = [], []
            for severity, stats in sorted(per_sev.items(), key=lambda kv: float(kv[0])):
                value = stats.get(metric)
                if value is not None:
                    severities.append(float(severity))
                    values.append(float(value))
            if values:
                ax.plot(severities, values, marker="o", label=label)
        ax.set_title(name)
        ax.set_xlabel("severity")
        if idx == 0:
            ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_accuracy_efficiency(points: dict[str, dict], out_path: str | Path, dpi: int = 200) -> Path | None:
    """FIGURE 10 — mAP vs FPS, mAP vs params, mAP vs FLOPs.

    Args:
        points: ``{label: {"mAP50_95": float, "fps": float, "params_M": float, "flops_G": float}}``.
            Entries missing the required axes are skipped.
    """
    plt = _mpl()
    panels = (("fps", "FPS", "higher is better"), ("params_M", "parameters (M)", "lower is better"),
              ("flops_G", "GFLOPs", "lower is better"))
    plotted = False
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, xlabel, note) in zip(axes, panels):
        for label, values in points.items():
            if values.get("mAP50_95") is None or values.get(key) is None:
                continue
            ax.scatter(float(values[key]), float(values["mAP50_95"]), label=label, s=60)
            ax.annotate(label, (float(values[key]), float(values["mAP50_95"])),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
            plotted = True
        ax.set_xlabel(f"{xlabel} ({note})")
        ax.set_ylabel("mAP50:95")
        ax.grid(alpha=0.3)
    if not plotted:
        plt.close(fig)
        return None
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_ablation_bars(ablation: dict[str, float], out_path: str | Path, baseline_key: str | None = None, dpi: int = 200) -> Path | None:
    """FIGURE — ablation bar chart, optionally showing the delta against a baseline row."""
    plt = _mpl()
    labels = [k for k, v in ablation.items() if v is not None]
    values = [float(ablation[k]) for k in labels]
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4))
    bars = ax.bar(labels, values, color="#2c7fb8")
    if baseline_key and baseline_key in ablation and ablation[baseline_key] is not None:
        base = float(ablation[baseline_key])
        ax.axhline(base, color="#e6550d", linestyle="--", linewidth=1, label=f"baseline ({base:.4f})")
        for bar, value in zip(bars, values):
            ax.annotate(f"{value - base:+.4f}", (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax.legend()
    ax.set_ylabel("mAP50:95")
    ax.set_title("Ablation")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_scale_ap(models: dict[str, dict], out_path: str | Path, dpi: int = 200) -> Path | None:
    """FIGURE — grouped bars of AP_small / AP_medium / AP_large per model."""
    plt = _mpl()
    ranges = ("small", "medium", "large")
    labels = [m for m, v in models.items() if any(v.get(f"AP_{r}") is not None for r in ranges)]
    if not labels:
        return None

    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(ranges))
    for i, label in enumerate(labels):
        values = [models[label].get(f"AP_{r}") for r in ranges]
        heights = [float(v) if v is not None else 0.0 for v in values]
        bars = ax.bar(x + i * width, heights, width, label=label)
        for bar, value in zip(bars, values):
            if value is None:
                # An unmeasurable range is marked, not drawn as a zero-height bar
                # that a reader would interpret as a real score.
                ax.annotate("n/a", (bar.get_x() + bar.get_width() / 2, 0.01), ha="center", fontsize=8)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([f"AP_{r}" for r in ranges])
    ax.set_ylabel("AP")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
