"""Feature-map visualization: what the SAR modules change about the representation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .attention_maps import feature_maps

__all__ = ["feature_maps", "save_feature_grid", "compare_feature_maps"]


def save_feature_grid(maps: np.ndarray, out_path: str | Path, title: str = "", max_channels: int = 8, dpi: int = 200) -> Path:
    """Save a grid of per-channel feature maps (activations normalised individually)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(maps), max_channels)
    cols = min(4, n)
    rows = int(np.ceil(n / max(cols, 1)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < n:
            m = maps[i]
            m = (m - m.min()) / max(m.max() - m.min(), 1e-9)
            ax.imshow(m, cmap="viridis")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def compare_feature_maps(
    models: dict,
    image_path: str | Path,
    layer_index: int,
    out_path: str | Path,
    imgsz: int = 640,
    max_channels: int = 4,
) -> Path | None:
    """Side-by-side feature maps for several models at the same layer index.

    Used to show, for example, that the baseline's activations on a cluttered
    region are diffuse while SAR-YOLO's are concentrated — the qualitative
    counterpart of the ablation table.
    """
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(image_path)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    panels = []
    for label, model in models.items():
        try:
            maps = feature_maps(model, img, layer_index, imgsz=imgsz, max_channels=max_channels)
        except Exception:
            continue
        panels.append((label, maps))
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), max_channels + 1, figsize=(3 * (max_channels + 1), 3 * len(panels)), squeeze=False)
    for r, (label, maps) in enumerate(panels):
        axes[r][0].imshow(img, cmap="gray")
        axes[r][0].set_title(label)
        axes[r][0].axis("off")
        for c in range(max_channels):
            ax = axes[r][c + 1]
            ax.axis("off")
            if c < len(maps):
                m = maps[c]
                m = (m - m.min()) / max(m.max() - m.min(), 1e-9)
                ax.imshow(m, cmap="viridis")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
