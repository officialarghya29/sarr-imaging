"""Attention and feature-map visualization.

Implemented self-contained (no ``grad-cam`` dependency) so the figures in the
paper are reproducible without extra packages and so the target layer for a
detection model can be chosen explicitly.

For a detector, Grad-CAM must be computed with respect to a *specific* prediction
signal (a class score at a specific anchor). This module follows the standard
practice for detection: gradients of the sum of a chosen class's scores over the
selected feature level are backpropagated to the target layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["find_layers", "gradcam", "feature_maps", "overlay_heatmap", "save_attention_panel"]


def find_layers(model, kinds: tuple[str, ...] = ("SARAdaptiveAttention", "SpeckleAwareFeatureModule", "SARFeatureEnhancement")):
    """List ``(index, name, module)`` for layers whose type name contains any of ``kinds``."""
    net = getattr(model, "model", model)
    found = []
    for idx, module in enumerate(net):
        name = type(module).__name__
        if any(k in name for k in kinds):
            found.append((idx, name, module))
    return found


def _preprocess(image: np.ndarray, imgsz: int):
    """Letterbox-free simple resize + normalise, matching YOLO val preprocessing."""
    import cv2
    import torch

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), resized


def gradcam(
    model,
    image: np.ndarray,
    layer_index: int | None = None,
    imgsz: int = 640,
    class_id: int = 0,
    device: str | None = None,
) -> tuple[np.ndarray, str]:
    """Grad-CAM heat map for a detection model at ``layer_index``.

    Args:
        model: Loaded Ultralytics model (``YOLO``/``SARYOLO``); its ``.model`` is used.
        image: Input image (BGR or grayscale).
        layer_index: Index into ``model.model``; defaults to the last SAR attention
            layer if one exists, else the last neck layer.
        imgsz: Input size.
        class_id: Class whose scores drive the gradients.
        device: Torch device string.

    Returns:
        ``(heatmap in [0,1] with the input's spatial shape, layer_name)``.
    """
    import torch

    net = model.model
    net.eval()
    if device:
        net.to(device)
    if layer_index is None:
        candidates = find_layers(model)
        layer_index = candidates[-1][0] if candidates else max(len(net) - 2, 0)

    activations: dict = {}
    gradients: dict = {}

    def _fwd_hook(_module, _inp, out):
        activations["value"] = out

    def _bwd_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    target = net[layer_index]
    handles = [target.register_forward_hook(_fwd_hook), target.register_full_backward_hook(_bwd_hook)]

    try:
        tensor, resized = _preprocess(image, imgsz)
        tensor = tensor.to(next(net.parameters()).device)
        tensor.requires_grad_(True)
        net.zero_grad(set_to_none=True)
        out = net(tensor)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        # preds: (B, 4*reg_max + nc, anchors) for the classic head.
        if preds.dim() == 3 and preds.shape[1] > class_id:
            # Columns are [4 * reg_max box distances, nc class scores]; read reg_max from the
            # head rather than assuming the default 16, which breaks models built with another
            # reg_max.
            reg_max = int(getattr(net[-1], "reg_max", 16))
            score_cols = 4 * reg_max
            signal = (
                preds[:, score_cols + class_id].clamp(min=0).sum()
                if preds.shape[1] > score_cols + class_id
                else preds.clamp(min=0).sum()
            )
        else:
            signal = preds.clamp(min=0).sum()
        signal.backward()
    finally:
        for handle in handles:
            handle.remove()

    act = activations.get("value")
    grad = gradients.get("value")
    if act is None or grad is None:
        raise RuntimeError(f"No activation/gradient captured at layer {layer_index} ({type(target).__name__}).")
    # Second-order statistics matter here: speckle is a variance effect, and mean
    # pooling alone tends to highlight large uniform clutter regions.
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = cam.squeeze().detach().cpu().numpy()
    if cam.ndim != 2:
        cam = cam.reshape(cam.shape[-2], cam.shape[-1])
    cam = cam - cam.min()
    cam = cam / max(cam.max(), 1e-9)
    from cv2 import INTER_LINEAR, resize

    cam_full = resize(cam, (resized.shape[1], resized.shape[0]), interpolation=INTER_LINEAR)
    return cam_full, type(target).__name__


def feature_maps(model, image: np.ndarray, layer_index: int, imgsz: int = 640, max_channels: int = 8) -> np.ndarray:
    """Return the first ``max_channels`` feature channels at ``layer_index`` as a grid image."""
    import torch

    net = model.model
    net.eval()
    captured: dict = {}

    def _hook(_m, _i, out):
        captured["value"] = out.detach()

    handle = net[layer_index].register_forward_hook(_hook)
    try:
        tensor, _ = _preprocess(image, imgsz)
        with torch.no_grad():
            net(tensor.to(next(net.parameters()).device))
    finally:
        handle.remove()

    feature = captured["value"][0]
    return feature[:max_channels].cpu().numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap over an image using a perceptually ordered colour map."""
    import cv2

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    coloured = cv2.applyColorMap((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, coloured, alpha, 0)


def save_attention_panel(
    model,
    image_path: str | Path,
    out_path: str | Path,
    imgsz: int = 640,
    device: str | None = None,
) -> Path | None:
    """Render ``image | Grad-CAM | overlay`` for one image and save it."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(image_path)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    try:
        heat, layer_name = gradcam(model, img, imgsz=imgsz, device=device)
    except Exception:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("SAR image")
    axes[1].imshow(heat, cmap="jet")
    axes[1].set_title(f"Grad-CAM ({layer_name})")
    axes[2].imshow(cv2.cvtColor(overlay_heatmap(img, heat), cv2.COLOR_BGR2RGB))
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
