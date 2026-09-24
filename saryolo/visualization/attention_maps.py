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

__all__ = ["find_layers", "gradcam", "feature_maps", "overlay_heatmap", "save_attention_panel",
           "CAM_TARGET_KINDS"]


def _detection_model(model):
    """The ``DetectionModel`` a figure is rendered from, from either handle a caller may pass.

    ``model.model`` means two different things depending on what the caller holds, and the
    metadata context lives on only one of them:

    * an Ultralytics facade (``YOLO``/``SARYOLO``) -- ``.model`` *is* the ``DetectionModel``;
    * a ``DetectionModel`` itself -- ``.model`` is its inner ``nn.Sequential``, which owns no
      metadata context, so conditioning a figure through it silently does nothing.

    Resolving explicitly keeps both call styles working, which matters because the probe command
    and the tests hold the model directly while these helpers document the facade.
    """
    from ultralytics.nn.tasks import DetectionModel

    if isinstance(model, DetectionModel):
        return model
    inner = getattr(model, "model", None)
    return inner if isinstance(inner, DetectionModel) else model


def _layers(model):
    """The indexable layer list a figure hooks into.

    This is ``DetectionModel.model`` (an ``nn.Sequential``) and *not* the ``DetectionModel``:
    the model object itself is not indexable in the installed Ultralytics version, so a figure
    that indexed the model rather than its layer list raised ``TypeError`` from ``enumerate``
    before ever rendering anything.
    """
    net = _detection_model(model)
    layers = getattr(net, "model", None)
    return layers if layers is not None else net


#: Every SAR-YOLO module is a valid Grad-CAM target. The list covers all components so a
#: figure can attribute a prediction to the target prior (8), the spectral branch (9),
#: context (10) or the deformable refinement (11), not only to the v1 modules.
#:
#: ``AcquisitionConditionedAdapter`` (Component 33) is included so the cross-sensor story can
#: also be *shown*: without it the module that carries the paper's headline claim was the one
#: component a figure could not attribute a prediction to.
CAM_TARGET_KINDS: tuple[str, ...] = (
    "SARAdaptiveAttention",
    "SpeckleAwareFeatureModule",
    "SARFeatureEnhancement",
    "TargetPriorModulation",
    "SpatialFrequencyRepresentation",
    "ContextAggregation",
    "TargetAwareRefinement",
    "AcquisitionConditionedAdapter",
)


def find_layers(model, kinds: tuple[str, ...] = CAM_TARGET_KINDS):
    """List ``(index, name, module)`` for layers whose type name contains any of ``kinds``."""
    found = []
    for idx, module in enumerate(_layers(model)):
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
    metadata: dict | None = None,
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
        metadata: optional encoded acquisition descriptor for *this* image, in the
            ``(continuous, categorical, availability)`` form with a batch dimension of
            one. A conditioned model rendered without it runs its "unknown
            acquisition" branch, so the published figure would attribute the prediction
            to a network that never saw the image's sensor.

    Returns:
        ``(heatmap in [0,1] with the input's spatial shape, layer_name)``.
    """
    import torch

    from saryolo.nn.model import set_batch_metadata

    detection_model = _detection_model(model)
    layers = _layers(model)
    detection_model.eval()
    if device:
        detection_model.to(device)
    if layer_index is None:
        candidates = find_layers(model)
        layer_index = candidates[-1][0] if candidates else max(len(layers) - 2, 0)
    # Refused rather than ignored -- the same contract as the representation probe. Handing a
    # descriptor to a model that cannot consume it renders an attribution that ignores the
    # acquisition while the figure implies otherwise, and nothing would say so.
    if metadata is not None and not set_batch_metadata(detection_model, metadata):
        raise RuntimeError(
            "acquisition metadata was supplied but no adapter in this model consumes it; "
            "the attribution would be rendered unconditioned"
        )

    activations: dict = {}
    gradients: dict = {}

    def _fwd_hook(_module, _inp, out):
        activations["value"] = out

    def _bwd_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    target = layers[layer_index]
    handles = [target.register_forward_hook(_fwd_hook), target.register_full_backward_hook(_bwd_hook)]

    try:
        tensor, resized = _preprocess(image, imgsz)
        tensor = tensor.to(next(detection_model.parameters()).device)
        tensor.requires_grad_(True)
        detection_model.zero_grad(set_to_none=True)
        # Forward through the *model*, not through its layer list: the graph has layers that
        # take several inputs (``Concat``) and read via negative layer indices, so a plain
        # ``nn.Sequential`` forward over the list cannot drive it -- it raises from the first
        # ``Concat``. ``DetectionModel.forward`` walks the graph and returns the head output.
        out = detection_model(tensor)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        # Column layout depends on the head's mode, and they disagree. Training emits the raw
        # head tensor ``(B, 4*reg_max + nc, anchors)``; eval emits the *decoded* tensor
        # ``(B, 4 + nc, anchors)`` whose class columns already passed a sigmoid. Reading eval
        # output with ``4 * reg_max`` (64) therefore indexed 64 rows into a 5-row tensor, and
        # the old code silently fell into the ``clamp(min=0).sum()`` fallback -- which summed
        # the *decoded box coordinates*, so the "attribution" was a picture of the boxes, not
        # of any class evidence. Distinguish the two layouts by the width the head declares.
        head = layers[-1]
        nc = int(getattr(head, "nc", 1))
        reg_max = int(getattr(head, "reg_max", 16))
        raw_width = 4 * reg_max + nc
        if preds.dim() == 3 and preds.shape[1] == raw_width:
            signal = preds[:, 4 * reg_max + class_id].clamp(min=0).sum()  # raw logits, untrained = negative
        elif preds.dim() == 3 and preds.shape[1] == 4 + nc:
            signal = preds[:, 4 + class_id].sum()  # decoded scores: sigmoid applied, never negative
        else:
            signal = preds.clamp(min=0).sum()
        # In the raw-logits layout, ``clamp(min=0)`` zeroes the signal whenever every logit is
        # negative -- the normal state of an untrained model. Backward through an exactly zero
        # signal contributes no gradient, the CAM weights come out all zero, and the published
        # figure is a uniform black square that reads as a confident "the model looks nowhere".
        # A blank attribution is a finding, not a picture, so refuse.
        if not signal.requires_grad or float(signal.detach()) == 0.0:
            raise RuntimeError(
                "Grad-CAM signal is identically zero: no positive class evidence for class_id "
                f"{class_id} in this image (the normal state of an untrained checkpoint). There "
                "is no attribution to render -- train the checkpoint first, or pick a class the "
                "image responds to."
            )
        signal.backward()
    finally:
        for handle in handles:
            handle.remove()
        if metadata is not None:
            # A figure pass must not leave the model conditioned on one image.
            set_batch_metadata(detection_model, None)

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


def feature_maps(
    model,
    image: np.ndarray,
    layer_index: int,
    imgsz: int = 640,
    max_channels: int = 8,
    metadata: dict | None = None,
) -> np.ndarray:
    """Return the first ``max_channels`` feature channels at ``layer_index`` as a grid image.

    ``metadata`` carries the image's acquisition descriptor for a conditioned model, for the
    same reason :func:`gradcam` takes one: without it the panel shows the model's
    unknown-acquisition features rather than the ones its prediction came from.
    """
    import torch

    from saryolo.nn.model import set_batch_metadata

    detection_model = _detection_model(model)
    layers = _layers(model)
    detection_model.eval()
    if metadata is not None and not set_batch_metadata(detection_model, metadata):
        raise RuntimeError(
            "acquisition metadata was supplied but no adapter in this model consumes it; "
            "the panel would show unconditioned features"
        )
    captured: dict = {}

    def _hook(_m, _i, out):
        captured["value"] = out.detach()

    handle = layers[layer_index].register_forward_hook(_hook)
    try:
        tensor, _ = _preprocess(image, imgsz)
        with torch.no_grad():
            # Through the model, for the same reason as :func:`gradcam`: the layer list alone
            # cannot be driven as a sequential stack.
            detection_model(tensor.to(next(detection_model.parameters()).device))
    finally:
        handle.remove()
        if metadata is not None:
            set_batch_metadata(detection_model, None)

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
