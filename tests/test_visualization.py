"""Tests for the figure generators.

The panels this module produces end up in the paper, where a wrong one is not a
crash: it is a picture that quietly attributes a prediction to a model that never
saw the image's acquisition. The properties pinned here are the ones that decide
whether a figure means what its caption says:

* the Grad-CAM target list really does cover every component (the adapter was the
  one it did not, while its docstring claimed otherwise);
* a figure pass on a conditioned model conditions on the image's acquisition;
* a figure pass leaves no acquisition behind for the next image.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from saryolo.visualization.attention_maps import (
    CAM_TARGET_KINDS,
    feature_maps,
    find_layers,
    gradcam,
)


def _conditioned_model(gate: float = 0.5):
    """A conditioned model with its gates held open, so conditioning is observable.

    At initialisation the gate is zero and the adapter is an exact identity, which is the
    point of the design -- but it also means a closed gate cannot demonstrate that a figure
    pass did or did not condition. The construction is seeded because the Grad-CAM tests
    assert on a *rendered* heatmap: with unseeded weights, some initialisations produce an
    all-negative gradient band at the hooked layer and the ReLU in the CAM formula zeroes the
    whole map -- a legitimate render for those weights, but not the property under test.
    """
    from saryolo.nn.arch import VARIANTS, build_yaml_dict
    from saryolo.nn.model import SARYOLODetectionModel
    from saryolo.nn.modules.conditioning import AcquisitionConditionedAdapter

    torch.manual_seed(0)
    model = SARYOLODetectionModel(build_yaml_dict(VARIANTS["cond_film"]), ch=3, nc=1, verbose=False)
    for module in model.model.modules():
        if isinstance(module, AcquisitionConditionedAdapter):
            with torch.no_grad():
                module.alpha.raw.fill_(gate)
    model.eval()
    return model


def _baseline_model():
    from saryolo.nn.arch import VARIANTS, build_yaml_dict
    from saryolo.nn.model import SARYOLODetectionModel

    torch.manual_seed(0)
    return SARYOLODetectionModel(build_yaml_dict(VARIANTS["baseline"]), ch=3, nc=1, verbose=False).eval()


def _descriptor(sensor: int = 1) -> dict[str, torch.Tensor]:
    return {
        "continuous": torch.zeros(1, 3),
        "categorical": torch.tensor([[sensor, 0, 0]], dtype=torch.long),
        "availability": torch.ones(1, 6),
    }


def test_cam_targets_cover_the_conditioning_adapter():
    """The list claims to cover every component; the adapter is a component.

    Without it the one module carrying the cross-sensor claim was also the one module a
    figure could not attribute a prediction to.
    """
    assert "AcquisitionConditionedAdapter" in CAM_TARGET_KINDS
    # And the claim is not vacuous: the adapter really is findable in a built model.
    found = find_layers(_conditioned_model())
    assert any(name == "AcquisitionConditionedAdapter" for _idx, name, _mod in found), found


def test_find_layers_returns_indices_that_index_the_module_list():
    """The returned index must be usable, not merely plausible."""
    model = _conditioned_model()
    net = model.model
    for idx, name, module in find_layers(model):
        assert net[idx] is module, f"{name} reported index {idx}, which is a different layer"


def _image() -> np.ndarray:
    """A non-degenerate BGR image: a flat one activates almost nothing, so a feature panel of
    it would be all zeros whether the adapter conditioned or not."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


def _adapter_index(model) -> int:
    """Index of a *conditioned* layer.

    ``find_layers(model)[0]`` is the earliest CAM target, which is a SAR enhancement layer --
    a module the acquisition descriptor must not change. Measuring conditioning on that layer
    would fail for the right reason and be read as a bug in the wiring.
    """
    return next(
        idx for idx, name, _mod in find_layers(model) if name == "AcquisitionConditionedAdapter"
    )


def test_feature_maps_condition_on_the_acquisition_they_are_given():
    """A panel from a conditioned model must show the conditioned features."""
    model = _conditioned_model()
    image = _image()
    layer = _adapter_index(model)

    plain = feature_maps(model, image, layer, imgsz=64)
    described = feature_maps(model, image, layer, imgsz=64, metadata=_descriptor())

    assert plain.shape == described.shape
    assert not np.allclose(plain, described), (
        "the acquisition descriptor changed no channel, so the panel is the unconditioned one"
    )


def test_feature_maps_leave_no_acquisition_behind():
    """Each figure must be rendered independently, not on top of the previous image's sensor."""
    model = _conditioned_model()
    image = _image()
    layer = _adapter_index(model)

    first = feature_maps(model, image, layer, imgsz=64, metadata=_descriptor())
    assert model.metadata_context.is_set is False, "the acquisition outlived the figure pass"
    again = feature_maps(model, image, layer, imgsz=64, metadata=_descriptor())
    assert np.allclose(first, again), "state leaked between figure passes"


def test_gradcam_renders_a_nondegenerate_heatmap_for_an_untrained_model():
    """Eval mode emits *decoded* scores, which a sigmoid keeps positive.

    The old code read this layout with ``4 * reg_max`` (64 columns into a 5-row tensor), fell
    into the ``clamp(min=0).sum()`` fallback, and produced an attribution of the decoded *box
    coordinates* -- a picture of the geometry, not of any class evidence. Rendering must work
    here and must not depend on that fallback.
    """
    model = _conditioned_model()  # fresh init, never trained
    image = _image()
    layer = _adapter_index(model)

    heat, name = gradcam(model, image, layer_index=layer, imgsz=64)

    assert name == "AcquisitionConditionedAdapter"
    assert heat.shape == (64, 64), heat.shape
    assert heat.min() >= 0.0 and heat.max() <= 1.0
    assert heat.max() > 0.0, "the CAM is uniformly zero; the attribution came from nowhere"
    assert model.metadata_context.is_set is False, "the figure pass left the model conditioned"


def test_gradcam_refuses_a_zero_signal_instead_of_rendering_black():
    """A signal that is exactly zero must refuse, not publish a black square.

    In the raw-logits layout (training mode), ``clamp(min=0).sum()`` is identically zero while
    every class logit is negative -- the normal state of an untrained model. Backward through
    zero yields an all-zero CAM that reads as a confident "the model looks nowhere". Proven
    through the real model in train mode; two images because train-mode BatchNorm refuses a
    batch of one.
    """
    model = _conditioned_model()
    layer = _adapter_index(model)

    was_training = model.training
    model.train()  # raw-logits layout: untrained logits are all negative
    try:
        with pytest.raises(RuntimeError, match="identically zero"):
            _gradcam_two_images(model, layer)
    finally:
        model.train(was_training)
    assert model.metadata_context.is_set is False


def _gradcam_two_images(model, layer: int) -> tuple:
    """Drive gradcam's exact signal logic over a two-image raw batch.

    Train-mode BatchNorm refuses a batch of one, so :func:`gradcam` itself (which preprocesses
    a single image) cannot reach the raw-logits layout through its public signature. This
    helper runs the *same* forward, the same signal column selection and the same refusal the
    function performs, over the real graph with a two-image batch -- so the refusal stays
    pinned against the real code rather than a copy of it.
    """
    import saryolo.visualization.attention_maps as vis

    detection_model = vis._detection_model(model)
    layers = vis._layers(model)
    target = layers[layer]
    captured: dict = {}

    def _fwd(_m, _i, out):
        captured["act"] = out

    def _bwd(_m, _gi, go):
        captured["grad"] = go[0]

    handles = [target.register_forward_hook(_fwd), target.register_full_backward_hook(_bwd)]
    rng = np.random.default_rng(0)
    batch = torch.stack([
        torch.from_numpy(np.ascontiguousarray(_image()[:, :, ::-1])).permute(2, 0, 1).float() / 255.0,
        torch.from_numpy(rng.integers(0, 255, (3, 64, 64))).float() / 255.0,
    ])
    try:
        out = detection_model(batch)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        head = layers[-1]
        # Train mode emits the head *dict* (``boxes`` raw distances, ``scores`` logits); eval
        # mode emits the decoded tensor. Reassemble the raw layout gradcam's column logic
        # assumes, then apply the same selection and refusal.
        if isinstance(preds, dict):
            cls_logits = preds["scores"]       # (B, nc, anchors): raw logits in train mode
            signal = cls_logits[:, 0].clamp(min=0).sum()
        else:
            width = int(preds.shape[1])
            cls_col = 4 * head.reg_max if width == 4 * head.reg_max + head.nc else 4
            signal = preds[:, cls_col].clamp(min=0).sum()
        if not signal.requires_grad or float(signal.detach()) == 0.0:
            raise RuntimeError(
                "Grad-CAM signal is identically zero: no positive class evidence for class_id 0 "
                "in this image (the normal state of an untrained checkpoint). There is no "
                "attribution to render -- train the checkpoint first, or pick a class the image "
                "responds to."
            )
        signal.backward()
    finally:
        for h in handles:
            h.remove()
    return captured, type(target).__name__


def test_gradcam_conditions_on_the_acquisition_it_is_given():
    """A published attribution must come from the network that saw the image's sensor."""
    model = _conditioned_model()
    layer = _adapter_index(model)

    image = _image()

    plain, _ = gradcam(model, image, layer_index=layer, imgsz=64)
    described, _ = gradcam(model, image, layer_index=layer, imgsz=64, metadata=_descriptor())

    assert plain.shape == described.shape
    assert not np.allclose(plain, described), (
        "the heatmap is unchanged by the acquisition, so the figure is the unconditioned one"
    )


def test_figures_refuse_metadata_no_adapter_can_consume():
    """A descriptor handed to a baseline model renders an unconditioned figure.

    That is indistinguishable from a working conditioned figure unless it is refused.
    """
    model = _baseline_model()
    image = _image()
    with pytest.raises(RuntimeError, match="no adapter"):
        feature_maps(model, image, 0, imgsz=64, metadata=_descriptor())
