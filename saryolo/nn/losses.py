"""Component 7 — SAR-Aware Loss.

The SAR-aware objective is deliberately kept *auxiliary and switchable*. Each term
defaults to weight 0, in which case the total loss is numerically identical to
the stock ``v8DetectionLoss`` (the loss vector simply carries a fourth zero
entry). That is what makes the ``+ SAR Loss`` row of the ablation table a fair
measurement of this contribution alone rather than a change of training regime.

Terms
-----
``w_sep`` — Target/Background Separation (TBS).
    SAR scenes are dominated by background anchors. A plain BCE drive on those
    anchors can satisfy the loss by shrinking *all* foreground confidence, which
    costs recall on weak, low-contrast targets. TBS adds a margin-ranking
    objective between the mean foreground confidence and the *hardest* background
    anchors (top fraction by score), so the model is explicitly rewarded for
    separating target from clutter instead of for being uniformly unconfident.

``w_small`` — Small-object emphasis.
    Small targets occupy few anchors, so their gradient contribution is diluted
    across the batch. This term adds an extra confidence objective restricted to
    anchors assigned to small ground-truth boxes (COCO area threshold), which is
    the loss-level counterpart of the P2 detection head.

``w_smooth`` — Background speckle regularisation.
    Speckle produces isolated, high-variance background responses. This term
    penalises total variation of the *background-masked* mean score map at every
    detection level, encouraging the head to suppress scattered responses while
    leaving foreground activations untouched.

All three terms are computed from quantities the baseline criterion already
produces (``fg_mask``, ``target_bboxes``, ``target_gt_idx``, ``preds["scores"]``),
so they add no extra assignment pass and negligible cost.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

__all__ = [
    "SARAwareDetectionLoss",
    "SAR_LOSS_DEFAULTS",
    "build_criterion",
    "perturb_batch",
    "sar_consistency_loss",
]

#: Below this a feature vector is treated as silent rather than as having a direction.
_EPS = 1e-6

#: Default hyperparameters; every weight defaults to 0 == stock YOLO loss exactly.
SAR_LOSS_DEFAULTS: dict[str, float] = {
    "w_sep": 0.0,
    "w_small": 0.0,
    "w_smooth": 0.0,
    "margin": 1.0,
    "bg_frac": 0.25,
    "small_area": 32.0 ** 2,  # COCO "small" definition, in input pixels
    # SEC. 4 -- representation consistency under controlled SAR perturbation. Off by default,
    # and it costs a second forward pass when on: the weight is the switch, and there is no
    # cheaper way to compare two views of the same image than to run both.
    "w_consistency": 0.0,
    "consistency_kind": "speckle",
    "consistency_severity": 4.0,
}


def perturb_batch(
    img: torch.Tensor, kind: str = "speckle", severity: float = 4.0, seed: int = 0
) -> torch.Tensor:
    """Apply one SAR degradation to a whole image batch, on the same physics as everywhere else.

    Reuses :func:`saryolo.evaluation.robustness.apply_corruption` one image at a time, so the
    degradation a model is trained to be invariant to is the *same* degradation the robustness
    benchmark later measures it under. A second implementation would let those drift apart.

    Args:
        img: ``(B, C, H, W)`` float tensor in ``[0, 1]`` (the trainer's batch layout).
        kind: Corruption name from ``robustness.CORRUPTIONS``.
        severity: Its scalar parameter, which must come from the published grid.
        seed: Seed for the draw. The same batch and seed give a byte-identical perturbation,
            so a consistency run is reproducible.

    Returns:
        A perturbed tensor of the same shape, on the same device and dtype.
    """
    import numpy as np

    from saryolo.evaluation.robustness import CORRUPTIONS, apply_corruption

    if kind not in CORRUPTIONS:
        raise KeyError(f"Unknown corruption {kind!r}. Known: {sorted(CORRUPTIONS)}")
    published = CORRUPTIONS[kind].severities
    if severity not in published:
        raise ValueError(
            f"consistency_severity {severity} for {kind!r} is outside the published grid "
            f"{published}; training and evaluation must refer to the same degradation."
        )

    out = img.detach().clone()
    rng = np.random.default_rng(seed)
    was_half = out.dtype == torch.float16
    # The corruptions are numpy/uint8, so the batch round-trips through that representation.
    # That is exactly what the robustness benchmark does too, which is the point: the model is
    # trained under the same quantisation it is evaluated under.
    arr = (out.float().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            arr[b, c] = apply_corruption(arr[b, c], kind, severity, rng)
    result = torch.from_numpy(arr).to(device=img.device, dtype=torch.float32) / 255.0
    return result.to(torch.float16) if was_half else result


def sar_consistency_loss(
    feats_a: list[torch.Tensor], feats_b: list[torch.Tensor], mode: str = "cosine"
) -> torch.Tensor:
    """Representation drift between two views of the same image.

    A SAR detector should not change its mind about *what is where* because the speckle draw
    changed. This penalises the drift between the two views' feature maps, at the same spatial
    scales, so the network is pushed towards a representation that depends on scene structure
    rather than on the particular noise realisation.

    Args:
        feats_a: Feature maps from the clean pass, one per detection level.
        feats_b: The same, from the perturbed pass. Shapes must match pairwise.
        mode: ``"cosine"`` (default) or ``"mse"``. Cosine is the default because the drift
            that matters is *directional*: a uniform scale change is a calibration difference,
            whereas a rotated feature vector means different structure is being encoded.
            ``"mse"`` is exactly ``0.0`` on identical views; ``"cosine"`` is ``0.0`` to within
            float32 rounding (measured ~1.6e-8), because normalising a vector by its own norm
            does not land exactly on 1.0.

    Returns:
        A scalar.

    Dead positions are excluded, not penalised
    ------------------------------------------
    ``F.cosine_similarity`` returns **0** when a vector has zero norm, so a feature position
    whose channels are all zero would contribute ``1 - 0 = 1`` -- the largest possible value --
    and the term would spend its gradient pushing dead positions to come alive. BatchNorm with
    a negative bias and any post-activation sparsity make such positions routine, so the
    penalty would fight sparsity rather than measure representation drift. Positions where
    either view is (near-)silent are therefore masked out of the average. With identical views
    the mask is what makes the result 0 rather than 1 at those positions.
    """
    if len(feats_a) != len(feats_b):
        raise ValueError(f"view A has {len(feats_a)} levels but view B has {len(feats_b)}")
    if not feats_a:
        raise ValueError("no feature levels supplied; the consistency term would be vacuous")

    total = None
    for level, (fa, fb) in enumerate(zip(feats_a, feats_b, strict=True)):
        if fa.shape != fb.shape:
            raise ValueError(f"level {level}: view shapes differ, {tuple(fa.shape)} vs {tuple(fb.shape)}")
        if mode == "cosine":
            a = fa.flatten(2)
            b = fb.flatten(2)
            # Each spatial position is a vector over channels; compare directions.
            norm_a = a.norm(dim=1)
            norm_b = b.norm(dim=1)
            live = (norm_a > _EPS) & (norm_b > _EPS)
            if not bool(live.any()):
                # Every position is silent in at least one view: there is no direction to
                # compare, so the honest value is no penalty rather than a full one.
                term = fa.sum() * 0.0
            else:
                sim = F.cosine_similarity(a, b, dim=1)
                term = (1.0 - sim)[live].mean()
        elif mode == "mse":
            term = F.mse_loss(fa, fb)
        else:
            raise ValueError(f"mode must be 'cosine' or 'mse', got {mode!r}")
        total = term if total is None else total + term
    return total / len(feats_a)


class SARAwareDetectionLoss(v8DetectionLoss):
    """``v8DetectionLoss`` plus switchable SAR-specific auxiliary terms.

    Args:
        model: The detection model (must be de-paralleled), as for ``v8DetectionLoss``.
        tal_topk: Task-aligned assignment top-k.
        tal_topk2: Secondary top-k (ultralytics >= 8.3).
        **sar: Overrides for :data:`SAR_LOSS_DEFAULTS`.
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, **sar):
        super().__init__(model, tal_topk, tal_topk2)
        unknown = set(sar) - set(SAR_LOSS_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown SAR loss option(s): {sorted(unknown)}")
        cfg = {**SAR_LOSS_DEFAULTS, **sar}
        self.sar_cfg = cfg
        self.w_sep = float(cfg["w_sep"])
        self.w_small = float(cfg["w_small"])
        self.w_smooth = float(cfg["w_smooth"])
        self.margin = float(cfg["margin"])
        self.bg_frac = float(cfg["bg_frac"])
        self.small_area = float(cfg["small_area"])
        self.w_consistency = float(cfg["w_consistency"])
        # The perturbed view's feature maps are supplied by the model's `loss()` before this
        # criterion is called, because producing them needs a forward pass the criterion
        # cannot perform (it receives predictions, not the model). Held as plain attributes
        # rather than registered buffers: they are training-scoped activations, and putting
        # them in the state dict would bloat every checkpoint with one batch's features.
        self._view_a: list[torch.Tensor] | None = None
        self._view_b: list[torch.Tensor] | None = None
        # Appending a 4th name is safe: the parent zips names against its own 3-entry
        # loss vector, so the extra name is simply unused there.
        self.loss_names = (*self.loss_names, "sar_loss")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_fg_mask(fg_mask: torch.Tensor, n_anchors: int) -> torch.Tensor:
        """Normalise ``fg_mask`` to a ``(b, a)`` bool tensor across ultralytics versions."""
        fg = fg_mask
        if fg.dim() == 3:
            fg = fg.squeeze(-1)
        if fg.shape[1] != n_anchors:
            raise RuntimeError(f"fg_mask has {fg.shape[1]} anchors but predictions have {n_anchors}.")
        return fg.bool()

    def _separation(self, pred_scores_ba: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        """Margin ranking between mean foreground and hardest background confidence."""
        conf = pred_scores_ba.max(dim=-1).values  # (b, a)
        fg_conf = conf[fg]
        bg_conf = conf[~fg]
        if fg_conf.numel() == 0 or bg_conf.numel() == 0:
            return conf.sum() * 0.0  # keeps the graph connected without adding a term
        k = max(int(bg_conf.numel() * self.bg_frac), 1)
        hard_bg = bg_conf.topk(k).values.mean()
        return F.relu(self.margin - (fg_conf.mean() - hard_bg))

    def _small_object(self, pred_scores_ba: torch.Tensor, target_bboxes: torch.Tensor,
                      target_gt_idx: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        """Extra confidence objective on anchors assigned to small ground-truth boxes."""
        if target_bboxes.numel() == 0:
            return pred_scores_ba.sum() * 0.0
        wh = (target_bboxes[..., 2:] - target_bboxes[..., :2]).clamp_min(0.0)
        areas = wh[..., 0] * wh[..., 1]  # (b, n_max) in input pixels
        valid = areas > 0
        small_gt = (areas < self.small_area) & valid
        if not small_gt.any():
            return pred_scores_ba.sum() * 0.0
        n_max = target_bboxes.shape[1]
        idx = target_gt_idx.clamp(0, n_max - 1)
        per_anchor_small = small_gt.gather(1, idx) & fg
        if not per_anchor_small.any():
            return pred_scores_ba.sum() * 0.0
        conf = pred_scores_ba.max(dim=-1).values[per_anchor_small]
        return F.binary_cross_entropy_with_logits(conf, torch.ones_like(conf))

    def _background_smoothness(self, pred_scores_ba: torch.Tensor, fg: torch.Tensor,
                               feats: list[torch.Tensor]) -> torch.Tensor:
        """Total-variation penalty on background-masked score maps (suppresses speckle spikes)."""
        b = pred_scores_ba.shape[0]
        total = pred_scores_ba.sum() * 0.0
        offset, n_levels = 0, 0
        for feat in feats:
            h, w = feat.shape[2], feat.shape[3]
            n = h * w
            if offset + n > pred_scores_ba.shape[1]:
                break
            score_map = pred_scores_ba[:, offset:offset + n, :].mean(-1).view(b, 1, h, w)
            bg = (~fg[:, offset:offset + n]).view(b, 1, h, w).to(score_map.dtype)
            sm = score_map * bg  # regularise background only
            total = total + (sm[..., 1:, :] - sm[..., :-1, :]).abs().mean()
            total = total + (sm[..., :, 1:] - sm[..., :, :-1]).abs().mean()
            offset += n
            n_levels += 1
        return total / max(n_levels, 1)

    # --------------------------------------------------------------------- loss
    def set_views(self, feats_a: list[torch.Tensor], feats_b: list[torch.Tensor]) -> None:
        """Supply the clean and perturbed feature maps for the consistency term.

        Called by the model when ``w_consistency > 0``. Kept as an explicit setter so the
        criterion never has to guess whether a second view exists: when it is not set, the
        consistency term is skipped entirely rather than silently comparing a view to itself.
        """
        self._view_a, self._view_b = feats_a, feats_b

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        """Return ``(loss * batch_size, loss_items)`` with the SAR term appended."""
        batch_size = preds["boxes"].shape[0]
        (fg_mask, target_gt_idx, target_bboxes, _anchor_points, _stride), det_loss, detach = (
            self.get_assigned_targets_and_loss(preds, batch)
        )

        total = torch.zeros(4, device=self.device, dtype=det_loss.dtype)
        total[:3] = det_loss

        wants_consistency = self.w_consistency and self._view_a is not None and self._view_b is not None
        if self.w_sep or self.w_small or self.w_smooth or wants_consistency:
            pred_scores_ba = preds["scores"].permute(0, 2, 1).contiguous()  # (b, a, nc)
            fg = self._as_fg_mask(fg_mask, pred_scores_ba.shape[1])
            sar = total.new_zeros(())
            if self.w_sep:
                sar = sar + self.w_sep * self._separation(pred_scores_ba, fg)
            if self.w_small:
                sar = sar + self.w_small * self._small_object(pred_scores_ba, target_bboxes, target_gt_idx, fg)
            if self.w_smooth:
                sar = sar + self.w_smooth * self._background_smoothness(pred_scores_ba, fg, preds["feats"])
            if wants_consistency:
                sar = sar + self.w_consistency * sar_consistency_loss(self._view_a, self._view_b)
            total[3] = sar
            detach = {**detach, "sar_loss": sar.detach()}
        else:
            detach = {**detach, "sar_loss": total[3].detach()}

        return total * batch_size, detach


def build_criterion(model, sar_cfg: dict | None = None):
    """Build the detection criterion for ``model``, honouring its YAML ``sar_loss`` block."""
    sar_cfg = dict(sar_cfg or {})
    active = any(
        float(sar_cfg.get(k, 0.0))
        for k in ("w_sep", "w_small", "w_smooth", "w_consistency")
    )
    if getattr(model.model[-1], "one2one_cv2", None) is not None:
        # End-to-end (YOLO26-style) heads use E2ELoss; our auxiliary terms are defined
        # against the classic head's prediction dict.
        if active:
            raise NotImplementedError(
                "The SAR-aware loss requires a classic (non end-to-end) Detect head. "
                "Set `end2end: False` in the model YAML or disable the SAR loss weights."
            )
        return E2ELoss(model)
    return SARAwareDetectionLoss(model, **sar_cfg)
