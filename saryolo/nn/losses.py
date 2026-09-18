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

__all__ = ["SARAwareDetectionLoss", "SAR_LOSS_DEFAULTS", "build_criterion"]

#: Default hyperparameters; every weight defaults to 0 == stock YOLO loss exactly.
SAR_LOSS_DEFAULTS: dict[str, float] = {
    "w_sep": 0.0,
    "w_small": 0.0,
    "w_smooth": 0.0,
    "margin": 1.0,
    "bg_frac": 0.25,
    "small_area": 32.0 ** 2,  # COCO "small" definition, in input pixels
}


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
    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        """Return ``(loss * batch_size, loss_items)`` with the SAR term appended."""
        batch_size = preds["boxes"].shape[0]
        (fg_mask, target_gt_idx, target_bboxes, _anchor_points, _stride), det_loss, detach = (
            self.get_assigned_targets_and_loss(preds, batch)
        )

        total = torch.zeros(4, device=self.device, dtype=det_loss.dtype)
        total[:3] = det_loss

        if self.w_sep or self.w_small or self.w_smooth:
            pred_scores_ba = preds["scores"].permute(0, 2, 1).contiguous()  # (b, a, nc)
            fg = self._as_fg_mask(fg_mask, pred_scores_ba.shape[1])
            sar = total.new_zeros(())
            if self.w_sep:
                sar = sar + self.w_sep * self._separation(pred_scores_ba, fg)
            if self.w_small:
                sar = sar + self.w_small * self._small_object(pred_scores_ba, target_bboxes, target_gt_idx, fg)
            if self.w_smooth:
                sar = sar + self.w_smooth * self._background_smoothness(pred_scores_ba, fg, preds["feats"])
            total[3] = sar
            detach = {**detach, "sar_loss": sar.detach()}
        else:
            detach = {**detach, "sar_loss": total[3].detach()}

        return total * batch_size, detach


def build_criterion(model, sar_cfg: dict | None = None):
    """Build the detection criterion for ``model``, honouring its YAML ``sar_loss`` block."""
    sar_cfg = dict(sar_cfg or {})
    active = any(float(sar_cfg.get(k, 0.0)) for k in ("w_sep", "w_small", "w_smooth"))
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
