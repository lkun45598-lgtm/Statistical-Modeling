from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .thermal import (
    alert_level_from_hotspot_dhw,
    hotspot,
    soft_dhw_from_sst,
    select_lead_indices,
)


def expand_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    while mask.dim() < ref.dim():
        mask = mask.unsqueeze(1)
    return mask.to(ref.device).float()


def masked_l1(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = expand_mask(mask, x)
    return (x.abs() * mask).sum() / (mask.sum() * x.shape[1] + eps)


def masked_mse(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = expand_mask(mask, x)
    return ((x ** 2) * mask).sum() / (mask.sum() * x.shape[1] + eps)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    dxp, dxt = pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    dyp, dyt = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    if mask is not None:
        mx = expand_mask(mask[..., :, 1:], dxp)
        my = expand_mask(mask[..., 1:, :], dyp)
        lx = ((dxp - dxt).abs() * mx).sum() / (mx.sum() * dxp.shape[1] + 1e-8)
        ly = ((dyp - dyt).abs() * my).sum() / (my.sum() * dyp.shape[1] + 1e-8)
        return lx + ly
    return F.l1_loss(dxp, dxt) + F.l1_loss(dyp, dyt)


def focal_loss_spatial(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for logits [B,T,K,H,W] and labels [B,T,H,W]."""
    B, T, K, H, W = logits.shape
    logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, K)
    labels_flat = labels.reshape(-1)
    mask_flat = mask.squeeze(1)
    while mask_flat.dim() < labels.dim():
        mask_flat = mask_flat.unsqueeze(1)
    mask_flat = mask_flat.expand_as(labels).reshape(-1).float()
    ce = F.cross_entropy(logits_flat, labels_flat, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce * mask_flat
    return loss.sum() / (mask_flat.sum() + 1e-8)


class ReefCastLoss:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.loss_cfg = cfg["loss"]
        self.forecast_cfg = cfg["forecast"]

    def __call__(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_ssta = outputs["pred_ssta"]
        risk_logits = outputs.get("risk_logits")
        true_ssta = batch["y_ssta"].to(pred_ssta.device)
        future_clim = batch["future_clim"].to(pred_ssta.device)
        past_sst = batch["past_sst"].to(pred_ssta.device)
        mmm = batch["mmm"].to(pred_ssta.device)
        ocean = batch["ocean_mask"].to(pred_ssta.device)
        reef = batch["reef_mask"].to(pred_ssta.device)
        buffer = batch["reef_buffer"].to(pred_ssta.device)

        reef_weight = float(self.loss_cfg.get("reef_weight", 4.0))
        buffer_weight = float(self.loss_cfg.get("reef_buffer_weight", 2.0))
        weight_mask = ocean * (1.0 + reef_weight * reef + buffer_weight * buffer)

        ssta_loss = masked_mse(pred_ssta - true_ssta, weight_mask)
        grad = gradient_loss(pred_ssta, true_ssta, ocean)

        pred_sst = pred_ssta + future_clim
        true_sst = true_ssta + future_clim
        pred_hs = hotspot(pred_sst, mmm)
        true_hs = hotspot(true_sst, mmm)
        hs_loss = masked_l1(pred_hs - true_hs, weight_mask)

        dhw_window = int(self.forecast_cfg.get("dhw_window_weeks", 12))
        threshold_c = float(self.forecast_cfg.get("bleaching_threshold_c", 1.0))
        gate_k = float(self.loss_cfg.get("dhw_soft_gate_k", 8.0))
        pred_dhw = soft_dhw_from_sst(pred_sst, past_sst, mmm, dhw_window, threshold_c, gate_k)
        true_dhw = soft_dhw_from_sst(true_sst, past_sst, mmm, dhw_window, threshold_c, gate_k)
        dhw_loss = masked_l1(pred_dhw - true_dhw, weight_mask)

        total = (
            float(self.loss_cfg.get("ssta_weight", 1.0)) * ssta_loss
            + float(self.loss_cfg.get("gradient_weight", 0.05)) * grad
            + float(self.loss_cfg.get("hotspot_weight", 0.2)) * hs_loss
            + float(self.loss_cfg.get("dhw_weight", 0.35)) * dhw_loss
        )
        log = {
            "loss_ssta": float(ssta_loss.detach().cpu()),
            "loss_grad": float(grad.detach().cpu()),
            "loss_hotspot": float(hs_loss.detach().cpu()),
            "loss_dhw": float(dhw_loss.detach().cpu()),
        }

        if risk_logits is not None and float(self.loss_cfg.get("alert_weight", 0.0)) > 0:
            labels = alert_level_from_hotspot_dhw(true_hs, true_dhw)
            alert_loss = focal_loss_spatial(
                risk_logits,
                labels,
                ocean,
                gamma=float(self.loss_cfg.get("focal_gamma", 2.0)),
            )
            total = total + float(self.loss_cfg.get("alert_weight", 0.2)) * alert_loss
            log["loss_alert"] = float(alert_loss.detach().cpu())

        log["loss_total"] = float(total.detach().cpu())
        return total, log
