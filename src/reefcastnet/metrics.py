from __future__ import annotations

from typing import Dict, Iterable

import torch

from .thermal import hard_dhw_from_sst, hotspot, alert_level_from_hotspot_dhw, select_lead_indices


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    mask = mask.to(x.device).float()
    return (x * mask).sum() / (mask.sum() * x.shape[1] + eps)


@torch.no_grad()
def sst_metrics(
    pred_ssta: torch.Tensor,
    true_ssta: torch.Tensor,
    ocean_mask: torch.Tensor,
    lead_weeks: list[int],
) -> Dict[str, float]:
    inds = select_lead_indices(true_ssta.shape[1], lead_weeks)
    out: Dict[str, float] = {}
    for lead, idx in zip(lead_weeks, inds):
        p = pred_ssta[:, idx:idx+1]
        y = true_ssta[:, idx:idx+1]
        diff = p - y
        mse = masked_mean(diff ** 2, ocean_mask)
        mae = masked_mean(diff.abs(), ocean_mask)
        out[f"rmse_w{lead}"] = float(torch.sqrt(mse).cpu())
        out[f"mae_w{lead}"] = float(mae.cpu())
    return out


@torch.no_grad()
def thermal_metrics(
    pred_ssta: torch.Tensor,
    true_ssta: torch.Tensor,
    future_clim: torch.Tensor,
    past_sst: torch.Tensor,
    mmm: torch.Tensor,
    ocean_mask: torch.Tensor,
    lead_weeks: list[int],
    window_weeks: int = 12,
    threshold_c: float = 1.0,
) -> Dict[str, float]:
    pred_sst = pred_ssta + future_clim
    true_sst = true_ssta + future_clim
    pred_hs = hotspot(pred_sst, mmm)
    true_hs = hotspot(true_sst, mmm)
    pred_dhw = hard_dhw_from_sst(pred_sst, past_sst, mmm, window_weeks, threshold_c)
    true_dhw = hard_dhw_from_sst(true_sst, past_sst, mmm, window_weeks, threshold_c)
    pred_alert = alert_level_from_hotspot_dhw(pred_hs, pred_dhw)
    true_alert = alert_level_from_hotspot_dhw(true_hs, true_dhw)
    inds = select_lead_indices(true_ssta.shape[1], lead_weeks)
    out: Dict[str, float] = {}
    mask = ocean_mask.to(pred_ssta.device).bool().squeeze(1)
    for lead, idx in zip(lead_weeks, inds):
        dhw_mae = masked_mean((pred_dhw[:, idx:idx+1] - true_dhw[:, idx:idx+1]).abs(), ocean_mask)
        out[f"dhw_mae_w{lead}"] = float(dhw_mae.cpu())
        pa = pred_alert[:, idx]
        ta = true_alert[:, idx]
        valid = mask
        if valid.dim() == 2:
            valid = valid[None].expand_as(pa)
        acc = ((pa == ta) & valid).sum().float() / (valid.sum().float() + 1e-8)
        out[f"alert_acc_w{lead}"] = float(acc.cpu())

        # Event skill for Alert Level 1+ and Alert Level 2+.
        for cls, name in [(3, "alert1p"), (4, "alert2p")]:
            p_event = (pa >= cls) & valid
            t_event = (ta >= cls) & valid
            tp = (p_event & t_event).sum().float()
            fp = (p_event & (~t_event)).sum().float()
            fn = ((~p_event) & t_event).sum().float()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            out[f"{name}_precision_w{lead}"] = float(precision.cpu())
            out[f"{name}_recall_w{lead}"] = float(recall.cpu())
            out[f"{name}_f1_w{lead}"] = float(f1.cpu())
    return out


def merge_metric_dicts(dicts: list[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: float(sum(d[k] for d in dicts) / len(dicts)) for k in keys}
