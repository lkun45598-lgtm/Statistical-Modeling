from __future__ import annotations

import torch
import torch.nn.functional as F


def hotspot(sst: torch.Tensor, mmm: torch.Tensor) -> torch.Tensor:
    """NOAA-style HotSpot magnitude: max(SST - MMM, 0)."""
    while mmm.dim() < sst.dim():
        mmm = mmm.unsqueeze(1)
    return torch.relu(sst - mmm)


def soft_dhw_from_sst(
    sst_future: torch.Tensor,
    past_sst: torch.Tensor,
    mmm: torch.Tensor,
    window_weeks: int = 12,
    threshold_c: float = 1.0,
    gate_k: float = 8.0,
) -> torch.Tensor:
    """Differentiable weekly approximation to Degree Heating Weeks.

    Args:
        sst_future: [B,T,1,H,W] predicted or true future SST in Celsius.
        past_sst: [B,Tin,1,H,W] past true SST, used to initialize the running 12-week window.
        mmm: [B,1,H,W] Maximum Monthly Mean.
    Returns:
        dhw: [B,T,1,H,W] weekly DHW approximation.
    """
    sst_all = torch.cat([past_sst, sst_future], dim=1)
    hs = hotspot(sst_all, mmm)
    gate = torch.sigmoid(gate_k * (hs - threshold_c))
    contrib = hs * gate
    outs = []
    tin = past_sst.shape[1]
    for t in range(sst_future.shape[1]):
        end = tin + t + 1
        start = max(0, end - window_weeks)
        outs.append(contrib[:, start:end].sum(dim=1, keepdim=True))
    return torch.cat(outs, dim=1)


def hard_dhw_from_sst(
    sst_future: torch.Tensor,
    past_sst: torch.Tensor,
    mmm: torch.Tensor,
    window_weeks: int = 12,
    threshold_c: float = 1.0,
) -> torch.Tensor:
    sst_all = torch.cat([past_sst, sst_future], dim=1)
    hs = hotspot(sst_all, mmm)
    contrib = torch.where(hs >= threshold_c, hs, torch.zeros_like(hs))
    outs = []
    tin = past_sst.shape[1]
    for t in range(sst_future.shape[1]):
        end = tin + t + 1
        start = max(0, end - window_weeks)
        outs.append(contrib[:, start:end].sum(dim=1, keepdim=True))
    return torch.cat(outs, dim=1)


def alert_level_from_hotspot_dhw(hs: torch.Tensor, dhw: torch.Tensor) -> torch.Tensor:
    """Five-class alert labels.

    0: No Stress, 1: Watch, 2: Warning, 3: Alert Level 1, 4: Alert Level 2+
    This merges high CRW Alert Levels into class 4 for class-balance stability.
    """
    out = torch.zeros_like(hs, dtype=torch.long)
    out = torch.where((hs > 0.0) & (hs < 1.0), torch.ones_like(out), out)
    out = torch.where((hs >= 1.0) & (dhw < 4.0), torch.full_like(out, 2), out)
    out = torch.where((hs >= 1.0) & (dhw >= 4.0) & (dhw < 8.0), torch.full_like(out, 3), out)
    out = torch.where((hs >= 1.0) & (dhw >= 8.0), torch.full_like(out, 4), out)
    return out.squeeze(2)  # [B,T,H,W]


def select_lead_indices(output_len: int, lead_weeks: list[int]) -> list[int]:
    return [min(max(int(w) - 1, 0), output_len - 1) for w in lead_weeks]
