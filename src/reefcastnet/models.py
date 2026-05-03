from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_conv(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Conv2d:
    pad = kernel_size // 2
    return nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, norm: bool = True):
        super().__init__()
        self.conv = make_conv(in_ch, out_ch, kernel_size)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch) if norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualConvBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(ch, ch, kernel_size),
            make_conv(ch, ch, kernel_size),
            nn.GroupNorm(num_groups=min(8, ch), num_channels=ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class GatedSpatiotemporalBlock(nn.Module):
    """Compact SimVPv2-style gated spatiotemporal attention block.

    This is a clean-room, lightweight implementation inspired by the SimVPv2
    design principle: recurrent-free convolutional temporal mixing with gating.
    It is not a copy of the official OpenSTL code.
    """
    def __init__(self, ch: int, kernel_size: int = 5, expansion: int = 2):
        super().__init__()
        hidden = ch * expansion
        pad = kernel_size // 2
        self.norm = nn.GroupNorm(num_groups=min(16, ch), num_channels=ch)
        self.proj_in = nn.Conv2d(ch, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size, padding=pad, groups=hidden)
        self.gate_conv = nn.Conv2d(hidden, hidden, kernel_size, padding=pad, groups=hidden)
        self.proj_out = nn.Conv2d(hidden, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        a, g = self.proj_in(x).chunk(2, dim=1)
        a = self.dwconv(a)
        g = torch.sigmoid(self.gate_conv(g))
        out = self.proj_out(F.gelu(a) * g)
        return residual + out


class ReefCastNetSimVP(nn.Module):
    """ReefCastNet-SimVP.

    A reef-aware SimVPv2-inspired backbone for weekly SST anomaly forecasting.
    Input:
        x_ssta: [B,Tin,1,H,W]
        static: [B,Cs,H,W]
        time_feats: [B,Tin,2]
    Output:
        pred_ssta: [B,Tout,1,H,W]
        risk_logits: [B,Tout,K,H,W]
    """
    def __init__(
        self,
        input_len: int,
        output_len: int,
        static_channels: int,
        hidden_dim: int = 48,
        encoder_depth: int = 2,
        temporal_depth: int = 4,
        decoder_depth: int = 2,
        kernel_size: int = 5,
        num_alert_classes: int = 5,
        use_alert_head: bool = True,
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.use_alert_head = use_alert_head

        frame_in_ch = 1 + static_channels + 2  # SSTA + static maps + week sin/cos maps
        enc = [ConvBlock(frame_in_ch, hidden_dim, kernel_size)]
        for _ in range(max(encoder_depth - 1, 0)):
            enc.append(ResidualConvBlock(hidden_dim, kernel_size))
        self.encoder = nn.Sequential(*enc)

        mix_ch = input_len * hidden_dim
        self.temporal = nn.Sequential(*[
            GatedSpatiotemporalBlock(mix_ch, kernel_size=kernel_size)
            for _ in range(temporal_depth)
        ])
        self.to_future = nn.Conv2d(mix_ch, output_len * hidden_dim, kernel_size=1)

        dec = []
        for _ in range(max(decoder_depth - 1, 0)):
            dec.append(ResidualConvBlock(hidden_dim, kernel_size))
        dec.append(make_conv(hidden_dim, 1, kernel_size=3))
        self.decoder = nn.Sequential(*dec)

        if use_alert_head:
            self.risk_head = nn.Sequential(
                ConvBlock(hidden_dim, hidden_dim, kernel_size=3),
                nn.Conv2d(hidden_dim, num_alert_classes, kernel_size=1),
            )
        else:
            self.risk_head = None

    def _make_frame_inputs(self, x_ssta: torch.Tensor, static: torch.Tensor, time_feats: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x_ssta.shape
        static_rep = static[:, None].expand(B, T, static.shape[1], H, W)
        tf = time_feats[:, :, :, None, None].expand(B, T, 2, H, W)
        x = torch.cat([x_ssta, static_rep, tf], dim=2)
        return x.reshape(B * T, x.shape[2], H, W)

    def forward(
        self,
        x_ssta: torch.Tensor,
        static: torch.Tensor,
        time_feats: torch.Tensor,
        future_time_feats: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _, H, W = x_ssta.shape
        x = self._make_frame_inputs(x_ssta, static, time_feats)
        z = self.encoder(x).reshape(B, T, self.hidden_dim, H, W)
        z = z.reshape(B, T * self.hidden_dim, H, W)
        z = self.temporal(z)
        z = self.to_future(z).reshape(B, self.output_len, self.hidden_dim, H, W)

        pred = self.decoder(z.reshape(B * self.output_len, self.hidden_dim, H, W))
        pred = pred.reshape(B, self.output_len, 1, H, W)

        out: Dict[str, torch.Tensor] = {"pred_ssta": pred}
        if self.risk_head is not None:
            logits = self.risk_head(z.reshape(B * self.output_len, self.hidden_dim, H, W))
            K = logits.shape[1]
            logits = logits.reshape(B, self.output_len, K, H, W)
            out["risk_logits"] = logits
        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = make_conv(in_ch + hidden_ch, 4 * hidden_ch, kernel_size)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMForecast(nn.Module):
    """Simple ConvLSTM autoregressive baseline."""
    def __init__(self, input_len: int, output_len: int, static_channels: int, hidden_dim: int = 48, kernel_size: int = 3):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.frame_in_ch = 1 + static_channels + 2
        self.cell = ConvLSTMCell(self.frame_in_ch, hidden_dim, kernel_size)
        self.out = nn.Conv2d(hidden_dim, 1, 1)
        self.risk_head = None

    def _frame(self, x: torch.Tensor, static: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        tfm = tf[:, :, None, None].expand(B, 2, H, W)
        return torch.cat([x, static, tfm], dim=1)

    def forward(self, x_ssta: torch.Tensor, static: torch.Tensor, time_feats: torch.Tensor, future_time_feats: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        B, T, _, H, W = x_ssta.shape
        h = torch.zeros(B, self.cell.hidden_ch, H, W, device=x_ssta.device, dtype=x_ssta.dtype)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(self._frame(x_ssta[:, t], static, time_feats[:, t]), (h, c))
        preds = []
        cur = x_ssta[:, -1]
        if future_time_feats is None:
            future_time_feats = time_feats[:, -1:].expand(B, self.output_len, 2)
        for t in range(self.output_len):
            h, c = self.cell(self._frame(cur, static, future_time_feats[:, t]), (h, c))
            cur = self.out(h)
            preds.append(cur[:, None])
        return {"pred_ssta": torch.cat(preds, dim=1)}


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    name = cfg["model"].get("name", "reefcastnet_simvp").lower()
    common = dict(
        input_len=int(cfg["forecast"]["input_len"]),
        output_len=int(cfg["forecast"]["output_len"]),
        static_channels=len(cfg["model"].get("static_channels", [])),
        hidden_dim=int(cfg["model"].get("hidden_dim", 48)),
        kernel_size=int(cfg["model"].get("kernel_size", 5)),
    )
    if name in {"reefcastnet_simvp", "simvp", "simvpv2"}:
        return ReefCastNetSimVP(
            **common,
            encoder_depth=int(cfg["model"].get("encoder_depth", 2)),
            temporal_depth=int(cfg["model"].get("temporal_depth", 4)),
            decoder_depth=int(cfg["model"].get("decoder_depth", 2)),
            num_alert_classes=int(cfg["model"].get("num_alert_classes", 5)),
            use_alert_head=bool(cfg["model"].get("use_alert_head", True)),
        )
    if name in {"convlstm", "conv_lstm"}:
        return ConvLSTMForecast(**common)
    raise ValueError(f"Unknown model name: {name}")
