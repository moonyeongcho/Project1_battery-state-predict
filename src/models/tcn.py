# src/models/tcn.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm as pweight_norm

class SafeChomp1d(nn.Module):
    """Causal TCN용 패딩 제거 (pad=0이면 그대로 반환)."""
    def __init__(self, c: int):
        super().__init__()
        self.c = int(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c <= 0:
            return x.contiguous()
        return x[:, :, :-self.c].contiguous()

class TBlock(nn.Module):
    """
    Temporal Block (TCN 기본 블록)
    - causal dilated Conv1d x2 + ReLU + Dropout + skip connection
    - weight_norm 적용
    """
    def __init__(self, cin: int, cout: int, k: int, d: int, drop: float = 0.1):
        super().__init__()
        pad = (k - 1) * d
        conv1 = pweight_norm(nn.Conv1d(cin,  cout, k, padding=pad, dilation=d))
        conv2 = pweight_norm(nn.Conv1d(cout, cout, k, padding=pad, dilation=d))
        self.net = nn.Sequential(
            conv1, SafeChomp1d(pad), nn.ReLU(),
            nn.Dropout(drop),
            conv2, SafeChomp1d(pad), nn.ReLU(),
            nn.Dropout(drop),
        )
        self.downsample = nn.Conv1d(cin, cout, 1) if cin != cout else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(y + res)

class TCNBackbone(nn.Module):
    """
    입력: x (B, T, F)
    출력: g (B, H), H (B, H, T)
    - cfg에서 channels(list) 제공 시 그 구조를 그대로 사용
    - 없으면 levels/k/drop로 고정 H 채널을 반복
    """
    def __init__(
        self,
        in_feat: int,
        hidden: int = 64,
        channels: list | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        levels: int = 4
    ):
        super().__init__()
        k = int(kernel_size)
        drop = float(dropout)

        layers = []
        c_in = in_feat

        if channels and isinstance(channels, (list, tuple)) and len(channels) > 0:
            # ex) channels=[64,64,64]
            for i, c_out in enumerate(channels):
                d = 2 ** i
                layers.append(TBlock(c_in, int(c_out), k, d, drop))
                c_in = int(c_out)
            final_hidden = int(channels[-1])
        else:
            # fallback: hidden으로 반복
            for i in range(int(levels)):
                d = 2 ** i
                layers.append(TBlock(c_in, int(hidden), k, d, drop))
                c_in = int(hidden)
            final_hidden = int(hidden)

        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.hidden = final_hidden

    def forward(self, x):  # x:(B,T,F)
        z = x.transpose(1, 2)       # (B,F,T)
        h = self.tcn(z)             # (B,H,T)
        g = self.gap(h).squeeze(-1) # (B,H)
        return g, h
