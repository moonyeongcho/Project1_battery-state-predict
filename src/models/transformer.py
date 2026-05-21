# src/models/transformer.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):    # 위치 정보 인코딩
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                          # (L,D) -> L: 시퀀스 길이, D: 임베딩 차원
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # unsqueeze(1) -> (L,1) -> 1차원 추가
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 2씩 증가
        pe[:, 0::2] = torch.sin(position * div_term) # 짝수 번째 인덱스에 sin 적용
        pe[:, 1::2] = torch.cos(position * div_term) # 홀수 번째 인덱스에 cos 적용
        self.register_buffer("pe", pe.unsqueeze(0))                 # (1,L,D)

    def forward(self, x):  # x: (B,T,D) -> B: 배치 크기, T: 시퀀스 길이, D: 임베딩 차원
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerBackbone(nn.Module):
    """
    입력: x (B,T,F)  -> proj(d_model) -> PE -> Encoder
    출력: g (B,D), H (B,D,T)
    """
    def __init__(
        self,
        in_feat: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.proj = nn.Linear(in_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            activation="gelu", layer_norm_eps=layer_norm_eps
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.posenc = PositionalEncoding(d_model)
        self.hidden = int(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):     # (B,T,F)
        z = self.proj(x)      # (B,T,D)
        z = self.posenc(z)
        y = self.encoder(z)   # (B,T,D)
        y = self.norm_out(y)
        g = y.mean(dim=1)     # (B,D)
        H = y.transpose(1, 2) # (B,D,T)
        return g, H
