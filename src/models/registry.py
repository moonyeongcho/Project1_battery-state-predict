# src/models/registry.py
# -*- coding: utf-8 -*-
from typing import Any, Dict

from .tcn import TCNBackbone
from .lstm import LSTMBackbone
from .transformer import TransformerBackbone
from .heads import MultiTaskHeads

def _resolve_hidden(cfg: Dict[str, Any], kind: str) -> int:
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if "hidden" in m and isinstance(m["hidden"], int) and m["hidden"] > 0:
        return int(m["hidden"])

    if kind == "tcn":
        tcn = m.get("tcn", {})
        nc = tcn.get("num_channels")
        if isinstance(nc, (list, tuple)) and len(nc) > 0 and isinstance(nc[-1], int):
            return int(nc[-1])
        return int(tcn.get("hidden", 64))
    if kind == "lstm":
        return int(m.get("lstm", {}).get("hidden_size", 128))
    if kind == "transformer":
        return int(m.get("transformer", {}).get("d_model", 128))
    return 64

def build_model(in_feat: int, horizon: int, cfg: Dict[str, Any], kind: str = None):
    """
    모델 빌더
      - TCN / LSTM / Transformer 백본 + MultiTaskHeads
      - 반환: nn.Module
    """
    model_name = (kind or cfg.get("model_name") or "tcn").lower()
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    horizon = int(horizon)

    if model_name == "tcn":
        tc = mcfg.get("tcn", {})
        channels = tc.get("num_channels", None)
        hidden = _resolve_hidden(cfg, "tcn")
        kernel_size = int(tc.get("kernel_size", 3))
        dropout = float(tc.get("dropout", 0.2))
        levels = int(tc.get("levels", 4))  # channels 없을 때만 의미
        backbone = TCNBackbone(
            in_feat, hidden=hidden,
            channels=channels,
            kernel_size=kernel_size, dropout=dropout, levels=levels
        )
        return MultiTaskHeads(backbone, horizon=horizon, hidden=backbone.hidden)

    if model_name == "lstm":
        lc = mcfg.get("lstm", {})
        hidden_size = int(lc.get("hidden_size", 128))
        num_layers  = int(lc.get("num_layers", 2))
        dropout     = float(lc.get("dropout", 0.2))
        backbone = LSTMBackbone(in_feat, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        return MultiTaskHeads(backbone, horizon=horizon, hidden=backbone.hidden)

    if model_name == "transformer":
        tc = mcfg.get("transformer", {})
        d_model = int(tc.get("d_model", 128))
        nhead   = int(tc.get("nhead", 4))
        num_encoder_layers = int(tc.get("num_encoder_layers", 3))
        dim_feedforward    = int(tc.get("dim_feedforward", 256))
        dropout = float(tc.get("dropout", 0.1))
        backbone = TransformerBackbone(
            in_feat, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        return MultiTaskHeads(backbone, horizon=horizon, hidden=backbone.hidden)

    raise NotImplementedError(
        f"Unsupported model_name='{model_name}'. Supported: 'tcn','lstm','transformer'."
    )
