# src/models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskHeads(nn.Module):
    """
    백본은 (B,T,F) 입력을 받아
      g: (B, hidden)     — 전역(풀링) 피처
      H: (B, hidden, T)  — 시퀀스 피처(채널=hidden, 길이=T)
    를 반환해야 함.

    변경 사항:
      - SOC, P_cell도 시퀀스 예측(0~Horizon)
      - 온도(High/Avg/Low)는 시퀀스 예측
    """
    def __init__(self, backbone, horizon, hidden=64):
        super().__init__()
        self.backbone = backbone
        self.horizon = int(horizon)

        # 시퀀스 헤드
        self.head_soc_seq = nn.Conv1d(hidden, 1, kernel_size=1)
        self.head_p_seq   = nn.Conv1d(hidden, 1, kernel_size=1)
        self.temp_proj    = nn.Conv1d(hidden, 3, kernel_size=1)

    def _align_horizon(self, seq):
        # seq: (B, C, T). horizon과 길이 맞춤
        T = seq.shape[-1]
        if self.horizon <= T:
            return seq[:, :, -self.horizon:]
        pad_len = self.horizon - T
        return F.pad(seq, (0, pad_len), mode="replicate")

    def forward(self, x):
        g, H = self.backbone(x)              # g 안 써도 무방(확장 여지)
        soc_seq = self._align_horizon(self.head_soc_seq(H))  # (B,1,H)
        p_seq   = self._align_horizon(self.head_p_seq(H))    # (B,1,H)
        tseq    = self._align_horizon(self.temp_proj(H))     # (B,3,H)
        th, ta, tl = tseq[:, 0, :], tseq[:, 1, :], tseq[:, 2, :]
        return soc_seq.squeeze(1), th, ta, tl, p_seq.squeeze(1)
