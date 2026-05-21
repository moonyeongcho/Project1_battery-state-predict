# src/labels.py
# -*- coding: utf-8 -*-
"""
라벨 생성 (멀티태스크 '시퀀스' 학습용, 벡터화 고속 버전)
- 입력 X: 전처리된 merged_clean.csv의 스케일된 입력
- 라벨: 시퀀스 (길이 = horizon_seconds / dt)
  * SOC(t..t+H-1)
  * Temp_high/avg/low(t..t+H-1)
  * P_cell(kW)(t..t+H-1): R0–SOC + 온도 보정 + 전압/열 제약
"""
import os, math, pickle
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from typing import List
from src.config import load_config

# ---------------------- R0–SOC 로드/보간 ----------------------
def load_r0(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"R0–SOC file not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    df.columns = [c.strip() for c in df.columns]
    soc_col, r_col = None, None
    for c in df.columns:
        cl = c.lower()
        if ("soc" in cl) and soc_col is None: soc_col = c
        if (c == "R0") or ("r0" in cl) or ("resistance" in cl): r_col = c
    if soc_col is None or r_col is None:
        raise ValueError(f"R0–SOC csv must have SOC and R0 columns. columns={list(df.columns)}")

    out = df[[soc_col, r_col]].copy(); out.columns = ["SOC", "R0"]
    r_numeric = pd.to_numeric(out["R0"], errors="coerce")
    med = np.nanmedian(r_numeric)
    out["R0_ohm"] = r_numeric / 1000.0 if (np.isfinite(med) and 1 <= med <= 50) else r_numeric
    out["SOC"] = pd.to_numeric(out["SOC"], errors="coerce")

    # SOC 단위 보정(0~1 → %)
    soc_min, soc_max = np.nanmin(out["SOC"].values), np.nanmax(out["SOC"].values)
    if np.isfinite(soc_min) and np.isfinite(soc_max) and soc_max <= 1.5:
        out["SOC"] = out["SOC"] * 100.0

    out = out.dropna(subset=["SOC", "R0_ohm"]).copy()
    out["SOC"] = out["SOC"].clip(0, 100)
    out = (out.groupby("SOC", as_index=False)["R0_ohm"].median()
              .sort_values("SOC").reset_index(drop=True))
    if len(out) < 2:
        raise ValueError(f"R0–SOC table too small: {len(out)}")
    return out[["SOC", "R0_ohm"]]

# ---------------------- 벡터화 유틸 ----------------------
def r0_temp_correction_vec(R0_25C: np.ndarray, temp_C: np.ndarray,
                           k_low: float = 0.03, k_high: float = 0.005) -> np.ndarray:
    """
    temp<25: R0 = R0_25C * (1 + k_low*(25-T))
    temp>=25: R0 = R0_25C * (1 + k_high*(T-25))
    벡터화 버전
    """
    R = np.asarray(R0_25C, dtype=float)
    T = np.asarray(temp_C, dtype=float)
    R = np.maximum(R, 1e-12)
    out = np.empty_like(R, dtype=float)
    mask_low = T < 25.0
    out[mask_low]  = R[mask_low]  * (1.0 + k_low  * (25.0 - T[mask_low]))
    out[~mask_low] = R[~mask_low] * (1.0 + k_high * (T[~mask_low] - 25.0))
    return out

def ploss_of_temp_vec(base_ploss: float, T: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """
    온도별 열허용치 derating: base_ploss * scale(T)
    curve: ndarray [[T, scale], ...]
    """
    if not np.isfinite(base_ploss) or base_ploss < 0:
        base_ploss = 0.0
    if curve is None or curve.size == 0:
        return np.full_like(T, max(0.0, base_ploss), dtype=float)
    Ts, S = curve[:, 0], curve[:, 1]
    scale = np.interp(T, Ts, S, left=S[0], right=S[-1])
    return np.maximum(0.0, base_ploss * scale)

def cell_limit_vec(R: np.ndarray, Vnom: float, Vmin: float, Ploss: np.ndarray) -> np.ndarray:
    """
    벡터화: 각 시점의 R, Ploss에 대해 셀 가용 출력(kW) 계산
    """
    R = np.maximum(R, 1e-12)
    I_star = Vnom / (2.0 * R)
    I_v    = (Vnom - Vmin) / R
    I_th   = np.sqrt(np.maximum(Ploss, 0.0) / R)
    I_max  = np.maximum(0.0, np.minimum.reduce([I_star, I_v, I_th]))
    P_cell_W = Vnom * I_max - (I_max ** 2) * R
    return np.maximum(0.0, P_cell_W) / 1000.0  # kW

# ---------------------- 메인 ----------------------
def main():
    cfg = load_config()
    interim = cfg["data"]["interim_dir"]
    processed = cfg["data"]["processed_dir"]
    os.makedirs(processed, exist_ok=True)

    # 윈도우 파라미터
    dt = float(cfg["window"]["dt"])
    past_sec = cfg["window"].get("past_seconds", cfg["window"].get("past", 15.0))
    horizon_sec = cfg["window"].get("horizon_seconds", cfg["window"].get("future", 30.0))
    P = int(round(past_sec / dt))     # past steps
    H = int(round(horizon_sec / dt))  # horizon steps

    # 입력 로드
    merged_csv = os.path.join(interim, "merged_clean.csv")
    if not os.path.exists(merged_csv):
        raise FileNotFoundError(f"merged_clean.csv not found at {merged_csv}")
    df = pd.read_csv(merged_csv, encoding="utf-8")

    # 스케일러 로드
    sc_cfg = cfg.get("preprocess", {}).get("scaling", {})
    scaler_path = sc_cfg.get("save_path", "artifacts/scalers/input_robust.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler pickle not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        pack = pickle.load(f)
    scaler = pack["scaler"]; use_cols: List[str] = pack["cols"]

    X_scaled = df[use_cols].values.astype(np.float32)         # (N, F)
    inv = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=use_cols)

    # 파생 열 이름
    th_col, ta_col, tl_col = "Temp_high", "Temp_avg", "Temp_low"

    # 길이/인덱스 계산
    N = X_scaled.shape[0]
    # 유효 중심 인덱스 i 범위: i in [P-1, N-H-1], 개수 M = N - P - H + 1
    M = N - P - H + 1
    if M <= 0:
        raise RuntimeError(f"Not enough rows for windows. Need N >= P+H. Got N={N}, P={P}, H={H}")
    start = P - 1
    stop  = start + M   # exclusive

    # ---------------------- 윈도우 벡터화 ----------------------
    # 입력: (M, P, F)
    X_w = sliding_window_view(X_scaled, window_shape=(P, X_scaled.shape[1]))  # (N-P+1, 1, P, F)
    X_w = X_w[:, 0, :, :]                    # (N-P+1, P, F)
    X_w = X_w[:M, :, :]                      # (M, P, F)  (j0=0..M-1 → i=P-1..P-1+M-1)

    # 타깃: 각 시퀀스 시작 k0 = i, 길이 H
    soc_all = inv["SOC"].values.astype(np.float32)
    th_all  = inv[th_col].values.astype(np.float32)
    ta_all  = inv[ta_col].values.astype(np.float32)
    tl_all  = inv[tl_col].values.astype(np.float32)

    soc_sw = sliding_window_view(soc_all, window_shape=H)  # (N-H+1, H)
    th_sw  = sliding_window_view(th_all,  window_shape=H)
    ta_sw  = sliding_window_view(ta_all,  window_shape=H)
    tl_sw  = sliding_window_view(tl_all,  window_shape=H)

    soc_seq = soc_sw[start:stop, :]   # (M, H)
    th_seq  = th_sw[start:stop, :]
    ta_seq  = ta_sw[start:stop, :]
    tl_seq  = tl_sw[start:stop, :]

    # 세션 id (i를 중심으로 선택)
    if "session" in df.columns:
        sid_all = np.asarray(df["session"].values, dtype=object)
    else:
        sid_all = np.array(["S"] * N, dtype=object)
    sid = sid_all[start:stop]         # (M,)

    # ---------------------- P_cell 시퀀스 (벡터화) ----------------------
    # R0–SOC 표/보간 준비
    r0_csv = cfg["data"].get(
        "r0_soc_file",
        os.path.join(cfg["data"]["raw_dir"], "20250808_R0_SOC_charge_discharge.csv")
    )
    rtab = load_r0(r0_csv)
    SOC_tab = rtab["SOC"].values.astype(float)
    R0_tab  = rtab["R0_ohm"].values.astype(float)

    # R0(25C) 보간: soc_seq (M,H) → R0_25C (M,H)
    R0_25C = np.interp(soc_seq, SOC_tab, R0_tab, left=R0_tab[0], right=R0_tab[-1])

    # 온도 보정 R(T): th_seq 사용(가장 보수적인 high 기준)
    R_T = r0_temp_correction_vec(R0_25C, th_seq)

    # 온도 derating → Ploss(T) (W)
    base_ploss = float(cfg.get("pack", {}).get("P_loss_base_W", 8.0))
    der_curve  = np.asarray(cfg.get("pack", {}).get("ploss_derating_curve",
                        [[0,0.6],[25,1.0],[40,0.9],[60,0.7]]), dtype=float)
    P_loss = ploss_of_temp_vec(base_ploss, th_seq, der_curve)

    # 전압/셀 파라미터
    Vnom = float(cfg.get("pack", {}).get("V_nom_cell", 3.7))
    Vmin = float(cfg.get("pack", {}).get("V_min_cell", 3.0))

    # P_cell(kW): 벡터식
    p_cell_seq = cell_limit_vec(R_T, Vnom, Vmin, P_loss)    # (M,H), kW

    # ---------------------- 저장 ----------------------
    out_path = os.path.join(processed, "dataset_windows_labels.npz")
    np.savez_compressed(
        out_path,
        X=X_w.astype(np.float32),
        soc=soc_seq.astype(np.float32),
        th=th_seq.astype(np.float32),
        ta=ta_seq.astype(np.float32),
        tl=tl_seq.astype(np.float32),
        p_cell=p_cell_seq.astype(np.float32),
        sid=sid
    )
    print("saved:", out_path,
          "| X:", X_w.shape, "| soc:", soc_seq.shape, "| th:", th_seq.shape, "| p_cell:", p_cell_seq.shape)

if __name__ == "__main__":
    main()
