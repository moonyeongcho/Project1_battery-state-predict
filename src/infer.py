# src/infer.py
# -*- coding: utf-8 -*-
"""
간이 추론/요약 저장 + 선택적 속도 벤치 (--bench)
- 체크포인트: artifacts/<model_name>/models/multitask_<model_name>_cellkW.pt
- 데이터: data/processed/dataset_windows_labels.npz
- split: train / val / test
- 결과: artifacts/<model_name>/<split>/preds_<split>.csv, mae_summary_<split>.json
- (옵션) --bench: n 배치 측정 → speed.json
"""
import os, json, argparse, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import load_config
from src.models.registry import build_model

class MTSet(Dataset):
    def __init__(self, pack, sid):
        self.X = pack["X"].astype(np.float32)
        self.soc = pack["soc"].astype(np.float32)       # (N,H)
        self.th  = pack["th"].astype(np.float32)        # (N,H)
        self.ta  = pack["ta"].astype(np.float32)        # (N,H)
        self.tl  = pack["tl"].astype(np.float32)        # (N,H)
        self.p_cell = pack["p_cell"].astype(np.float32) # (N,H)
        self.sid = np.asarray(sid)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.soc[i]),
            torch.from_numpy(self.th[i]),
            torch.from_numpy(self.ta[i]),
            torch.from_numpy(self.tl[i]),
            torch.from_numpy(self.p_cell[i]),
            str(self.sid[i]),
        )

@torch.no_grad()
def run_eval(model, loader, device, save_limit=1000):
    L1 = torch.nn.L1Loss(reduction="sum")
    sums = {"soc":0.0,"th":0.0,"ta":0.0,"tl":0.0,"p_cell":0.0}
    n = 0; rows = []
    for X, soc, th, ta, tl, pc, sid in loader:
        X, soc, th, ta, tl, pc = (X.to(device), soc.to(device), th.to(device),
                                  ta.to(device), tl.to(device), pc.to(device))
        psoc, pth, pta, ptl, ppc = model(X)
        bs = X.size(0); n += bs
        sums["soc"]    += float(L1(psoc, soc).item())
        sums["th"]     += float(L1(pth, th).item())
        sums["ta"]     += float(L1(pta, ta).item())
        sums["tl"]     += float(L1(ptl, tl).item())
        sums["p_cell"] += float(L1(ppc, pc).item())

        take = min(bs, max(0, save_limit - len(rows)))
        if take > 0:
            for i in range(take):
                rows.append(dict(
                    session=str(sid[i]),
                    soc_pred_last=float(psoc[i, -1].detach().cpu().item()),
                    soc_true_last=float(soc[i, -1].detach().cpu().item()),
                    pcell_pred_last=float(ppc[i, -1].detach().cpu().item()),
                    pcell_true_last=float(pc[i, -1].detach().cpu().item()),
                    th_pred_last=float(pth[i, -1].detach().cpu().item()),
                    th_true_last=float(th[i, -1].detach().cpu().item()),
                    ta_pred_last=float(pta[i, -1].detach().cpu().item()),
                    ta_true_last=float(ta[i, -1].detach().cpu().item()),
                    tl_pred_last=float(ptl[i, -1].detach().cpu().item()),
                    tl_true_last=float(tl[i, -1].detach().cpu().item()),
                ))
        if len(rows) >= save_limit: break
    mae = {k: (sums[k] / max(n,1)) for k in sums}
    return mae, pd.DataFrame(rows)

@torch.no_grad()
def bench_speed(model, loader, device, max_batches=20, warmup=3):
    use_amp = torch.cuda.is_available() and device.startswith("cuda")
    autocast_dtype = torch.float16 if use_amp else torch.bfloat16
    it = iter(loader)
    for _ in range(min(warmup, len(loader))):
        try: X, *_ = next(it)
        except StopIteration: break
        X = X.to(device)
        with torch.autocast(device_type=("cuda" if use_amp else "cpu"), dtype=autocast_dtype):
            _ = model(X)
    tot_samp = 0; t0 = time.perf_counter()
    it = iter(loader)
    for b in range(min(max_batches, len(loader))):
        try: X, *_ = next(it)
        except StopIteration: break
        X = X.to(device)
        with torch.autocast(device_type=("cuda" if use_amp else "cpu"), dtype=autocast_dtype):
            _ = model(X)
        tot_samp += X.size(0)
    dur = max(time.perf_counter() - t0, 1e-9)
    return dict(
        batches=int(min(max_batches, len(loader))),
        samples=int(tot_samp),
        total_seconds=float(dur),
        avg_latency_ms=float(1000.0 * dur / max(1, tot_samp)),
        throughput_sps=float(tot_samp / dur)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_name", type=str, default="tcn", choices=["tcn","lstm","transformer"])
    parser.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    parser.add_argument("--limit_rows", type=int, default=1000)
    parser.add_argument("--bench", type=int, default=0, help="앞쪽 n배치 속도 벤치(0=off)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    npz_path = os.path.join(cfg["data"]["processed_dir"], "dataset_windows_labels.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    sid_all = np.array([str(s).strip() for s in npz["sid"]], dtype=object)

    split_key = dict(train="sessions_train", val="sessions_val", test="sessions_test")[args.split]
    want = [str(s).strip() for s in cfg["data"][split_key]]
    mask = np.isin(sid_all, np.array(want, dtype=object))
    if not mask.any():
        raise RuntimeError(f"No samples matched {split_key}={want}. Available: {sorted(set(sid_all))}")

    keys = ["X","soc","th","ta","tl","p_cell"]
    pack = {k: npz[k][mask] for k in keys}
    sid_sel = sid_all[mask]

    dt = float(cfg["window"]["dt"])
    horizon = pack["th"].shape[-1]
    in_feat = pack["X"].shape[-1]

    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    loader = DataLoader(MTSet(pack, sid_sel), batch_size=256, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory,
                        persistent_workers=(num_workers > 0))

    model = build_model(in_feat, horizon, cfg, kind=args.model_name).to(device).eval()
    ckpt = os.path.join("artifacts", args.model_name, "models", f"multitask_{args.model_name}_cellkW.pt")
    if not os.path.exists(ckpt):
        legacy_ckpt = "artifacts/models/multitask_tcn_cellkW.pt"
        ckpt = legacy_ckpt if os.path.exists(legacy_ckpt) else ckpt
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)

    mae, df = run_eval(model, loader, device, save_limit=int(args.limit_rows))
    out_dir = os.path.join("artifacts", args.model_name, args.split)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"preds_{args.split}.csv"), index=False, encoding="utf-8", float_format="%.6f")
    with open(os.path.join(out_dir, f"mae_summary_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(mae, f, ensure_ascii=False, indent=2)

    if int(args.bench) > 0:
        speed = bench_speed(model, loader, device, max_batches=int(args.bench), warmup=3)
        with open(os.path.join(out_dir, "speed.json"), "w", encoding="utf-8") as f:
            json.dump(speed, f, ensure_ascii=False, indent=2)

    print("saved:", out_dir)

if __name__ == "__main__":
    main()
