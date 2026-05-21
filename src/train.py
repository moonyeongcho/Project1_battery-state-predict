# src/train.py
# -*- coding: utf-8 -*-
import os, json, random, argparse, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.config import load_config
from src.models.registry import build_model

SUPPORTS_UTF8 = (sys.stdout.encoding or "").lower().startswith("utf")
CHECK = "✅" if SUPPORTS_UTF8 else "[BEST]"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_train_defaults(cfg):
    tr = dict(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=64, lr=1e-3, epochs=50,
        loss_weight=dict(soc=1.0, temp=0.33, pavail=1.0),
        use_amp=True, num_workers=0, pin_memory=True
    )
    user = cfg.get("train", {})
    tr.update({k: user[k] for k in user if k != "loss_weight"})
    if "loss_weight" in user:
        tr["loss_weight"].update(user["loss_weight"])
    model_name = (cfg.get("model_name") or "tcn").lower()
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    per_model = mcfg.get(model_name, {}) if isinstance(mcfg, dict) else {}
    for k in ["lr", "batch_size", "epochs"]:
        if k in per_model: tr[k] = per_model[k]
    return tr

def normalize_device_amp(cfg_train):
    want_cuda = str(cfg_train.get("device", "")).lower().startswith("cuda")
    has_cuda = torch.cuda.is_available()
    if want_cuda and not has_cuda:
        print("[warn] CUDA not available. Falling back to CPU.")
        cfg_train["device"] = "cpu"
    if str(cfg_train["device"]).lower().startswith("cpu"):
        if cfg_train.get("use_amp", False):
            print("[info] AMP disabled on CPU.")
        cfg_train["use_amp"] = False
        cfg_train["pin_memory"] = False
    return cfg_train

class MTSet(Dataset):
    def __init__(self, pack):
        self.X = pack["X"].astype(np.float32)
        self.soc = pack["soc"].astype(np.float32)       # (N,H)
        self.th  = pack["th"].astype(np.float32)        # (N,H)
        self.ta  = pack["ta"].astype(np.float32)        # (N,H)
        self.tl  = pack["tl"].astype(np.float32)        # (N,H)
        self.p_cell = pack["p_cell"].astype(np.float32) # (N,H)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.soc[i]),
            torch.from_numpy(self.th[i]),
            torch.from_numpy(self.ta[i]),
            torch.from_numpy(self.tl[i]),
            torch.from_numpy(self.p_cell[i]),
        )

def train_epoch(model, loader, opt, cfg_train, scaler=None):
    model.train()
    crit = nn.SmoothL1Loss()
    lw = cfg_train["loss_weight"]; device = cfg_train["device"]
    use_amp = bool(cfg_train.get("use_amp", False))
    total = 0.0; n = 0
    for X, soc, th, ta, tl, pc in loader:
        X, soc, th, ta, tl, pc = (X.to(device), soc.to(device), th.to(device),
                                  ta.to(device), tl.to(device), pc.to(device))
        opt.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.autocast(device_type=("cuda" if "cuda" in device else "cpu"),
                                dtype=torch.float16 if "cuda" in device else torch.bfloat16):
                psoc, pth, pta, ptl, ppc = model(X)
                L = (lw["soc"]*crit(psoc, soc)
                     + lw["temp"]*(crit(pth, th)+crit(pta, ta)+crit(ptl, tl))
                     + lw["pavail"]*crit(ppc, pc))
            scaler.scale(L).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            psoc, pth, pta, ptl, ppc = model(X)
            L = (lw["soc"]*crit(psoc, soc)
                 + lw["temp"]*(crit(pth, th)+crit(pta, ta)+crit(ptl, tl))
                 + lw["pavail"]*crit(ppc, pc))
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = X.size(0); total += float(L.item())*bs; n += bs
    return total/max(n,1)

@torch.no_grad()
def eval_epoch(model, loader, cfg_train):
    model.eval()
    L1 = nn.L1Loss()
    m = {"soc": 0.0, "temp": 0.0, "p_cell": 0.0, "n": 0}
    device = cfg_train["device"]
    for X, soc, th, ta, tl, pc in loader:
        X, soc, th, ta, tl, pc = (X.to(device), soc.to(device), th.to(device),
                                  ta.to(device), tl.to(device), pc.to(device))
        psoc, pth, pta, ptl, ppc = model(X)
        bs = X.size(0)
        m["soc"]    += float(L1(psoc, soc).item()) * bs
        m["temp"]   += float(L1(pth, th).item() + L1(pta, ta).item() + L1(ptl, tl).item()) * bs
        m["p_cell"] += float(L1(ppc, pc).item()) * bs
        m["n"] += bs
    for k in ["soc", "temp", "p_cell"]:
        m[k] = m[k] / max(m["n"], 1)
    lw = cfg_train.get("loss_weight", {})
    w_soc  = float(lw.get("soc", 1.0))
    w_temp = float(lw.get("temp", 0.33))
    w_p    = float(lw.get("pavail", lw.get("p_cell", 1.0)))
    m["total"] = w_soc * m["soc"] + w_temp * m["temp"] + w_p * m["p_cell"]
    return m

def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.optim as optim
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_name", type=str, default="tcn", choices=["tcn","lstm","transformer"])
    args = parser.parse_args()

    set_seed(42)
    cfg = load_config(args.config)
    cfg["model_name"] = args.model_name
    cfg_train = get_train_defaults(cfg)
    cfg_train = normalize_device_amp(cfg_train)

    npz_path = os.path.join(cfg["data"]["processed_dir"], "dataset_windows_labels.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}. Run: python -m src.labels --config {args.config}")
    npz = np.load(npz_path, allow_pickle=True)

    sid = np.array([str(s).strip() for s in npz["sid"]], dtype=object)
    tr_list = [str(s).strip() for s in cfg["data"]["sessions_train"]]
    va_list = [str(s).strip() for s in cfg["data"]["sessions_val"]]
    mask_tr = np.isin(sid, np.array(tr_list, dtype=object))
    mask_va = np.isin(sid, np.array(va_list, dtype=object))
    for who, m in [("train", mask_tr), ("val", mask_va)]:
        if not m.any():
            avail = sorted(set(sid))
            raise RuntimeError(f"No samples for {who}. Available sessions: {avail}")

    keys = ["X","soc","th","ta","tl","p_cell"]
    pack_tr = {k: npz[k][mask_tr] for k in keys}
    pack_va = {k: npz[k][mask_va] for k in keys}

    in_feat = npz["X"].shape[-1]
    horizon = npz["th"].shape[-1]
    model = build_model(in_feat, horizon, cfg, kind=args.model_name)

    device = cfg_train["device"]
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=float(cfg_train["lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg_train.get("use_amp", False)))

    train_loader = DataLoader(MTSet(pack_tr), batch_size=int(cfg_train["batch_size"]),
                              shuffle=True, num_workers=int(cfg_train["num_workers"]),
                              pin_memory=bool(cfg_train["pin_memory"]))
    val_loader   = DataLoader(MTSet(pack_va), batch_size=int(cfg_train["batch_size"]),
                              shuffle=False, num_workers=int(cfg_train["num_workers"]),
                              pin_memory=bool(cfg_train["pin_memory"]))

    hist = []; best = (1e9, 0)
    art_dir = os.path.join("artifacts", args.model_name, "models")
    os.makedirs(art_dir, exist_ok=True)
    ckpt_path = os.path.join(art_dir, f"multitask_{args.model_name}_cellkW.pt")

    for ep in range(1, int(cfg_train["epochs"])+1):
        tr_loss = train_epoch(model, train_loader, opt, cfg_train, scaler)
        va = eval_epoch(model, val_loader, cfg_train)
        hist.append(va["total"])
        print(f"epoch {ep:03d} | train={tr_loss:.6f} | val_total={va['total']:.6f} "
              f"(soc={va['soc']:.6f}, temp={va['temp']:.6f}, p={va['p_cell']:.6f})")
        if va["total"] < best[0]:
            best = (va["total"], ep)
            torch.save(model.state_dict(), ckpt_path)
            print(CHECK, "best updated:", best)

    try:
        xs = np.arange(1, len(hist)+1)
        plt.figure(figsize=(8,5))
        plt.plot(xs, np.array(hist), linewidth=2.2)
        plt.xlabel("epoch"); plt.ylabel("val_total_loss"); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(ckpt_path), "val_curve.png"), dpi=600, bbox_inches="tight"); plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
