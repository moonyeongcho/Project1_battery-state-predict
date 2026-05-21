import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==========================
# ===== 고정 하이퍼파라미터
# ==========================
seed = 4
n_steps_out = 300
epochs = 200  # 최종 재학습 epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
np.random.seed(seed)

# ===== 모델 저장 경로 (바탕화면) =====
DESKTOP_DIR = os.path.expanduser("~/Desktop")
os.makedirs(DESKTOP_DIR, exist_ok=True)
model_path = os.path.join(DESKTOP_DIR, "best_multi_target_lstm_model.pth")
print(f"모델은 여기 저장됩니다: {model_path}")

# ===== 파일 리스트 =====
file_list = [
    r"C:\Users\Me_llamo_Show\Desktop\RACE 2025 기술아이더\배터리 테스트\6060_regen70 데이터\preprocessed_raw&diff_last.xlsx",
    r"C:\Users\Me_llamo_Show\Desktop\RACE 2025 기술아이더\배터리 테스트\6070_regen70 데이터\preprocessed_raw&diff_last.xlsx",
    r"C:\Users\Me_llamo_Show\Desktop\RACE 2025 기술아이더\배터리 테스트\6080_regen70 데이터\preprocessed_raw&diff_last.xlsx",
    r"C:\Users\Me_llamo_Show\Desktop\RACE 2025 기술아이더\배터리 테스트\6090_regen70 데이터\preprocessed_raw&diff_last.xlsx",
]

# ===== 피처/타겟 =====
features = ['Total Pressure','Temp1_trend','Current','Power','Current.1','Total Pressure_diff',
            'Remaining Capacity_diff','Power_diff','SOC','SOC_diff','Temp3']
target_cols = ['Temp1_trend','SOC']

# ==========================
# ===== 유틸 & 모델
# ==========================
class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best = float('inf')
        self.early_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class MultiTargetLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MultiTargetLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_direct_sequences(X, Y, look_back, n_steps_out):
    Xs, Ys = [], []
    for i in range(len(X) - look_back - n_steps_out + 1):
        Xs.append(X[i:i+look_back])
        Ys.append(Y[i+look_back:i+look_back+n_steps_out])
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    return Xs, Ys.reshape(len(Ys), Ys.shape[1] * Ys.shape[2])

def build_dataloaders(look_back: int, batch_size: int):
    train_files = file_list[:-1]
    X_train_all, Y_train_all = [], []
    for file_path in train_files:
        df = pd.read_excel(file_path)
        rolling_ratio = 0.15
        rolling_window = max(1, int(len(df) * rolling_ratio))
        df['Temp1_trend'] = df['Temp1'].rolling(window=rolling_window, min_periods=1).mean()
        X_scaled = df[features].values
        Y_scaled = df[target_cols].values
        X_seq, Y_seq = create_direct_sequences(X_scaled, Y_scaled, look_back, n_steps_out)
        X_train_all.append(X_seq); Y_train_all.append(Y_seq)
    X_train_all = np.concatenate(X_train_all, axis=0)
    Y_train_all = np.concatenate(Y_train_all, axis=0)

    val_file = file_list[-1]
    dfv = pd.read_excel(val_file)
    rolling_ratio = 0.15
    rolling_window = max(1, int(len(dfv) * rolling_ratio))
    dfv['Temp1_trend'] = dfv['Temp1'].rolling(window=rolling_window, min_periods=1).mean()
    X_val_scaled = dfv[features].values
    Y_val_scaled = dfv[target_cols].values
    X_val, Y_val = create_direct_sequences(X_val_scaled, Y_val_scaled, look_back, n_steps_out)

    train_dataset = TensorDataset(torch.tensor(X_train_all, dtype=torch.float32),
                                  torch.tensor(Y_train_all, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train_all.shape[2]
    output_size = Y_train_all.shape[1]
    return train_loader, val_loader, input_size, output_size

def l1_penalty(model: nn.Module, exclude_bias=True, exclude_norm=True):
    l1 = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if exclude_bias and ('bias' in name):
            continue
        if exclude_norm and any(k in name.lower() for k in ['norm', 'bn', 'layernorm', 'ln']):
            continue
        l1 = l1 + p.abs().sum()
    return l1

def build_adamw_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or any(k in name.lower() for k in ['norm', 'bn', 'layernorm', 'ln']):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# ==========================
# ===== 최종 학습 함수
# ==========================
def run_training_final_and_save(best_params: dict):
    look_back = best_params['look_back']
    batch_size = best_params['batch_size']
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    learning_rate = best_params['learning_rate']
    alpha = best_params['alpha']
    rho = best_params['rho']
    max_grad_norm = best_params['max_grad_norm']
    early_patience = best_params['early_patience']
    sched_factor = best_params['sched_factor']
    sched_patience = best_params['sched_patience']

    # Elastic-net 스타일: L1/L2 분해
    l1_lambda = alpha * rho
    l2_lambda = alpha * (1 - rho)

    train_loader, val_loader, input_size, output_size = build_dataloaders(look_back, batch_size)
    model = MultiTargetLSTM(input_size=input_size, hidden_size=hidden_size,
                            output_size=output_size, num_layers=num_layers).to(device)

    criterion = nn.MSELoss()
    param_groups = build_adamw_param_groups(model, weight_decay=l2_lambda)
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=sched_factor, patience=sched_patience, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    early_stopper = EarlyStopping(patience=early_patience)

    file_best_val_loss = float('inf')

    for epoch in range(epochs):
        # ----- Train -----
        model.train()
        train_total_loss = 0.0
        train_mse_accum = 0.0
        train_l1_accum = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                output = model(X_batch)
                mse_loss = criterion(output, Y_batch)
                l1 = l1_penalty(model)
                loss = mse_loss + l1_lambda * l1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_total_loss += loss.item()
            train_mse_accum += mse_loss.item()
            train_l1_accum += (l1_lambda * l1).item()

        avg_train_total = train_total_loss / max(1, len(train_loader))
        avg_train_mse   = train_mse_accum / max(1, len(train_loader))
        avg_train_l1    = train_l1_accum / max(1, len(train_loader))

        # ----- Validation -----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    output = model(X_batch)
                    loss = criterion(output, Y_batch)
                val_loss_sum += loss.item()
        avg_val_loss = val_loss_sum / max(1, len(val_loader))

        # 스케줄러 & 로깅
        scheduler.step(avg_val_loss)
        lr_now = float(optimizer.param_groups[0]['lr'])

        # 베스트 모델 저장(바탕화면)
        is_best = ""
        if avg_val_loss < file_best_val_loss:
            file_best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            is_best = "  <-- ✅ best & saved"

        # === 매 epoch 출력 ===
        print(f"Epoch {epoch+1:03d} | "
              f"train_total={avg_train_total:.6e} (mse={avg_train_mse:.6e}, l1={avg_train_l1:.6e}) | "
              f"val={avg_val_loss:.6e} | lr={lr_now:.6g}{is_best}")

        # Early stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("⏹ Early stopping")
            break

    print(f"🏁 최종 완료. Best Val Loss: {file_best_val_loss:.6e}")
    print(f"🧩 저장 경로: {model_path}")

# ==========================
# ===== main
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # 여기에 직접 최적 파라미터 기입
    best_params = {
        "look_back": 26,
        "batch_size": 48,
        "hidden_size": 256,
        "num_layers": 1,
        "learning_rate": 0.01022822555443743,
        "alpha": 1.386770467366391e-07,
        "rho": 0.4877932782501319,
        "max_grad_norm": 4.1207492984840925,
        "early_patience": 22,
        "sched_factor": 0.4973206496293012,
        "sched_patience": 4
    }

    run_training_final_and_save(best_params)
