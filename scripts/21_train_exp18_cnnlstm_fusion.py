
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
PATCH_FILE = DATA_DIR / "station_patches_15x15.npz"
PARQUET_FILE = DATA_DIR / "advanced_features.parquet"
OUT_DIR = ROOT / "models" / "exp18"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOOKBACK = 7 
TARGET = "PM25" # We can loop or parameterize
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
PATIENCE = 15

# Updated Tabular Columns from Exp 17
BASE_TABULAR = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "Elevation_m", "Dist_Coast_km", "Dist_Corniche_km", "Dist_E11_km",
    "Wind_U", "Wind_V", "VentilationIndex", "StabilityIndex",
    "UrbanDensity_5km"
]

# We will add DRP columns dynamically based on target

class AQIDataset(Dataset):
    def __init__(self, tabular_df, patch_data, patch_dates, patch_stations, target_col, tabular_cols, lookback=7):
        self.lookback = lookback
        self.target_col = target_col
        self.tabular_cols = tabular_cols
        
        # Map date/station to patch index
        date_map = {str(d)[:10]: i for i, d in enumerate(patch_dates)}
        stat_map = {s: i for i, s in enumerate(patch_stations)}
        
        self.sequences = []
        log.info(f"Building sequences for {target_col} (Lookback: {lookback})...")
        
        for station, group in tabular_df.groupby("StationName"):
            group = group.sort_values("Date")
            s_idx = stat_map.get(station)
            if s_idx is None: continue
            
            records = group.to_dict('records')
            
            for i in range(lookback - 1, len(records)):
                window = records[i - (lookback-1) : i + 1]
                valid = True
                window_patches = []
                window_tabular = []
                
                for rec in window:
                    d_str = str(rec['Date'])[:10]
                    p_idx = date_map.get(d_str)
                    if p_idx is None:
                        valid = False
                        break
                    
                    # Patch from npz
                    p = patch_data[p_idx, s_idx] # (15, 15, 2)
                    p = np.nan_to_num(p)
                    window_patches.append(p.transpose(2, 0, 1)) # (C, H, W)
                    
                    window_tabular.append([rec[c] for c in self.tabular_cols])
                
                if valid:
                    self.sequences.append({
                        'patches': np.array(window_patches),
                        'tabular': np.array(window_tabular),
                        'target': records[i][target_col]
                    })
        
        log.info(f"Generated {len(self.sequences)} valid sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        return (
            torch.tensor(item['patches'], dtype=torch.float32),
            torch.tensor(item['tabular'], dtype=torch.float32),
            torch.tensor(item['target'], dtype=torch.float32)
        )

class CNNLSTM(nn.Module):
    def __init__(self, tabular_dim):
        super(CNNLSTM, self).__init__()
        # CNN for spatial patches (2 channels: AOD, NO2)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 15x15 -> 7x7
        cnn_out_dim = 32 * 7 * 7
        
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU()
        )
        
        # Bi-LSTM for temporal dynamics
        self.lstm = nn.LSTM(input_size=cnn_out_dim + 64, hidden_size=128, num_layers=2, 
                            batch_first=True, dropout=0.2, bidirectional=True)
        
        self.head = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, patches, tabular):
        B, T, C, H, W = patches.size()
        cnn_out = self.cnn(patches.view(B*T, C, H, W)).view(B, T, -1)
        tab_out = self.tab_mlp(tabular.view(B*T, -1)).view(B, T, -1)
        
        combined = torch.cat([cnn_out, tab_out], dim=2)
        lstm_out, _ = self.lstm(combined)
        return self.head(lstm_out[:, -1, :]).squeeze(-1)

def run_training(target, args):
    log.info(f"===== Starting Exp 18 Training: {target} =====")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(PARQUET_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    patch_raw = np.load(PATCH_FILE, allow_pickle=True)
    patches, dates, stations = patch_raw['patches'], patch_raw['dates'], patch_raw['stations']
    
    # Define features for this target
    drp_cols = [f"DRP_{target}_lag1", f"DRP_{target}_roll3", f"DRP_{target}_roll7"]
    tabular_cols = BASE_TABULAR + drp_cols
    
    # Scale tabular (RobustScaler)
    # Fill NaNs in tabular first
    df[tabular_cols] = df[tabular_cols].fillna(df[tabular_cols].median())
    scaler = RobustScaler()
    df[tabular_cols] = scaler.fit_transform(df[tabular_cols])
    
    # Splits
    train_df = df[df["Date"] <= "2023-06-30"].copy()
    val_df   = df[(df["Date"] > "2023-06-30") & (df["Date"] <= "2023-12-31")].copy()
    test_df  = df[df["Date"] > "2023-12-31"].copy()
    
    # Drop rows where target is NaN
    train_df = train_df.dropna(subset=[target])
    val_df   = val_df.dropna(subset=[target])
    test_df  = test_df.dropna(subset=[target])
    
    train_ds = AQIDataset(train_df, patches, dates, stations, target, tabular_cols, args.lookback)
    val_ds   = AQIDataset(val_df, patches, dates, stations, target, tabular_cols, args.lookback)
    test_ds  = AQIDataset(test_df, patches, dates, stations, target, tabular_cols, args.lookback)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)
    
    model = CNNLSTM(len(tabular_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    best_rmse = float('inf')
    no_improve = 0
    
    for epoch in range(args.epochs):
        model.train()
        t_loss = 0
        for p, t, y in train_loader:
            p, t, y = p.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(p, t)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * p.size(0)
        
        model.eval()
        v_rmse = 0
        with torch.no_grad():
            for p, t, y in val_loader:
                p, t, y = p.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
                v_rmse += nn.functional.mse_loss(model(p, t), y).item() * p.size(0)
        v_rmse = np.sqrt(v_rmse / len(val_ds))
        
        if v_rmse < best_rmse:
            best_rmse = v_rmse
            torch.save(model.state_dict(), OUT_DIR / f"{target}_best.pt")
            no_improve = 0
        else:
            no_improve += 1
            
        if (epoch + 1) % 5 == 0:
            log.info(f"Epoch {epoch+1} | Train Loss: {t_loss/len(train_ds):.2f} | Val RMSE: {v_rmse:.2f}")
        
        if no_improve >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final Test
    model.load_state_dict(torch.load(OUT_DIR / f"{target}_best.pt"))
    model.eval()
    all_preds, all_truth = [], []
    with torch.no_grad():
        for p, t, y in test_loader:
            p, t, y = p.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
            all_preds.extend(model(p, t).cpu().numpy())
            all_truth.extend(y.cpu().numpy())
    
    r2 = r2_score(all_truth, all_preds)
    rmse = np.sqrt(mean_squared_error(all_truth, all_preds))
    log.info(f"FINAL TEST [{target}]: R2={r2:.4f} | RMSE={rmse:.2f}")
    return r2, rmse

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    results = {}
    for target in [TARGET, "PM10"]:
        r2, rmse = run_training(target, args)
        results[target] = (r2, rmse)
        
    with open(OUT_DIR / "exp18_report.txt", "w") as f:
        f.write("EXPERIMENT 18: CNN-LSTM Fusion (DRP + LUR + Patches)\n")
        f.write("="*60 + "\n")
        for target, (r2, rmse) in results.items():
            f.write(f"{target} R2: {r2:.4f}, RMSE: {rmse:.2f}\n")
