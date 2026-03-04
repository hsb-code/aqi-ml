"""
scripts/16_train_exp14_cnnlstm.py
======================================
Experiment 14: CNN-LSTM Hybrid Model
Spatial Patches + Tabular Features + Temporal lookback.

Architecture:
1. CNN (2D) for 15x15 NO2/AOD patches.
2. MLP for tabular ERA5 + static geography.
3. LSTM to fuse sequences (Lookback = 3 days).
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
PATCH_FILE    = PROCESSED_DIR / "station_patches_15x15.npz"
PARQUET_FILE  = PROCESSED_DIR / "training_data_full.parquet"
OUT_DIR       = ROOT / "models" / "exp14"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOOKBACK = 7 # Increased from 3
TARGET   = "PM25"
BATCH_SIZE = 32
EPOCHS     = 100 # Increased from 30
LR         = 1e-4
PATIENCE   = 15  # Early stopping patience

TABULAR_COLS = [
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude", "Elevation_m", "Dist_Coast_km"
]

# ---------------------------------------------------------------------------
# Dataset & Model
# ---------------------------------------------------------------------------

class AQIDataset(Dataset):
    def __init__(self, tabular_df, patch_data, patch_dates, patch_stations, target_col, lookback=3):
        self.lookback = lookback
        self.target_col = target_col
        
        # Align patches and tabular
        # Map date/station to patch index
        date_map = {str(d)[:10]: i for i, d in enumerate(patch_dates)}
        stat_map = {s: i for i, s in enumerate(patch_stations)}
        
        # We need continuous sequences per station
        self.sequences = []
        
        log.info("Building sequences for CNN-LSTM...")
        for station, group in tabular_df.groupby("StationName"):
            group = group.sort_values("Date")
            
            # Map patches to this group
            s_idx = stat_map.get(station)
            if s_idx is None: continue
            
            # Convert group to list for easier sliding window
            records = group.to_dict('records')
            
            for i in range(lookback - 1, len(records)):
                # Check if we have patches for all days in window
                window = records[i - (lookback-1) : i + 1]
                
                # Check date continuity (optional but good)
                # For this dataset, we'll allow gaps but check if patches exist
                valid = True
                window_patches = []
                window_tabular = []
                
                for rec in window:
                    d_str = str(rec['Date'])[:10]
                    p_idx = date_map.get(d_str)
                    if p_idx is None:
                        valid = False
                        break
                    
                    p = patch_data[p_idx, s_idx] # (15, 15, 2)
                    # Normalize patch (naive: divide by max or standardize)
                    p = np.nan_to_num(p) 
                    window_patches.append(p.transpose(2, 0, 1)) # (C, H, W)
                    
                    tab = [rec[c] for c in TABULAR_COLS]
                    window_tabular.append(tab)
                
                if valid:
                    self.sequences.append({
                        'patches': np.array(window_patches), # (T, C, H, W)
                        'tabular': np.array(window_tabular), # (T, F)
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
    def __init__(self, tabular_dim, patch_size=15):
        super(CNNLSTM, self).__init__()
        
        # CNN branch for spatial patches
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output dim
        # 15x15 -> MaxPool -> 7x7
        cnn_out_dim = 32 * 7 * 7
        
        # MLP for tabular features (applied to each step)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU()
        )
        
        # LSTM layer
        # Input size = CNN features + MLP features
        self.lstm = nn.LSTM(input_size=cnn_out_dim + 64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, patches, tabular):
        # patches: (B, T, C, H, W)
        # tabular: (B, T, F)
        
        batch_size, seq_len, C, H, W = patches.size()
        
        # Run CNN on each time step
        cnn_in = patches.view(batch_size * seq_len, C, H, W)
        cnn_out = self.cnn(cnn_in) # (B*T, cnn_out_dim)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        
        # Run MLP on tabular
        tab_in = tabular.view(batch_size * seq_len, -1)
        tab_out = self.tabular_mlp(tab_in)
        tab_out = tab_out.view(batch_size, seq_len, -1)
        
        # Combine
        combined = torch.cat([cnn_out, tab_out], dim=2) # (B, T, features)
        
        # LSTM
        lstm_out, _ = self.lstm(combined)
        
        # Take last time step
        last_out = lstm_out[:, -1, :]
        
        # Final prediction
        return self.head(last_out).squeeze(-1)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info("Loading data files...")
    df = pd.read_parquet(PARQUET_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    patch_data_raw = np.load(PATCH_FILE, allow_pickle=True)
    patches  = patch_data_raw['patches']
    dates    = patch_data_raw['dates']
    stations = patch_data_raw['stations']
    
    # Scale tabular data
    scaler = RobustScaler()
    df[TABULAR_COLS] = scaler.fit_transform(df[TABULAR_COLS])
    joblib.dump(scaler, OUT_DIR / "tabular_scaler.pkl")
    
    # Splits based on date (consistent with other exps)
    train_df = df[df["Date"] <= "2023-06-30"]
    val_df   = df[(df["Date"] > "2023-06-30") & (df["Date"] <= "2023-12-31")]
    test_df  = df[df["Date"] > "2023-12-31"]
    
    train_ds = AQIDataset(train_df, patches, dates, stations, TARGET, args.lookback)
    val_ds   = AQIDataset(val_df, patches, dates, stations, TARGET, args.lookback)
    test_ds  = AQIDataset(test_df, patches, dates, stations, TARGET, args.lookback)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)
    
    model = CNNLSTM(tabular_dim=len(TABULAR_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    
    log.info(f"Starting training for {args.epochs} epochs on {device} (Lookback: {args.lookback})...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for p, t, target in train_loader:
            p, t, target = p.to(device), t.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(p, t)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * p.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Val
        model.eval()
        val_rmse = 0
        with torch.no_grad():
            for p, t, target in val_loader:
                p, t, target = p.to(device), t.to(device), target.to(device)
                output = model(p, t)
                val_rmse += nn.functional.mse_loss(output, target).item() * p.size(0)
        
        val_rmse = np.sqrt(val_rmse / len(val_loader.dataset))
        
        # Step scheduler
        scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), OUT_DIR / f"{TARGET}_cnnlstm_best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 5 == 0:
            log.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.2f} | Val RMSE: {val_rmse:.2f} | Best: {best_val_rmse:.2f}")

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final Eval on Test
    log.info("Final evaluation on 2024 Test set...")
    model.load_state_dict(torch.load(OUT_DIR / f"{TARGET}_cnnlstm_best.pt"))
    model.eval()
    all_preds = []
    all_truth = []
    with torch.no_grad():
        for p, t, target in test_loader:
            p, t, target = p.to(device), t.to(device), target.to(device)
            output = model(p, t)
            all_preds.extend(output.cpu().numpy())
            all_truth.extend(target.cpu().numpy())
    
    r2 = r2_score(all_truth, all_preds)
    rmse = np.sqrt(mean_squared_error(all_truth, all_preds))
    log.info(f"COMBINED TEST RESULTS [{TARGET}]: R2={r2:.4f}  RMSE={rmse:.2f}")
    
    # Save report
    with open(OUT_DIR / "exp14b_report.txt", "w") as f:
        f.write(f"EXPERIMENT 14b: Refined CNN-LSTM ({TARGET} only)\n")
        f.write("="*40 + "\n")
        f.write(f"Lookback: {args.lookback}\n")
        f.write(f"R2:   {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")

if __name__ == "__main__":
    main()
