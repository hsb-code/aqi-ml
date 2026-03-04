"""
scripts/10_train_exp8_lstm.py
==============================
Experiment 8: Many-to-One LSTM for AQI Prediction

Uses PyTorch to train a Recurrent Neural Network (LSTM):
- Input: Sequence of historical features (e.g., last 7 days)
- Target: Next day PM2.5 and PM10
- Scaling: StandardScaler for features and targets
- GPU: Fully supported

Usage:
  conda activate aqi-ml
  python scripts/10_train_exp8_lstm.py --epochs 50 --n-trials 20
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Project Root Setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp8")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

# We use the raw features (no manual lags needed, LSTM handles history)
BASE_FEATURES = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Preparation (LSTM Sequences)
# ---------------------------------------------------------------------------

def create_sequences(df, features, targets, seq_length=7):
    """
    Creates (N, seq_length, n_features) and (N, n_targets) arrays.
    Handles gaps by sliding within each StationName group.
    """
    X, y = [], []
    for station, group in df.groupby("StationName"):
        # Ensure dates are continuous and sorted
        group = group.sort_values("Date")
        features_data = group[features].values
        targets_data = group[targets].values
        
        for i in range(len(group) - seq_length):
            X.append(features_data[i:i+seq_length])
            y.append(targets_data[i+seq_length])
            
    return np.array(X), np.array(y)

def load_and_scale_data(seq_length=7):
    log.info("Loading data from parquet...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet", engine='fastparquet')
    full["Date"] = pd.to_datetime(full["Date"])
    
    # Fill missing values (simple forward fill within station)
    full = full.sort_values(["StationName", "Date"])
    # Standard way to ffill within groups and keep columns
    full = full.groupby("StationName", group_keys=False).apply(lambda x: x.ffill().bfill())
    full = full.reset_index(drop=True)
    
    # Identify splits
    train_raw = full[full["Date"] <= "2023-06-30"].copy()
    val_raw   = full[(full["Date"] > "2023-06-30") & (full["Date"] <= "2023-12-31")].copy()
    test_raw  = full[full["Date"] > "2023-12-31"].copy()
    
    # Scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    train_raw[BASE_FEATURES] = scaler_x.fit_transform(train_raw[BASE_FEATURES])
    val_raw[BASE_FEATURES]   = scaler_x.transform(val_raw[BASE_FEATURES])
    test_raw[BASE_FEATURES]  = scaler_x.transform(test_raw[BASE_FEATURES])
    
    train_raw[TARGETS] = scaler_y.fit_transform(train_raw[TARGETS])
    val_raw[TARGETS]   = scaler_y.transform(val_raw[TARGETS])
    test_raw[TARGETS]  = scaler_y.transform(test_raw[TARGETS])
    
    # Create Sequences
    X_train, y_train = create_sequences(train_raw, BASE_FEATURES, TARGETS, seq_length)
    X_val,   y_val   = create_sequences(val_raw,   BASE_FEATURES, TARGETS, seq_length)
    X_test,  y_test  = create_sequences(test_raw,  BASE_FEATURES, TARGETS, seq_length)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y

# ---------------------------------------------------------------------------
# LSTM Model Definition
# ---------------------------------------------------------------------------

class AQILSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(AQILSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # We take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                preds = model(b_x)
                val_loss += criterion(preds, b_y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            
    model.load_state_dict(best_model_state)
    return best_val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=7)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y = load_and_scale_data(args.seq_length)
    
    # Save scalers
    joblib.dump(scaler_x, OUT_DIR / "scaler_x.pkl")
    joblib.dump(scaler_y, OUT_DIR / "scaler_y.pkl")
    
    # Convert to tensors
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),   torch.tensor(y_val, dtype=torch.float32)),   batch_size=64, shuffle=False)
    
    # Optuna
    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout    = trial.suggest_float("dropout", 0.0, 0.4)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        model = AQILSTM(len(BASE_FEATURES), hidden_dim, num_layers, len(TARGETS), dropout).to(device)
        return train_model(model, train_loader, val_loader, epochs=args.epochs, lr=lr, device=device)

    log.info(f"Starting Optuna search ({args.n_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    
    log.info(f"Best trial parameters: {study.best_params}")
    
    # Final Model
    log.info("Training final model with best parameters...")
    final_model = AQILSTM(len(BASE_FEATURES), study.best_params["hidden_dim"], study.best_params["num_layers"], len(TARGETS), study.best_params["dropout"]).to(device)
    train_model(final_model, train_loader, val_loader, epochs=args.epochs, lr=study.best_params["lr"], device=device)
    
    torch.save(final_model.state_dict(), OUT_DIR / "lstm_model.pt")
    
    # Evaluation
    final_model.eval()
    with torch.no_grad():
        test_preds_scaled = final_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        test_preds = scaler_y.inverse_transform(test_preds_scaled)
        y_test_raw = scaler_y.inverse_transform(y_test)
        
    # Metrics
    report = [f"EXP 8 LSTM REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
              f"Sequence Length: {args.seq_length}", f"Best Params: {study.best_params}", "-"*50]
    
    for i, target in enumerate(TARGETS):
        r2 = r2_score(y_test_raw[:, i], test_preds[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_raw[:, i], test_preds[:, i]))
        mae = mean_absolute_error(y_test_raw[:, i], test_preds[:, i])
        
        report.append(f"\nTARGET: {target}")
        report.append(f"  Test R2:   {r2:.4f}")
        report.append(f"  Test RMSE: {rmse:.2f}")
        report.append(f"  Test MAE:  {mae:.2f}")
        
        # Plot
        plt.figure(figsize=(8,8))
        plt.scatter(y_test_raw[:, i], test_preds[:, i], alpha=0.3)
        plt.plot([y_test_raw[:, i].min(), y_test_raw[:, i].max()], [y_test_raw[:, i].min(), y_test_raw[:, i].max()], 'r--')
        plt.title(f"LSTM {target} - R2: {r2:.3f}")
        plt.savefig(PLOTS_DIR / f"{target}_scatter.png")
        plt.close()
        
    with open(OUT_DIR / "exp8_report.txt", "w") as f:
        f.write("\n".join(report))
        
    log.info("Exp 8 finished. Results in %s", OUT_DIR)

if __name__ == "__main__":
    main()
