"""
scripts/12_train_exp10_xgboost.py
==================================
Experiment 10: XGBoost + Optuna + Extended Lags

Completes the GBM trifecta comparison:
- Base Model: XGBoost Regressor.
- Tuning: Optuna (50 trials).
- Features: Extended lags from Exp 4 (our best feature set).
- GPU: Accelerated training on RTX 4050.

Usage:
  conda activate aqi-ml
  python scripts/12_train_exp10_xgboost.py --n-trials 50
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Project Root Setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp10")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

# Features from Exp 4 (Extended Lags)
BASE_FEATURE_COLS = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]
LAG_ORIG = ["PM25_lag1", "PM10_lag1", "PM25_roll3", "PM10_roll3", "PM25_roll7", "PM10_roll7", "AOD_roll3", "NO2_roll3"]
LAG_NEW  = ["PM25_lag2", "PM10_lag2", "PM25_lag3", "PM10_lag3", "PM25_roll14", "PM10_roll14", "AOD_roll7", "NO2_roll7", "PM25_ewm7", "PM10_ewm7", "WindSpeed_roll3", "T2M_roll3"]
STATION_COL = "Station_enc"

FEATURES = BASE_FEATURE_COLS + LAG_ORIG + LAG_NEW + [STATION_COL]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Data Loading & Prep
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame):
    df = df.sort_values(["StationName", "Date"]).copy()
    
    # 1. 1-3 Day Lags
    for target in ["PM25", "PM10"]:
        df[f"{target}_lag1"] = df.groupby("StationName")[target].shift(1)
        df[f"{target}_lag2"] = df.groupby("StationName")[target].shift(2)
        df[f"{target}_lag3"] = df.groupby("StationName")[target].shift(3)
        
    # 2. Rolling Statistics
    for target in ["PM25", "PM10"]:
        df[f"{target}_roll3"] = df.groupby("StationName")[target].shift(1).rolling(3, min_periods=2).mean()
        df[f"{target}_roll7"] = df.groupby("StationName")[target].shift(1).rolling(7, min_periods=3).mean()
        df[f"{target}_roll14"] = df.groupby("StationName")[target].shift(1).rolling(14, min_periods=5).mean()
        
    # 3. Satellite/Met Persistence
    df["AOD_roll3"] = df.groupby("StationName")["AOD"].shift(1).rolling(3, min_periods=2).mean()
    df["NO2_roll3"] = df.groupby("StationName")["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean()
    df["AOD_roll7"] = df.groupby("StationName")["AOD"].shift(1).rolling(7, min_periods=3).mean()
    df["NO2_roll7"] = df.groupby("StationName")["NO2_ugm3"].shift(1).rolling(7, min_periods=3).mean()
    
    # 4. EWM
    df["PM25_ewm7"] = df.groupby("StationName")["PM25"].shift(1).ewm(span=7, min_periods=3).mean()
    df["PM10_ewm7"] = df.groupby("StationName")["PM10"].shift(1).ewm(span=7, min_periods=3).mean()
    
    # 5. Met roll
    df["WindSpeed_roll3"] = df.groupby("StationName")["WindSpeed"].shift(1).rolling(3, min_periods=2).mean()
    df["T2M_roll3"] = df.groupby("StationName")["T2M_C"].shift(1).rolling(3, min_periods=2).mean()
    
    # Encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[STATION_COL] = le.fit_transform(df["StationName"])
    
    return df, le

def load_data():
    log.info("Loading training data...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet", engine='fastparquet')
    full["Date"] = pd.to_datetime(full["Date"])
    
    full, le = engineer_features(full)
    
    train = full[full["Date"] <= "2023-06-30"].copy()
    val   = full[(full["Date"] > "2023-06-30") & (full["Date"] <= "2023-12-31")].copy()
    test  = full[full["Date"] > "2023-12-31"].copy()
    
    return train, val, test, le

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def train_target(target, train, val, test, n_trials):
    log.info("-" * 40)
    log.info(f"TRAINING TARGET: {target}")
    log.info("-" * 40)
    
    # Cleanup NAs
    train = train.dropna(subset=[target] + FEATURES)
    val   = val.dropna(subset=[target] + FEATURES)
    test  = test.dropna(subset=[target] + FEATURES)
    
    X_train, y_train = train[FEATURES], train[target]
    X_val,   y_val   = val[FEATURES], val[target]
    
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "n_estimators": 2000,
            "early_stopping_rounds": 50,
            "verbosity": 0
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Final Model
    log.info(f"Best trial parameters: {study.best_params}")
    final_params = {
        **study.best_params,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": 5000,
        "early_stopping_rounds": 50,
        "verbosity": 0
    }
    
    model = xgb.XGBRegressor(**final_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluation
    X_test, y_test = test[FEATURES], test[target]
    y_pred = model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred)
    }
    
    test_res = test[["Date", "StationName", target]].copy()
    test_res[f"{target}_pred"] = y_pred
    
    return model, metrics, test_res

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    train, val, test, le = load_data()
    joblib.dump(le, OUT_DIR / "station_encoder.pkl")
    
    report = [f"EXP 10 XGBOOST REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
              "Approach: XGBoost + Extended Lags (Exp4 features)", f"Trials: {args.n_trials}", "-"*50]
    
    for target in TARGETS:
        model, metrics, test_res = train_target(target, train, val, test, args.n_trials)
        model.save_model(str(OUT_DIR / f"{target}_xgboost.json"))
        
        report.append(f"\nTARGET: {target}")
        report.append(f"  Test R2:   {metrics['R2']:.4f}")
        report.append(f"  Test RMSE: {metrics['RMSE']:.2f}")
        report.append(f"  Test MAE:  {metrics['MAE']:.2f}")
        
        # Plot
        plt.figure(figsize=(8,8))
        plt.scatter(test_res[target], test_res[f"{target}_pred"], alpha=0.3)
        plt.plot([test_res[target].min(), test_res[target].max()], [test_res[target].min(), test_res[target].max()], 'r--')
        plt.title(f"XGBoost {target} - R2: {metrics['R2']:.3f}")
        plt.savefig(PLOTS_DIR / f"{target}_scatter.png")
        plt.close()
        
    with open(OUT_DIR / "exp10_report.txt", "w") as f:
        f.write("\n".join(report))
        
    log.info("Exp 10 finished. Results in %s", OUT_DIR)

if __name__ == "__main__":
    main()
