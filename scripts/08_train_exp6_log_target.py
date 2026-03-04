"""
scripts/08_train_exp6_log_target.py
====================================
Experiment 6: LightGBM + Log-Transform Target + Extended Lags + Optuna

Improves on Exp4 by addressing target skewness:
- Transformation: y_log = log1p(y)
- Optimization: RMSE on log scale
- Evaluation: Back-transform with expm1(y_pred) for R2, RMSE, MAE on raw scale

Usage:
  conda activate aqi-ml
  python scripts/08_train_exp6_log_target.py --n-trials 50
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
import lightgbm as lgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Project Root Setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp6")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

# Re-use Exp4 feature definitions for consistency
BASE_FEATURES = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

# Lag Specs (Matching Exp4/Exp5)
LAG_SPECS = [("PM25", 1), ("PM10", 1), ("PM25", 2), ("PM10", 2), ("PM25", 3), ("PM10", 3)]
ROLL_SPECS = [
    ("PM25", 3), ("PM10", 3), ("PM25", 7), ("PM10", 7), ("PM25", 14), ("PM10", 14),
    ("AOD", 3), ("NO2_ugm3", 3), ("AOD", 7), ("NO2_ugm3", 7),
    ("WindSpeed", 3), ("T2M_C", 3)
]
EWM_SPECS = [("PM25", 7), ("PM10", 7)]

STATION_COL = "Station_enc"

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
# Data Engineering (Modular & Clean)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame):
    df = df.sort_values(["StationName", "Date"]).copy()
    lag_cols = []
    
    for station, group in df.groupby("StationName", sort=False):
        idx = group.index
        for target, n in LAG_SPECS:
            col = f"{target}_lag{n}"
            df.loc[idx, col] = group[target].shift(n)
            if col not in lag_cols: lag_cols.append(col)
        for target, w in ROLL_SPECS:
            col = f"{target}_roll{w}"
            df.loc[idx, col] = group[target].shift(1).rolling(w, min_periods=max(2, w//3)).mean()
            if col not in lag_cols: lag_cols.append(col)
        for target, s in EWM_SPECS:
            col = f"{target}_ewm{s}"
            df.loc[idx, col] = group[target].shift(1).ewm(span=s, min_periods=3).mean()
            if col not in lag_cols: lag_cols.append(col)
            
    return df, lag_cols

def load_data():
    log.info("Loading training data...")
    # Use fastparquet as it was confirmed working in Exp5
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet", engine='fastparquet')
    full["Date"] = pd.to_datetime(full["Date"])
    
    full, lag_cols = engineer_features(full)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    full[STATION_COL] = le.fit_transform(full["StationName"])
    
    all_features = BASE_FEATURES + lag_cols + [STATION_COL]
    
    train = full[full["Date"] <= "2023-06-30"].copy()
    val   = full[(full["Date"] > "2023-06-30") & (full["Date"] <= "2023-12-31")].copy()
    test  = full[full["Date"] > "2023-12-31"].copy()
    
    return train, val, test, all_features, le

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def calculate_metrics_raw(y_true_raw, y_pred_log):
    """Back-transforms predictions and calculates metrics on original scale."""
    y_pred_raw = np.expm1(y_pred_log)
    return {
        "R2": r2_score(y_true_raw, y_pred_raw),
        "RMSE": np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)),
        "MAE": mean_absolute_error(y_true_raw, y_pred_raw)
    }

def train_target(target, train, val, test, feature_cols, n_trials):
    log.info("-" * 40)
    log.info(f"TARGET: {target} (Log-Transform)")
    log.info("-" * 40)
    
    # Drop rows where target is NaN
    train = train.dropna(subset=[target])
    val   = val.dropna(subset=[target])
    test  = test.dropna(subset=[target])
    
    # Transform Target
    y_train_raw = train[target].values
    y_val_raw   = val[target].values
    y_test_raw  = test[target].values
    
    y_train_log = np.log1p(y_train_raw)
    y_val_log   = np.log1p(y_val_raw)
    
    X_train, X_val, X_test = train[feature_cols], val[feature_cols], test[feature_cols]
    
    # Optuna Study
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }
        
        dtrain = lgb.Dataset(X_train, label=y_train_log)
        dval   = lgb.Dataset(X_val,   label=y_val_log, reference=dtrain)
        
        model = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds_log = model.predict(X_val)
        # Minimize RMSE on log scale (standard for log-transformed targets)
        return np.sqrt(mean_squared_error(y_val_log, preds_log))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    log.info(f"Best log-RMSE: {study.best_value:.4f}")
    
    # Retrain final model with best params
    best_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": 42,
        **study.best_params
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train_log)
    dval   = lgb.Dataset(X_val,   label=y_val_log, reference=dtrain)
    
    final_model = lgb.train(
        best_params, dtrain, num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Evaluate
    metrics = {
        "Train": calculate_metrics_raw(y_train_raw, final_model.predict(X_train)),
        "Val":   calculate_metrics_raw(y_val_raw,   final_model.predict(X_val)),
        "Test":  calculate_metrics_raw(y_test_raw,  final_model.predict(X_test))
    }
    
    # Save predictions
    test_preds_raw = np.expm1(final_model.predict(X_test))
    test_res = test[["Date", "StationName", target]].copy()
    test_res[f"{target}_pred"] = test_preds_raw
    
    return final_model, metrics, test_res

# ---------------------------------------------------------------------------
# Reporting & Visuals
# ---------------------------------------------------------------------------

def plot_results(target, test_df):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    y_true = test_df[target]
    y_pred = test_df[f"{target}_pred"]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel(f"Observed {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Exp 6 - {target} (Log-Transform)\nR2: {r2_score(y_true, y_pred):.3f}")
    plt.savefig(PLOTS_DIR / f"{target}_scatter_test.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train, val, test, feature_cols, le = load_data()
    joblib.dump(le, OUT_DIR / "station_encoder.pkl")
    
    report = [f"EXP 6 REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
              f"Approach: LightGBM + log1p(target)", f"Trials: {args.n_trials}", "-"*50]
    
    for target in TARGETS:
        model, metrics, test_res = train_target(target, train, val, test, feature_cols, args.n_trials)
        
        # Save model & predictions
        model.save_model(str(OUT_DIR / f"{target}_lgbm.txt"))
        test_res.to_csv(OUT_DIR / f"{target}_test_preds.csv", index=False)
        
        # Plot
        plot_results(target, test_res)
        
        # Report
        report.append(f"\nTARGET: {target}")
        for split, m in metrics.items():
            report.append(f"  {split:5}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}")
            
    with open(OUT_DIR / "exp6_report.txt", "w") as f:
        f.write("\n".join(report))
        
    log.info("Exp 6 finished. Results in %s", OUT_DIR)

if __name__ == "__main__":
    main()
