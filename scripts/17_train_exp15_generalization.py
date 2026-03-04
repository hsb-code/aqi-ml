"""
scripts/17_train_exp15_generalization.py
======================================
Experiment 15: Generalization & Physics-Informed Features
Based on Exp 11, but evaluates new features and ablation of Station_enc.

New Features:
- Physical: Wind_U, Wind_V, VentilationIndex, StabilityIndex
- LUR/Proxies: Dist_Corniche_km, Dist_E11_km
- Goal: Maintain R2 > 0.55 WITHOUT using Station_enc.
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp15")
TARGETS       = ["PM25", "PM10"]

TRAIN_END = "2023-06-30"
VAL_END   = "2023-12-31"

BASE_FEATURE_COLS = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "Elevation_m", "Dist_Coast_km", "Dist_Corniche_km", "Dist_E11_km",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
    "Wind_U", "Wind_V", "VentilationIndex", "StabilityIndex"
]

LAG_COLS = [
    "PM25_lag1", "PM25_roll3", "PM25_roll7", "PM25_lag2", "PM25_lag3", "PM25_roll14", "PM25_ewm7",
    "PM10_lag1", "PM10_roll3", "PM10_roll7", "PM10_lag2", "PM10_lag3", "PM10_roll14", "PM10_ewm7",
    "AOD_roll3", "NO2_roll3", "AOD_roll7", "NO2_roll7",
    "WindSpeed_roll3", "T2M_roll3"
]

STATION_COL = "Station_enc"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["StationName", "Date"]).copy()
    for station, grp in df.groupby("StationName", sort=False):
        idx = grp.index
        # PM25
        df.loc[idx, "PM25_lag1"]   = grp["PM25"].shift(1)
        df.loc[idx, "PM25_roll3"]  = grp["PM25"].shift(1).rolling(3,  min_periods=2).mean()
        df.loc[idx, "PM25_roll7"]  = grp["PM25"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "PM25_lag2"]   = grp["PM25"].shift(2)
        df.loc[idx, "PM25_lag3"]   = grp["PM25"].shift(3)
        df.loc[idx, "PM25_roll14"] = grp["PM25"].shift(1).rolling(14, min_periods=5).mean()
        df.loc[idx, "PM25_ewm7"]   = grp["PM25"].shift(1).ewm(span=7, min_periods=3).mean()
        
        # PM10
        df.loc[idx, "PM10_lag1"]   = grp["PM10"].shift(1)
        df.loc[idx, "PM10_roll3"]  = grp["PM10"].shift(1).rolling(3,  min_periods=2).mean()
        df.loc[idx, "PM10_roll7"]  = grp["PM10"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "PM10_lag2"]   = grp["PM10"].shift(2)
        df.loc[idx, "PM10_lag3"]   = grp["PM10"].shift(3)
        df.loc[idx, "PM10_roll14"] = grp["PM10"].shift(1).rolling(14, min_periods=5).mean()
        df.loc[idx, "PM10_ewm7"]   = grp["PM10"].shift(1).ewm(span=7, min_periods=3).mean()

        # Shared
        df.loc[idx, "AOD_roll3"]   = grp["AOD"].shift(1).rolling(3,   min_periods=2).mean()
        df.loc[idx, "NO2_roll3"]   = grp["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "AOD_roll7"]   = grp["AOD"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "NO2_roll7"]   = grp["NO2_ugm3"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "WindSpeed_roll3"] = grp["WindSpeed"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "T2M_roll3"]       = grp["T2M_C"].shift(1).rolling(3, min_periods=2).mean()
    return df

def prepare_xy(df: pd.DataFrame, target: str, feature_cols: list):
    df = df.dropna(subset=[target] + feature_cols).copy()
    X = df[feature_cols].copy()
    y = df[target].values
    return X, y, df

def make_objective(X_train, y_train, X_val, y_val, cat_features):
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, categorical_feature=cat_features, free_raw_data=False)
    def objective(trial):
        params = {
            "objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 1.0, log=True),
        }
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], callbacks=callbacks)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    return objective

def run_experiment(name, train, val, test, features, cat_features, args):
    log.info(f"--- Running Training: {name} ---")
    target = "PM25"
    X_train, y_train, _ = prepare_xy(train, target, features)
    X_val, y_val, _     = prepare_xy(val,   target, features)
    X_test, y_test, _   = prepare_xy(test,  target, features)

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(X_train, y_train, X_val, y_val, cat_features), n_trials=args.n_trials)
    
    best_params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42, **study.best_params}
    dtrain_full = lgb.Dataset(pd.concat([X_train, X_val]), label=np.concatenate([y_train, y_val]), categorical_feature=cat_features)
    
    # Get optimal rounds via ES
    model_es = lgb.train(best_params, lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features), 
                         num_boost_round=1000, valid_sets=[lgb.Dataset(X_val, label=y_val)], 
                         callbacks=[lgb.early_stopping(50, verbose=False)])
    
    model = lgb.train(best_params, dtrain_full, num_boost_round=int(model_es.best_iteration * 1.1))
    
    # Save model
    model_path = OUT_DIR / f"{target}_{name.lower().replace(' ', '_')}.txt"
    model.save_model(str(model_path))
    log.info(f"Model saved: {model_path}")

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    log.info(f"[{name}] Test R2: {r2:.4f} | RMSE: {rmse:.2f}")
    
    return r2, rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet")
    full = add_lag_features(full)
    
    le = LabelEncoder()
    full[STATION_COL] = le.fit_transform(full["StationName"])
    
    train, val, test = full[full["Date"] <= TRAIN_END], \
                       full[(full["Date"] > TRAIN_END) & (full["Date"] <= VAL_END)], \
                       full[full["Date"] > VAL_END]

    results = []
    for target in TARGETS:
        log.info(f"===== Target: {target} =====")
        # 1. Baseline + Station Enc
        f1 = BASE_FEATURE_COLS + LAG_COLS + [STATION_COL]
        # Filter lags for target
        target_lags = [c for c in LAG_COLS if target in c or not any(t in c for t in TARGETS)]
        f1_actual = BASE_FEATURE_COLS + target_lags + [STATION_COL]
        
        c1 = [STATION_COL, "Season"]
        r2_1, rmse_1 = run_experiment(f"{target} with Station ID", train, val, test, f1_actual, c1, args)
        results.append(f"{target} With Station ID: R2={r2_1:.4f}, RMSE={rmse_1:.2f}")

        # 2. Advanced Features WITHOUT Station Enc
        f2_actual = BASE_FEATURE_COLS + target_lags
        c2 = ["Season"]
        r2_2, rmse_2 = run_experiment(f"{target} Physics-Only", train, val, test, f2_actual, c2, args)
        results.append(f"{target} Physics-Only:   R2={r2_2:.4f}, RMSE={rmse_2:.2f}")
        results.append("-" * 30)

    with open(OUT_DIR / "exp15_report.txt", "w") as f:
        f.write("EXPERIMENT 15: Generalization and Physics-Informed Features\n")
        f.write("="*60 + "\n")
        for res in results:
            f.write(res + "\n")
    
    log.info(f"Report saved to {OUT_DIR}/exp15_report.txt")

if __name__ == "__main__":
    main()
