"""
scripts/25_train_exp22_regularized_mixed.py
===========================================
Experiment 22: Regularized Mixed Split Evaluation
Based on Exp 20, but with STRICT regularization to prevent overfitting.

Modifications:
- High L1 (lambda_l1): [1.0, 100.0]
- High L2 (lambda_l2): [1.0, 100.0]
- Low Learning Rate: [0.001, 0.05]
- Forced broader nodes (min_data_in_leaf): [50, 500]
- Min gain to split: [0.1, 5.0]
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp22")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

BASE_FEATURE_COLS = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

ALL_LAG_COLS = [
    "PM25_lag1", "PM10_lag1", "PM25_roll3", "PM10_roll3", "PM25_roll7", "PM10_roll7", 
    "AOD_roll3", "NO2_roll3", "PM25_lag2", "PM10_lag2", "PM25_lag3", "PM10_lag3",
    "PM25_roll14", "PM10_roll14", "AOD_roll7", "NO2_roll7", "PM25_ewm7", "PM10_ewm7",
    "WindSpeed_roll3", "T2M_roll3"
]

STATION_COL = "Station_enc"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering (Identical to Exp 20)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["StationName", "Date"]).copy()
    for station, grp in df.groupby("StationName", sort=False):
        idx = grp.index
        df.loc[idx, "PM25_lag1"]   = grp["PM25"].shift(1)
        df.loc[idx, "PM10_lag1"]   = grp["PM10"].shift(1)
        df.loc[idx, "PM25_roll3"]  = grp["PM25"].shift(1).rolling(3,  min_periods=2).mean()
        df.loc[idx, "PM10_roll3"]  = grp["PM10"].shift(1).rolling(3,  min_periods=2).mean()
        df.loc[idx, "PM25_roll7"]  = grp["PM25"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "PM10_roll7"]  = grp["PM10"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "AOD_roll3"]   = grp["AOD"].shift(1).rolling(3,   min_periods=2).mean()
        df.loc[idx, "NO2_roll3"]   = grp["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "PM25_lag2"]   = grp["PM25"].shift(2)
        df.loc[idx, "PM10_lag2"]   = grp["PM10"].shift(2)
        df.loc[idx, "PM25_lag3"]   = grp["PM25"].shift(3)
        df.loc[idx, "PM10_lag3"]   = grp["PM10"].shift(3)
        df.loc[idx, "PM25_roll14"] = grp["PM25"].shift(1).rolling(14, min_periods=5).mean()
        df.loc[idx, "PM10_roll14"] = grp["PM10"].shift(1).rolling(14, min_periods=5).mean()
        df.loc[idx, "AOD_roll7"]   = grp["AOD"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "NO2_roll7"]   = grp["NO2_ugm3"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "PM25_ewm7"]   = grp["PM25"].shift(1).ewm(span=7, min_periods=3).mean()
        df.loc[idx, "PM10_ewm7"]   = grp["PM10"].shift(1).ewm(span=7, min_periods=3).mean()
        df.loc[idx, "WindSpeed_roll3"] = grp["WindSpeed"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "T2M_roll3"]       = grp["T2M_C"].shift(1).rolling(3, min_periods=2).mean()
    return df

def encode_station(full: pd.DataFrame):
    le = LabelEncoder()
    full = full.copy()
    full[STATION_COL] = le.fit_transform(full["StationName"])
    return full, le

def prepare_xy(df: pd.DataFrame, target: str, feature_cols: list):
    df = df.dropna(subset=[target] + feature_cols).copy()
    X = df[feature_cols].copy()
    y = df[target].values
    return X, y, df

# ---------------------------------------------------------------------------
# Optuna objective with STRICT Regularization
# ---------------------------------------------------------------------------

def make_objective(X_train, y_train, X_val, y_val, cat_features):
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, categorical_feature=cat_features, free_raw_data=False)
    
    def objective(trial):
        params = {
            "objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500), # Increased for stability
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            # STRICT REGULARIZATION
            "lambda_l1": trial.suggest_float("lambda_l1", 1.0, 100.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 100.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.1, 5.0),
        }
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)] # Higher patience for lower LR
        model = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval], callbacks=callbacks)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    return objective

def compute_metrics(y_true, y_pred, split_name: str) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    log.info("  %-12s R\u00b2=%.4f | RMSE=%.2f | MAE=%.2f", split_name, r2, rmse, mae)
    return {"split": split_name, "R2": r2, "RMSE": rmse, "MAE": mae}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading full dataset...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet")
    
    log.info("Engineering lag features...")
    full = add_lag_features(full)
    
    log.info("Encoding station names...")
    full, station_le = encode_station(full)
    joblib.dump(station_le, OUT_DIR / "station_label_encoder.pkl")

    all_features = BASE_FEATURE_COLS + ALL_LAG_COLS + [STATION_COL]
    cat_features = [STATION_COL, "Season"]

    report = [f"EXPERIMENT 22: Regularized Mixed Split Evaluation\n", "="*60 + "\n"]
    report.append(f"Strategy: Strict L1/L2, Lower LR, Forced broader nodes\n")
    report.append(f"Optuna trials: {args.n_trials}\n\n")

    for target in TARGETS:
        log.info(f"===== Targeting {target} =====")
        X, y, df_clean = prepare_xy(full, target, all_features)
        
        # Mixed Split: 80% Train, 10% Val, 10% Test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
        
        log.info(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(X_train, y_train, X_val, y_val, cat_features), n_trials=args.n_trials)
        
        best_params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42, **study.best_params}
        
        # Get optimal rounds via ES
        dtrain_es = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        dval_es   = lgb.Dataset(X_val, label=y_val, reference=dtrain_es, categorical_feature=cat_features)
        model_es  = lgb.train(best_params, dtrain_es, num_boost_round=3000, valid_sets=[dval_es], 
                              callbacks=[lgb.early_stopping(100, verbose=False)])
        
        # Retrain on Train+Val
        X_tv = pd.concat([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])
        dtrain_full = lgb.Dataset(X_tv, label=y_tv, categorical_feature=cat_features)
        model = lgb.train(best_params, dtrain_full, num_boost_round=int(model_es.best_iteration * 1.1))
        
        # Save model
        model.save_model(str(OUT_DIR / f"{target}_lgbm_regularized.txt"))
        
        # Eval
        log.info("--- Metrics ---")
        tr_m = compute_metrics(y_train, model.predict(X_train), "Train")
        vl_m = compute_metrics(y_val,   model.predict(X_val),   "Val")
        ts_m = compute_metrics(y_test,  model.predict(X_test),  "Test")
        
        # Combined Holdout
        X_vt = pd.concat([X_val, X_test])
        y_vt = np.concatenate([y_val, y_test])
        vt_m = compute_metrics(y_vt, model.predict(X_vt), "Val+Test")
        
        report.append(f"TARGET: {target}\n" + "-"*30 + "\n")
        report.append(f"  Train:     R2={tr_m['R2']:.4f}  RMSE={tr_m['RMSE']:.2f}\n")
        report.append(f"  Val:       R2={vl_m['R2']:.4f}  RMSE={vl_m['RMSE']:.2f}\n")
        report.append(f"  Test:      R2={ts_m['R2']:.4f}  RMSE={ts_m['RMSE']:.2f}\n")
        report.append(f"  Val+Test:  R2={vt_m['R2']:.4f}  RMSE={vt_m['RMSE']:.2f} (REGULARIZED MIXED HOLDOUT)\n\n")

    with open(OUT_DIR / "exp22_report.txt", "w") as f:
        f.writelines(report)
    
    log.info(f"Experiment 22 complete. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
