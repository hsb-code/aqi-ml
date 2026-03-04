"""
scripts/13_train_exp11_geography.py
======================================
Experiment 11: LightGBM + Static Geography + Extended Lag Features + Optuna
Based on Exp 4, but adds newly engineered spatial content:
- Elevation_m
- Dist_Coast_km

Goal: Breaks the 0.60 ceiling by helping the model understand station topography.
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
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp11")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

TRAIN_END = "2023-06-30"
VAL_END   = "2023-12-31"

BASE_FEATURE_COLS = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "Elevation_m", "Dist_Coast_km", # NEW IN EXP 11
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

# Lag features (from Exp 4)
LAG_COLS = [
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
# Helpers
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

def temporal_split(df: pd.DataFrame):
    train = df[df["Date"] <= TRAIN_END]
    val   = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)]
    test  = df[df["Date"] > VAL_END]
    return train, val, test

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
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "feature_pre_filter": False,
        }
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dval], callbacks=callbacks)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    return objective

def plot_feature_importance(model, target, out_dir):
    importances = pd.Series(model.feature_importance(importance_type="gain"), index=model.feature_name()).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = []
    for n in importances.index:
        if n in ["Elevation_m", "Dist_Coast_km"]: colors.append("#d62728") # Red for NEW geography
        elif n == STATION_COL: colors.append("#2d8b2d")
        elif any(x in n for x in ["lag", "roll", "ewm"]): colors.append("#e07b00")
        else: colors.append("#1a6fa8")
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Feature Importance (gain) — {target} [Exp 11]\nred=NEW geography  blue=base  orange=temporal")
    plt.tight_layout()
    plt.savefig(out_dir / f"{target}_feature_importance.png", dpi=150)
    plt.close()

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--n-trials", type=int, default=50)
    args = args_parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading training data...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet")
    log.info(f"Full dataset: {len(full):,} rows")

    log.info("Engineering lag features...")
    full = add_lag_features(full)
    
    log.info("Encoding stations...")
    full, station_le = encode_station(full)
    joblib.dump(station_le, OUT_DIR / "station_label_encoder.pkl")

    train_all, val_all, test_all = temporal_split(full)
    all_feature_cols = BASE_FEATURE_COLS + LAG_COLS + [STATION_COL]
    cat_features = [STATION_COL, "Season"]
    
    report = [f"EXPERIMENT 11: LightGBM + Static Geography\n", "="*50 + "\n"]

    for target in TARGETS:
        log.info(f"Targeting {target}...")
        X_train, y_train, _ = prepare_xy(train_all, target, all_feature_cols)
        X_val, y_val, _     = prepare_xy(val_all,   target, all_feature_cols)
        X_test, y_test, _   = prepare_xy(test_all,  target, all_feature_cols)

        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(X_train, y_train, X_val, y_val, cat_features), n_trials=args.n_trials)
        
        log.info(f"Best RMSE for {target}: {study.best_value:.4f}")
        
        # Retrain on train+val
        best_params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42, **study.best_params}
        X_tv = pd.concat([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])
        dtrain_full = lgb.Dataset(X_tv, label=y_tv, categorical_feature=cat_features)
        
        # Final fine-tuning round for iterations
        dtrain_es = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        dval_es   = lgb.Dataset(X_val, label=y_val, reference=dtrain_es, categorical_feature=cat_features)
        model_es  = lgb.train(best_params, dtrain_es, num_boost_round=1500, valid_sets=[dval_es], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        model = lgb.train(best_params, dtrain_full, num_boost_round=int(model_es.best_iteration * 1.1))
        
        # Eval
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        log.info(f"TEST RESULTS [{target}]: R2={r2:.4f}  RMSE={rmse:.2f}")
        report.append(f"TARGET: {target}\n  R2:   {r2:.4f}\n  RMSE: {rmse:.2f}\n  MAE:  {mae:.2f}\n\n")
        
        plot_feature_importance(model, target, PLOTS_DIR)
        model.save_model(str(OUT_DIR / f"{target}_lgbm.txt"))

    with open(OUT_DIR / "exp11_report.txt", "w") as f:
        f.writelines(report)
    log.info(f"Experiment complete. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
