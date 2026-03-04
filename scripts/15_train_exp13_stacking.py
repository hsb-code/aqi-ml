"""
scripts/15_train_exp13_stacking.py
======================================
Experiment 13: Advanced Stacking Regressor
Combines LightGBM, XGBoost, and CatBoost using a meta-learner.

Mechanism:
1. TimeSeriesSplit (5 folds) on train+val data.
2. Generate Out-of-Fold (OOF) predictions for each base model.
3. Train a Ridge Regression meta-model on OOF predictions.
4. Evaluate on 2024 Test set.
"""

import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp13")
TARGETS       = ["PM25", "PM10"]

TRAIN_VAL_END = "2023-12-31"

BASE_FEATURES = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "Elevation_m", "Dist_Coast_km",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

LAG_COLS = [
    "PM25_lag1", "PM10_lag1", "PM25_roll3", "PM10_roll3", "PM25_roll7", "PM10_roll7", 
    "AOD_roll3", "NO2_roll3", "PM25_lag2", "PM10_lag2", "PM25_lag3", "PM10_lag3",
    "PM25_roll14", "PM10_roll14", "AOD_roll7", "NO2_roll7", "PM25_ewm7", "PM10_ewm7",
    "WindSpeed_roll3", "T2M_roll3"
]

STATION_COL = "Station_enc"
ALL_FEATURES = BASE_FEATURES + LAG_COLS + [STATION_COL]
CAT_FEATURES = [STATION_COL, "Season"]

# Standardized params from previous best trials
LGBM_PARAMS = {
    "objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42,
    "learning_rate": 0.05, "num_leaves": 127, "max_depth": 10, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5, "feature_pre_filter": False
}

XGB_PARAMS = {
    "objective": "reg:squarederror", "tree_method": "hist", "device": "cuda",
    "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8,
    "n_estimators": 1000, "random_state": 42
}

CB_PARAMS = {
    "iterations": 1000, "learning_rate": 0.05, "depth": 8, "loss_function": "RMSE",
    "random_seed": 42, "verbose": False, "task_type": "CPU"
}

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

def train_stacker(target, train_val_df, test_df):
    log.info(f"--- Stacking for {target} ---")
    
    # Prep data
    data = train_val_df.dropna(subset=[target] + ALL_FEATURES).copy()
    test = test_df.dropna(subset=[target] + ALL_FEATURES).copy()
    
    X = data[ALL_FEATURES]
    y = data[target].values
    X_test = test[ALL_FEATURES]
    y_test = test[target].values
    
    # Sort by date for TimeSeriesSplit
    data = data.sort_values("Date")
    X = data[ALL_FEATURES]
    y = data[target].values

    # 1. Generate OOF Predictions
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros((len(X), 3)) # LGBM, XGB, CatBoost
    
    log.info("Generating Out-of-Fold predictions (5-fold TimeSeriesSplit)...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        
        # LGBM
        model_lgb = lgb.train(LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_FEATURES), num_boost_round=500)
        oof_preds[val_idx, 0] = model_lgb.predict(X_vl)
        
        # XGB
        model_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        model_xgb.fit(X_tr, y_tr)
        oof_preds[val_idx, 1] = model_xgb.predict(X_vl)
        
        # CatBoost (Note: Uses StationName raw strings)
        X_tr_cb = X_tr.copy()
        X_vl_cb = X_vl.copy()
        # CatBoost script uses raw StationName, but our ALL_FEATURES uses enc. We'll stick to enc for consistency here.
        model_cb = CatBoostRegressor(**CB_PARAMS)
        model_cb.fit(X_tr, y_tr, cat_features=CAT_FEATURES)
        oof_preds[val_idx, 2] = model_cb.predict(X_vl)
        
        log.info(f"  Fold {fold+1} complete.")

    # Only keep rows where we have OOF predictions (TSC skip first fold)
    mask = np.any(oof_preds != 0, axis=1)
    X_meta = oof_preds[mask]
    y_meta = y[mask]

    # 2. Train Base Models on FULL train+val
    log.info("Training final base models on full train+val data...")
    final_lgb = lgb.train(LGBM_PARAMS, lgb.Dataset(X, label=y, categorical_feature=CAT_FEATURES), num_boost_round=1000)
    final_xgb = xgb.XGBRegressor(**XGB_PARAMS)
    final_xgb.fit(X, y)
    final_cb = CatBoostRegressor(**CB_PARAMS)
    final_cb.fit(X, y, cat_features=CAT_FEATURES)

    # 3. Train Meta-Learner (Tier 2)
    log.info("Training meta-learner (Ridge Regression)...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta, y_meta)
    
    # 4. Final Inference on Test set
    log.info("Evaluating on 2024 Test set...")
    base_test_preds = np.column_stack([
        final_lgb.predict(X_test),
        final_xgb.predict(X_test),
        final_cb.predict(X_test)
    ])
    
    final_preds = meta_model.predict(base_test_preds)
    
    # Metrics
    r2 = r2_score(y_test, final_preds)
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    mae = mean_absolute_error(y_test, final_preds)
    
    log.info(f"RESULTS [{target}]: R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    
    # Save test results
    test_res = test[["Date", "StationName", target]].copy()
    test_res[f"{target}_pred"] = final_preds
    test_res[f"{target}_lgbm"] = base_test_preds[:, 0]
    test_res[f"{target}_xgb"]  = base_test_preds[:, 1]
    test_res[f"{target}_cb"]   = base_test_preds[:, 2]
    
    return {
        "target": target,
        "r2": r2, "rmse": rmse, "mae": mae,
        "meta_model": meta_model,
        "test_df": test_res
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info("Loading training data...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet")
    full = add_lag_features(full)
    
    le = LabelEncoder()
    full[STATION_COL] = le.fit_transform(full["StationName"])
    
    train_val = full[full["Date"] <= TRAIN_VAL_END]
    test      = full[full["Date"] > TRAIN_VAL_END]
    
    results = []
    report = [f"EXPERIMENT 13: Advanced Stacking (LGBM + XGB + CB)\n", "="*50 + "\n"]
    
    for target in TARGETS:
        res = train_stacker(target, train_val, test)
        results.append(res)
        
        report.append(f"TARGET: {target}\n")
        report.append(f"  Final Stacker R2: {res['r2']:.4f}\n")
        report.append(f"  RMSE: {res['rmse']:.2f} | MAE: {res['mae']:.2f}\n")
        
        # Meta-model coefficients
        coefs = res["meta_model"].coef_
        report.append(f"  Weights: LGBM={coefs[0]:.3f}, XGB={coefs[1]:.3f}, CatBoost={coefs[2]:.3f}\n")
        report.append("-" * 30 + "\n")
        
        res["test_df"].to_csv(OUT_DIR / f"{target}_stacking_preds.csv", index=False)
        joblib.dump(res["meta_model"], OUT_DIR / f"{target}_meta_learner.pkl")

    with open(OUT_DIR / "exp13_report.txt", "w") as f:
        f.writelines(report)
        
    log.info(f"Experiment 13 complete. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
