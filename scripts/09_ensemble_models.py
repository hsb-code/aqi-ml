"""
scripts/09_ensemble_models.py
==============================
Experiment 7: Ensembling LightGBM (Exp 4) and CatBoost (Exp 5)

This script loads the best models from Experiment 4 and Experiment 5,
generates predictions, and finds the optimal weighted average using the
Validation set to maximize R2. Final results are reported on the 2024 Test set.

Usage:
  conda activate aqi-ml
  python scripts/09_ensemble_models.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Project Root Setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration
PROCESSED_DIR = Path("data/processed")
EXP4_DIR      = Path("models/exp4")
EXP5_DIR      = Path("models/exp5")
OUT_DIR       = Path("models/exp7")
TARGETS       = ["PM25", "PM10"]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Feature Definitions (Matching Exp4/Exp5 exactly)
BASE_FEATURE_COLS = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

# Exp 4 Feature List (LightGBM)
LAG_ORIG = ["PM25_lag1", "PM10_lag1", "PM25_roll3", "PM10_roll3", "PM25_roll7", "PM10_roll7", "AOD_roll3", "NO2_roll3"]
LAG_NEW  = ["PM25_lag2", "PM10_lag2", "PM25_lag3", "PM10_lag3", "PM25_roll14", "PM10_roll14", "AOD_roll7", "NO2_roll7", "PM25_ewm7", "PM10_ewm7", "WindSpeed_roll3", "T2M_roll3"]
LGBM_FEATURES = BASE_FEATURE_COLS + LAG_ORIG + LAG_NEW + ["Station_enc"]

# Exp 5 Feature List (CatBoost)
# Produced in 07 by: [Lags 1-3] + [Rolls] + [EWM]
CB_LAGS = ["PM25_lag1", "PM10_lag1", "PM25_lag2", "PM10_lag2", "PM25_lag3", "PM10_lag3"]
CB_ROLLS = [
    "PM25_roll3", "PM10_roll3", "PM25_roll7", "PM10_roll7", "PM25_roll14", "PM10_roll14",
    "AOD_roll3", "NO2_ugm3_roll3", "AOD_roll7", "NO2_ugm3_roll7",
    "WindSpeed_roll3", "T2M_C_roll3"
]
CB_EWM = ["PM25_ewm7", "PM10_ewm7"]
CB_FEATURES = BASE_FEATURE_COLS + CB_LAGS + CB_ROLLS + CB_EWM + ["StationName"]

# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame):
    """
    Mirroring the feature engineering from Exp4/Exp5 scripts.
    Since they share the same lag/rolling specs, we calculate them once here.
    """
    df = df.sort_values(["StationName", "Date"]).copy()
    
    for station, group in df.groupby("StationName", sort=False):
        idx = group.index
        # 1-day lags
        df.loc[idx, "PM25_lag1"] = group["PM25"].shift(1)
        df.loc[idx, "PM10_lag1"] = group["PM10"].shift(1)
        # 2 and 3 day lags
        df.loc[idx, "PM25_lag2"] = group["PM25"].shift(2)
        df.loc[idx, "PM10_lag2"] = group["PM10"].shift(2)
        df.loc[idx, "PM25_lag3"] = group["PM25"].shift(3)
        df.loc[idx, "PM10_lag3"] = group["PM10"].shift(3)
        # Rolling means
        df.loc[idx, "PM25_roll3"] = group["PM25"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "PM10_roll3"] = group["PM10"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "PM25_roll7"] = group["PM25"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "PM10_roll7"] = group["PM10"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "PM25_roll14"] = group["PM25"].shift(1).rolling(14, min_periods=5).mean()
        df.loc[idx, "PM10_roll14"] = group["PM10"].shift(1).rolling(14, min_periods=5).mean()
        # Satellite rolling
        df.loc[idx, "AOD_roll3"]  = group["AOD"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "NO2_roll3"]  = group["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean() # Match LGBM naming
        df.loc[idx, "NO2_ugm3_roll3"] = group["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean() # Match CatBoost naming
        df.loc[idx, "AOD_roll7"]  = group["AOD"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "NO2_roll7"]  = group["NO2_ugm3"].shift(1).rolling(7, min_periods=3).mean() # Match LGBM
        df.loc[idx, "NO2_ugm3_roll7"] = group["NO2_ugm3"].shift(1).rolling(7, min_periods=3).mean() # Match CatBoost
        # EWM
        df.loc[idx, "PM25_ewm7"]  = group["PM25"].shift(1).ewm(span=7, min_periods=3).mean()
        df.loc[idx, "PM10_ewm7"]  = group["PM10"].shift(1).ewm(span=7, min_periods=3).mean()
        # Met roll
        df.loc[idx, "WindSpeed_roll3"] = group["WindSpeed"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "T2M_roll3"] = group["T2M_C"].shift(1).rolling(3, min_periods=2).mean() # Match LGBM
        df.loc[idx, "T2M_C_roll3"] = group["T2M_C"].shift(1).rolling(3, min_periods=2).mean() # Match CatBoost
            
    return df

def load_data():
    log.info("Loading training data (using fastparquet)...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet", engine='fastparquet')
    full["Date"] = pd.to_datetime(full["Date"])
    
    full = engineer_features(full)
    
    # LightGBM needs the station label encoding from Exp4
    log.info("Loading station encoder from Exp4...")
    le = joblib.load(EXP4_DIR / "station_label_encoder.pkl")
    full["Station_enc"] = le.transform(full["StationName"])
    
    val   = full[(full["Date"] > "2023-06-30") & (full["Date"] <= "2023-12-31")].copy()
    test  = full[full["Date"] > "2023-12-31"].copy()
    
    return val, test

# ---------------------------------------------------------------------------
# Ensembling Logic
# ---------------------------------------------------------------------------

def get_predictions(target, val_df, test_df):
    """Loads Exp4 (LGBM) and Exp5 (CatBoost) and gets predictions."""
    
    # Drop NaNs for this specific target
    val_clean = val_df.dropna(subset=[target])
    test_clean = test_df.dropna(subset=[target])
    
    # Features (Must match Exp4/Exp5 setup)
    BASE_FEATURES = [
        "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
        "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
        "DayOfYear", "Month", "Season", "IsWeekend",
        "Latitude", "Longitude",
        "NO2_log", "AOD_log", "BLH_log", "f_RH",
    ]
    lag_cols = [c for c in val_df.columns if any(x in c for x in ["lag", "roll", "ewm"])]
    
    # LGBM Features
    lgbm_features = BASE_FEATURES + lag_cols + ["Station_enc"]
    # CatBoost Features
    cat_features = BASE_FEATURES + lag_cols + ["StationName"]
    
    # Load Models
    log.info(f"Loading Models for {target}...")
    lgbm_model = lgb.Booster(model_file=str(EXP4_DIR / f"{target}_lgbm.txt"))
    cat_model  = CatBoostRegressor()
    cat_model.load_model(str(EXP5_DIR / f"{target}_catboost.cbm"))
    
    # Predict
    log.info("Generating predictions...")
    
    val_lgbm = lgbm_model.predict(val_clean[LGBM_FEATURES])
    val_cat  = cat_model.predict(val_clean[CB_FEATURES])
    
    test_lgbm = lgbm_model.predict(test_clean[LGBM_FEATURES])
    test_cat  = cat_model.predict(test_clean[CB_FEATURES])
    
    return {
        "target": val_clean[target].values,
        "lgbm": val_lgbm,
        "cat": val_cat,
        "test_target": test_clean[target].values,
        "test_lgbm": test_lgbm,
        "test_cat": test_cat,
        "test_info": test_clean[["Date", "StationName"]].copy()
    }

def find_optimal_weights(y_true, y_lgbm, y_cat):
    """Finds best weights by searching a grid (0.0 to 1.0)."""
    best_r2 = -np.inf
    best_w = 0.5
    
    for w in np.linspace(0, 1, 101):
        y_ensemble = w * y_lgbm + (1 - w) * y_cat
        r2 = r2_score(y_true, y_ensemble)
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
            
    return best_w, best_r2

def calculate_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val, test = load_data()
    
    report = [f"EXP 7 ENSEMBLE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              "Ensembling: LightGBM (Exp4) + CatBoost (Exp5)", "-"*50]
    
    final_test_results = {}

    for target in TARGETS:
        log.info(f"\nProcessing {target}...")
        data = get_predictions(target, val, test)
        
        # Find optimal weights on Val
        w, val_r2 = find_optimal_weights(data['target'], data['lgbm'], data['cat'])
        log.info(f"Optimal Weight for {target} (LGBM): {w:.2f} (Val R2: {val_r2:.4f})")
        
        # Apply to Test
        test_ensemble = w * data['test_lgbm'] + (1 - w) * data['test_cat']
        test_metrics = calculate_metrics(data['test_target'], test_ensemble)
        lgbm_test_metrics = calculate_metrics(data['test_target'], data['test_lgbm'])
        cat_test_metrics  = calculate_metrics(data['test_target'], data['test_cat'])
        
        # Report
        report.append(f"\nTARGET: {target}")
        report.append(f"  Optimal Weight (LGBM): {w:.2f}")
        report.append(f"  LGBM Test R2:      {lgbm_test_metrics['R2']:.4f}")
        report.append(f"  CatBoost Test R2:  {cat_test_metrics['R2']:.4f}")
        report.append(f"  Ensemble Test R2:  {test_metrics['R2']:.4f}")
        report.append(f"  Ensemble Test RMSE: {test_metrics['RMSE']:.2f}")
        report.append(f"  Ensemble Test MAE:  {test_metrics['MAE']:.2f}")
        
        # Save ensemble predictions
        res_df = data['test_info'].copy()
        res_df[target] = data['test_target']
        res_df[f"{target}_pred"] = test_ensemble
        res_df.to_csv(OUT_DIR / f"{target}_ensemble_preds.csv", index=False)

    # Write Final Report
    report_text = "\n".join(report)
    with open(OUT_DIR / "exp7_report.txt", "w") as f:
        f.write(report_text)
    
    log.info("\n" + report_text)
    log.info(f"\nEnsemble results saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
