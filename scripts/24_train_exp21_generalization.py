"""
scripts/24_train_exp21_generalization.py
=======================================
Experiment 21: Generalization & Regularization (Final Push)
- REMOVES Station ID (Station_enc)
- ADDS Regional Context (Regional_AOD_Avg, Regional_NO2_Avg)
- ADDS Physics-Plus features (Stability, Ventilation)
- SHUFFLE & TEMPORAL splits evaluation
- STRICT Regularization via Optuna
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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp21")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

# Physics + LUR + Seasonal Features
BASE_PHYSICAL = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "Elevation_m", "Dist_Coast_km", "Dist_Corniche_km", "Dist_E11_km",
    "Wind_U", "Wind_V", "VentilationIndex", "StabilityIndex",
    "PBLH_Wind_Index", "DewPoint_Depression", "Coastal_Exposure", "UrbanDensity_5km"
]

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
# Feature Engineering
# ---------------------------------------------------------------------------

def add_regional_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily averages of AOD and NO2 across all stations."""
    log.info("Computing regional pollution trends...")
    regional = df.groupby("Date")[["AOD", "NO2_ugm3"]].mean().reset_index()
    regional.columns = ["Date", "Regional_AOD_Avg", "Regional_NO2_Avg"]
    df = df.merge(regional, on="Date", how="left")
    return df

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add DRP lags and standard feature lags."""
    df = df.sort_values(["StationName", "Date"]).copy()
    for station, grp in df.groupby("StationName", sort=False):
        idx = grp.index
        # AOD and NO2 persistence
        df.loc[idx, "AOD_lag1"] = grp["AOD"].shift(1)
        df.loc[idx, "NO2_lag1"] = grp["NO2_ugm3"].shift(1)
        # Regional persistence lags (already in data but making sure)
        df.loc[idx, "Reg_AOD_lag1"] = df.loc[idx, "Regional_AOD_Avg"].shift(1)
    return df

def make_objective(X_train, y_train, X_val, y_val, cat_features):
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, categorical_feature=cat_features, free_raw_data=False)
    
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            # STRICT REGULARIZATION
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 50.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 50.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.1, 5.0),
        }
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval], callbacks=callbacks)
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
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading advanced features...")
    df = pd.read_parquet(PROCESSED_DIR / "advanced_features.parquet")
    
    df = add_regional_trends(df)
    df = add_lags(df)
    
    # Define Target Specific Lags (DRP)
    DRP_COLS = ["DRP_PM25_lag1", "DRP_PM25_roll3", "DRP_PM25_roll7", 
                "DRP_PM10_lag1", "DRP_PM10_roll3", "DRP_PM10_roll7"]
    
    GLOBAL_COLS = ["Regional_AOD_Avg", "Regional_NO2_Avg", "Reg_AOD_lag1", "AOD_lag1", "NO2_lag1"]
    
    report = [f"EXPERIMENT 21: Generalization & Regularization\n", "="*60 + "\n"]

    for target in TARGETS:
        log.info(f"\n===== TRAINING MODEL FOR {target} =====")
        
        # Select target specific DRP columns
        target_drp = [c for c in DRP_COLS if target in c]
        features = BASE_PHYSICAL + GLOBAL_COLS + target_drp
        
        # NO Station_enc here!
        cat_features = ["Season"]
        
        df_clean = df.dropna(subset=[target] + features).copy()
        
        # 1. EVALUATE ON MIXED SPLIT
        log.info("Evaluating on MIXED split...")
        X_mix, y_mix = df_clean[features], df_clean[target]
        X_train_m, X_temp_m, y_train_m, y_temp_m = train_test_split(X_mix, y_mix, test_size=0.2, random_state=42)
        X_val_m, X_test_m, y_val_m, y_test_m     = train_test_split(X_temp_m, y_temp_m, test_size=0.5, random_state=42)
        
        study_m = optuna.create_study(direction="minimize")
        study_m.optimize(make_objective(X_train_m, y_train_m, X_val_m, y_val_m, cat_features), n_trials=args.n_trials)
        
        best_p_m = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42, **study_m.best_params}
        dtrain_m = lgb.Dataset(X_train_m, label=y_train_m, categorical_feature=cat_features)
        dval_m   = lgb.Dataset(X_val_m,   label=y_val_m,   reference=dtrain_m, categorical_feature=cat_features)
        model_es_m = lgb.train(best_p_m, dtrain_m, num_boost_round=2000, valid_sets=[dval_m], 
                               callbacks=[lgb.early_stopping(50, verbose=False)])
        
        # Retrain on full mix training set
        model_mix = lgb.train(best_p_m, lgb.Dataset(pd.concat([X_train_m, X_val_m]), 
                                                     label=np.concatenate([y_train_m, y_val_m]), 
                                                     categorical_feature=cat_features), 
                              num_boost_round=int(model_es_m.best_iteration * 1.1))
        
        mix_res = compute_metrics(y_test_m, model_mix.predict(X_test_m), "Test (Mixed)")
        
        # 2. EVALUATE ON TEMPORAL SPLIT
        log.info("Evaluating on TEMPORAL split...")
        train_t = df_clean[df_clean["Date"] <= "2023-06-30"]
        val_t   = df_clean[(df_clean["Date"] > "2023-06-30") & (df_clean["Date"] <= "2023-12-31")]
        test_t  = df_clean[df_clean["Date"] > "2023-12-31"]
        
        X_train_t, y_train_t = train_t[features], train_t[target]
        X_val_t, y_val_t     = val_t[features],   val_t[target]
        X_test_t, y_test_t   = test_t[features],  test_t[target]
        
        study_t = optuna.create_study(direction="minimize")
        study_t.optimize(make_objective(X_train_t, y_train_t, X_val_t, y_val_t, cat_features), n_trials=args.n_trials)
        
        best_p_t = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42, **study_t.best_params}
        dtrain_t = lgb.Dataset(X_train_t, label=y_train_t, categorical_feature=cat_features)
        dval_t   = lgb.Dataset(X_val_t,   label=y_val_t,   reference=dtrain_t, categorical_feature=cat_features)
        model_es_t = lgb.train(best_p_t, dtrain_t, num_boost_round=2000, valid_sets=[dval_t], 
                               callbacks=[lgb.early_stopping(50, verbose=False)])
        
        model_temp = lgb.train(best_p_t, lgb.Dataset(pd.concat([X_train_t, X_val_t]), 
                                                      label=np.concatenate([y_train_t, y_val_t]), 
                                                      categorical_feature=cat_features), 
                               num_boost_round=int(model_es_t.best_iteration * 1.1))
        
        temp_res = compute_metrics(y_test_t, model_temp.predict(X_test_t), "Test (Temporal)")
        
        # Save results
        report.append(f"TARGET: {target}\n" + "-"*30 + "\n")
        report.append(f"  MIXED SPLIT TEST R2:    {mix_res['R2']:.4f}\n")
        report.append(f"  TEMPORAL SPLIT TEST R2: {temp_res['R2']:.4f}\n\n")
        
        model_temp.save_model(str(OUT_DIR / f"{target}_general_model.txt"))

    with open(OUT_DIR / "exp21_report.txt", "w") as f:
        f.writelines(report)
        
    log.info(f"Experiment 21 complete. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
