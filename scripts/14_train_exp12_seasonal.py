"""
scripts/14_train_exp12_seasonal.py
======================================
Experiment 12: Seasonal LightGBM Models
Separate optimization and training for:
- Cool Season (Oct - Mar)
- Warm Season (Apr - Sep)

Goal: Specialized models for different atmospheric regimes in Abu Dhabi.
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
OUT_DIR       = Path("models/exp12")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

TRAIN_END = "2023-06-30"
VAL_END   = "2023-12-31"

BASE_FEATURE_COLS = [
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
        # Fix a few missing rolls from exp4
        df.loc[idx, "AOD_roll7"]   = grp["AOD"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "NO2_roll7"]   = grp["NO2_ugm3"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "PM25_ewm7"]   = grp["PM25"].shift(1).ewm(span=7, min_periods=3).mean()
        df.loc[idx, "PM10_ewm7"]   = grp["PM10"].shift(1).ewm(span=7, min_periods=3).mean()
        df.loc[idx, "WindSpeed_roll3"] = grp["WindSpeed"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "T2M_roll3"]       = grp["T2M_C"].shift(1).rolling(3, min_periods=2).mean()
    return df

def get_season_split(df: pd.DataFrame):
    """Split into Warm (Apr-Sep) and Cool (Oct-Mar)"""
    df = df.copy()
    # Month is already in processed data
    # Cool = Oct, Nov, Dec, Jan, Feb, Mar
    cool_months = [10, 11, 12, 1, 2, 3]
    df["MeteorologicalSeason"] = np.where(df["Month"].isin(cool_months), "Cool", "Warm")
    return df

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

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--n-trials", type=int, default=20)
    args = args_parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading training data...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet")
    full = add_lag_features(full)
    full = get_season_split(full)
    
    # Encode station once
    le = LabelEncoder()
    full[STATION_COL] = le.fit_transform(full["StationName"])
    joblib.dump(le, OUT_DIR / "station_label_encoder.pkl")

    all_features = BASE_FEATURE_COLS + LAG_COLS + [STATION_COL]
    cat_features = [STATION_COL, "Season"]

    # Splits
    train_full = full[full["Date"] <= TRAIN_END]
    val_full   = full[(full["Date"] > TRAIN_END) & (full["Date"] <= VAL_END)]
    test_full  = full[full["Date"] > VAL_END]

    report = [f"EXPERIMENT 12: Seasonal Models\n", "="*50 + "\n"]
    
    final_test_preds = test_full[["Date", "StationName", "PM25", "PM10", "MeteorologicalSeason"]].copy()

    for target in TARGETS:
        log.info(f"\n--- Targeting {target} ---")
        
        for m_season in ["Cool", "Warm"]:
            log.info(f"Training specialized model for {m_season} season...")
            
            # Filter splits for season
            tr = train_full[train_full["MeteorologicalSeason"] == m_season].dropna(subset=[target] + all_features)
            vl = val_full[val_full["MeteorologicalSeason"] == m_season].dropna(subset=[target] + all_features)
            te = test_full[test_full["MeteorologicalSeason"] == m_season].dropna(subset=[target] + all_features)
            
            if len(tr) < 50 or len(vl) < 10:
                log.warning(f"Not enough data for {m_season}/{target}. Skipping.")
                continue

            X_tr, y_tr = tr[all_features], tr[target]
            X_vl, y_vl = vl[all_features], vl[target]
            
            # Optuna
            study = optuna.create_study(direction="minimize")
            study.optimize(make_objective(X_tr, y_tr, X_vl, y_vl, cat_features), n_trials=args.n_trials)
            
            # Retrain on tr+vl
            best_params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42, 
                           "feature_pre_filter": False, **study.best_params}
            X_tv = pd.concat([X_tr, X_vl])
            y_tv = np.concatenate([y_tr, y_vl])
            
            # Early stopping check for iterations
            dtrain_es = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
            dval_es   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain_es, categorical_feature=cat_features)
            model_es  = lgb.train(best_params, dtrain_es, num_boost_round=1500, valid_sets=[dval_es], callbacks=[lgb.early_stopping(50, verbose=False)])
            
            dtrain_full = lgb.Dataset(X_tv, label=y_tv, categorical_feature=cat_features)
            model = lgb.train(best_params, dtrain_full, num_boost_round=int(model_es.best_iteration * 1.1))
            
            # Save model
            model.save_model(str(OUT_DIR / f"{target}_{m_season}.txt"))
            
            # Predict on corresponding test slice
            if not te.empty:
                X_te = te[all_features]
                preds = model.predict(X_te)
                final_test_preds.loc[te.index, f"{target}_pred"] = preds
        
        # Aggregate evaluation for the target
        mask = final_test_preds[f"{target}_pred"].notna() & final_test_preds[target].notna()
        y_true = final_test_preds.loc[mask, target]
        y_pred = final_test_preds.loc[mask, f"{target}_pred"]
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        log.info(f"COMBINED TEST RESULTS [{target}]: R2={r2:.4f}  RMSE={rmse:.2f}")
        report.append(f"TARGET: {target}\n  R2:   {r2:.4f}\n  RMSE: {rmse:.2f}\n  MAE:  {mae:.2f}\n\n")

    with open(OUT_DIR / "exp12_report.txt", "w") as f:
        f.writelines(report)
    
    final_test_preds.to_csv(OUT_DIR / "exp12_test_predictions.csv", index=False)
    log.info(f"Experiment complete. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
