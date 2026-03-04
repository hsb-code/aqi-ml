
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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Config
DATA_DIR = Path("data/processed")
OUT_DIR  = Path("models/exp17")
TARGETS  = ["PM25", "PM10"]

TRAIN_END = "2023-06-30"
VAL_END   = "2023-12-31"

BASE_FEATURES = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "Elevation_m", "Dist_Coast_km", "Dist_Corniche_km", "Dist_E11_km",
    "Wind_U", "Wind_V", "VentilationIndex", "StabilityIndex",
    "UrbanDensity_5km"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

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
            "feature_pre_filter": False
        }
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], callbacks=callbacks)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    return objective

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_DIR / "advanced_features.parquet")
    
    train = df[df["Date"] <= TRAIN_END]
    val   = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)]
    test  = df[df["Date"] > VAL_END]
    
    results = []
    
    for target in TARGETS:
        log.info(f"===== Training Exp 17: {target} (DRP Windows + LUR) =====")
        
        drp_cols = [f"DRP_{target}_lag1", f"DRP_{target}_roll3", f"DRP_{target}_roll7"]
        features = BASE_FEATURES + drp_cols
        
        train_clean = train.dropna(subset=[target] + features)
        val_clean   = val.dropna(subset=[target] + features)
        test_clean  = test.dropna(subset=[target] + features)
        
        X_train, y_train = train_clean[features], train_clean[target]
        X_val, y_val     = val_clean[features], val_clean[target]
        X_test, y_test   = test_clean[features], test_clean[target]
        
        cat_features = ["Season"]
        
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(X_train, y_train, X_val, y_val, cat_features), n_trials=args.n_trials)
        
        best_params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42, **study.best_params, "feature_pre_filter": False}
        
        dtrain_full = lgb.Dataset(pd.concat([X_train, X_val]), 
                                  label=np.concatenate([y_train, y_val]), 
                                  categorical_feature=cat_features)
        
        model = lgb.train(best_params, dtrain_full, num_boost_round=600)
        
        model_path = OUT_DIR / f"{target}_exp17_drp_windows.txt"
        model.save_model(str(model_path))
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        log.info(f"[{target}] Test R2: {r2:.4f} | RMSE: {rmse:.2f}")
        results.append(f"{target} R2: {r2:.4f}, RMSE: {rmse:.2f}")
        
    with open(OUT_DIR / "exp17_report.txt", "w") as f:
        f.write("EXPERIMENT 17: 7-Day DRP Windows & Advanced LUR\n")
        f.write("="*60 + "\n")
        for res in results:
            f.write(res + "\n")
            
    log.info(f"Report saved to {OUT_DIR}/exp17_report.txt")

if __name__ == "__main__":
    main()
