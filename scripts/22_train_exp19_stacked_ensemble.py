
import sys
import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Config
DATA_DIR = Path("data/processed")
OUT_DIR  = Path("models/exp19")
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
    "UrbanDensity_5km", "PBLH_Wind_Index", "DewPoint_Depression", "Coastal_Exposure"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

def train_base_models(X, y, cat_features):
    # LGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=64, random_state=42, verbose=-1)
    lgb_model.fit(X, y, categorical_feature=cat_features)
    
    # CatBoost
    cb_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, random_state=42, verbose=0)
    cb_model.fit(X, y, cat_features=cat_features)
    
    # XGBoost
    # XGB needs numeric cat features
    X_xgb = X.copy()
    for col in cat_features:
        X_xgb[col] = X_xgb[col].astype('category').cat.codes
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42)
    xgb_model.fit(X_xgb, y)
    
    return lgb_model, cb_model, xgb_model

def get_predictions(lgb_model, cb_model, xgb_model, X, cat_features):
    p1 = lgb_model.predict(X)
    p2 = cb_model.predict(X)
    X_xgb = X.copy()
    for col in cat_features:
        X_xgb[col] = X_xgb[col].astype('category').cat.codes
    p3 = xgb_model.predict(X_xgb)
    return np.column_stack([p1, p2, p3])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-agnostic", action="store_true", default=True)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_DIR / "advanced_features.parquet")
    
    train_full = df[df["Date"] <= VAL_END]
    test       = df[df["Date"] > VAL_END]
    
    results = []
    
    for target in TARGETS:
        log.info(f"===== Training Exp 19 Stack: {target} =====")
        
        drp_cols = [f"DRP_{target}_lag1", f"DRP_{target}_roll3", f"DRP_{target}_roll7"]
        features = BASE_FEATURES + drp_cols
        
        train_clean = train_full.dropna(subset=[target] + features)
        test_clean  = test.dropna(subset=[target] + features)
        
        X_train, y_train = train_clean[features], train_clean[target]
        X_test, y_test   = test_clean[features], test_clean[target]
        
        cat_features = ["Season"]
        
        # 5-Fold OOF Predictions for Meta-Learner
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(X_train), 3))
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            m1, m2, m3 = train_base_models(X_tr, y_tr, cat_features)
            oof_preds[val_idx] = get_predictions(m1, m2, m3, X_val, cat_features)
            
        # Meta-Learner
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(oof_preds, y_train)
        
        # Final train on all training data
        m1, m2, m3 = train_base_models(X_train, y_train, cat_features)
        
        # Test predictions
        test_base_preds = get_predictions(m1, m2, m3, X_test, cat_features)
        final_preds = meta_model.predict(test_base_preds)
        
        r2 = r2_score(y_test, final_preds)
        rmse = np.sqrt(mean_squared_error(y_test, final_preds))
        
        log.info(f"[{target}] STACKED Test R2: {r2:.4f} | RMSE: {rmse:.2f}")
        results.append(f"{target} Stacked R2: {r2:.4f}, RMSE: {rmse:.2f}")
        
        # Save models
        joblib.dump(meta_model, OUT_DIR / f"{target}_meta_ridge.joblib")
        joblib.dump(m1, OUT_DIR / f"{target}_lgb.joblib")
        joblib.dump(m2, OUT_DIR / f"{target}_cb.joblib")
        joblib.dump(m3, OUT_DIR / f"{target}_xgb.joblib")

    with open(OUT_DIR / "exp19_report.txt", "w") as f:
        f.write("EXPERIMENT 19: Stacked Ensemble (LGBM + CB + XGB + Ridge)\n")
        f.write("Enriched with Stability and Exposure Phys-Plus Features\n")
        f.write("="*60 + "\n")
        for res in results:
            f.write(res + "\n")
            
    log.info(f"Report saved to {OUT_DIR}/exp19_report.txt")

if __name__ == "__main__":
    main()
