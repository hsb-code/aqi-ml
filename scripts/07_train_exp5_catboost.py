"""
scripts/07_train_exp5_catboost.py
==================================
Experiment 5: CatBoost + Native Categorical + Extended Lags + Optuna

Key differences from previous LightGBM experiments:
1. NATIVE CATEGORICALS: Uses 'StationName' raw strings as cat_features.
2. ORDERED BOOSTING: CatBoost's algorithm is better suited for time-series.
3. GPU SUPPORT: Integrated task_type toggle.
4. MODULAR DESIGN: Cleaner separation of data, engineering, and training.

Usage:
  conda activate aqi-ml
  python scripts/07_train_exp5_catboost.py --n-trials 50
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
from catboost import CatBoostRegressor, Pool, CatBoostError
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Environment Setup
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp5")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

# Temporal Splits
TRAIN_END = "2023-06-30"
VAL_END   = "2023-12-31"

# Feature Definition
BASE_FEATURES = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

# We will dynamically generate lag/rolling features to keep it "new" but functionally equivalent to Exp4 scope
LAG_SPECS = [
    ("PM25", 1), ("PM10", 1), ("PM25", 2), ("PM10", 2), ("PM25", 3), ("PM10", 3)
]
ROLL_SPECS = [
    ("PM25", 3), ("PM10", 3), ("PM25", 7), ("PM10", 7), ("PM25", 14), ("PM10", 14),
    ("AOD", 3), ("NO2_ugm3", 3), ("AOD", 7), ("NO2_ugm3", 7),
    ("WindSpeed", 3), ("T2M_C", 3)
]
EWM_SPECS = [
    ("PM25", 7), ("PM10", 7)
]

CAT_FEATURES = ["StationName"]

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data & Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fresh implementation of feature engineering with a focus on modularity."""
    log.info("Starting fresh feature engineering...")
    df = df.sort_values(["StationName", "Date"]).copy()
    
    generated_cols = []

    for station, group in df.groupby("StationName", sort=False):
        idx = group.index
        
        # Lags
        for target, n in LAG_SPECS:
            col = f"{target}_lag{n}"
            df.loc[idx, col] = group[target].shift(n)
            if col not in generated_cols: generated_cols.append(col)
            
        # Rolling Means (shift 1 to avoid leakage)
        for target, window in ROLL_SPECS:
            col = f"{target}_roll{window}"
            min_p = max(2, window // 3)
            df.loc[idx, col] = group[target].shift(1).rolling(window, min_periods=min_p).mean()
            if col not in generated_cols: generated_cols.append(col)
            
        # EWM
        for target, span in EWM_SPECS:
            col = f"{target}_ewm{span}"
            df.loc[idx, col] = group[target].shift(1).ewm(span=span, min_periods=3).mean()
            if col not in generated_cols: generated_cols.append(col)

    log.info(f"Generated {len(generated_cols)} lag/rolling/EWM features.")
    return df, generated_cols

def load_and_split():
    log.info("Loading data from %s", PROCESSED_DIR / "training_data_full.parquet")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet", engine='fastparquet')
    
    # Pre-process: ensure Date is datetime
    full["Date"] = pd.to_datetime(full["Date"])
    
    # Feature Engineering
    full, lag_cols = engineer_features(full)
    
    # All features
    all_features = BASE_FEATURES + lag_cols + CAT_FEATURES
    
    # Splits (Temporal)
    train = full[full["Date"] <= TRAIN_END].copy()
    val   = full[(full["Date"] > TRAIN_END) & (full["Date"] <= VAL_END)].copy()
    test  = full[full["Date"] > VAL_END].copy()
    
    return train, val, test, all_features

# ---------------------------------------------------------------------------
# Model Training & Optimization
# ---------------------------------------------------------------------------

def train_catboost_target(target, train_df, val_df, test_df, feature_cols, n_trials, use_gpu=False):
    log.info("-" * 40)
    log.info(f"TRAINING TARGET: {target}")
    log.info("-" * 40)
    
    # Drop NaNs in target
    train_df = train_df.dropna(subset=[target])
    val_df   = val_df.dropna(subset=[target])
    test_df  = test_df.dropna(subset=[target])
    
    X_train, y_train = train_df[feature_cols], train_df[target]
    X_val,   y_val   = val_df[feature_cols],   val_df[target]
    X_test,  y_test  = test_df[feature_cols],  test_df[target]
    
    log.info(f"Samples - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Optuna Objective
    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 128, 254),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            "loss_function": "RMSE",
            "eval_metric": "R2",
            "random_seed": 42,
            "task_type": "GPU" if use_gpu else "CPU",
            "verbose": False
        }
        
        # Specific constraints for certain grow policies
        if params["grow_policy"] == "Depthwise":
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)
        elif params["grow_policy"] == "Lossguide":
            params["num_leaves"] = trial.suggest_int("num_leaves", 16, 64)

        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=CAT_FEATURES,
            early_stopping_rounds=100,
            use_best_model=True
        )
        
        return model.get_best_score()["validation"]["R2"]

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    log.info(f"Best trial R2: {study.best_value:.4f}")
    log.info(f"Best params: {study.best_params}")
    
    # Final Training with best params
    best_params = {
        "iterations": 3000,
        "loss_function": "RMSE",
        "eval_metric": "R2",
        "random_seed": 42,
        "task_type": "GPU" if use_gpu else "CPU",
        "verbose": 100,
        **study.best_params
    }
    
    model = CatBoostRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=CAT_FEATURES,
        early_stopping_rounds=100,
        use_best_model=True
    )
    
    # Evaluation
    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)
    test_preds  = model.predict(X_test)
    
    metrics = {
        "Train": calculate_metrics(y_train, train_preds),
        "Val":   calculate_metrics(y_val, val_preds),
        "Test":  calculate_metrics(y_test, test_preds)
    }
    
    return model, metrics, test_preds, test_df[["Date", "StationName", target]].copy()

def calculate_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# ---------------------------------------------------------------------------
# Reporting & Visualization
# ---------------------------------------------------------------------------

def save_plots(target, y_true, y_pred, model, feature_cols):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel(f"Observed {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"CatBoost {target} - Test Set\nR2: {r2_score(y_true, y_pred):.3f}")
    plt.savefig(PLOTS_DIR / f"{target}_scatter.png", dpi=150)
    plt.close()
    
    # 2. Feature Importance
    importance = model.get_feature_importance()
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='skyblue')
    plt.gca().invert_yaxis()
    plt.title(f"Top 20 Features - {target} (CatBoost)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{target}_importance.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    train, val, test, feature_cols = load_and_split()
    
    results_report = []
    results_report.append(f"EXP 5 CATBOOST REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_report.append(f"Trials per target: {args.n_trials}")
    results_report.append(f"GPU: {args.use_gpu}")
    results_report.append("-" * 50)
    
    for target in TARGETS:
        model, metrics, preds, test_info = train_catboost_target(
            target, train, val, test, feature_cols, args.n_trials, args.use_gpu
        )
        
        # Save model
        model_path = OUT_DIR / f"{target}_catboost.cbm"
        model.save_model(str(model_path))
        
        # Save Predictions
        test_info[f"{target}_pred"] = preds
        test_info.to_csv(OUT_DIR / f"{target}_test_predictions.csv", index=False)
        
        # Reporting
        results_report.append(f"\nTARGET: {target}")
        for split, m in metrics.items():
            results_report.append(f"  {split:5}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}")
            
        # Plots
        save_plots(target, test_info[target], preds, model, feature_cols)
        
    # Write Final Report
    with open(OUT_DIR / "exp5_report.txt", "w") as f:
        f.write("\n".join(results_report))
        
    log.info("Experiment 5 complete. Results saved in %s", OUT_DIR)

if __name__ == "__main__":
    main()
