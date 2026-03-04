"""
scripts/11_train_exp9_geospatial.py
====================================
Experiment 9: Geospatial Features + LightGBM + Optuna

Improves on Exp4 by adding spatial context:
- Spatial Features: Mean PM2.5/PM10 of 3 nearest stations (shifted by 1 day to prevent leakage).
- Base Model: LightGBM (Raw scale).
- Goal: Break the 0.60 R2 barrier by leveraging correlated pollution trends from neighbors.

Usage:
  conda activate aqi-ml
  python scripts/11_train_exp9_geospatial.py --n-trials 50
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
import lightgbm as lgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial import KDTree

# Project Root Setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp9")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

# Base Features (from Exp4)
BASE_FEATURES = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

# Static categorical
STATION_COL = "Station_enc"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Geospatial Feature Engineering
# ---------------------------------------------------------------------------

def engineer_geospatial_features(df: pd.DataFrame, k=3):
    """
    1. Finds k-nearest stations for each station.
    2. Calculates the mean target values of neighbors.
    3. Shifts by 1 day to avoid leakage.
    """
    log.info(f"Calculating geospatial features using {k} nearest neighbors...")
    
    # Get unique stations and their locations
    stations = df[["StationName", "Latitude", "Longitude"]].drop_duplicates().set_index("StationName")
    coords = stations.values # [Lat, Lon]
    
    tree = KDTree(coords)
    
    # Map each station to its k-nearest neighbors (excluding itself)
    # query returns (distances, indices)
    # d, i = tree.query(coords, k=k+1) # k+1 because index 0 is the station itself
    # But some stations might have identical coordinates or be very close. 
    # To be safe, we just use the indices.
    dist, indices = tree.query(coords, k=k+1)
    
    station_neighbors = {}
    for idx, station_name in enumerate(stations.index):
        # neighbor_indices = indices[idx][1:] # Skip self
        # Actually, if there are multiple stations at the same location, distance will be 0.
        # We just want the names.
        neighbor_names = stations.index[indices[idx][1:]].tolist()
        station_neighbors[station_name] = neighbor_names

    # Pivot data to [Date, StationName] matrix for easy cross-station access
    # PM25
    pm25_matrix = df.pivot(index="Date", columns="StationName", values="PM25")
    pm10_matrix = df.pivot(index="Date", columns="StationName", values="PM10")
    
    spatial_features = []
    
    for target_name, matrix in [("PM25", pm25_matrix), ("PM10", pm10_matrix)]:
        # For each station, average its neighbors
        neighbor_means = pd.DataFrame(index=matrix.index, columns=matrix.columns)
        for station_name, neighbors in station_neighbors.items():
            neighbor_means[station_name] = matrix[neighbors].mean(axis=1)
        
        # Shift by 1 to ensure we use "Yesterday's Neighbor Status" for Today's Prediction
        col_name = f"{target_name}_spatial_lag1"
        shifted_means = neighbor_means.shift(1).stack().reset_index()
        shifted_means.columns = ["Date", "StationName", col_name]
        
        # Merge back
        df = df.merge(shifted_means, on=["Date", "StationName"], how="left")
        spatial_features.append(col_name)
        
    log.info(f"Added spatial features: {spatial_features}")
    return df, spatial_features

def engineer_lags(df: pd.DataFrame):
    """Reuse the standard lag/rolling features from Exp4."""
    df = df.sort_values(["StationName", "Date"]).copy()
    
    # Simple Lags 1-3
    for target in ["PM25", "PM10"]:
        df[f"{target}_lag1"] = df.groupby("StationName")[target].shift(1)
        df[f"{target}_lag2"] = df.groupby("StationName")[target].shift(2)
        df[f"{target}_lag3"] = df.groupby("StationName")[target].shift(3)
        
    # Standard Rolling (shift 1 to avoid leakage)
    for target in ["PM25", "PM10"]:
        df[f"{target}_roll7"] = df.groupby("StationName")[target].shift(1).rolling(7, min_periods=3).mean()
        df[f"{target}_roll14"] = df.groupby("StationName")[target].shift(1).rolling(14, min_periods=5).mean()
        
    # Met/Satellite Persistence
    df["AOD_roll3"] = df.groupby("StationName")["AOD"].shift(1).rolling(3, min_periods=2).mean()
    df["NO2_roll3"] = df.groupby("StationName")["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean()
    
    lag_cols = [c for c in df.columns if any(x in c for x in ["lag", "roll", "ewm"])]
    return df, lag_cols

def load_data():
    log.info("Loading data...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet", engine='fastparquet')
    full["Date"] = pd.to_datetime(full["Date"])
    
    # 1. Geospatial
    full, spatial_cols = engineer_geospatial_features(full, k=3)
    
    # 2. Lags
    full, lag_cols = engineer_lags(full)
    
    # Encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    full[STATION_COL] = le.fit_transform(full["StationName"])
    
    feature_cols = list(set(BASE_FEATURES + lag_cols + spatial_cols + [STATION_COL]))
    # Ensure consistent ordering
    feature_cols.sort()
    
    train = full[full["Date"] <= "2023-06-30"].copy()
    val   = full[(full["Date"] > "2023-06-30") & (full["Date"] <= "2023-12-31")].copy()
    test  = full[full["Date"] > "2023-12-31"].copy()
    
    return train, val, test, feature_cols, le

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def train_target(target, train, val, test, feature_cols, n_trials):
    log.info("-" * 40)
    log.info(f"TRAINING TARGET: {target}")
    log.info("-" * 40)
    
    train = train.dropna(subset=[target] + feature_cols)
    val   = val.dropna(subset=[target] + feature_cols)
    test  = test.dropna(subset=[target] + feature_cols)
    
    X_train, y_train = train[feature_cols], train[target]
    X_val,   y_val   = val[feature_cols], val[target]
    
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        return model.best_score["valid_0"]["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Retrain
    best_params = {**study.best_params, "objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    final_model = lgb.train(
        best_params, dtrain, num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Evaluation
    X_test, y_test = test[feature_cols], test[target]
    y_pred = final_model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred)
    }
    
    test_res = test[["Date", "StationName", target]].copy()
    test_res[f"{target}_pred"] = y_pred
    
    return final_model, metrics, test_res

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    train, val, test, feature_cols, le = load_data()
    joblib.dump(le, OUT_DIR / "station_encoder.pkl")
    
    report = [f"EXP 9 GEOSPATIAL REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
              "Approach: LightGBM + 3-Nearest Neighbor Spatial Lags", f"Trials: {args.n_trials}", "-"*50]
    
    for target in TARGETS:
        model, metrics, test_res = train_target(target, train, val, test, feature_cols, args.n_trials)
        model.save_model(str(OUT_DIR / f"{target}_lgbm.txt"))
        
        # Plot
        plt.figure(figsize=(8,8))
        plt.scatter(test_res[target], test_res[f"{target}_pred"], alpha=0.3)
        plt.plot([test_res[target].min(), test_res[target].max()], [test_res[target].min(), test_res[target].max()], 'r--')
        plt.title(f"Exp 9 {target} - R2: {metrics['R2']:.3f}")
        plt.savefig(PLOTS_DIR / f"{target}_scatter.png")
        plt.close()
        
        report.append(f"\nTARGET: {target}")
        report.append(f"  Test R2:   {metrics['R2']:.4f}")
        report.append(f"  Test RMSE: {metrics['RMSE']:.2f}")
        report.append(f"  Test MAE:  {metrics['MAE']:.2f}")
        
    with open(OUT_DIR / "exp9_report.txt", "w") as f:
        f.write("\n".join(report))
        
    log.info("Exp 9 finished. Results in %s", OUT_DIR)

if __name__ == "__main__":
    main()
