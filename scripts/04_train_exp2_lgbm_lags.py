"""
scripts/06_train_exp4_lgbm_lags.py
=====================================
Experiment 4: LightGBM + Lag Features

Builds on Exp 3 by adding temporal persistence features:
  - PM25_lag1, PM10_lag1        : yesterday's measured PM (strongest predictors)
  - PM25_roll3, PM10_roll3      : 3-day rolling mean
  - PM25_roll7, PM10_roll7      : 7-day rolling mean
  - AOD_roll3, NO2_roll3        : 3-day satellite rolling mean

IMPORTANT: lags are created on the FULL dataset before splitting so that
val rows correctly use train's last rows as lag source (no boundary leak).

Usage:
  conda activate aqi-ml
  python scripts/06_train_exp4_lgbm_lags.py
"""

import sys
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp4")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

# Temporal split boundaries (same as preprocess.py)
TRAIN_END = "2023-06-30"
VAL_END   = "2023-12-31"

BASE_FEATURE_COLS = [
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    "DayOfYear", "Month", "Season", "IsWeekend",
    "Latitude", "Longitude",
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

LAG_FEATURE_COLS = [
    "PM25_lag1", "PM10_lag1",
    "PM25_roll3", "PM10_roll3",
    "PM25_roll7", "PM10_roll7",
    "AOD_roll3",  "NO2_roll3",
]

# LightGBM params — slightly more conservative to help generalisation
LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "learning_rate":     0.03,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "verbose":           -1,
    "n_jobs":            -1,
    "seed":              42,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING  = 75

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
# Lag feature engineering
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features PER STATION sorted by date.
    Must be called on the FULL dataset before temporal splitting.
    """
    df = df.sort_values(["StationName", "Date"]).copy()

    for station, grp in df.groupby("StationName", sort=False):
        idx = grp.index

        # Yesterday's PM
        df.loc[idx, "PM25_lag1"] = grp["PM25"].shift(1)
        df.loc[idx, "PM10_lag1"] = grp["PM10"].shift(1)

        # 3-day rolling mean (min 2 valid days)
        df.loc[idx, "PM25_roll3"] = grp["PM25"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "PM10_roll3"] = grp["PM10"].shift(1).rolling(3, min_periods=2).mean()

        # 7-day rolling mean (min 3 valid days)
        df.loc[idx, "PM25_roll7"] = grp["PM25"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "PM10_roll7"] = grp["PM10"].shift(1).rolling(7, min_periods=3).mean()

        # Satellite rolling means
        df.loc[idx, "AOD_roll3"] = grp["AOD"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "NO2_roll3"] = grp["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean()

    added = df[LAG_FEATURE_COLS].notna().mean() * 100
    log.info("Lag feature coverage:")
    for col, pct in added.items():
        log.info("  %s: %.1f%% non-NaN", col, pct)

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame):
    train = df[df["Date"] <= TRAIN_END]
    val   = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)]
    test  = df[df["Date"] > VAL_END]
    return train, val, test


def prepare_xy(df: pd.DataFrame, target: str, feature_cols: list):
    df = df.dropna(subset=[target]).copy()
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df[target].values
    return X, y, df


def compute_metrics(y_true, y_pred, split_name: str) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    log.info("  %-6s R²=%.4f | RMSE=%.2f | MAE=%.2f", split_name, r2, rmse, mae)
    return {"split": split_name, "R2": r2, "RMSE": rmse, "MAE": mae}


def plot_scatter(y_true, y_pred, target: str, split: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#e07b00")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line")
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_xlabel(f"Observed {target} (ug/m3)")
    ax.set_ylabel(f"Predicted {target} (ug/m3)")
    ax.set_title(f"Exp4 LightGBM+Lags — {target} ({split})\nR²={r2:.3f}  RMSE={rmse:.2f}")
    ax.legend()
    plt.tight_layout()
    path = out_dir / f"{target}_{split}_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved scatter: %s", path.name)


def plot_feature_importance(model: lgb.Booster, target: str, out_dir: Path):
    importances = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=model.feature_name(),
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#e07b00" if "lag" in n or "roll" in n else "#2d7dd2"
              for n in importances.index]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Feature Importance (gain) — {target} [orange = lag features]")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    path = out_dir / f"{target}_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved feature importance: %s", path.name)


def plot_shap(model: lgb.Booster, X: pd.DataFrame, target: str, out_dir: Path):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(9, 8))
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False, max_display=25)
        plt.title(f"SHAP Feature Importance — {target} (Exp4)")
        plt.tight_layout()
        path = out_dir / f"{target}_shap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved SHAP plot: %s", path.name)
    except Exception as e:
        log.warning("SHAP skipped: %s", e)


# ---------------------------------------------------------------------------
# Train one target
# ---------------------------------------------------------------------------

def train_target(target: str, train: pd.DataFrame, val: pd.DataFrame,
                 test: pd.DataFrame, feature_cols: list):
    log.info("=" * 60)
    log.info("TARGET: %s  (Exp4 — LightGBM + Lag Features)", target)
    log.info("=" * 60)

    X_train, y_train, _ = prepare_xy(train, target, feature_cols)
    X_val,   y_val,   _ = prepare_xy(val,   target, feature_cols)
    X_test,  y_test, test_clean = prepare_xy(test, target, feature_cols)

    log.info("Train: %s | Val: %s | Test: %s | Features: %d",
             f"{len(X_train):,}", f"{len(X_val):,}",
             f"{len(X_test):,}", X_train.shape[1])

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, free_raw_data=False)

    log.info("Training (max %d rounds, early stop=%d)...", NUM_BOOST_ROUND, EARLY_STOPPING)
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    log.info("Best iteration: %d", model.best_iteration)

    log.info("--- Metrics ---")
    results = [
        compute_metrics(y_train, model.predict(X_train), "Train"),
        compute_metrics(y_val,   model.predict(X_val),   "Val"),
        compute_metrics(y_test,  model.predict(X_test),  "Test"),
    ]

    plot_scatter(y_val,  model.predict(X_val),  target, "val",  PLOTS_DIR)
    plot_scatter(y_test, model.predict(X_test), target, "test", PLOTS_DIR)
    plot_feature_importance(model, target, PLOTS_DIR)
    plot_shap(model, X_val, target, PLOTS_DIR)

    # Save model
    model.save_model(str(OUT_DIR / f"{target}_lgbm.txt"))
    joblib.dump(model, OUT_DIR / f"{target}_lgbm.pkl")
    log.info("Model saved: %s_lgbm.txt", target)

    # Save test predictions
    test_clean = test_clean.copy()
    test_clean[f"{target}_pred"] = model.predict(X_test)
    pred_path = OUT_DIR / f"{target}_test_predictions.csv"
    test_clean[["Date", "StationName", target, f"{target}_pred"]].to_csv(pred_path, index=False)
    log.info("Test predictions saved: %s", pred_path.name)

    return model, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXPERIMENT 4 — LightGBM + Lag Features")
    log.info("=" * 60)

    # Load full dataset so lag computation spans train→val boundary correctly
    log.info("Loading full dataset...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet")
    log.info("Full dataset: %s rows", f"{len(full):,}")

    # Create lag features on the full dataset before splitting
    log.info("Engineering lag features...")
    full = add_lag_features(full)

    # Re-apply temporal split
    train, val, test = temporal_split(full)
    log.info(
        "Splits after lags — Train: %s | Val: %s | Test: %s",
        f"{len(train):,}", f"{len(val):,}", f"{len(test):,}",
    )

    # Full feature set = base + lag
    all_feature_cols = BASE_FEATURE_COLS + LAG_FEATURE_COLS
    log.info("Total features: %d (%d base + %d lag)",
             len(all_feature_cols), len(BASE_FEATURE_COLS), len(LAG_FEATURE_COLS))

    all_results = {}
    for target in TARGETS:
        model, results = train_target(target, train, val, test, all_feature_cols)
        all_results[target] = results

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("EXPERIMENT 4 SUMMARY")
    log.info("=" * 60)

    report_lines = [
        "EXPERIMENT 4 — LightGBM + Lag Features (Temporal Split)\n",
        "=" * 60 + "\n",
        f"Train: 2022-01-01 to 2023-06-30  ({len(train):,} rows)\n",
        f"Val:   2023-07-01 to 2023-12-31  ({len(val):,} rows)\n",
        f"Test:  2024-01-01 to 2024-12-30  ({len(test):,} rows)\n\n",
        f"Base features: {len(BASE_FEATURE_COLS)}\n",
        f"Lag features:  {len(LAG_FEATURE_COLS)}   "
        f"({', '.join(LAG_FEATURE_COLS)})\n",
        f"Total features: {len(all_feature_cols)}\n\n",
    ]

    for target, results in all_results.items():
        report_lines.append(f"{target}\n" + "-" * 40 + "\n")
        for r in results:
            report_lines.append(
                f"  {r['split']:<6}: R2={r['R2']:.4f}  "
                f"RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}\n"
            )
        report_lines.append("\n")
        test_r = next(r for r in results if r["split"] == "Test")
        val_r  = next(r for r in results if r["split"] == "Val")
        log.info("%s — Val R²=%.4f RMSE=%.2f | Test R²=%.4f RMSE=%.2f",
                 target, val_r["R2"], val_r["RMSE"], test_r["R2"], test_r["RMSE"])

    report_lines.append("\nExp3 vs Exp4 comparison (Test set):\n")
    report_lines.append("  Exp3 PM25 R²=0.2356 | Exp4 PM25 see above\n")
    report_lines.append("  Exp3 PM10 R²=0.2607 | Exp4 PM10 see above\n")

    report_path = OUT_DIR / "exp4_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    log.info("Report saved: %s", report_path)
    log.info("All outputs in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
