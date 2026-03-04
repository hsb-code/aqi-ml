"""
scripts/05_train_exp3_lgbm.py
==============================
Experiment 3: LightGBM on the clean temporally-split dataset.

Improvements over Exp 1 & 2:
  - Correct temporal split (no data leakage): train 2022-H1 2023, val H2 2023, test 2024
  - LightGBM (faster than XGBoost, handles missing natively)
  - StationName as a native categorical feature
  - Separate model trained per target (PM25, PM10)
  - Early stopping on val set (no manual epoch count)
  - SHAP feature importance
  - Full metrics on val AND test

Usage:
  conda activate aqi-ml
  python scripts/05_train_exp3_lgbm.py
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
OUT_DIR       = Path("models/exp3")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

FEATURE_COLS = [
    # Satellite
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    # ERA5 meteorology
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    # Temporal
    "DayOfYear", "Month", "Season", "IsWeekend",
    # Spatial
    "Latitude", "Longitude",
    # Derived / physics
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
]

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             42,
}
NUM_BOOST_ROUND  = 2000
EARLY_STOPPING   = 50   # stop if val RMSE doesn't improve for 50 rounds

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

def load_splits():
    """Load the temporally-split parquet files."""
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    log.info(
        "Loaded splits — Train: %s (%s to %s) | Val: %s (%s to %s) | Test: %s (%s to %s)",
        f"{len(train):,}", train.Date.min().date(), train.Date.max().date(),
        f"{len(val):,}",   val.Date.min().date(),   val.Date.max().date(),
        f"{len(test):,}",  test.Date.min().date(),  test.Date.max().date(),
    )
    return train, val, test


def prepare_xy(df: pd.DataFrame, target: str):
    """Return (X, y) dropping rows where target is NaN."""
    df = df.dropna(subset=[target]).copy()
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (will skip): %s", missing)
    X = df[available].copy()
    y = df[target].values
    return X, y


def metrics(y_true, y_pred, split_name: str) -> dict:
    """Compute and log R², RMSE, MAE."""
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    log.info("  %s — R²=%.4f | RMSE=%.2f | MAE=%.2f", split_name, r2, rmse, mae)
    return {"split": split_name, "R2": r2, "RMSE": rmse, "MAE": mae}


def plot_scatter(y_true, y_pred, target: str, split: str, out_dir: Path):
    """Predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#2d7dd2")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line")
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_xlabel(f"Observed {target} (ug/m3)")
    ax.set_ylabel(f"Predicted {target} (ug/m3)")
    ax.set_title(f"Exp3 LightGBM — {target} ({split})\nR²={r2:.3f}  RMSE={rmse:.2f} ug/m3")
    ax.legend()
    plt.tight_layout()
    path = out_dir / f"{target}_{split}_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved scatter plot: %s", path.name)


def plot_feature_importance(model: lgb.Booster, target: str, out_dir: Path):
    """LightGBM native split-gain importance."""
    importances = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=model.feature_name(),
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    importances.plot(kind="barh", ax=ax, color="#2d7dd2")
    ax.set_title(f"Feature Importance (gain) — {target}")
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    path = out_dir / f"{target}_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved feature importance: %s", path.name)


def plot_shap(model: lgb.Booster, X: pd.DataFrame, target: str, out_dir: Path):
    """SHAP summary bar plot (mean |SHAP| per feature)."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(8, 7))
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — {target}")
        plt.tight_layout()
        path = out_dir / f"{target}_shap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved SHAP plot: %s", path.name)
    except Exception as e:
        log.warning("SHAP plot skipped: %s", e)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_target(target: str, train: pd.DataFrame, val: pd.DataFrame,
                 test: pd.DataFrame, out_dir: Path, plots_dir: Path):
    """Train one LightGBM model for a single target variable."""
    log.info("=" * 60)
    log.info("TARGET: %s", target)
    log.info("=" * 60)

    X_train, y_train = prepare_xy(train, target)
    X_val,   y_val   = prepare_xy(val,   target)
    X_test,  y_test  = prepare_xy(test,  target)

    log.info("Train: %s rows | Val: %s rows | Test: %s rows",
             f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}")
    log.info("Features: %d", X_train.shape[1])

    # Build LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, free_raw_data=False)

    # Train with early stopping
    log.info("Training LightGBM (max %d rounds, early stop=%d)...", NUM_BOOST_ROUND, EARLY_STOPPING)
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    log.info("Best iteration: %d", model.best_iteration)

    # Evaluate
    log.info("--- Metrics ---")
    results = []
    results.append(metrics(y_train, model.predict(X_train), "Train"))
    results.append(metrics(y_val,   model.predict(X_val),   "Val"))
    results.append(metrics(y_test,  model.predict(X_test),  "Test"))

    # Plots
    plot_scatter(y_val,  model.predict(X_val),  target, "val",  plots_dir)
    plot_scatter(y_test, model.predict(X_test), target, "test", plots_dir)
    plot_feature_importance(model, target, plots_dir)
    plot_shap(model, X_val, target, plots_dir)

    # Save model
    model_path = out_dir / f"{target}_lgbm.txt"
    model.save_model(str(model_path))
    log.info("Model saved: %s", model_path)

    # Save test predictions
    test_copy = test.dropna(subset=[target]).copy()
    test_copy[f"{target}_pred"] = model.predict(X_test)
    pred_path = out_dir / f"{target}_test_predictions.csv"
    test_copy[["Date", "StationName", target, f"{target}_pred"]].to_csv(pred_path, index=False)
    log.info("Test predictions saved: %s", pred_path.name)

    return model, results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXPERIMENT 3 — LightGBM (Temporal Split)")
    log.info("=" * 60)

    train, val, test = load_splits()

    all_results = {}
    models = {}

    for target in TARGETS:
        model, results = train_target(target, train, val, test, OUT_DIR, PLOTS_DIR)
        models[target] = model
        all_results[target] = results

    # Save joblib versions as well (for pipeline use)
    for target, model in models.items():
        joblib.dump(model, OUT_DIR / f"{target}_lgbm.pkl")

    # Final summary report
    log.info("")
    log.info("=" * 60)
    log.info("EXPERIMENT 3 SUMMARY")
    log.info("=" * 60)

    report_lines = [
        "EXPERIMENT 3 — LightGBM (Temporal Split, Clean Data)\n",
        "=" * 60 + "\n",
        f"Train: 2022-01-01 to 2023-06-30  ({len(train):,} rows)\n",
        f"Val:   2023-07-01 to 2023-12-31  ({len(val):,} rows)\n",
        f"Test:  2024-01-01 to 2024-12-30  ({len(test):,} rows)\n",
        f"Features: {len(FEATURE_COLS)}\n\n",
    ]

    for target, results in all_results.items():
        report_lines.append(f"{target}\n" + "-" * 40 + "\n")
        for r in results:
            report_lines.append(
                f"  {r['split']:<6}: R2={r['R2']:.4f}  RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}\n"
            )
        report_lines.append("\n")

        # Log summary
        test_r = [r for r in results if r["split"] == "Test"][0]
        val_r  = [r for r in results if r["split"] == "Val"][0]
        log.info("%s — Val R²=%.4f RMSE=%.2f | Test R²=%.4f RMSE=%.2f",
                 target, val_r["R2"], val_r["RMSE"], test_r["R2"], test_r["RMSE"])

    report_path = OUT_DIR / "exp3_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    log.info("Report saved: %s", report_path)
    log.info("All outputs in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
