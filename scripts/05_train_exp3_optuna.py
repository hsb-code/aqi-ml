"""
scripts/07_train_exp5_optuna.py
================================
Experiment 5: LightGBM + Lag Features + Optuna Hyperparameter Tuning
                                + StationName as categorical feature

Builds on Exp 4 by:
  1. Adding StationName as a native LightGBM categorical feature
     (gives the model location-specific bias correction per station)
  2. Using Optuna Bayesian search to find the best hyperparameters
     (smarter than GridSearch: 50 trials vs thousands of combinations)
  3. Optimising on Val RMSE, evaluating on Test

Usage:
  conda activate aqi-ml
  python scripts/07_train_exp5_optuna.py
  python scripts/07_train_exp5_optuna.py --n-trials 100   # more tuning
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
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUT_DIR       = Path("models/exp5")
PLOTS_DIR     = OUT_DIR / "plots"
TARGETS       = ["PM25", "PM10"]

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

# StationName will be encoded and appended as a categorical feature
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
# Feature engineering (same as Exp4 — reused here)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["StationName", "Date"]).copy()
    for station, grp in df.groupby("StationName", sort=False):
        idx = grp.index
        df.loc[idx, "PM25_lag1"]  = grp["PM25"].shift(1)
        df.loc[idx, "PM10_lag1"]  = grp["PM10"].shift(1)
        df.loc[idx, "PM25_roll3"] = grp["PM25"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "PM10_roll3"] = grp["PM10"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "PM25_roll7"] = grp["PM25"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "PM10_roll7"] = grp["PM10"].shift(1).rolling(7, min_periods=3).mean()
        df.loc[idx, "AOD_roll3"]  = grp["AOD"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "NO2_roll3"]  = grp["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean()
    return df


def encode_station(full: pd.DataFrame):
    """Encode StationName as integer, fit on full dataset."""
    le = LabelEncoder()
    full = full.copy()
    full[STATION_COL] = le.fit_transform(full["StationName"])
    return full, le


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


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(X_train, y_train, X_val, y_val, cat_features):
    """Return an Optuna objective function for one target."""
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features,
                         free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain,
                         categorical_feature=cat_features, free_raw_data=False)

    def objective(trial):
        params = {
            "objective":         "regression",
            "metric":            "rmse",
            "verbosity":         -1,
            "n_jobs":            -1,
            "seed":              42,
            # Tuned hyperparameters
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "max_depth":         trial.suggest_int("max_depth", 5, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        }
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        model = lgb.train(params, dtrain, num_boost_round=2000,
                          valid_sets=[dval], callbacks=callbacks)
        preds = model.predict(X_val)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    return objective


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, split_name: str) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    log.info("  %-6s R²=%.4f | RMSE=%.2f | MAE=%.2f", split_name, r2, rmse, mae)
    return {"split": split_name, "R2": r2, "RMSE": rmse, "MAE": mae}


def plot_scatter(y_true, y_pred, target, split, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#7b2d8b")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_xlabel(f"Observed {target} (ug/m3)")
    ax.set_ylabel(f"Predicted {target} (ug/m3)")
    ax.set_title(f"Exp5 LightGBM+Optuna — {target} ({split})\nR²={r2:.3f}  RMSE={rmse:.2f}")
    plt.tight_layout()
    path = out_dir / f"{target}_{split}_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved scatter: %s", path.name)


def plot_feature_importance(model, target, out_dir):
    importances = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=model.feature_name(),
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = []
    for n in importances.index:
        if "lag" in n or "roll" in n:
            colors.append("#e07b00")
        elif n == STATION_COL:
            colors.append("#2d8b2d")
        else:
            colors.append("#7b2d8b")
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(
        f"Feature Importance — {target} (Exp5)\n"
        "purple=base  orange=lag  green=station"
    )
    ax.set_xlabel("Gain")
    plt.tight_layout()
    path = out_dir / f"{target}_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved importance: %s", path.name)


def plot_shap(model, X, target, out_dir):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        plt.figure(figsize=(9, 8))
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False, max_display=25)
        plt.title(f"SHAP Feature Importance — {target} (Exp5)")
        plt.tight_layout()
        path = out_dir / f"{target}_shap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved SHAP: %s", path.name)
    except Exception as e:
        log.warning("SHAP skipped: %s", e)


def plot_optuna_history(study, target, out_dir):
    """Plot Optuna optimisation history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Optimisation history
    trials_df = study.trials_dataframe()
    axes[0].plot(trials_df.index, trials_df["value"], "o-", markersize=3, color="#7b2d8b")
    axes[0].axhline(study.best_value, color="red", linestyle="--", label=f"Best={study.best_value:.3f}")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Val RMSE")
    axes[0].set_title(f"Optuna History — {target}")
    axes[0].legend()
    # Parameter importances
    try:
        param_imp = optuna.importance.get_param_importances(study)
        params = list(param_imp.keys())[:8]
        vals   = [param_imp[p] for p in params]
        axes[1].barh(params[::-1], vals[::-1], color="#7b2d8b")
        axes[1].set_title(f"Optuna Param Importance — {target}")
        axes[1].set_xlabel("Importance")
    except Exception:
        axes[1].text(0.5, 0.5, "Param importance\nnot available", ha="center", va="center")
    plt.tight_layout()
    path = out_dir / f"{target}_optuna_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved Optuna history: %s", path.name)


# ---------------------------------------------------------------------------
# Main train function per target
# ---------------------------------------------------------------------------

def train_target(target, train, val, test, feature_cols, cat_features, n_trials):
    log.info("=" * 60)
    log.info("TARGET: %s  (Exp5 — Optuna + Station Category)", target)
    log.info("=" * 60)

    X_train, y_train, _          = prepare_xy(train, target, feature_cols)
    X_val,   y_val,   _          = prepare_xy(val,   target, feature_cols)
    X_test,  y_test,  test_clean = prepare_xy(test,  target, feature_cols)

    log.info("Train: %s | Val: %s | Test: %s | Features: %d",
             f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}", X_train.shape[1])

    # ── Optuna search ──────────────────────────────────────────────────────
    log.info("Running Optuna search (%d trials)...", n_trials)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        make_objective(X_train, y_train, X_val, y_val, cat_features),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    log.info("Best trial: RMSE=%.4f", study.best_value)
    log.info("Best params: %s", study.best_params)

    # Save Optuna history plot
    plot_optuna_history(study, target, PLOTS_DIR)

    # ── Retrain with best params on train+val combined ─────────────────────
    best_params = {
        "objective":         "regression",
        "metric":            "rmse",
        "verbosity":         -1,
        "n_jobs":            -1,
        "seed":              42,
        **study.best_params,
    }

    # Combine train+val for final model, evaluate on test only
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    dtrain_full = lgb.Dataset(X_trainval, label=y_trainval,
                              categorical_feature=cat_features, free_raw_data=False)
    # Use best iteration from the study's best trial * 1.1 as fixed rounds
    # (common practice when retraining on train+val without early stopping)
    # First: get best n_estimators by retraining with best params and early stopping on val
    dtrain_only = lgb.Dataset(X_train,   label=y_train,
                              categorical_feature=cat_features, free_raw_data=False)
    dval_only   = lgb.Dataset(X_val,     label=y_val,  reference=dtrain_only,
                              categorical_feature=cat_features, free_raw_data=False)
    callbacks_es = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    model_es = lgb.train(best_params, dtrain_only, num_boost_round=2000,
                         valid_sets=[dval_only], callbacks=callbacks_es)
    best_rounds = int(model_es.best_iteration * 1.1)
    log.info("Retraining on train+val for %d rounds...", best_rounds)

    model = lgb.train(best_params, dtrain_full, num_boost_round=best_rounds,
                      callbacks=[lgb.log_evaluation(-1)])

    # ── Evaluate ───────────────────────────────────────────────────────────
    log.info("--- Metrics ---")
    results = [
        compute_metrics(y_train, model.predict(X_train), "Train"),
        compute_metrics(y_val,   model.predict(X_val),   "Val"),
        compute_metrics(y_test,  model.predict(X_test),  "Test"),
    ]

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_scatter(y_val,  model.predict(X_val),  target, "val",  PLOTS_DIR)
    plot_scatter(y_test, model.predict(X_test), target, "test", PLOTS_DIR)
    plot_feature_importance(model, target, PLOTS_DIR)
    plot_shap(model, X_val, target, PLOTS_DIR)

    # ── Save ───────────────────────────────────────────────────────────────
    model.save_model(str(OUT_DIR / f"{target}_lgbm.txt"))
    joblib.dump({"model": model, "best_params": best_params,
                 "best_rounds": best_rounds}, OUT_DIR / f"{target}_lgbm.pkl")
    joblib.dump(study, OUT_DIR / f"{target}_optuna_study.pkl")

    test_clean = test_clean.copy()
    test_clean[f"{target}_pred"] = model.predict(X_test)
    test_clean[["Date","StationName",target,f"{target}_pred"]].to_csv(
        OUT_DIR / f"{target}_test_predictions.csv", index=False)

    log.info("Model + study saved to %s", OUT_DIR)
    return model, study, results


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of Optuna trials per target (default 50)")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXPERIMENT 5 — LightGBM + Optuna + Station (Temporal Split)")
    log.info("=" * 60)
    log.info("Optuna trials per target: %d", args.n_trials)

    # ── Load + prepare data ─────────────────────────────────────────────────
    log.info("Loading full dataset...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet")

    log.info("Engineering lag features...")
    full = add_lag_features(full)

    log.info("Encoding station names...")
    full, station_le = encode_station(full)
    joblib.dump(station_le, OUT_DIR / "station_label_encoder.pkl")
    log.info("Stations: %s", list(station_le.classes_))

    # ── Split ────────────────────────────────────────────────────────────────
    train, val, test = temporal_split(full)
    log.info("Train: %s | Val: %s | Test: %s",
             f"{len(train):,}", f"{len(val):,}", f"{len(test):,}")

    all_feature_cols = BASE_FEATURE_COLS + LAG_FEATURE_COLS + [STATION_COL]
    cat_features     = [STATION_COL]
    log.info("Total features: %d (%d base + %d lag + 1 station)",
             len(all_feature_cols), len(BASE_FEATURE_COLS), len(LAG_FEATURE_COLS))

    # ── Train ────────────────────────────────────────────────────────────────
    all_results = {}
    for target in TARGETS:
        _, _, results = train_target(
            target, train, val, test, all_feature_cols, cat_features, args.n_trials
        )
        all_results[target] = results

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("EXPERIMENT 5 SUMMARY")
    log.info("=" * 60)

    report = [
        "EXPERIMENT 5 — LightGBM + Optuna Tuning + Station Category\n",
        "=" * 60 + "\n",
        f"Optuna trials per target: {args.n_trials}\n",
        f"Train: 2022-01-01 to 2023-06-30  ({len(train):,} rows)\n",
        f"Val:   2023-07-01 to 2023-12-31  ({len(val):,} rows)\n",
        f"Test:  2024-01-01 to 2024-12-30  ({len(test):,} rows)\n\n",
        f"Features: {len(all_feature_cols)} ({len(BASE_FEATURE_COLS)} base + "
        f"{len(LAG_FEATURE_COLS)} lag + 1 station)\n\n",
    ]

    for target, results in all_results.items():
        report.append(f"{target}\n" + "-" * 40 + "\n")
        for r in results:
            report.append(f"  {r['split']:<6}: R2={r['R2']:.4f}  "
                          f"RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}\n")
        report.append("\n")
        test_r = next(r for r in results if r["split"] == "Test")
        val_r  = next(r for r in results if r["split"] == "Val")
        log.info("%s — Val R²=%.4f RMSE=%.2f | Test R²=%.4f RMSE=%.2f",
                 target, val_r["R2"], val_r["RMSE"], test_r["R2"], test_r["RMSE"])

    report.append("\nExp3 vs Exp4 vs Exp5 (Test):\n")
    report.append("  Exp3 PM25 R²=0.2356  PM10 R²=0.2607\n")
    report.append("  Exp4 PM25 R²=0.5357  PM10 R²=0.5290\n")
    report.append("  Exp5 PM25 see above  PM10 see above\n")

    report_path = OUT_DIR / "exp5_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report)
    log.info("Report saved: %s", report_path)
    log.info("All outputs in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
