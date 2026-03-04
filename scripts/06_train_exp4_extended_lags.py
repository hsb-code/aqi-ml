"""
scripts/06_train_exp4_extended_lags.py
=======================================
Experiment 4: LightGBM + Extended Lag Features + Optuna + Station Category

Builds on Exp3 by adding more temporal persistence features:
  Existing (from Exp3):
    PM25_lag1, PM10_lag1
    PM25_roll3, PM10_roll3, PM25_roll7, PM10_roll7
    AOD_roll3, NO2_roll3

  NEW in Exp4:
    PM25_lag2, PM10_lag2          – 2-day lag
    PM25_lag3, PM10_lag3          – 3-day lag
    PM25_roll14, PM10_roll14      – 2-week rolling mean
    AOD_roll7, NO2_roll7          – 7-day satellite rolling mean
    PM25_ewm7, PM10_ewm7          – exponentially weighted mean (span=7)
    WindSpeed_roll3               – 3-day wind persistence
    T2M_roll3                     – 3-day temperature persistence

Total new lag features: 12 (on top of existing 8 = 20 lag features total)
Grand total features: 21 base + 20 lag + 1 station = 42

Usage:
  conda activate aqi-ml
  python scripts/06_train_exp4_extended_lags.py
  python scripts/06_train_exp4_extended_lags.py --n-trials 100
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
OUT_DIR       = Path("models/exp_test")
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

# Original lag features (from Exp2/Exp3)
LAG_FEATURE_COLS_ORIGINAL = [
    "PM25_lag1", "PM10_lag1",
    "PM25_roll3", "PM10_roll3",
    "PM25_roll7", "PM10_roll7",
    "AOD_roll3",  "NO2_roll3",
]

# NEW extended lag features for Exp4
LAG_FEATURE_COLS_NEW = [
    "PM25_lag2", "PM10_lag2",
    "PM25_lag3", "PM10_lag3",
    "PM25_roll14", "PM10_roll14",
    "AOD_roll7", "NO2_roll7",
    "PM25_ewm7", "PM10_ewm7",
    "WindSpeed_roll3", "T2M_roll3",
]

ALL_LAG_COLS = LAG_FEATURE_COLS_ORIGINAL + LAG_FEATURE_COLS_NEW

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
# Feature engineering
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add original + extended lag/rolling features PER STATION sorted by date.
    Must be called on the FULL dataset before temporal splitting.
    """
    df = df.sort_values(["StationName", "Date"]).copy()

    for station, grp in df.groupby("StationName", sort=False):
        idx = grp.index

        # ── Original lags (same as Exp2/Exp3) ─────────────────────────────
        df.loc[idx, "PM25_lag1"]   = grp["PM25"].shift(1)
        df.loc[idx, "PM10_lag1"]   = grp["PM10"].shift(1)
        df.loc[idx, "PM25_roll3"]  = grp["PM25"].shift(1).rolling(3,  min_periods=2).mean()
        df.loc[idx, "PM10_roll3"]  = grp["PM10"].shift(1).rolling(3,  min_periods=2).mean()
        df.loc[idx, "PM25_roll7"]  = grp["PM25"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "PM10_roll7"]  = grp["PM10"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "AOD_roll3"]   = grp["AOD"].shift(1).rolling(3,   min_periods=2).mean()
        df.loc[idx, "NO2_roll3"]   = grp["NO2_ugm3"].shift(1).rolling(3, min_periods=2).mean()

        # ── NEW extended lags ──────────────────────────────────────────────
        df.loc[idx, "PM25_lag2"]   = grp["PM25"].shift(2)
        df.loc[idx, "PM10_lag2"]   = grp["PM10"].shift(2)
        df.loc[idx, "PM25_lag3"]   = grp["PM25"].shift(3)
        df.loc[idx, "PM10_lag3"]   = grp["PM10"].shift(3)

        # 14-day rolling mean (min 5 valid days)
        df.loc[idx, "PM25_roll14"] = grp["PM25"].shift(1).rolling(14, min_periods=5).mean()
        df.loc[idx, "PM10_roll14"] = grp["PM10"].shift(1).rolling(14, min_periods=5).mean()

        # 7-day satellite rolling
        df.loc[idx, "AOD_roll7"]   = grp["AOD"].shift(1).rolling(7,  min_periods=3).mean()
        df.loc[idx, "NO2_roll7"]   = grp["NO2_ugm3"].shift(1).rolling(7, min_periods=3).mean()

        # Exponentially-weighted mean (recent days matter more, span=7 ≈ 7-day halflife)
        df.loc[idx, "PM25_ewm7"]   = grp["PM25"].shift(1).ewm(span=7, min_periods=3).mean()
        df.loc[idx, "PM10_ewm7"]   = grp["PM10"].shift(1).ewm(span=7, min_periods=3).mean()

        # Meteorological persistence
        df.loc[idx, "WindSpeed_roll3"] = grp["WindSpeed"].shift(1).rolling(3, min_periods=2).mean()
        df.loc[idx, "T2M_roll3"]       = grp["T2M_C"].shift(1).rolling(3, min_periods=2).mean()

    coverage = df[ALL_LAG_COLS].notna().mean() * 100
    log.info("Lag feature coverage (non-NaN %):")
    for col, pct in coverage.items():
        log.info("  %s: %.1f%%", col, pct)

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
        return np.sqrt(mean_squared_error(y_val, preds))

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
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#1a6fa8")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_xlabel(f"Observed {target} (µg/m³)")
    ax.set_ylabel(f"Predicted {target} (µg/m³)")
    ax.set_title(f"Exp4 LightGBM+ExtLags — {target} ({split})\nR²={r2:.3f}  RMSE={rmse:.2f}")
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

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = []
    for n in importances.index:
        if n == STATION_COL:
            colors.append("#2d8b2d")
        elif any(x in n for x in ["lag", "roll", "ewm"]):
            colors.append("#e07b00")
        else:
            colors.append("#1a6fa8")
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(
        f"Feature Importance (gain) — {target} [Exp4]\n"
        "blue=base  orange=lag/rolling  green=station"
    )
    ax.set_xlabel("Gain")
    plt.tight_layout()
    path = out_dir / f"{target}_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved importance: %s", path.name)


def plot_shap(model, X, target, out_dir):
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)
        plt.figure(figsize=(10, 10))
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False, max_display=30)
        plt.title(f"SHAP Feature Importance — {target} (Exp4)")
        plt.tight_layout()
        path = out_dir / f"{target}_shap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved SHAP: %s", path.name)
    except Exception as e:
        log.warning("SHAP skipped: %s", e)


def plot_optuna_history(study, target, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    trials_df = study.trials_dataframe()
    axes[0].plot(trials_df.index, trials_df["value"], "o-", markersize=3, color="#1a6fa8")
    axes[0].axhline(study.best_value, color="red", linestyle="--",
                    label=f"Best={study.best_value:.3f}")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Val RMSE")
    axes[0].set_title(f"Optuna History — {target} (Exp4)")
    axes[0].legend()
    try:
        param_imp = optuna.importance.get_param_importances(study)
        params = list(param_imp.keys())[:8]
        vals   = [param_imp[p] for p in params]
        axes[1].barh(params[::-1], vals[::-1], color="#1a6fa8")
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
# Train one target
# ---------------------------------------------------------------------------

def train_target(target, train, val, test, feature_cols, cat_features, n_trials):
    log.info("=" * 60)
    log.info("TARGET: %s  (Exp4 — Extended Lags + Optuna + Station)", target)
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
        show_progress_bar=True,
    )
    log.info("Best trial: RMSE=%.4f", study.best_value)
    log.info("Best params: %s", study.best_params)

    plot_optuna_history(study, target, PLOTS_DIR)

    # ── Retrain with best params ────────────────────────────────────────────
    best_params = {
        "objective": "regression",
        "metric":    "rmse",
        "verbosity": -1,
        "n_jobs":    -1,
        "seed":      42,
        **study.best_params,
    }

    # Find best n_estimators with early stopping on val
    dtrain_only = lgb.Dataset(X_train, label=y_train,
                              categorical_feature=cat_features, free_raw_data=False)
    dval_only   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain_only,
                              categorical_feature=cat_features, free_raw_data=False)
    model_es = lgb.train(best_params, dtrain_only, num_boost_round=2000,
                         valid_sets=[dval_only],
                         callbacks=[lgb.early_stopping(50, verbose=False),
                                    lgb.log_evaluation(-1)])
    best_rounds = int(model_es.best_iteration * 1.1)
    log.info("Retraining on train+val for %d rounds...", best_rounds)

    # Final model: train+val combined
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    dtrain_full = lgb.Dataset(X_trainval, label=y_trainval,
                              categorical_feature=cat_features, free_raw_data=False)
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
    test_clean[["Date", "StationName", target, f"{target}_pred"]].to_csv(
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
    log.info("EXPERIMENT 4 — LightGBM + Extended Lags + Optuna + Station")
    log.info("=" * 60)
    log.info("New lag features: PM25/10_lag2/lag3, roll14, AOD/NO2_roll7, ewm7, wind/temp roll3")
    log.info("Optuna trials per target: %d", args.n_trials)

    # ── Load + prepare data ─────────────────────────────────────────────────
    log.info("Loading full dataset...")
    full = pd.read_parquet(PROCESSED_DIR / "training_data_full.parquet")
    log.info("Full dataset: %s rows", f"{len(full):,}")

    log.info("Engineering lag features (original + extended)...")
    full = add_lag_features(full)

    log.info("Encoding station names...")
    full, station_le = encode_station(full)
    joblib.dump(station_le, OUT_DIR / "station_label_encoder.pkl")
    log.info("Stations (%d): %s", len(station_le.classes_), list(station_le.classes_))

    # ── Split ────────────────────────────────────────────────────────────────
    train, val, test = temporal_split(full)
    log.info("Train: %s | Val: %s | Test: %s",
             f"{len(train):,}", f"{len(val):,}", f"{len(test):,}")

    all_feature_cols = BASE_FEATURE_COLS + ALL_LAG_COLS + [STATION_COL]
    cat_features     = [STATION_COL]
    log.info("Total features: %d (%d base + %d lag + 1 station)",
             len(all_feature_cols), len(BASE_FEATURE_COLS), len(ALL_LAG_COLS))

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
    log.info("EXPERIMENT 4 SUMMARY")
    log.info("=" * 60)

    report = [
        "EXPERIMENT 4 — LightGBM + Extended Lag Features + Optuna + Station\n",
        "=" * 60 + "\n",
        f"Optuna trials per target: {args.n_trials}\n",
        f"Train: 2022-01-01 to 2023-06-30  ({len(train):,} rows)\n",
        f"Val:   2023-07-01 to 2023-12-31  ({len(val):,} rows)\n",
        f"Test:  2024-01-01 to 2024-12-30  ({len(test):,} rows)\n\n",
        f"Features: {len(all_feature_cols)} ({len(BASE_FEATURE_COLS)} base + "
        f"{len(ALL_LAG_COLS)} lag + 1 station)\n\n",
        "New lag features vs Exp3:\n",
        "  + PM25_lag2, PM10_lag2  (2-day lag)\n",
        "  + PM25_lag3, PM10_lag3  (3-day lag)\n",
        "  + PM25_roll14, PM10_roll14  (14-day rolling mean)\n",
        "  + AOD_roll7, NO2_roll7  (7-day satellite rolling)\n",
        "  + PM25_ewm7, PM10_ewm7  (exponentially weighted, span=7)\n",
        "  + WindSpeed_roll3, T2M_roll3  (met persistence)\n\n",
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

    report.append("\nExp1 vs Exp2 vs Exp3 vs Exp4 (Test):\n")
    report.append("  Exp1 PM25 R²=0.2356  PM10 R²=0.2607\n")
    report.append("  Exp2 PM25 R²=0.5357  PM10 R²=0.5290\n")
    report.append("  Exp3 PM25 R²=0.5531  PM10 R²=0.5547\n")
    report.append("  Exp4 PM25 see above  PM10 see above\n")

    report_path = OUT_DIR / "exp4_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report)
    log.info("Report saved: %s", report_path)
    log.info("All outputs in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
