"""
Top-level preprocessing pipeline orchestrator.

Runs all steps:
  1. Load & filter ground station data
  2. Load & process S5P NO2 (unit conversion + QA)
  3. Load & process MODIS AOD (scale factor)
  4. Load & process ERA5 (unit conversion + derived vars)
  5. Merge all sources
  6. Feature engineering
  7. Final QC
  8. Temporal split: Train / Val / Test
  9. Scale features (RobustScaler, fit on train only)
  10. Save parquet files + scaler

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --ground-station path/to/file.csv
"""

import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import RobustScaler

from src.preprocessing.ground_station import load_ground_station, aggregate_daily
from src.preprocessing.features import (
    merge_sources,
    build_features,
    final_qc,
    FEATURE_COLS,
    TARGET_COLS,
)

logger = logging.getLogger(__name__)

# ── Default paths (relative to project root) ──────────────────────────────
DEFAULT_PATHS = {
    "ground_station": Path("00_Ancillary_Data/EAD_Hourly_2022-2024_AQ_Points_AQI_dedup.csv"),
    "no2_csv":        Path("data/processed/no2_at_stations.csv"),
    "aod_csv":        Path("data/processed/aod_at_stations.csv"),
    "era5_csv":       Path("data/processed/era5_at_stations.csv"),
    "output_dir":     Path("data/processed"),
    "model_dir":      Path("models"),
}

# ── Temporal split boundaries ─────────────────────────────────────────────
TRAIN_END = "2023-06-30"   # up to and including
VAL_END   = "2023-12-31"   # up to and including
# Test = everything after VAL_END (2024-01-01 → 2024-12-31)


def load_no2_csv(path: Path) -> pd.DataFrame:
    """Load pre-extracted S5P NO2 at stations (must have NO2, BLH columns)."""
    df = pd.read_csv(path, parse_dates=["Date"])

    # Rename raw column if needed
    if "NO2" in df.columns and "NO2_mol_m2" not in df.columns:
        df = df.rename(columns={"NO2": "NO2_mol_m2"})

    required = {"StationName", "Date", "NO2_mol_m2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NO2 CSV missing columns: {missing}")

    logger.info(f"Loaded NO2 CSV: {len(df):,} rows from {path}")
    return df


def load_aod_csv(path: Path) -> pd.DataFrame:
    """Load pre-extracted MODIS AOD at stations (raw DN or already scaled)."""
    df = pd.read_csv(path, parse_dates=["Date"])

    # Accept column named AOD_DN (from extract_at_stations.py) or AOD / AOD_raw
    if "AOD_DN" in df.columns:
        df = df.rename(columns={"AOD_DN": "AOD_raw"})

    # Detect whether scale factor was already applied
    if "AOD" in df.columns:
        if df["AOD"].max() > 10:
            logger.warning("AOD values appear to be raw DN -- applying x0.001 scale factor")
            df["AOD_raw"] = df["AOD"]
            df["AOD"] = df["AOD"] * 0.001
        else:
            logger.info("AOD values appear already scaled (max < 10) -- no scale factor applied")
    elif "AOD_raw" in df.columns:
        df["AOD"] = df["AOD_raw"] * 0.001
    else:
        raise ValueError("AOD CSV must have 'AOD', 'AOD_DN', or 'AOD_raw' column")

    # Range filter
    df = df[(df["AOD"] >= 0) & (df["AOD"] <= 5.0)]
    logger.info("Loaded AOD CSV: %s rows from %s", f"{len(df):,}", path)
    return df



def load_era5_csv(path: Path) -> pd.DataFrame:
    """Load pre-extracted ERA5 at stations and apply unit conversions."""
    from src.preprocessing.era5 import process_era5
    df = pd.read_csv(path, parse_dates=["Date"])
    df = process_era5(df)
    logger.info(f"Loaded ERA5 CSV: {len(df):,} rows from {path}")
    return df


def temporal_split(df: pd.DataFrame):
    """
    Temporal train / val / test split — NO random shuffling.

        Train: <= 2023-06-30
        Val:   2023-07-01 – 2023-12-31
        Test:  2024-01-01 – 2024-12-31
    """
    df = df.sort_values("Date")
    train = df[df["Date"] <= TRAIN_END]
    val   = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)]
    test  = df[df["Date"] > VAL_END]

    logger.info(
        "Split sizes -- Train: %s (%s to %s) | Val: %s (%s to %s) | Test: %s (%s to %s)",
        f"{len(train):,}", train.Date.min().date(), train.Date.max().date(),
        f"{len(val):,}",   val.Date.min().date(),   val.Date.max().date(),
        f"{len(test):,}",  test.Date.min().date(),  test.Date.max().date(),
    )
    return train, val, test


def run_pipeline(
    ground_station_path: Path = None,
    no2_path: Path = None,
    aod_path: Path = None,
    era5_path: Path = None,
    output_dir: Path = None,
    model_dir: Path = None,
) -> dict:
    """
    Run the full preprocessing pipeline end-to-end.

    Returns a dict with keys: train, val, test, scaler, feature_cols, target_cols
    """
    # ── Resolve paths ──────────────────────────────────────────────────────
    gs_path  = ground_station_path or DEFAULT_PATHS["ground_station"]
    no2_p    = no2_path   or DEFAULT_PATHS["no2_csv"]
    aod_p    = aod_path   or DEFAULT_PATHS["aod_csv"]
    era5_p   = era5_path  or DEFAULT_PATHS["era5_csv"]
    out_dir  = output_dir or DEFAULT_PATHS["output_dir"]
    mdl_dir  = model_dir  or DEFAULT_PATHS["model_dir"]

    out_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1-2: Ground station ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading ground station data")
    gs_hourly = load_ground_station(gs_path)
    gs_daily  = aggregate_daily(gs_hourly)

    # ── Step 3: S5P NO2 ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Loading S5P NO2")
    no2_df = load_no2_csv(no2_p)
    # NO2 conversion requires BLH — merge ERA5 BLH first
    era5_raw = pd.read_csv(era5_p, parse_dates=["Date"])
    if "BLH" in era5_raw.columns:
        no2_df = no2_df.merge(
            era5_raw[["StationName","Date","BLH"]],
            on=["StationName","Date"], how="left"
        )
        from src.preprocessing.satellite import convert_no2_to_ugm3
        no2_df["NO2_ugm3"] = convert_no2_to_ugm3(
            no2_df["NO2_mol_m2"].values,
            no2_df["BLH"].values,
        )
        no2_df = no2_df[no2_df["NO2_ugm3"].notna() & (no2_df["NO2_ugm3"] >= 0)]
    else:
        logger.warning("BLH not found in ERA5 — NO2 conversion will be skipped")
        no2_df["NO2_ugm3"] = np.nan

    # ── Step 4: MODIS AOD ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Loading MODIS AOD")
    aod_df = load_aod_csv(aod_p)

    # ── Step 5: ERA5 ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Processing ERA5")
    era5_df = load_era5_csv(era5_p)

    # ── Step 6: Merge ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Merging all sources")
    merged = merge_sources(gs_daily, no2_df, aod_df, era5_df)

    # ── Step 7: Feature engineering ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Feature engineering")
    merged = build_features(merged)

    # ── Step 8: Final QC ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Final QC")
    merged = final_qc(merged)

    # Save full dataset
    full_path = out_dir / "training_data_full.parquet"
    merged.to_parquet(full_path, index=False)
    logger.info(f"Full dataset saved: {full_path} ({len(merged):,} rows)")

    # ── Step 9: Temporal split ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Temporal split")
    train, val, test = temporal_split(merged)

    # ── Step 10: Feature scaling ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9: Scaling features (RobustScaler, fit on train only)")

    # Only scale rows where all features are available for scaler fit
    train_complete = train[FEATURE_COLS].dropna()
    scaler = RobustScaler()
    scaler.fit(train_complete)

    def apply_scale(df_split):
        df_out = df_split.copy()
        mask = df_out[FEATURE_COLS].notna().all(axis=1)
        df_out.loc[mask, [f + "_scaled" for f in FEATURE_COLS]] = scaler.transform(
            df_out.loc[mask, FEATURE_COLS]
        )
        return df_out

    train = apply_scale(train)
    val   = apply_scale(val)
    test  = apply_scale(test)

    scaler_path = mdl_dir / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved: {scaler_path}")

    # ── Step 11: Save split parquets ───────────────────────────────────────
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.parquet"
        split.to_parquet(path, index=False)
        logger.info(f"Saved {name}: {path} ({len(split):,} rows)")

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"  Total rows:  {len(merged):,}")
    logger.info(f"  Train:       {len(train):,}")
    logger.info(f"  Val:         {len(val):,}")
    logger.info(f"  Test:        {len(test):,}")
    logger.info(f"  Features:    {len(FEATURE_COLS)}")
    logger.info(f"  Targets:     {TARGET_COLS}")
    logger.info("=" * 60)

    return {
        "train": train,
        "val":   val,
        "test":  test,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "target_cols":  TARGET_COLS,
    }
