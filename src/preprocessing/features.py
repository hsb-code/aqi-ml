"""
Feature engineering: merges all data sources and builds the 21-feature matrix.

Features per final_preprocessing_spec.md:
  Satellite:       NO2_ugm3, AOD, AOD_corrected, AOD_BLH_ratio
  ERA5 met:        T2M_C, D2M_C, SP_hPa, BLH, WindSpeed, WindDirection, RH
  Temporal:        DayOfYear, Month, Season, IsWeekend
  Spatial:         Latitude, Longitude
  Derived/physics: NO2_log, AOD_log, BLH_log, f_RH
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# UAE weekend: Friday = 4, Saturday = 5
UAE_WEEKEND_DAYS = {4, 5}

# Season mapping (month → season index)
SEASON_MAP = {
    12: 1, 1: 1, 2: 1,   # Winter
    3: 2,  4: 2, 5: 2,   # Spring
    6: 3,  7: 3, 8: 3,   # Summer
    9: 4, 10: 4, 11: 4,  # Autumn
}

# Final feature columns (order matters for scaler)
FEATURE_COLS = [
    # Satellite
    "NO2_ugm3", "AOD", "AOD_corrected", "AOD_BLH_ratio",
    # ERA5 meteorology
    "T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH",
    # Temporal
    "DayOfYear", "Month", "Season", "IsWeekend",
    # Spatial
    "Latitude", "Longitude",
    # Static Geography (NEW in Phase 2)
    "Elevation_m", "Dist_Coast_km", "Dist_Corniche_km", "Dist_E11_km",
    # Derived / physics-based
    "NO2_log", "AOD_log", "BLH_log", "f_RH",
    "Wind_U", "Wind_V", "VentilationIndex", "StabilityIndex"
]

TARGET_COLS = ["PM25", "PM10"]


def merge_sources(
    ground: pd.DataFrame,
    no2: pd.DataFrame,
    aod: pd.DataFrame,
    era5: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily ground station data with satellite and ERA5 features.

    All DataFrames must have columns: StationName (or Station), Date.
    Merge is a left join on ground station — keeps all days with ground truth.

    Args:
        ground: daily ground station (from ground_station.aggregate_daily)
        no2:    daily S5P NO2 at stations (processed, has NO2_ugm3)
        aod:    daily MODIS AOD at stations (processed, has AOD)
        era5:   daily ERA5 at stations (processed, has T2M_C, BLH, etc.)
    Returns:
        Merged DataFrame
    """
    # Normalise station/date column names
    for df_ref in [no2, aod, era5]:
        if "Station" in df_ref.columns and "StationName" not in df_ref.columns:
            df_ref.rename(columns={"Station": "StationName"}, inplace=True)

    merge_keys = ["StationName", "Date"]

    merged = ground.merge(no2[merge_keys + ["NO2_ugm3"]], on=merge_keys, how="left")
    merged = merged.merge(aod[merge_keys + ["AOD"]],       on=merge_keys, how="left")
    merged = merged.merge(
        era5[merge_keys + ["T2M_C", "D2M_C", "SP_hPa", "BLH", "WindSpeed", "WindDirection", "RH"]],
        on=merge_keys,
        how="left",
    )

    # ── Merge Static Geography (Option A) ──────────────────────────────────
    import os
    geo_path = Path("data/processed/station_geography.csv")
    if geo_path.exists():
        geo_df = pd.read_csv(geo_path)
        # Drop Lat/Lon from geo_df as we already have them or calculate them in build_features
        geo_df = geo_df.drop(columns=["Latitude", "Longitude"], errors="ignore")
        merged = merged.merge(geo_df, on="StationName", how="left")
        logger.info(f"Merged static geography: {len(geo_df)} stations")
    else:
        logger.warning("Station geography CSV not found — adding placeholder zeros")
        merged["Elevation_m"] = 0.0
        merged["Dist_Coast_km"] = 0.0

    logger.info(
        f"Merged: {len(merged):,} rows | "
        f"NO2 coverage: {merged['NO2_ugm3'].notna().mean():.1%} | "
        f"AOD coverage: {merged['AOD'].notna().mean():.1%} | "
        f"ERA5 coverage: {merged['BLH'].notna().mean():.1%}"
    )
    return merged


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the merged DataFrame.

    Requires columns after merge:
        NO2_ugm3, AOD, BLH, RH, Date, x (lon), y (lat)

    Adds:
        AOD_corrected   – humidity-corrected AOD (removes hygroscopic growth)
        AOD_BLH_ratio   – key predictor: aerosol loading per unit boundary layer
        DayOfYear, Month, Season, IsWeekend
        Latitude, Longitude
        NO2_log, AOD_log, BLH_log
        f_RH            – hygroscopic growth factor
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # ── Hygroscopic growth correction ──────────────────────────────────────
    rh_frac = (df["RH"] / 100.0).clip(0.0, 0.98)
    df["f_RH"] = 1.0 / (1.0 - 0.95 * rh_frac)
    df["AOD_corrected"] = df["AOD"] / df["f_RH"]

    # ── AOD / BLH ratio (core physics feature) ─────────────────────────────
    df["AOD_BLH_ratio"] = df["AOD"] / (df["BLH"] + 1e-6)
    # Cap extreme ratios caused by data errors
    df["AOD_BLH_ratio"] = df["AOD_BLH_ratio"].clip(upper=0.05)

    # ── Temporal features ──────────────────────────────────────────────────
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Month"]     = df["Date"].dt.month
    df["Season"]    = df["Month"].map(SEASON_MAP)
    df["IsWeekend"] = df["Date"].dt.dayofweek.isin(UAE_WEEKEND_DAYS).astype(int)

    # ── Map spatial features ───────────────────────────────────────────────
    df["Latitude"]  = df["y"]
    df["Longitude"] = df["x"]

    # ── Physics-Informed Features ──────────────────────────────────────────
    # 1. Wind Vectors (U = East-West, V = North-South)
    # WindDirection is degrees (0-360), WindSpeed is m/s
    wd_rad = np.deg2rad(df["WindDirection"])
    df["Wind_U"] = df["WindSpeed"] * np.sin(wd_rad)
    df["Wind_V"] = df["WindSpeed"] * np.cos(wd_rad)

    # 2. Ventilation Index (AOD per unit of wind dispersion)
    df["VentilationIndex"] = df["AOD"] / (df["WindSpeed"] + 1.0)

    # 3. Stability Index (Temperature-Dewpoint spread as proxy for inversion)
    df["StabilityIndex"] = df["T2M_C"] - df["D2M_C"]

    # ── Log transforms (reduce skewness of heavy-tailed distributions) ─────
    df["NO2_log"] = np.log1p(df["NO2_ugm3"].clip(lower=0))
    df["AOD_log"] = np.log1p(df["AOD"].clip(lower=0))
    df["BLH_log"] = np.log1p(df["BLH"].clip(lower=0))

    logger.info(f"Feature engineering complete — {len(FEATURE_COLS)} features built")
    return df


def final_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final quality control checks after feature engineering.
    Removes physically impossible combinations.
    """
    n_before = len(df)

    df = df[df["RH"].between(0, 100)]
    df = df[df["BLH"] > 0]
    df = df[df["AOD_BLH_ratio"].between(0, 0.05)]
    df = df[df["SP_hPa"].between(900, 1100)]

    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Final QC removed {n_removed:,} rows")

    # Report missing rates for key features
    logger.info("Missing rates after QC:")
    for col in ["NO2_ugm3", "AOD", "T2M_C", "BLH", "RH"]:
        if col in df.columns:
            pct = 100 * df[col].isna().sum() / len(df)
            logger.info(f"  {col}: {pct:.1f}%")

    return df
