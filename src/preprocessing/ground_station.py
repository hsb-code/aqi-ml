"""
Ground station data loading, filtering, deduplication and daily aggregation.
Source: EAD_Hourly_2022-2024_AQ_Points_AQI_dedup.csv
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# QC thresholds (based on EAD sensor spec + literature)
QC = {
    "pm25_max": 500,    # µg/m³ — above this = instrument error
    "pm10_max": 1000,   # µg/m³ — above this = instrument error
    "co_max":   10.0,   # mg/m³ — clip at 10 (sensor saturation)
    "min_valid_hours": 12,  # min hourly obs per day to keep that day
}


def load_ground_station(path: Path) -> pd.DataFrame:
    """
    Load the deduplicated hourly ground station CSV.

    Returns a clean DataFrame with:
        StationName, datetime, PM25, PM10, NO2, SO2, O3, CO, x (lon), y (lat)

    All PM / gas columns remain in their original units:
        PM25, PM10, NO2, SO2, O3  →  µg/m³ (integer)
        CO                        →  mg/m³ (float, clipped at 10)
    """
    logger.info(f"Loading ground station data from {path}")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ── Rename columns for consistency ──────────────────────────────────────
    df = df.rename(columns={"PM2P5": "PM25"})

    # ── Date filter: keep 2022-01-01 onwards only ───────────────────────────
    df = df[df["datetime"] >= "2022-01-01"].copy()
    logger.info(f"  After date filter: {len(df):,} rows")

    # ── Outlier removal ──────────────────────────────────────────────────────
    n_before = len(df)
    df = df[(df["PM25"] >= 0) & (df["PM25"] <= QC["pm25_max"])]
    df = df[(df["PM10"] >= 0) & (df["PM10"] <= QC["pm10_max"])]
    df["CO"] = df["CO"].clip(upper=QC["co_max"])
    logger.info(f"  After QC filter: {len(df):,} rows (removed {n_before - len(df):,})")

    # ── Keep only needed columns ─────────────────────────────────────────────
    cols = ["StationName", "datetime", "PM25", "PM10", "NO2", "SO2", "O3", "CO", "x", "y"]
    df = df[cols].sort_values(["StationName", "datetime"]).reset_index(drop=True)

    logger.info(f"  Stations: {df.StationName.nunique()} | Date range: {df.datetime.min().date()} -> {df.datetime.max().date()}")

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly ground station data to daily means per station.

    Only keeps days with at least `min_valid_hours` valid hourly readings.
    This matches the daily cadence of satellite (S5P, MODIS) data.

    Returns DataFrame with columns:
        StationName, Date, x, y, PM25, PM10, NO2, SO2, O3, CO, valid_hours
    """
    logger.info("Aggregating to daily means per station...")

    df = df.copy()
    df["Date"] = df["datetime"].dt.date

    agg = df.groupby(["StationName", "Date", "x", "y"]).agg(
        PM25=("PM25", "mean"),
        PM10=("PM10", "mean"),
        NO2=("NO2", "mean"),
        SO2=("SO2", "mean"),
        O3=("O3", "mean"),
        CO=("CO", "mean"),
        valid_hours=("PM25", "count"),
    ).reset_index()

    agg["Date"] = pd.to_datetime(agg["Date"])

    # Drop days with insufficient coverage
    n_before = len(agg)
    agg = agg[agg["valid_hours"] >= QC["min_valid_hours"]]
    logger.info(f"  Daily records: {len(agg):,} (dropped {n_before - len(agg):,} low-coverage days)")

    return agg
