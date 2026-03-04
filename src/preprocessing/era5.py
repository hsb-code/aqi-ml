"""
ERA5 reanalysis data processing: unit conversions and derived variables.

Unit conversions per final_preprocessing_spec.md:
  T2M, D2M: Kelvin  →  Celsius  (subtract 273.15)
  SP:       Pascals →  hPa      (divide by 100)
  BLH, U10, V10: no conversion needed

Derived variables:
  WindSpeed:     √(U10² + V10²)
  WindDirection: atan2(V10, U10) in degrees [0, 360)
  RH:            Magnus formula from T2M and D2M
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

KELVIN_OFFSET = 273.15

# Magnus formula constants (Alduchov & Eskridge 1996)
MAGNUS_A = 17.625
MAGNUS_B = 243.04


def kelvin_to_celsius(t_k: np.ndarray) -> np.ndarray:
    return t_k - KELVIN_OFFSET


def pascals_to_hpa(p_pa: np.ndarray) -> np.ndarray:
    return p_pa / 100.0


def wind_speed(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    return np.sqrt(u10 ** 2 + v10 ** 2)


def wind_direction(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    """Meteorological wind direction in degrees [0, 360)."""
    return np.degrees(np.arctan2(v10, u10)) % 360.0


def relative_humidity(t2m_c: np.ndarray, d2m_c: np.ndarray) -> np.ndarray:
    """
    Relative humidity (%) via the Magnus formula.

    RH = 100 × exp(A × Td / (B + Td)) / exp(A × T / (B + T))

    Args:
        t2m_c: 2m temperature in Celsius
        d2m_c: 2m dewpoint temperature in Celsius
    Returns:
        RH in % clipped to [0, 100]
    """
    sat_vp   = np.exp((MAGNUS_A * t2m_c)  / (MAGNUS_B + t2m_c))
    dp_vp    = np.exp((MAGNUS_A * d2m_c)  / (MAGNUS_B + d2m_c))
    rh = 100.0 * dp_vp / sat_vp
    return np.clip(rh, 0.0, 100.0)


def process_era5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all ERA5 unit conversions and compute derived features.

    Expected input columns (raw ERA5 names):
        T2M   – 2m temperature (K)
        D2M   – 2m dewpoint temperature (K)
        SP    – surface pressure (Pa)
        BLH   – boundary layer height (m)
        U10   – 10m u-wind component (m/s)
        V10   – 10m v-wind component (m/s)

    Adds columns:
        T2M_C, D2M_C, SP_hPa, WindSpeed, WindDirection, RH
    """
    df = df.copy()

    df["T2M_C"]         = kelvin_to_celsius(df["T2M"].values)
    df["D2M_C"]         = kelvin_to_celsius(df["D2M"].values)
    df["SP_hPa"]        = pascals_to_hpa(df["SP"].values)
    df["WindSpeed"]     = wind_speed(df["U10"].values, df["V10"].values)
    df["WindDirection"] = wind_direction(df["U10"].values, df["V10"].values)
    df["RH"]            = relative_humidity(df["T2M_C"].values, df["D2M_C"].values)

    # Validate BLH > 0 (required for NO2 conversion in satellite.py)
    invalid_blh = (df["BLH"] <= 0).sum()
    if invalid_blh > 0:
        logger.warning(f"  {invalid_blh} rows with BLH <= 0 — setting to NaN")
        df.loc[df["BLH"] <= 0, "BLH"] = np.nan

    logger.info(
        f"  ERA5 processed: {len(df):,} rows | "
        f"T2M {df['T2M_C'].min():.1f}–{df['T2M_C'].max():.1f} °C | "
        f"SP {df['SP_hPa'].min():.0f}–{df['SP_hPa'].max():.0f} hPa | "
        f"BLH {df['BLH'].min():.0f}–{df['BLH'].max():.0f} m | "
        f"RH {df['RH'].min():.0f}–{df['RH'].max():.0f} %"
    )
    return df
