"""
Satellite data processing: S5P TROPOMI NO2 and MODIS MCD19A2 AOD.

Unit conversions per final_preprocessing_spec.md:
  S5P NO2:   mol/m²  →  µg/m³  via  (mol_m2 × 46.0055 × 1e6) / BLH_m
  MODIS AOD: integer DN  →  dimensionless  via  DN × 0.001
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Physical constants
NO2_MOLAR_MASS = 46.0055   # g/mol
MOL_TO_UGM3_FACTOR = 1e6  # g/m³ → µg/m³

# QA / validity ranges
QA_VALUE_MIN    = 0.75     # Copernicus TROPOMI official QA threshold
AOD_SCALE       = 0.001    # MODIS MCD19A2 scale factor (DN → optical depth)
AOD_VALID_MAX   = 5.0      # Physical upper limit for optical depth
AOD_VALID_MIN   = 0.0


# ── S5P NO2 ─────────────────────────────────────────────────────────────────

def convert_no2_to_ugm3(no2_mol_m2: np.ndarray, blh_m: np.ndarray) -> np.ndarray:
    """
    Convert S5P NO2 column density to near-surface concentration.

    Formula:
        NO2_ugm3 = (no2_mol_m2 × NO2_MOLAR_MASS × 1e6) / blh_m

    This divides the total column loading (mol/m²) by the boundary layer
    height to estimate the near-surface concentration (µg/m³), assuming
    uniform vertical mixing within the boundary layer.

    Args:
        no2_mol_m2: NO2 column density in mol/m²  (raw S5P value, ~0.0001)
        blh_m:      Boundary layer height in metres from ERA5

    Returns:
        NO2 near-surface concentration in µg/m³
    """
    blh_safe = np.where(blh_m > 0, blh_m, np.nan)
    no2_ugm3 = (no2_mol_m2 * NO2_MOLAR_MASS * MOL_TO_UGM3_FACTOR) / blh_safe
    return no2_ugm3


def apply_no2_qa_filter(df: pd.DataFrame, qa_col: str = "qa_value") -> pd.DataFrame:
    """
    Apply the official Copernicus TROPOMI QA filter.
    Rows with qa_value <= 0.75 are set to NaN and dropped.
    """
    if qa_col not in df.columns:
        logger.warning(f"Column '{qa_col}' not found — skipping QA filter")
        return df

    n_before = len(df)
    df = df[df[qa_col] > QA_VALUE_MIN].copy()
    logger.info(f"  S5P QA filter (qa > {QA_VALUE_MIN}): kept {len(df):,}/{n_before:,} rows")
    return df


def process_s5p(df: pd.DataFrame, blh_col: str = "BLH") -> pd.DataFrame:
    """
    Full S5P NO2 processing:
      1. Apply QA filter
      2. Convert col density (mol/m²) → surface concentration (µg/m³)
      3. Drop rows where conversion produces invalid values

    Expects columns: NO2_mol_m2, qa_value (optional), BLH (m)
    Adds column:     NO2_ugm3
    """
    df = df.copy()

    # QA filter
    df = apply_no2_qa_filter(df)

    # Unit conversion
    df["NO2_ugm3"] = convert_no2_to_ugm3(
        df["NO2_mol_m2"].values,
        df[blh_col].values,
    )

    # Sanity check — Abu Dhabi range 0-200 µg/m³
    n_invalid = df["NO2_ugm3"].isna().sum() + (df["NO2_ugm3"] < 0).sum()
    if n_invalid > 0:
        logger.warning(f"  {n_invalid} rows with invalid NO2_ugm3 (negative or NaN)")
    df = df[df["NO2_ugm3"].notna() & (df["NO2_ugm3"] >= 0)]

    logger.info(
        f"  S5P NO2 processed: {len(df):,} rows | "
        f"mean={df['NO2_ugm3'].mean():.2f} µg/m³ | "
        f"max={df['NO2_ugm3'].max():.2f} µg/m³"
    )
    return df


# ── MODIS AOD ────────────────────────────────────────────────────────────────

def scale_aod(raw_dn: np.ndarray) -> np.ndarray:
    """
    Apply MODIS MCD19A2 scale factor: optical_depth = DN × 0.001

    Raw DN values in your data are integers (~0–3924).
    After scaling: ~0.0–3.9 (dimensionless optical depth).
    """
    return raw_dn * AOD_SCALE


def process_modis(df: pd.DataFrame, raw_col: str = "AOD_raw") -> pd.DataFrame:
    """
    Full MODIS AOD processing:
      1. Apply ×0.001 scale factor
      2. Filter to valid optical depth range [0, 5]
      3. Optionally apply QA flag filter

    Expects column: AOD_raw (integer DN)
    Adds column:    AOD
    """
    df = df.copy()
    df["AOD"] = scale_aod(df[raw_col].values)

    # Range filter
    n_before = len(df)
    df = df[(df["AOD"] >= AOD_VALID_MIN) & (df["AOD"] <= AOD_VALID_MAX)]
    logger.info(
        f"  MODIS AOD processed: {len(df):,}/{n_before:,} valid rows | "
        f"mean={df['AOD'].mean():.3f} | max={df['AOD'].max():.3f}"
    )
    return df
