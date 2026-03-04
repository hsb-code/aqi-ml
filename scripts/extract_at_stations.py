"""
scripts/extract_at_stations.py
===============================
Extract satellite (S5P NO2, MODIS AOD) and ERA5 values at the 20 ground
station coordinates from the raw raster/NetCDF files.

Outputs written to data/processed/:
  no2_at_stations.csv   -- Date, StationName, NO2_mol_m2
  aod_at_stations.csv   -- Date, StationName, AOD_DN  (raw DN; x0.001 = true AOD)
  era5_at_stations.csv  -- Date, StationName, T2M, D2M, U10, V10, SP, BLH

Usage:
  conda activate aqi-ml
  python scripts/extract_at_stations.py
  python scripts/extract_at_stations.py --skip-era5   # just NO2 + AOD
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import rowcol

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
GROUND_STATION_CSV = ROOT / "00_Ancillary_Data" / "EAD_Hourly_2022-2024_AQ_Points_AQI_dedup.csv"

DEFAULT_NO2_DIR   = ROOT / "data" / "raw" / "NO2"
DEFAULT_AOD_DIR   = ROOT / "data" / "raw" / "MODIS_AOD"
DEFAULT_ERA5_FILE = ROOT / "data" / "raw" / "ERA5" / "ERA5_merged_20220101_20241231.nc"
OUTPUT_DIR        = ROOT / "data" / "processed"

# Empty/sentinel TIF files have exactly 2540 bytes; skip anything smaller than 5 KB
MIN_TIF_BYTES = 5_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stations(ground_csv: Path) -> pd.DataFrame:
    """Return unique (StationName, x=lon, y=lat) for the 20 stations."""
    df = pd.read_csv(ground_csv, usecols=["StationName", "x", "y"])
    stations = df.drop_duplicates("StationName").reset_index(drop=True)
    log.info("Loaded %d station locations", len(stations))
    return stations


def sample_tif(tif_path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Sample the first band of a GeoTIFF at each (lon, lat) point.
    Returns float64 array of shape (n_stations,); out-of-bounds or nodata -> NaN.
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float64)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        rows, cols = rowcol(src.transform, lons, lats)
        rows = np.array(rows)
        cols = np.array(cols)

        out = np.full(len(lons), np.nan, dtype=np.float64)
        in_bounds = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        out[in_bounds] = data[rows[in_bounds], cols[in_bounds]]
    return out


# ---------------------------------------------------------------------------
# NO2 extraction
# ---------------------------------------------------------------------------

def extract_no2(no2_dir: Path, stations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Sample S5P NO2 (mol/m^2) from daily TIFs at each station."""
    log.info("=== Extracting NO2 ===")
    tifs = sorted(
        f for f in no2_dir.glob("NO2_*.tif")
        if f.suffix == ".tif" and f.stat().st_size >= MIN_TIF_BYTES
    )
    log.info("  %d valid NO2 TIF files found", len(tifs))

    lons  = stations["x"].values
    lats  = stations["y"].values
    names = stations["StationName"].values

    rows = []
    for tif in tifs:
        date_str = tif.stem[4:]          # strip "NO2_"
        try:
            pd.to_datetime(date_str)
        except Exception:
            continue

        vals = sample_tif(tif, lons, lats)
        for name, val in zip(names, vals):
            rows.append({"Date": date_str, "StationName": name, "NO2_mol_m2": val})

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year.between(2022, 2024)].copy()
    df = df.sort_values(["Date", "StationName"]).reset_index(drop=True)

    out = output_dir / "no2_at_stations.csv"
    df.to_csv(out, index=False)
    log.info(
        "  Saved %s rows (%s non-NaN) -> %s",
        f"{len(df):,}", f"{df['NO2_mol_m2'].notna().sum():,}", out.name
    )
    return df


# ---------------------------------------------------------------------------
# MODIS AOD extraction
# ---------------------------------------------------------------------------

def extract_aod(aod_dir: Path, stations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Sample MODIS AOD from daily TIFs at each station.
    Raw values are DN (integer); true AOD = DN x 0.001 (applied in features.py).
    """
    log.info("=== Extracting MODIS AOD ===")
    tifs = sorted(
        f for f in aod_dir.glob("MODIS_AOD_*.tif")
        if f.suffix == ".tif" and f.stat().st_size >= MIN_TIF_BYTES
    )
    log.info("  %d valid MODIS AOD TIF files found", len(tifs))

    lons  = stations["x"].values
    lats  = stations["y"].values
    names = stations["StationName"].values

    rows = []
    for tif in tifs:
        date_str = tif.stem[10:]         # strip "MODIS_AOD_"
        try:
            pd.to_datetime(date_str)
        except Exception:
            continue

        vals = sample_tif(tif, lons, lats)
        vals[vals == 0] = np.nan         # MODIS uses 0 as nodata in sparse tiles

        for name, val in zip(names, vals):
            rows.append({"Date": date_str, "StationName": name, "AOD_DN": val})

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year.between(2022, 2024)].copy()
    df = df.sort_values(["Date", "StationName"]).reset_index(drop=True)

    out = output_dir / "aod_at_stations.csv"
    df.to_csv(out, index=False)
    log.info(
        "  Saved %s rows (%s non-NaN) -> %s",
        f"{len(df):,}", f"{df['AOD_DN'].notna().sum():,}", out.name
    )
    return df


# ---------------------------------------------------------------------------
# ERA5 extraction
# ---------------------------------------------------------------------------

def extract_era5(era5_file: Path, stations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Resample ERA5 hourly data to daily means, then sample nearest grid point
    for each station (vectorised over all dates at once).

    Variable mapping expected in the NetCDF:
      t2m, d2m, u10, v10, sp, blh
    """
    log.info("=== Extracting ERA5 ===")
    ds = xr.open_dataset(era5_file)

    # Drop scalar non-dimension coords that cause pandas duplicate-index errors
    drop_coords = [c for c in ["number", "expver", "step"] if c in ds.coords and c not in ds.dims]
    if drop_coords:
        ds = ds.drop_vars(drop_coords)
        log.info("  Dropped extra coords: %s", drop_coords)

    # Normalise coordinate names (ERA5 CDS uses 'valid_time', 'latitude', 'longitude')
    renames = {}
    for old, new in [("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon")]:
        if old in ds.coords and new not in ds.coords:
            renames[old] = new
    if renames:
        ds = ds.rename(renames)

    vars_found = list(ds.data_vars)
    log.info("  Variables: %s", vars_found)
    log.info(
        "  Raw time range: %s -> %s",
        str(ds.time.values[0])[:10],
        str(ds.time.values[-1])[:10],
    )

    # Resample hourly -> daily mean
    log.info("  Resampling to daily means...")
    ds_daily = ds.resample(time="1D").mean()

    # Filter 2022-2024
    ds_daily = ds_daily.sel(time=slice("2022-01-01", "2024-12-31"))
    n_days = ds_daily.dims["time"]
    log.info("  Daily timesteps in 2022-2024: %d", n_days)

    frames = []
    for _, row in stations.iterrows():
        station = row["StationName"]
        lon = float(row["x"])
        lat = float(row["y"])

        # nearest-neighbour point selection -- returns (time,) DataArrays
        pt = ds_daily.sel(lat=lat, lon=lon, method="nearest")
        dates = pd.to_datetime(pt.time.values).normalize()

        frame = pd.DataFrame({"Date": dates, "StationName": station})
        for var, col in [("t2m","T2M"), ("d2m","D2M"), ("u10","U10"),
                         ("v10","V10"), ("sp","SP"), ("blh","BLH")]:
            frame[col] = pt[var].values if var in pt else np.nan

        frames.append(frame)
        log.info("    %s: %d records", station, len(frame))

    ds.close()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Date", "StationName"]).reset_index(drop=True)

    out = output_dir / "era5_at_stations.csv"
    df.to_csv(out, index=False)
    log.info("  Saved %s rows -> %s", f"{len(df):,}", out.name)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract satellite + ERA5 data at ground station coordinates"
    )
    p.add_argument("--no2-dir",    type=Path, default=DEFAULT_NO2_DIR)
    p.add_argument("--aod-dir",    type=Path, default=DEFAULT_AOD_DIR)
    p.add_argument("--era5-file",  type=Path, default=DEFAULT_ERA5_FILE)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--skip-no2",   action="store_true")
    p.add_argument("--skip-aod",   action="store_true")
    p.add_argument("--skip-era5",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stations = load_stations(GROUND_STATION_CSV)

    if not args.skip_no2:
        extract_no2(args.no2_dir, stations, args.output_dir)

    if not args.skip_aod:
        extract_aod(args.aod_dir, stations, args.output_dir)

    if not args.skip_era5:
        extract_era5(args.era5_file, stations, args.output_dir)

    log.info("Extraction complete. Files saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
