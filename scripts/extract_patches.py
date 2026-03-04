"""
scripts/extract_patches.py
==========================
Extract 15x15 pixel patches for S5P NO2 and MODIS AOD around ground stations.
Saves data into a consolidated .npz file for deep learning.
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
STATION_CSV = ROOT / "00_Ancillary_Data" / "EAD_Hourly_2022-2024_AQ_Points_AQI_dedup.csv"
NO2_DIR     = ROOT / "data" / "raw" / "NO2"
AOD_DIR     = ROOT / "data" / "raw" / "MODIS_AOD"
OUTPUT_FILE = ROOT / "data" / "processed" / "station_patches_15x15.npz"

PATCH_SIZE  = 15 # pixels
HALF_PATCH  = PATCH_SIZE // 2

def load_stations():
    df = pd.read_csv(STATION_CSV)
    return df.drop_duplicates("StationName")[["StationName", "x", "y"]].sort_values("StationName")

def extract_patches_from_tif(tif_path, stations):
    """Extract PATCH_SIZE x PATCH_SIZE patches for each station."""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        
        # Pad data to handle edges
        padded = np.pad(data, HALF_PATCH, mode='edge')
        
        patches = []
        rows, cols = rowcol(src.transform, stations["x"].values, stations["y"].values)
        
        for r, c in zip(rows, cols):
            # Adjust for padding
            r_pad, c_pad = r + HALF_PATCH, c + HALF_PATCH
            patch = padded[r_pad-HALF_PATCH : r_pad+HALF_PATCH+1, 
                           c_pad-HALF_PATCH : c_pad+HALF_PATCH+1]
            
            # Ensure exact size
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                patch = np.full((PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
            patches.append(patch)
            
    return np.array(patches)

def main():
    stations = load_stations()
    station_names = stations["StationName"].values
    
    no2_tifs = sorted(NO2_DIR.glob("NO2_*.tif"))
    aod_tifs = sorted(AOD_DIR.glob("MODIS_AOD_*.tif"))
    
    # We'll use a dictionary to align by date
    # Date -> { 'no2': [patches], 'aod': [patches] }
    all_data = {}
    
    log.info(f"Processing {len(no2_tifs)} NO2 TIFs...")
    for tif in no2_tifs:
        date = tif.stem[4:]
        patches = extract_patches_from_tif(tif, stations)
        if date not in all_data: all_data[date] = {}
        all_data[date]['no2'] = patches
        
    log.info(f"Processing {len(aod_tifs)} AOD TIFs...")
    for tif in aod_tifs:
        date = tif.stem[10:]
        patches = extract_patches_from_tif(tif, stations)
        if date not in all_data: all_data[date] = {}
        all_data[date]['aod'] = patches
        
    # Final cleanup: only keep dates with both
    common_dates = sorted([d for d in all_data if 'no2' in all_data[d] and 'aod' in all_data[d]])
    
    # Pack into arrays
    # Shape: (Dates, Stations, Patch, Patch, Channels)
    dates_arr = np.array(common_dates)
    final_patches = np.zeros((len(common_dates), len(station_names), PATCH_SIZE, PATCH_SIZE, 2), dtype=np.float32)
    
    for i, date in enumerate(common_dates):
        final_patches[i, :, :, :, 0] = all_data[date]['no2']
        final_patches[i, :, :, :, 1] = all_data[date]['aod']
        
    log.info(f"Extraction complete. Found {len(common_dates)} common dates.")
    log.info(f"Saving to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE, 
        patches=final_patches, 
        dates=dates_arr, 
        stations=station_names
    )
    log.info("Done.")

if __name__ == "__main__":
    main()
