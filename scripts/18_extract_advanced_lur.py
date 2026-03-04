
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

# Config
DATA_DIR = Path("data/processed")
ANCILLARY_DIR = Path("00_Ancillary_Data")
OUTPUT_FILE = DATA_DIR / "advanced_features.parquet"
TRAINING_DATA = DATA_DIR / "training_data_full.parquet"

def haversine_vectorized(lat1, lon1, lat2, lon2):
    # Vectorized haversine distance
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def extract_lur_features(stations_df):
    log.info("Extracting LUR features from shapefiles...")
    
    # Load Urban Regions
    urban_shp = ANCILLARY_DIR / "AbuDhabi_MixedUrban_Regions.shp"
    if not urban_shp.exists():
        log.warning(f"Shapefile not found: {urban_shp}")
        return stations_df
        
    urban_gdf = gpd.read_file(urban_shp)
    # Ensure CRS matches or use lat/lon
    if urban_gdf.crs != "EPSG:4326":
        urban_gdf = urban_gdf.to_crs("EPSG:4326")
        
    # Create Point geometries for stations
    stations_gdf = gpd.GeoDataFrame(
        stations_df, 
        geometry=gpd.points_from_xy(stations_df.Longitude, stations_df.Latitude),
        crs="EPSG:4326"
    )
    
    # 1. Urban Density (within 5km buffer)
    log.info("Computing Urban Density (5km buffer)...")
    # Project to metric CRS for accurate buffering (UTM 40N is good for UAE)
    stations_projected = stations_gdf.to_crs("EPSG:32640")
    urban_projected = urban_gdf.to_crs("EPSG:32640")
    
    buffers = stations_projected.buffer(5000) # 5km
    urban_area_5km = []
    
    for i, buf in enumerate(buffers):
        # Intersection of buffer and urban polygons
        intersect = urban_projected.intersection(buf)
        area = intersect.area.sum() / (np.pi * 5000**2) # Fraction of buffer that is urban
        urban_area_5km.append(area)
        
    stations_df['UrbanDensity_5km'] = urban_area_5km
    return stations_df

def compute_drp_features(df):
    log.info("Computing Dynamic Regional Persistence (DRP) with 7-day windows...")
    
    stations = df[['StationName', 'Latitude', 'Longitude']].drop_duplicates()
    dates = df['Date'].unique()
    
    # We use multiple lags to capture regional memory
    # target_cols = ['PM25', 'PM10']
    
    # Pre-calculate distances and weights
    coords = stations[['Latitude', 'Longitude']].values
    dist_matrix = np.zeros((len(stations), len(stations)))
    for i in range(len(stations)):
        dist_matrix[i] = haversine_vectorized(coords[i,0], coords[i,1], coords[:,0], coords[:,1])
    
    np.fill_diagonal(dist_matrix, np.inf)
    weights = 1.0 / (dist_matrix**2 + 0.1)
    weights = weights / weights.sum(axis=1)[:, None]
    
    # Create pivot for PM values for each date
    pivots = {
        'PM25': df.pivot(index='Date', columns='StationName', values='PM25').ffill().fillna(0),
        'PM10': df.pivot(index='Date', columns='StationName', values='PM10').ffill().fillna(0)
    }
    
    # Compute regional averages for each day for all stations
    regional_daily = {}
    for target, pivot in pivots.items():
        # Shape: (Dates, Stations)
        regional_daily[target] = pivot.values @ weights.T # (Dates, Stations)
    
    result_dfs = []
    
    log.info("Computing sliding regional windows...")
    for station_idx, station_name in enumerate(stations['StationName']):
        station_drp = pd.DataFrame(index=dates)
        station_drp['StationName'] = station_name
        
        for target in ['PM25', 'PM10']:
            # Regional values for THIS station (IDW sum of others)
            reg_series = pd.Series(regional_daily[target][:, station_idx], index=dates)
            
            # Now compute lags and rolls on the regional series
            station_drp[f"DRP_{target}_lag1"] = reg_series.shift(1)
            station_drp[f"DRP_{target}_roll3"] = reg_series.shift(1).rolling(3, min_periods=1).mean()
            station_drp[f"DRP_{target}_roll7"] = reg_series.shift(1).rolling(7, min_periods=1).mean()
            
        result_dfs.append(station_drp.reset_index().rename(columns={'index': 'Date'}))
        
    drp_df = pd.concat(result_dfs)
    return drp_df

def add_local_lags(df):
    log.info("Adding local lags for DRP calculation...")
    df = df.sort_values(["StationName", "Date"]).copy()
    for station, grp in df.groupby("StationName", sort=False):
        idx = grp.index
        df.loc[idx, "PM25_lag1"] = grp["PM25"].shift(1)
        df.loc[idx, "PM10_lag1"] = grp["PM10"].shift(1)
    return df

def add_physics_plus(df):
    log.info("Adding Physics-Plus features (Stability, Exposure)...")
    df['PBLH_Wind_Index'] = df['BLH'] / (df['WindSpeed'] + 1.0)
    df['DewPoint_Depression'] = df['T2M_C'] - df['D2M_C']
    df['Coastal_Exposure'] = 1.0 / (df['Dist_Coast_km'] + 0.1)
    return df

def main():
    if not TRAINING_DATA.exists():
        log.error(f"Training data not found: {TRAINING_DATA}")
        return

    df = pd.read_parquet(TRAINING_DATA)
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract unique station info
    stations = df[['StationName', 'Latitude', 'Longitude', 'Dist_Coast_km']].drop_duplicates().reset_index(drop=True)
    
    # 0. Add lags and physics+
    df = add_local_lags(df)
    df = add_physics_plus(df)
    
    # 1. LUR Features
    stations = extract_lur_features(stations)
    
    # 2. DRP Features
    drp_df = compute_drp_features(df)
    
    # Merge back
    log.info("Merging new features into dataset...")
    df = df.merge(stations[['StationName', 'UrbanDensity_5km']], on='StationName', how='left')
    df = df.merge(drp_df, on=['StationName', 'Date'], how='left')
    
    log.info(f"Final shape: {df.shape}")
    log.info(f"Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE)
    log.info("Done!")

if __name__ == "__main__":
    main()
