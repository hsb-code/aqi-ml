"""
Data extraction module for satellite and weather data at ground station locations.
Handles GeoTIFF (NO2, MODIS AOD) and NetCDF (ERA5) extraction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import rasterio
import xarray as xr
from datetime import datetime


class DataExtractor:
    """Extract satellite and weather data at specific geographic points."""
    
    def __init__(self, logger=None):
        """
        Initialize data extractor.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def extract_from_geotiff(
        self,
        file_path: Path,
        lat: float,
        lon: float
    ) -> Tuple[float, bool]:
        """
        Extract pixel value at specific coordinates from GeoTIFF.
        
        Args:
            file_path: Path to GeoTIFF file
            lat: Latitude
            lon: Longitude
        
        Returns:
            Tuple of (value, success_flag)
        """
        try:
            with rasterio.open(file_path) as src:
                # Convert lat/lon to pixel coordinates
                row, col = src.index(lon, lat)
                
                # Read the value at that pixel
                value = src.read(1)[row, col]
                
                # Check for nodata
                if src.nodata is not None and value == src.nodata:
                    return (np.nan, False)
                
                # Check for invalid values (NaN, Inf)
                if not np.isfinite(value):
                    return (np.nan, False)
                
                return (float(value), True)
                
        except Exception as e:
            self.logger.warning(f"Failed to extract from {file_path.name}: {e}")
            return (np.nan, False)
    
    def extract_geotiff_series(
        self,
        file_dir: Path,
        stations_df: pd.DataFrame,
        product_name: str
    ) -> pd.DataFrame:
        """
        Extract values from multiple GeoTIFF files for all stations.
        
        Args:
            file_dir: Directory containing GeoTIFF files
            stations_df: DataFrame with columns [Station, Latitude, Longitude]
            product_name: Name of product (e.g., 'NO2', 'AOD')
        
        Returns:
            DataFrame with columns [Date, Station, {product_name}]
        """
        self.logger.info(f"Extracting {product_name} from {file_dir}")
        
        # Find all .tif files
        tif_files = sorted(file_dir.glob("*.tif"))
        self.logger.info(f"Found {len(tif_files)} files")
        
        results = []
        successful = 0
        failed = 0
        
        for i, tif_file in enumerate(tif_files):
            # Extract date from filename (assuming format: PRODUCT_YYYY-MM-DD.tif)
            try:
                date_str = tif_file.stem.split('_')[-1]
                date = pd.to_datetime(date_str)
            except:
                self.logger.warning(f"Could not parse date from filename: {tif_file.name}")
                continue
            
            # Extract for each station
            for _, station in stations_df.iterrows():
                value, success = self.extract_from_geotiff(
                    tif_file,
                    station['Latitude'],
                    station['Longitude']
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                results.append({
                    'Date': date,
                    'Station': station['Station'],
                    product_name: value
                })
            
            # Progress logging
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(tif_files)} files")
        
        self.logger.info(f"Extraction complete: {successful} successful, {failed} failed")
        
        return pd.DataFrame(results)
    
    def extract_from_netcdf(
        self,
        netcdf_path: Path,
        stations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract ERA5 variables at station locations from NetCDF.
        
        Args:
            netcdf_path: Path to merged ERA5 NetCDF file
            stations_df: DataFrame with columns [Station, Latitude, Longitude]
        
        Returns:
            DataFrame with columns [Date, Station, T2m, D2m, U10, V10, SP, BLH]
        """
        self.logger.info(f"Extracting ERA5 from {netcdf_path}")
        
        # Open NetCDF
        ds = xr.open_dataset(netcdf_path)
        
        # Determine time coordinate name
        time_coord = 'time' if 'time' in ds.coords else 'valid_time'
        
        # Map variable names (CDS long names vs NetCDF short names)
        var_mapping = {
            't2m': ['t2m', '2m_temperature'],
            'd2m': ['d2m', '2m_dewpoint_temperature'],
            'u10': ['u10', '10m_u_component_of_wind'],
            'v10': ['v10', '10m_v_component_of_wind'],
            'sp': ['sp', 'surface_pressure'],
            'blh': ['blh', 'boundary_layer_height']
        }
        
        # Find actual variable names in dataset
        actual_vars = {}
        for short_name, possible_names in var_mapping.items():
            for name in possible_names:
                if name in ds.data_vars:
                    actual_vars[short_name] = name
                    break
            if short_name not in actual_vars:
                self.logger.warning(f"Variable {short_name} not found in NetCDF")
        
        self.logger.info(f"Found variables: {actual_vars}")
        
        results = []
        
        # Extract for each station
        for _, station in stations_df.iterrows():
            station_name = station['Station']
            lat = station['Latitude']
            lon = station['Longitude']
            
            self.logger.debug(f"Extracting for {station_name} at ({lat}, {lon})")
            
            # Select nearest grid point
            station_data = ds.sel(
                latitude=lat,
                longitude=lon,
                method='nearest'
            )
            
            # Convert to DataFrame for easier aggregation
            temp_data = []
            for t in range(len(station_data[time_coord])):
                timestamp = pd.to_datetime(station_data[time_coord].values[t])
                date = timestamp.normalize()  # Get date only
                
                row = {
                    'Date': date,
                    'Station': station_name
                }
                
                # Extract each variable
                for short_name, nc_name in actual_vars.items():
                    try:
                        value = float(station_data[nc_name].values[t])
                        row[short_name.upper()] = value
                    except:
                        row[short_name.upper()] = np.nan
                
                temp_data.append(row)
            
            # Aggregate to daily means (handles multiple timestamps per day)
            temp_df = pd.DataFrame(temp_data)
            daily_means = temp_df.groupby(['Date', 'Station']).mean().reset_index()
            
            results.extend(daily_means.to_dict('records'))
        
        ds.close()
        
        self.logger.info(f"Extracted {len(results)} station-day records from ERA5 (aggregated to daily)")
        
        return pd.DataFrame(results)
    
    def load_ground_stations(
        self,
        csv_path: Path,
        start_date: str = '2022-01-01',
        end_date: str = '2024-12-31'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and process ground station data.
        
        Args:
            csv_path: Path to ground station CSV
            start_date: Start date for filtering
            end_date: End date for filtering
        
        Returns:
            Tuple of (station_metadata_df, daily_measurements_df)
        """
        self.logger.info(f"Loading ground station data from {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        self.logger.info(f"Loaded {len(df)} hourly records")
        self.logger.info(f"Columns: {list(df.columns)}")
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['datetime'])
        
        # Rename columns for clarity
        df = df.rename(columns={
            'StationName': 'Station',
            'PM2P5': 'PM2.5',
            'x': 'Longitude',
            'y': 'Latitude'
        })
        
        self.logger.info(f"Unique stations: {df['Station'].nunique()}")
        self.logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Filter date range
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        self.logger.info(f"After date filtering: {len(df)} records")
        
        # Quality control: Remove invalid PM values
        initial_count = len(df)
        df = df[
            (df['PM2.5'] >= 0) & (df['PM2.5'] < 500) &
            (df['PM10'] >= 0) & (df['PM10'] < 1000)
        ]
        removed = initial_count - len(df)
        self.logger.info(f"Removed {removed} records with invalid PM values")
        
        # Extract station metadata (unique stations with coordinates)
        stations = df[['Station', 'Latitude', 'Longitude']].drop_duplicates()
        stations = stations.sort_values('Station').reset_index(drop=True)
        self.logger.info(f"Station metadata extracted: {len(stations)} stations")
        
        # Aggregate hourly to daily (mean of all hourly measurements per day)
        daily = df.groupby([df['Date'].dt.date, 'Station']).agg({
            'PM2.5': 'mean',
            'PM10': 'mean',
            'Latitude': 'first',
            'Longitude': 'first'
        }).reset_index()
        
        # Convert Date back to datetime
        daily['Date'] = pd.to_datetime(daily['Date'])
        
        self.logger.info(f"Aggregated to {len(daily)} daily measurements")
        self.logger.info(f"PM2.5 range: {daily['PM2.5'].min():.1f} - {daily['PM2.5'].max():.1f} µg/m³")
        self.logger.info(f"PM10 range: {daily['PM10'].min():.1f} - {daily['PM10'].max():.1f} µg/m³")
        
        return stations, daily


if __name__ == "__main__":
    # Test extraction
    import sys
    logging.basicConfig(level=logging.INFO)
    
    extractor = DataExtractor()
    
    # Test with sample data
    print("Data extractor module loaded successfully")
