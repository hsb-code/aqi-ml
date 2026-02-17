"""
Copernicus CDS downloader for ERA5 meteorological data.
Handles multi-year downloads with chunking and retry logic.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Tuple
import cdsapi
import xarray as xr

from .base_downloader import BaseDownloader
from ..config.settings import config


class CDSDownloader(BaseDownloader):
    """
    Download ERA5 reanalysis data from Copernicus Climate Data Store.
    
    Features:
    - Automatic CDS API authentication
    - Multi-year chunking (avoid timeouts)
    - NetCDF file merging
    - Retry mechanism for queue delays
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize CDS downloader.
        
        Args:
            output_dir: Directory to save downloads
        """
        super().__init__(output_dir)
        self.bbox = config.download.bbox
        self.variables = config.download.era5_variables
        self.chunk_months = config.download.chunk_size_months
        self.max_retries = config.download.max_retries
    
    def _initialize_client(self):
        """Initialize CDS API client"""
        try:
            self.client = cdsapi.Client()
            self.logger.info("CDS API client initialized successfully")
            self.logger.info("Using credentials from ~/.cdsapirc")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize CDS client: {e}\n"
                "Please ensure ~/.cdsapirc file exists with your UID and API key.\n"
                "See: https://cds.climate.copernicus.eu/api-how-to"
            ) from e
    
    def _create_date_chunks(
        self,
        start_date: datetime,
        end_date: datetime,
        chunk_months: int = 12
    ) -> List[Tuple[datetime, datetime]]:
        """
        Split date range into chunks to avoid CDS timeouts.
        
        Args:
            start_date: Start date
            end_date: End date
            chunk_months: Chunk size in months
        
        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        chunks = []
        current = start_date
        
        while current < end_date:
            # Calculate chunk end (either chunk_months ahead or end_date)
            if chunk_months == 12:
                # Yearly chunks
                chunk_end = datetime(current.year + 1, 1, 1) - timedelta(days=1)
            else:
                # Monthly chunks
                chunk_end = current + timedelta(days=30 * chunk_months)
            
            chunk_end = min(chunk_end, end_date)
            chunks.append((current, chunk_end))
            
            current = chunk_end + timedelta(days=1)
        
        self.logger.info(f"Split into {len(chunks)} chunks of ~{chunk_months} months each")
        return chunks
    
    def _fetch_data(
        self,
        product_name: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[Path]:
        """
        Fetch ERA5 data with chunking.
        
        Args:
            product_name: Product name (should be 'ERA5')
            start_date: Start date
            end_date: End date
        
        Returns:
            List of paths to downloaded NetCDF files
        """
        if product_name != 'ERA5':
            raise ValueError(f"CDSDownloader only supports ERA5, got: {product_name}")
        
        # Split into chunks
        chunks = self._create_date_chunks(start_date, end_date, self.chunk_months)
        
        # Download each chunk
        chunk_files = []
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            self.logger.info(f"Downloading chunk {i}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}")
            
            chunk_file = self._download_chunk(chunk_start, chunk_end, i)
            chunk_files.append(chunk_file)
        
        return chunk_files
    
    def _download_chunk(
        self,
        start_date: datetime,
        end_date: datetime,
        chunk_id: int
    ) -> Path:
        """
        Download a single date chunk from CDS.
        
        Args:
            start_date: Chunk start date
            end_date: Chunk end date
            chunk_id: Chunk number for naming
        
        Returns:
            Path to downloaded NetCDF file
        """
        # Prepare output path
        output_file = self.output_dir / "ERA5" / f"ERA5_chunk{chunk_id:02d}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists
        if output_file.exists():
            self.logger.info(f"Chunk file already exists: {output_file.name}")
            return output_file
        
        # Prepare CDS request
        lon_min, lat_min, lon_max, lat_max = self.bbox
        
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': self.variables,
            'year': [str(y) for y in range(start_date.year, end_date.year + 1)],
            'month': [f'{m:02d}' for m in range(1, 13)],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': ['12:00'],  # Single daily snapshot at 12:00 UTC (~4pm Abu Dhabi time)
            'area': [lat_max, lon_min, lat_min, lon_max],  # North, West, South, East
        }
        
        # Submit download
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Submitting CDS request (attempt {attempt + 1}/{self.max_retries})...")
                
                self.client.retrieve(
                    'reanalysis-era5-single-levels',
                    request,
                    str(output_file)
                )
                
                self.logger.info(f"Downloaded chunk to: {output_file}")
                return output_file
                
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = config.download.retry_delay_seconds * (2 ** attempt)
                    self.logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to download chunk after {self.max_retries} attempts") from e
    
    def _merge_netcdf_files(self, file_list: List[Path]) -> Path:
        """
        Merge multiple NetCDF files into one.
        
        Args:
            file_list: List of NetCDF file paths
        
        Returns:
            Path to merged file
        """
        if len(file_list) == 1:
            self.logger.info("Only one chunk, no merging needed")
            return file_list[0]
        
        self.logger.info(f"Merging {len(file_list)} NetCDF files...")
        
        # Load all datasets
        datasets = []
        for f in file_list:
            try:
                ds = xr.open_dataset(f)
                datasets.append(ds)
            except Exception as e:
                self.logger.error(f"Failed to open {f}: {e}")
        
        if not datasets:
            raise RuntimeError("No datasets were successfully opened for merging")

        # Concatenate along time dimension (find which name is used)
        time_coord = 'time'
        if 'time' not in datasets[0].coords and 'valid_time' in datasets[0].coords:
            time_coord = 'valid_time'
            self.logger.info("Using 'valid_time' as time coordinate for merging")
        
        # Concatenate
        merged = xr.concat(datasets, dim=time_coord)
        
        # Sort by time
        merged = merged.sortby(time_coord)
        
        # Get start/end dates for naming
        start_t = datasets[0].coords[time_coord].values[0]
        end_t = datasets[-1].coords[time_coord].values[-1]
        
        # Convert to daily string for filename
        start_str = str(start_t)[:10].replace('-', '')
        end_str = str(end_t)[:10].replace('-', '')
        
        output_file = self.output_dir / "ERA5" / f"ERA5_merged_{start_str}_{end_str}.nc"
        
        # Save merged file
        self.logger.info(f"Saving merged file: {output_file}")
        merged.to_netcdf(output_file)
        
        # Close datasets
        for ds in datasets:
            ds.close()
        
        self.logger.info(f"Merge complete: {output_file}")
        return output_file
    
    def _validate_output(self, data: List[Path], product_name: str):
        """
        Validate downloaded NetCDF files.
        
        Args:
            data: List of NetCDF file paths
            product_name: Product name
        
        Raises:
            ValueError: If data is invalid
        """
        if not data or len(data) == 0:
            raise ValueError(f"No files downloaded for {product_name}")
        
        # Typical mapping from CDS long names to NetCDF short names
        var_mapping = {
            '2m_temperature': 't2m',
            '2m_dewpoint_temperature': 'd2m',
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10',
            'surface_pressure': 'sp',
            'boundary_layer_height': 'blh'
        }
        
        # Check each file exists and can be opened
        for file_path in data:
            if not file_path.exists():
                raise ValueError(f"Downloaded file does not exist: {file_path}")
            
            try:
                ds = xr.open_dataset(file_path)
                
                # Check coordinates (expect time or valid_time)
                if 'time' not in ds.coords and 'valid_time' not in ds.coords:
                    self.logger.warning(f"No time coordinate found in {file_path.name}. Coordinates: {list(ds.coords)}")

                # Check variables
                available_vars = set(ds.data_vars)
                missing_vars = []
                for var in self.variables:
                    short_name = var_mapping.get(var, var)
                    if var not in available_vars and short_name not in available_vars:
                        missing_vars.append(var)
                
                if missing_vars:
                    self.logger.warning(f"Missing variables in {file_path.name}: {missing_vars}")
                    self.logger.info(f"Available variables: {list(available_vars)}")
                
                ds.close()
                
            except Exception as e:
                raise ValueError(f"Cannot open NetCDF file {file_path}: {e}") from e
        
        self.logger.info("Output validation passed")
    
    def _save_data(
        self,
        data: List[Path],
        product_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """
        Save ERA5 data (merge if multiple chunks).
        
        Args:
            data: List of NetCDF file paths
            product_name: Product name
            start_date: Start date
            end_date: End date
        
        Returns:
            Path to final merged file
        """
        if len(data) > 1:
            return self._merge_netcdf_files(data)
        else:
            return data[0]


if __name__ == "__main__":
    # Test CDS downloader
    downloader = CDSDownloader()
    
    # Test with small date range
    test_start = datetime(2024, 1, 1)
    test_end = datetime(2024, 1, 31)
    
    try:
        output = downloader.download('ERA5', test_start, test_end)
        print(f"Download test completed: {output}")
    except Exception as e:
        print(f"Test failed: {e}")
