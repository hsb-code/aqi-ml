"""
Google Earth Engine downloader for Sentinel-5P and MODIS data.
Handles authentication, data filtering, and robust download with retries.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import ee

from .base_downloader import BaseDownloader
from ..config.settings import config


class GEEDownloader(BaseDownloader):
    """
    Download satellite data from Google Earth Engine.
    
    Supports:
    - Sentinel-5P TROPOMI (NO2, SO2, CO, O3)
    - MODIS AOD (MCD19A2)
    
    Features:
    - Automatic authentication
    - Quality filtering
    - Retry mechanism with exponential backoff
    - Multi-year handling
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize GEE downloader.
        
        Args:
            output_dir: Directory to save downloads
        """
        super().__init__(output_dir)
        self.bbox = config.download.bbox
        self.max_retries = config.download.max_retries
        self.retry_delay = config.download.retry_delay_seconds
    
    def _initialize_client(self):
        """Initialize and authenticate with Google Earth Engine"""
        try:
            # Initialize with your GEE project
            ee.Initialize(project='aqi-ml-project')
            self.logger.info("Google Earth Engine initialized successfully with project: aqi-ml-project")
            self.client = ee  # Store reference
            
        except Exception as e:
            self.logger.error(f"GEE initialization failed: {e}")
            self.logger.info("Attempting authentication...")
            
            try:
                ee.Authenticate()
                ee.Initialize(project='aqi-ml-project')
                self.logger.info("GEE authenticated and initialized")
                self.client = ee
            except Exception as auth_error:
                raise RuntimeError(
                    f"Failed to initialize GEE: {auth_error}. "
                    "Run 'earthengine authenticate' from command line."
                ) from auth_error
    
    def _get_region_geometry(self):
        """Get Abu Dhabi region as ee.Geometry"""
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    
    def _fetch_data(
        self,
        product_name: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> ee.ImageCollection:
        """
        Fetch data from GEE.
        
        Args:
            product_name: Product name (e.g., 'NO2', 'SO2', 'MODIS_AOD')
            start_date: Start date
            end_date: End date
        
        Returns:
            ImageCollection with filtered data
        """
        # Determine GEE product ID
        if product_name in config.download.s5p_products:
            gee_product = config.download.s5p_products[product_name]
            variable = config.download.s5p_variables[product_name]
            is_modis = False
        elif product_name == 'MODIS_AOD':
            gee_product = config.download.modis_product
            variable = config.download.modis_variable
            is_modis = True
        else:
            raise ValueError(f"Unknown product: {product_name}")
        
        self.logger.info(f"Fetching {product_name} from {gee_product}")
        
        # Create image collection
        collection = ee.ImageCollection(gee_product)
        
        # Filter by date and region
        filtered = collection.filterDate(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        ).filterBounds(self._get_region_geometry())
        
        # Apply quality filters
        filtered = self._apply_quality_filters(filtered, product_name, is_modis)
        
        # Get count
        try:
            count = filtered.size().getInfo()
            self.logger.info(f"Found {count} images for {product_name}")
            
            if count == 0:
                self.logger.warning(f"No data found for {product_name} in date range")
        except Exception as e:
            self.logger.warning(f"Could not get image count: {e}")
        
        return filtered
    
    def _apply_quality_filters(
        self,
        collection: ee.ImageCollection,
        product_name: str,
        is_modis: bool
    ) -> ee.ImageCollection:
        """
        Apply quality filtering to image collection.
        
        Args:
            collection: Input ImageCollection
            product_name: Product name
            is_modis: Whether this is MODIS data
        
        Returns:
            Filtered ImageCollection
        """
        if is_modis:
            # MODIS AOD quality filtering
            # For now, return collection as-is
            # Can add QA filtering later if needed
            return collection
        
        else:
            # Sentinel-5P quality filtering
            # S5P products don't have qa_value band consistently
            # Use cloud fraction filter instead
            def filter_s5p(image):
                # Filter out images with >70% cloud cover
                cloud_fraction = image.select('cloud_fraction')
                return image.updateMask(cloud_fraction.lt(0.7))
            
            return collection.map(filter_s5p)
    
    def _validate_output(self, data: ee.ImageCollection, product_name: str):
        """
        Validate GEE ImageCollection.
        
        Args:
            data: ImageCollection to validate
            product_name: Product name
        
        Raises:
            ValueError: If data is invalid
        """
        if data is None:
            raise ValueError(f"No data retrieved for {product_name}")
        
        try:
            # Try to get collection size
            size = data.size().getInfo()
            if size == 0:
                self.logger.warning(f"Empty collection for {product_name}")
        except Exception as e:
            self.logger.warning(f"Could not validate collection size: {e}")
    
    def _save_data(
        self,
        data: ee.ImageCollection,
        product_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """
        Save GEE ImageCollection to GeoTIFF files (daily composites).
        
        Args:
            data: ImageCollection to save
            product_name: Product name
            start_date: Start date
            end_date: End date
        
        Returns:
            Path to output directory
        """
        import geemap
        
        # Create output directory
        output_dir = self.output_dir / product_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get variable name
        if product_name in config.download.s5p_variables:
            variable = config.download.s5p_variables[product_name]
        elif product_name == 'MODIS_AOD':
            variable = config.download.modis_variable
        else:
            raise ValueError(f"Unknown product: {product_name}")
        
        self.logger.info(f"Exporting {product_name} data to GeoTIFF files...")
        self.logger.info(f"Variable: {variable}")
        self.logger.info("Creating daily composite images (mean of all overpasses per day)")
        
        # Get region geometry
        region = self._get_region_geometry()
        
        # Calculate total days
        total_days = (end_date - start_date).days + 1
        self.logger.info(f"Processing {total_days} days from {start_date.date()} to {end_date.date()}")
        
        exported_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Process day by day
        current_date = start_date
        
        while current_date <= end_date:
            try:
                date_str = current_date.strftime('%Y-%m-%d')
                output_file = output_dir / f"{product_name}_{date_str}.tif"
                
                # Skip if already exists
                if output_file.exists():
                    self.logger.debug(f"Skipping {date_str} (already exists)")
                    skipped_count += 1
                    current_date += timedelta(days=1)
                    continue
                
                # Filter collection for this day
                next_date = current_date + timedelta(days=1)
                daily = data.filterDate(
                    current_date.strftime("%Y-%m-%d"),
                    next_date.strftime("%Y-%m-%d")
                )
                
                # Check if we have data for this day
                count = daily.size().getInfo()
                
                if count == 0:
                    self.logger.debug(f"No data for {date_str}, skipping")
                    failed_count += 1
                    current_date += timedelta(days=1)
                    continue
                
                # Create daily composite (mean of all images)
                daily_composite = daily.select(variable).mean()
                
                # Export using geemap
                geemap.ee_export_image(
                    daily_composite,
                    filename=str(output_file),
                    scale=1000,  # 1km resolution
                    region=region,
                    file_per_band=False
                )
                
                exported_count += 1
                
                # Log progress every 30 days
                if exported_count % 30 == 0:
                    self.logger.info(f"Progress: {exported_count}/{total_days} days exported ({failed_count} no data, {skipped_count} skipped)")
                
            except Exception as e:
                self.logger.warning(f"Failed to export {current_date.date()}: {e}")
                failed_count += 1
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"Export complete!")
        self.logger.info(f"  Exported: {exported_count} days")
        self.logger.info(f"  No data: {failed_count} days")  
        self.logger.info(f"  Skipped (already existed): {skipped_count} days")
        
        # Save summary info
        info_file = output_dir / f"{product_name}_summary.txt"
        with open(info_file, 'w') as f:
            f.write(f"Product: {product_name}\n")
            f.write(f"Variable: {variable}\n")
            f.write(f"Date range: {start_date.date()} to {end_date.date()}\n")
            f.write(f"Bounding box: {self.bbox}\n")
            f.write(f"Total days in range: {total_days}\n")
            f.write(f"Successfully exported: {exported_count}\n")
            f.write(f"Days with no data: {failed_count}\n")
            f.write(f"Already existed (skipped): {skipped_count}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"\nNote: Each file is a daily composite (mean of all satellite overpasses)\n")
        
        return output_dir
    
    def download_daily_images(
        self,
        product_name: str,
        start_date: datetime,
        end_date: datetime,
        scale: int = 1000
    ) -> Path:
        """
        Download daily images for a product.
        
        This method downloads individual daily images rather than a collection.
        Useful for processing day-by-day.
        
        Args:
            product_name: Product to download
            start_date: Start date
            end_date: End date
            scale: Spatial resolution in meters
        
        Returns:
            Path to output directory
        """
        self.logger.info(f"Downloading daily images for {product_name}")
        
        # Initialize if needed
        if self.client is None:
            self._initialize_client()
        
        # Get collection
        collection = self._fetch_data(product_name, start_date, end_date)
        
        # Output directory
        output_dir = self.output_dir / product_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate through dates
        current_date = start_date
        downloaded_count = 0
        
        while current_date <= end_date:
            next_date = current_date + timedelta(days=1)
            
            # Filter for single day
            daily = collection.filterDate(
                current_date.strftime("%Y-%m-%d"),
                next_date.strftime("%Y-%m-%d")
            )
            
            try:
                size = daily.size().getInfo()
                if size > 0:
                    self.logger.debug(f"Processing {current_date.date()}: {size} images")
                    downloaded_count += 1
            except Exception as e:
                self.logger.warning(f"Error processing {current_date.date()}: {e}")
            
            current_date = next_date
        
        self.logger.info(f"Processed {downloaded_count} days with data")
        
        return output_dir


if __name__ == "__main__":
    # Test GEE downloader
    downloader = GEEDownloader()
    
    # Test with small date range
    test_start = datetime(2024, 12, 1)
    test_end = datetime(2024, 12, 7)
    
    try:
        output = downloader.download('NO2', test_start, test_end)
        print(f"Download test completed: {output}")
    except Exception as e:
        print(f"Test failed: {e}")
