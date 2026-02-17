"""
Central configuration module for AQI ML system.
Contains settings for data downloads, processing, and model training.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Paths:
    """File paths configuration"""
    # Root directory
    project_root: Path = Path("H:/AQI")
    
    # Data directories
    data_root: Path = field(default_factory=lambda: Path("H:/AQI/data"))
    raw_data: Path = field(default_factory=lambda: Path("H:/AQI/data/raw"))
    processed_data: Path = field(default_factory=lambda: Path("H:/AQI/data/processed"))
    
    # Ancillary data
    ancillary: Path = field(default_factory=lambda: Path("H:/AQI/00_Ancillary_Data"))
    ground_stations_csv: Path = field(
        default_factory=lambda: Path("H:/AQI/00_Ancillary_Data/EAD_Hourly_2022-2024_AQ_Points_AQI.csv")
    )
    abu_dhabi_shapefile: Path = field(
        default_factory=lambda: Path("H:/AQI/00_Ancillary_Data/AbuDhabi_Regions.shp")
    )
    
    # Model directories
    models: Path = field(default_factory=lambda: Path("H:/AQI/models"))
    checkpoints: Path = field(default_factory=lambda: Path("H:/AQI/models/checkpoints"))
    
    # Logs
    logs: Path = field(default_factory=lambda: Path("H:/AQI/logs"))
    
    def ensure_directories(self):
        """Create all directories if they don't exist"""
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, Path) and 'csv' not in str(attr) and 'shp' not in str(attr):
                    attr.mkdir(parents=True, exist_ok=True)


@dataclass
class DownloadConfig:
    """Configuration for satellite data downloads"""
    # Training period (3 years as per requirements)
    start_date: datetime = datetime(2022, 1, 1)
    end_date: datetime = datetime(2024, 12, 31)
    
    # Prediction period (December 2025)
    prediction_start: datetime = datetime(2025, 12, 1)
    prediction_end: datetime = datetime(2025, 12, 31)
    
    # Region of Interest
    region_name: str = "Abu Dhabi"
    
    # Abu Dhabi bounding box [lon_min, lat_min, lon_max, lat_max]
    # Approximate bounds for Abu Dhabi Emirate
    bbox: Tuple[float, float, float, float] = (51.5, 22.5, 56.0, 25.5)
    
    # Google Earth Engine Products
    s5p_products: Dict[str, str] = field(default_factory=lambda: {
        'NO2': 'COPERNICUS/S5P/OFFL/L3_NO2',
        'SO2': 'COPERNICUS/S5P/OFFL/L3_SO2',
        'CO': 'COPERNICUS/S5P/OFFL/L3_CO',
        'O3': 'COPERNICUS/S5P/OFFL/L3_O3'
    })
    
    # MODIS AOD product
    modis_product: str = 'MODIS/061/MCD19A2_GRANULES'
    modis_variable: str = 'Optical_Depth_055'
    
    # Sentinel-5P variables
    s5p_variables: Dict[str, str] = field(default_factory=lambda: {
        'NO2': 'NO2_column_number_density',
        'SO2': 'SO2_column_number_density',
        'CO': 'CO_column_number_density',
        'O3': 'O3_column_number_density'
    })
    
    # ERA5 variables for meteorology
    era5_variables: List[str] = field(default_factory=lambda: [
        '2m_temperature',
        '2m_dewpoint_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'surface_pressure',
        'boundary_layer_height'
    ])
    
    # Download settings
    chunk_size_months: int = 1  # Download monthly chunks (yearly chunks exceed CDS cost limits)
    max_retries: int = 3
    retry_delay_seconds: int = 60


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    # Target CRS (Web Mercator - same as existing pipeline)
    target_crs: str = "EPSG:3857"
    
    # Target spatial resolution (meters)
    target_resolution: int = 1000  # 1 km
    
    # Temporal aggregation
    temporal_resolution: str = "daily"  # 'hourly' or 'daily'
    
    # Quality control thresholds
    pm25_max: float = 500.0  # µg/m³
    pm10_max: float = 1000.0  # µg/m³
    aod_max: float = 5.0  # dimensionless
    
    # Missing data handling
    max_missing_fraction: float = 0.4  # Allow up to 40% missing data
    interpolation_method: str = "linear"
    
    # Zarr compression
    zarr_compressor: str = "zlib"
    zarr_compression_level: int = 5


@dataclass
class ModelConfig:
    """Configuration for ML model training"""
    # Model type
    model_type: str = "RandomForest"
    
    # Random Forest hyperparameters
    n_estimators: int = 200
    max_depth: int = 20
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    
    # Training settings
    test_size: float = 0.15
    val_size: float = 0.15
    train_size: float = 0.70
    
    # Feature engineering
    temporal_features: List[str] = field(default_factory=lambda: [
        'hour', 'day_of_week', 'day_of_year', 'month', 'season', 'is_weekend'
    ])
    
    # Target variables
    target_variables: List[str] = field(default_factory=lambda: ['PM2.5', 'PM10'])
    
    # Performance thresholds
    min_r2_pm25: float = 0.75
    min_r2_pm10: float = 0.73
    max_rmse_pm25: float = 15.0  # µg/m³
    max_rmse_pm10: float = 25.0  # µg/m³


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Log file settings
    log_to_file: bool = True
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_file_backup_count: int = 5


class Config:
    """Main configuration class that combines all configs"""
    
    def __init__(self):
        self.paths = Paths()
        self.download = DownloadConfig()
        self.processing = ProcessingConfig()
        self.model = ModelConfig()
        self.logging = LoggingConfig()
        
        # Ensure all directories exist
        self.paths.ensure_directories()
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  Training: {self.download.start_date.date()} to {self.download.end_date.date()}\n"
            f"  Region: {self.download.region_name}\n"
            f"  Target CRS: {self.processing.target_crs}\n"
            f"  Model: {self.model.model_type}\n"
            f")"
        )


# Global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    print(config)
    print(f"\nData paths:")
    print(f"  Raw data: {config.paths.raw_data}")
    print(f"  Processed: {config.paths.processed_data}")
    print(f"  Models: {config.paths.models}")
    print(f"\nTraining period: {config.download.start_date.date()} to {config.download.end_date.date()}")
    print(f"Prediction period: {config.download.prediction_start.date()} to {config.download.prediction_end.date()}")
