"""
Abstract base class for all data downloaders.
Implements template method pattern for consistent download workflow.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from ..config.settings import config
from ..utils.logger import LoggerMixin


class BaseDownloader(ABC, LoggerMixin):
    """
    Abstract base class for satellite and meteorological data downloaders.
    
    Implements the Template Method pattern to enforce consistent workflow:
    1. Validate inputs
    2. Initialize client (GEE, CDS, etc.)
    3. Fetch data
    4. Validate output
    5. Save data
    
    Subclasses must implement abstract methods for specific data sources.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded data (defaults to config)
        """
        self._init_logger()
        self.output_dir = output_dir or config.paths.raw_data
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def download(
        self,
        product_name: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Path:
        """
        Template method for download workflow.
        
        Args:
            product_name: Name of the product to download
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional product-specific parameters
        
        Returns:
            Path to downloaded data
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If download fails
        """
        self.logger.info(f"Starting download: {product_name}")
        self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        try:
            # Step 1: Validate inputs
            self._validate_inputs(product_name, start_date, end_date)
            
            # Step 2: Initialize client
            if self.client is None:
                self.logger.info("Initializing API client...")
                self._initialize_client()
            
            # Step 3: Fetch data
            self.logger.info("Fetching data from remote source...")
            data = self._fetch_data(product_name, start_date, end_date, **kwargs)
            
            # Step 4: Validate output
            self.logger.info("Validating downloaded data...")
            self._validate_output(data, product_name)
            
            # Step 5: Save data
            self.logger.info("Saving data to disk...")
            output_path = self._save_data(data, product_name, start_date, end_date)
            
            self.logger.info(f"Download complete: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Download failed for {product_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to download {product_name}") from e
    
    def _validate_inputs(
        self,
        product_name: str,
        start_date: datetime,
        end_date: datetime
    ):
        """
        Validate download inputs.
        
        Args:
            product_name: Product name
            start_date: Start date
            end_date: End date
        
        Raises:
            ValueError: If inputs are invalid
        """
        if start_date >= end_date:
            raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")
        
        if not product_name:
            raise ValueError("product_name cannot be empty")
        
        self.logger.debug(f"Input validation passed for {product_name}")
    
    @abstractmethod
    def _initialize_client(self):
        """
        Initialize API client (GEE, CDS, etc.).
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _fetch_data(
        self,
        product_name: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Any:
        """
        Fetch data from remote source.
        Must be implemented by subclasses.
        
        Args:
            product_name: Product to download
            start_date: Start date
            end_date: End date
            **kwargs: Product-specific parameters
        
        Returns:
            Downloaded data (format depends on source)
        """
        pass
    
    @abstractmethod
    def _validate_output(self, data: Any, product_name: str):
        """
        Validate downloaded data.
        Must be implemented by subclasses.
        
        Args:
            data: Downloaded data
            product_name: Product name
        
        Raises:
            ValueError: If data is invalid
        """
        pass
    
    @abstractmethod
    def _save_data(
        self,
        data: Any,
        product_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """
        Save data to disk.
        Must be implemented by subclasses.
        
        Args:
            data: Data to save
            product_name: Product name
            start_date: Start date
            end_date: End date
        
        Returns:
            Path to saved data
        """
        pass
    
    def _get_output_path(
        self,
        product_name: str,
        start_date: datetime,
        end_date: datetime,
        extension: str = ""
    ) -> Path:
        """
        Generate standardized output path.
        
        Args:
            product_name: Product name
            start_date: Start date
            end_date: End date
            extension: File extension (e.g., '.nc', '.tif')
        
        Returns:
            Full path for output file
        """
        # Create subdirectory for product
        product_dir = self.output_dir / product_name
        product_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        filename = f"{product_name}_{start_str}_{end_str}{extension}"
        
        return product_dir / filename
    
    def cleanup(self):
        """Clean up resources (close connections, etc.)"""
        self.logger.info(f"Cleaning up {self.__class__.__name__}")
        self.client = None


if __name__ == "__main__":
    # This is an abstract class, so we can't instantiate it directly
    print("BaseDownloader is an abstract base class.")
    print("See GEEDownloader or CDSDownloader for concrete implementations.")
