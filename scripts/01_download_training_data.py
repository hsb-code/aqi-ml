"""
Main download script for AQI training data (2022-2024).
Downloads satellite data (S5P, MODIS) and meteorological data (ERA5).
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import config
from src.data_acquisition.gee_downloader import GEEDownloader
from src.data_acquisition.cds_downloader import CDSDownloader
from src.utils.logger import setup_logger


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Download AQI ML training data (satellite + meteorology)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default='2022-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default='2024-12-31',
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--products',
        nargs='+',
        choices=['NO2', 'SO2', 'CO', 'O3', 'MODIS_AOD', 'ERA5', 'all'],
        default=['all'],
        help='Products to download'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (defaults to config setting)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip products that already have data'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def check_existing_data(product: str, output_dir: Path) -> bool:
    """
    Check if data already exists for a product.
    
    Args:
        product: Product name
        output_dir: Output directory
    
    Returns:
        True if data exists, False otherwise
    """
    product_dir = output_dir / product
    if product_dir.exists() and any(product_dir.iterdir()):
        return True
    return False


def download_s5p_products(
    products: list,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    skip_existing: bool,
    logger
):
    """
    Download Sentinel-5P products.
    
    Args:
        products: List of products to download
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        skip_existing: Whether to skip existing data
        logger: Logger instance
    """
    s5p_products = ['NO2', 'SO2', 'CO', 'O3']
    to_download = [p for p in products if p in s5p_products or 'all' in products]
    
    if not to_download and 'all' not in products:
        logger.info("No S5P products selected")
        return
    
    if 'all' in products:
        to_download = s5p_products
    
    logger.info(f"Downloading S5P products: {to_download}")
    
    # Initialize GEE downloader
    gee = GEEDownloader(output_dir)
    
    for product in to_download:
        if skip_existing and check_existing_data(product, output_dir):
            logger.info(f"Skipping {product} (data already exists)")
            continue
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Downloading {product}")
            logger.info(f"{'='*60}")
            
            output_path = gee.download(product, start_date, end_date)
            
            logger.info(f"✓ {product} download complete: {output_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {product}: {e}", exc_info=True)
            logger.warning(f"Continuing with next product...")


def download_modis(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    skip_existing: bool,
    logger
):
    """
    Download MODIS AOD data.
    
    Args:
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        skip_existing: Whether to skip existing data
        logger: Logger instance
    """
    product = 'MODIS_AOD'
    
    if skip_existing and check_existing_data(product, output_dir):
        logger.info(f"Skipping {product} (data already exists)")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading {product}")
    logger.info(f"{'='*60}")
    
    try:
        gee = GEEDownloader(output_dir)
        output_path = gee.download(product, start_date, end_date)
        
        logger.info(f"✓ {product} download complete: {output_path}")
        
    except Exception as e:
        logger.error(f"✗ Failed to download {product}: {e}", exc_info=True)


def download_era5(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    skip_existing: bool,
    logger
):
    """
    Download ERA5 meteorology data.
    
    Args:
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        skip_existing: Whether to skip existing data
        logger: Logger instance
    """
    product = 'ERA5'
    
    if skip_existing and check_existing_data(product, output_dir):
        logger.info(f"Skipping {product} (data already exists)")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading {product}")
    logger.info(f"{'='*60}")
    
    try:
        cds = CDSDownloader(output_dir)
        output_path = cds.download(product, start_date, end_date)
        
        logger.info(f"✓ {product} download complete: {output_path}")
        
    except Exception as e:
        logger.error(f"✗ Failed to download {product}: {e}", exc_info=True)


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logger('data_download', level=args.log_level)
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else config.paths.raw_data
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info("="*60)
    logger.info("AQI ML Training Data Download")
    logger.info("="*60)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Products: {args.products}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("="*60)
    
    # Download products
    products = args.products
    
    # S5P products
    if any(p in products for p in ['NO2', 'SO2', 'CO', 'O3', 'all']):
        download_s5p_products(
            products, start_date, end_date, output_dir, args.skip_existing, logger
        )
    
    # MODIS AOD
    if 'MODIS_AOD' in products or 'all' in products:
        download_modis(start_date, end_date, output_dir, args.skip_existing, logger)
    
    # ERA5
    if 'ERA5' in products or 'all' in products:
        download_era5(start_date, end_date, output_dir, args.skip_existing, logger)
    
    logger.info("\n" + "="*60)
    logger.info("Download script completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
