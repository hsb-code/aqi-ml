"""
Main preprocessing script for AQI ML training data.

This script:
1. Loads ground station data (20 stations, 2022-2024)
2. Extracts NO2 values at station locations
3. Extracts MODIS AOD values at station locations  
4. Extracts ERA5 weather variables at station locations
5. Merges all features with ground truth PM data
6. Performs feature engineering
7. Quality control and cleaning
8. Saves final training dataset

Usage:
    python scripts/02_preprocess_data.py
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Import custom modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.extractor import DataExtractor
from src.preprocessing.feature_engineering import create_all_features
from src.config.settings import config
from src.utils.logger import setup_logger


def main():
    """Main preprocessing workflow."""
    
    # Setup logging
    logger = setup_logger('preprocessing', log_file='preprocessing.log')
    logger.info("="*60)
    logger.info("AQI ML Preprocessing Pipeline")
    logger.info("="*60)
    
    # Initialize paths
    raw_data_dir = Path(config.paths.raw_data)
    processed_dir = Path(config.paths.processed_data)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    ground_csv = config.paths.ground_stations_csv
    
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Output directory: {processed_dir}")
    
    # Initialize extractor
    extractor = DataExtractor(logger)
    
    # ========================================
    # Step 1: Load Ground Station Data
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading ground station data")
    logger.info("="*60)
    
    stations, ground_data = extractor.load_ground_stations(
        csv_path=ground_csv,
        start_date=str(config.download.start_date.date()),
        end_date=str(config.download.end_date.date())
    )
    
    logger.info(f"\nStations ({len(stations)}):")
    for _, station in stations.iterrows():
        logger.info(f"  - {station['Station']}: ({station['Latitude']:.4f}, {station['Longitude']:.4f})")
    
    logger.info(f"\nGround measurements: {len(ground_data)} station-days")
    logger.info(f"Date range: {ground_data['Date'].min()} to {ground_data['Date'].max()}")
    
    # Save intermediate
    ground_data.to_csv(processed_dir / 'ground_data_daily.csv', index=False)
    logger.info(f"Saved to: {processed_dir / 'ground_data_daily.csv'}")
    
    # ========================================
    # Step 2: Extract NO2 at Stations
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Extracting NO2 at station locations")
    logger.info("="*60)
    
    no2_dir = raw_data_dir / "NO2"
    
    if no2_dir.exists():
        no2_data = extractor.extract_geotiff_series(
            file_dir=no2_dir,
            stations_df=stations,
            product_name='NO2'
        )
        
        # Save intermediate
        no2_data.to_csv(processed_dir / 'no2_at_stations.csv', index=False)
        logger.info(f"Extracted {len(no2_data)} NO2 measurements")
        logger.info(f"Saved to: {processed_dir / 'no2_at_stations.csv'}")
    else:
        logger.error(f"NO2 directory not found: {no2_dir}")
        no2_data = pd.DataFrame()
    
    # ========================================
    # Step 3: Extract MODIS AOD at Stations
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Extracting MODIS AOD at station locations")
    logger.info("="*60)
    
    aod_dir = raw_data_dir / "MODIS_AOD"
    
    if aod_dir.exists():
        aod_data = extractor.extract_geotiff_series(
            file_dir=aod_dir,
            stations_df=stations,
            product_name='AOD'
        )
        
        # Save intermediate
        aod_data.to_csv(processed_dir / 'aod_at_stations.csv', index=False)
        logger.info(f"Extracted {len(aod_data)} AOD measurements")
        logger.info(f"Saved to: {processed_dir / 'aod_at_stations.csv'}")
    else:
        logger.error(f"MODIS_AOD directory not found: {aod_dir}")
        aod_data = pd.DataFrame()
    
    # ========================================
    # Step 4: Extract ERA5 at Stations
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Extracting ERA5 weather at station locations")
    logger.info("="*60)
    
    era5_file = raw_data_dir / "ERA5" / "ERA5_merged_20220101_20241231.nc"
    
    if era5_file.exists():
        era5_data = extractor.extract_from_netcdf(
            netcdf_path=era5_file,
            stations_df=stations
        )
        
        # Save intermediate
        era5_data.to_csv(processed_dir / 'era5_at_stations.csv', index=False)
        logger.info(f"Extracted {len(era5_data)} ERA5 measurements")
        logger.info(f"Saved to: {processed_dir / 'era5_at_stations.csv'}")
    else:
        logger.error(f"ERA5 file not found: {era5_file}")
        era5_data = pd.DataFrame()
    
    # ========================================
    # Step 5: Merge All Features
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Merging all features")
    logger.info("="*60)
    
    # Start with ground data
    merged = ground_data.copy()
    logger.info(f"Starting with ground data: {len(merged)} rows")
    
    # Merge NO2
    if not no2_data.empty:
        merged = merged.merge(
            no2_data[['Date', 'Station', 'NO2']],
            on=['Date', 'Station'],
            how='left'
        )
        logger.info(f"After NO2 merge: {len(merged)} rows, {merged['NO2'].notna().sum()} with NO2 data")
    
    # Merge AOD
    if not aod_data.empty:
        merged = merged.merge(
            aod_data[['Date', 'Station', 'AOD']],
            on=['Date', 'Station'],
            how='left'
        )
        logger.info(f"After AOD merge: {len(merged)} rows, {merged['AOD'].notna().sum()} with AOD data")
    
    # Merge ERA5
    if not era5_data.empty:
        merged = merged.merge(
            era5_data,
            on=['Date', 'Station'],
            how='left'
        )
        logger.info(f"After ERA5 merge: {len(merged)} rows")
    
    logger.info(f"\nMerged dataset shape: {merged.shape}")
    logger.info(f"Columns: {list(merged.columns)}")
    
    # ========================================
    # Step 6: Feature Engineering
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Feature engineering")
    logger.info("="*60)
    
    merged = create_all_features(merged)
    logger.info(f"After feature engineering: {merged.shape[1]} columns")
    
    # ========================================
    # Step 7: Quality Control
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 7: Quality control and cleaning")
    logger.info("="*60)
    
    initial_count = len(merged)
    
    # Remove rows where BOTH satellite features are missing
    merged = merged.dropna(subset=['NO2', 'AOD'], how='all')
    logger.info(f"Removed {initial_count - len(merged)} rows with no satellite data")
    
    # Remove rows with invalid PM values (should already be clean, but double-check)
    initial_count = len(merged)
    merged = merged[
        (merged['PM2.5'].notna()) &
        (merged['PM10'].notna()) &
        (merged['PM2.5'] >= 0) &
        (merged['PM10'] >= 0)
    ]
    logger.info(f"Removed {initial_count - len(merged)} rows with invalid PM values")
    
    # ========================================
    # Step 8: Save Final Dataset
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 8: Saving training dataset")
    logger.info("="*60)
    
    # Save as CSV
    csv_path = processed_dir / 'training_data_2022-2024.csv'
    merged.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    # Save as Parquet (better for ML)
    parquet_path = processed_dir / 'training_data_2022-2024.parquet'
    merged.to_parquet(parquet_path, index=False)
    logger.info(f"Saved Parquet: {parquet_path}")
    
    # ========================================
    # Step 9: Generate Summary Report
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    
    summary_path = processed_dir / 'training_data_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("AQI ML Training Dataset Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write(f"Total samples: {len(merged)}\n")
        f.write(f"Date range: {merged['Date'].min()} to {merged['Date'].max()}\n")
        f.write(f"Number of stations: {merged['Station'].nunique()}\n\n")
        
        f.write("Features:\n")
        feature_cols = [c for c in merged.columns if c not in ['Date', 'Station', 'PM2.5', 'PM10']]
        for col in feature_cols:
            missing_pct = 100 * merged[col].isna().sum() / len(merged)
            f.write(f"  - {col}: {missing_pct:.1f}% missing\n")
        
        f.write("\nTargets:\n")
        f.write(f"  - PM2.5: {merged['PM2.5'].min():.1f} - {merged['PM2.5'].max():.1f} µg/m³ (mean: {merged['PM2.5'].mean():.1f})\n")
        f.write(f"  - PM10: {merged['PM10'].min():.1f} - {merged['PM10'].max():.1f} µg/m³ (mean: {merged['PM10'].mean():.1f})\n")
        
        f.write(f"\nOutput files:\n")
        f.write(f"  - {csv_path}\n")
        f.write(f"  - {parquet_path}\n")
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"\n✓ Preprocessing complete!")
    logger.info(f"  Total training samples: {len(merged):,}")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Ready for ML training!")


if __name__ == "__main__":
    main()
