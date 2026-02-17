"""
Main ML training script for AQI prediction models.

This script:
1. Loads preprocessed training data
2. Splits into train/validation/test sets
3. Trains Random Forest models for PM2.5 and PM10
4. Evaluates model performance
5. Generates validation plots and feature importance charts
6. Saves trained models

Usage:
    python scripts/03_train_models.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.trainer import AQIModelTrainer
from src.config.settings import config
from src.utils.logger import setup_logger


def prepare_data(df: pd.DataFrame, target_col: str):
    """Prepare features and target for training.
    
    Args:
        df: Training dataframe
        target_col: Name of target column ('PM2.5' or 'PM10')
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Define feature columns (exclude non-predictive columns)
    exclude_cols = ['Date', 'Station', 'PM2.5', 'PM10']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y


def main():
    """Main training workflow."""
    
    # Setup logging
    logger = setup_logger('ml_training', log_file='ml_training.log')
    logger.info("=" * 70)
    logger.info("AQI ML MODEL TRAINING")
    logger.info("=" * 70)
    
    # Initialize paths
    processed_dir = Path(config.paths.processed_data)
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = models_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # ========================================
    # Step 1: Load Data
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Loading training data")
    logger.info("=" * 70)
    
    data_path = processed_dir / 'training_data_2022-2024.parquet'
    logger.info(f"Loading from: {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"✓ Loaded {len(df):,} samples")
    logger.info(f"✓ Features: {df.shape[1]} columns")
    
    # ========================================
    # Step 2: Handle Missing Values
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Handling missing values")
    logger.info("=" * 70)
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_features = missing[missing > 0]
    
    if len(missing_features) > 0:
        logger.info("Missing values detected:")
        for col, count in missing_features.items():
            pct = 100 * count / len(df)
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
        
        # Strategy: For satellite data (NO2, AOD), drop rows if both missing
        # For other features, fill with median
        logger.info("\nApplying missing value strategy...")
        
        # Drop rows with both NO2 and AOD missing (already done in preprocessing, but double-check)
        initial_len = len(df)
        df = df.dropna(subset=['NO2', 'AOD'], how='all')
        logger.info(f"  Dropped {initial_len - len(df)} rows with no satellite data")
        
        # Fill remaining missing values with median
        for col in df.columns:
            if df[col].isnull().any() and col not in ['Date', 'Station']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled {col} with median: {median_val:.2f}")
    else:
        logger.info("✓ No missing values detected")
    
    logger.info(f"\n✓ Clean dataset: {len(df):,} samples")
    
    # ========================================
    # Step 3: Train/Val/Test Split
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Splitting data")
    logger.info("=" * 70)
    
    # Split: 70% train, 15% val, 15% test
    # Use stratification by station to ensure geographic diversity
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42,
        stratify=df['Station']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['Station']
    )
    
    logger.info(f"Training set:   {len(train_df):,} samples ({100*len(train_df)/len(df):.1f}%)")
    logger.info(f"Validation set: {len(val_df):,} samples ({100*len(val_df)/len(df):.1f}%)")
    logger.info(f"Test set:       {len(test_df):,} samples ({100*len(test_df)/len(df):.1f}%)")
    
    # ========================================
    # Step 4: Train PM2.5 Model
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Training PM2.5 model")
    logger.info("=" * 70)
    
    X_train, y_train = prepare_data(train_df, 'PM2.5')
    X_val, y_val = prepare_data(val_df, 'PM2.5')
    X_test, y_test = prepare_data(test_df, 'PM2.5')
    
    logger.info(f"Feature columns ({len(X_train.columns)}):")
    for i, col in enumerate(X_train.columns, 1):
        logger.info(f"  {i:2d}. {col}")
    
    # Train model
    pm25_trainer = AQIModelTrainer(logger)
    pm25_model = pm25_trainer.train_random_forest(
        X_train, y_train,
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Evaluate
    logger.info("\nEvaluating PM2.5 model...")
    train_metrics = pm25_trainer.evaluate_model(X_train, y_train, "Training")
    val_metrics = pm25_trainer.evaluate_model(X_val, y_val, "Validation")
    test_metrics = pm25_trainer.evaluate_model(X_test, y_test, "Test")
    
    # Generate plots
    logger.info("\nGenerating PM2.5 plots...")
    pm25_trainer.plot_predictions(
        X_val, y_val,
        plots_dir / 'pm25_predictions_val.png',
        title='PM2.5 Predictions (Validation Set)',
        target_name='PM2.5'
    )
    
    pm25_trainer.plot_predictions(
        X_test, y_test,
        plots_dir / 'pm25_predictions_test.png',
        title='PM2.5 Predictions (Test Set)',
        target_name='PM2.5'
    )
    
    pm25_importance = pm25_trainer.plot_feature_importance(
        plots_dir / 'pm25_feature_importance.png',
        top_n=15,
        title='PM2.5 Model Feature Importance'
    )
    
    # Save model
    pm25_trainer.save_model(models_dir / 'pm25_rf_model.pkl')
    
    # ========================================
    # Step 5: Train PM10 Model
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Training PM10 model")
    logger.info("=" * 70)
    
    X_train, y_train = prepare_data(train_df, 'PM10')
    X_val, y_val = prepare_data(val_df, 'PM10')
    X_test, y_test = prepare_data(test_df, 'PM10')
    
    # Train model
    pm10_trainer = AQIModelTrainer(logger)
    pm10_model = pm10_trainer.train_random_forest(
        X_train, y_train,
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Evaluate
    logger.info("\nEvaluating PM10 model...")
    train_metrics_pm10 = pm10_trainer.evaluate_model(X_train, y_train, "Training")
    val_metrics_pm10 = pm10_trainer.evaluate_model(X_val, y_val, "Validation")
    test_metrics_pm10 = pm10_trainer.evaluate_model(X_test, y_test, "Test")
    
    # Generate plots
    logger.info("\nGenerating PM10 plots...")
    pm10_trainer.plot_predictions(
        X_val, y_val,
        plots_dir / 'pm10_predictions_val.png',
        title='PM10 Predictions (Validation Set)',
        target_name='PM10'
    )
    
    pm10_trainer.plot_predictions(
        X_test, y_test,
        plots_dir / 'pm10_predictions_test.png',
        title='PM10 Predictions (Test Set)',
        target_name='PM10'
    )
    
    pm10_importance = pm10_trainer.plot_feature_importance(
        plots_dir / 'pm10_feature_importance.png',
        top_n=15,
        title='PM10 Model Feature Importance'
    )
    
    # Save model
    pm10_trainer.save_model(models_dir / 'pm10_rf_model.pkl')
    
    # ========================================
    # Step 6: Save Test Set & Predictions
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Saving test set and predictions")
    logger.info("=" * 70)
    
    # Save test set with predictions
    test_results_df = test_df.copy()
    
    # Add PM2.5 predictions
    X_test_pm25, y_test_pm25 = prepare_data(test_df, 'PM2.5')
    test_results_df['PM2.5_Predicted'] = pm25_trainer.model.predict(X_test_pm25)
    
    # Add PM10 predictions
    X_test_pm10, y_test_pm10 = prepare_data(test_df, 'PM10')
    test_results_df['PM10_Predicted'] = pm10_trainer.model.predict(X_test_pm10)
    
    # Save test results
    test_results_path = models_dir / 'test_set_results.csv'
    test_results_df.to_csv(test_results_path, index=False)
    logger.info(f"✓ Saved test set with predictions: {test_results_path}")
    logger.info(f"  Contains {len(test_results_df)} samples for verification")
    
    # ========================================
    # Step 7: Generate Summary Report
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Generating training report")
    logger.info("=" * 70)
    
    report_path = models_dir / 'training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("AQI ML MODEL TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples: {len(df):,}\n")
        f.write(f"Training:      {len(train_df):,} ({100*len(train_df)/len(df):.1f}%)\n")
        f.write(f"Validation:    {len(val_df):,} ({100*len(val_df)/len(df):.1f}%)\n")
        f.write(f"Test:          {len(test_df):,} ({100*len(test_df)/len(df):.1f}%)\n")
        f.write(f"Features:      {len(X_train.columns)}\n\n")
        
        f.write("PM2.5 MODEL PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Validation R²:   {val_metrics['r2']:.4f}\n")
        f.write(f"Validation RMSE: {val_metrics['rmse']:.2f} µg/m³\n")
        f.write(f"Validation MAE:  {val_metrics['mae']:.2f} µg/m³\n")
        f.write(f"Test R²:         {test_metrics['r2']:.4f}\n")
        f.write(f"Test RMSE:       {test_metrics['rmse']:.2f} µg/m³\n")
        f.write(f"Test MAE:        {test_metrics['mae']:.2f} µg/m³\n\n")
        
        f.write("PM2.5 TOP 10 FEATURES\n")
        f.write("-" * 70 + "\n")
        for i, row in pm25_importance.head(10).iterrows():
            f.write(f"  {i+1:2d}. {row['Feature']:20s} {row['Importance']:.4f}\n")
        f.write("\n")
        
        f.write("PM10 MODEL PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Validation R²:   {val_metrics_pm10['r2']:.4f}\n")
        f.write(f"Validation RMSE: {val_metrics_pm10['rmse']:.2f} µg/m³\n")
        f.write(f"Validation MAE:  {val_metrics_pm10['mae']:.2f} µg/m³\n")
        f.write(f"Test R²:         {test_metrics_pm10['r2']:.4f}\n")
        f.write(f"Test RMSE:       {test_metrics_pm10['rmse']:.2f} µg/m³\n")
        f.write(f"Test MAE:        {test_metrics_pm10['mae']:.2f} µg/m³\n\n")
        
        f.write("PM10 TOP 10 FEATURES\n")
        f.write("-" * 70 + "\n")
        for i, row in pm10_importance.head(10).iterrows():
            f.write(f"  {i+1:2d}. {row['Feature']:20s} {row['Importance']:.4f}\n")
        f.write("\n")
        
        f.write("OUTPUT FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"  • pm25_rf_model.pkl\n")
        f.write(f"  • pm10_rf_model.pkl\n")
        f.write(f"  • test_set_results.csv (test data + predictions)\n")
        f.write(f"  • plots/pm25_predictions_val.png\n")
        f.write(f"  • plots/pm25_predictions_test.png\n")
        f.write(f"  • plots/pm25_feature_importance.png\n")
        f.write(f"  • plots/pm10_predictions_val.png\n")
        f.write(f"  • plots/pm10_predictions_test.png\n")
        f.write(f"  • plots/pm10_feature_importance.png\n")
    
    logger.info(f"✓ Report saved: {report_path}")
    
    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\n✓ PM2.5 Model: R² = {val_metrics['r2']:.4f}, RMSE = {val_metrics['rmse']:.2f} µg/m³")
    logger.info(f"✓ PM10 Model:  R² = {val_metrics_pm10['r2']:.4f}, RMSE = {val_metrics_pm10['rmse']:.2f} µg/m³")
    logger.info(f"\n✓ Models saved to: {models_dir}")
    logger.info(f"✓ Plots saved to: {plots_dir}")
    logger.info(f"✓ Report: {report_path}")


if __name__ == "__main__":
    main()
