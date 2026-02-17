"""
Improved ML training script - Experiment 2

Improvements over Experiment 1:
1. Hyperparameter tuning with GridSearchCV
2. XGBoost algorithm (usually better than Random Forest)
3. Lag features (yesterday's PM, rolling averages)
4. Comparison of multiple algorithms

Usage:
    python scripts/04_train_models_exp2.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.trainer import AQIModelTrainer
from src.config.settings import config
from src.utils.logger import setup_logger


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling window features.
    
    Args:
        df: DataFrame with Date, Station, PM2.5, PM10
        
    Returns:
        DataFrame with additional lag features
    """
    df = df.sort_values(['Station', 'Date']).copy()
    
    # Group by station
    for station in df['Station'].unique():
        mask = df['Station'] == station
        
        # Lag features (yesterday's values)
        df.loc[mask, 'PM2.5_lag1'] = df.loc[mask, 'PM2.5'].shift(1)
        df.loc[mask, 'PM10_lag1'] = df.loc[mask, 'PM10'].shift(1)
        
        # Rolling averages (3-day, 7-day)
        df.loc[mask, 'PM2.5_rolling3'] = df.loc[mask, 'PM2.5'].rolling(3, min_periods=1).mean()
        df.loc[mask, 'PM10_rolling3'] = df.loc[mask, 'PM10'].rolling(3, min_periods=1).mean()
        df.loc[mask, 'PM2.5_rolling7'] = df.loc[mask, 'PM2.5'].rolling(7, min_periods=1).mean()
        df.loc[mask, 'PM10_rolling7'] = df.loc[mask, 'PM10'].rolling(7, min_periods=1).mean()
        
        # Rolling AOD and NO2
        df.loc[mask, 'AOD_rolling3'] = df.loc[mask, 'AOD'].rolling(3, min_periods=1).mean()
        df.loc[mask, 'NO2_rolling3'] = df.loc[mask, 'NO2'].rolling(3, min_periods=1).mean()
    
    return df


def prepare_data(df: pd.DataFrame, target_col: str):
    """Prepare features and target for training.
    
    Args:
        df: Training dataframe
        target_col: Name of target column ('PM2.5' or 'PM10')
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Define feature columns (exclude non-predictive columns and target)
    exclude_cols = ['Date', 'Station', 'PM2.5', 'PM10']
    
    # Remove lag features of the target we're predicting
    if target_col == 'PM2.5':
        exclude_cols.extend(['PM2.5_lag1', 'PM2.5_rolling3', 'PM2.5_rolling7'])
    else:
        exclude_cols.extend(['PM10_lag1', 'PM10_rolling3', 'PM10_rolling7'])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y


def train_xgboost(X_train, y_train, X_val, y_val, logger):
    """Train XGBoost model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        logger: Logger instance
        
    Returns:
        Best model and GridSearch results
    """
    logger.info("Training XGBoost with hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [10, 15, 20],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Base model
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    
    # Grid search
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} combinations...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV R²: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search


def train_tuned_rf(X_train, y_train, logger):
    """Train Random Forest with tuned hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        logger: Logger instance
        
    Returns:
        Tuned Random Forest model
    """
    logger.info("Training Random Forest with improved hyperparameters...")
    
    # Tuned parameters (based on common best practices)
    model = RandomForestRegressor(
        n_estimators=300,  # More trees
        max_depth=30,  # Deeper trees
        min_samples_split=3,  # More flexible splitting
        min_samples_leaf=1,
        max_features='sqrt',  # Feature sampling
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    return model


def main():
    """Main training workflow for Experiment 2."""
    
    # Setup logging
    logger = setup_logger('ml_training_exp2', log_file='ml_training_exp2.log')
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: IMPROVED ML MODELS")
    logger.info("=" * 70)
    logger.info("\nImprovements:")
    logger.info("  1. Hyperparameter tuning")
    logger.info("  2. XGBoost algorithm")
    logger.info("  3. Lag features (yesterday's PM, rolling averages)")
    
    # Initialize paths
    processed_dir = Path(config.paths.processed_data)
    exp_dir = Path('models/exp2')
    plots_dir = exp_dir / 'plots'
    
    # Load data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Loading and engineering features")
    logger.info("=" * 70)
    
    data_path = processed_dir / 'training_data_2022-2024.parquet'
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} samples")
    
    # Create lag features
    logger.info("\nCreating lag features...")
    df = create_lag_features(df)
    logger.info(f"Added lag features: PM_lag1, PM_rolling3, PM_rolling7, AOD_rolling3, NO2_rolling3")
    
    # Handle missing values
    logger.info("\nHandling missing values...")
    initial_len = len(df)
    df = df.dropna(subset=['NO2', 'AOD'], how='all')
    logger.info(f"  Dropped {initial_len - len(df)} rows with no satellite data")
    
    # Fill remaining NaN with median
    for col in df.columns:
        if df[col].isnull().any() and col not in ['Date', 'Station']:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    logger.info(f"Clean dataset: {len(df):,} samples")
    
    # Split data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Splitting data")
    logger.info("=" * 70)
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Station'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Station'])
    
    logger.info(f"Training:   {len(train_df):,} samples")
    logger.info(f"Validation: {len(val_df):,} samples")
    logger.info(f"Test:       {len(test_df):,} samples")
    
    # PM2.5 Models
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Training PM2.5 models")
    logger.info("=" * 70)
    
    X_train, y_train = prepare_data(train_df, 'PM2.5')
    X_val, y_val = prepare_data(val_df, 'PM2.5')
    X_test, y_test = prepare_data(test_df, 'PM2.5')
    
    logger.info(f"\nFeatures: {X_train.shape[1]} columns")
    logger.info(f"New features: {[c for c in X_train.columns if 'lag' in c or 'rolling' in c]}")
    
    # Train XGBoost
    xgb_pm25, pm25_grid = train_xgboost(X_train, y_train, X_val, y_val, logger)
    
    # Train tuned RF
    rf_pm25 = train_tuned_rf(X_train, y_train, logger)
    
    # Evaluate both
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    logger.info("\n--- XGBoost PM2.5 Performance ---")
    y_pred = xgb_pm25.predict(X_val)
    r2_xgb = r2_score(y_val, y_pred)
    rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred))
    mae_xgb = mean_absolute_error(y_val, y_pred)
    logger.info(f"Validation R²:   {r2_xgb:.4f}")
    logger.info(f"Validation RMSE: {rmse_xgb:.2f} ug/m3")
    logger.info(f"Validation MAE:  {mae_xgb:.2f} ug/m3")
    
    logger.info("\n--- Tuned RF PM2.5 Performance ---")
    y_pred_rf = rf_pm25.predict(X_val)
    r2_rf = r2_score(y_val, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
    mae_rf = mean_absolute_error(y_val, y_pred_rf)
    logger.info(f"Validation R²:   {r2_rf:.4f}")
    logger.info(f"Validation RMSE: {rmse_rf:.2f} ug/m3")
    logger.info(f"Validation MAE:  {mae_rf:.2f} ug/m3")
    
    # Choose best
    best_pm25 = xgb_pm25 if r2_xgb > r2_rf else rf_pm25
    best_pm25_name = "XGBoost" if r2_xgb > r2_rf else "Random Forest"
    logger.info(f"\nBest PM2.5 model: {best_pm25_name}")
    
    # PM10 Models
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Training PM10 models")
    logger.info("=" * 70)
    
    X_train, y_train = prepare_data(train_df, 'PM10')
    X_val, y_val = prepare_data(val_df, 'PM10')
    X_test, y_test = prepare_data(test_df, 'PM10')
    
    # Train XGBoost
    xgb_pm10, pm10_grid = train_xgboost(X_train, y_train, X_val, y_val, logger)
    
    # Train tuned RF
    rf_pm10 = train_tuned_rf(X_train, y_train, logger)
    
    # Evaluate both
    logger.info("\n--- XGBoost PM10 Performance ---")
    y_pred = xgb_pm10.predict(X_val)
    r2_xgb_pm10 = r2_score(y_val, y_pred)
    rmse_xgb_pm10 = np.sqrt(mean_squared_error(y_val, y_pred))
    mae_xgb_pm10 = mean_absolute_error(y_val, y_pred)
    logger.info(f"Validation R²:   {r2_xgb_pm10:.4f}")
    logger.info(f"Validation RMSE: {rmse_xgb_pm10:.2f} ug/m3")
    logger.info(f"Validation MAE:  {mae_xgb_pm10:.2f} ug/m3")
    
    logger.info("\n--- Tuned RF PM10 Performance ---")
    y_pred_rf = rf_pm10.predict(X_val)
    r2_rf_pm10 = r2_score(y_val, y_pred_rf)
    rmse_rf_pm10 = np.sqrt(mean_squared_error(y_val, y_pred_rf))
    mae_rf_pm10 = mean_absolute_error(y_val, y_pred_rf)
    logger.info(f"Validation R²:   {r2_rf_pm10:.4f}")
    logger.info(f"Validation RMSE: {rmse_rf_pm10:.2f} ug/m3")
    logger.info(f"Validation MAE:  {mae_rf_pm10:.2f} ug/m3")
    
    # Choose best
    best_pm10 = xgb_pm10 if r2_xgb_pm10 > r2_rf_pm10 else rf_pm10
    best_pm10_name = "XGBoost" if r2_xgb_pm10 > r2_rf_pm10 else "Random Forest"
    logger.info(f"\nBest PM10 model: {best_pm10_name}")
    
    # Save models
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Saving models and results")
    logger.info("=" * 70)
    
    import joblib
    joblib.dump(best_pm25, exp_dir / 'pm25_best_model.pkl')
    joblib.dump(best_pm10, exp_dir / 'pm10_best_model.pkl')
    logger.info(f"Saved models to {exp_dir}")
    
    # Save test predictions
    X_test_pm25, y_test_pm25 = prepare_data(test_df, 'PM2.5')
    X_test_pm10, y_test_pm10 = prepare_data(test_df, 'PM10')
    
    test_results = test_df.copy()
    test_results['PM2.5_Predicted'] = best_pm25.predict(X_test_pm25)
    test_results['PM10_Predicted'] = best_pm10.predict(X_test_pm10)
    test_results.to_csv(exp_dir / 'test_set_results.csv', index=False)
    
    # Generate report
    report_path = exp_dir / 'training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EXPERIMENT 2: IMPROVED ML MODELS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("IMPROVEMENTS\n")
        f.write("-" * 70 + "\n")
        f.write("1. Hyperparameter tuning with GridSearchCV\n")
        f.write("2. XGBoost algorithm\n")
        f.write("3. Lag features (PM_lag1, rolling3, rolling7)\n")
        f.write("4. Additional rolling features (AOD, NO2)\n\n")
        
        f.write("DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples: {len(df):,}\n")
        f.write(f"Features: {X_train.shape[1]} (up from 23 in exp1)\n\n")
        
        f.write(f"PM2.5 BEST MODEL: {best_pm25_name}\n")
        f.write("-" * 70 + "\n")
        best_r2 = r2_xgb if best_pm25_name == "XGBoost" else r2_rf
        best_rmse = rmse_xgb if best_pm25_name == "XGBoost" else rmse_rf
        best_mae = mae_xgb if best_pm25_name == "XGBoost" else mae_rf
        f.write(f"Validation R²:   {best_r2:.4f}\n")
        f.write(f"Validation RMSE: {best_rmse:.2f} ug/m³\n")
        f.write(f"Validation MAE:  {best_mae:.2f} ug/m³\n\n")
        
        if best_pm25_name == "XGBoost":
            f.write("XGBoost Parameters:\n")
            for k, v in pm25_grid.best_params_.items():
                f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        f.write(f"PM10 BEST MODEL: {best_pm10_name}\n")
        f.write("-" * 70 + "\n")
        best_r2_pm10 = r2_xgb_pm10 if best_pm10_name == "XGBoost" else r2_rf_pm10
        best_rmse_pm10 = rmse_xgb_pm10 if best_pm10_name == "XGBoost" else rmse_rf_pm10
        best_mae_pm10 = mae_xgb_pm10 if best_pm10_name == "XGBoost" else mae_rf_pm10
        f.write(f"Validation R²:   {best_r2_pm10:.4f}\n")
        f.write(f"Validation RMSE: {best_rmse_pm10:.2f} ug/m³\n")
        f.write(f"Validation MAE:  {best_mae_pm10:.2f} ug/m³\n\n")
        
        if best_pm10_name == "XGBoost":
            f.write("XGBoost Parameters:\n")
            for k, v in pm10_grid.best_params_.items():
                f.write(f"  {k}: {v}\n")
    
    logger.info(f"Report saved: {report_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\nPM2.5 {best_pm25_name}: R² = {best_r2:.4f}")
    logger.info(f"PM10 {best_pm10_name}:  R² = {best_r2_pm10:.4f}")


if __name__ == "__main__":
    main()
