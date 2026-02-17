"""ML training utilities for AQI prediction models."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from typing import Tuple, Dict, Any
import logging


class AQIModelTrainer:
    """Trainer for Random Forest AQI prediction models."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize trainer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.model = None
        self.feature_names = None
        
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        **kwargs
    ) -> RandomForestRegressor:
        """Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
            **kwargs: Additional RF parameters
            
        Returns:
            Trained model
        """
        self.logger.info(f"Training Random Forest with {n_estimators} trees...")
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Features: {X_train.shape[1]}")
        
        self.feature_names = list(X_train.columns)
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=1,
            **kwargs
        )
        
        self.model.fit(X_train, y_train)
        
        self.logger.info("✓ Training complete")
        
        return self.model
    
    def evaluate_model(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        dataset_name: str = "Validation"
    ) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X: Features
            y_true: True targets
            dataset_name: Name of dataset (for logging)
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X)
        
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'samples': len(y_true)
        }
        
        self.logger.info(f"\n{dataset_name} Metrics:")
        self.logger.info(f"  R²:   {metrics['r2']:.4f}")
        self.logger.info(f"  RMSE: {metrics['rmse']:.2f} µg/m³")
        self.logger.info(f"  MAE:  {metrics['mae']:.2f} µg/m³")
        
        return metrics
    
    def plot_predictions(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        output_path: Path,
        title: str = "Predicted vs Actual",
        target_name: str = "PM"
    ):
        """Create predicted vs actual scatter plot.
        
        Args:
            X: Features
            y_true: True targets
            output_path: Path to save plot
            title: Plot title
            target_name: Name of target variable
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel(f'Actual {target_name} (µg/m³)', fontsize=12)
        ax.set_ylabel(f'Predicted {target_name} (µg/m³)', fontsize=12)
        ax.set_title(f'{title}\nR² = {r2:.4f}, RMSE = {rmse:.2f} µg/m³', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved plot: {output_path}")
    
    def plot_feature_importance(
        self,
        output_path: Path,
        top_n: int = 15,
        title: str = "Feature Importance"
    ):
        """Plot feature importance.
        
        Args:
            output_path: Path to save plot
            top_n: Number of top features to show
            title: Plot title
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': [self.feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.barplot(
            data=importance_df,
            x='Importance',
            y='Feature',
            palette='viridis',
            ax=ax
        )
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'{title} (Top {top_n})', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved feature importance: {output_path}")
        
        return importance_df
    
    def save_model(self, output_path: Path):
        """Save trained model.
        
        Args:
            output_path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, output_path)
        self.logger.info(f"✓ Saved model: {output_path}")
    
    @staticmethod
    def load_model(model_path: Path) -> Tuple[RandomForestRegressor, list]:
        """Load trained model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Tuple of (model, feature_names)
        """
        model_data = joblib.load(model_path)
        return model_data['model'], model_data['feature_names']
