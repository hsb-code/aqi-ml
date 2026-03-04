
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/training_data_full.parquet")

def analyze():
    if not DATA_PATH.exists():
        print(f"File not found: {DATA_PATH}")
        return

    try:
        df = pd.read_parquet(DATA_PATH)
        print(f"Dataset Shape: {df.shape}")
        print("\nInfo:")
        print(df.info())
        print("\nHead:")
        print(df.head())
        
        # Test a small correlation
        targets = ["PM25", "PM10"]
        cols_to_corr = [c for c in df.columns if df[c].dtype in ['float64', 'int64', 'float32']]
        for target in targets:
            if target in df.columns:
                print(f"\nCorrelations with {target}:")
                # Drop non-numeric and target itself
                numeric_df = df[cols_to_corr]
                corr = numeric_df.corr()[target].sort_values(ascending=False)
                print(corr.head(10))
    except Exception as e:
        print(f"Error reading parquet: {e}")

if __name__ == "__main__":
    analyze()
