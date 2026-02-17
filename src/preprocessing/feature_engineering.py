"""
Feature engineering module for creating derived features from satellite and weather data.
"""

import numpy as np
import pandas as pd
from typing import List


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features from Date column.
    
    Args:
        df: DataFrame with 'Date' column
    
    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract temporal features
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Week'] = df['Date'].dt.isocalendar().week
    
    # Season (meteorological seasons for Northern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    df['Season'] = df['Month'].apply(get_season)
    
    # Is weekend
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    return df


def add_meteorological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived meteorological features from ERA5 variables.
    
    Requires columns: U10, V10, T2M, D2M
    
    Args:
        df: DataFrame with ERA5 variables
    
    Returns:
        DataFrame with additional meteorological features
    """
    df = df.copy()
    
    # Wind speed (m/s)
    if 'U10' in df.columns and 'V10' in df.columns:
        df['WindSpeed'] = np.sqrt(df['U10']**2 + df['V10']**2)
        
        # Wind direction (degrees, 0 = North, 90 = East)
        df['WindDirection'] = np.degrees(np.arctan2(df['V10'], df['U10']))
        # Convert from -180-180 to 0-360
        df['WindDirection'] = (df['WindDirection'] + 360) % 360
    
    # Relative Humidity (approximation from T2m and D2m)
    # Using Magnus formula
    if 'T2M' in df.columns and 'D2M' in df.columns:
        def calc_relative_humidity(temp_k, dewpoint_k):
            """Calculate relative humidity from temperature and dewpoint (Kelvin)"""
            # Convert to Celsius
            temp_c = temp_k - 273.15
            dewpoint_c = dewpoint_k - 273.15
            
            # Magnus formula parameters
            a = 17.27
            b = 237.7
            
            # Saturation vapor pressure at temperature
            es = 6.112 * np.exp((a * temp_c) / (b + temp_c))
            
            # Actual vapor pressure at dewpoint
            e = 6.112 * np.exp((a * dewpoint_c) / (b + dewpoint_c))
            
            # Relative humidity (%)
            rh = 100 * (e / es)
            
            # Clip to valid range
            return np.clip(rh, 0, 100)
        
        df['RelativeHumidity'] = calc_relative_humidity(df['T2M'], df['D2M'])
    
    # Temperature in Celsius (for readability)
    if 'T2M' in df.columns:
        df['TempCelsius'] = df['T2M'] - 273.15
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between different variables.
    
    Args:
        df: DataFrame with satellite and weather features
    
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    # AOD × Temperature interaction (high AOD + high temp often means more PM)
    if 'AOD' in df.columns and 'TempCelsius' in df.columns:
        df['AOD_Temp'] = df['AOD'] * df['TempCelsius']
    
    # NO2 × AOD interaction
    if 'NO2' in df.columns and 'AOD' in df.columns:
        df['NO2_AOD'] = df['NO2'] * df['AOD']
    
    # Wind speed bins (calm, moderate, strong)
    if 'WindSpeed' in df.columns:
        df['WindSpeedBin'] = pd.cut(
            df['WindSpeed'],
            bins=[0, 3, 7, float('inf')],
            labels=[0, 1, 2]  # 0=calm, 1=moderate, 2=strong
        ).astype(float)
    
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Args:
        df: DataFrame with basic features
    
    Returns:
        DataFrame with all engineered features
    """
    df = add_temporal_features(df)
    df = add_meteorological_features(df)
    df = add_interaction_features(df)
    
    return df


if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    
    # Sample data
    test_df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=10),
        'U10': [3, 4, 2, 5, 3, 4, 2, 1, 3, 4],
        'V10': [2, 3, 1, 2, 4, 3, 2, 1, 2, 3],
        'T2M': [298, 299, 297, 300, 301, 299, 298, 297, 298, 299],  # Kelvin
        'D2M': [290, 291, 289, 292, 293, 291, 290, 289, 290, 291],  # Kelvin
        'AOD': [0.3, 0.5, 0.2, 0.6, 0.4, 0.3, 0.5, 0.2, 0.4, 0.5],
        'NO2': [45, 50, 40, 55, 48, 46, 51, 42, 47, 52]
    })
    
    print("Original data:")
    print(test_df.head())
    
    print("\nWith engineered features:")
    result = create_all_features(test_df)
    print(result.head())
    print(f"\nTotal features: {len(result.columns)}")
