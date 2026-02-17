"""Regenerate the training data summary file with proper encoding."""
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load the data
df = pd.read_csv('data/processed/training_data_2022-2024.csv')

# Generate summary
summary_path = Path('data/processed/training_data_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("AQI ML Training Dataset Summary\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"Total samples: {len(df):,}\n")
    f.write(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")
    f.write(f"Number of stations: {df['Station'].nunique()}\n")
    f.write(f"Number of features: {df.shape[1]}\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("FEATURES\n")
    f.write("=" * 70 + "\n")
    feature_cols = [c for c in df.columns if c not in ['Date', 'Station', 'PM2.5', 'PM10']]
    for col in feature_cols:
        missing_count = df[col].isna().sum()
        missing_pct = 100 * missing_count / len(df)
        if missing_pct > 0:
            f.write(f"  • {col:20s}: {missing_pct:5.1f}% missing ({missing_count:,} values)\n")
        else:
            f.write(f"  ✓ {col:20s}: Complete\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("TARGET VARIABLES\n")
    f.write("=" * 70 + "\n")
    f.write(f"  PM2.5: {df['PM2.5'].min():.1f} - {df['PM2.5'].max():.1f} µg/m³ ")
    f.write(f"(mean: {df['PM2.5'].mean():.1f}, std: {df['PM2.5'].std():.1f})\n")
    f.write(f"  PM10:  {df['PM10'].min():.1f} - {df['PM10'].max():.1f} µg/m³ ")
    f.write(f"(mean: {df['PM10'].mean():.1f}, std: {df['PM10'].std():.1f})\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("OUTPUT FILES\n")
    f.write("=" * 70 + "\n")
    f.write(f"  • training_data_2022-2024.csv\n")
    f.write(f"  • training_data_2022-2024.parquet\n")
    f.write(f"  • VERIFICATION_REPORT.txt\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("STATUS: ✓ Ready for ML Training\n")
    f.write("=" * 70 + "\n")

print(f"✓ Summary regenerated: {summary_path}")
print("\nPreview:")
print("-" * 70)
with open(summary_path, 'r', encoding='utf-8') as f:
    print(f.read())
