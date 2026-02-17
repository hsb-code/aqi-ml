"""Generate clean dataset verification report."""
import pandas as pd

# Load data
df = pd.read_csv('data/processed/training_data_2022-2024.csv')

with open('data/processed/VERIFICATION_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("AQI ML TRAINING DATASET - VERIFICATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("ALL 27 COLUMNS\n")
    f.write("=" * 70 + "\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"{i:2d}. {col}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("MISSING DATA ANALYSIS\n")
    f.write("=" * 70 + "\n")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    
    complete_cols = []
    partial_cols = []
    
    for col in df.columns:
        if missing[col] == 0:
            complete_cols.append(col)
        else:
            partial_cols.append((col, missing[col], missing_pct[col]))
    
    f.write(f"\nComplete columns ({len(complete_cols)}):\n")
    for col in complete_cols:
        f.write(f"  ✓ {col}\n")
    
    if partial_cols:
        f.write(f"\nColumns with missing data ({len(partial_cols)}):\n")
        for col, count, pct in partial_cols:
            f.write(f"  • {col:20s}: {count:5d} missing ({pct:5.1f}%)\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("ERA5 WEATHER VARIABLES (Critical Verification)\n")
    f.write("=" * 70 + "\n")
    era5_vars = ['T2M', 'D2M', 'U10', 'V10', 'SP', 'BLH']
    all_era5_complete = True
    for v in era5_vars:
        present = df[v].notna().sum()
        pct = 100 * present / len(df)
        status = "PASS" if pct == 100 else "FAIL"
        if pct < 100:
            all_era5_complete = False
        f.write(f"  {v:5s}: {present:5d}/{len(df)} ({pct:6.2f}%) - {status}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("TARGET VARIABLES\n")
    f.write("=" * 70 + "\n")
    f.write(f"  PM2.5: {df['PM2.5'].min():6.1f} - {df['PM2.5'].max():6.1f} µg/m³\n")
    f.write(f"  PM10:  {df['PM10'].min():6.1f} - {df['PM10'].max():6.1f} µg/m³\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("SATELLITE DATA\n")
    f.write("=" * 70 + "\n")
    f.write(f"  NO2: {df['NO2'].notna().sum()}/{len(df)} ({100*df['NO2'].notna().sum()/len(df):.1f}%)\n")
    f.write(f"  AOD: {df['AOD'].notna().sum()}/{len(df)} ({100*df['AOD'].notna().sum()/len(df):.1f}%)\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("FINAL VERIFICATION\n")
    f.write("=" * 70 + "\n")
    if all_era5_complete:
        f.write("  ✓ ERA5 data: COMPLETE (100% coverage)\n")
    else:
        f.write("  ✗ ERA5 data: INCOMPLETE\n")
    
    f.write(f"  ✓ Total samples: {df.shape[0]:,} station-days\n")
    f.write(f"  ✓ Feature count: {df.shape[1]} columns\n")
    f.write(f"  ✓ Date range: {df['Date'].min()} to {df['Date'].max()}\n")
    f.write(f"  ✓ Stations: {df['Station'].nunique()} unique\n")
    f.write("\n")
    if all_era5_complete and df.shape[0] > 20000:
        f.write("  ★ DATASET READY FOR ML TRAINING! ★\n")
    
    f.write("\n" + "=" * 70 + "\n")

print(f"Verification report saved to: data/processed/VERIFICATION_REPORT.txt")

# Also print to screen
with open('data/processed/VERIFICATION_REPORT.txt', 'r', encoding='utf-8') as f:
    print(f.read())
