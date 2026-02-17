# Data Preprocessing Guide

## 🎯 Overview

This preprocessing pipeline transforms raw satellite and weather data into a machine learning training dataset.

**Input**: Raw data (GeoTIFF, NetCDF)  
**Output**: Training dataset with ~12,000-15,000 samples

---

## 💡 What We're Actually Doing (Simplified Approach)

### ✅ What This Pipeline DOES:

**Point-Based Extraction** - We extract satellite values at 20 specific ground station locations:
- ✓ Load 20 ground stations with their coordinates
- ✓ Extract NO₂ pixel value at each station's (lat, lon) from 1,094 GeoTIFF files
- ✓ Extract AOD pixel value at each station's (lat, lon) from 1,095 GeoTIFF files
- ✓ Extract ERA5 weather at each station's nearest grid point from NetCDF
- ✓ Match with ground-measured PM2.5 and PM10
- ✓ Create tabular training dataset (rows × columns)

### ❌ What We're NOT Doing:

**Full Grid Processing** - We skip these steps because we only need values at 20 points:
- ✗ Creating new aligned raster grids covering all of Abu Dhabi
- ✗ Reprojecting to EPSG:3857 coordinate system
- ✗ Resampling to uniform 1km resolution
- ✗ Temporal interpolation to fill missing days
- ✗ Saving as Zarr format (that's for gridded data)

### 🧠 Why This Approach is Better:

**Original plan**: Create full aligned raster grids → Extract at stations  
**Our approach**: Directly extract at stations

**Benefits:**
- ⚡ **Much faster** (10-15 min vs hours)
- 💾 **Less storage** (only keep values we need)
- 🎯 **Simpler** (fewer steps = fewer bugs)
- ✅ **Same ML result** (we get identical training data!)

**Output format**: Tabular data (CSV/Parquet), not raster grids
```
Date        | Station      | NO2  | AOD  | Temp | ... | PM2.5 | PM10
------------|--------------|------|------|------|-----|-------|------
2022-01-01  | Khalifa Sch. | 45.2 | 0.35 | 298K | ... | 38.5  | 82.1
2022-01-01  | Baniyas Sch. | 42.1 | 0.32 | 297K | ... | 35.2  | 75.3
```

---

## 📋 Quick Start

### Run Preprocessing

```bash
conda activate aqi-ml
python scripts/02_preprocess_data.py
```

### Expected Runtime
- **Total**: 30-60 minutes
- Steps:
  - Load ground data: ~1 min
  - Extract NO₂: ~10-15 min (1,094 files)
  - Extract AOD: ~10-15 min (1,095 files)
  - Extract ERA5: ~5 min (1 file)
  - Merge & feature engineering: ~2 min
  - Save: ~1 min

---

## 🔄 Processing Workflow

### Step 1: Load Ground Station Data
- **Input**: `H:\AQI\00_Ancillary_Data\EAD_Hourly_2022-2024_AQ_Points_AQI.csv`
- **Process**:
  - Parse hourly measurements from 20 stations
  - Filter 2022-2024 date range
  - Quality control (remove invalid PM values)
  - Aggregate hourly → daily (mean)
- **Output**: `ground_data_daily.csv` (~21,920 station-days)

### Step 2: Extract NO₂ at Stations
- **Input**: 1,094 NO₂ GeoTIFF files
- **Process**: For each station, extract pixel value at (lat, lon) from each file
- **Output**: `no2_at_stations.csv`

### Step 3: Extract MODIS AOD at Stations
- **Input**: 1,095 MODIS AOD GeoTIFF files
- **Process**: For each station, extract pixel value from each file
- **Output**: `aod_at_stations.csv`

### Step 4: Extract ERA5 Weather at Stations
- **Input**: ERA5 merged NetCDF file
- **Process**: For each station, select nearest grid point and extract 6 variables
- **Output**: `era5_at_stations.csv`

### Step 5-7: Merge, Engineer, Clean
- Merge all features on (Date, Station)
- Add derived features (wind speed, humidity, temporal)
- Remove rows with missing satellite data
- Remove invalid PM values

### Step 8: Save Training Dataset
- **CSV**: `training_data_2022-2024.csv` (for inspection)
- **Parquet**: `training_data_2022-2024.parquet` (for ML)
- **Summary**: `training_data_summary.txt` (metadata)

---

## 📊 Output Dataset Structure

### Expected Samples
- **Theoretical max**: 20 stations × 1,096 days = 21,920
- **Actual**: ~12,000-15,000 (60-70% coverage due to clouds/gaps)

### Features (~20 columns)

**Satellite Features:**
-  `NO2` - Nitrogen dioxide column density
- `AOD` - Aerosol Optical Depth at 550nm

**Weather Features (ERA5):**
- `T2M` - 2m temperature (Kelvin)
- `D2M` - 2m dewpoint temperature (Kelvin)
- `U10` - 10m U-wind component (m/s)
- `V10` - 10m V-wind component (m/s)
- `SP` - Surface pressure (Pa)
- `BLH` - Boundary layer height (m)

**Derived Meteorological:**
- `WindSpeed` - Calculated from U10, V10
- `WindDirection` - Degrees (0-360)
- `RelativeHumidity` - Percent (0-100)
- `TempCelsius` - Temperature in °C

**Temporal Features:**
- `DayOfYear` - 1-365
- `Month` - 1-12
- `Season` - 0=Winter, 1=Spring, 2=Summer, 3=Autumn
- `DayOfWeek` - 0=Monday, 6=Sunday
- `IsWeekend` - 0/1

**Location:**
- `Latitude`, `Longitude`

**Targets:**
- `PM2.5` - Ground-measured PM2.5 (µg/m³)
- `PM10` - Ground-measured PM10 (µg/m³)

---

## 🔍 Data Quality Checks

The pipeline automatically:
- ✅ Removes rows where both NO₂ and AOD are missing
- ✅ Removes rows with PM2.5 < 0 or PM10 < 0
- ✅ Logs missing data percentages
- ✅ Validates value ranges

### Expected Missing Data
- **NO₂**: 20-40% (cloud cover, sensor gaps)
- **AOD**: 20-40% (cloud cover, algorithm failures)
- **ERA5**: <1% (very complete)

### Value Ranges (Abu Dhabi)
- **PM2.5**: 5-150 µg/m³ (typical), up to 200-300 (dust events)
- **PM10**: 20-300 µg/m³ (typical), up to 500+ (dust events)
- **AOD**: 0.1-1.5 (typical), up to 3+ (dust storms)
- **Temperature**: 15-45°C (293-318K)

---

## 📁 Output Files

All files saved to: `H:\AQI\data\processed\`

### Main Outputs
- `training_data_2022-2024.csv` - Full dataset (inspection)
- `training_data_2022-2024.parquet` - Optimized for ML
- `training_data_summary.txt` - Statistics

### Intermediate Files
- `ground_data_daily.csv` - Aggregated ground measurements
- `no2_at_stations.csv` - NO₂ extracted at stations
- `aod_at_stations.csv` - AOD extracted at stations
- `era5_at_stations.csv` - Weather extracted at stations

---

## 🐛 Troubleshooting

### Issue: "NO2 directory not found"
**Solution**: Ensure NO₂ data was downloaded in Phase 1
```bash
ls H:\AQI\data\raw\NO2\*.tif | wc -l  # Should show ~1094
```

### Issue: "ERA5 file not found"
**Solution**: Check ERA5 merge completed successfully
```bash
ls H:\AQI\data\raw\ERA5\ERA5_merged_*.nc
```

### Issue: Low sample count (<10,000)
**Cause**: Excessive missing satellite data
**Solution**: Review download quality, may need to re-download certain date ranges

### Issue: Memory error
**Solution**: Process in chunks or increase available RAM

---

## ✅ Success Criteria

Before proceeding to ML training, verify:

1. ✅ Training dataset has 12,000+  samples
2. ✅ All 20 stations represented
3. ✅ PM2.5 and PM10 have no missing values
4. ✅ NO₂ and AOD coverage >50%
5. ✅ Value ranges look reasonable
6. ✅ Parquet file created successfully

---

## 🚀 Next Steps

Once preprocessing completes successfully:

1. **Review summary**: Check `training_data_summary.txt`
2. **Inspect data**: Quick look at CSV to verify
3. **Proceed to ML**: Run `scripts/03_train_models.py`

Expected training accuracy:
- PM2.5: R² > 0.75
- PM10: R² > 0.73
