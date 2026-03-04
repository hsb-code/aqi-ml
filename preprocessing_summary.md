# AQI-ML: Data Preprocessing Summary
### Full Pipeline Documentation — Sources, Transformations, Features, Quality Control
**Script entry-point:** `scripts/02_preprocess.py`  
**Pipeline module:** `src/preprocessing/pipeline.py`  
**Date range:** 2022-01-01 to 2024-12-31  
**Targets:** PM2.5 and PM10 (µg/m³, daily means per station)

---

## Data Sources

| Source | Provider | Product | Resolution | Key Variables |
|--------|----------|---------|------------|---------------|
| EAD Ground Stations | Environment Agency Abu Dhabi | Hourly AQ monitoring | Hourly / ~10 stations | PM2.5, PM10, NO2, SO2, O3, CO |
| Sentinel-5P TROPOMI | ESA / Copernicus | L2 NO2 column density | ~3.5×5.5 km, daily | NO2 (mol/m²), QA value |
| MODIS MCD19A2 | NASA | Multi-Angle AOD | 1 km, daily | Aerosol Optical Depth (DN) |
| ERA5 Reanalysis | ECMWF | Single-level hourly | ~31 km, daily | T2M, D2M, SP, BLH, U10, V10 |

---

## Pipeline Steps (Run Order)

### Step 1 — Ground Station Loading and Quality Control
**Module:** `src/preprocessing/ground_station.py`

**Source file:** `00_Ancillary_Data/EAD_Hourly_2022-2024_AQ_Points_AQI_dedup.csv`

**Operations:**
1. Parse `datetime` column and filter to `>= 2022-01-01`
2. Rename `PM2P5` → `PM25` for consistency
3. **Outlier removal** using sensor specification limits:
   - PM2.5: keep `0 ≤ PM25 ≤ 500 µg/m³` (above = instrument error)
   - PM10: keep `0 ≤ PM10 ≤ 1000 µg/m³`
   - CO: clip at `10 mg/m³` (sensor saturation)
4. Retain columns: `StationName, datetime, PM25, PM10, NO2, SO2, O3, CO, x (lon), y (lat)`

**Daily Aggregation:**
- Group by `(StationName, Date, x, y)`
- Compute mean of all pollutant columns
- Track `valid_hours` (count of non-null PM2.5 readings per day)
- **Drop days with fewer than 12 valid hourly readings** (minimum 50% daily coverage)

**Output:** One row per (station, day) with reliable daily mean pollutant levels.

---

### Step 2 — Sentinel-5P NO2 Processing
**Module:** `src/preprocessing/satellite.py`

**Source file:** `data/processed/no2_at_stations.csv`

**Operations:**
1. Load pre-extracted S5P NO2 values sampled at station coordinates
2. Apply **Copernicus official QA filter**: retain only rows with `qa_value > 0.75` (removes cloud-contaminated, snow-covered, or low-quality pixels)
3. **Unit conversion** from column density to near-surface concentration:

```
NO2_ugm3 = (NO2_mol_m2 × 46.0055 × 1e6) / BLH_m
```

This divides the total atmospheric column loading (mol/m²) by the ERA5 Boundary Layer Height (m), assuming uniform vertical mixing within the planetary boundary layer. The result is a near-surface NO2 concentration in µg/m³.

4. Validity check: drop rows where `NO2_ugm3 < 0` or `NaN` (caused by BLH ≤ 0)

**Physical constants:**
- NO2 molar mass: `46.0055 g/mol`
- Conversion factor: `×1e6` (g/m³ → µg/m³)
- `BLH_m` sourced from ERA5 (required, merged before conversion)

---

### Step 3 — MODIS AOD Processing
**Module:** `src/preprocessing/satellite.py`

**Source file:** `data/processed/aod_at_stations.csv`

**Operations:**
1. Load pre-extracted MODIS MCD19A2 AOD values at station coordinates
2. **Scale factor application**: `AOD = AOD_DN × 0.001`
   - Raw integer DN values (~0–3924) → dimensionless optical depth (~0.0–3.9)
   - Auto-detects whether scale factor was already applied (checks if `max > 10`)
3. **Range filter**: retain `0.0 ≤ AOD ≤ 5.0` (physical upper limit for optical depth; values above indicate cloud contamination or instrument error)

**Note on cloud gaps:** MODIS cannot retrieve AOD through clouds. Days with cloud cover result in `NaN` for AOD. These missing values propagate to features using AOD (e.g., `AOD_BLH_ratio`, `AOD_corrected`) and are handled by LightGBM's native NaN-aware split logic.

---

### Step 4 — ERA5 Reanalysis Processing
**Module:** `src/preprocessing/era5.py`

**Source file:** `data/processed/era5_at_stations.csv`

**Unit conversions:**

| Raw ERA5 Column | Unit | Conversion | Output Column | Unit |
|----------------|------|-----------|---------------|------|
| T2M | Kelvin | − 273.15 | T2M_C | °C |
| D2M | Kelvin | − 273.15 | D2M_C | °C |
| SP | Pascals | ÷ 100 | SP_hPa | hPa |
| BLH | metres | none | BLH | m |
| U10 | m/s | none | U10 | m/s |
| V10 | m/s | none | V10 | m/s |

**Derived meteorological variables:**

```python
WindSpeed     = sqrt(U10² + V10²)                   # m/s
WindDirection = atan2(V10, U10) × (180/π) % 360     # degrees meteorological

# Magnus formula (Alduchov & Eskridge 1996)
RH = 100 × exp(17.625 × D2M_C / (243.04 + D2M_C))
         / exp(17.625 × T2M_C / (243.04 + T2M_C))  # %
```

**Validation:** BLH ≤ 0 is set to NaN (required for valid NO2 unit conversion).

---

### Step 5 — Spatial Merge
**Module:** `src/preprocessing/features.py → merge_sources()`

All four sources are joined on `(StationName, Date)` using a **left join anchored to the ground station data**. This ensures:
- Every row has a verified ground truth PM value
- Days without satellite coverage (clouds, orbit gaps) remain as NaN — not dropped
- Station geography (`station_geography.csv`) is merged here if available

**Coverage logging** after merge:
- NO2 coverage: typically ~75–85% (QA filter + cloud gaps)
- AOD coverage: typically ~60–75% (cloud contamination higher)
- ERA5 coverage: ~100% (reanalysis has no missing days)

---

### Step 6 — Feature Engineering
**Module:** `src/preprocessing/features.py → build_features()`

#### 6a. Humidity-Corrected AOD
AOD measured by satellite includes a hygroscopic growth component: particles absorb water in humid air and scatter more light. The correction removes this artifact:

```python
f_RH = 1 / (1 − 0.95 × (RH/100))      # hygroscopic growth factor
AOD_corrected = AOD / f_RH              # dry-equivalent AOD
```

#### 6b. AOD/BLH Ratio (Core Physics Feature)
The most important engineered feature in the entire project:

```python
AOD_BLH_ratio = AOD / (BLH + 1e-6)     # clipped at 0.05
```

**Physical meaning:** AOD measures total aerosol loading in the atmosphere (the full column). BLH tells us how thick the mixing layer is. Their ratio approximates aerosol **concentration in the layer people actually breathe**. On days when BLH is low (temperature inversion, stable atmosphere), the same aerosol loading is compressed into a shallower layer — producing dangerously high surface concentrations.

#### 6c. Wind Vector Decomposition
```python
Wind_U = WindSpeed × sin(WindDirection_rad)   # east–west component
Wind_V = WindSpeed × cos(WindDirection_rad)   # north–south component
```
Encoding wind direction as two continuous components (U, V) removes the circular discontinuity at 0°/360° that would confuse gradient boosting.

#### 6d. Ventilation Index
```python
VentilationIndex = AOD / (WindSpeed + 1.0)
```
Low wind + high AOD = poor dispersion. High wind = pollution flushed away. This ratio captures the day's atmospheric capacity to clear aerosols.

#### 6e. Stability Index
```python
StabilityIndex = T2M_C − D2M_C
```
A large temperature–dewpoint spread indicates a dry, stable atmosphere prone to inversions. Small spread (high humidity, near-saturation) indicates convective mixing. Used as a proxy for temperature inversion strength.

#### 6f. Temporal Encodings
```python
DayOfYear = 1–365/366             # annual cycle proxy
Month     = 1–12
Season    = {1:winter, 2:spring, 3:summer, 4:autumn}
IsWeekend = 1 if Fri or Sat else 0   # UAE weekend (Fri–Sat)
```

#### 6g. Log Transforms
Right-skewed variables are log-transformed to reduce the influence of rare extreme events during tree splits:

```python
NO2_log = log1p(NO2_ugm3)   # NO2 values span 0–200 µg/m³
AOD_log = log1p(AOD)        # AOD spans 0–5
BLH_log = log1p(BLH)        # BLH spans 50–3000 m
```

---

### Step 7 — Final Quality Control
**Module:** `src/preprocessing/features.py → final_qc()`

Removes rows with physically impossible combinations:

| Check | Condition kept | Rationale |
|-------|---------------|-----------|
| Relative humidity | `0 ≤ RH ≤ 100` | Sensor error outside this range |
| Boundary layer height | `BLH > 0` | Required for NO2 conversion |
| AOD/BLH ratio | `0 ≤ ratio ≤ 0.05` | Beyond 0.05 = data error |
| Surface pressure | `900 ≤ SP ≤ 1100 hPa` | Outside normal atmospheric range |

---

### Step 8 — Temporal Split
**Module:** `src/preprocessing/pipeline.py → temporal_split()`

Hard date-based split — **no random shuffling at any stage**:

| Split | Date Range | Purpose |
|-------|-----------|---------|
| **Train** | 2022-01-01 → 2023-06-30 | Model learning |
| **Validation** | 2023-07-01 → 2023-12-31 | Early stopping and hyperparameter tuning |
| **Test** | 2024-01-01 → 2024-12-31 | Final held-out evaluation |

**Critical:** All lag features and rolling means are computed on the **full dataset before splitting**, so that the first days of the validation set correctly use the last days of training as their lag source. Splitting before computing lags would create artificial boundary gaps.

---

### Step 9 — Feature Scaling
**Module:** `src/preprocessing/pipeline.py`

- **Scaler:** `sklearn.preprocessing.RobustScaler`
- **Fit on training data only** — never on validation or test
- Applied to all 28 features, producing `*_scaled` columns alongside the originals
- RobustScaler uses median and IQR (not mean and std), making it robust to the extreme pollution events (dust storms, fire) present in UAE AQI data
- **Saved to:** `models/feature_scaler.pkl` for reproducible inference

---

### Step 10 — Output Files
**Directory:** `data/processed/`

| File | Description |
|------|-------------|
| `training_data_full.parquet` | Complete cleaned dataset, all rows |
| `train.parquet` | Training split (2022–Jun 2023) |
| `val.parquet` | Validation split (Jul–Dec 2023) |
| `test.parquet` | Test split (2024, held-out) |
| `models/feature_scaler.pkl` | Fitted RobustScaler |
| `station_geography.csv` | Static station metadata (elevation, distances) |

---

## Final Feature Set (28 Features)

| # | Feature | Source | Type | Notes |
|---|---------|--------|------|-------|
| 1 | `NO2_ugm3` | S5P TROPOMI | Satellite | Column-to-surface converted |
| 2 | `AOD` | MODIS MCD19A2 | Satellite | Scale factor applied |
| 3 | `AOD_corrected` | Derived | Physics | Humidity-corrected AOD |
| 4 | `AOD_BLH_ratio` | Derived | Physics | **Top SHAP feature** |
| 5 | `T2M_C` | ERA5 | Met | 2m temperature (°C) |
| 6 | `D2M_C` | ERA5 | Met | 2m dewpoint (°C) |
| 7 | `SP_hPa` | ERA5 | Met | Surface pressure |
| 8 | `BLH` | ERA5 | Met | Boundary layer height (m) |
| 9 | `WindSpeed` | ERA5 | Met | √(U10²+V10²) |
| 10 | `WindDirection` | ERA5 | Met | Meteorological degrees |
| 11 | `RH` | ERA5 | Met | Magnus formula (%) |
| 12 | `DayOfYear` | Calendar | Temporal | Annual cycle |
| 13 | `Month` | Calendar | Temporal | Seasonality |
| 14 | `Season` | Calendar | Temporal | 4-class (categorical) |
| 15 | `IsWeekend` | Calendar | Temporal | UAE Fri–Sat weekend |
| 16 | `Latitude` | Ground station | Spatial | Station y-coordinate |
| 17 | `Longitude` | Ground station | Spatial | Station x-coordinate |
| 18 | `Elevation_m` | Open-Elevation API | Geography | Station elevation |
| 19 | `Dist_Coast_km` | GeoPandas | Geography | Distance to UAE coast |
| 20 | `Dist_Corniche_km` | GeoPandas | Geography | Distance to Corniche |
| 21 | `Dist_E11_km` | GeoPandas | Geography | Distance to E11 highway |
| 22 | `NO2_log` | Derived | Log-transform | `log1p(NO2_ugm3)` |
| 23 | `AOD_log` | Derived | Log-transform | `log1p(AOD)` |
| 24 | `BLH_log` | Derived | Log-transform | `log1p(BLH)` |
| 25 | `f_RH` | Derived | Physics | Hygroscopic growth factor |
| 26 | `Wind_U` | Derived | Physics | E-W wind component |
| 27 | `Wind_V` | Derived | Physics | N-S wind component |
| 28 | `VentilationIndex` | Derived | Physics | AOD/(WindSpeed+1) |

**Additional features added per experiment:**
- `StabilityIndex` = T2M_C − D2M_C (added from Exp 16 onward)
- `PM25_lag1/2/3`, `PM10_lag1/2/3` — temporal memory lags (added in Exp 2+)
- `PM25_roll3/7/14`, `PM10_roll3/7/14` — rolling means (added in Exp 2+)
- `PM25_ewm7`, `PM10_ewm7` — exponentially weighted means (Exp 4+)
- `DRP_PM25_lag1`, `DRP_PM10_lag1` — Dynamic Regional Persistence IDW features (Exp 16+)

---

## Data Volume (Approximate)

| Split | Rows | Stations | Date Range |
|-------|------|---------|------------|
| Train | ~7,000–8,000 | ~10 | 2022-01-01 to 2023-06-30 |
| Validation | ~1,800–2,000 | ~10 | 2023-07-01 to 2023-12-31 |
| Test | ~3,500–3,700 | ~10 | 2024-01-01 to 2024-12-31 |

*Actual row counts vary due to cloud coverage gaps in satellite data and the 12-hour minimum hourly coverage filter.*

---

## Running the Pipeline

```bash
conda activate aqi-ml

# Full pipeline with default paths
python scripts/02_preprocess.py

# With custom data paths
python scripts/02_preprocess.py \
  --ground-station path/to/ead_data.csv \
  --no2  data/processed/no2_at_stations.csv \
  --aod  data/processed/aod_at_stations.csv \
  --era5 data/processed/era5_at_stations.csv \
  --output-dir data/processed \
  --model-dir models
```

Logs saved to: `logs/preprocess.log`

---

*Generated: March 2026 | AQI-ML Project | UAE Environment & Air Quality*
