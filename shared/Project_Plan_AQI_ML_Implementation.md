# Project Plan: Air Quality Index (AQI) System Enhancement
## Machine Learning Implementation for PM2.5/PM10 Estimation

---

**Project Title:** AQI System Enhancement - ML-Based PM Estimation
**Project Duration:** 4-6 Weeks
**Start Date:** [To Be Confirmed]
**Prepared By:** [Your Name]
**Date:** January 2026
**Version:** 1.0

---

## Executive Summary

This project plan outlines the implementation of a Machine Learning (ML) approach to estimate PM2.5 and PM10 concentrations from satellite data, replacing the current dependency on CAMS reanalysis data which has a 2-5 month lag. This enhancement will enable near real-time Air Quality Index (AQI) calculation with only a 1-day operational lag.

### Current State
- **Problem:** CAMS reanalysis data (PM2.5, PM10) has 2-5 month availability lag
- **Impact:** Cannot calculate AQI for recent months (e.g., December 2025)
- **Affected Pollutants:** PM2.5, PM10 only (other 4 pollutants are near real-time)

### Proposed Solution
- Develop Random Forest ML models trained on local ground station data
- Use MODIS AOD + ERA5 meteorology as input features
- Integrate predictions into existing AQI pipeline

### Expected Outcome
- Reduce data lag from **2-5 months** to **1 day**
- Maintain accuracy: R² > 0.75, RMSE < 15 µg/m³
- Enable near real-time AQI for all 6 pollutants

---

## 1. Project Objectives

### 1.1 Primary Objectives

| # | Objective | Success Metric |
|---|-----------|----------------|
| 1 | Develop ML model for PM2.5 estimation | R² > 0.75, RMSE < 15 µg/m³ |
| 2 | Develop ML model for PM10 estimation | R² > 0.73, RMSE < 25 µg/m³ |
| 3 | Reduce operational data lag | From 2-5 months to ≤1 day |
| 4 | Generate December 2025 AQI | Complete AQI maps for Dec 2025 |
| 5 | Integrate with existing pipeline | Seamless AQI calculation |

### 1.2 Secondary Objectives

- Document methodology for future maintenance
- Validate predictions against EAD ground stations
- Create reproducible workflow for monthly operations

---

## 2. Project Scope

### 2.1 In Scope

| Item | Description |
|------|-------------|
| Data Download | 2024 full year satellite + meteorological data |
| Data Processing | Run existing pipeline for 2024 training data |
| ML Development | Train Random Forest models for PM2.5 and PM10 |
| Prediction | Generate PM estimates for December 2025 |
| Integration | Connect ML output to existing AQI calculation |
| Validation | Compare with ground station measurements |
| Documentation | Technical report and user guide |

### 2.2 Out of Scope

| Item | Reason |
|------|--------|
| Changes to NO₂, SO₂, CO, O₃ processing | Already working with S5P (no lag) |
| Real-time dashboard development | Separate project |
| Ground station data collection | Provided by EAD |
| Hardware procurement | Using existing infrastructure |

---

## 3. System Architecture

### 3.1 Current Pipeline (Before)

```
┌─────────────────────────────────────────────────────────────────┐
│                     CURRENT AQI PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Sentinel-5P ──→ NO₂, SO₂, CO, O₃ ──→  ┐                        │
│                     (3-6 hrs lag)      │                        │
│                                        ├──→ AQI Calculation     │
│  CAMS Reanalysis ──→ PM2.5, PM10 ────→ ┘                        │
│                     (2-5 MONTHS LAG) ❌                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Enhanced Pipeline (After)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENHANCED AQI PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Sentinel-5P ──→ NO₂, SO₂, CO, O₃ ──→  ┐                        │
│                     (3-6 hrs lag)      │                        │
│                                        ├──→ AQI Calculation     │
│  MODIS AOD ─┐                          │                        │
│             ├──→ ML Model ──→ PM2.5 ─→ ┤                        │
│  ERA5 ──────┘    (Random    PM10 ────→ ┘                        │
│                   Forest)                                       │
│                  (1 DAY LAG) ✅                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow

```
Training Phase (One-time):
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ MODIS AOD    │    │ ERA5 Weather │    │ Ground       │
│ (2024)       │    │ (2024)       │    │ Stations     │
└──────┬───────┘    └──────┬───────┘    │ (2024)       │
       │                   │            └──────┬───────┘
       └─────────┬─────────┘                   │
                 │                             │
                 ▼                             ▼
       ┌─────────────────┐           ┌─────────────────┐
       │ Feature Matrix  │           │ Target Values   │
       │ (X: 12 features)│           │ (Y: PM2.5,PM10) │
       └────────┬────────┘           └────────┬────────┘
                │                             │
                └─────────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Random Forest   │
                    │ Training        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Trained Models  │
                    │ PM25_model.pkl  │
                    │ PM10_model.pkl  │
                    └─────────────────┘

Operational Phase (Daily/Monthly):
┌──────────────┐    ┌──────────────┐
│ MODIS AOD    │    │ ERA5 Weather │
│ (Dec 2025)   │    │ (Dec 2025)   │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
       ┌─────────────────┐
       │ Trained Models  │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ PM2.5, PM10     │
       │ Predictions     │
       │ (Dec 2025)      │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ AQI Calculation │
       │ (All 6 pollut.) │
       └─────────────────┘
```

### 3.4 Integration Strategy: Zarr Compatibility (Drop-in Replacement)

**Key Design Decision:** ML predictions will be saved in the **exact same zarr format** as CAMS files, enabling seamless integration with zero changes to the existing pipeline.

#### Why Zarr Format?

The existing pipeline uses zarr files for all pollutant data:
- `S5P_NO2_ugm3.zarr`
- `S5P_SO2_ugm3.zarr`
- `S5P_CO_ugm3.zarr`
- `S5P_O3_ugm3.zarr`
- `CAMS_PM2P5_ugm3.zarr` ← **Will be replaced by ML**
- `CAMS_PM10_ugm3.zarr` ← **Will be replaced by ML**

#### ML Output Format (Identical to CAMS)

```
ML_PM2P5_ugm3.zarr
├── PM2P5 (time, y, x)     # Variable name matches existing
├── time                    # datetime64 coordinates
├── y                       # Same y coordinates as other zarr files
├── x                       # Same x coordinates as other zarr files
└── spatial_ref            # CRS: EPSG:3857

ML_PM10_ugm3.zarr
├── PM10 (time, y, x)
├── time
├── y
├── x
└── spatial_ref            # CRS: EPSG:3857
```

#### Zarr Structure Comparison

| Attribute | CAMS Zarr (Current) | ML Zarr (New) | Compatible? |
|-----------|---------------------|---------------|-------------|
| **Dimensions** | (time, y, x) | (time, y, x) | ✅ Yes |
| **CRS** | EPSG:3857 | EPSG:3857 | ✅ Yes |
| **Resolution** | ~1 km (resampled) | ~1 km | ✅ Yes |
| **Variable name** | PM2P5 / PM10 | PM2P5 / PM10 | ✅ Yes |
| **Units** | µg/m³ | µg/m³ | ✅ Yes |
| **Time format** | datetime64[ns] | datetime64[ns] | ✅ Yes |
| **Chunking** | Auto | Auto | ✅ Yes |

#### Pipeline Integration (No Code Changes Required)

The existing `load_data()` function in `functions.ipynb` searches for pollutant names:

```python
# Existing code (NO CHANGES NEEDED)
pollutant_zarrs = list(filter(lambda x: pollutant in str(x), zarr_files))
```

This means:
- When searching for `PM2P5`, it will find `ML_PM2P5_ugm3.zarr`
- When searching for `PM10`, it will find `ML_PM10_ugm3.zarr`

#### Before vs After (File Replacement)

```
BEFORE (CAMS - Not Available):
data/processed/
├── S5P_NO2_ugm3.zarr        ✅ Available
├── S5P_SO2_ugm3.zarr        ✅ Available
├── S5P_CO_ugm3.zarr         ✅ Available
├── S5P_O3_ugm3.zarr         ✅ Available
├── CAMS_PM2P5_ugm3.zarr     ❌ NOT AVAILABLE (2-5 month lag)
└── CAMS_PM10_ugm3.zarr      ❌ NOT AVAILABLE

AFTER (ML Replacement):
data/processed/
├── S5P_NO2_ugm3.zarr        ✅ Available (unchanged)
├── S5P_SO2_ugm3.zarr        ✅ Available (unchanged)
├── S5P_CO_ugm3.zarr         ✅ Available (unchanged)
├── S5P_O3_ugm3.zarr         ✅ Available (unchanged)
├── ML_PM2P5_ugm3.zarr       ✅ ML Prediction (NEW)
└── ML_PM10_ugm3.zarr        ✅ ML Prediction (NEW)
```

#### Code to Save ML Predictions as Zarr

```python
import xarray as xr

# After ML prediction, create xarray Dataset
pm25_ds = xr.Dataset({
    'PM2P5': (['time', 'y', 'x'], pm25_predicted_values)
}, coords={
    'time': time_coordinates,
    'y': y_coordinates,
    'x': x_coordinates
})

# Set CRS to match existing pipeline
pm25_ds = pm25_ds.rio.write_crs("EPSG:3857")

# Save as zarr (same format as CAMS would have)
pm25_ds.to_zarr('data/processed/ML_PM2P5_ugm3.zarr', mode='w')

# Same for PM10
pm10_ds = xr.Dataset({
    'PM10': (['time', 'y', 'x'], pm10_predicted_values)
}, coords={
    'time': time_coordinates,
    'y': y_coordinates,
    'x': x_coordinates
})
pm10_ds = pm10_ds.rio.write_crs("EPSG:3857")
pm10_ds.to_zarr('data/processed/ML_PM10_ugm3.zarr', mode='w')
```

#### Benefits of This Approach

| Benefit | Description |
|---------|-------------|
| **Zero Pipeline Changes** | Existing notebooks (03, 04, 05) work without modification |
| **Easy Rollback** | Can switch back to CAMS when available by changing file names |
| **Consistent Format** | All pollutants use same zarr structure |
| **Parallel Development** | ML module developed independently |
| **Future-Proof** | Can upgrade ML model without pipeline changes |

#### Summary

| Question | Answer |
|----------|--------|
| Does ML output match CAMS format? | ✅ Yes (identical zarr structure) |
| Are pipeline changes needed? | ❌ No (drop-in replacement) |
| Does `load_data()` work? | ✅ Yes (searches by pollutant name) |
| Does normalization work? | ✅ Yes (same data structure) |
| Does AQI calculation work? | ✅ Yes (same variable names) |

**The ML module is designed as a "drop-in replacement" for CAMS - the rest of the pipeline remains 100% unchanged.**

---

## 4. Project Timeline

### 4.1 High-Level Schedule (6 Weeks)

```
Week 1  ████████████████████████  Data Download (2024)
Week 2  ████████████████████████  Data Processing (Pipeline)
Week 3  ████████████████████████  ML Model Development
Week 4  ████████████████████████  Dec 2025 Processing & Integration
Week 5  ████████████████████████  Validation & Testing
Week 6  ████████████████████████  Documentation & Buffer
        ─────────────────────────────────────────────────────────
        Mon  Tue  Wed  Thu  Fri
```

### 4.2 Detailed Gantt Chart

```
WEEK 1: DATA DOWNLOAD (2024 Training Data)
├── Mon │████████│ S5P NO₂ download (365 days)
├── Tue │████████│ S5P SO₂, CO download
├── Wed │████████│ S5P O₃ download
├── Thu │████████│ MODIS AOD download (GEE)
└── Fri │████████│ ERA5 meteorology download (CDS)

WEEK 2: DATA PROCESSING (Run Existing Pipeline)
├── Mon │████████│ Organize data, run 01_data_download
├── Tue │████████│ Run 02_data_preparation (S5P)
├── Wed │████████│ Run 02_data_preparation (MODIS)
├── Thu │████████│ Run 03_data_norm (normalize)
└── Fri │████████│ Prepare ground station data (filter 2024)

WEEK 3: ML MODEL DEVELOPMENT
├── Mon │████████│ Extract satellite values at stations
├── Tue │████████│ Feature engineering, data cleaning
├── Wed │████████│ Train PM2.5 Random Forest
├── Thu │████████│ Train PM10 Random Forest
└── Fri │████████│ Model validation, performance metrics

WEEK 4: DECEMBER 2025 PROCESSING
├── Mon │████████│ Verify/download Dec 2025 S5P data
├── Tue │████████│ Download ERA5 Dec 2025
├── Wed │████████│ Run pipeline (NO₂, SO₂, CO, O₃)
├── Thu │████████│ Generate ML predictions (PM2.5, PM10)
└── Fri │████████│ Integrate, calculate AQI

WEEK 5: VALIDATION & TESTING
├── Mon │████████│ Compare with ground stations
├── Tue │████████│ Debug issues, fix errors
├── Wed │████████│ Re-run if needed, fine-tune
├── Thu │████████│ Regional validation statistics
└── Fri │████████│ Export final outputs (GeoTIFF, CSV)

WEEK 6: DOCUMENTATION & BUFFER
├── Mon │████████│ Technical documentation
├── Tue │████████│ User guide / methodology
├── Wed │████████│ Disclaimer & metadata
├── Thu │████████│ Final review & cleanup
└── Fri │████████│ DELIVERY
```

---

## 5. Detailed Work Breakdown Structure (WBS)

### 5.1 Week 1: Data Download

| ID | Task | Duration | Dependencies | Output |
|----|------|----------|--------------|--------|
| 1.1 | Download S5P NO₂ (2024, 365 days) | 6-8 hrs | GEE access | `S5P_NO2_2024/` |
| 1.2 | Download S5P SO₂ (2024) | 4-6 hrs | 1.1 complete | `S5P_SO2_2024/` |
| 1.3 | Download S5P CO (2024) | 4-6 hrs | GEE access | `S5P_CO_2024/` |
| 1.4 | Download S5P O₃ (2024) | 4-6 hrs | GEE access | `S5P_O3_2024/` |
| 1.5 | Download MODIS AOD (2024) | 6-8 hrs | GEE access | `MODIS_AOD_2024/` |
| 1.6 | Download ERA5 meteorology (2024) | 6-8 hrs | CDS API | `ERA5_2024.nc` |
| 1.7 | Verify data completeness | 2 hrs | 1.1-1.6 | Data checklist |

**Week 1 Deliverables:**
- [ ] S5P data for all 4 gas pollutants (2024)
- [ ] MODIS AOD data (2024)
- [ ] ERA5 meteorological data (2024)
- [ ] Data completeness report

---

### 5.2 Week 2: Data Processing

| ID | Task | Duration | Dependencies | Output |
|----|------|----------|--------------|--------|
| 2.1 | Configure pipeline for 2024 dates | 1 hr | Week 1 | Updated `00_config.ipynb` |
| 2.2 | Run `01_data_download` (organize) | 2-4 hrs | 2.1 | Organized folder structure |
| 2.3 | Run `02_data_preparation` (S5P) | 6-8 hrs | 2.2 | S5P zarr files |
| 2.4 | Run `02_data_preparation` (MODIS) | 4-6 hrs | 2.2 | MODIS zarr files |
| 2.5 | Run `03_data_norm` (normalize) | 4-6 hrs | 2.3, 2.4 | Normalized zarr |
| 2.6 | Load ground station data | 2 hrs | EAD data | Filtered 2024 CSV |
| 2.7 | Quality check processed data | 2 hrs | 2.3-2.6 | QC report |

**Week 2 Deliverables:**
- [ ] Processed S5P zarr files (NO₂, SO₂, CO, O₃)
- [ ] Processed MODIS AOD zarr files
- [ ] Normalized datasets
- [ ] 2024 ground station data (filtered)

---

### 5.3 Week 3: ML Model Development

| ID | Task | Duration | Dependencies | Output |
|----|------|----------|--------------|--------|
| 3.1 | Create spatial extraction script | 3-4 hrs | Week 2 | `extract_at_stations.py` |
| 3.2 | Extract MODIS AOD at 20 stations | 2-3 hrs | 3.1 | AOD time series |
| 3.3 | Extract ERA5 at 20 stations | 2-3 hrs | 3.1 | Weather time series |
| 3.4 | Extract S5P NO₂ at 20 stations | 2-3 hrs | 3.1 | NO₂ time series |
| 3.5 | Merge features with ground truth | 3-4 hrs | 3.2-3.4 | Training DataFrame |
| 3.6 | Feature engineering | 4-6 hrs | 3.5 | Final feature matrix |
| 3.7 | Train/test split (70/15/15) | 1 hr | 3.6 | Split datasets |
| 3.8 | Train PM2.5 Random Forest | 2-3 hrs | 3.7 | `PM25_RF_model.pkl` |
| 3.9 | Train PM10 Random Forest | 2-3 hrs | 3.7 | `PM10_RF_model.pkl` |
| 3.10 | Hyperparameter tuning (optional) | 4-6 hrs | 3.8, 3.9 | Optimized models |
| 3.11 | Model validation | 3-4 hrs | 3.8, 3.9 | Validation metrics |
| 3.12 | Generate validation plots | 2-3 hrs | 3.11 | Scatter plots, time series |

**Week 3 Deliverables:**
- [ ] Training dataset (~120,000 samples)
- [ ] Trained PM2.5 model (R² > 0.75)
- [ ] Trained PM10 model (R² > 0.73)
- [ ] Validation report with metrics
- [ ] Feature importance analysis

---

### 5.4 Week 4: December 2025 Processing

| ID | Task | Duration | Dependencies | Output |
|----|------|----------|--------------|--------|
| 4.1 | Verify Dec 2025 S5P data | 2 hrs | Existing data | Data checklist |
| 4.2 | Download missing S5P (if any) | 2-4 hrs | 4.1 | Complete S5P Dec 2025 |
| 4.3 | Download ERA5 Dec 2025 | 2-3 hrs | CDS API | `ERA5_Dec2025.nc` |
| 4.4 | Verify MODIS AOD Dec 2025 | 1 hr | Existing data | Data checklist |
| 4.5 | Run pipeline for Dec 2025 (S5P) | 4-6 hrs | 4.2 | S5P zarr (Dec 2025) |
| 4.6 | Prepare prediction features | 3-4 hrs | 4.3, 4.4 | Feature grid (Dec 2025) |
| 4.7 | Generate PM2.5 predictions | 2-3 hrs | 4.6, Week 3 | `ML_PM2P5.zarr` |
| 4.8 | Generate PM10 predictions | 2-3 hrs | 4.6, Week 3 | `ML_PM10.zarr` |
| 4.9 | Integrate into pipeline | 3-4 hrs | 4.7, 4.8 | Modified `functions.ipynb` |
| 4.10 | Calculate AQI (all 6 pollutants) | 3-4 hrs | 4.5, 4.7, 4.8 | Dec 2025 AQI |

**Week 4 Deliverables:**
- [ ] December 2025 S5P data (NO₂, SO₂, CO, O₃)
- [ ] December 2025 ERA5 data
- [ ] ML predictions: PM2.5 and PM10 (Dec 2025)
- [ ] Complete AQI for December 2025

---

### 5.5 Week 5: Validation & Testing

| ID | Task | Duration | Dependencies | Output |
|----|------|----------|--------------|--------|
| 5.1 | Extract predictions at stations | 2-3 hrs | Week 4 | Prediction vs observed |
| 5.2 | Calculate validation metrics | 2-3 hrs | 5.1 | R², RMSE, MAE, bias |
| 5.3 | Generate comparison plots | 3-4 hrs | 5.2 | Scatter, time series plots |
| 5.4 | Identify systematic errors | 2-3 hrs | 5.2, 5.3 | Error analysis |
| 5.5 | Debug and fix issues | 6-8 hrs | 5.4 | Bug fixes |
| 5.6 | Re-run pipeline if needed | 4-6 hrs | 5.5 | Updated outputs |
| 5.7 | Regional statistics by emirate | 3-4 hrs | 5.6 | Regional report |
| 5.8 | Export GeoTIFF files | 2-3 hrs | 5.6 | `*.tif` files |
| 5.9 | Export CSV statistics | 2-3 hrs | 5.6 | `*_Region_Stats.csv` |

**Week 5 Deliverables:**
- [ ] Validation report with accuracy metrics
- [ ] Comparison plots (satellite vs ground)
- [ ] Final GeoTIFF outputs
- [ ] Regional statistics CSV files

---

### 5.6 Week 6: Documentation & Buffer

| ID | Task | Duration | Dependencies | Output |
|----|------|----------|--------------|--------|
| 6.1 | Write technical methodology | 4-6 hrs | All weeks | Technical report |
| 6.2 | Create user guide | 3-4 hrs | Week 4-5 | User documentation |
| 6.3 | Write data disclaimer | 2-3 hrs | Week 5 | Disclaimer document |
| 6.4 | Document ML model details | 2-3 hrs | Week 3 | Model documentation |
| 6.5 | Create operational workflow | 2-3 hrs | Week 4 | SOP document |
| 6.6 | Code cleanup and comments | 3-4 hrs | All code | Clean codebase |
| 6.7 | Final review | 2-3 hrs | 6.1-6.6 | Review checklist |
| 6.8 | Package deliverables | 2-3 hrs | 6.7 | Delivery package |
| 6.9 | **Project Delivery** | - | 6.8 | **COMPLETE** |

**Week 6 Deliverables:**
- [ ] Technical methodology report
- [ ] User guide / operational manual
- [ ] Data disclaimer document
- [ ] Complete code with documentation
- [ ] Final delivery package

---

## 6. Deliverables

### 6.1 Data Products

| Deliverable | Format | Description |
|-------------|--------|-------------|
| December 2025 AQI | GeoTIFF, Zarr | Daily AQI maps (all 6 pollutants) |
| PM2.5 predictions | GeoTIFF, Zarr | ML-estimated PM2.5 concentrations |
| PM10 predictions | GeoTIFF, Zarr | ML-estimated PM10 concentrations |
| Regional statistics | CSV | AQI by region/emirate |
| Validation comparison | CSV | Satellite vs ground station comparison |

### 6.2 Models & Code

| Deliverable | Format | Description |
|-------------|--------|-------------|
| PM2.5 trained model | `.pkl` | Random Forest model for PM2.5 |
| PM10 trained model | `.pkl` | Random Forest model for PM10 |
| Training notebook | `.ipynb` | `00b_train_ml_model.ipynb` |
| Prediction notebook | `.ipynb` | `00c_predict_pm.ipynb` |
| Modified pipeline | `.ipynb` | Updated `functions.ipynb` |

### 6.3 Documentation

| Deliverable | Format | Description |
|-------------|--------|-------------|
| Technical report | PDF/MD | Methodology, validation, results |
| User guide | PDF/MD | How to run the pipeline |
| Data disclaimer | PDF/MD | Satellite vs ground station differences |
| ML implementation plan | PDF/MD | Scientific foundation (existing) |

---

## 7. Resource Requirements

### 7.1 Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 16 GB | 32 GB |
| **CPU** | 4 cores | 8 cores |
| **Storage** | 100 GB | 200 GB |
| **Internet** | 10 Mbps | 50+ Mbps |

### 7.2 Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Core programming |
| Jupyter | Latest | Notebook execution |
| scikit-learn | 1.0+ | ML model training |
| xarray | 0.19+ | Geospatial data handling |
| rioxarray | 0.7+ | Raster I/O |
| earthengine-api | 0.1.300+ | GEE data access |
| cdsapi | 0.5+ | ERA5 download |

### 7.3 Data Access

| Service | Access Required | Status |
|---------|-----------------|--------|
| Google Earth Engine | Authenticated account | ☐ Verify |
| CDS (Copernicus) | API key configured | ☐ Verify |
| EAD Ground Data | Data file provided | ☐ Verify |

### 7.4 Human Resources

| Role | Effort | Responsibility |
|------|--------|----------------|
| Data Scientist/Developer | 100% (6 weeks) | Implementation |
| Project Manager | 10% | Oversight, reporting |
| Domain Expert (optional) | As needed | Technical guidance |

---

## 8. Risk Assessment

### 8.1 Risk Matrix

| Risk | Probability | Impact | Severity | Mitigation |
|------|-------------|--------|----------|------------|
| Data download delays (CDS queue) | Medium | Medium | **Medium** | Run downloads overnight, parallel requests |
| GEE quota exceeded | Low | Medium | **Low** | Split requests, multiple days |
| ML model underperforms | Medium | High | **High** | Feature engineering, ensemble methods |
| Data gaps in satellite coverage | Medium | Medium | **Medium** | Temporal interpolation, climatology |
| Pipeline integration issues | Medium | Medium | **Medium** | Week 5 buffer for debugging |
| Ground station data quality | Low | High | **Medium** | Data cleaning, outlier removal |
| Computational resource limits | Low | Medium | **Low** | Cloud computing if needed |

### 8.2 Contingency Plans

| Scenario | Trigger | Action |
|----------|---------|--------|
| Download takes >2 days | CDS queue >24 hrs | Use parallel requests, split by month |
| R² < 0.70 | Validation results | Add features, try XGBoost/ensemble |
| Missing >30% AOD data | Cloud coverage | Use S5P AAI as backup, interpolation |
| Integration fails | AQI calculation errors | Debug, create standalone script |

---

## 9. Success Criteria

### 9.1 Technical Success

| Criterion | Target | Minimum Acceptable |
|-----------|--------|-------------------|
| PM2.5 R² | > 0.80 | > 0.70 |
| PM2.5 RMSE | < 12 µg/m³ | < 18 µg/m³ |
| PM10 R² | > 0.78 | > 0.70 |
| PM10 RMSE | < 20 µg/m³ | < 28 µg/m³ |
| Data lag | 1 day | ≤ 3 days |
| Coverage | 100% Abu Dhabi | > 95% |

### 9.2 Project Success

| Criterion | Target |
|-----------|--------|
| On-time delivery | Within 6 weeks |
| December 2025 AQI generated | Complete for all days |
| Pipeline integration | Seamless with existing code |
| Documentation complete | All deliverables provided |
| Validation report | Comprehensive with metrics |

---

## 10. Communication Plan

### 10.1 Status Reporting

| Report | Frequency | Audience | Content |
|--------|-----------|----------|---------|
| Daily standup | Daily | Team | Progress, blockers |
| Weekly summary | Weekly | Manager | Milestone status, risks |
| Final report | End of project | All stakeholders | Complete results |

### 10.2 Milestone Reviews

| Milestone | Week | Review Items |
|-----------|------|--------------|
| Data download complete | End Week 1 | Data completeness, quality |
| Processing complete | End Week 2 | Zarr files, data integrity |
| Models trained | End Week 3 | Performance metrics, validation |
| AQI generated | End Week 4 | December 2025 outputs |
| Validation complete | End Week 5 | Accuracy assessment |
| Project complete | End Week 6 | All deliverables |

---

## 11. Assumptions & Dependencies

### 11.1 Assumptions

1. Ground station data (EAD) for 2024 is complete and quality-controlled
2. GEE and CDS API access is available and functioning
3. Existing pipeline code is stable and documented
4. December 2025 satellite data (S5P, MODIS) is already downloaded
5. Computational resources are sufficient (16 GB RAM, 8 cores)
6. No major changes to UAE EAQI breakpoints during project

### 11.2 Dependencies

| Dependency | Type | Impact if Unavailable |
|------------|------|----------------------|
| EAD ground station data | Data | Cannot train ML models |
| GEE access | Service | Cannot download satellite data |
| CDS API access | Service | Cannot download ERA5 |
| Existing pipeline | Code | Cannot integrate predictions |
| Python environment | Software | Cannot execute code |

---

## 12. Approval

### 12.1 Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Lead | | | |
| Technical Lead | | | |
| Manager | | | |

### 12.2 Change Control

Any changes to scope, timeline, or deliverables must be approved by the project manager and documented in a change request form.

---

## Appendix A: Technical Specifications

### A.1 ML Model Parameters

```python
# PM2.5 Random Forest
PM25_MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# PM10 Random Forest
PM10_MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}
```

### A.2 Feature List (12 Features)

| # | Feature | Source | Unit |
|---|---------|--------|------|
| 1 | AOD_550nm | MODIS MCD19A2 | dimensionless |
| 2 | NO2_column | Sentinel-5P | mol/m² |
| 3 | Temperature_2m | ERA5 | °C |
| 4 | Relative_humidity | ERA5 (derived) | % |
| 5 | Wind_speed | ERA5 (derived) | m/s |
| 6 | Wind_direction | ERA5 (derived) | degrees |
| 7 | Surface_pressure | ERA5 | hPa |
| 8 | Boundary_layer_height | ERA5 | m |
| 9 | Hour_of_day | Timestamp | 0-23 |
| 10 | Day_of_year | Timestamp | 1-365 |
| 11 | Month | Timestamp | 1-12 |
| 12 | Is_weekend | Timestamp | 0/1 |

### A.3 Data Sources

| Source | Product | Resolution | Access |
|--------|---------|------------|--------|
| Sentinel-5P | TROPOMI L2 | 7 × 3.5 km | GEE |
| MODIS | MCD19A2 v061 | 1 km | GEE |
| ERA5 | Reanalysis | 0.25° (~25 km) | CDS |
| EAD | Ground stations | Point | CSV |

---

## Appendix B: File Structure

```
aqi_pipeline_2026-01-08/
├── 00_config.ipynb                    # Configuration
├── 01_data_download_dynamic.ipynb     # Data download
├── 02_data_preparation_dynamic.ipynb  # Data processing
├── 03_data_norm_dynamic.ipynb         # Normalization
├── 04_mean_dynamic.ipynb              # Temporal aggregation
├── 05_Air Quality Index.ipynb         # AQI calculation
├── functions.ipynb                    # Utility functions
│
├── ML_PM_Prediction/                  # NEW: ML Module
│   ├── 00a_download_training_data.ipynb
│   ├── 00b_train_ml_model.ipynb
│   ├── 00c_predict_pm.ipynb
│   ├── models/
│   │   ├── PM25_RF_model.pkl
│   │   └── PM10_RF_model.pkl
│   └── training_data/
│       └── training_dataset_2024.csv
│
├── data/
│   ├── raw/                           # Downloaded data
│   ├── processed/                     # Zarr files
│   └── output/                        # Final products
│
├── docs/
│   ├── ML_PM_Prediction_Implementation_Plan.md
│   ├── Project_Plan_AQI_ML_Implementation.md
│   ├── Technical_Report.md
│   └── Data_Disclaimer.md
│
└── validation/
    ├── validation_report.md
    └── plots/
```

---

## Appendix C: Contact Information

| Role | Name | Email | Phone |
|------|------|-------|-------|
| Project Lead | | | |
| Technical Support | | | |
| Data Provider (EAD) | | | |

---

**Document Control:**
- Version: 1.0
- Created: January 2026
- Last Updated: January 2026
- Status: Draft / Approved

---

*End of Project Plan*