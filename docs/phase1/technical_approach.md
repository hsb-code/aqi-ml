# Technical Approach - Real-time Satellite-based AQI Monitoring System

*Proposed methodology based on Phase 1 research*

---

## Project Goal

Build a deep learning system that calculates Air Quality Index (AQI) from real-time satellite imagery, enabling air quality monitoring in areas without ground monitoring stations, with specific focus on detecting pollution from fire incidents.

---

## System Architecture Overview

### Layer 1: Data Acquisition
Download satellite and ground truth data:
- **Sentinel-5P** (NO₂, SO₂, CO, O₃) → Downloaded via Copernicus
- **MODIS** (AOD → PM2.5, PM10) → Downloaded via NASA
- **ERA5 Meteorological Data** → Downloaded via ECMWF
- **OpenAQ Ground Truth** → API access for validation

### Layer 2: Preprocessing
Process and clean the data:
- Cloud masking and quality filtering
- Geospatial reprojection and cropping
- Gap-filling for missing data
- Normalization and scaling
- Feature extraction (satellite + meteorological)

### Layer 3: Model
Machine learning predictions:
- **Phase 1 (MVP)**: Random Forest for NO₂ prediction
- **Phase 2**: CNN-LSTM for PM2.5 from AOD
- **Phase 3**: Multi-pollutant fusion model

### Layer 4: AQI Calculation
Calculate final air quality index:
- Calculate sub-index for each pollutant
- Apply EPA AQI formula with breakpoints
- Select maximum AQI as overall index
- Generate confidence scores

### Layer 5: API
Serve predictions via REST API:
- **POST /predict** - Real-time AQI for location
- **GET /history** - Historical data retrieval
- **GET /status** - System health check

---

## Data Sources (Final Selection)

### Primary Sources

| Data | Source | Pollutants | Resolution | Update Frequency | Purpose |
|------|--------|-----------|------------|------------------|---------|
| **Gas pollutants** | Sentinel-5P TROPOMI | NO₂, SO₂, CO, O₃ | 3.5×5.5 km | Daily | Direct measurements |
| **Aerosol data** | MODIS MAIAC | AOD (→PM2.5/PM10) | 1 km | Twice daily | Particulate matter |
| **Meteorology** | ERA5 Reanalysis | Temp, Wind, Humidity, BLH | 0.25° (~25km) | Hourly | Model inputs |
| **Ground truth** | OpenAQ | All 6 pollutants | Point data | Varies | Training/validation |

### Backup/Supplementary
- **CAMS**: Gap-filling when satellite data unavailable (clouds)
- **EPA AirNow**: High-quality US validation data

---

## Phased Development Approach

### Phase 1: MVP - NO₂ Prediction (Weeks 1-4)

**Goal**: Prove concept with single pollutant

**Data**:
- Sentinel-5P NO₂ columns
- ERA5 meteorological data
- OpenAQ ground truth

**Model**:
- **Random Forest Regressor**
- Features: NO₂ column, temperature, wind speed/direction, boundary layer height, land use, time
- Target: Ground-level NO₂ concentration
- Why RF: Simple, fast, proven effective (R² = 0.75-0.80 expected)

**Pipeline**:
1. Download 1 month of Sentinel-5P NO₂ data for test region
2. Match with OpenAQ ground stations (temporal/spatial alignment)
3. Extract meteorological features from ERA5
4. Train/validate Random Forest model
5. Test inference on new satellite overpass

**Success Criteria**:
- R² > 0.70 on validation set
- Can process satellite pass within 1 hour
- Successfully predicts NO₂ for no-station location

---

### Phase 2: PM2.5 from AOD (Weeks 5-8)

**Goal**: Add most important pollutant for health/fires

**Data**:
- MODIS MAIAC AOD
- ERA5 meteorology (expanded: relative humidity critical for PM)
- OpenAQ PM2.5 ground truth

**Model Option A** (Simple):
- Random Forest with AOD + meteorology
- Expected R² = 0.70-0.75

**Model Option B** (Advanced):
- CNN-LSTM hybrid
- CNN: Extract spatial AOD patterns
- LSTM: Temporal trends
- Expected R² = 0.80-0.90
- Use if RF insufficient

**Specific Challenges**:
- AOD-PM2.5 relationship varies with humidity
- Cloud gaps in MODIS data → implement gap-filling
- May need MODIS fire product for fire scenario

---

### Phase 3: Multi-Pollutant AQI (Weeks 9-12)

**Goal**: Complete system with all 6 pollutants

**Extensions**:
1. Add **SO₂** and **CO** from Sentinel-5P (similar to NO₂ approach)
2. Add **O₃** from Sentinel-5P (may need UV correction)
3. Add **PM10** estimation (from MODIS, similar to PM2.5)

**AQI Calculation**:
```python
def calculate_aqi(pollutants_dict):
    """
    pollutants_dict = {
        'PM2.5': value,
        'PM10': value,
        'NO2': value,
        'SO2': value,
        'CO': value,
        'O3': value
    }
    """
    aqi_values = []
    for pollutant, concentration in pollutants_dict.items():
        sub_aqi = apply_epa_formula(pollutant, concentration)
        aqi_values.append((sub_aqi, pollutant))
    
    overall_aqi, dominant_pollutant = max(aqi_values)
    return overall_aqi, dominant_pollutant
```

**Integration**:
- Combine all individual pollutant models
- Run inference for each pollutant
- Calculate AQI using EPA formula
- Return overall AQI + dominant pollutant

---

## Model Architecture Details

### Starting Point: Random Forest for NO₂

**Why Random Forest**:
✓ Simple to implement and train
✓ Fast inference (real-time capable)
✓ Robust to overfitting
✓ Handles missing features gracefully
✓ Provides feature importance
✓ Proven in literature (R² = 0.75-0.85)

**Input Features** (~15-20 features):
1. **Satellite**: NO₂ vertical column density
2. **Meteorological**: Temperature, wind (U/V), relative humidity, pressure, boundary layer height
3. **Spatial**: Latitude, longitude, elevation
4. **Temporal**: Hour of day, day of year, season
5. **Land use**: Urban/rural classification, population density
6. **Quality**: Satellite quality flags

**Hyperparameters**:
- n_estimators: 200-500 trees
- max_depth: 10-20
- min_samples_split: 10
- Cross-validation: 5-fold spatial CV

---

### If Needed: CNN-LSTM for PM2.5

**When to use**: If Random Forest R² < 0.70 for PM2.5

**Architecture**:

**Input**: Multi-day AOD images (e.g., 7 days × H × W) + Meteorological fields (Temperature, Relative Humidity, Wind, Boundary Layer Height)

**Step 1 - CNN Encoder** (ResNet-18 base):
- Extracts spatial features from AOD imagery
- Identifies spatial pollution patterns

**Step 2 - LSTM Layers** (2 layers, 128 hidden units):
- Captures temporal dependencies across multiple days
- Learns how pollution evolves over time

**Step 3 - Dense Layers** (Fully Connected → ReLU → Fully Connected):
- Regression head for final prediction
- Outputs single PM2.5 value

**Output**: Ground-level PM2.5 concentration prediction

**Training**:
- Loss: Huber loss (robust to outliers)
- Optimizer: AdamW
- Learning rate: 1e-4 with cosine decay
- Batch size: 32
- Data augmentation: Spatial flips, rotations

---

## Gap-Filling Strategy

**Problem**: Clouds cause ~40-60% missing satellite data

**Solution** (Two-stage):

**Stage 1 - Temporal Interpolation**:
- If data missing for Day N, average Day N-1 and Day N+1
- Works for short gaps (1-2 days)

**Stage 2 - Model-based Filling**:
- Use CAMS model outputs to fill longer gaps
- CAMS provides all pollutants without cloud gaps
- Calibrate CAMS against satellite data when available

**Stage 3 - Uncertainty Quantification**:
- Flag predictions as:
  - "High confidence" (satellite data available)
  - "Medium confidence" (1-day gap, interpolated)
  - "Low confidence" (>2 days gap, model-based)

---

## Fire Incident Detection (Our Use Case)

**Scenario**: Fire near office, no monitoring station nearby

**Approach**:

1. **Detect Fire**:
   - Use MODIS Active Fire product (MOD14/MYD14)
   - Identifies active fires with thermal anomaly detection
   
2. **Identify Satellite Overpass**:
   - Check Sentinel-5P pass times for fire location/date
   - Check MODIS Terra/Aqua overpass times

3. **Extract Pollutants**:
   - **Expected spikes**:
     - PM2.5: 10-50× increase (massive!)
     - CO: 5-20× increase
     - NO₂: 2-5× increase
     - SO₂: Variable (depends on fuel)

4. **Calculate AQI**:
   - Likely dominated by PM2.5 (fire produces lots of particulates)
   - AQI could reach "Very Unhealthy" (200-300) or "Hazardous" (300+)

5. **Compare Baseline**:
   - Compare with average AQI for previous week
   - Quantify pollution spike magnitude

6. **Visualize**:
   - Create before/after satellite image comparison
   - Show AQI heatmap around fire location

**Implementation**:
```python
def detect_fire_pollution(location, date):
    # 1. Check for active fire
    fire_detected = query_modis_fire(location, date)
    
    if fire_detected:
        # 2. Get satellite data
        s5p_data = get_sentinel5p(location, date)
        modis_data = get_modis_aod(location, date)
        
        # 3. Run inference
        pollutants = {
            'PM2.5': predict_pm25(modis_data),
            'NO2': predict_no2(s5p_data),
            'CO': predict_co(s5p_data),
            'SO2': predict_so2(s5p_data)
        }
        
        # 4. Calculate AQI
        aqi, dominant = calculate_aqi(pollutants)
        
        # 5. Get baseline
        baseline_aqi = get_historical_avg(location, days=7)
        
        # 6. Report
        return {
            'fire_detected': True,
            'current_aqi': aqi,
            'baseline_aqi': baseline_aqi,
            'spike': aqi - baseline_aqi,
            'dominant_pollutant': dominant,
            'pollutant_levels': pollutants
        }
```

---

## Validation Strategy

### Stage 1: With Ground Stations
- Compare predictions against OpenAQ/EPA measurements
- Metrics: R², RMSE, MAE, correlation
- Goal: Validate model accuracy where ground truth exists

### Stage 2: Cross-validation
- Spatial cross-validation (leave-out entire regions)
- Temporal cross-validation (leave-out time periods)
- Ensures model generalizes

### Stage 3: No-Station Areas
- Use multiple satellites as cross-check (Sentinel-5P vs CAMS)
- Consistency checks (physical constraints)
- Expert visual inspection of outputs

### Stage 4: Fire Scenario Testing
- Identify historical fire events from MODIS fire product
- Retroactively predict AQI for those events
- Validate against any available nearby station data
- Check if predicted spikes match expected fire pollution patterns

---

## Technical Stack

### Programming
- **Python 3.9+**
- **Core Libraries**: NumPy, Pandas, Xarray (for NetCDF data)
- **Geospatial**: Rasterio, GDAL, GeoPandas
- **ML**: Scikit-learn (Random Forest), PyTorch (if using CNN-LSTM)
- **API**: FastAPI + Uvicorn

### Data Processing
- **Satellite Access**: `sentinelsat`, NASA `earthaccess`
- **Format Handling**: NetCDF4, h5py
- **Meteorology**: `cdsapi` (for ERA5)

### Deployment
- **Containerization**: Docker
- **Orchestration**: Could use Kubernetes if scaling
- **Scheduling**: APScheduler for daily satellite downloads
- **Caching**: Redis for API responses

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Cloud cover** (40-60% missing data) | Gap-filling with temporal interpolation + CAMS |
| **Column to surface conversion** | Use Random Forest/NN with meteorological features |
| **AOD-PM2.5 varies by location** | Train separate models per region or use local features |
| **Real-time requirements** | Optimize for speed: RF faster than deep learning |
| **Model accuracy in no-station areas** | Cross-validate with multiple satellites, uncertainty quantification |
| **Fire detection specificity** | Combine MODIS fire product with pollution spikes |
| **Data storage** | Process and discard raw data, keep only predictions |
| **API latency** | Pre-compute for known locations, cache results |

---

## Success Metrics

### Model Performance
- **NO₂ model**: R² > 0.75
- **PM2.5 model**: R² > 0.70 (RF) or > 0.85 (CNN-LSTM)
- **Overall AQI**: Within ±20 AQI units of ground truth

### System Performance
- **Latency**: < 5 seconds for API response (cached)
- **Processing**: Complete satellite pass within 2 hours
- **Uptime**: > 95% availability
- **Coverage**: Daily updates for defined geographic region

### Use Case Validation
- **Fire detection**: Successfully identify >80% of fire pollution events
- **No-station accuracy**: Predictions consistent with nearby stations (within 30 km)

---

## Timeline & Next Steps

**Phase 1 Complete** ✅: Research & Documentation

**Next: Phase 2 - Data Acquisition (Weeks 1-2)**:
1. Register for Copernicus & NASA Earthdata accounts
2. Download 1 month of sample data (Sentinel-5P, MODIS, ERA5)
3. Set up data storage structure
4. Test data reading and basic preprocessing

**Then: Phase 3 - Model Development (Weeks 3-6)**:
1. Prepare training dataset (satellite + ground truth matching)
2. Train Random Forest for NO₂
3. Evaluate and tune model
4. Test on held-out data

**Future: Complete System (Weeks 7-13)**:
1. Add PM2.5 prediction
2. Add remaining pollutants
3. Build AQI calculation pipeline
4. Develop API
5. Test fire incident scenario
6. Deploy prototype

---

## Conclusion

This approach is:
- **Grounded in literature**: Uses proven techniques from recent research
- **Practical**: Starts simple (RF), scales up if needed (CNN-LSTM)
- **Real-time capable**: Designed for operational deployment
- **Use-case focused**: Fire incident detection as primary goal
- **Feasible**: Achievable with available data and resources
- **Extensible**: Can add more pollutants and features incrementally

**Ready to proceed to Phase 2: Data Acquisition!**
