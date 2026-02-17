# Phase 1: Research & Foundation

## Objectives

1. Understand AQI calculation methodology
2. Identify satellite data sources
3. Study existing approaches for satellite-based AQI estimation
4. Document technical approach

---

## 1. AQI Methodology Research

### Required Pollutants

Research and document the following pollutants:

| Pollutant | Full Name | Primary Sources | Health Impact |
|-----------|-----------|-----------------|---------------|
| PM2.5 | Particulate Matter < 2.5μm | Combustion, fires, industrial | Respiratory, cardiovascular |
| PM10 | Particulate Matter < 10μm | Dust, construction, pollen | Respiratory irritation |
| NO₂ | Nitrogen Dioxide | Vehicle emissions, power plants | Respiratory problems |
| SO₂ | Sulfur Dioxide | Fossil fuel burning, volcanoes | Respiratory issues, acid rain |
| CO | Carbon Monoxide | Incomplete combustion, vehicles | Reduces oxygen delivery |
| O₃ | Ozone | Secondary pollutant (sunlight + NOx + VOCs) | Respiratory damage |

### AQI Calculation

**Standard AQI Formula** (EPA):

```
AQI = [(I_high - I_low) / (BP_high - BP_low)] × (C - BP_low) + I_low
```

Where:
- `C` = Pollutant concentration
- `BP` = Breakpoint (concentration range)
- `I` = AQI index range

### AQI Categories

| AQI Range | Category | Color | Health Impact |
|-----------|----------|-------|---------------|
| 0-50 | Good | Green | Minimal |
| 51-100 | Moderate | Yellow | Acceptable |
| 101-150 | Unhealthy for Sensitive Groups | Orange | Some concern |
| 151-200 | Unhealthy | Red | Everyone affected |
| 201-300 | Very Unhealthy | Purple | Serious effects |
| 301+ | Hazardous | Maroon | Emergency conditions |

### Tasks

- [ ] Document AQI breakpoint tables for each pollutant
- [ ] Create `aqi_calculator.py` utility function
- [ ] Validate against official calculators

---

## 2. Satellite Data Sources

### Sentinel-5P (TROPOMI)

**Provides**: NO₂, SO₂, CO, O₃, CH₄, HCHO

- **Satellite**: Sentinel-5 Precursor
- **Instrument**: TROPOMI
- **Temporal Resolution**: Daily overpass
- **Spatial Resolution**: 7 × 3.5 km (very high for atmospheric monitoring)
- **Latency**: 3-5 hours (near real-time)

**Access**:
- Portal: https://dataspace.copernicus.eu/
- API: `sentinelsat` Python library
- Data format: NetCDF

**Advantages**:
✓ High spatial resolution
✓ Daily coverage
✓ Free and open access
✓ Multiple pollutants

**Limitations**:
✗ Cloud interference
✗ No direct PM2.5/PM10

---

### MODIS (Terra/Aqua)

**Provides**: Aerosol Optical Depth (AOD) → PM2.5/PM10 estimation

- **Satellites**: Terra (morning), Aqua (afternoon)
- **Temporal Resolution**: 1-2 days
- **Spatial Resolution**: 250m - 1km (depends on product)
- **Latency**: Near real-time to 48 hours

**Access**:
- Portal: https://earthdata.nasa.gov/
- API: NASA Earthdata API
- Products: MOD04 (Terra), MYD04 (Aqua)

**PM Estimation**:
Use AOD + meteorological data → Statistical/ML model → PM2.5

---

### CAMS (Copernicus Atmosphere Monitoring Service)

**Provides**: Near real-time atmospheric composition analysis

- Combines satellite observations with models
- Forecast data available
- Lower spatial resolution but higher temporal

**Access**: https://ads.atmosphere.copernicus.eu/

---

### Ground Truth Data

For training and validation:

1. **OpenAQ**: https://openaq.org/
   - Global air quality data from monitoring stations
   - API available
   - Free and open

2. **EPA AirNow** (US): https://www.airnow.gov/
   - Real-time and historical data
   - API available

3. **Local Environmental Agencies**
   - Pakistan EPA, India CPCB, etc.

---

## 3. Literature Review

### Key Research Areas

1. **Deep Learning for Atmospheric Remote Sensing**
   - CNNs for satellite image analysis
   - Attention mechanisms for multi-spectral data
   - Transfer learning approaches

2. **PM2.5 Estimation from AOD**
   - AOD-PM2.5 relationship
   - Meteorological corrections
   - Spatiotemporal models

3. **Multi-Satellite Fusion**
   - Combining multiple data sources
   - Temporal alignment
   - Data quality handling

### Papers to Review

- "Deep Learning for Air Quality Forecasting with Satellite Imagery"
- "Estimating PM2.5 from Satellite AOD using Deep Neural Networks"
- "Real-time Air Quality Monitoring using Multi-Source Satellite Data"
- "Attention-based Deep Learning for Atmospheric Pollutant Estimation"

### Tasks

- [ ] Create literature review document
- [ ] Identify best practices and methodologies
- [ ] Document potential challenges and solutions

---

## 4. Technical Approach Document

Based on research, document:

### Data Flow

```
Satellite Data → Download → Preprocessing → Feature Extraction → 
    DL Model → Pollutant Predictions → AQI Calculation → Output
```

### Model Architecture (Proposed)

1. **Multi-Input CNN**
   - Input: Multi-channel satellite imagery
   - Backbone: Pre-trained ResNet/EfficientNet
   - Fusion layer: Combine spectral + spatial features
   - Output: Pollutant concentrations

2. **Alternative: U-Net**
   - For spatial AQI mapping
   - Pixel-wise predictions

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Cloud cover | Multi-day averaging, quality flags |
| Data latency | Use CAMS for near real-time |
| Limited ground truth | Transfer learning, synthetic data |
| PM not directly measured | Use AOD + meteorological model |

---

## Deliverables

By end of Phase 1:

- ✓ `docs/aqi_methodology.md` - Complete AQI calculation guide
- ✓ `docs/pollutants_reference.md` - Pollutant specifications
- ✓ `docs/data_sources.md` - Comparison of satellite sources
- ✓ `docs/literature_review.md` - Research summary
- ✓ `docs/technical_approach.md` - Proposed methodology
- ✓ Sample satellite datasets downloaded

---

## Estimated Timeline

**2 weeks**

- Week 1: AQI research + Data source exploration
- Week 2: Literature review + Technical approach documentation
