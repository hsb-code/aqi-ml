# Satellite Data Sources - Research Documentation

## Overview

This document summarizes the satellite data sources available for extracting air quality pollutants needed for AQI calculation.

---

## Sentinel-5P (TROPOMI)

### What it provides:
- **NO₂** (Nitrogen Dioxide)
- **SO₂** (Sulfur Dioxide)  
- **CO** (Carbon Monoxide)
- **O₃** (Ozone)
- **CH₄** (Methane)
- **HCHO** (Formaldehyde)

### Key Specs:
- **Satellite**: Sentinel-5 Precursor (S5P)
- **Instrument**: TROPOMI (TROPOspheric Monitoring Instrument)
- **Launch Date**: October 13, 2017 (operational since)
- **Coverage**: **Near-daily** global coverage (complete coverage for latitudes > 7° and < -7°)
- **Swath Width**: 2600 km (very wide - allows daily global coverage)
- **Orbit**: Sun-synchronous, 824 km altitude, 14 orbits per day
- **Resolution**: **3.5 × 5.5 km** at nadir (upgraded from 7×7 km in August 2019)
  - Among the highest resolution atmospheric monitoring satellites
- **Latency**: **3-5 hours** (near real-time!) - excellent for our use case
- **Data Format**: NetCDF

### Access:
- **Primary Portal**: https://dataspace.copernicus.eu/ (Copernicus Data Space Ecosystem)
- **Alternative Portals**: 
  - NASA GES DISC (https://disc.gsfc.nasa.gov/)
  - WEkEO (https://www.wekeo.eu/)
- **API**: `sentinelsat` Python library  
- **Registration**: https://dataspace.copernicus.eu/ (Click "Register" button on homepage)
- **My Username**: *(will fill after registration)*
- **Registration Date**: *(pending)*

### Products Available:
- **Level 2 Products**: Individual pollutant retrievals (NO₂, SO₂, CO, O₃)
- **Level 3 Products**: Gridded, aggregated data (easier to use)
- **Product IDs**:
  - NO₂: `L2__NO2___`
  - CO: `L2__CO____`
  - SO₂: `L2__SO2___`
  - O₃: `L2__O3____`

### Advantages:
✓ **High spatial resolution** (3.5 × 5.5 km - very detailed)
✓ **Near real-time** (3-5 hour latency)
✓ **Daily global coverage**  
✓ **Multiple pollutants** in one mission
✓ **Free and open access**
✓ **Excellent for NO₂ monitoring** (primary strength)
✓ **Perfect for fire detection use case** (detects SO₂ and CO from fires)

### Limitations:
✗ **Cloud interference** (cannot see through clouds)
✗ **No direct PM2.5/PM10 measurement** (need AOD instead)
✗ **Daytime only** (sun-synchronous orbit)
✗ **Vertical column density** (total atmosphere column, not just ground level)

### Notes from exploration:
- TROPOMI has the highest spatial resolution for NO₂ measurements globally
- Data is available within hours of satellite overpass - critical for real-time system
- Can detect pollution from individual power plants and major roadways
- Successfully used for COVID-19 lockdown pollution monitoring
- **Best choice for starting our project** - excellent NO₂ data, near real-time, easy access

---

## MODIS (Terra/Aqua)

### What it provides:
- **AOD (Aerosol Optical Depth)** → Can be converted to **PM2.5/PM10** estimates
- Not direct pollutant measurement, but can be converted using statistical/ML models

### Key Specs:
- **Satellites**: Terra (morning overpass ~10:30 AM) & Aqua (afternoon overpass ~1:30 PM)
- **Launch Dates**: Terra (December 1999), Aqua (May 2002)
- **Coverage**: Near-daily global coverage (combined Terra + Aqua provides 2× daily coverage)
- **Resolution**: **250m, 500m, and 1km** depending on band
  - **AOD products**: 1 km resolution (MAIAC algorithm)
  - **Standard products**: 3 km or 10 km (Dark Target/Deep Blue algorithms)
- **Latency**: Near real-time to 48 hours
- **Data Format**: HDF, NetCDF

### Access:
- **Primary Portal**: https://earthdata.nasa.gov/
- **LAADS DAAC**: https://ladsweb.modaps.eosdis.nasa.gov/search/
- **Google Earth Engine**: Search "MCD19A2" for MAIAC AOD
- **Registration**: https://urs.earthdata.nasa.gov/users/new
- **My Username**: *(will fill after registration)*
- **Registration Date**: *(pending)*

### Products Available:
- **MOD04** (Terra) / **MYD04** (Aqua): Standard aerosol products (3km, 10km)
- **MCD19A2**: MAIAC AOD product (1km resolution, daily, best for PM2.5 estimation)
- **MOD/MYD08**: Atmospheric gridded products

### PM2.5 Estimation Method:
AOD measures how much light is blocked by aerosols in entire atmospheric column. To convert to ground-level PM2.5:

1. **Statistical Approach**:
   - Simple regression: `PM2.5 = AOD × 46.7 + 7.13` (example formula)
   - Varies by location, season, humidity
   
2. **Machine Learning Approach** (recommended):
   - Random Forest, CNN models
   - Input: AOD + meteorological data (humidity, temperature, wind, boundary layer height)
   - Output: Ground-level PM2.5
   - Accuracy: R² typically 0.7-0.85
   
3. **Chemical Transport Model** (complex):
   - GEOS-Chem model to relate column AOD to surface PM2.5
   - Requires meteorological reanalysis data

### Advantages:
✓ **Very high spatial resolution** (250m-1km)
✓ **Long data record** (20+ years)
✓ **Twice-daily coverage** (Terra + Aqua)
✓ **Proven PM2.5 estimation** (many research papers)
✓ **Free and open access**
✓ **Google Earth Engine support** (easy processing)

### Limitations:
✗ **No direct PM measurement** (requires conversion from AOD)
✗ **Cloud interference** (major issue - no data over cloudy areas)
✗ **Requires meteorological data** for accurate PM2.5 estimation
✗ **AOD-PM2.5 relationship varies** by location and season
✗ **Cannot detect other pollutants** (only aerosols)

### Notes:
- MAIAC algorithm (MCD19A2) provides best AOD data for PM2.5 estimation
- Correlation between AOD and PM2.5 is stronger in dry conditions
- Terra (morning) + Aqua (afternoon) provides temporal variation
- **Critical for PM2.5 in fire scenario** - will show massive AOD spikes during fires

---

## CAMS (Copernicus Atmosphere Monitoring Service)

### What it provides:
- **Near real-time atmospheric composition** analysis and forecasts
- All 6 AQI pollutants: PM2.5, PM10, NO₂, SO₂, CO, O₃
- **Model-based** (combines satellite observations with atmospheric chemistry models)

### Key Specs:
- **Type**: Data assimilation system (not pure satellite observations)
- **Resolution**: ~40 km (coarser than Sentinel-5P or MODIS)
- **Temporal**: Hourly updates, 5-day forecasts
- **Latency**: Near real-time (within hours)
- **Data Format**: GRIB, NetCDF

### Access:
- **Portal**: https://ads.atmosphere.copernicus.eu/
- **Requires**: Copernicus ADS account (free)
- **API**: Python CDS API client

### Advantages:
✓ **All pollutants** in one source
✓ **Gap-filling** where satellite data is missing (clouds)
✓ **Forecast capability** (predict future pollution)
✓ **Global coverage** including polar regions
✓ **No cloud issues** (model-based)

### Limitations:
✗ **Lower spatial resolution** (~40 km vs 3-5 km for satellites)
✗ **Model-based** (not direct measurements)
✗ **May have lower accuracy** than pure satellite-based estimates

### Use Case:
- **Backup/supplement** to satellite data
- **Fill gaps** when satellite data unavailable due to clouds
- **Forecasting** future AQI
- **Validation** of our satellite-based estimates

---

## Ground Truth Data

### OpenAQ

**URL**: https://openaq.org/  
**API Documentation**: https://docs.openaq.org/

**What it provides**:
- Ground-level measurements from **monitoring stations worldwide**
- Pollutants: PM2.5, PM10, NO₂, SO₂, CO, O₃, BC
- **Thousands of locations** across 100+ countries
- **Real-time and historical data**

**Coverage**:
- Primary: Urban areas with government monitoring networks
- US, Europe, China, India have extensive coverage
- Developing countries have sparse coverage

**Access**:
- **REST API v3** (latest, v1/v2 deprecated Jan 2025)
- **Free** access, no API key required
- **JSON format**
- Python example:
  ```python
  import requests
  url = "https://api.openaq.org/v3/locations"
  response = requests.get(url)
  data = response.json()
  ```

**Use Case for Our Project**:
- **Training data**: Match satellite observations with ground measurements
- **Validation**: Compare our predictions against actual measurements
- **Ground truth**: Check model accuracy
- **Fire scenario**: Check if ground stations detected fire pollution (if any nearby)

---

### EPA AirNow (US)

**URL**: https://www.airnow.gov/  
**API**: https://docs.airnowapi.org/

**Coverage**: United States (comprehensive)
**Data**: All 6 AQI pollutants, hourly updates
**Format**: JSON, XML
**Requires**: Free API key (request via email)

**Use Case**:
- Validation in US regions
- Higher quality/frequency than OpenAQ in US
- Real-time AQI values for comparison

---

### Local Environmental Agencies

**Pakistan EPA**: http://environment.gov.pk/ (if available)
**India CPCB**: https://cpcb.nic.in/ (comprehensive network)
**China MEE**: http://www.mee.gov.cn/ (extensive but access may be limited)

**Use Case**:
- Region-specific validation
- May have data not in OpenAQ
- Check for local fire incidents

---

## Data Comparison Summary

| Source | Pollutants | Resolution | Update Frequency | Coverage | Best For |
|--------|-----------|------------|------------------|----------|----------|
| **Sentinel-5P** | NO₂, SO₂, CO, O₃ | 3.5×5.5 km | Daily | Global | Real-time gas pollutants, fire detection |
| **MODIS** | AOD → PM | 1 km | Twice daily | Global | PM2.5/PM10 estimation, fire aerosols |
| **CAMS** | All 6 | 40 km | Hourly | Global | Gap-filling, forecasts |
| **OpenAQ** | All 6 | Point | Varies (hourly to daily) | Global (sparse) | Training, validation |
| **EPA AirNow** | All 6 | Point | Hourly | US only | High-quality validation |

---

## Recommended Data Strategy for Our Project

### Phase 1 (MVP - Start Simple):
1. **Primary**: **Sentinel-5P for NO₂**
   - Easiest to access and process
   - High resolution, near real-time
   - Direct measurement (no conversion needed)
   - Perfect for testing our pipeline
   
2. **Ground Truth**: **OpenAQ**
   - Free, global, easy API
   - Use for training and validation

### Phase 2 (Add PM):
3. **Add**: **MODIS AOD for PM2.5/PM10**
   - More complex (requires AOD→PM conversion)
   - Critical for fire scenario (fires produce lots of PM2.5)
   - May need meteorological data

### Phase 3 (Complete System):
4. **Add**: **More pollutants from Sentinel-5P** (SO₂, CO, O₃)
5. **Add**: **CAMS for gap-filling**
6. **Integrate**: **Multi-satellite fusion**

---

## Fire Incident Use Case - Data Strategy

For detecting pollution from fire incidents (our specific example):

**Required Data**:
- **Sentinel-5P**: SO₂, CO, NO₂ (fires produce all three)
- **MODIS**: AOD/PM2.5 (fires produce massive particulate matter)
- **MODIS Fire Products**: MOD14/MYD14 (active fire detection)
- **Time**: Check satellite overpasses for date/time of fire

**Approach**:
1. Identify fire location and time
2. Find Sentinel-5P overpass near that time
3. Find MODIS overpass (Terra/Aqua)
4. Extract pollutant values in area
5. Calculate AQI spike
6. Compare with normal (non-fire) days
7. Validate with any ground station data if available

**Expected Results**:
- **PM2.5**: 10-50× increase
- **CO**: 5-20× increase  
- **NO₂**: 2-5× increase
- **SO₂**: Variable (depends on what's burning)

---

## Next Steps

- [x] Research satellite data sources
- [ ] Register for Copernicus account (Sentinel-5P)
- [ ] Register for NASA Earthdata account (MODIS)
- [ ] Test OpenAQ API access
- [ ] Download sample Sentinel-5P NO₂ data
- [ ] Download sample MODIS AOD data
- [ ] Visualize sample data in Python
