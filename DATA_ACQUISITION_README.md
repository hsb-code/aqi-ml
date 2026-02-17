# AQI ML Implementation - Professional Data Acquisition System

## 🎉 Phase 1 Completed: Code Architecture Ready!

### What's Been Built

We've created a **professional, production-ready** data acquisition system for downloading satellite and meteorological data.

---

## 📁 Project Structure

```
H:\AQI\
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          ✅ Central configuration
│   │
│   ├── data_acquisition/
│   │   ├── __init__.py
│   │   ├── base_downloader.py   ✅ Abstract base class
│   │   ├── gee_downloader.py    ✅ Google Earth Engine
│   │   └── cds_downloader.py    ✅ Copernicus CDS (ERA5)
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py             ✅ Logging system
│
├── scripts/
│   └── 01_download_training_data.py  ✅ Main download script
│
├── data/
│   └── raw/              ← Downloads will go here
│
└── logs/                 ← Log files will be created here
```

---

## 🚀 Usage Guide

### Prerequisites

1. **Google Earth Engine** access:
   ```bash
   # Authenticate GEE (first time only)
   earthengine authenticate
   ```

2. **Copernicus CDS** account:
   - Register: https://cds.climate.copernicus.eu/user/register
   - Create `~/.cdsapirc` file with:
     ```
     url: https://cds.climate.copernicus.eu/api/v2
     key: <YOUR_UID>:<YOUR_API_KEY>
     ```

---

### Download Training Data (2022-2024)

#### Option 1: Download Everything
```bash
cd H:\AQI
python scripts/01_download_training_data.py --products all
```

This will download:
- Sentinel-5P: NO₂, SO₂, CO, O₃ (2022-2024)
- MODIS AOD (2022-2024)
- ERA5 meteorology (2022-2024)

#### Option 2: Download Specific Products
```bash
# Download only S5P NO2 and SO2
python scripts/01_download_training_data.py --products NO2 SO2

# Download only ERA5
python scripts/01_download_training_data.py --products ERA5

# Download only MODIS AOD
python scripts/01_download_training_data.py --products MODIS_AOD
```

#### Option 3: Custom Date Range
```bash
# Download specific date range
python scripts/01_download_training_data.py \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --products all
```

#### Option 4: Skip Existing Data
```bash
# Resume interrupted download
python scripts/01_download_training_data.py \
    --products all \
    --skip-existing
```

---

### Command-Line Options

```
usage: 01_download_training_data.py [-h] [--start START] [--end END]
                                     [--products {NO2,SO2,CO,O3,MODIS_AOD,ERA5,all} ...]
                                     [--output-dir OUTPUT_DIR]
                                     [--skip-existing]
                                     [--log-level {DEBUG,INFO,WARNING,ERROR}]

Options:
  --start START         Start date (YYYY-MM-DD) [default: 2022-01-01]
  --end END             End date (YYYY-MM-DD) [default: 2024-12-31]
  --products            Products to download [default: all]
  --output-dir          Output directory [default: H:/AQI/data/raw]
  --skip-existing       Skip products that already have data
  --log-level           Logging verbosity [default: INFO]
```

---

## 🔍 Testing the System

### Test Configuration
```bash
cd H:\AQI
python -c "from src.config.settings import config; print(config)"
```

**Expected Output:**
```
Config(
  Training: 2022-01-01 to 2024-12-31
  Region: Abu Dhabi
  Target CRS: EPSG:3857
  Model: RandomForest
)
```

### Test GEE Downloader
```python
from src.data_acquisition.gee_downloader import GEEDownloader
from datetime import datetime

# Initialize
gee = GEEDownloader()

# Test with small date range
output = gee.download('NO2', datetime(2024, 12, 1), datetime(2024, 12, 7))
print(f"Test download: {output}")
```

### Test CDS Downloader
```python
from src.data_acquisition.cds_downloader import CDSDownloader
from datetime import datetime

# Initialize
cds = CDSDownloader()

# Test with small date range
output = cds.download('ERA5', datetime(2024, 1, 1), datetime(2024, 1, 7))
print(f"Test download: {output}")
```

---

## 📊 Expected Downloads

### File Structure After Download
```
H:\AQI\data\raw\
├── NO2/
│   ├── NO2_20220101_20241231/
│   └── NO2_info.txt
│
├── SO2/
│   └── SO2_20220101_20241231/
│
├── CO/
│   └── CO_20220101_20241231/
│
├── O3/
│   └── O3_20220101_20241231/
│
├── MODIS_AOD/
│   └── MODIS_AOD_20220101_20241231/
│
└── ERA5/
    ├── ERA5_chunk01_20220101_20221231.nc
    ├── ERA5_chunk02_20230101_20231231.nc
    ├── ERA5_chunk03_20240101_20241231.nc
    └── ERA5_merged_2022-01-01_2024-12-31.nc
```

### Storage Requirements
- **S5P data** (4 pollutants × 3 years): ~40 GB
- **MODIS AOD** (3 years): ~30 GB
- **ERA5** (6 variables × 3 years): ~50 GB
- **Total**: ~120 GB

---

## 🔧 Configuration

All settings are in `src/config/settings.py`:

### Download Configuration
- **Training Period**: 2022-01-01 to 2024-12-31
- **Prediction Period**: 2025-12-01 to 2025-12-31
- **Region**: Abu Dhabi (bbox: [51.5, 22.5, 56.0, 25.5])
- **Chunk Size**: 12 months (for ERA5)
- **Retries**: 3 attempts with 60s delay

### Processing Configuration
- **Target CRS**: EPSG:3857 (Web Mercator)
- **Resolution**: 1000m (1km)
- **Temporal Resolution**: Daily

### Model Configuration
- **Algorithm**: Random Forest
- **Estimators**: 200 trees
- **Max Depth**: 20
- **Target R²**: PM2.5 > 0.75, PM10 > 0.73

---

## 📝 Logging

Logs are saved to `H:\AQI\logs\`:

- **Console**: INFO level
- **File**: DEBUG level
- **Rotation**: 10 MB per file, 5 backups

View logs:
```bash
# View latest download log
tail -f H:\AQI\logs\data_download.log

# View GEE downloader log
tail -f H:\AQI\logs\GEEDownloader.log
```

---

## ⚠️ Troubleshooting

### GEE Authentication Fails
```bash
# Solution: Re-authenticate
earthengine authenticate --auth_mode=notebook
```

### CDS Downloads Stuck
- **Reason**: CDS queue can be slow (12-24 hours)
- **Solution**: Check queue status at https://cds.climate.copernicus.eu/live/queue
- **Tip**: Run downloads overnight or during weekends

### Missing Data Days
- **Expected**: 20-40% missing days due to clouds
- **Solution**: ML models can handle missing data

---

## ✅ Next Steps

After downloads complete:
1. **Verify data completeness** (Week 2)
2. **Process satellite data** (Week 2)
3. **Train ML models** (Week 3)
4. **Generate December 2025 predictions** (Week 4)

See `task.md` for detailed checklist!

---

## 🆘 Need Help?

- **Configuration issues**: Check `src/config/settings.py`
- **Download errors**: Check logs in `H:\AQI\logs\`
- **API access**: Verify GEE and CDS credentials
- **Questions**: Review `implementation_plan.md`
