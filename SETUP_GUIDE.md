# AQI ML Project - Quick Start Guide

## 🚀 Complete Setup Steps (For NO₂ Download)

Follow these steps to download NO₂ data from Sentinel-5P (2022-2024):

---

## Step 1: Create Conda Environment

```bash
# Create new conda environment with Python 3.9
conda create -n aqi-ml python=3.9 -y

# Activate the environment
conda activate aqi-ml
```

---

## Step 2: Install Python Dependencies

```bash
# Navigate to project directory
cd H:\AQI

# Install all required packages
pip install -r requirements.txt
```

**This will install:**
- `earthengine-api` - For Google Earth Engine access
- `xarray`, `pandas`, `numpy` - For data processing
- `scikit-learn` - For ML (later phases)
- Other utilities

---

## Step 3: Authenticate Google Earth Engine

**First time setup only:**

```bash
# Method 1: Standard authentication
earthengine authenticate

# Method 2: If Method 1 fails, try notebook mode
earthengine authenticate --auth_mode=notebook
```

**This will:**
1. Open a browser window
2. Ask you to login with your Google account
3. Give you an authorization code
4. Paste the code back in the terminal

**Note:** You need a Google account. If you don't have GEE access, register at: https://code.earthengine.google.com/register

---

## Step 4: Test GEE Connection

```bash
# Quick test to verify GEE works
python -c "import ee; ee.Initialize(); print('✓ GEE connected successfully!')"
```

**Expected output:** `✓ GEE connected successfully!`

If this fails, your GEE authentication didn't work. Re-run Step 3.

---

## Step 5: Download NO₂ Data (2022-2024)

```bash
# Download ONLY NO₂ (about 3 years of data)
python scripts/01_download_training_data.py \
    --products NO2 \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --log-level INFO
```

**What happens:**
- Script will extract NO₂ data for Abu Dhabi region
- Date range: Jan 2022 → Dec 2024 (3 years)
- Output: `H:\AQI\data\raw\NO2\`
- Logs: `H:\AQI\logs\data_download.log`

**Expected time:** 30-60 minutes (depends on GEE queue)

**Expected size:** ~10-15 GB

---

## 📊 Monitor Progress

While downloading, you can check logs:

```bash
# View download logs in real-time
Get-Content H:\AQI\logs\data_download.log -Wait -Tail 50
```

---

## ✅ Verify Download Completed

```bash
# Check if NO2 directory exists and has files
dir H:\AQI\data\raw\NO2
```

You should see:
- `NO2_info.txt` - Download metadata
- Folders with actual NO₂ data

---

## 🔄 Complete Command Reference

### Download NO₂ only:
```bash
conda activate aqi-ml
cd H:\AQI
python scripts/01_download_training_data.py --products NO2
```

### Download NO₂ + MODIS AOD:
```bash
python scripts/01_download_training_data.py --products NO2 MODIS_AOD
```

### Download everything:
```bash
python scripts/01_download_training_data.py --products all
```

### Resume interrupted download:
```bash
python scripts/01_download_training_data.py --products NO2 --skip-existing
```

### Download specific date range:
```bash
python scripts/01_download_training_data.py \
    --products NO2 \
    --start 2024-01-01 \
    --end 2024-12-31
```

---

## 🛠️ Troubleshooting

### GEE Authentication Fails
```bash
# Clear existing credentials and re-authenticate
earthengine authenticate --force
```

### Import Errors
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Script Not Found
```bash
# Make sure you're in the right directory
cd H:\AQI
pwd  # Should show H:\AQI
```

### Download Stalls
- **Normal!** GEE can be slow during peak hours
- Check GEE status: https://status.earthengine.google.com/
- Wait 10-15 minutes, script has retry logic

---

## 📁 Expected Directory Structure After Download

```
H:\AQI\
├── data\
│   └── raw\
│       └── NO2\
│           ├── NO2_info.txt
│           └── NO2_20220101_20241231\
│               └── (NO2 data files)
│
├── logs\
│   ├── data_download.log
│   └── GEEDownloader.log
│
├── src\
│   └── (your Python modules)
│
└── scripts\
    └── 01_download_training_data.py
```

---

## ⏭️ Next Steps

After NO₂ download completes:

1. **Download MODIS AOD and ERA5** (Week 1-2)
   ```bash
   python scripts/01_download_training_data.py --products MODIS_AOD ERA5
   ```

2. **Process satellite data** (Week 2)
   - Run preprocessing scripts (to be created)

3. **Train ML model** (Week 3)
   - Match NO₂ + AOD with ground station data
   - Train Random Forest

4. **Predict December 2025 PM** (Week 4)

---

## 🆘 Need Help?

- Check logs: `H:\AQI\logs\data_download.log`
- Review configuration: `H:\AQI\src\config\settings.py`
- Read full docs: `H:\AQI\DATA_ACQUISITION_README.md`

---

## 📝 Quick Summary

```bash
# Complete setup in 5 steps:
conda create -n aqi-ml python=3.9 -y
conda activate aqi-ml
cd H:\AQI
pip install -r requirements.txt
earthengine authenticate
python scripts/01_download_training_data.py --products NO2
```

That's it! 🎉
