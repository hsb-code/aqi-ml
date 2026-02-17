# Getting Started Guide

## Prerequisites

- Python 3.9 or higher
- Git
- 10GB+ free disk space for satellite data
- (Optional) CUDA-capable GPU for model training

## Step 1: Environment Setup

### Create Conda Environment

```bash
# Create conda environment
conda create -n aqi python=3.9 -y

# Activate it
conda activate aqi
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Register for Satellite Data Access

### Copernicus (Sentinel-5P)

1. Visit: https://dataspace.copernicus.eu/ and click "Register"
2. Create an account
3. Add credentials to `.env` file

### NASA Earthdata

1. Visit: https://urs.earthdata.nasa.gov/users/new
2. Create an account
3. Approve applications for:
   - NASA GESDISC DATA ARCHIVE
   - LAADS DAAC
4. Add credentials to `.env` file

## Step 3: Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
notepad .env  # or use your preferred editor
```

## Step 4: Verify Installation

```bash
# Run a simple test
python scripts/verify_setup.py
```

## Next Steps

1. **Research Phase**: Review [implementation_plan.md](file:///C:/Users/hasee/.gemini/antigravity/brain/69f2a25c-2813-4381-aee5-20262f72515a/implementation_plan.md)
2. **Download Sample Data**: Use scripts in `src/data_acquisition/`
3. **Explore Data**: Use notebooks in `notebooks/exploration/`
4. **Start Development**: Follow Phase 1 tasks in [task.md](file:///C:/Users/hasee/.gemini/antigravity/brain/69f2a25c-2813-4381-aee5-20262f72515a/task.md)

## Helpful Resources

- [AQI Calculation Methodology](https://www.airnow.gov/aqi/aqi-basics/)
- [Sentinel-5P Documentation](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p)
- [MODIS Documentation](https://modis.gsfc.nasa.gov/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Troubleshooting

### Issue: GDAL installation fails

**Solution**: 
```bash
# On Windows, use pre-built wheels:
pip install GDAL‑3.4.3‑cp39‑cp39‑win_amd64.whl
```

### Issue: Cannot access satellite data

**Solution**: 
- Verify credentials in `.env`
- Check if API endpoints are accessible
- Ensure you've approved required applications

## Need Help?

Refer to the detailed [implementation_plan.md](file:///C:/Users/hasee/.gemini/antigravity/brain/69f2a25c-2813-4381-aee5-20262f72515a/implementation_plan.md) for comprehensive guidance.
