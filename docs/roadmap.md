# Project Roadmap - Quick Reference

## Phase 1: Research & Foundation (Week 1-2) 🔍
**Goal**: Understand the science and identify data sources

**Key Activities**:
- Learn AQI calculation (which pollutants, how to calculate)
- Find satellite data sources (Sentinel-5P, MODIS)
- Study existing research papers
- Create technical approach document

**Deliverables**: Documentation in `docs/` folder

---

## Phase 2: Data Acquisition (Week 3-4) 📡
**Goal**: Download and store satellite data

**Key Activities**:
- Register accounts (Copernicus, NASA Earthdata)
- Build download scripts
- Get sample satellite images
- Download ground station data for validation

**Deliverables**: Working data pipeline, sample datasets

---

## Phase 3: Data Processing (Week 5-6) 🔧
**Goal**: Prepare data for model training

**Key Activities**:
- Clean and preprocess satellite images
- Extract pollutant values
- Match with ground truth data
- Create training datasets

**Deliverables**: Processed datasets ready for ML

---

## Phase 4: Model Development (Week 7-9) 🤖
**Goal**: Build and train deep learning model

**Key Activities**:
- Design model architecture (CNN/Transformer)
- Train model on historical data
- Validate accuracy
- Optimize performance

**Deliverables**: Trained model achieving target accuracy

---

## Phase 5: Real-time System (Week 10-11) ⚡
**Goal**: Enable real-time predictions

**Key Activities**:
- Build real-time data fetching
- Create inference pipeline
- Develop REST API
- (Optional) JavaScript SDK

**Deliverables**: Working API for real-time AQI

---

## Phase 6: Testing & Deployment (Week 12+) 🚀
**Goal**: Validate and deploy

**Key Activities**:
- Test fire incident scenario
- Compare with ground stations
- Deploy to cloud
- Document everything

**Deliverables**: Production-ready system

---

## Success Criteria

✓ Model predicts AQI with R² > 0.75  
✓ Works in areas without monitoring stations  
✓ Real-time inference < 5 seconds  
✓ Successfully detects pollution from fire incidents

---

## Current Status

📍 **You are here**: Project setup complete, ready to start Phase 1

**Next immediate steps**:
1. Read [getting_started.md](file:///h:/AQI/docs/getting_started.md)
2. Set up Python environment
3. Register for satellite data access
4. Begin Phase 1 research (see [phase1_START_HERE.md](file:///h:/AQI/docs/phase1/phase1_START_HERE.md))
