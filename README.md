# Real-time Satellite-based AQI Monitoring System

A deep learning project to estimate Air Quality Index (AQI) from real-time satellite imagery, primarily designed for regions lacking ground-based monitoring stations.

## 🚀 Project Overview

This repository contains a full pipeline for air quality monitoring:
1.  **Data Acquisition**: Automated downloading of Sentinel-5P (pollutants) and MODIS (AOD) data.
2.  **Preprocessing**: Grid-based alignment and temporal feature engineering.
3.  **Machine Learning**: Advanced model training using Random Forest and XGBoost.
4.  **Comparison**: Detailed evaluation of model performance across experiments.

---

## ✅ What Has Been Done

- [x] **Phase 1 & 2: Infrastructure**: Established directory structure and automated data acquisition pipelines.
- [x] **Phase 3: Preprocessing**: Implemented robust data cleaning and lag feature engineering (PM lag, rolling trends).
- [x] **Phase 4: Experimentation**:
    - **Experiment 1 (Baseline)**: Established Random Forest models for PM2.5 and PM10.
    - **Experiment 2 (Optimization)**: Upgraded to XGBoost with GridSearch hyperparameter tuning.
    - **Results**: Achieved **+20.7% improvement in PM2.5 R²** and **+19.0% improvement in PM10 R²**.
- [x] **Analytics**: Created comprehensive [EXPERIMENT_COMPARISON.md](models/EXPERIMENT_COMPARISON.md) documenting key wins.

---

## 🛠 Project Structure

```text
AQI/
├── data/               # Data storage (raw & processed) - [Ignored by Git]
├── src/                # Core Python package
│   ├── data_acquisition/ # Satellite downloaders
│   ├── preprocessing/    # Data pipelines
│   ├── ml/              # Training & Model logic
│   └── inference/        # Prediction endpoints
├── models/             # Experiment results & checkpoints
│   ├── exp1/           # Baseline Random Forest results
│   ├── exp2/           # Optimized XGBoost results
│   └── checkpoints/    # Model weights - [Ignored by Git]
├── scripts/            # Executable workflows (01-04)
├── docs/               # Technical guides & roadmaps
└── shared/             # Shared utilities & ancillary data
```

---

## 🏃 Next Steps

- [ ] **Phase 5**: Develop real-time inference API for live satellite passes.
- [ ] **Phase 6**: Deploy monitoring dashboard for "Areas of Interest" (AOI).

---

## 🏁 How to Run

1.  **Setup**: Follow [SETUP_GUIDE.md](SETUP_GUIDE.md).
2.  **Data**: Run `python scripts/01_download_training_data.py`.
3.  **Process**: Run `python scripts/02_preprocess_data.py`.
4.  **Train**: Run `python scripts/03_train_models.py` (Exp1) or `python scripts/04_train_models_exp2.py` (Exp2).
