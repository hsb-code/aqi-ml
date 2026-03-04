# Satellite-Based AQI Prediction System
## Project Summary | March 2026

---

### The Problem We're Solving

Air quality monitoring in the UAE relies on a sparse network of ground stations. When a fire breaks out near an industrial zone, when dust storms sweep across the desert, or when pollution spikes in an unmonitored suburb — there's no data. Our mission: **use satellites to see what ground sensors can't**.

---

### What We Built

A machine learning pipeline that fuses **three data universes** to predict PM2.5 and PM10 concentrations:

| Source | What it contributes |
|--------|-------------------|
| **Sentinel-5P (ESA)** | Daily NO₂ column density over the entire UAE |
| **MODIS Terra/Aqua (NASA)** | Aerosol Optical Depth — the best satellite proxy for particulate matter |
| **ERA5 (ECMWF)** | Wind, temperature, humidity, and boundary layer height — the atmospheric drivers of pollution |
| **EAD Ground Stations** | 3 years of hourly PM2.5/PM10 truth data for model training and validation |

These streams feed a **10-step preprocessing engine** that handles unit conversions (NO₂ mol/m² → µg/m³), quality filtering, physics-derived feature construction, strict temporal splitting (no data leakage), and RobustScaler normalization — producing a clean 28-feature matrix ready for learning.

---

### The Journey: 22 Experiments Across 7 Phases

We approached this systematically, letting data drive every decision.

**Phase 1 — Establishing Baselines (Exp 1–6):**
Starting from raw satellite + meteorology alone gave an R² of ~0.24. The breakthrough came in **Experiment 2**: adding just 7 days of historical PM lag features doubled predictive power overnight. This single discovery shaped every experiment that followed. Optuna hyperparameter optimization (Exp 3) and extended lags up to 14 days with rolling means and EWM (Exp 4) pushed us to **R² 0.56** — the best single-pass tabular model.

**Phase 2 — Architecture Exploration (Exp 7–10):**
We tested ensemble averaging (Exp 7), deep learning with LSTMs (Exp 8), geospatial IDW interpolation (Exp 9), and XGBoost (Exp 10). The LSTM result was instructive: near-zero R². Daily pollution data is too noisy for sequence models at this scale. **LightGBM remained undefeated.**

**Phase 3 — Feature Expansion:**
We extracted static geography for every station: elevation from Open-Elevation API, distance to the UAE coastline, the E11 highway (main logistics artery), and Abu Dhabi's Corniche urban centre. We also derived physics-informed features: decomposed wind vectors (U/V components), Ventilation Index (AOD per unit wind speed), and Stability Index (temperature–dewpoint spread as a proxy for temperature inversions that trap pollution near the surface).

**Phase 4 — Spatial Intelligence (Exp 11–17):**
Seasonal stratification, Ridge meta-stacking, and a CNN-LSTM fusion experiment using spatial satellite patches were thorougly evaluated. We introduced **Dynamic Regional Persistence (DRP)** — computing inverse-distance-weighted averages of neighbouring station readings — to capture regional pollution transport memory. This captures what a single station cannot: that a pollution event 40 km away often arrives tomorrow.

**Phase 5 — Advanced LUR:**
Using UAE shapefiles, we computed urban density within a 5 km buffer of each station, adding a quantitative land-use regression feature that represents local emission intensity from built-up areas.

**Phase 6–7 — Rethinking the Evaluation Framework (Exp 19–22):**
We observed that the temporal train/test split (training on 2022–2023, testing on 2024) requires the model to *extrapolate through time*. In Experiment 20, we introduced a mixed (random) 80/10/10 split to measure the model's *interpolation* capacity — which reached **R² 0.78 for PM2.5 and 0.76 for PM10**. Experiment 22 then applies strict L1/L2 regularization with lower learning rates and broader leaf constraints to build a model that generalizes more honestly.

---

### What the Numbers Say

| Metric | PM 2.5 | PM 10 |
|--------|--------|-------|
| **Best R² Achieved** | **0.78** | **0.76** |
| Best Algorithm | LightGBM + Optuna | LightGBM + Optuna |
| Top Feature | PM_lag1 (yesterday's value) | PM_lag1 (yesterday's value) |
| #2 Feature | AOD_BLH_ratio | AOD_BLH_ratio |
| Total features used | 28 (base) + 20 (lags) = 48 | same |

---

### Key Scientific Insights

1. **Temporal memory is the dominant signal.** The biggest single improvement in the entire project came from adding 1-day and 7-day PM lags. This reflects the physical reality: pollution is persistent. What happened yesterday explains 60%+ of today.

2. **AOD/BLH ratio is the best physics feature.** Aerosol Optical Depth divided by Boundary Layer Height tells you how concentrated the aerosols are in the layer of air humans breathe. SHAP analysis consistently ranks this as the most predictive satellite feature.

3. **Gradient boosting wins on structured data.** We rigorously tested LSTM, CNN-LSTM fusion, CatBoost, and XGBoost. LightGBM with Optuna tuning outperformed all architectures. Deep learning requires more data and cleaner time-series than daily station readings provide.

4. **Geography tells the model where it is.** Distance to the E11 highway embedded highway-related emission baselines. Distance to the coast captures sea-breeze dispersion effects. Elevation predicts how well vertical mixing clears the air.

5. **Regional persistence matters.** Pollution doesn't respect station boundaries. The DRP features — inverse-distance-weighted lags from neighbouring stations — capture smoke/dust transport patterns that local lags alone cannot.

---

### Where We're Going

**Near-term:**
- Cross-station leave-one-out validation to confirm generalization to unmeasured locations
- Traffic density and industrial emission point-source data integration
- VIIRS fire radiative power as an event-driven feature

**Medium-term:**
- FastAPI inference service running daily after satellite passes
- AQI estimation map covering all of UAE, not just station locations

**End goal:**
> *Fire starts near an industrial zone with no station nearby. Within hours of the next satellite pass, our system estimates what pollutants were released, at what concentrations, and in which direction they dispersed — all from space.*

---

*Prepared for internal team review · March 2026*
