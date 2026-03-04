# AQI Prediction: Experiment Summary (1–10)

This document provides a strategic overview of the nine experiments conducted to predict PM25 and PM10 levels using Satellite AOD and meteorological data.

## 📊 Performance Comparison (2024 Test Set R²)

| Exp | Description | PM2.5 R² | PM10 R² | Key Outcome |
| :--- | :--- | :---: | :---: | :--- |
| 1 | Baseline LightGBM (Raw Features) | 0.2356 | 0.2607 | Established low baseline. |
| 2 | LGBM + Basic Lags (1-7 days) | 0.5357 | 0.5290 | **Massive jump** in predictive power. |
| 3 | LGBM + Optuna + Station Encoding | 0.5531 | 0.5547 | Fine-tuning the lag-driven approach. |
| 4 | **LGBM + Extended Lags (Winner)** | **0.5599** | **0.5605** | **Best single-model performance.** |
| 5 | CatBoost (Native Categorical) | 0.5338 | 0.5388 | Strong training fit, but overfitted. |
| 6 | LGBM + Log-Transform Target | 0.5305 | 0.5245 | Log-scale didn't improve R². |
| 7 | Ensemble (Exp 4 + Exp 5) | 0.5599 | 0.5605 | Weighted average favored LGBM. |
| 8 | Deep Learning (LSTM) | 0.0278 | -0.0833 | Time-series sequences were too noisy. |
| 9 | Geospatial Features (3-Station NN) | 0.5594 | 0.5526 | High correlation, but redundant with lags. |
| 10 | XGBoost + Optuna | 0.5455 | 0.5374 | Completed GBM trifecta; sitting behind LGBM. |

---

## 🔍 Strategic Insights

1. **Temporal Context is King**: The largest performance gain occurred in **Experiment 2**, when we introduced historical lags. This proves that past pollution levels are significantly more predictive than current meteorology or AOD alone.
2. **The 0.56 Plateau**: We have tested model architectures (GBMs, RNNs), hyperparameter tuning (Optuna), and target transformations (Log). Despite this, we are hitting a consistent plateau at **R² ≈ 0.56**.
3. **Robustness of LightGBM**: Optimized LightGBM (Exp 4) consistently outperforms more "modern" approaches like CatBoost or LSTMs on this specific tabular time-series dataset.

---

## 🛑 Current Blockers
- **Data Limits**: Satellite AOD and ERA5 meteorology provide a "macro" view, but AQI is often driven by "micro" factors (local traffic, industry, topography) not currently captured in our feature set.
- **Noise vs. Signal**: Pollution data is highly volatile. The negative R² in the LSTM trial suggests that complex models are easily distracted by noise in the temporal sequences.

---

## 🚀 Way Forward (The Path to 0.75 R²)

To break the 0.60 ceiling and move toward the project goal, we recommend:

1. **Feature Expansion (External Data)**:
   - **Land Use Regression (LUR)**: Add features for road density, distance to factories, and green cover.
   - **High-Res Topography**: Add Elevation and Slope features to capture how pollution "pools" in valleys.
2. **Temporal Window Optimization**:
   - Experiment with **dynamic lags** (e.g., using different lookback windows depending on the season).
3. **Specialized Models**:
   - Train separate models for **Extreme Pollution Events** (high AQI) vs. **Baseline Days** (low AQI).
4. **Ensemble Refinement**:
   - Moving from simple "Weighted Averaging" to **Ridge Stacking**, where a meta-model learns which base model to trust on specific days.
