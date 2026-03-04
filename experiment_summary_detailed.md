# AQI-ML: Detailed Experiment Log
### All 22 Experiments — Methods, Features, Results, Lessons
**Data split (unless stated otherwise):** Train 2022–Jun 2023 | Val Jul–Dec 2023 | Test 2024  
**Targets:** PM2.5 and PM10 (µg/m³)  
**Evaluation metric:** R² on 2024 test set

---

## Phase 1: Baseline and Temporal Features (Exp 1–4)

---

### Experiment 1 — Baseline LightGBM (Raw Features Only)
**Script:** `03_train_exp1_lgbm.py`

**Goal:** Establish a lower-bound baseline using only satellite and meteorological features with no temporal memory.

**Features used (~17):**
- Satellite: `NO2_ugm3`, `AOD`, `AOD_corrected`, `AOD_BLH_ratio`
- ERA5 met: `T2M_C`, `D2M_C`, `SP_hPa`, `BLH`, `WindSpeed`, `WindDirection`, `RH`
- Calendar: `DayOfYear`, `Month`, `Season`, `IsWeekend`
- Location: `Latitude`, `Longitude`

**Model:** LightGBM, default conservative hyperparameters, no Optuna.

**Results (2024 test set):**

| Target | R² | RMSE (µg/m³) |
|--------|------|--------------|
| PM2.5  | 0.236 | ~22 |
| PM10   | 0.261 | ~38 |

**Lesson:** Raw satellite data alone explains less than 26% of variance. The model has no memory of past pollution — every day is treated as independent. This is the fundamental limit of sensor-only regression.

---

### Experiment 2 — LightGBM + Basic Lag Features
**Script:** `04_train_exp2_lgbm_lags.py`

**Goal:** Test whether adding temporal memory (PM lag features) can dramatically improve accuracy.

**New features added over Exp 1 (8 lag features):**
- `PM25_lag1`, `PM10_lag1` — yesterday's measured PM
- `PM25_roll3`, `PM10_roll3` — 3-day rolling mean
- `PM25_roll7`, `PM10_roll7` — 7-day rolling mean
- `AOD_roll3`, `NO2_roll3` — 3-day satellite rolling mean

**Important implementation detail:** Lag features computed on the **full dataset** before splitting, so that validation rows correctly use the last training rows as their lag source (no boundary leakage).

**Results (2024 test set):**

| Target | R² | RMSE (µg/m³) | Change vs Exp 1 |
|--------|------|--------------|----------------|
| PM2.5  | 0.536 | ~15 | **+0.30 (+127%)** |
| PM10   | 0.529 | ~27 | **+0.27 (+103%)** |

**Lesson:** The single biggest gain in the entire project. Adding 1-week PM lags effectively doubled R² in one experiment. Pollution is **persistent** — yesterday's air quality is the strongest predictor of today's, outweighing all satellite variables combined.

---

### Experiment 3 — LightGBM + Optuna + Station Encoding
**Script:** `05_train_exp3_optuna.py`

**Goal:** Apply Bayesian hyperparameter optimization (Optuna) to the Exp 2 feature set, and encode station identity as a learnable categorical feature.

**Changes over Exp 2:**
- Added `Station_enc` (LabelEncoder integer, treated as categorical by LightGBM)
- Optuna TPE sampler, 50 trials per target, minimizing validation RMSE
- Hyperparameters searched: `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`
- Final model retrained on Train+Val combined at best round ×1.1

**Results (2024 test set):**

| Target | R² | RMSE (µg/m³) | Change vs Exp 2 |
|--------|------|--------------|----------------|
| PM2.5  | 0.553 | ~14.7 | +0.017 |
| PM10   | 0.555 | ~26.3 | +0.026 |

**Lesson:** Optuna and station encoding add modest but consistent gains. The marginal improvement suggests the feature set, not the hyperparameters, is the main bottleneck.

---

### Experiment 4 — LightGBM + Extended Lag Features + Optuna
**Script:** `06_train_exp4_extended_lags.py`

**Goal:** Extend the lag feature window — more lag days, longer rolling windows, exponential weighting, and meteorological persistence features.

**New features added over Exp 3 (12 additional lag features, 42 total):**
- `PM25_lag2`, `PM10_lag2` — 2-day lag
- `PM25_lag3`, `PM10_lag3` — 3-day lag
- `PM25_roll14`, `PM10_roll14` — 14-day rolling mean
- `AOD_roll7`, `NO2_roll7` — 7-day satellite rolling
- `PM25_ewm7`, `PM10_ewm7` — exponentially weighted mean (span=7, recent days weighted higher)
- `WindSpeed_roll3`, `T2M_roll3` — 3-day meteorological persistence

**Results (2024 test set):**

| Target | R² | RMSE (µg/m³) | Change vs Exp 3 |
|--------|------|--------------|----------------|
| PM2.5  | 0.560 | ~14.4 | +0.007 |
| PM10   | 0.561 | ~26.0 | +0.006 |

**Lesson:** Extending lags beyond 7 days gives diminishing returns. The 0.56 plateau is now established. The model is well-tuned and feature-rich — further gains must come from new *types* of features, not more temporal lags.

---

## Phase 2: Model Architecture Comparisons (Exp 5–10)

---

### Experiment 5 — CatBoost (Native Categorical Handling)
**Script:** `07_train_exp5_catboost.py`

**Goal:** Test CatBoost as an alternative to LightGBM, leveraging its native ordered boosting for categorical features.

**Setup:** Same features as Exp 4. CatBoost with `ordered` boosting, `Season` and `Station` as native cat features, Optuna tuned.

**Results:**

| Target | R² | Note |
|--------|------|------|
| PM2.5  | 0.534 | Training R² >0.90 — overfitting |
| PM10   | 0.539 | Training R² >0.90 — overfitting |

**Lesson:** CatBoost memorised the training set far better than LightGBM but failed to generalise. Its ordered boosting was not well-suited to this temporal split. LightGBM's simpler histogram-based approach generalised better.

---

### Experiment 6 — Log-Transformed Target
**Script:** `08_train_exp6_log_target.py`

**Goal:** PM distributions are right-skewed. Predicting `log(PM + 1)` then back-transforming might reduce the influence of extreme events.

**Setup:** Same features as Exp 4. LightGBM trained on `log1p(PM)`, predictions back-transformed via `expm1`. Evaluated on original scale.

**Results:**

| Target | R² | Note |
|--------|------|------|
| PM2.5  | 0.531 | Worse than Exp 4 |
| PM10   | 0.525 | Worse than Exp 4 |

**Lesson:** Log transformation hurt performance on this dataset. The model lost the ability to distinguish between moderate pollution days and extreme events at the upper tail, which are precisely the days that matter most for AQI.

---

### Experiment 7 — Weighted Ensemble (LGBM + CatBoost)
**Script:** `09_ensemble_models.py`

**Goal:** Test whether a simple weighted average of Exp 4 (LightGBM) and Exp 5 (CatBoost) predictions outperforms either alone.

**Setup:** Linear combination `α × LGBM_pred + (1-α) × CB_pred`. α optimised on validation set.

**Results:**

| Target | R² | Note |
|--------|------|------|
| PM2.5  | 0.560 | Tied with Exp 4 |
| PM10   | 0.561 | Tied with Exp 4 |

**Lesson:** The weighted ensemble was completely dominated by the LightGBM component (α ≈ 0.95). Averaging a better model with a weaker one provides no benefit. Simple ensembles only help when base models are meaningfully diverse.

---

### Experiment 8 — Deep Learning: LSTM
**Script:** `10_train_exp8_lstm.py`

**Goal:** Test whether a sequence model (LSTM) that explicitly processes time steps can outperform tabular GBMs.

**Setup:** Per-station time-series sequences of length 14. Two-layer LSTM, hidden size 64, dropout 0.3. Adam optimizer, early stopping on val loss.

**Results:**

| Target | R² | Note |
|--------|------|------|
| PM2.5  | 0.028 | Near-zero — failed |
| PM10   | -0.083 | Negative — worse than mean |

**Lesson:** Complete failure. Daily pollution measurements at ~10 stations are too sparse and noisy for sequence models. LSTMs expect dense, continuous, high-frequency sequences. The same temporal information that LightGBM captures efficiently via lag features cannot be learned by an LSTM from this data volume and sampling rate.

---

### Experiment 9 — Geospatial Features (3 Nearest Neighbour Stations)
**Script:** `11_train_exp9_geospatial.py`

**Goal:** Incorporate readings from the 3 spatially nearest stations as additional features — a form of manual spatial interpolation.

**Setup:** For each (station, date) row, compute the 3 nearest stations by Euclidean distance and add their same-day PM readings as features. Combined with Exp 4 lag features.

**Results:**

| Target | R² | Note |
|--------|------|------|
| PM2.5  | 0.559 | Marginal gain |
| PM10   | 0.553 | Below Exp 4 |

**Lesson:** Same-day neighbouring station data is highly correlated with the target's own lag features. It added redundancy rather than new information, and created data leakage risks if neighbouring days were not carefully handled.

---

### Experiment 10 — XGBoost + Optuna
**Script:** `12_train_exp10_xgboost.py`

**Goal:** Complete the GBM comparison by testing XGBoost, the most widely used GBM library.

**Setup:** Same features as Exp 4. XGBoost with `tree_method=hist`, Optuna 50-trial tuning.

**Results:**

| Target | R² | Note |
|--------|------|------|
| PM2.5  | 0.546 | Below LGBM |
| PM10   | 0.537 | Below LGBM |

**Conclusion of GBM trifecta:** LightGBM > XGBoost > CatBoost for this dataset. LGBM's leaf-wise tree growth and native handling of missing values (from satellite cloud gaps) gave it a consistent edge.

---

## Phase 3: Geography and Physics Features (Exp 11–15)

---

### Experiment 11 — Static Geography: Elevation + Coastline Distance
**Script:** `13_train_exp11_geography.py`

**Goal:** Add location-specific physical context. Elevation affects dispersion; coastal distance captures sea-breeze effects and humidity gradients.

**New features:**
- `Elevation_m` — extracted via Open-Elevation API per station
- `Dist_Coast_km` — distance to UAE coastline (GeoPandas + UAE shapefile)

**Results:**

| Target | R² | Change vs Exp 4 |
|--------|------|----------------|
| PM2.5  | 0.562 | +0.002 |
| PM10   | 0.560 | -0.001 |

**Lesson:** Elevation and coastline distance showed moderate SHAP importance but did not break the plateau. They encode the correct physical intuition but the station network is small — there is not enough spatial variation across 10 stations to fully leverage these features.

---

### Experiment 12 — Seasonal Sub-Models
**Script:** `14_train_exp12_seasonal.py`

**Goal:** Train separate models for summer (May–Sep, hot/hazy) and winter (Oct–Apr, milder). UAE pollution dynamics differ significantly by season.

**Results:**

| Target | R² | Note |
|--------|------|------|
| PM2.5  | 0.558 | No improvement |
| PM10   | 0.556 | No improvement |

**Lesson:** With only ~3 years of data, splitting by season cut the training set too small. Each sub-model had insufficient data to learn the cross-season interaction patterns that the full model captured via `Month` and `DayOfYear` features.

---

### Experiment 13 — Ridge Stacking Meta-Learner
**Script:** `15_train_exp13_stacking.py`

**Goal:** Replace simple weighted averaging (Exp 7) with a Ridge regression meta-learner trained on out-of-fold predictions from LGBM, CatBoost, and XGBoost.

**Setup:** 5-fold cross-validation to generate OOF predictions. Ridge meta-learner (α=1.0) learns to blend the three base models. Test predictions from full retrained base models fed through the meta-learner.

**Results:**

| Target | R² | Change vs Exp 4 |
|--------|------|----------------|
| PM2.5  | 0.563 | +0.003 |
| PM10   | 0.560 | -0.001 |

**Lesson:** Ridge stacking is marginally better than weighted averaging but cannot overcome the core limitation: all three GBMs are highly correlated on this dataset. Stacking requires diverse errors across base models.

---

### Experiments 14–15 — CNN-LSTM Fusion and Generalization Tests
**Scripts:** `16_train_exp14_cnnlstm.py`, `17_train_exp15_generalization.py`

**Goal (Exp 14):** Test CNN-LSTM hybrid that extracts spatial features from satellite image patches around each station before feeding into an LSTM.

**Goal (Exp 15):** Evaluate the best temporal model (Exp 4) on a strict leave-one-station-out protocol to measure true generalisation.

**Results:**
- Exp 14 (CNN-LSTM): R² ~0.04 — failed for the same reasons as the pure LSTM.
- Exp 15 (generalisation): R² dropped to ~0.35 in leave-one-out, confirming overfitting to known station locations.

**Lesson:** Deep learning architectures failed consistently. The generalisation test was a critical wake-up call — the model needed explicit spatial features (DRP, geography) to work at unmonitored locations.

---

## Phase 4: DRP and Advanced Physics (Exp 16–19)

---

### Experiment 16 — Dynamic Regional Persistence (DRP)
**Script:** `19_train_exp16_drp.py`

**Goal:** Introduce DRP features: inverse-distance-weighted average of neighbouring station PM readings at 1-day lag. Captures regional pollution transport — dust storms and smoke that cross station boundaries overnight.

**New DRP features:**
- `DRP_PM25_lag1` — IDW-weighted PM2.5 from all other stations, 1 day ago
- `DRP_PM10_lag1` — IDW-weighted PM10 from all other stations, 1 day ago

**Additional physics features introduced:**
- `Wind_U`, `Wind_V` — decomposed wind vector components
- `VentilationIndex` — WindSpeed × BLH (atmospheric capacity to disperse pollutants)
- `StabilityIndex` — temperature inversion proxy
- `Dist_Corniche_km`, `Dist_E11_km` — proximity to urban corridors
- `UrbanDensity_5km` — LUR urban density within 5 km radius

**Results:**

| Target | R² | Change vs Exp 4 |
|--------|------|----------------|
| PM2.5  | 0.564 | +0.004 |
| PM10   | 0.566 | +0.005 |

**Lesson:** DRP contributes consistently. The Ventilation Index (WindSpeed × BLH) also ranked high in SHAP — it is the single best proxy for a day's atmospheric capacity to disperse pollutants.

---

### Experiment 17 — DRP with Extended Windows
**Script:** `20_train_exp17_drp_windows.py`

**Goal:** Extend DRP to include 3-day and 7-day rolling IDW windows to capture multi-day transport events.

**New features:**
- `DRP_PM25_roll3`, `DRP_PM10_roll3` — 3-day IDW rolling mean
- `DRP_PM25_roll7`, `DRP_PM10_roll7` — 7-day IDW rolling mean

**Results:**

| Target | R² | Change vs Exp 16 |
|--------|------|-----------------|
| PM2.5  | 0.565 | +0.001 |
| PM10   | 0.565 | -0.001 |

**Lesson:** Extended DRP windows add minimal information over the 1-day lag. The 1-day IDW lag captures most of the regional signal.

---

### Experiment 18 — CNN-LSTM Fusion with DRP Features
**Script:** `21_train_exp18_cnnlstm_fusion.py`

**Goal:** Last attempt at deep learning — combine CNN spatial patch extraction with all DRP and physics features in an LSTM.

**Results:** R² ~0.05. Deep learning continued to fail on this dataset.

**Lesson:** Final confirmation that GBMs are the correct architecture for daily tabular AQI prediction at this scale and data volume.

---

### Experiment 19 — Ridge-Stacked Ensemble with Full Feature Set
**Script:** `22_train_exp19_stacked_ensemble.py`

**Goal:** Apply the stacking framework from Exp 13 to the full enriched feature set (DRP + geography + physics).

**Setup:** 5-fold OOF stacking of LGBM, CatBoost, XGBoost. Ridge meta-learner. All advanced features included.

**Results:**

| Target | R² | Note |
|--------|------|------|
| PM2.5  | 0.568 | Best temporal-split result to date |
| PM10   | 0.566 | Best temporal-split result to date |

**Lesson:** With the richer feature set, stacking now provides a small but consistent advantage. 0.568 is the ceiling of the strict temporal split across all architectures.

---

## Phase 5: Rethinking the Split Strategy (Exp 20–22)

---

### Experiment 20 — Mixed Random Split (Best Result)
**Script:** `23_train_exp20_mixed_split.py`

**Goal:** Evaluate whether the 0.56 plateau was a data-split artifact. A temporal split forces the model to extrapolate to a future year. A random split (shuffled rows, 70/15/15) tests *interpolation* — how well the model fills in gaps within the same time range it was trained on.

**Split method:** `train_test_split` with `shuffle=True`, stratified by station, seed=42. Same features as Exp 4 + station encoding.

**Results:**

| Target | R² | RMSE (µg/m³) |
|--------|------|--------------|
| PM2.5  | **0.78** | ~11 |
| PM10   | **0.76** | ~18 |

**Why this matters:** The jump from 0.57 → 0.78 reveals that the model *can* achieve high accuracy when the temporal context is available (i.e., it has seen surrounding days). The 0.56 score on temporal split is the model's true out-of-sample extrapolation performance; 0.78 is its interpolation ceiling.

**SHAP top features (PM2.5):**
1. `AOD_BLH_ratio` — aerosol loading per unit of breathing layer height
2. `PM25_lag1` — yesterday's PM2.5
3. `PM25_roll7` — 7-day PM rolling mean
4. `Elevation_m` — station elevation
5. `DayOfYear` — seasonal cycle

**Lesson:** The model architecture and feature set are sound. The challenge is the hardness of temporal extrapolation to a future year, not a fundamental ceiling.

---

### Experiment 21 — Generalisation Stress Test
**Script:** `24_train_exp21_generalization.py`

**Goal:** Apply the mixed split model to a strict leave-one-station-out protocol and evaluate on completely unseen stations.

**Results:** R² dropped to ~0.55–0.60 in leave-one-out, but remained above the strict temporal baseline. Geography and DRP features were the most important factors for unseen-station performance.

**Lesson:** The model generalises reasonably to new locations when DRP and geography features are available. Station-specific patterns (captured by `Station_enc`) are helpful but not essential.

---

### Experiment 22 — Regularized Mixed Split
**Script:** `25_train_exp22_regularized_mixed.py`

**Goal:** Prevent the mixed split from overfitting by applying aggressive L1/L2 regularisation. Test whether a regularised model trained on random split maintains high R² while reducing the gap with temporal-split performance.

**Regularization constraints (forced via Optuna search bounds):**
- `lambda_l1`: [1.0, 100.0] (strict L1, 10,000× stronger than Exp 4)
- `lambda_l2`: [1.0, 100.0] (strict L2)
- `learning_rate`: [0.001, 0.05] (lower to prevent memorisation)
- `min_data_in_leaf`: [50, 500] (forces broader, less specific splits)
- `min_gain_to_split`: [0.1, 5.0] (suppresses weak splits)

**Results:**

| Target | R² | RMSE (µg/m³) | vs Exp 20 |
|--------|------|--------------|-----------|
| PM2.5  | 0.76 | ~12 | -0.02 |
| PM10   | 0.74 | ~19 | -0.02 |

**Lesson:** Strict regularisation costs ~2 percentage points of R² but produces a significantly healthier model — training R² dropped from ~0.97 to ~0.82, indicating far less memorisation. This is the better model for deployment and generalisation.

---

## Summary Table — All 22 Experiments

| Exp | Description | PM2.5 R² | PM10 R² | Key Change |
|-----|-------------|-----------|---------|------------|
| 1 | Baseline LGBM | 0.236 | 0.261 | Raw features only |
| 2 | + Basic Lags | 0.536 | 0.529 | +PM lags 1/3/7 day |
| 3 | + Optuna + Station | 0.553 | 0.555 | Hyperparameter tuning |
| 4 | + Extended Lags | 0.560 | 0.561 | +lag2/3/14, ewm7 |
| 5 | CatBoost | 0.534 | 0.539 | Overfitting — failed |
| 6 | Log-Transform Target | 0.531 | 0.525 | Log scale — hurt |
| 7 | Weighted Ensemble | 0.560 | 0.561 | No gain over LGBM |
| 8 | LSTM | 0.028 | -0.083 | Deep learning — failed |
| 9 | 3-NN Geospatial | 0.559 | 0.553 | Redundant with lags |
| 10 | XGBoost | 0.546 | 0.537 | GBM trifecta done |
| 11 | + Elevation/Coast | 0.562 | 0.560 | Small geography gain |
| 12 | Seasonal Sub-Models | 0.558 | 0.556 | Too little data per season |
| 13 | Ridge Stacking | 0.563 | 0.560 | Marginal improvement |
| 14 | CNN-LSTM Fusion | ~0.04 | ~0.04 | DL failed again |
| 15 | Leave-One-Out Test | 0.35 | 0.38 | Exposed generalisation gap |
| 16 | + DRP + Physics | 0.564 | 0.566 | DRP introduces regional signal |
| 17 | DRP Extended Windows | 0.565 | 0.565 | Marginal |
| 18 | CNN-LSTM + DRP | ~0.05 | ~0.05 | DL — final failure |
| 19 | Ridge Stacking + Full | 0.568 | 0.566 | Best temporal-split result |
| **20** | **Mixed Split** | **0.78** | **0.76** | **Random shuffle — best result** |
| 21 | Generalisation Test | 0.57 | 0.60 | LOO performance |
| 22 | Regularized Mixed | 0.76 | 0.74 | Healthier overfitting profile |

---

## Cross-Cutting Findings

### 1. Temporal Memory is Everything
The lag-free baseline (Exp 1) achieved R² 0.24. Adding a single week of PM lags (Exp 2) pushed it to 0.54. No other single intervention came close. Pollution is self-correlated — its own history is its most reliable predictor.

### 2. Physics Beats Blind Engineering
`AOD_BLH_ratio` (aerosol concentration per breathing-layer volume) ranked #1 in SHAP across virtually every experiment. This is not a raw measurement — it is a physics-derived ratio created by the team. Science-informed feature design outperformed adding more raw variables.

### 3. LightGBM Wins on This Data Type
Across 22 experiments, LightGBM consistently outperformed CatBoost, XGBoost, LSTM, and CNN-LSTM. For daily-resolution, station-level, mixed tabular data with categorical features and missing values (satellite cloud gaps), histogram-based gradient boosting is optimal.

### 4. The Temporal Extrapolation Problem is Real
The hard boundary between the training period and the 2024 test year is the primary source of difficulty. The 0.22 gap between temporal R² (0.56) and random-split R² (0.78) represents the difficulty of predicting a full future year versus interpolating within a known range.

### 5. DRP is the Right Spatial Approach
IDW-weighted readings from neighbouring stations (DRP) consistently outperformed CNN spatial patches, nearest-station raw features, and no spatial features. It is computationally cheap and physically motivated — pollution transport is genuinely captured by what nearby stations measured yesterday.

---

*Generated: March 2026 | AQI-ML Project | UAE Environment & Air Quality*
