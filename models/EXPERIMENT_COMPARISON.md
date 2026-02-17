# Experiment Comparison: Exp1 (Baseline) vs Exp2 (Improved)

## 🎯 Results Summary

### PM2.5 Model Performance

| Metric | Exp1 (Baseline RF) | Exp2 (XGBoost + Tuning) | **Improvement** |
|--------|-------------------|------------------------|----------------|
| **Test R²** | 0.4490 | **0.5421** | **+20.7%** ✓ |
| **Test RMSE** | 16.89 µg/m³ | **15.40 µg/m³** | **-8.8%** ✓ |
| **Test MAE** | 10.77 µg/m³ | **9.57 µg/m³** | **-11.1%** ✓ |

### PM10 Model Performance

| Metric | Exp1 (Baseline RF) | Exp2 (XGBoost + Tuning) | **Improvement** |
|--------|-------------------|------------------------|----------------|
| **Test R²** | 0.5471 | **0.6508** | **+19.0%** ✓ |
| **Test RMSE** | 48.67 µg/m³ | **42.74 µg/m³** | **-12.2%** ✓ |
| **Test MAE** | 31.07 µg/m³ | **26.81 µg/m³** | **-13.7%** ✓ |

---

## ✅ SUCCESS! Significant Improvements Achieved

### Key Wins:

1. **PM2.5 R² improved from 0.45 → 0.54** (+20% improvement)
   - Now explains 54% of variance (vs 45% before)
   - Prediction error reduced by ~1.5 µg/m³

2. **PM10 R² improved from 0.55 → 0.65** (+19% improvement)
   - Now explains 65% of variance (vs 55% before)
   - Prediction error reduced by ~6 µg/m³
   - **R² > 0.65 is excellent** for air quality!

3. **Both models chose XGBoost** over Random Forest
   - Grid search found optimal hyperparameters
   - Lag features contributed significantly

---

## 📊 What Changed (Exp2 Improvements)

### 1. Algorithm Upgrade
- **Exp1**: Basic Random Forest (200 trees, depth 20)
- **Exp2**: XGBoost with GridSearch tuning
  - Best params: 300 trees, depth 10, lr=0.05

### 2. Feature Engineering
- **Exp1**: 23 features
- **Exp2**: 28 features (added lag + rolling features)
  - PM_lag1 (yesterday's values)
  - PM_rolling3, PM_rolling7 (temporal trends)
  - AOD_rolling3, NO2_rolling3

### 3. Hyperparameter Optimization
- **Exp1**: Default parameters
- **Exp2**: GridSearch over 48 combinations

---

## 🎓 Assessment

**PM2.5 Model**:
- R² = 0.54 is **solid** for satellite-based PM2.5
- Published research typically: 0.4-0.7
- You're in the **upper-middle range**! ✓

**PM10 Model**:
- R² = 0.65 is **excellent**!
- Approaching state-of-the-art for this type of model
- PM10 harder to predict globally, this is impressive ✓

---
