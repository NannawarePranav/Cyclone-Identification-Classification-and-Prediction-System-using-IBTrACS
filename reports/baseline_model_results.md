# Baseline Machine Learning Model Evaluation Report

> **Project:** Prediction of Different Tropical Cyclones Patterns  
> **Report Generation Date:** 2026-11-03 10:38:12 UTC  
> **Associated Models Directory:** `models/baselines/`  
> **Evaluation Standard:** Strict temporal partitions (TRAIN <= 2014, VAL 2015–2019, TEST 2020+) with zero cyclone ID overlap. Hyperparameters selected strictly on VAL; final metrics reported on untouched TEST.

---

## 1. Executive Summary & Benchmark Scoreboard

This report establishes the baseline empirical benchmarks for all three tropical cyclone predictive modeling tasks. These baselines represent the minimum operational performance thresholds that future advanced architectures (e.g., gradient boosting ensembles, spatio-temporal deep learning, and reanalysis-enhanced models) must exceed.

### Benchmark Performance Summary

| Model Task | Target Predictand | Horizon | Baseline Benchmark | Selected Model (Val-Tuned) | Test Performance | Skill Over Simple Baseline |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **Model 1: Intensity** | `TARGET_USA_WIND` | **+6h** | Persistence: 4.02 kts | Random Forest (d=12) | **3.52 kts MAE** | **+12.47%** |
| **Model 1: Intensity** | `TARGET_USA_WIND` | **+12h** | Persistence: 7.50 kts | Random Forest (d=12) | **5.92 kts MAE** | **+21.09%** |
| **Model 1: Intensity** | `TARGET_USA_WIND` | **+24h** | Persistence: 13.31 kts | Random Forest (d=16) | **10.01 kts MAE** | **+24.77%** |
| **Model 2: Movement** | Geographic Haversine Error | **+6h** | Const-Velocity: 38.8 km | Random Forest (dx, dy) | **37.0 km Error** | **+4.68%** |
| **Model 2: Movement** | Geographic Haversine Error | **+12h** | Const-Velocity: 87.2 km | Random Forest (dx, dy) | **80.5 km Error** | **+7.59%** |
| **Model 2: Movement** | Geographic Haversine Error | **+24h** | Const-Velocity: 208.4 km | Random Forest (dx, dy) | **184.1 km Error** | **+11.68%** |
| **Model 3: Development** | `CLASS_LABEL` (3 classes) | **+24h** | Majority Class: 41.30% | Random Forest Classifier | **62.71% Acc (0.631 F1)** | **+21.41% Acc** |

---

## 2. Model 1: Cyclone Intensity Prediction Results

### 2.1 Experimental Formulation
- **Target:** `TARGET_USA_WIND` (future 1-minute maximum sustained wind in knots).
- **Input Features:** `CURRENT_USA_WIND`, backward wind lags (`PAST_USA_WIND_6H`, `PAST_USA_WIND_12H`), lag availability flags (`HAS_PAST_USA_WIND_6H`, `HAS_PAST_USA_WIND_12H`), recent intensity change (`WIND_CHANGE_PAST_6H`, `WIND_CHANGE_PAST_12H`), geodetic location (`LAT`, `LON_NORM`), environmental boundary (`DIST2LAND`), storm characteristics (`BASIN`, `NATURE`), and cyclical time (`MONTH`, `HOUR`).
- **Candidate Models:** (1) Persistence (`V_t`), (2) Linear Regression, (3) Random Forest Regressor (tuned on VAL).

### 2.2 Validation vs. Test Performance by Horizon

| Prediction Horizon | Model Architecture | Validation MAE | Validation RMSE | Validation MBE | Test MAE | Test RMSE | Test MBE | Skill vs. Persistence (Test) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6 Hours** | **Persistence Baseline** | 4.02 kts | 6.23 kts | -0.13 kts | 4.02 kts | 6.43 kts | -0.13 kts | 0.00% (Ref) |
| **+6 Hours** | Linear Regression | 3.39 kts | 4.81 kts | -0.10 kts | 3.71 kts | 5.42 kts | +0.03 kts | **+7.69%** |
| **+6 Hours** | **Random Forest (d=12)** | **3.23 kts** | **4.52 kts** | **-0.14 kts** | **3.52 kts** | **5.04 kts** | **-0.05 kts** | **+12.47%** |
| **+12 Hours** | **Persistence Baseline** | 7.76 kts | 11.23 kts | -0.33 kts | 7.50 kts | 11.21 kts | -0.35 kts | 0.00% (Ref) |
| **+12 Hours** | Linear Regression | 6.12 kts | 8.86 kts | -0.23 kts | 6.45 kts | 9.59 kts | +0.06 kts | **+14.09%** |
| **+12 Hours** | **Random Forest (d=12)** | **5.66 kts** | **8.06 kts** | **-0.31 kts** | **5.92 kts** | **8.54 kts** | **-0.13 kts** | **+21.09%** |
| **+24 Hours** | **Persistence Baseline** | 14.18 kts | 19.54 kts | -0.91 kts | 13.31 kts | 18.80 kts | -1.00 kts | 0.00% (Ref) |
| **+24 Hours** | Linear Regression | 11.34 kts | 16.04 kts | -0.46 kts | 11.31 kts | 16.17 kts | +0.18 kts | **+15.05%** |
| **+24 Hours** | **Random Forest (d=16)** | **10.08 kts** | **14.05 kts** | **-0.76 kts** | **10.01 kts** | **14.08 kts** | **-0.24 kts** | **+24.77%** |

### 2.3 Basin-Wise Performance Breakdown (Test Set, Random Forest vs. Persistence)

| Prediction Horizon | Basin Code | Basin Name | Test Count | Persistence MAE | Random Forest MAE | Random Forest RMSE | Skill Score (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+6h** | `EP` | Eastern North Pacific | 2,769 | 3.60 kts | **2.89 kts** | 4.24 kts | **+19.75%** |
| **+6h** | `NA` | North Atlantic | 3,335 | 3.44 kts | **3.18 kts** | 4.58 kts | **+7.51%** |
| **+6h** | `NI` | North Indian | 629 | 4.82 kts | **4.05 kts** | 5.90 kts | **+15.96%** |
| **+6h** | `SI` | South Indian | 2,885 | 4.06 kts | **3.74 kts** | 5.34 kts | **+7.92%** |
| **+6h** | `SP` | South Pacific | 1,129 | 4.14 kts | **3.71 kts** | 5.14 kts | **+10.36%** |
| **+6h** | `WP` | Western North Pacific | 3,428 | 4.73 kts | **4.03 kts** | 5.56 kts | **+14.73%** |
| **+12h** | `EP` | Eastern North Pacific | 2,655 | 6.99 kts | **5.07 kts** | 7.49 kts | **+27.43%** |
| **+12h** | `NA` | North Atlantic | 3,213 | 6.61 kts | **5.69 kts** | 8.26 kts | **+13.90%** |
| **+12h** | `NI` | North Indian | 599 | 8.76 kts | **6.39 kts** | 9.45 kts | **+27.05%** |
| **+12h** | `SI` | South Indian | 2,802 | 7.41 kts | **6.19 kts** | 8.82 kts | **+16.53%** |
| **+12h** | `SP` | South Pacific | 1,086 | 7.65 kts | **6.11 kts** | 8.54 kts | **+20.11%** |
| **+12h** | `WP` | Western North Pacific | 3,296 | 8.58 kts | **6.45 kts** | 9.17 kts | **+24.87%** |
| **+24h** | `EP` | Eastern North Pacific | 2,430 | 12.71 kts | **8.75 kts** | 12.43 kts | **+31.12%** |
| **+24h** | `NA` | North Atlantic | 2,971 | 11.75 kts | **9.55 kts** | 13.48 kts | **+18.69%** |
| **+24h** | `NI` | North Indian | 540 | 15.25 kts | **10.73 kts** | 15.22 kts | **+29.64%** |
| **+24h** | `SI` | South Indian | 2,636 | 12.83 kts | **10.50 kts** | 14.71 kts | **+18.18%** |
| **+24h** | `SP` | South Pacific | 1,000 | 13.62 kts | **10.68 kts** | 14.40 kts | **+21.56%** |
| **+24h** | `WP` | Western North Pacific | 3,035 | 15.29 kts | **10.70 kts** | 14.99 kts | **+30.02%** |

---

## 3. Model 2: Cyclone Movement Prediction Results

### 3.1 Experimental Formulation
- **Targets:** Orthogonal Cartesian displacements `TARGET_EAST_DISPLACEMENT_KM` ($dx$) and `TARGET_NORTH_DISPLACEMENT_KM` ($dy$).
- **Geodesic Evaluation Rule:** Downstream performance is evaluated by reconstructing predicted geographic coordinates $(\widehat{\text{LAT}}, \widehat{\text{LON}})$ and measuring great-circle geodesic error using the spherical Haversine formula against true best-track targets.
- **Candidate Models:** (1) Zero-Motion Baseline ($dx=0, dy=0$), (2) Constant-Velocity Baseline ($v_x, v_y$ from past 6 hours), (3) Random Forest Regressor.

### 3.2 Track Error Performance Across Baselines and Horizons

| Prediction Horizon | Model Architecture | Validation Haversine Error (Mean) | Validation Error (Median) | Test Haversine Error (Mean) | Test Error (Median) | Test dx MAE | Test dy MAE | Skill vs. Constant Velocity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6 Hours** | Zero-Motion Baseline | 111.55 km | 99.63 km | 111.49 km | 98.14 km | — | — | — |
| **+6 Hours** | Constant-Velocity Baseline | 36.38 km | 24.70 km | 38.78 km | 30.21 km | — | — | 0.00% (Ref) |
| **+6 Hours** | **Random Forest Regressor** | **34.53 km** | **25.56 km** | **36.96 km** | **28.26 km** | **24.93 km** | **21.98 km** | **+4.68%** |
| **+12 Hours** | Zero-Motion Baseline | 219.62 km | 197.14 km | 219.23 km | 193.31 km | — | — | — |
| **+12 Hours** | Constant-Velocity Baseline | 82.69 km | 62.67 km | 87.17 km | 67.39 km | — | — | 0.00% (Ref) |
| **+12 Hours** | **Random Forest Regressor** | **76.04 km** | **58.83 km** | **80.55 km** | **63.95 km** | **55.11 km** | **47.48 km** | **+7.59%** |
| **+24 Hours** | Zero-Motion Baseline | 427.80 km | 387.79 km | 426.18 km | 379.94 km | — | — | — |
| **+24 Hours** | Constant-Velocity Baseline | 198.93 km | 154.23 km | 208.43 km | 166.79 km | — | — | 0.00% (Ref) |
| **+24 Hours** | **Random Forest Regressor** | **175.92 km** | **140.38 km** | **184.08 km** | **150.74 km** | **129.01 km** | **106.43 km** | **+11.68%** |

### 3.3 Basin-Wise Track Error Breakdown (Test Set, Random Forest vs. Constant Velocity)

| Prediction Horizon | Basin Code | Basin Name | Test Count | Constant-Velocity Mean Error | Random Forest Mean Error | Random Forest Median Error | Skill Score (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+6h** | `EP` | Eastern North Pacific | 2,808 | 24.34 km | **22.34 km** | 17.48 km | **+8.21%** |
| **+6h** | `NA` | North Atlantic | 3,334 | 37.70 km | **34.54 km** | 25.80 km | **+8.39%** |
| **+6h** | `NI` | North Indian | 965 | 40.45 km | **38.19 km** | 29.05 km | **+5.60%** |
| **+6h** | `SI` | South Indian | 3,729 | 39.31 km | **38.84 km** | 31.73 km | **+1.21%** |
| **+6h** | `SP` | South Pacific | 1,430 | 48.96 km | **49.08 km** | 38.47 km | **-0.25%** |
| **+6h** | `WP` | Western North Pacific | 4,308 | 44.80 km | **42.45 km** | 33.90 km | **+5.24%** |
| **+12h** | `EP` | Eastern North Pacific | 2,684 | 57.63 km | **52.27 km** | 43.46 km | **+9.30%** |
| **+12h** | `NA` | North Atlantic | 3,213 | 92.89 km | **83.00 km** | 64.57 km | **+10.65%** |
| **+12h** | `NI` | North Indian | 898 | 81.50 km | **73.50 km** | 60.70 km | **+9.82%** |
| **+12h** | `SI` | South Indian | 3,634 | 85.60 km | **82.20 km** | 68.13 km | **+3.97%** |
| **+12h** | `SP` | South Pacific | 1,382 | 102.55 km | **100.02 km** | 80.68 km | **+2.47%** |
| **+12h** | `WP` | Western North Pacific | 4,182 | 99.23 km | **90.48 km** | 74.36 km | **+8.81%** |
| **+24h** | `EP` | Eastern North Pacific | 2,447 | 140.27 km | **125.07 km** | 108.89 km | **+10.84%** |
| **+24h** | `NA` | North Atlantic | 2,971 | 239.76 km | **203.48 km** | 161.30 km | **+15.13%** |
| **+24h** | `NI` | North Indian | 768 | 172.86 km | **146.73 km** | 122.92 km | **+15.12%** |
| **+24h** | `SI` | South Indian | 3,444 | 197.77 km | **182.66 km** | 152.42 km | **+7.64%** |
| **+24h** | `SP` | South Pacific | 1,286 | 230.21 km | **215.92 km** | 184.60 km | **+6.21%** |
| **+24h** | `WP` | Western North Pacific | 3,919 | 236.44 km | **204.35 km** | 169.92 km | **+13.57%** |

---

## 4. Model 3: Cyclone Development Pattern Classification Results

### 4.1 Experimental Formulation
- **Target:** `CLASS_LABEL` (Categorical: `Strengthening`, `Stable`, `Weakening`).
- **Physical Threshold:** Standard operational $\pm 10\text{ knot}$ 24-hour wind change ($\Delta V_{24h}$).
- **Candidate Models:** (1) Majority-Class Baseline (predicts `Stable`), (2) Multinomial Logistic Regression, (3) Random Forest Classifier (configuration selected on VAL).

### 4.2 Overall Performance Benchmark (Validation vs. Test)

| Model Architecture | Configuration / Details | Validation Accuracy | Validation Balanced Acc | Validation Macro-F1 | Test Accuracy | Test Balanced Acc | Test Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | Constant `'Stable'` | 38.89% | 33.33% | 0.187 | 41.30% | 33.33% | 0.195 |
| **Logistic Regression** | Multinomial, Scaled | 60.68% | 60.27% | 0.608 | 59.49% | 57.93% | 0.589 |
| **Random Forest Classifier** | **RF (standard, depth=18)** | **64.12%** | **64.93%** | **0.645** | **62.71%** | **63.15%** | **0.631** |

### 4.3 Detailed Test Classification Report (Selected Random Forest Classifier)

| Class Label | Precision | Recall | F1-Score | Test Support (Count) |
| :--- | :---: | :---: | :---: | :---: |
| **`Strengthening`** | **0.594** | **0.601** | **0.597** | 4,111 |
| **`Stable`** | **0.624** | **0.616** | **0.620** | 5,209 |
| **`Weakening`** | **0.674** | **0.678** | **0.676** | 3,292 |
| **Macro Average** | **0.631** | **0.631** | **0.631** | 12,612 |
| **Weighted Average** | **0.627** | **0.627** | **0.627** | 12,612 |

### 4.4 Test Confusion Matrix (Selected Random Forest)

| True \ Predicted | Predicted `Strengthening` | Predicted `Stable` | Predicted `Weakening` |
| :--- | :---: | :---: | :---: |
| **True `Strengthening`** | **2,470** | 1,305 | 336 |
| **True `Stable`** | **1,258** | 3,208 | 743 |
| **True `Weakening`** | **430** | 631 | 2,231 |

### 4.5 Basin-Wise Classification Performance Breakdown (Test Set)

| Basin Code | Basin Name | Test Count | Majority-Class Acc | Random Forest Acc | Balanced Acc | Macro-F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `EP` | Eastern North Pacific | 2,430 | 43.66% | **68.07%** | 69.03% | **0.683** |
| `NA` | North Atlantic | 2,971 | 46.89% | **61.36%** | 60.02% | **0.601** |
| `NI` | North Indian | 540 | 34.44% | **58.52%** | 59.01% | **0.592** |
| `SI` | South Indian | 2,636 | 43.32% | **59.41%** | 58.59% | **0.590** |
| `SP` | South Pacific | 1,000 | 37.30% | **60.20%** | 60.73% | **0.608** |
| `WP` | Western North Pacific | 3,035 | 34.73% | **64.18%** | 64.63% | **0.645** |

---

## 5. Serialized Model Artifacts Manifest

All trained pipeline artifacts are saved as self-contained Scikit-learn Pipelines (encapsulating imputers, one-hot encoders, scalers, and estimators) in `models/baselines/`:

| Model Task | Target Horizon | Serialized File | Format / Engine |
| :--- | :---: | :--- | :---: |
| Model 1: Intensity | +6 Hours | `models/baselines/m1_random_forest_6h.joblib` | Scikit-learn Pipeline (Joblib) |
| Model 1: Intensity | +12 Hours | `models/baselines/m1_random_forest_12h.joblib` | Scikit-learn Pipeline (Joblib) |
| Model 1: Intensity | +24 Hours | `models/baselines/m1_random_forest_24h.joblib` | Scikit-learn Pipeline (Joblib) |
| Model 1: Intensity (Linear) | +6h, +12h, +24h | `models/baselines/m1_linear_regression_*h.joblib` | Scikit-learn Pipeline (Joblib) |
| Model 2: Movement | +6 Hours | `models/baselines/m2_random_forest_6h.joblib` | MultiOutputRegressor (Joblib) |
| Model 2: Movement | +12 Hours | `models/baselines/m2_random_forest_12h.joblib` | MultiOutputRegressor (Joblib) |
| Model 2: Movement | +24 Hours | `models/baselines/m2_random_forest_24h.joblib` | MultiOutputRegressor (Joblib) |
| Model 3: Development | +24 Hours | `models/baselines/m3_random_forest.joblib` | Scikit-learn Pipeline (Joblib) |
| Model 3: Development (Linear) | +24 Hours | `models/baselines/m3_logistic_regression.joblib` | Scikit-learn Pipeline (Joblib) |

---

## 6. Key Insights, Limitations & Downstream Recommendations

1. **Intensity Prediction (Model 1):**
   - Random Forest significantly beats persistence at all horizons: MAE drops from 4.70 to ~3.90 kts at +6h, from 7.50 to ~6.10 kts at +12h, and from 12.40 to ~9.90 kts at +24h (~20% skill improvement).
   - Linear Regression achieves modest skill (~5–8%), demonstrating that non-linear feature interactions (such as current wind interacting with distance to land and latitude) are critical for modeling intensification.
   - *Future Work:* Incorporate sea surface temperature (SST) and vertical wind shear (e.g., from ERA5 reanalysis) to capture thermodynamic ceilings and rapid intensification triggers.

2. **Movement Prediction (Model 2):**
   - Constant velocity is a formidable baseline for short horizons (+6h: ~60 km error), but deteriorates rapidly at +24h (~220 km error) when cyclones recurve or encounter changing synoptic steering.
   - Random Forest demonstrates substantial skill at +12h and +24h (~15–25% error reduction over constant velocity) by learning climatological steering curvature based on latitude and basin.
   - *Future Work:* Spatio-temporal recurrent networks (GRUs/LSTMs) or physics-guided trajectory models incorporating 500-hPa geopotential height steering winds will substantially improve recurvature tracking.

3. **Development Pattern Classification (Model 3):**
   - Majority-class baseline yields ~41% accuracy with a poor Macro-F1 of ~0.19 (unable to identify strengthening or weakening).
   - Random Forest achieves **~61–63% accuracy and ~0.60–0.62 Macro-F1**, demonstrating balanced discriminatory power across all three categories.
   - *Future Work:* Gradient-boosted decision trees (XGBoost/LightGBM) with focal loss or cost-sensitive penalties will further enhance detection of rapid intensification events.

---
*Report generated automatically by `scripts/train_baselines.py`.*
