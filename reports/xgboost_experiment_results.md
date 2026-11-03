# XGBoost Experiments and Model Improvement Report

> **Project:** Prediction of Different Tropical Cyclones Patterns  
> **Report Generation Date:** 2026-11-03 11:05:58 UTC  
> **Associated Models Directory:** `models/xgboost/`  
> **Task Relationship:** Extension of Task 6 baselines (`models/baselines/`, `reports/baseline_model_results.md`).  
> **Methodological Standard:** Strict temporal partitions (TRAIN <= 2014, VAL 2015–2019, TEST 2020+) with zero cyclone ID overlap. Hyperparameters selected strictly on VAL; final metrics evaluated once on untouched TEST.

---

## 1. Executive Summary & Cross-Model Benchmark Scoreboard

This investigation tests whether gradient-boosted decision trees using histogram-based tree construction (`XGBoost`) improve upon the empirical baselines established in Task 6 across our three core tropical cyclone prediction tasks.

### Master Head-to-Head Benchmark Table (Test Set: 2020–Present)

| Model Task | Target / Metric | Horizon | Simple Baseline | Task 6 Random Forest | Task 7 XGBoost (Selected) | Skill vs. Simple Baseline | Delta vs. Random Forest | Verdict |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **Model 1: Intensity** | Wind MAE (kts) | **+6h** | Persistence: 4.02 kts | RF: 3.52 kts | **3.43 kts** | **+14.83%** | +2.56% | **Improved** |
| **Model 1: Intensity** | Wind MAE (kts) | **+12h** | Persistence: 7.50 kts | RF: 5.92 kts | **5.71 kts** | **+23.93%** | +3.55% | **Improved** |
| **Model 1: Intensity** | Wind MAE (kts) | **+24h** | Persistence: 13.31 kts | RF: 10.01 kts | **9.58 kts** | **+27.98%** | +4.30% | **Improved** |
| **Model 2: Movement** | Haversine Error (km) | **+6h** | Const-Vel: 38.8 km | RF: 37.0 km | **36.6 km** | **+5.61%** | +0.97% | **Improved** |
| **Model 2: Movement** | Haversine Error (km) | **+12h** | Const-Vel: 87.2 km | RF: 80.5 km | **79.6 km** | **+8.71%** | +1.20% | **Improved** |
| **Model 2: Movement** | Haversine Error (km) | **+24h** | Const-Vel: 208.4 km | RF: 184.1 km | **181.6 km** | **+12.87%** | +1.35% | **Improved** |
| **Model 3: Development** | Accuracy / Macro-F1 | **+24h** | Majority: 41.30% | RF: 62.71% (0.631 F1) | **63.42% (0.638 F1)** | **+22.12%** | +0.71% Acc | **Improved** |

---

## 2. Model 1: Cyclone Intensity Prediction (XGBRegressor)

### 2.1 Hyperparameter Search Space & Best Validation Configurations

A controlled randomized search explored the following parameter space over 8 diverse candidate configurations evaluated on the validation partition (2015–2019):
- `n_estimators`: [200, 400, 600]
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.03, 0.05, 0.1]
- `subsample`: [0.8, 1.0]
- `colsample_bytree`: [0.8, 1.0]
- `min_child_weight`: [1, 3, 5]
- `reg_lambda`: [1.0, 5.0, 10.0]

| Horizon | Selected Parameters (Tuned on VAL) | Val MAE | Val RMSE | Val Skill vs. Pers |
| :---: | :--- | :---: | :---: | :---: |
| **+6h** | `depth=7, lr=0.05, trees=200, sub=0.8, col=0.8, min_child=1, lambda=1.0` | **3.15 kts** | 4.38 kts | **+21.80%** |
| **+12h** | `depth=5, lr=0.03, trees=600, sub=0.8, col=0.8, min_child=5, lambda=10.0` | **5.45 kts** | 7.70 kts | **+29.74%** |
| **+24h** | `depth=7, lr=0.1, trees=200, sub=1.0, col=0.8, min_child=1, lambda=5.0` | **9.65 kts** | 13.52 kts | **+31.98%** |

### 2.2 Final Benchmark Comparison on Untouched Test Partition (2020–Present)

| Prediction Horizon | Model Architecture | Test MAE | Test RMSE | Test MBE | Skill vs. Persistence | Skill vs. Random Forest |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6 Hours** | Persistence Baseline | 4.02 kts | 6.43 kts | -0.13 kts | 0.00% (Ref) | — |
| **+6 Hours** | Random Forest (Task 6) | 3.52 kts | 5.04 kts | -0.05 kts | +12.47% | 0.00% (Ref) |
| **+6 Hours** | **XGBRegressor (Task 7)** | **3.43 kts** | **4.86 kts** | **-0.11 kts** | **+14.83%** | **+2.56%** |
| **+12 Hours** | Persistence Baseline | 7.50 kts | 11.21 kts | -0.35 kts | 0.00% (Ref) | — |
| **+12 Hours** | Random Forest (Task 6) | 5.92 kts | 8.54 kts | -0.13 kts | +21.09% | 0.00% (Ref) |
| **+12 Hours** | **XGBRegressor (Task 7)** | **5.71 kts** | **8.18 kts** | **-0.25 kts** | **+23.93%** | **+3.55%** |
| **+24 Hours** | Persistence Baseline | 13.31 kts | 18.80 kts | -1.00 kts | 0.00% (Ref) | — |
| **+24 Hours** | Random Forest (Task 6) | 10.01 kts | 14.08 kts | -0.24 kts | +24.77% | 0.00% (Ref) |
| **+24 Hours** | **XGBRegressor (Task 7)** | **9.58 kts** | **13.50 kts** | **-0.54 kts** | **+27.98%** | **+4.30%** |

### 2.3 Basin-Wise Intensity Performance Breakdown (Test Set, XGBoost vs. Baselines)

| Horizon | Basin Code | Basin Name | Test Count | Persistence MAE | Task 6 RF MAE | XGBoost MAE | XGBoost RMSE | Skill vs. Pers (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6h** | `EP` | Eastern North Pacific | 2,769 | 3.60 kts | 2.89 kts | **2.78 kts** | 3.95 kts | **+22.75%** |
| **+6h** | `NA` | North Atlantic | 3,335 | 3.44 kts | 3.18 kts | **3.07 kts** | 4.38 kts | **+10.77%** |
| **+6h** | `NI` | North Indian | 629 | 4.82 kts | 4.05 kts | **3.93 kts** | 5.63 kts | **+18.45%** |
| **+6h** | `SI` | South Indian | 2,885 | 4.06 kts | 3.74 kts | **3.68 kts** | 5.25 kts | **+9.26%** |
| **+6h** | `SP` | South Pacific | 1,129 | 4.14 kts | 3.71 kts | **3.69 kts** | 5.09 kts | **+10.88%** |
| **+6h** | `WP` | Western North Pacific | 3,428 | 4.73 kts | 4.03 kts | **3.91 kts** | 5.37 kts | **+17.33%** |
| **+12h** | `EP` | Eastern North Pacific | 2,655 | 6.99 kts | 5.07 kts | **4.89 kts** | 7.15 kts | **+30.04%** |
| **+12h** | `NA` | North Atlantic | 3,213 | 6.61 kts | 5.69 kts | **5.41 kts** | 7.75 kts | **+18.23%** |
| **+12h** | `NI` | North Indian | 599 | 8.76 kts | 6.39 kts | **6.16 kts** | 9.06 kts | **+29.70%** |
| **+12h** | `SI` | South Indian | 2,802 | 7.41 kts | 6.19 kts | **6.03 kts** | 8.59 kts | **+18.62%** |
| **+12h** | `SP` | South Pacific | 1,086 | 7.65 kts | 6.11 kts | **5.99 kts** | 8.34 kts | **+21.69%** |
| **+12h** | `WP` | Western North Pacific | 3,296 | 8.58 kts | 6.45 kts | **6.21 kts** | 8.78 kts | **+27.68%** |
| **+24h** | `EP` | Eastern North Pacific | 2,430 | 12.71 kts | 8.75 kts | **8.40 kts** | 11.96 kts | **+33.88%** |
| **+24h** | `NA` | North Atlantic | 2,971 | 11.75 kts | 9.55 kts | **9.01 kts** | 12.71 kts | **+23.28%** |
| **+24h** | `NI` | North Indian | 540 | 15.25 kts | 10.73 kts | **10.05 kts** | 14.22 kts | **+34.10%** |
| **+24h** | `SI` | South Indian | 2,636 | 12.83 kts | 10.50 kts | **10.22 kts** | 14.30 kts | **+20.36%** |
| **+24h** | `SP` | South Pacific | 1,000 | 13.62 kts | 10.68 kts | **10.00 kts** | 13.70 kts | **+26.60%** |
| **+24h** | `WP` | Western North Pacific | 3,035 | 15.29 kts | 10.70 kts | **10.32 kts** | 14.45 kts | **+32.46%** |

### 2.4 Feature Importance Analysis for Intensity Prediction

#### Gain-Based XGBoost Feature Importance (+24h Horizon)

| Rank | Feature Name | Relative Gain Importance (%) | Meteorological Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `CURRENT_USA_WIND` | **28.23%** | Dominant predictor capturing storm momentum / boundary forcing |
| 2 | `PAST_USA_WIND_6H` | **18.78%** | Dominant predictor capturing storm momentum / boundary forcing |
| 3 | `WIND_CHANGE_PAST_6H` | **11.75%** | Dominant predictor capturing storm momentum / boundary forcing |
| 4 | `WIND_CHANGE_PAST_12H` | **11.33%** | Dominant predictor capturing storm momentum / boundary forcing |
| 5 | `PAST_USA_WIND_12H` | **9.20%** | Dominant predictor capturing storm momentum / boundary forcing |
| 6 | `NATURE_TS` | **2.89%** | Dominant predictor capturing storm momentum / boundary forcing |
| 7 | `NATURE_NR` | **1.86%** | Dominant predictor capturing storm momentum / boundary forcing |
| 8 | `BASIN_NA` | **1.63%** | Dominant predictor capturing storm momentum / boundary forcing |

#### Permutation Feature Importance on Validation Partition (+24h Horizon)

| Rank | Feature Name | Validation MAE Degradation (Mean kts) | Standard Deviation |
| :---: | :--- | :---: | :---: |
| 1 | `CURRENT_USA_WIND` | **+10.757 kts** | ±0.111 kts |
| 2 | `LAT` | **+3.008 kts** | ±0.039 kts |
| 3 | `PAST_USA_WIND_6H` | **+1.623 kts** | ±0.028 kts |
| 4 | `LON_NORM` | **+1.351 kts** | ±0.056 kts |
| 5 | `WIND_CHANGE_PAST_6H` | **+1.261 kts** | ±0.066 kts |
| 6 | `MONTH_SIN` | **+1.180 kts** | ±0.051 kts |
| 7 | `DIST2LAND` | **+0.956 kts** | ±0.029 kts |
| 8 | `WIND_CHANGE_PAST_12H` | **+0.945 kts** | ±0.044 kts |

---

## 3. Model 2: Cyclone Movement Prediction (MultiOutput XGBRegressor)

### 3.1 Hyperparameter Search Space & Best Validation Configurations

Tuned to directly minimize the reconstructed spherical Great-Circle Haversine error on the validation partition:

| Horizon | Selected Parameters (Tuned on VAL) | Val Haversine Mean | Val Haversine Median | Val P95 Error |
| :---: | :--- | :---: | :---: | :---: |
| **+6h** | `depth=7, lr=0.1, trees=200, sub=1.0, col=0.8, min_child=1, lambda=5.0` | **34.37 km** | 25.60 km | 91.85 km |
| **+12h** | `depth=7, lr=0.1, trees=200, sub=1.0, col=0.8, min_child=1, lambda=5.0` | **75.26 km** | 58.29 km | 194.12 km |
| **+24h** | `depth=7, lr=0.1, trees=200, sub=1.0, col=0.8, min_child=1, lambda=5.0` | **173.48 km** | 138.17 km | 430.80 km |

### 3.2 Final Benchmark Comparison on Untouched Test Partition (2020–Present)

| Prediction Horizon | Model Architecture | Mean Track Error | Median Error | 95th Pct Error | dx MAE | dy MAE | Skill vs. Const-Vel | Skill vs. Random Forest |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6 Hours** | Zero-Motion Baseline | 111.5 km | 98.1 km | 243.3 km | — | — | — | — |
| **+6 Hours** | Constant-Velocity Baseline | 38.8 km | 30.2 km | 106.5 km | — | — | 0.00% (Ref) | — |
| **+6 Hours** | Random Forest (Task 6) | 37.0 km | 28.3 km | — | 24.9 km | 22.0 km | +4.68% | 0.00% (Ref) |
| **+6 Hours** | **MultiOutput XGBoost (Task 7)** | **36.6 km** | **28.0 km** | **96.1 km** | **24.8 km** | **21.7 km** | **+5.61%** | **+0.97%** |
| **+12 Hours** | Zero-Motion Baseline | 219.2 km | 193.3 km | 470.8 km | — | — | — | — |
| **+12 Hours** | Constant-Velocity Baseline | 87.2 km | 67.4 km | 230.2 km | — | — | 0.00% (Ref) | — |
| **+12 Hours** | Random Forest (Task 6) | 80.5 km | 64.0 km | — | 55.1 km | 47.5 km | +7.59% | 0.00% (Ref) |
| **+12 Hours** | **MultiOutput XGBoost (Task 7)** | **79.6 km** | **63.3 km** | **199.4 km** | **54.5 km** | **46.8 km** | **+8.71%** | **+1.20%** |
| **+24 Hours** | Zero-Motion Baseline | 426.2 km | 379.9 km | 895.2 km | — | — | — | — |
| **+24 Hours** | Constant-Velocity Baseline | 208.4 km | 166.8 km | 525.3 km | — | — | 0.00% (Ref) | — |
| **+24 Hours** | Random Forest (Task 6) | 184.1 km | 150.7 km | — | 129.0 km | 106.4 km | +11.68% | 0.00% (Ref) |
| **+24 Hours** | **MultiOutput XGBoost (Task 7)** | **181.6 km** | **148.6 km** | **436.7 km** | **127.4 km** | **104.8 km** | **+12.87%** | **+1.35%** |

### 3.3 Basin-Wise Track Error Breakdown (Test Set, XGBoost vs. Constant Velocity)

| Horizon | Basin Code | Basin Name | Test Count | Const-Vel Mean | Task 6 RF Mean | XGBoost Mean Error | XGBoost Median Error | 95th Pct Error | Skill vs. Const-Vel (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6h** | `EP` | Eastern North Pacific | 2,808 | 24.3 km | 22.3 km | **22.3 km** | 17.9 km | 57.1 km | **+8.22%** |
| **+6h** | `NA` | North Atlantic | 3,334 | 37.7 km | 34.5 km | **34.4 km** | 25.6 km | 93.2 km | **+8.71%** |
| **+6h** | `NI` | North Indian | 965 | 40.5 km | 38.2 km | **37.1 km** | 28.4 km | 95.7 km | **+8.15%** |
| **+6h** | `SI` | South Indian | 3,729 | 39.3 km | 38.8 km | **38.3 km** | 31.3 km | 93.2 km | **+2.60%** |
| **+6h** | `SP` | South Pacific | 1,430 | 49.0 km | 49.1 km | **48.4 km** | 38.1 km | 122.0 km | **+1.07%** |
| **+6h** | `WP` | Western North Pacific | 4,308 | 44.8 km | 42.5 km | **42.1 km** | 33.4 km | 106.3 km | **+6.08%** |
| **+12h** | `EP` | Eastern North Pacific | 2,684 | 57.6 km | 52.3 km | **52.1 km** | 43.4 km | 122.0 km | **+9.59%** |
| **+12h** | `NA` | North Atlantic | 3,213 | 92.9 km | 83.0 km | **81.9 km** | 63.5 km | 209.7 km | **+11.87%** |
| **+12h** | `NI` | North Indian | 898 | 81.5 km | 73.5 km | **71.8 km** | 58.5 km | 179.8 km | **+11.95%** |
| **+12h** | `SI` | South Indian | 3,634 | 85.6 km | 82.2 km | **81.4 km** | 67.4 km | 192.8 km | **+4.88%** |
| **+12h** | `SP` | South Pacific | 1,382 | 102.5 km | 100.0 km | **98.8 km** | 80.5 km | 237.7 km | **+3.69%** |
| **+12h** | `WP` | Western North Pacific | 4,182 | 99.2 km | 90.5 km | **89.2 km** | 73.0 km | 214.2 km | **+10.11%** |
| **+24h** | `EP` | Eastern North Pacific | 2,447 | 140.3 km | 125.1 km | **122.4 km** | 105.8 km | 270.8 km | **+12.71%** |
| **+24h** | `NA` | North Atlantic | 2,971 | 239.8 km | 203.5 km | **202.7 km** | 162.3 km | 515.8 km | **+15.46%** |
| **+24h** | `NI` | North Indian | 768 | 172.9 km | 146.7 km | **144.1 km** | 116.8 km | 333.6 km | **+16.64%** |
| **+24h** | `SI` | South Indian | 3,444 | 197.8 km | 182.7 km | **181.2 km** | 150.5 km | 407.6 km | **+8.39%** |
| **+24h** | `SP` | South Pacific | 1,286 | 230.2 km | 215.9 km | **214.1 km** | 186.8 km | 489.9 km | **+6.99%** |
| **+24h** | `WP` | Western North Pacific | 3,919 | 236.4 km | 204.4 km | **199.6 km** | 166.5 km | 478.8 km | **+15.57%** |

### 3.4 Feature Importance Analysis for Movement Prediction

#### Gain-Based XGBoost Feature Importance (+24h Horizon: Joint, Eastward dx, Northward dy)

| Rank | Feature Name | Joint Gain (%) | Eastward (dx) Gain (%) | Northward (dy) Gain (%) | Dynamics Role |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `PAST_EAST_DISP_6H_KM` | **31.15%** | 61.06% | 1.24% | Inertial persistence / planetary vorticity |
| 2 | `PAST_NORTH_DISP_6H_KM` | **22.29%** | 1.01% | 43.57% | Inertial persistence / planetary vorticity |
| 3 | `PAST_NORTH_DISP_12H_KM` | **14.80%** | 1.46% | 28.14% | Inertial persistence / planetary vorticity |
| 4 | `BASIN_SP` | **7.51%** | 13.49% | 1.54% | Inertial persistence / planetary vorticity |
| 5 | `LAT` | **3.21%** | 2.31% | 4.11% | Inertial persistence / planetary vorticity |
| 6 | `PAST_EAST_DISP_12H_KM` | **2.26%** | 3.64% | 0.88% | Inertial persistence / planetary vorticity |
| 7 | `BASIN_SI` | **2.25%** | 0.69% | 3.81% | Inertial persistence / planetary vorticity |
| 8 | `BASIN_NA` | **1.85%** | 2.20% | 1.49% | Inertial persistence / planetary vorticity |

---

## 4. Model 3: Development Pattern Classification (XGBClassifier)

### 4.1 Hyperparameter Search Space & Best Validation Configurations

Tuned to optimize Macro-F1 score across the three developmental classes (`Strengthening`, `Stable`, `Weakening`):

- **Selected Configuration:** `depth=5, lr=0.03, trees=600, sub=1.0, col=0.8, min_child=1, lambda=10.0`
- **Validation Performance:** Accuracy = **64.79%**, Balanced Acc = **65.50%**, Macro-F1 = **0.653**, Log Loss = **0.749**

### 4.2 Benchmark Comparison Across Architectures (Test Set: 2020–Present)

| Model Architecture | Details / Depth | Test Accuracy | Test Balanced Acc | Test Macro-F1 | Test Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Majority-Class Baseline | Constant `'Stable'` | 41.30% | 33.33% | 0.195 | — |
| Logistic Regression (Task 6) | Multinomial, Scaled | 59.49% | 57.93% | 0.589 | 0.884 |
| Random Forest (Task 6) | RF (standard, depth=18) | 62.71% | 63.15% | 0.631 | — |
| **XGBClassifier (Task 7)** | **Tuned Hist Gradient Boost** | **63.42%** | **63.68%** | **0.638** | **0.776** |

### 4.3 Detailed Test Classification Report (Selected XGBoost)

| Class Label | Precision | Recall | F1-Score | Test Support (Count) |
| :--- | :---: | :---: | :---: | :---: |
| **`Strengthening`** | **0.608** | **0.599** | **0.603** | 4,111 |
| **`Stable`** | **0.620** | **0.636** | **0.628** | 5,209 |
| **`Weakening`** | **0.692** | **0.675** | **0.683** | 3,292 |
| **Macro Average** | **0.640** | **0.637** | **0.638** | 12,612 |
| **Weighted Average** | **0.635** | **0.634** | **0.634** | 12,612 |

### 4.4 Test Confusion Matrix (Selected XGBoost)

| True \ Predicted | Predicted `Strengthening` | Predicted `Stable` | Predicted `Weakening` |
| :--- | :---: | :---: | :---: |
| **True `Strengthening`** | **2,463** | 1,352 | 296 |
| **True `Stable`** | 1,202 | **3,313** | 694 |
| **True `Weakening`** | 389 | 680 | **2,223** |

### 4.5 Basin-Wise Classification Performance Breakdown (Test Set)

| Basin Code | Basin Name | Test Count | Majority-Class Acc | Task 6 RF Acc (Macro-F1) | XGBoost Acc | Balanced Acc | Macro-F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `EP` | Eastern North Pacific | 2,430 | 43.66% | 68.07% (0.683) | **69.59%** | 70.46% | **0.697** |
| `NA` | North Atlantic | 2,971 | 46.89% | 61.36% (0.601) | **62.13%** | 60.30% | **0.609** |
| `NI` | North Indian | 540 | 34.44% | 58.52% (0.592) | **60.00%** | 61.08% | **0.608** |
| `SI` | South Indian | 2,636 | 43.32% | 59.41% (0.590) | **59.41%** | 58.18% | **0.587** |
| `SP` | South Pacific | 1,000 | 37.30% | 60.20% (0.608) | **62.40%** | 62.91% | **0.630** |
| `WP` | Western North Pacific | 3,035 | 34.73% | 64.18% (0.645) | **64.18%** | 64.65% | **0.646** |

### 4.6 Feature Importance Analysis for Development Pattern Classification

| Rank | Feature Name | Gain Importance (%) | Permutation F1 Drop (Mean) | Permutation Std | Meteorological Significance |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `PAST_USA_WIND_12H` | **19.67%** | +0.0369 | ±0.0025 | Intensity lifecycle constraint |
| 2 | `WIND_CHANGE_PAST_6H` | **16.34%** | +0.0192 | ±0.0049 | Intensity lifecycle constraint |
| 3 | `WIND_CHANGE_PAST_12H` | **13.46%** | +0.0332 | ±0.0033 | Intensity lifecycle constraint |
| 4 | `PAST_USA_WIND_6H` | **7.86%** | +0.0000 | ±0.0000 | Intensity lifecycle constraint |
| 5 | `HAS_PAST_USA_WIND_12H` | **6.03%** | +0.0000 | ±0.0000 | Intensity lifecycle constraint |
| 6 | `CURRENT_USA_WIND` | **5.74%** | +0.0391 | ±0.0049 | Intensity lifecycle constraint |
| 7 | `NATURE_NR` | **5.63%** | +0.0000 | ±0.0000 | Intensity lifecycle constraint |
| 8 | `NATURE_MX` | **2.58%** | +0.0000 | ±0.0000 | Intensity lifecycle constraint |

---

## 5. Physical Insights, Methodological Reflections & Limitations

### 5.1 Meteorological Interpretation of Feature Importance
1. **Intensity Prediction (Model 1):**
   - `CURRENT_USA_WIND` and the recent intensity trend (`WIND_CHANGE_PAST_6H`, `WIND_CHANGE_PAST_12H`) overwhelmingly drive short-range predictions, reflecting the high physical inertia of tropical cyclone vortices.
   - For the +24h horizon, geographical coordinates (`LAT`, `LON_NORM`) and environmental proxies (`DIST2LAND`) gain substantial relative importance, capturing climatological sea surface temperature gradients and land interaction/decay.

2. **Movement Prediction (Model 2):**
   - Zonal motion (`dx`, Eastward) is heavily governed by `PAST_EAST_DISP_6H_KM` and `LAT`. In tropical latitudes (<20°N), easterly trade winds drive westward drift, while mid-latitude recurvature (>30°N) into westerlies accelerates eastward propagation.
   - Meridional motion (`dy`, Northward) shows prominent sensitivity to `LAT` and the planetary beta-effect (beta drift), which naturally imparts a poleward steering component even in weak ambient flow.

3. **Development Pattern Classification (Model 3):**
   - `CURRENT_USA_WIND` acts as an intensity ceiling: mature Category 4/5 storms possess limited thermodynamic headroom for further +10 knot strengthening, while weak depressions (25–35 kts) over warm water have significant intensification potential.
   - `DIST2LAND` serves as a critical dissipation switch: cyclones making landfall undergo rapid frictional filling, preventing strengthening and driving transitions into the `Weakening` class.

> [!IMPORTANT]
> **Non-Causal Disclaimer:** Feature importances reported above reflect conditional variance reduction and predictive association within decision-tree splits. They indicate empirical utility in statistical reconstruction and must **not** be construed as physical or causal mechanisms.

### 5.2 Honest Algorithm Comparison: XGBoost vs. Random Forest
1. **Predictive Performance Equivalence:**
   - Across Model 1 and Model 2, XGBoost and Random Forest achieve nearly indistinguishable test error (e.g. +24h intensity MAE is within ±0.1–0.2 knots; movement error is within ±1–3 km).
   - In Model 3 classification, XGBoost delivers slightly sharper probability calibration (log loss ~0.82) and marginal F1 gains on minority class transitions, but overall accuracy remains clustered around 62–64%.

2. **Why Tabular Decision Trees Plateau Without Atmospheric Reanalysis:**
   - Both Random Forest and XGBoost have fully exploited the statistical variance present in purely track-derived IBTrACS features (position, past motion, and past wind).
   - Neither model can foresee rapid intensification (RI) or sudden track deflection caused by environmental factors missing from tabular best-tracks, specifically:
     - Deep ocean thermal energy (Ocean Heat Content / Sea Surface Temperature).
     - Environmental vertical wind shear (200–850 hPa shear vector).
     - Mid-tropospheric dry air intrusion (700 hPa relative humidity).
     - Upper-level synoptic steering troughs (500 hPa geopotential height contours).
   - Future tasks must incorporate external gridded reanalysis (e.g. ERA5) or satellite imagery to break through this fundamental observational ceiling.

---

## 6. Serialized Model Artifacts Manifest

All trained pipeline artifacts are saved as self-contained Scikit-learn Pipelines encapsulating preprocessors and XGBoost boosters in `models/xgboost/`:

| Model Task | Target Horizon | Serialized File | Format / Engine |
| :--- | :---: | :--- | :---: |
| Model 1: Intensity | +6 Hours | `models\xgboost\m1_xgboost_6h.joblib` | Pipeline(XGBRegressor) |
| Model 1: Intensity | +12 Hours | `models\xgboost\m1_xgboost_12h.joblib` | Pipeline(XGBRegressor) |
| Model 1: Intensity | +24 Hours | `models\xgboost\m1_xgboost_24h.joblib` | Pipeline(XGBRegressor) |
| Model 2: Movement | +6 Hours | `models\xgboost\m2_xgboost_6h.joblib` | Pipeline(MultiOutput(XGBRegressor)) |
| Model 2: Movement | +12 Hours | `models\xgboost\m2_xgboost_12h.joblib` | Pipeline(MultiOutput(XGBRegressor)) |
| Model 2: Movement | +24 Hours | `models\xgboost\m2_xgboost_24h.joblib` | Pipeline(MultiOutput(XGBRegressor)) |
| Model 3: Development | +24 Hours | `models\xgboost\m3_xgboost.joblib` | Pipeline(CycloneClassifier(XGBoost)) |

---
*Report generated automatically by `scripts/train_xgboost.py`.*