# Cyclone Identification, Classification, and Prediction System using NOAA IBTrACS

An end-to-end, reproducible machine learning system for predicting tropical cyclone intensity, movement trajectory, and 24-hour rapid development patterns across global oceanic basins using the NOAA International Best Track Archive for Climate Stewardship (IBTrACS v04r01).

---

## 🌪️ Project Overview

This system formulates and solves three distinct meteorological prediction tasks across strict historical partitions (**TRAIN** $\le 2014$, **VAL** $2015\text{--}2019$, **TEST** $2020+$) with zero cyclone ID overlap and rigorous anti-leakage safeguards:

1. **Model 1: Intensity Prediction (Regression)**
   - **Target:** Future 1-minute maximum sustained wind speed (`TARGET_USA_WIND`) in knots.
   - **Horizons:** +6 hours, +12 hours, and +24 hours.
   - **Algorithms:** Persistence baseline, Linear Regression, Random Forest Regressor, and Tuned XGBoost Regressor.
2. **Model 2: Movement & Trajectory Prediction (Multi-Output Regression)**
   - **Target:** Orthogonal Eastward (`dx`) and Northward (`dy`) displacements in kilometers on a local tangent plane with spherical geodetic inversion into true $(\widehat{\text{LAT}}, \widehat{\text{LON}})$.
   - **Evaluation Metric:** Great-Circle spherical Haversine error in kilometers.
   - **Horizons:** +6 hours, +12 hours, and +24 hours.
   - **Algorithms:** Zero-motion baseline, Constant-velocity baseline, Random Forest Regressor, and MultiOutput XGBoost Regressor.
3. **Model 3: Development Pattern Classification (Multi-Class Classification)**
   - **Target:** 24-hour intensification regime (`CLASS_LABEL` $\in$ `['Strengthening', 'Stable', 'Weakening']`) based on operational $\pm 10\text{ knot}$ wind change threshold ($\Delta V_{24h}$).
   - **Algorithms:** Majority-class baseline (`Stable`), Multinomial Logistic Regression, Random Forest Classifier, and Tuned XGBoost Classifier.

---

## 📊 Benchmark Scoreboard (`TEST` Partition: 2020–Present)

| Model Task | Target / Metric | Horizon | Simple Baseline | Random Forest | XGBoost (Selected) | Skill vs. Baseline | Delta vs. RF |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: | :---: |
| **Model 1: Intensity** | Wind MAE (kts) | **+6h** | Persistence: 4.02 kts | RF: 3.52 kts | **3.43 kts** | **+14.83%** | +2.56% |
| **Model 1: Intensity** | Wind MAE (kts) | **+12h** | Persistence: 7.50 kts | RF: 5.92 kts | **5.71 kts** | **+23.93%** | +3.55% |
| **Model 1: Intensity** | Wind MAE (kts) | **+24h** | Persistence: 13.31 kts | RF: 10.01 kts | **9.58 kts** | **+27.98%** | +4.30% |
| **Model 2: Movement** | Haversine Track Error (km) | **+6h** | Const-Vel: 38.8 km | RF: 37.0 km | **36.6 km** | **+5.61%** | +0.97% |
| **Model 2: Movement** | Haversine Track Error (km) | **+12h** | Const-Vel: 87.2 km | RF: 80.5 km | **79.6 km** | **+8.71%** | +1.20% |
| **Model 2: Movement** | Haversine Track Error (km) | **+24h** | Const-Vel: 208.4 km | RF: 184.1 km | **181.6 km** | **+12.87%** | +1.35% |
| **Model 3: Development** | Accuracy / Macro-F1 | **+24h** | Majority: 41.30% | RF: 62.71% (0.631 F1) | **63.42% (0.638 F1)** | **+22.12%** | +0.71% Acc |

---

## 📁 Repository Structure

```text
├── models/
│   ├── baselines/          # Serialized Linear, Logistic, and RF pipelines (README included)
│   └── xgboost/            # Serialized XGBoost pipelines (~1.5–4.7 MB each)
├── processed/              # Clean model-ready Parquet partitions (TRAIN, VAL, TEST)
│   ├── model1_intensity.parquet
│   ├── model2_movement.parquet
│   └── model3_development.parquet
├── raw/                    # Raw NOAA IBTrACS data folder (download instructions in README)
│   └── README.md
├── reports/                # Comprehensive technical and analytical audit reports
│   ├── dataset_inspection.md
│   ├── modelling_cohort_audit.md
│   ├── iflag_and_basin_validation.md
│   ├── preprocessing_report.md
│   ├── model_feature_manifest.md
│   ├── baseline_model_results.md
│   ├── xgboost_experiment_results.md
│   └── model_portability_validation.md
├── scripts/                # Standalone CLI tools for preprocessing, training, and testing
│   ├── inspect_dataset.py
│   ├── audit_modelling_cohort.py
│   ├── audit_iflag_and_basins.py
│   ├── prepare_datasets.py
│   ├── validate_processed_datasets.py
│   ├── train_baselines.py
│   ├── train_xgboost.py
│   └── test_model_loading.py
├── src/
│   └── cyclone_ml/         # Modular Python package for estimators, geodesy, and features
│       ├── __init__.py
│       ├── estimators.py   # CycloneClassifier Scikit-learn wrapper
│       ├── geodesy.py      # Spherical coordinate reconstruction & Haversine distance
│       └── features.py     # Cyclical Month/Hour feature transforms
├── requirements-models.txt # Exact frozen library versions for model serving
└── .gitignore              # Git exclusion rules for large datasets, binaries, and caches
```

---

## 🚀 Getting Started

### 1. Environment Setup
Clone the repository and install the verified dependencies:
```bash
git clone https://github.com/NannawarePranav/Cyclone-Identification-Classification-and-Prediction-System-using-IBTrACS.git
cd Cyclone-Identification-Classification-and-Prediction-System-using-IBTrACS

python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-models.txt
```

### 2. Verify Model Portability & Run Inference Smoke Tests
Test all 18 models across Model 1, Model 2, and Model 3:
```bash
python scripts/test_model_loading.py
```

### 3. Training and Experiments
- **Reproduce Baseline Models:**
  ```bash
  python scripts/train_baselines.py --model all
  ```
- **Run XGBoost Experiments & Hyperparameter Tuning:**
  ```bash
  python scripts/train_xgboost.py --model all --n-iter 8
  ```

---

## 🔬 Scientific Methodology & Quality Safeguards

- **NOAA Agency Wind Harmonization:** Uses 1-minute maximum sustained wind (`USA_WIND`) from JTWC and NHC, filtering strictly for original/verified observations using character 0 of the IBTrACS `IFLAG` (`'O'` and `'V'`).
- **Antimeridian & Latitude Continuity:** Longitudes are normalized to $[-180^\circ, 180^\circ]$. Movement targets are computed using local tangent-plane orthogonal eastward and northward displacements ($dx, dy$) in kilometers, avoiding antimeridian discontinuity artifacts.
- **Strict Anti-Leakage Discipline:** Preprocessors (imputers, one-hot encoders, scalers) are fitted strictly on `TRAIN` ($\le 2014$). Hyperparameters are tuned on `VAL` ($2015\text{--}2019$). Final metrics are reported on untouched `TEST` ($2020+$).

---

## 📄 License and Data Attribution
Data provided by the National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Information (NCEI) [IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive).
