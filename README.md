# Prediction of Different Tropical Cyclone Patterns

An end-to-end, reproducible machine learning system and interactive web dashboard for predicting tropical cyclone intensity, movement trajectory, and 24-hour development classes across global oceanic basins using the NOAA International Best Track Archive for Climate Stewardship (IBTrACS v04r01).

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
3. **Model 3: Development Classification (Multi-Class Classification)**
   - **Target:** 24-hour development class (`CLASS_LABEL` $\in$ `['Strengthening', 'Stable', 'Weakening']`) based on the project's $\pm 10\text{ knot}$ wind change threshold ($\Delta V_{24h}$).
   - **Algorithms:** Majority-class baseline (`Stable`), Multinomial Logistic Regression, Random Forest Classifier, and Tuned XGBoost Classifier.

---

## 🌐 Interactive Dashboard

The dashboard presents **Prediction of Different Tropical Cyclone Patterns** with Streamlit and Plotly: explore IBTrACS records, generate XGBoost predictions, replay storm lifecycles, and compare models against baseline benchmarks.

```bash
# Launch the dashboard locally
pip install -r requirements-dashboard.txt
streamlit run app.py
```

### Dashboard Architecture & Core Modules
```text
dashboard/
├── config.py              # Dark ocean color tokens, SVG icons, and custom CSS
├── data_loader.py         # Cached Parquet data loaders and cyclone catalog indexer
├── model_loader.py        # Cached XGBoost model loaders and artifact validator
├── inference.py           # Decoupled inference pipelines and geodetic reconstruction
├── charts.py              # Plotly dark ocean maps, time-series, and performance plots
├── components.py          # Standardized headers, metric cards, and cyclone selectors
└── pages/
    ├── overview.py        # Global activity map, basin metrics, and AI architecture
    ├── intensity.py       # Multi-horizon intensity forecasting (+6h, +12h, +24h)
    ├── movement.py        # Multi-horizon trajectory forecast with geodetic reconstruction
    ├── development.py     # 24-hour intensification regime classification & probabilities
    ├── historical_replay.py # Chronological step-through playback and intensity profiles
    └── model_performance.py # Academic-grade scoreboards on independent test cohort (2020+)
```

### Key Interactive Features
- **Global Climatology & Activity:** High-performance Plotly Scattergeo visualizations of historical cyclone paths, basin distributions, and annual activity trends (1980–2024).
- **Live Multi-Horizon Inference:** Run all 7 trained XGBoost models on any historical storm observation with zero data leakage.
- **Spherical Geodesy Reconstruction:** Converts Model 2 tangent-plane displacement ($dx, dy$) into true latitude and longitude while strictly respecting antimeridian $[-180^\circ, 180^\circ]$ boundaries.
- **Historical Lifecycle Replay:** Chronologically scrub through any recorded cyclone, analyzing elapsed translation speed, cumulative track distance, and peak sustained wind.
- **Uncalibrated Probability Disclosure:** Explicitly documents raw softmax confidence for Model 3, emphasizing relative ranking rather than calibrated likelihoods.
- **Strict Dark Ocean Aesthetic:** Designed according to [`design-system/cyclone-atlas/MASTER.md`](design-system/cyclone-atlas/MASTER.md) with custom typography, responsive metric cards, and persistent research disclaimers.

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
| **Model 3: Development** | Accuracy / Macro-F1 | **+24h** | Majority: 41.30% | RF: 62.71% (0.631 F1) | **64.05% (0.646 F1)** | **+22.75 pp** | +1.34 pp Acc |

Current Model 3 uses longer wind history and pressure features selected on validation data. The archived Random Forest numbers are historical references; the dashboard computes current-model and simple-baseline results from saved artifacts. See [upgrade results](reports/development_upgrade.md) and [machine-readable scores](reports/benchmark_results.json).

### Presentation walkthrough
Open **Guided Presentation Demo** from the sidebar to select an unseen cyclone, predict, then reveal the recorded outcome. Models 1–3, Historical Replay and Guided Demo share the selected cyclone, UTC observation, basin, seasons, partition and name/ID search. The demo only supports TEST observations with all forecast horizons available; unavailable selections are explained without overwriting the original selection. Overview and benchmarks show global results while preserving the selection, which is visible in the sidebar on every page.

### Updated local workflow
```bash
python scripts/export_benchmarks.py
python scripts/test_development_upgrade.py
streamlit run app.py
```
The dashboard refreshes saved scores when model/data fingerprints change. The original Model 3 is preserved in `models/archive/pre_development_v2/`; the enhanced development inputs are in `processed/model3_development_enriched.parquet`.

---

## 📁 Repository Structure

```text
├── app.py                  # Main entry point for Cyclone Atlas web dashboard
├── dashboard/              # Modular dashboard package (config, loaders, charts, pages)
├── design-system/          # Design tokens and visual specifications
│   └── cyclone-atlas/
│       └── MASTER.md       # Dark ocean scientific design tokens and guidelines
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
│   ├── test_model_loading.py
│   ├── test_dashboard_integrity.py
│   └── test_dashboard_refinements.py
├── src/
│   └── cyclone_ml/         # Modular Python package for estimators, geodesy, and features
│       ├── __init__.py
│       ├── estimators.py   # CycloneClassifier Scikit-learn wrapper
│       ├── geodesy.py      # Spherical coordinate reconstruction & Haversine distance
│       └── features.py     # Cyclical Month/Hour feature transforms
├── requirements-models.txt    # Exact frozen library versions for model serving
├── requirements-dashboard.txt # Extended dependencies for Streamlit and Plotly dashboard
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
pip install -r requirements-dashboard.txt
```

### 2. Verify System & Model Integrity
Run the automated verification suite to validate all models, data pipelines, and visualizations:
```bash
python scripts/test_dashboard_integrity.py
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

### 4. Training and Reproducibility
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
- **Persistent Research Notice:** All dashboard views incorporate a research disclaimer clarifying that predictions are derived from historical best-track data and are not intended as live operational weather warnings.

---

## 📄 License and Data Attribution
Data provided by the National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Information (NCEI) [IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive).
