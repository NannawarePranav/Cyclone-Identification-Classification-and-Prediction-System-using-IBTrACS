# PREDICTION OF DIFFERENT TROPICAL CYCLONE PATTERNS USING NOAA IBTrACS

## A Machine Learning Approach for Cyclone Intensity Forecasting, Track Movement Prediction, and 24-Hour Development Classification

**Project Report**  
**Submitted in partial fulfilment of the requirements for the award of the degree of**  
**Bachelor of Technology in Artificial Intelligence and Data Science**

**Submitted by:** Pranav Nannaware  
**Department:** Artificial Intelligence and Data Science  
**Institute:** CSMSS Chh. Shahu College of Engineering, Chhatrapati Sambhajinagar, Maharashtra  
**University:** Dr. Babasaheb Ambedkar Technological University (BATU), Lonere  
**Academic Year:** [Insert Academic Year]  
**Project Guide:** [Insert Guide Name]  
**Head of Department:** [Insert HOD Name]

---

# CERTIFICATE

This is to certify that the project entitled **“Prediction of Different Tropical Cyclone Patterns using NOAA IBTrACS”** is a bonafide work carried out by **Pranav Nannaware**, student of Bachelor of Technology in Artificial Intelligence and Data Science, CSMSS Chh. Shahu College of Engineering, Chhatrapati Sambhajinagar, during the academic year **[Insert Academic Year]**, under the guidance and supervision of **[Guide Name]**.

The work is submitted in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in Artificial Intelligence and Data Science under Dr. Babasaheb Ambedkar Technological University, Lonere.

| Project Guide | Head of Department | Principal |
|---|---|---|
| Name: __________ | Name: __________ | Name: __________ |
| Signature: ______ | Signature: ______ | Signature: ______ |
| Date: ___________ | Date: ___________ | Date: ___________ |

---

# DECLARATION

I hereby declare that the project report entitled **“Prediction of Different Tropical Cyclone Patterns using NOAA IBTrACS”** is based on the work carried out by me under the guidance of **[Guide Name]**. The implementation, data preparation, experiments, and results described in this report are documented in the associated project repository. Wherever external datasets, publications, software libraries, or documentation have been used, the corresponding sources have been acknowledged.

I further declare that this report has not been submitted, in whole or in part, for the award of any other degree or qualification.

**Submitted by:** Pranav Nannaware  
**Signature:** __________________  
**Date:** _______________________

---

# ACKNOWLEDGEMENT

I express my sincere gratitude to **[Project Guide Name]** for guidance, suggestions, and support throughout the development of this project. I also thank the Head of the Department and the faculty members of the Department of Artificial Intelligence and Data Science, CSMSS Chh. Shahu College of Engineering, for providing the academic environment and resources required to carry out this work.

I acknowledge the National Oceanic and Atmospheric Administration (NOAA), National Centers for Environmental Information (NCEI), and the contributors to the International Best Track Archive for Climate Stewardship (IBTrACS) for providing the historical tropical cyclone dataset used in this project. I also acknowledge the developers and maintainers of the open-source Python libraries used for data processing and machine learning.

Finally, I thank my family, friends, and everyone who provided support and encouragement during the completion of this project.

---

# ABSTRACT

Tropical cyclones are complex atmospheric systems that can cause severe damage through strong winds, heavy rainfall, storm surges, and coastal impacts. Understanding their intensity, movement, and short-term development is important for meteorological analysis and disaster-risk planning. This project develops a machine learning framework that uses historical global tropical cyclone records to predict three related aspects of cyclone behaviour: future wind intensity, future geographic movement, and short-term intensity-development category.

The project uses the **NOAA International Best Track Archive for Climate Stewardship (IBTrACS), version v04r01**, as its primary data source. The global CSV archive inspected for this work contains 730,416 observations and 174 fields, representing 13,631 unique cyclone identifiers across the historical record. The raw data is examined for completeness, temporal coverage, track quality, reporting-agency consistency, and interpolation flags before model-specific datasets are generated.

Three supervised learning tasks are formulated. **Model 1** performs regression to estimate the future USA agency maximum sustained wind speed at +6, +12, and +24 hours. **Model 2** performs multi-output regression to estimate eastward and northward cyclone displacement over the same horizons; predicted displacements are converted into geographic coordinates and assessed using great-circle Haversine distance. **Model 3** performs three-class classification to identify whether a cyclone is likely to strengthen, remain stable, or weaken over the next 24 hours, using defined wind-change thresholds of +10 knots and -10 knots.

The modelling pipeline uses time-based training, validation, and test partitions to evaluate generalisation to later cyclone seasons. Storm identifiers are kept within a single partition to reduce leakage between related observations. Original or verified USA wind observations are selected for intensity-based targets, and current/future wind pairs are required to come from the same reporting agency. The project first establishes simple and Random Forest baselines, then evaluates tuned XGBoost models. On the held-out 2020-and-later test partition, XGBoost achieves a wind-speed MAE of 3.43, 5.71, and 9.58 knots at +6, +12, and +24 hours respectively; movement errors of 36.6, 79.6, and 181.6 km; and 63.42% accuracy with 0.638 macro-F1 for the 24-hour development classification task.

The project also implements reusable geodesy and feature-processing modules and validates the portability of 18 serialized model artifacts in a fresh Python process. The present completed milestone is the data and model-engineering pipeline. A Streamlit dashboard for interactive exploration and prediction is planned as the next development phase and is not represented as a completed component in this report.

**Keywords:** Tropical Cyclone, IBTrACS, NOAA, Machine Learning, XGBoost, Random Forest, Intensity Forecasting, Track Prediction, Multi-class Classification, Haversine Distance.

---

# TABLE OF CONTENTS

1. Introduction  
2. Problem Definition and Project Scope  
3. Objectives and Expected Outcomes  
4. Background and Literature Basis  
5. Dataset Description  
6. Data Inspection and Quality Auditing  
7. Data Preprocessing and Feature Engineering  
8. Machine Learning Methodology  
9. Experimental Setup and Evaluation Metrics  
10. Results and Discussion  
11. Software Architecture and Implementation  
12. Model Portability and Reproducibility  
13. Limitations and Challenges  
14. Future Scope  
15. Conclusion  
16. References  
17. Appendices

> In a word processor, generate or refresh the page-numbered table of contents after formatting the report.

---

# LIST OF ABBREVIATIONS

| Abbreviation | Meaning |
|---|---|
| AI | Artificial Intelligence |
| CSV | Comma-Separated Values |
| ET | Extratropical |
| F1 | F1-score |
| IBTrACS | International Best Track Archive for Climate Stewardship |
| IFLAG | IBTrACS interpolation/observation flag field |
| MAE | Mean Absolute Error |
| MBE | Mean Bias Error |
| ML | Machine Learning |
| NCEI | National Centers for Environmental Information |
| NOAA | National Oceanic and Atmospheric Administration |
| RMSE | Root Mean Squared Error |
| SID | Storm Identifier |
| UTC | Coordinated Universal Time |
| WMO | World Meteorological Organization |
| XGBoost | Extreme Gradient Boosting |

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background

Tropical cyclones are large rotating weather systems that develop over warm tropical or subtropical oceans. Their evolution involves interacting atmospheric and oceanic processes. A cyclone may intensify, weaken, remain at a similar intensity, or change its direction and speed as it moves. These properties are important for understanding a storm's potential impacts.

Cyclone analysis commonly considers several related variables. **Intensity** describes the strength of the system and is often represented using maximum sustained wind speed or minimum central pressure. **Movement** describes the change in the cyclone's geographic position over time. **Development pattern** describes whether the system is strengthening, remaining relatively stable, or weakening during a defined interval.

Historical best-track archives provide observations of cyclone location, intensity, timing, and other characteristics. These records can be transformed into supervised machine learning datasets in which past and current storm information is used to estimate a future value or category.

## 1.2 Motivation

The motivation for this project is to investigate whether machine learning models can learn useful patterns from historical global cyclone records and provide estimates for multiple aspects of short-term cyclone behaviour. Instead of treating cyclone forecasting as a single output problem, the project divides it into three focused tasks: intensity prediction, movement prediction, and development classification.

The use of a global archive enables the experiments to include cyclones from multiple ocean basins. The project also emphasises data quality and evaluation design. Historical storm observations are temporally related, and records from different agencies may use different conventions. These issues must be considered when preparing training data and assessing model performance.

## 1.3 Project Overview

The project is an offline, historical-data-based machine learning study. It processes the NOAA IBTrACS global CSV archive, prepares three separate model-ready datasets, trains baseline and XGBoost models, evaluates them on later seasons, and packages the trained models for reuse.

The three tasks are related but have different target variables and evaluation metrics:

- **Intensity prediction:** estimate future maximum sustained wind speed in knots.
- **Movement prediction:** estimate future eastward and northward displacement and reconstruct a future latitude/longitude.
- **Development classification:** classify the 24-hour change in wind intensity as Strengthening, Stable, or Weakening.

## 1.4 Project Boundary

The current implementation is limited to historical best-track data and supervised machine learning. It does not perform satellite-image recognition, cyclone-genesis detection, real-time meteorological data ingestion, operational warning generation, or automatic disaster alerts. These are outside the completed scope of this college-level project.

---

# CHAPTER 2 — PROBLEM DEFINITION AND PROJECT SCOPE

## 2.1 Problem Statement

To design and evaluate a machine learning framework that uses historical global tropical cyclone best-track observations to predict future cyclone intensity, geographic movement, and short-term intensity-development category at defined forecast horizons.

## 2.2 Problem Formulation

Let a cyclone observation at time `t` contain available historical and current information such as location, basin, cyclone nature, wind intensity, recent intensity changes, and previous movement. The objective is to learn a mapping from these permitted inputs to a future target at time `t + h`, where `h` is 6, 12, or 24 hours for the regression tasks and 24 hours for classification.

### Model 1 — Intensity Prediction

Predict future one-minute maximum sustained wind speed:

`V_hat(t+h) = f(X_t)`

where `V_hat(t+h)` is the predicted USA wind speed in knots and `X_t` contains only information available at or before the prediction time.

Horizons: **+6 hours, +12 hours, +24 hours**.

### Model 2 — Movement Prediction

Predict local eastward and northward displacement:

`(dx_hat, dy_hat) = g(X_t)`

The predicted displacement is converted into a future geographic position `(LAT_hat, LON_hat)`. The location error is measured as the spherical great-circle distance between the predicted and reference positions.

Horizons: **+6 hours, +12 hours, +24 hours**.

### Model 3 — Development Pattern Classification

Calculate the 24-hour wind change:

`Delta V_24h = V(t+24h) - V(t)`

Then assign one of three classes:

| Class | Rule | Interpretation |
|---|---|---|
| Strengthening | `Delta V_24h >= +10 kt` | Wind intensity increases by at least 10 knots |
| Stable | `-10 kt < Delta V_24h < +10 kt` | Wind change remains within the stable interval |
| Weakening | `Delta V_24h <= -10 kt` | Wind intensity decreases by at least 10 knots |

The classification target is the future change category, not a prediction of cyclone formation or genesis.

## 2.3 Project Scope

**Included:**

1. Global IBTrACS v04r01 CSV data inspection.
2. Quality checks for observation flags, agencies, missing values, time stamps, cyclone identifiers, and geographic coordinates.
3. Creation of model-specific datasets for intensity, movement, and development classification.
4. Temporal training/validation/test separation.
5. Baseline model development and evaluation.
6. XGBoost model tuning and comparison against baselines.
7. Geodesic coordinate reconstruction and Haversine track-error evaluation.
8. Serialization, model loading, inference smoke testing, and reproducibility documentation.

**Not included in the current completed milestone:**

1. Satellite or radar image processing.
2. Cyclone-genesis detection.
3. Real-time data feeds or live operational forecasting.
4. Public warning or emergency alert delivery.
5. Integration with official meteorological agency systems.
6. A completed Streamlit front end or deployed public application.

---

# CHAPTER 3 — OBJECTIVES AND EXPECTED OUTCOMES

## 3.1 Primary Objectives

1. Study and prepare a reliable global historical cyclone dataset.
2. Formulate three prediction problems with clearly defined input and target variables.
3. Train machine learning models to predict wind intensity, cyclone displacement, and 24-hour intensity-development category.
4. Compare advanced gradient-boosted models with simple baselines and Random Forest models.
5. Evaluate model performance using metrics suited to each prediction task.
6. reduce temporal and storm-identity leakage during dataset partitioning.
7. Build reusable software modules for feature handling, coordinate conversion, and model inference.
8. Record the experimental procedure so that model training and evaluation can be reproduced.

## 3.2 Expected Outcomes

The expected deliverables are:

- Three processed Parquet datasets.
- Training and validation scripts for baseline and XGBoost models.
- Serialized model artifacts.
- Analytical reports documenting data audits, feature definitions, preprocessing, and evaluation.
- A reproducible Python environment specification.
- A future dashboard that can expose model predictions and historical cyclone information through an interactive interface.

---

# CHAPTER 4 — BACKGROUND AND LITERATURE BASIS

## 4.1 Tropical Cyclone Best-Track Data

A best-track dataset is a post-analysis record of a tropical cyclone's estimated centre position and intensity over time. IBTrACS combines historical and recent information from multiple meteorological agencies into a global archive. It provides a common framework for examining cyclone records across ocean basins and historical periods.

IBTrACS is valuable for a global machine learning study because it includes cyclone identifiers, season, time, position, basin, storm nature, and agency-specific intensity information. However, it is important to recognise that a best-track archive is not equivalent to an unprocessed stream of real-time sensor measurements. Best-track positions and intensities may reflect post-event analysis, agency practices, and interpolation.

## 4.2 Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees using randomized samples and feature subsets. For regression, individual tree predictions are combined, commonly by averaging. For classification, tree votes or averaged class scores are used to produce a class prediction. Random Forest provides a useful non-linear baseline and can model feature interactions without requiring a linear relationship between every input and target.

## 4.3 Extreme Gradient Boosting (XGBoost)

XGBoost is a gradient-boosted tree learning system. It builds an ensemble of decision trees sequentially, with later trees improving the fit to errors from earlier trees. It includes regularization and computational techniques intended to improve predictive performance and scalability. In this project, histogram-based tree construction is used and a controlled randomized search evaluates candidate hyperparameter configurations on the validation partition.

## 4.4 Geospatial Distance Evaluation

Latitude and longitude are angular coordinates on a spherical surface. A simple difference in degrees is not a consistent physical distance, and longitude behaves discontinuously at the antimeridian. The project therefore predicts local eastward and northward displacement components and uses a spherical Haversine calculation to evaluate the distance between the predicted and reference geographic locations.

## 4.5 Position of the Present Work

The project applies conventional supervised learning methods to three related cyclone prediction tasks using a global best-track database. The contribution is primarily in the formulation of separate targets, data-quality controls, temporal evaluation, geographic treatment of movement, and comparison of reproducible model baselines with XGBoost. The work should be interpreted as an experimental historical-data modelling project rather than an operational meteorological forecasting service.

---

# CHAPTER 5 — DATASET DESCRIPTION

## 5.1 Primary Dataset

**Dataset name:** International Best Track Archive for Climate Stewardship (IBTrACS)  
**Version:** v04r01  
**Subset:** Global / ALL  
**Format:** CSV  
**Source organisation:** NOAA National Centers for Environmental Information (NCEI)

**Official dataset page:** https://www.ncei.noaa.gov/products/international-best-track-archive  
**Project CSV source:** https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv

## 5.2 Dataset Size and Coverage

The downloaded global file inspected for this project contains:

| Property | Inspected value |
|---|---:|
| Data observations | 730,416 rows |
| Variables / columns | 174 |
| Unique cyclone identifiers | 13,631 |
| Earliest recorded observation | 1842-10-25 03:00 UTC |
| Latest observation in the inspected file | 2026-11-01 00:00 UTC |
| Season coverage reported by the inspection | 1842–2027 |
| Core coordinate completeness | LAT and LON fully populated in the inspected archive |

The archive is global and includes seven primary basin codes:

| Code | Basin |
|---|---|
| WP | Western North Pacific |
| SI | South Indian |
| NA | North Atlantic |
| SP | South Pacific |
| EP | Eastern North Pacific |
| NI | North Indian |
| SA | South Atlantic |

The season field is used for temporal splitting. This is important because Southern Hemisphere cyclone seasons can span two calendar years; therefore, season-based separation is more appropriate than treating every cyclone as belonging to a simple January-to-December period.

## 5.3 Important Dataset Fields

| Field | Description / use in project |
|---|---|
| `SID` | Unique cyclone identifier; used for grouping and leakage checks, not as a predictive feature |
| `SEASON` | Tropical cyclone season; used for chronological partitioning |
| `BASIN` | Global ocean basin identifier |
| `SUBBASIN` | Sub-region identifier where available |
| `NAME` | Cyclone name where assigned |
| `ISO_TIME` | Observation timestamp in UTC |
| `NATURE` | Combined storm nature/type information |
| `LAT`, `LON` | Best-track geographic coordinates |
| `USA_WIND` | USA-agency maximum sustained wind estimate used for intensity tasks |
| `USA_AGENCY` | Reporting agency/source stream used for consistency checks |
| `IFLAG` | Observation/interpolation flag string; first character is inspected for USA wind quality |
| `TRACK_TYPE` | Track classification used to retain main tracks for movement modelling |
| `DIST2LAND` | Distance to nearest land, used as a contextual feature |

## 5.4 Why USA_WIND Was Selected for Intensity Tasks

Wind estimates from different agencies may use different averaging periods and reporting procedures. Combining wind values from different sources without accounting for those differences can introduce inconsistency into the training target. The project therefore uses the USA wind series for intensity and development tasks, requires current and future observations to belong to the same `USA_AGENCY`, and filters the USA observation flag to the original/verified categories used in the project's audit (`IFLAG[0]` equal to `O` or `V`).

The implementation does not silently substitute WMO_WIND when USA_WIND is missing. This avoids mixing wind estimates whose averaging conventions may differ.

## 5.5 Best-Track Interpretation

IBTrACS records are consolidated best-track data. In particular, the position targets used by the movement model should be understood as post-analysis best-track positions rather than raw real-time sensor fixes. The model therefore learns patterns in the historical best-track archive and its reference conventions.

---

# CHAPTER 6 — DATA INSPECTION AND QUALITY AUDITING

## 6.1 Initial Dataset Inspection

The first stage examined the CSV structure, number of records, data types, missing values, temporal span, unique cyclone identifiers, coordinate completeness, and basin distribution. The IBTrACS CSV contains a units row below the header; this row must be skipped when parsing so that numeric columns are interpreted correctly.

The parser also preserves the literal basin code `NA`. This is important because default missing-value parsing can interpret the string `NA` as a null value, even though it is the valid code for the North Atlantic basin.

## 6.2 Interpolation Flag Audit

The project audited the 15-character `IFLAG` field to distinguish original/verified observations from interpolated records. For USA wind, the first flag character is used. In the post-1980 cohort examined by the audit:

| USA wind observation type | Count | Share of available USA wind fixes |
|---|---:|---:|
| Original / verified (`O` or `V`) | 133,085 | 50.61% |
| Interpolated (`P` or `I`) | 129,859 | 49.39% |

This audit matters because treating interpolated values as independent observations can make the model learn the archive's interpolation process instead of only observed agency reports. The intensity and development datasets therefore use original/verified USA wind observations for the current and relevant future target records.

## 6.3 Agency Consistency

For wind-based tasks, historical lag values and future target values are matched within the same cyclone (`SID`) and the same reporting agency stream (`USA_AGENCY`). A value from a different agency is not treated as a direct continuation of the same wind series. This reduces unintended changes caused by switching reporting sources.

## 6.4 Movement Observation Frequency

The movement model uses the main track (`TRACK_TYPE == 'main'`) and a standard six-hourly UTC synoptic grid: 00:00, 06:00, 12:00, and 18:00 UTC. Intermediate off-synoptic points are excluded from this model's principal sample construction to reduce the risk of learning values created by interpolation routines rather than observed synoptic track behaviour.

## 6.5 Duplicate and Coordinate Checks

The preprocessing audit reports that the composite key `(SID, ISO_TIME)` was unique in the raw archive, with no unexpected duplicates found in the inspected data. Latitude and longitude were available for the full raw observation set. Longitudes were normalised for model use to the interval [-180°, +180°] to handle the antimeridian consistently.

## 6.6 Quality Audit Conclusion

The inspection and audit stages informed the model-specific cohorts and filters. They are an important part of the system because the quality and meaning of the training labels directly affect the validity of subsequent model comparisons.

---

# CHAPTER 7 — DATA PREPROCESSING AND FEATURE ENGINEERING

## 7.1 Preprocessing Pipeline Overview

The raw global CSV is converted into three model-specific Parquet datasets:

1. `processed/model1_intensity.parquet`
2. `processed/model2_movement.parquet`
3. `processed/model3_development.parquet`

The preprocessing script is `scripts/prepare_datasets.py`. Processed datasets are validated using `scripts/validate_processed_datasets.py`.

The pipeline handles the IBTrACS units row, preserves valid categorical basin values, checks cyclone/time records, applies task-specific temporal and observation filters, creates historical lag features, constructs future targets, assigns train/validation/test splits, and stores model-ready tables in Parquet format.

## 7.2 Temporal and Cyclone-Level Data Splitting

Random row-level splitting is avoided because observations from the same cyclone are correlated. If records from the same cyclone were placed in both training and test datasets, the test score could be overly optimistic.

The project uses season-based chronological partitions and verifies that a cyclone identifier does not cross partitions:

| Task | Training seasons | Validation seasons | Test seasons |
|---|---|---|---|
| Model 1 — Intensity | 1980–2014 | 2015–2019 | 2020 onward |
| Model 2 — Movement | 1970–2014 | 2015–2019 | 2020 onward |
| Model 3 — Development | 1980–2014 | 2015–2019 | 2020 onward |

The split is based on the IBTrACS `SEASON` field. No cyclone ID crossover between the three partitions was reported by the validation workflow.

## 7.3 Model 1 Preprocessing — Intensity

- Historical cohort: season 1980 onward.
- Target field: `USA_WIND` at the future observation.
- Horizons: +6h, +12h, and +24h.
- Current and future wind records must pass the original/verified IFLAG condition.
- Current and future records must use the same USA agency stream.
- Backward wind lags are matched by cyclone identifier, exact timestamp, and reporting agency.
- Missing lag information is retained with availability indicators rather than being mistaken for a measured zero.
- Future-derived variables are excluded from the feature matrix.

Main feature families include current wind intensity, past 6-hour and 12-hour wind values, past intensity changes, lag-availability flags, current latitude, normalised longitude, distance to land, basin, storm nature, month, and hour.

The report records the following lag availability:

| Feature | Available | Missing | Availability |
|---|---:|---:|---:|
| `HAS_PAST_USA_WIND_6H` | 338,434 | 13,499 | 96.16% |
| `HAS_PAST_USA_WIND_12H` | 325,270 | 26,663 | 92.42% |

## 7.4 Model 2 Preprocessing — Movement

- Historical cohort: season 1970 onward.
- Track type: main tracks only.
- Sampling: six-hourly synoptic UTC observations.
- Forecast horizons: +6h, +12h, and +24h.
- Current and target locations are joined within the same cyclone track.
- Longitudes are normalised to [-180°, +180°].
- Backward eastward and northward displacements are included as motion-history features.
- Future coordinate and displacement values are stored only as target fields and are not used as predictors.

The main input feature families include current latitude and normalised longitude, previous 6-hour and 12-hour east/north displacement, distance to land, basin, storm nature, month, and UTC hour.

### Local Tangent-Plane Targets

The model predicts local orthogonal displacement components:

- `TARGET_EAST_DISPLACEMENT_KM` (dx)
- `TARGET_NORTH_DISPLACEMENT_KM` (dy)

These components allow a multi-output regression formulation. The predicted displacement is then converted to geographic coordinates using a spherical-earth approximation and longitude wrapping. Great-circle Haversine distance is used for final error evaluation.

The preprocessing report compares planar displacement magnitude with spherical Haversine distance:

| Horizon | Mean tangent distance | Mean Haversine distance | Mean absolute difference |
|---|---:|---:|---:|
| +6h | 110.98 km | 110.98 km | 0.0024 km |
| +12h | 218.39 km | 218.38 km | 0.0172 km |
| +24h | 424.95 km | 424.84 km | 0.1148 km |

This comparison validates the local approximation over the sampled movement targets while retaining Haversine distance as the evaluation standard.

## 7.5 Model 3 Preprocessing — Development Classification

- Historical cohort: season 1980 onward.
- Prediction horizon: +24 hours.
- Current and future wind values must be original/verified and from the same USA agency stream.
- The continuous 24-hour change in USA wind is converted into the three target categories using the ±10-knot rule.
- Future wind and the computed future change are target-only fields and are not passed into the classifier as inputs.
- Backward lag wind values and their availability flags are constructed with the same-agency matching rule used for Model 1.

The observed class distribution in the final Model 3 dataset is:

| Class | Samples | Share |
|---|---:|---:|
| Stable | 45,842 | 41.71% |
| Strengthening | 36,367 | 33.09% |
| Weakening | 27,692 | 25.20% |
| **Total** | **109,901** | **100%** |

The largest class is approximately 1.66 times the smallest class. This is not perfectly balanced, but it is sufficiently less skewed than a highly imbalanced classification problem. Accuracy is therefore reported alongside macro-F1 and balanced accuracy.

## 7.6 Missing-Value Treatment

Missing historical lags are legitimate for some observations, particularly near cyclone genesis or at agency handover. They are not silently interpreted as real zero wind or zero motion. The feature manifest defines binary availability indicators, and the model pipelines handle missing values according to the estimator's requirements. The data validation report records zero missing values in the final target columns.

## 7.7 Leakage Prevention

The following controls are used:

1. Future target columns are kept out of the predictor matrix.
2. Storm IDs and timestamps are retained for grouping and auditing, not as ordinary predictive features.
3. `LANDFALL` is excluded because it can contain forward-looking information.
4. Centred `STORM_SPEED` and `STORM_DIR` are excluded because they may use future track fixes.
5. Raw longitude is not used as a direct numeric feature; normalised longitude is used to avoid an artificial boundary at the antimeridian.
6. Preprocessing transformations are fitted on training data only and applied to validation/test data.
7. Hyperparameter selection uses validation data; final reported metrics are calculated on the held-out test partition.

---

# CHAPTER 8 — MACHINE LEARNING METHODOLOGY

## 8.1 Baseline Models

Simple baselines establish a minimum reference level that a more complex model should outperform.

| Prediction task | Simple baseline | Purpose |
|---|---|---|
| Intensity | Persistence | Assumes future wind remains equal to current wind |
| Movement | Zero motion and constant velocity | Provides no-motion and linear extrapolation references |
| Development | Majority-class classifier | Predicts the most frequent class, Stable |

The project also evaluates conventional machine learning models, including Linear Regression, Random Forest regression/classification, and XGBoost.

## 8.2 Random Forest Models

Random Forest models were trained as non-linear baselines for the three tasks. For movement prediction, the model estimates the eastward and northward displacement outputs. For classification, a Random Forest classifier predicts the three 24-hour development classes.

The baseline training and evaluation workflow is implemented in `scripts/train_baselines.py`. The baseline models are saved using Joblib in `models/baselines/` in the development environment. The repository excludes the large baseline model binaries and provides instructions for regenerating them.

## 8.3 XGBoost Models

XGBoost was introduced as the advanced model family. The experiments use histogram-based tree construction (`tree_method=hist`) and a controlled randomized search over selected hyperparameters.

The report specifies a search across candidate values for:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`

Eight candidate configurations are evaluated on the validation partition in the reported experiment. The validation data is used for selection, and the test partition is held out for final evaluation.

## 8.4 Model 1 — Intensity Regression

Separate horizon-specific regression models estimate future wind speed at +6h, +12h, and +24h. The target is expressed in knots. The primary reported metric is Mean Absolute Error (MAE), supported by RMSE and mean bias error in the experiment logs.

## 8.5 Model 2 — Multi-Output Movement Regression

Movement is formulated as a two-output regression problem. One output predicts eastward displacement and the other predicts northward displacement. The predictions are used to reconstruct future latitude and longitude. The final score is based on Haversine error in kilometres rather than a raw degree difference.

## 8.6 Model 3 — Three-Class Classification

The classifier produces one of three labels: Strengthening, Stable, or Weakening. The project evaluates accuracy, macro-F1, and balanced accuracy. Macro-F1 gives equal weight to each class and helps make the evaluation less dependent on the most frequent category.

## 8.7 Model Artifact Serialisation

Trained pipelines are stored in Joblib format. A custom classifier wrapper originally defined in a training script was moved into the importable package `src/cyclone_ml/estimators.py` so that the serialized model can locate the class during loading in a separate process. The package also contains geodesy and feature transformation modules.

---

# CHAPTER 9 — EXPERIMENTAL SETUP AND EVALUATION METRICS

## 9.1 Experimental Partitions

The final test partition contains cyclone seasons from 2020 onward. Training and validation use earlier seasons. This approximates the practical problem of learning from historical records and evaluating on later cyclone seasons.

| Dataset | Train rows | Validation rows | Test rows | Total rows |
|---|---:|---:|---:|---:|
| Model 1 — Intensity | 270,694 | 40,801 | 40,438 | 351,933 |
| Model 2 — Movement | 421,827 | 46,857 | 47,402 | 516,086 |
| Model 3 — Development | 84,531 | 12,758 | 12,612 | 109,901 |

The Model 1 row count combines the 6-hour, 12-hour, and 24-hour horizon records. Model 2 is also stored as horizon-specific rows. Model 3 is a single +24-hour classification dataset.

## 9.2 Regression Metrics

### Mean Absolute Error

`MAE = (1/n) * sum(|y_i - y_hat_i|)`

MAE is the average absolute difference between the model's prediction and the true value. It is reported in knots for intensity and kilometres for movement.

### Root Mean Squared Error

`RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))`

RMSE penalises larger errors more strongly than MAE.

### Skill Relative to a Baseline

For an error metric where lower is better:

`Skill (%) = (1 - Model_Error / Baseline_Error) * 100`

A positive skill score means the model error is lower than the selected baseline error.

### Haversine Distance

For two geographic points, the Haversine formula calculates the great-circle distance on a sphere. In this project it is used to calculate the distance, in kilometres, between the predicted future cyclone location and the reference IBTrACS location.

## 9.3 Classification Metrics

### Accuracy

Accuracy is the fraction of samples for which the predicted class equals the actual class.

### Macro-F1

Macro-F1 calculates the F1-score separately for each class and averages the three scores equally. This gives each of Strengthening, Stable, and Weakening the same weight in the final metric.

### Balanced Accuracy

Balanced accuracy is the average recall across the classes. It is useful when class frequencies are not identical.

## 9.4 Evaluation Discipline

The test data is not used to select hyperparameters. Baseline and XGBoost results are compared on the same held-out temporal test period. Results should be interpreted in the context of best-track data quality, archive coverage, and the limitations of using historical track and intensity variables without a complete set of atmospheric environmental predictors.

---

# CHAPTER 10 — RESULTS AND DISCUSSION

## 10.1 Master Benchmark Results

The following table summarises the reported test results for the selected XGBoost models compared with simple baselines and Random Forest models.

| Model | Forecast horizon | Simple baseline | Random Forest | XGBoost | XGBoost skill vs. simple baseline |
|---|---:|---:|---:|---:|---:|
| Intensity MAE | +6h | 4.02 kt | 3.52 kt | **3.43 kt** | **14.83%** |
| Intensity MAE | +12h | 7.50 kt | 5.92 kt | **5.71 kt** | **23.93%** |
| Intensity MAE | +24h | 13.31 kt | 10.01 kt | **9.58 kt** | **27.98%** |
| Movement Haversine error | +6h | 38.8 km | 37.0 km | **36.6 km** | **5.61%** |
| Movement Haversine error | +12h | 87.2 km | 80.5 km | **79.6 km** | **8.71%** |
| Movement Haversine error | +24h | 208.4 km | 184.1 km | **181.6 km** | **12.87%** |
| Development classification | +24h | 41.30% majority accuracy | 62.71% accuracy / 0.631 macro-F1 | **63.42% accuracy / 0.638 macro-F1** | **22.12% relative accuracy skill** |

The classification comparison is also reported as a +0.71 percentage-point accuracy gain over Random Forest. Skill percentages use the project's stated baseline definitions; they should not be confused with percentage-point changes.

## 10.2 Model 1 — Intensity Prediction Results

| Horizon | Persistence MAE | Random Forest MAE | XGBoost MAE | XGBoost skill vs. persistence |
|---|---:|---:|---:|---:|
| +6h | 4.02 kt | 3.52 kt | **3.43 kt** | 14.83% |
| +12h | 7.50 kt | 5.92 kt | **5.71 kt** | 23.93% |
| +24h | 13.31 kt | 10.01 kt | **9.58 kt** | 27.98% |

**Discussion:** Error increases with the forecast horizon, which is expected because uncertainty in cyclone intensity evolution accumulates over time. XGBoost reduces MAE relative to both persistence and Random Forest at all three horizons. The largest relative improvement over persistence occurs at +24 hours, but the absolute error is also largest at this horizon.

The experiment's feature-importance analysis for the +24-hour intensity model reports current wind, past wind, and recent wind change among the leading features. This indicates that recent intensity history is informative to the trained model. Feature importance is a model diagnostic, not proof of physical causation.

## 10.3 Model 2 — Movement Prediction Results

| Horizon | Constant-velocity error | Random Forest error | XGBoost error | XGBoost skill vs. constant velocity |
|---|---:|---:|---:|---:|
| +6h | 38.8 km | 37.0 km | **36.6 km** | 5.61% |
| +12h | 87.2 km | 80.5 km | **79.6 km** | 8.71% |
| +24h | 208.4 km | 184.1 km | **181.6 km** | 12.87% |

**Discussion:** The mean location error increases as the forecast horizon grows. XGBoost gives a modest improvement over Random Forest at each horizon and improves on the constant-velocity reference. The difference between XGBoost and Random Forest is relatively small compared with the overall error, so the result should be described as an incremental improvement rather than a major performance leap.

The use of east/north displacement targets avoids direct regression across a longitude discontinuity. The reconstructed position is evaluated geographically using Haversine distance.

## 10.4 Model 3 — Development Classification Results

| Metric | Majority baseline | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 41.30% | 62.71% | **63.42%** |
| Macro-F1 | 0.195 (reported baseline) | 0.631 | **0.638** |
| Balanced accuracy | 33.33% (majority baseline) | 63.15% | **63.68%** |

**Discussion:** The XGBoost classifier outperforms the majority-class baseline and gives a modest improvement over Random Forest. Accuracy alone does not provide a complete picture of a multi-class problem; macro-F1 and balanced accuracy are also reported to account for performance across all three categories.

The selected XGBoost classifier has the following test-set class-wise results:

| Class | Precision | Recall | F1-score | Test samples |
|---|---:|---:|---:|---:|
| Strengthening | 0.608 | 0.599 | 0.603 | 4,111 |
| Stable | 0.620 | 0.636 | 0.628 | 5,209 |
| Weakening | 0.692 | 0.675 | 0.683 | 3,292 |
| **Macro average** | **0.640** | **0.637** | **0.638** | **12,612** |

The corresponding test confusion matrix (rows = true class, columns = predicted class) is:

| True class | Predicted Strengthening | Predicted Stable | Predicted Weakening |
|---|---:|---:|---:|
| Strengthening | 2,463 | 1,352 | 296 |
| Stable | 1,202 | 3,313 | 694 |
| Weakening | 389 | 680 | 2,223 |

The model's class probabilities are prediction scores for the three categories. The fact that they sum to one is a normalisation property and does not, by itself, demonstrate probability calibration. Formal calibration analysis would require additional evaluation, for example reliability diagrams or calibration-sensitive scores.

## 10.5 Overall Interpretation

The experimental results indicate that XGBoost is the best-performing tested model family across the three tasks on the selected temporal test partition. The size of the improvement varies:

- Intensity: consistent improvement over Random Forest across all forecast horizons.
- Movement: small but consistent improvement over Random Forest.
- Development classification: a modest improvement in accuracy and macro-F1 over Random Forest.

These results support the use of XGBoost as the default model family for the next application-integration phase. They do not prove that the models are suitable for operational warning or safety-critical decisions.

---

# CHAPTER 11 — SOFTWARE ARCHITECTURE AND IMPLEMENTATION

## 11.1 Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data manipulation | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Model persistence | Joblib |
| Processed dataset format | Parquet / PyArrow |
| Geospatial calculations | Project geodesy module and Haversine distance |
| Source control | Git and GitHub |
| Planned dashboard | Streamlit (not yet completed in the repository at this report milestone) |

## 11.2 Project Repository Structure

```text
Cyclone-Identification-Classification-and-Prediction-System-using-IBTrACS/
│
├── models/
│   ├── baselines/                  # Reproducible baseline artifact instructions;
│   │                               # heavy binaries are excluded from Git
│   └── xgboost/                    # Seven committed XGBoost artifacts
│
├── processed/
│   ├── model1_intensity.parquet
│   ├── model2_movement.parquet
│   └── model3_development.parquet
│
├── raw/
│   └── README.md                  # Instructions for obtaining the source CSV
│
├── reports/
│   ├── dataset_inspection.md
│   ├── modelling_cohort_audit.md
│   ├── iflag_and_basin_validation.md
│   ├── preprocessing_report.md
│   ├── model_feature_manifest.md
│   ├── baseline_model_results.md
│   ├── xgboost_experiment_results.md
│   └── model_portability_validation.md
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── audit_modelling_cohort.py
│   ├── audit_iflag_and_basins.py
│   ├── prepare_datasets.py
│   ├── validate_processed_datasets.py
│   ├── train_baselines.py
│   ├── train_xgboost.py
│   └── test_model_loading.py
│
├── src/
│   └── cyclone_ml/
│       ├── __init__.py
│       ├── estimators.py
│       ├── geodesy.py
│       └── features.py
│
├── requirements-models.txt
├── README.md
└── .gitignore
```

## 11.3 Main Scripts

| Script | Function |
|---|---|
| `inspect_dataset.py` | Inspects the source IBTrACS CSV and reports dataset properties |
| `audit_modelling_cohort.py` | Examines suitable historical cohorts and target feasibility |
| `audit_iflag_and_basins.py` | Audits original/interpolated observations and basin representation |
| `prepare_datasets.py` | Generates the three model-specific Parquet datasets |
| `validate_processed_datasets.py` | Checks dataset structure, splits, targets, and lag construction |
| `train_baselines.py` | Trains and evaluates simple and Random Forest/linear baselines |
| `train_xgboost.py` | Runs XGBoost training, validation-based tuning, and test evaluation |
| `test_model_loading.py` | Loads serialized models and performs inference smoke tests |

## 11.4 Reusable Python Package

The `src/cyclone_ml/` package separates reusable functionality from the command-line training scripts:

- `estimators.py` contains the importable custom classifier wrapper required for portable model deserialization.
- `geodesy.py` contains coordinate reconstruction and geographic distance utilities.
- `features.py` contains shared feature transformations, including cyclical time transformations.
- `__init__.py` defines the package interface.

## 11.5 Data and Model Storage

The processed datasets are stored in Parquet format to make repeated analysis more efficient than re-reading and reparsing the full CSV file. Seven XGBoost artifacts are committed to the public GitHub repository. Large baseline Random Forest model binaries are excluded from Git because of their size; the project includes training scripts and documentation to regenerate them.

The raw IBTrACS CSV is also excluded from Git. A copy should be downloaded from the official NOAA NCEI source using the instructions in `raw/README.md`.

---

# CHAPTER 12 — MODEL PORTABILITY AND REPRODUCIBILITY

## 12.1 Portability Validation

A separate validation stage tested the serialized model artifacts outside the original training execution. The recorded portability audit reports:

| Check | Result |
|---|---:|
| Model artifacts audited | 18 |
| Successfully loaded | 18 / 18 |
| Successful smoke inference | 18 / 18 |
| Reported environment | Python 3.13.14, Windows AMD64 |

The 18 audited artifacts consist of 11 baseline models and 7 XGBoost models. The portability issue involving the custom `CycloneClassifier` class was resolved by moving the class into the importable `cyclone_ml.estimators` module. The model artifacts were migrated without retraining, according to the portability report.

## 12.2 Reproduction Workflow

The repository documents the following high-level workflow:

1. Clone the GitHub repository.
2. Create and activate a Python virtual environment.
3. Install the pinned modelling dependencies.
4. Download the raw IBTrACS CSV into the expected `raw/` location.
5. Run the data inspection and audit scripts as required.
6. Generate the processed Parquet datasets.
7. Validate the processed datasets.
8. Regenerate excluded baseline model artifacts.
9. Run baseline and XGBoost training experiments.
10. Run the model-loading and inference smoke test.

Example commands from the repository:

```powershell
# Clone project
git clone https://github.com/NannawarePranav/Cyclone-Identification-Classification-and-Prediction-System-using-IBTrACS.git
cd Cyclone-Identification-Classification-and-Prediction-System-using-IBTrACS

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install modelling dependencies
pip install -r requirements-models.txt

# Prepare and validate model-ready data
python scripts/prepare_datasets.py
python scripts/validate_processed_datasets.py

# Recreate baseline models if they are not present
python scripts/train_baselines.py --model all

# Run XGBoost experiments
python scripts/train_xgboost.py --model all --n-iter 8

# Validate serialized model loading and inference
python scripts/test_model_loading.py
```

The exact command-line options and expected source-file locations should be checked against the current scripts and README before executing the full reproduction workflow. A fresh clone requires the source CSV, and the excluded baseline models must be regenerated before a test that expects all 18 artifacts can pass.

## 12.3 Version Information

The portability report records Python 3.13.14 on 64-bit Windows AMD64. The dependency manifest pins the numerical, dataframe, machine-learning, serialization, and Parquet libraries. The exact pins are maintained in `requirements-models.txt`; that file should be treated as the authoritative environment specification for the committed model artifacts.

---

# CHAPTER 13 — LIMITATIONS AND CHALLENGES

## 13.1 Historical Best-Track Dependence

The models learn from historical best-track records. These records are post-analysis products and may contain interpolation, agency differences, or revised positions and intensities. The predictions should not be interpreted as forecasts generated from a live operational observation stream.

## 13.2 Lack of Environmental Predictors

The current feature sets primarily use cyclone history, current position, intensity history, basin, storm nature, time, and distance to land. They do not yet include a comprehensive set of atmospheric and oceanic predictors such as environmental wind shear, sea-surface temperature, humidity, pressure fields, or upper-level steering flow. These variables may improve future models but require additional data sources and careful temporal alignment.

## 13.3 Agency and Averaging-Period Differences

Wind estimates may differ across agencies and measurement conventions. The project reduces one source of inconsistency by using the same USA agency stream for current and target wind records. However, agency practices and historical coverage can still affect the representativeness of the available data.

## 13.4 Uneven Basin Representation

The global data contains records from all seven basin codes, but the sample counts are not equal. The South Atlantic basin has far fewer usable samples than the Western North Pacific or North Atlantic. Aggregate global metrics can hide regional performance differences. Basin-level reporting is therefore important, and conclusions for sparsely represented basins should be treated cautiously.

## 13.5 Forecast-Horizon Uncertainty

Intensity and movement errors increase with forecast horizon. A +24-hour estimate is more uncertain than a +6-hour estimate, and a single point prediction cannot capture the full range of possible future cyclone behaviour.

## 13.6 Movement Approximation

The movement model uses local tangent-plane displacement targets for regression and spherical Haversine distance for evaluation. The preprocessing audit shows small average discrepancies between planar and Haversine distance, but the local approximation is not a complete model of atmospheric motion or a replacement for numerical weather prediction.

## 13.7 Classification Thresholds

Strengthening, Stable, and Weakening are defined using a fixed ±10-knot threshold over 24 hours. This is a project-defined target formulation. It simplifies continuous intensity change and does not represent every possible operational definition of rapid intensification or weakening.

## 13.8 Probability Calibration

The classifier produces class probability outputs, but normalized probabilities are not necessarily calibrated probabilities. Calibration has not been established by the current smoke tests and should be evaluated separately before probabilistic interpretation.

## 13.9 No Operational Deployment

The current milestone does not include live data ingestion, real-time inference scheduling, system monitoring, alert generation, or deployment for public use. The project is an academic machine learning prototype and must not be used as the sole basis for safety decisions.

---

# CHAPTER 14 — FUTURE SCOPE

The following improvements are possible extensions and are not claimed as completed features:

1. **Interactive Streamlit dashboard:** Develop an interface with an overview page, intensity forecast, track movement map, development classification, historical cyclone replay, and model-performance visualisations.
2. **Historical cyclone exploration:** Add filtering by cyclone name, SID, season, and basin, with historical track visualisation.
3. **Environmental data integration:** Evaluate reanalysis datasets such as ERA5 for variables including wind shear, sea-surface temperature, humidity, and steering flow, provided that temporal and spatial alignment can be implemented correctly.
4. **Additional model families:** Compare gradient boosting with other suitable time-series and sequence models after creating a defensible temporal evaluation setup.
5. **Uncertainty estimation:** Add prediction intervals or ensembles for regression and calibrated probabilities for classification.
6. **Basin-specific evaluation:** Perform deeper regional analysis and assess whether global models or basin-specific models are more appropriate.
7. **Ablation studies:** Measure the contribution of individual feature groups such as intensity history, movement history, location, basin, and distance to land.
8. **Robustness testing:** Test sensitivity to missing inputs, agency transitions, rare basins, extreme intensities, and unusual movement patterns.
9. **Deployment and monitoring:** If the project is later connected to operational data, build monitoring, versioning, audit logs, and domain-expert review into the system.
10. **Official forecast comparison:** Compare model outputs with suitable official or research forecast baselines only after matching forecast initialisation times, lead times, and evaluation samples.

---

# CHAPTER 15 — CONCLUSION

This project developed a historical-data machine learning framework for three tropical cyclone prediction problems using NOAA IBTrACS v04r01. The work began with a detailed inspection of the global dataset and proceeded through observation-quality auditing, model-specific preprocessing, feature engineering, baseline training, XGBoost experiments, and model portability validation.

The system treats intensity, movement, and development category as distinct prediction tasks. The intensity model estimates future USA wind speed at +6, +12, and +24 hours. The movement model predicts local eastward and northward displacement and evaluates reconstructed positions with Haversine distance. The classification model assigns a 24-hour wind-change category using the project's ±10-knot threshold.

The experiments show that XGBoost achieved the best reported test performance among the evaluated model families. It achieved intensity MAE values of 3.43, 5.71, and 9.58 knots for +6, +12, and +24 hours; movement errors of 36.6, 79.6, and 181.6 km; and 63.42% accuracy with 0.638 macro-F1 for development classification. The improvement over Random Forest was consistent but modest for movement and classification, while intensity prediction showed a clearer reduction in MAE.

The project also establishes important safeguards for historical cyclone modelling: consistent wind-source matching, original/verified observation filtering, storm-level temporal partitions, exclusion of future-derived fields, correct handling of geographic longitude, and evaluation on later seasons. The portability audit reports successful loading and smoke inference for all 18 baseline and XGBoost artifacts in a fresh Python process.

The completed work provides a reproducible foundation for a college-level machine learning project. The next major step is the planned Streamlit interface, followed by further evaluation and possible integration of environmental datasets. The results remain experimental and should not be interpreted as official or operational cyclone forecasts.

---

# REFERENCES

1. Gahtan, J., Knapp, K. R., Schreck, C. J. III, Diamond, H. J., Kossin, J. P., and Kruk, M. J. (2024). *International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4.01*. NOAA National Centers for Environmental Information. https://doi.org/10.25921/82ty-9e16
2. NOAA National Centers for Environmental Information. *International Best Track Archive for Climate Stewardship (IBTrACS).* https://www.ncei.noaa.gov/products/international-best-track-archive
3. NOAA NCEI. *IBTrACS v04r01 Column Documentation.* https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
5. Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
6. scikit-learn developers. *Model Evaluation: Quantifying the Quality of Predictions.* https://scikit-learn.org/stable/modules/model_evaluation.html
7. Project source repository. *Cyclone Identification, Classification, and Prediction System using IBTrACS.* https://github.com/NannawarePranav/Cyclone-Identification-Classification-and-Prediction-System-using-IBTrACS
8. Project report: `reports/dataset_inspection.md` — IBTrACS Global Tropical Cyclone Dataset Inspection Report.
9. Project report: `reports/modelling_cohort_audit.md` — IBTrACS Modelling Cohort & Feature Feasibility Audit.
10. Project report: `reports/iflag_and_basin_validation.md` — IBTrACS Original Observation & Basin Representation Audit.
11. Project report: `reports/preprocessing_report.md` — IBTrACS Model-Specific Dataset Preprocessing Report.
12. Project report: `reports/model_feature_manifest.md` — IBTrACS Machine Learning Feature and Target Manifest.
13. Project report: `reports/baseline_model_results.md` — Baseline Machine Learning Model Evaluation Report.
14. Project report: `reports/xgboost_experiment_results.md` — XGBoost Experiments and Model Improvement Report.
15. Project report: `reports/model_portability_validation.md` — Model Artifact Portability and Inference Validation Report.

---

# APPENDIX A — PROJECT REPOSITORY AND SOURCE FILES

**GitHub repository:**  
https://github.com/NannawarePranav/Cyclone-Identification-Classification-and-Prediction-System-using-IBTrACS

**Important source files:**

- `scripts/inspect_dataset.py`
- `scripts/audit_modelling_cohort.py`
- `scripts/audit_iflag_and_basins.py`
- `scripts/prepare_datasets.py`
- `scripts/validate_processed_datasets.py`
- `scripts/train_baselines.py`
- `scripts/train_xgboost.py`
- `scripts/test_model_loading.py`
- `src/cyclone_ml/estimators.py`
- `src/cyclone_ml/geodesy.py`
- `src/cyclone_ml/features.py`

# APPENDIX B — COMPLETED DELIVERABLES AT REPORT DATE

| Deliverable | Status at report preparation |
|---|---|
| Global IBTrACS data inspection | Completed |
| Modelling cohort audit | Completed |
| IFLAG and basin validation | Completed |
| Model-specific preprocessing | Completed |
| Processed dataset validation | Completed |
| Baseline model training and evaluation | Completed |
| XGBoost experiments and evaluation | Completed |
| Model artifact portability validation | Completed locally; repository contains the seven XGBoost artifacts |
| Public GitHub repository | Created and pushed to `main` |
| Streamlit dashboard | Planned; not yet part of the repository at this milestone |
| Application deployment | Not completed |

# APPENDIX C — ITEMS TO COMPLETE BEFORE FINAL COLLEGE SUBMISSION

- [ ] Replace all guide, HOD, principal, academic-year, and date placeholders.
- [ ] Confirm the exact department and university formatting required by the institute.
- [ ] Add the college logo and approved cover-page layout, if required.
- [ ] Add actual screenshots from the data inspection, model evaluation, and application once the dashboard is implemented.
- [ ] Add charts generated from the committed experimental results: target distribution, train/validation/test split, intensity performance by horizon, movement error by horizon, and classification confusion matrix.
- [ ] Refresh the table of contents and page numbers after formatting.
- [ ] Confirm all reported metrics against the final committed experiment reports before printing or submitting.
- [ ] Update the report after completing the Streamlit dashboard so that planned functionality is not described as completed.

---

**End of Report**
