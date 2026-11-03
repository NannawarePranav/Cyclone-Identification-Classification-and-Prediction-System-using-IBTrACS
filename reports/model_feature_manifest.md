# IBTrACS Machine Learning Feature and Target Manifest

> **Project:** Prediction of Different Tropical Cyclones Patterns  
> **Source Dataset:** NOAA IBTrACS v04r01 Global Best-Track Database  
> **Target Models:**
> 1. **Model 1:** Tropical Cyclone Intensity Prediction (Regression: Future Maximum Sustained Wind Speed)
> 2. **Model 2:** Tropical Cyclone Movement Prediction (Regression: Future Position Coordinates & Geodesic Displacements)
> 3. **Model 3:** Tropical Cyclone Development Pattern Classification (Multi-Class: Strengthening, Stable, Weakening)
>
> **Associated Processed Parquet Files:**
> - `processed/model1_intensity.parquet`
> - `processed/model2_movement.parquet`
> - `processed/model3_development.parquet`

---

## 1. Architectural Principles & Anti-Leakage Guardrails

To ensure absolute scientific validity, physical consistency, and zero target leakage across downstream modeling, all three datasets adhere to the following universal architectural rules:

1. **Strict Lookahead Prohibition (Zero Future Inputs):**
   - No feature derived from $t > t_{\text{current}}$ may ever enter a feature matrix.
   - For Model 1 and Model 3, all future intensity observations (`TARGET_USA_WIND`, `TARGET_USA_WIND_24H`, `WIND_CHANGE_24H`) and classification labels are strictly quarantined as target outputs.
   - For Model 2, all future tracking fields (`TARGET_LAT`, `TARGET_LON`, `TARGET_LON_NORM`, `TARGET_EAST_DISPLACEMENT_KM`, `TARGET_NORTH_DISPLACEMENT_KM`, `TARGET_TOTAL_DISTANCE_KM`, `TARGET_HAVERSINE_DISTANCE_KM`) are strictly quarantined as target outputs.

2. **Prohibition of Non-Predictive Identifiers:**
   - Primary tracking identifiers (`SID`, `ISO_TIME`, `SPLIT`) must **never** be passed as continuous or nominal numeric features. 
   - Feeding storm IDs (`SID`) causes machine learning algorithms (e.g. gradient boosted trees or neural networks) to memorize individual historical storm trajectories rather than learning generalized atmospheric dynamics.
   - Feeding raw UTC timestamps (`ISO_TIME`) introduces artificial ordinal trend biases. Temporal dynamics must be represented exclusively through seasonal cycle features (`MONTH`, `HOUR`).

3. **Coordinate Normalization Standard (`LON_NORM` vs. `LON`):**
   - IBTrACS natively reports longitudes in degrees East $[0^\circ, 360^\circ]$, with numerous historical records exceeding $180^\circ$ East in the Western Hemisphere.
   - Downstream models must strictly ingest `LON_NORM` (normalized to $[-180.0^\circ, +180.0^\circ]$) rather than raw `LON`.
   - Antimeridian crossing deltas are wrapped geodesically to prevent artificial discontinuities when storms cross $180^\circ$.

4. **Agency Reporting Stream Consistency:**
   - In Model 1 and Model 3, all current ($t$), historical lag ($t-6h, t-12h$), and target ($t+h$) wind values are verified to originate from the identical operational reporting stream (`USA_AGENCY`, typically JTWC or NHC) and require original observations (`IFLAG[0] in ['O', 'V']`). 
   - Cross-agency wind comparisons are strictly prohibited to prevent artificial intensity jumps arising from differing agency averaging intervals (e.g., 1-minute vs 10-minute sustained wind standards).

5. **Exclusion of Contaminated Post-Analysis Fields:**
   - Raw IBTrACS fields `LANDFALL`, `STORM_SPEED`, and `STORM_DIR` are strictly barred from all feature sets.
   - `LANDFALL` contains forward-looking information regarding future landfall timing.
   - Raw `STORM_SPEED` and `STORM_DIR` are centered difference quantities calculated using forward track fixes ($t+3h$ or $t+6h$), creating direct target leakage. Past kinematic motion must be captured solely through the backward lag displacements provided (`PAST_EAST_DISP_*`, `PAST_NORTH_DISP_*`).

---

## 2. Model 1 Manifest: Tropical Cyclone Intensity Prediction

### 2.1 Task Overview
- **Objective:** Predict the future maximum sustained wind speed (1-minute mean, in knots) of a tropical cyclone at discrete forward horizons.
- **Task Type:** Multi-horizon Continuous Regression.
- **Prediction Horizons:** $+6\text{ hours}$, $+12\text{ hours}$, and $+24\text{ hours}$ (indicated by `PRED_HORIZON_HOURS`).
- **Data File:** `processed/model1_intensity.parquet` (351,933 rows × 22 columns).

### 2.2 Input Feature Columns

| Column Name | Data Type | Physical Units | Valid Range / Categories | Missing Treatment | Description & Meteorological Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`CURRENT_USA_WIND`** | `float64` | Knots (kts) | $[10.0, 185.0]$ | None (0 missing) | Current maximum 1-minute sustained wind speed at observation time $t$. Baseline intensity predictor. |
| **`PAST_USA_WIND_6H`** | `float64` | Knots (kts) | $[10.0, 185.0]$ | Impute 0 / Masked | Original wind speed exactly 6 hours prior ($t-6h$) from the same agency stream. Missing only at storm genesis. |
| **`PAST_USA_WIND_12H`** | `float64` | Knots (kts) | $[10.0, 185.0]$ | Impute 0 / Masked | Original wind speed exactly 12 hours prior ($t-12h$) from the same agency stream. Missing during initial storm fixes. |
| **`HAS_PAST_USA_WIND_6H`** | `int64` | Binary Flag | `0` or `1` | None (0 missing) | Availability indicator: `1` if 6h past wind exists from same agency; `0` if storm was initiated $<6h$ ago. |
| **`HAS_PAST_USA_WIND_12H`** | `int64` | Binary Flag | `0` or `1` | None (0 missing) | Availability indicator: `1` if 12h past wind exists from same agency; `0` if storm was initiated $<12h$ ago. |
| **`WIND_CHANGE_PAST_6H`** | `float64` | Knots (kts) | $[-60.0, +75.0]$ | Impute 0 / Masked | Past 6-hour intensity trend ($V_t - V_{t-6h}$). Captures recent intensification/decay rate. |
| **`WIND_CHANGE_PAST_12H`** | `float64` | Knots (kts) | $[-85.0, +110.0]$ | Impute 0 / Masked | Past 12-hour intensity trend ($V_t - V_{t-12h}$). Captures rapid intensification momentum. |
| **`LAT`** | `float64` | Degrees North | $[-68.2, +69.0]$ | None (0 missing) | Current latitude coordinate. Proxy for sea surface temperature (SST) and Coriolis parameter. |
| **`LON_NORM`** | `float64` | Degrees East | $[-180.0, +180.0]$ | None (0 missing) | Current normalized longitude coordinate. Spatial climatological locator. |
| **`DIST2LAND`** | `float64` | Kilometers (km) | $[0.0, 3192.0]$ | None (0 missing) | Distance to nearest landmass. Critical for modeling friction and latent heat disruption upon landfall. |
| **`BASIN`** | `string` | Nominal Code | `WP`, `SI`, `EP`, `NA`, `SP`, `NI`, `SA` | None (0 missing) | Ocean basin identifier. Encodes regional environmental differences (e.g. shear profiles, SSTs). |
| **`NATURE`** | `string` | Nominal Code | `TS`, `ET`, `SS`, `NR`, `MX`, `DS` | None (0 missing) | Storm structural stage (e.g., Tropical, Extratropical transition, Subtropical). |
| **`MONTH`** | `int64` | Calendar Month | $[1, 12]$ | None (0 missing) | Annual climatological cycle indicator. Governs seasonal thermal peak. |
| **`HOUR`** | `int64` | UTC Hour | $[0, 23]$ | None (0 missing) | Diurnal cycle indicator (convective maximums frequently occur in early morning hours). |

### 2.3 Target Columns (Labels Only)

| Column Name | Data Type | Physical Units | Prediction Horizon | Missing Count | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`TARGET_USA_WIND`** | `float64` | Knots (kts) | $+6h, +12h, +24h$ | **0 (Complete)** | Future 1-minute maximum sustained wind speed at $t + h$. Verified original observation from same agency. |

### 2.4 Metadata Columns (Grouping, Slicing & Auditing Only)

| Column Name | Data Type | Usage Description |
| :--- | :---: | :--- |
| **`SID`** | `string` | Unique storm identifier (e.g. `1980186N14260`). Reserved strictly for grouped cross-validation. |
| **`SEASON`** | `int64` | Meteorological tropical cyclone season year. Governs temporal partition assignment. |
| **`ISO_TIME`** | `string` | UTC timestamp string (`YYYY-MM-DD HH:MM:SS`). Temporal auditing and verification key. |
| **`USA_AGENCY`** | `string` | Reporting agency code (`JTWC` or `hurdat_atl` / `hurdat_epa`). Source audit key. |
| **`PRED_HORIZON_HOURS`** | `int64` | Horizon indicator (`6`, `12`, or `24`). Used to slice multi-task models or train horizon-specific submodels. |
| **`SPLIT`** | `string` | Partition assignment: `train` (1980–2014), `val` (2015–2019), or `test` (2020–present). |
| **`LON`** | `float64` | Raw un-normalized longitude ($[0^\circ, 360^\circ]$). Retained for backward traceability. Do not feed to models. |

### 2.5 Prohibited Columns (Target Leakage & Snooping Hazards)
- `TARGET_USA_WIND` (Direct target leakage if included in inputs).
- `LANDFALL` (Lookahead variable; indicates future distance to landfall point).
- Raw `STORM_SPEED` and `STORM_DIR` (Computed using future fixes $t+3h$ or $t+6h$).
- Raw `LON` (Use `LON_NORM` to prevent artificial boundaries).
- `SID`, `ISO_TIME`, `SPLIT` (Identities and non-stationary administrative strings).

### 2.6 Downstream Preprocessing & Encoding Recommendations
1. **Categorical Features:**
   - `BASIN` (7 categories): One-hot encoding or learned low-dimensional entity embedding (e.g., $d=3$).
   - `NATURE` (6 categories): One-hot encoding.
2. **Cyclical Temporal Features:**
   - `MONTH`: Trigonometric encoding: $\sin(2\pi \cdot \text{MONTH} / 12)$, $\cos(2\pi \cdot \text{MONTH} / 12)$.
   - `HOUR`: Trigonometric encoding: $\sin(2\pi \cdot \text{HOUR} / 24)$, $\cos(2\pi \cdot \text{HOUR} / 24)$.
3. **Missing Lag Imputation:**
   - For decision trees (XGBoost, LightGBM, CatBoost): Retain `NaN` natively; models branch automatically on `HAS_PAST_USA_WIND_*`.
   - For linear models / neural networks: Impute missing lag values (`PAST_USA_WIND_*`, `WIND_CHANGE_PAST_*`) with `0.0` while supplying `HAS_PAST_USA_WIND_*` as active binary indicators.

---

## 3. Model 2 Manifest: Tropical Cyclone Movement Prediction

### 3.1 Task Overview
- **Objective:** Predict the future geographic position and orthogonal displacement components of a tropical cyclone from standard synoptic best-track fixes.
- **Task Type:** Multi-horizon Vector Regression (Coordinates & Cartesian Displacements).
- **Prediction Horizons:** $+6\text{ hours}$, $+12\text{ hours}$, and $+24\text{ hours}$ (indicated by `PRED_HORIZON_HOURS`).
- **Sampling Grid:** Standard 6-hourly UTC synoptic grid (`00:00, 06:00, 12:00, 18:00 UTC`).
- **Data File:** `processed/model2_movement.parquet` (516,086 rows × 24 columns).

### 3.2 Input Feature Columns

| Column Name | Data Type | Physical Units | Valid Range / Categories | Missing Treatment | Description & Meteorological Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`LAT`** | `float64` | Degrees North | $[-68.2, +69.0]$ | None (0 missing) | Current latitude coordinate. Dictates steering regime (e.g. tropical easterlies vs. mid-latitude westerlies). |
| **`LON_NORM`** | `float64` | Degrees East | $[-180.0, +180.0]$ | None (0 missing) | Current normalized longitude coordinate. Continuous geographic locator across the antimeridian. |
| **`PAST_EAST_DISP_6H_KM`** | `float64` | Kilometers (km) | $[-478.0, +473.0]$ | Impute 0 / Masked | Zonal motion over previous 6 hours ($t-6h$ to $t$). Captures current eastward speed component. |
| **`PAST_NORTH_DISP_6H_KM`** | `float64` | Kilometers (km) | $[-445.0, +556.0]$ | Impute 0 / Masked | Meridional motion over previous 6 hours ($t-6h$ to $t$). Captures current northward speed component. |
| **`PAST_EAST_DISP_12H_KM`** | `float64` | Kilometers (km) | $[-825.0, +913.0]$ | Impute 0 / Masked | Zonal motion over previous 12 hours ($t-12h$ to $t$). Captures persistent steering flow. |
| **`PAST_NORTH_DISP_12H_KM`** | `float64` | Kilometers (km) | $[-889.0, +1001.0]$ | Impute 0 / Masked | Meridional motion over previous 12 hours ($t-12h$ to $t$). Captures recurvature momentum. |
| **`DIST2LAND`** | `float64` | Kilometers (km) | $[0.0, 3192.0]$ | None (0 missing) | Distance to nearest landmass. Coastal terrain causes frictional track deflections and deceleration. |
| **`BASIN`** | `string` | Nominal Code | `WP`, `SI`, `EP`, `NA`, `SP`, `NI`, `SA` | None (0 missing) | Ocean basin identifier. Captures dominant synoptic steering features (e.g., Bermuda High, Subtropical Ridge). |
| **`NATURE`** | `string` | Nominal Code | `TS`, `ET`, `SS`, `NR`, `MX`, `DS` | None (0 missing) | Cyclone nature. Extratropical cyclones accelerate poleward rapidly compared to tropical systems. |
| **`MONTH`** | `int64` | Calendar Month | $[1, 12]$ | None (0 missing) | Annual climatological cycle indicator (subtropical ridge position migrates seasonally). |
| **`HOUR`** | `int64` | UTC Synoptic Hour | `0`, `6`, `12`, `18` | None (0 missing) | Synoptic observation cycle key. |

### 3.3 Target Columns (Labels Only)

| Column Name | Data Type | Physical Units | Prediction Horizon | Missing Count | Description & Recommended Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`TARGET_EAST_DISPLACEMENT_KM`** | `float64` | Kilometers (km) | $+6h, +12h, +24h$ | **0 (Complete)** | **Primary Training Target:** Eastward tangent-plane displacement ($dx$). Orthogonal Cartesian predictand. |
| **`TARGET_NORTH_DISPLACEMENT_KM`** | `float64` | Kilometers (km) | $+6h, +12h, +24h$ | **0 (Complete)** | **Primary Training Target:** Northward tangent-plane displacement ($dy$). Orthogonal Cartesian predictand. |
| **`TARGET_TOTAL_DISTANCE_KM`** | `float64` | Kilometers (km) | $+6h, +12h, +24h$ | **0 (Complete)** | Tangent-plane Euclidean displacement $\sqrt{dx^2 + dy^2}$. Orthogonal check target. |
| **`TARGET_HAVERSINE_DISTANCE_KM`** | `float64` | Kilometers (km) | $+6h, +12h, +24h$ | **0 (Complete)** | **Primary Evaluation Target:** Exact spherical great-circle geodesic distance. Standard benchmark metric. |
| **`TARGET_LAT`** | `float64` | Degrees North | $+6h, +12h, +24h$ | **0 (Complete)** | Future latitude at $t + h$. Coordinate target for direct latitude regression. |
| **`TARGET_LON_NORM`** | `float64` | Degrees East | $+6h, +12h, +24h$ | **0 (Complete)** | Future normalized longitude at $t + h$ ($[-180^\circ, +180^\circ]$). Wrapped coordinate target. |

### 3.4 Metadata Columns (Grouping, Slicing & Auditing Only)

| Column Name | Data Type | Usage Description |
| :--- | :---: | :--- |
| **`SID`** | `string` | Unique storm identifier. Reserved for grouped split verification and trajectory visualization. |
| **`SEASON`** | `int64` | Tropical cyclone season year. Governs temporal partition assignment. |
| **`ISO_TIME`** | `string` | UTC timestamp string (`YYYY-MM-DD HH:MM:SS`). Temporal primary key. |
| **`PRED_HORIZON_HOURS`** | `int64` | Horizon indicator (`6`, `12`, or `24`). Slicing key for horizon-specific movement models. |
| **`SPLIT`** | `string` | Partition assignment: `train` (1970–2014), `val` (2015–2019), or `test` (2020–present). |
| **`LON`** | `float64` | Raw un-normalized longitude. Retained for raw data validation only. |
| **`TARGET_LON`** | `float64` | Raw un-normalized target longitude. Retained for backward auditability. Do not feed to models. |

### 3.5 Prohibited Columns (Target Leakage Hazards)
- All `TARGET_*` fields (`TARGET_EAST_DISPLACEMENT_KM`, `TARGET_NORTH_DISPLACEMENT_KM`, `TARGET_TOTAL_DISTANCE_KM`, `TARGET_HAVERSINE_DISTANCE_KM`, `TARGET_LAT`, `TARGET_LON`, `TARGET_LON_NORM`).
- `LANDFALL`, centered `STORM_SPEED`, centered `STORM_DIR`.
- Raw `LON`.
- `SID`, `ISO_TIME`, `SPLIT`.

### 3.6 Downstream Modeling Architecture Recommendation
- **Recommended Formulation:** Train models to predict the orthogonal Cartesian displacements $(dx, dy) = (\text{TARGET\_EAST\_DISPLACEMENT\_KM}, \text{TARGET\_NORTH\_DISPLACEMENT\_KM})$.
- **Reconstruction:** Predicted future coordinates are recovered analytically:
  $$\widehat{\text{LAT}}_{t+h} = \text{LAT}_t + \frac{\widehat{dy}}{R \cdot (\pi / 180)}$$
  $$\widehat{\text{LON\_NORM}}_{t+h} = \text{normalize}\left(\text{LON\_NORM}_t + \frac{\widehat{dx}}{R \cdot \cos(\bar{\phi}) \cdot (\pi / 180)}\right)$$
- **Evaluation Metric:** Evaluate trajectory error using the spherical Great-Circle Distance (Haversine formula) between reconstructed $(\widehat{\text{LAT}}, \widehat{\text{LON}})$ and true $(\text{TARGET\_LAT}, \text{TARGET\_LON\_NORM})$.

---

## 4. Model 3 Manifest: Cyclone Development Pattern Classification

### 4.1 Task Overview
- **Objective:** Classify the 24-hour intensification or decay pattern of a tropical cyclone into three balanced physical categories based on future 24-hour wind change ($\Delta V_{24h} = V_{t+24h} - V_t$).
- **Task Type:** 3-Class Categorical Classification.
- **Physical Boundary Threshold:** $\pm 10\text{ knots}$ ($10\text{ kts} \approx 5.14\text{ m/s}$, the standard operational threshold for tropical cyclone development).
  - **`Strengthening` (Class ID `2`):** $\Delta V_{24h} \ge +10\text{ kts}$ (Rapid or steady intensification).
  - **`Stable` (Class ID `1`):** $-10\text{ kts} < \Delta V_{24h} < +10\text{ kts}$ (Steady intensity maintenance).
  - **`Weakening` (Class ID `0`):** $\Delta V_{24h} \le -10\text{ kts}$ (Steady decay or dissipation).
- **Prediction Horizon:** $+24\text{ hours}$.
- **Data File:** `processed/model3_development.parquet` (109,901 rows × 24 columns).

### 4.2 Input Feature Columns

| Column Name | Data Type | Physical Units | Valid Range / Categories | Missing Treatment | Description & Meteorological Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`CURRENT_USA_WIND`** | `float64` | Knots (kts) | $[10.0, 185.0]$ | None (0 missing) | Current maximum 1-minute sustained wind speed at observation time $t$. Proximity to Maximum Potential Intensity (MPI). |
| **`PAST_USA_WIND_6H`** | `float64` | Knots (kts) | $[10.0, 185.0]$ | Impute 0 / Masked | Original wind speed 6 hours prior from same agency stream. |
| **`PAST_USA_WIND_12H`** | `float64` | Knots (kts) | $[10.0, 185.0]$ | Impute 0 / Masked | Original wind speed 12 hours prior from same agency stream. |
| **`HAS_PAST_USA_WIND_6H`** | `int64` | Binary Flag | `0` or `1` | None (0 missing) | Indicator flag: `1` if 6h past wind is available from same agency; `0` if storm initiated $<6h$ ago. |
| **`HAS_PAST_USA_WIND_12H`** | `int64` | Binary Flag | `0` or `1` | None (0 missing) | Indicator flag: `1` if 12h past wind is available from same agency; `0` if storm initiated $<12h$ ago. |
| **`WIND_CHANGE_PAST_6H`** | `float64` | Knots (kts) | $[-60.0, +75.0]$ | Impute 0 / Masked | Past 6-hour wind change ($V_t - V_{t-6h}$). Recent intensification trend. |
| **`WIND_CHANGE_PAST_12H`** | `float64` | Knots (kts) | $[-85.0, +110.0]$ | Impute 0 / Masked | Past 12-hour wind change ($V_t - V_{t-12h}$). Momentum indicator. |
| **`LAT`** | `float64` | Degrees North | $[-68.2, +69.0]$ | None (0 missing) | Current latitude coordinate. Proxy for ocean heat content and vertical wind shear. |
| **`LON_NORM`** | `float64` | Degrees East | $[-180.0, +180.0]$ | None (0 missing) | Current normalized longitude coordinate. Spatial climatological locator. |
| **`DIST2LAND`** | `float64` | Kilometers (km) | $[0.0, 3192.0]$ | None (0 missing) | Distance to nearest land. Land interaction is the primary driver of rapid weakening. |
| **`BASIN`** | `string` | Nominal Code | `WP`, `SI`, `EP`, `NA`, `SP`, `NI`, `SA` | None (0 missing) | Ocean basin code. Captures regional climate regimes. |
| **`NATURE`** | `string` | Nominal Code | `TS`, `ET`, `SS`, `NR`, `MX`, `DS` | None (0 missing) | Storm structural stage (extratropical systems rarely strengthen via tropical mechanisms). |
| **`MONTH`** | `int64` | Calendar Month | $[1, 12]$ | None (0 missing) | Climatological season indicator. |
| **`HOUR`** | `int64` | UTC Hour | $[0, 23]$ | None (0 missing) | Diurnal cycle convective key. |

### 4.3 Target Columns (Labels Only)

| Column Name | Data Type | Physical Units / Format | Missing Count | Description & Recommended Role |
| :--- | :---: | :---: | :---: | :--- |
| **`CLASS_LABEL`** | `string` | Categorical Label | **0 (Complete)** | **Primary Classification Target:** `Strengthening`, `Stable`, or `Weakening`. Human-readable classification label. |
| **`CLASS_ID`** | `int64` | Ordinal Class ID (`0, 1, 2`) | **0 (Complete)** | **Primary Numerical Target:** Integer ID for multi-class cross-entropy loss: `2` = Strengthening, `1` = Stable, `0` = Weakening. |
| **`WIND_CHANGE_24H`** | `float64` | Knots (kts) | **0 (Complete)** | Continuous 24h wind difference ($V_{t+24h} - V_t$). Reserved for boundary analysis or joint multi-task learning. |
| **`TARGET_USA_WIND_24H`** | `float64` | Knots (kts) | **0 (Complete)** | Future wind speed at $t+24h$. Strict label verification key. |

### 4.4 Metadata Columns (Grouping, Slicing & Auditing Only)

| Column Name | Data Type | Usage Description |
| :--- | :---: | :--- |
| **`SID`** | `string` | Storm unique identifier. Grouping key for group-based k-fold cross validation. |
| **`SEASON`** | `int64` | Tropical cyclone season year. Governs temporal partition assignment. |
| **`ISO_TIME`** | `string` | UTC timestamp string (`YYYY-MM-DD HH:MM:SS`). |
| **`USA_AGENCY`** | `string` | Reporting agency code (`JTWC`, `hurdat_atl`, `hurdat_epa`). |
| **`SPLIT`** | `string` | Partition assignment: `train` (1980–2014), `val` (2015–2019), or `test` (2020–present). |
| **`LON`** | `float64` | Raw un-normalized longitude. Retained for data traceability. Do not feed to models. |

### 4.5 Prohibited Columns (Target Leakage Hazards)
- `CLASS_LABEL`, `CLASS_ID`, `WIND_CHANGE_24H`, `TARGET_USA_WIND_24H` (All target labels!).
- `LANDFALL`, centered `STORM_SPEED`, centered `STORM_DIR`.
- Raw `LON`.
- `SID`, `ISO_TIME`, `SPLIT`.

### 4.6 Evaluation Metrics Recommendation
Due to the well-balanced nature of the dataset (Stable: 41.71%, Strengthening: 33.09%, Weakening: 25.20%, Max/Min Class Ratio: 1.66):
- **Primary Metric:** Macro-averaged F1-Score ($\text{Macro-F1}$) to ensure equal performance across all three categories.
- **Secondary Metrics:** Multi-class Cohen's Kappa ($\kappa$), Categorical Cross-Entropy Loss, and Per-Class Confusion Matrix (Precision and Recall for Strengthening and Weakening).

---

## 5. Summary Matrix of Feature Roles Across All Three Models

| Column Name | Model 1: Intensity | Model 2: Movement | Model 3: Development | Role Category | Notes |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`SID`** | Metadata | Metadata | Metadata | Administrative Identity | Barred from ordinary features. |
| **`SEASON`** | Metadata | Metadata | Metadata | Administrative Partition | Barred from ordinary features. |
| **`BASIN`** | **Input Feature** | **Input Feature** | **Input Feature** | Environmental Categorical | One-hot or learned embedding. |
| **`ISO_TIME`** | Metadata | Metadata | Metadata | Administrative Timestamp | Barred from ordinary features. |
| **`USA_AGENCY`** | Metadata | — | Metadata | Operational Metadata | Agency stream audit key. |
| **`PRED_HORIZON_HOURS`** | Metadata / Slice | Metadata / Slice | — | Horizon Indicator | Multi-task horizon key ($+6h, +12h, +24h$). |
| **`CURRENT_USA_WIND`** | **Input Feature** | — | **Input Feature** | Core Physical Predictor | Current wind speed in kts. |
| **`TARGET_USA_WIND`** | **Target Output** | — | — | Target Label | Regressed predictand at $t+h$. |
| **`TARGET_USA_WIND_24H`** | — | — | **Target Output** | Target Label | Auxiliary label verification. |
| **`WIND_CHANGE_24H`** | — | — | **Target Output** | Target Label | Continuous wind change label. |
| **`CLASS_LABEL`** | — | — | **Target Output** | Target Label | Human-readable class name. |
| **`CLASS_ID`** | — | — | **Target Output** | Target Label | Integer class ID (`0, 1, 2`). |
| **`LAT`** | **Input Feature** | **Input Feature** | **Input Feature** | Core Geodetic Predictor | Current latitude in degrees. |
| **`LON`** | Prohibited | Prohibited | Prohibited | Legacy Coordinate | Raw longitude; use `LON_NORM`. |
| **`LON_NORM`** | **Input Feature** | **Input Feature** | **Input Feature** | Core Geodetic Predictor | Normalized $[-180^\circ, +180^\circ]$. |
| **`TARGET_LAT`** | — | **Target Output** | — | Target Label | Future latitude at $t+h$. |
| **`TARGET_LON`** | — | Prohibited | — | Legacy Coordinate | Raw target longitude; use `TARGET_LON_NORM`. |
| **`TARGET_LON_NORM`** | — | **Target Output** | — | Target Label | Future normalized longitude at $t+h$. |
| **`TARGET_EAST_DISPLACEMENT_KM`** | — | **Target Output** | — | Target Label | Primary zonal displacement target. |
| **`TARGET_NORTH_DISPLACEMENT_KM`** | — | **Target Output** | — | Target Label | Primary meridional displacement target. |
| **`TARGET_TOTAL_DISTANCE_KM`** | — | **Target Output** | — | Target Label | Tangent-plane Euclidean distance. |
| **`TARGET_HAVERSINE_DISTANCE_KM`** | — | **Target Output** | — | Evaluation Benchmark | Exact spherical great-circle distance. |
| **`DIST2LAND`** | **Input Feature** | **Input Feature** | **Input Feature** | Physical Boundary Predictor | Distance to nearest landmass in km. |
| **`NATURE`** | **Input Feature** | **Input Feature** | **Input Feature** | Physical Structural Predictor | Storm development phase code. |
| **`MONTH`** | **Input Feature** | **Input Feature** | **Input Feature** | Climatological Predictor | Seasonal cycle (trig encoded). |
| **`HOUR`** | **Input Feature** | **Input Feature** | **Input Feature** | Diurnal Predictor | Diurnal cycle (trig encoded). |
| **`PAST_USA_WIND_6H`** | **Input Feature** | — | **Input Feature** | Historical Intensity Predictor | Past intensity 6h prior. |
| **`PAST_USA_WIND_12H`** | **Input Feature** | — | **Input Feature** | Historical Intensity Predictor | Past intensity 12h prior. |
| **`HAS_PAST_USA_WIND_6H`** | **Input Feature** | — | **Input Feature** | Quality Indicator | Binary availability flag. |
| **`HAS_PAST_USA_WIND_12H`** | **Input Feature** | — | **Input Feature** | Quality Indicator | Binary availability flag. |
| **`WIND_CHANGE_PAST_6H`** | **Input Feature** | — | **Input Feature** | Historical Trend Predictor | Past 6h intensification trend. |
| **`WIND_CHANGE_PAST_12H`** | **Input Feature** | — | **Input Feature** | Historical Trend Predictor | Past 12h intensification trend. |
| **`PAST_EAST_DISP_6H_KM`** | — | **Input Feature** | — | Historical Motion Predictor | Past 6h zonal motion. |
| **`PAST_NORTH_DISP_6H_KM`** | — | **Input Feature** | — | Historical Motion Predictor | Past 6h meridional motion. |
| **`PAST_EAST_DISP_12H_KM`** | — | **Input Feature** | — | Historical Motion Predictor | Past 12h zonal motion. |
| **`PAST_NORTH_DISP_12H_KM`** | — | **Input Feature** | — | Historical Motion Predictor | Past 12h meridional motion. |
| **`SPLIT`** | Metadata | Metadata | Metadata | Administrative Partition | Slicing key: `train`, `val`, `test`. |

---
*Manifest document finalized for the Tropical Cyclone Pattern Prediction project.*
