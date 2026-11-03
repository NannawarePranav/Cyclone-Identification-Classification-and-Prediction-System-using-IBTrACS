# IBTrACS Model-Specific Dataset Preprocessing Report

> **Project:** Prediction of Different Tropical Cyclones Patterns  
> **Report Generation Date:** 2026-11-03 10:15:19 UTC  
> **Source Dataset:** `raw/ibtracs.ALL.list.v04r01.csv`  
> **Processed Deliverables:**
> 1. `processed/model1_intensity.parquet`
> 2. `processed/model2_movement.parquet`
> 3. `processed/model3_development.parquet`

---

## 1. Executive Summary & Quality Filtering Audit

This report documents the end-to-end data preparation pipeline that converts the raw NOAA IBTrACS v04r01 archive into three clean, physically consistent, leakage-free machine learning datasets.

### Common Quality Filtering Audit Trail

| Filtering Step | Raw Count | Filtered Count | Discarded Count | Operational Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Global Archive** | **730,416 rows** (13,631 cyclones) | — | — | Complete NOAA global database (1842–2026/2027). |
| **Header & Unit Row Handling** | Row 1 (Header), Row 2 (Units) | 730,416 data rows | 1 row | Row 2 skipped (`skiprows=[1]`) to preserve numeric column types. |
| **Duplicate `(SID, ISO_TIME)` Verification** | 730,416 rows | 730,416 rows | **0 rows** | Verified strict uniqueness of primary key composite. Zero unexpected duplicates. |
| **Track Type Filtering (`TRACK_TYPE == 'main'`)** | 730,416 rows | **710,515 rows** (13,170 cyclones) | **19,901 rows** | Excluded secondary bifurcating spurs (`spur-split`), merging tracks (`spur-merge`), and provisional tracks (`PROVISIONAL`, `US-PROVISIONAL`). |

### Track Types Discard Breakdown

| `TRACK_TYPE` Value | Observation Count | Retained Status |
| :--- | :--- | :--- |
| `main` | 710,515 | **RETAINED (Core Training Set)** |
| `US-PROVISIONAL` | 6,353 | EXCLUDED (Secondary / Provisional) |
| `spur-split` | 4,790 | EXCLUDED (Secondary / Provisional) |
| `PROVISIONAL` | 4,241 | EXCLUDED (Secondary / Provisional) |
| `spur-merge` | 2,871 | EXCLUDED (Secondary / Provisional) |
| `spur-other` | 1,621 | EXCLUDED (Secondary / Provisional) |
| `US-PROVISIONAL_spur` | 25 | EXCLUDED (Secondary / Provisional) |

---

## 2. Model 1 — Intensity Prediction Dataset

### Pipeline Specifications
- **Historical Cohort:** 1980–Present (`SEASON >= 1980`).
- **Target Attribute:** `USA_WIND` (1-minute maximum sustained wind speed in knots).
- **Prediction Horizons:** 6 hours, 12 hours, and 24 hours.
- **Observation Quality Requirement:** Current ($t$) and target ($t+h$) observations must have original wind records (`IFLAG[0] in ['O', 'V']`).
- **Source Consistency Requirement:** Current and target observations must come from the identical `USA_AGENCY` stream.
- **Predictor Feature Restrictions:** Strict exclusion of future-derived features (`LANDFALL`, unverified centered `STORM_SPEED`/`DIR`). All predictors are strictly concurrent or backward-looking ($t, t-6h, t-12h$).

### Usable Samples by Prediction Horizon

| Prediction Horizon | Total Usable Pairs | Unique Cyclones | Train Pairs (1980–2014) | Val Pairs (2015–2019) | Test Pairs (2020+) | Mean Target (kts) | Median Target (kts) | Target Std (kts) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6 Hours** | **123,234** | **4,364** | 94,783 | 14,276 | 14,175 | 48.16 | 40.0 | 27.66 |
| **+12 Hours** | **118,798** | **4,365** | 91,380 | 13,767 | 13,651 | 49.03 | 40.0 | 27.77 |
| **+24 Hours** | **109,901** | **4,336** | 84,531 | 12,758 | 12,612 | 50.7 | 45.0 | 28.1 |

**Total Model 1 Dataset Rows:** **351,933** instances across **4,366 unique cyclones**.

### Historical Lag Feature Availability (Model 1)

Historical wind observations at $t-6h$ and $t-12h$ are matched strictly from the same cyclone ID (`SID`), exact timestamp, and the identical reporting stream (`USA_AGENCY`), requiring original wind observations (`IFLAG[0] in ['O', 'V']`). Missing values occur legitimately only during cyclone genesis or agency handover, and are explicitly indicated by binary availability flags:

| Feature Indicator | Positive Count (Available) | Missing Count (Genesis / Transition) | Availability % |
| :--- | :---: | :---: | :---: |
| `HAS_PAST_USA_WIND_6H` | **338,434** | 13,499 | **96.16%** |
| `HAS_PAST_USA_WIND_12H` | **325,270** | 26,663 | **92.42%** |

### Basin Distribution (Model 1)

| Ocean Basin Code | Basin Name | Total Instances | % of Dataset |
| :--- | :--- | :--- | :--- |
| **`WP`** | Western North Pacific | 113,471 | 32.24% |
| **`EP`** | Eastern North Pacific | 66,855 | 19.00% |
| **`SI`** | South Indian | 66,808 | 18.98% |
| **`NA`** | North Atlantic | 57,128 | 16.23% |
| **`SP`** | South Pacific | 32,292 | 9.18% |
| **`NI`** | North Indian | 15,217 | 4.32% |
| **`SA`** | South Atlantic | 162 | 0.05% |

---

## 3. Model 2 — Movement Prediction Dataset

### Pipeline Specifications
- **Historical Cohort:** 1970–Present (`SEASON >= 1970`).
- **Track Filter:** `TRACK_TYPE == 'main'`.
- **Observation Selection:** Standard 6-hourly UTC synoptic grid (`00:00, 06:00, 12:00, 18:00 UTC`). Sub-synoptic 3-hourly points are excluded to prevent training models on NOAA's linear interpolation routine.
- **Coordinate Normalization:** All longitudes normalized to [-180 deg, +180 deg]. Antimeridian crossings wrapped using continuous geodesic angular differences.
- **Target Targets:** Target coordinates (`TARGET_LAT`, `TARGET_LON`, `TARGET_LON_NORM`), Cartesian tangent-plane displacements (`TARGET_EAST_DISPLACEMENT_KM`, `TARGET_NORTH_DISPLACEMENT_KM`, `TARGET_TOTAL_DISTANCE_KM`), and spherical geodesic distance (`TARGET_HAVERSINE_DISTANCE_KM`).

### Consolidated IBTrACS Best-Track Position Characteristics

> [!NOTE]
> **Consolidated Best-Track Positions:** `LAT` and `LON` are consolidated IBTrACS best-track positions compiled across international regional specialized meteorological centres (RSMCs) and warning centers. While selecting the 6-hourly synoptic grid (`00, 06, 12, 18 UTC`) eliminates exposure to NOAA's intermediate 3-hourly linear interpolations, it does not imply that every consolidated position is an independently observed raw fix. Best tracks are post-season consensus trajectories produced by human reanalysis teams who smooth fixes using cubic splines and physical constraints. Downstream models should recognize that targets represent smoothed best-track consensus paths rather than raw sensor fixes.

### Usable Samples by Prediction Horizon

| Prediction Horizon | Total Usable Pairs | Unique Cyclones | Train Pairs (1970–2014) | Val Pairs (2015–2019) | Test Pairs (2020+) | Mean Tangent Dist | Median Dist | Max Dist |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6 Hours** | **180,022** | **6,006** | 147,108 | 16,340 | 16,574 | 110.98 km | 97.98 km | 1037.29 km |
| **+12 Hours** | **174,016** | **5,996** | 142,225 | 15,798 | 15,993 | 218.39 km | 193.24 km | 1713.39 km |
| **+24 Hours** | **162,048** | **5,915** | 132,494 | 14,719 | 14,835 | 424.95 km | 380.4 km | 3138.17 km |

### Local Tangent-Plane Approximation vs. Haversine Great-Circle Geodesic Distance

Eastward and northward displacements ($dx, dy$) and planar distance ($d_{plane} = \sqrt{dx^2 + dy^2}$) are local tangent-plane equirectangular approximations centered on the trajectory midpoint latitude $\bar{\phi}$. The approximation was rigorously validated against the exact spherical Haversine great-circle distance $d_{haversine}$ across all prediction horizons:

| Prediction Horizon | Tangent Mean Dist | Haversine Mean Dist | Mean Error | Median Error | 95th % Error | Max Error | Mean Rel Error | Max Rel Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6 Hours** | 110.98 km | 110.98 km | **0.0024 km** | 0.0 km | 0.01 km | **1.47 km** | **0.0008%** | **0.1869%** |
| **+12 Hours** | 218.39 km | 218.38 km | **0.0172 km** | 0.0 km | 0.05 km | **10.3 km** | **0.0030%** | **0.6961%** |
| **+24 Hours** | 424.95 km | 424.84 km | **0.1148 km** | 0.01 km | 0.34 km | **36.62 km** | **0.0109%** | **1.5491%** |

> [!TIP]
> **Evaluation Standard:** While Cartesian components (`TARGET_EAST_DISPLACEMENT_KM`, `TARGET_NORTH_DISPLACEMENT_KM`) provide orthogonal training targets for trajectory regression, the provided `TARGET_HAVERSINE_DISTANCE_KM` field represents the exact spherical great-circle distance and should be used as the standard benchmark metric during future model evaluation.

**Total Model 2 Dataset Rows:** **516,086** instances across **6,006 unique cyclones**.

### Basin Distribution (Model 2)

| Ocean Basin Code | Basin Name | Total Instances | % of Dataset |
| :--- | :--- | :--- | :--- |
| **`WP`** | Western North Pacific | 178,294 | 34.55% |
| **`SI`** | South Indian | 107,406 | 20.81% |
| **`EP`** | Eastern North Pacific | 78,592 | 15.23% |
| **`NA`** | North Atlantic | 69,863 | 13.54% |
| **`SP`** | South Pacific | 55,266 | 10.71% |
| **`NI`** | North Indian | 26,503 | 5.14% |
| **`SA`** | South Atlantic | 162 | 0.03% |

---

## 4. Model 3 — Development Pattern Classification Dataset

### Pipeline Specifications
- **Historical Cohort:** 1980–Present (`SEASON >= 1980`).
- **Target Formulation:** 24-hour wind change ΔV_24h = V(t+24h) - V(t) classified into 3 balanced categories:
  - **`Strengthening`:** ΔV_24h >= +10 kts (Class ID `2`)
  - **`Stable`:** -10 kts < ΔV_24h < +10 kts (Class ID `1`)
  - **`Weakening`:** ΔV_24h <= -10 kts (Class ID `0`)
- **Quality Control:** Both current and 24h future winds must be original (`IFLAG[0] in ['O', 'V']`) and originate from the identical `USA_AGENCY` stream.
- **Label Safety:** Future winds (`TARGET_USA_WIND_24H`), ΔV_24h, and class labels are strictly designated as labels, never as inputs.

### Historical Lag Feature Availability (Model 3)

Historical wind observations at $t-6h$ and $t-12h$ are matched strictly from the same cyclone ID (`SID`), exact timestamp, and the identical reporting stream (`USA_AGENCY`), requiring original wind observations (`IFLAG[0] in ['O', 'V']`):

| Feature Indicator | Positive Count (Available) | Missing Count (Genesis / Transition) | Availability % |
| :--- | :---: | :---: | :---: |
| `HAS_PAST_USA_WIND_6H` | **105,438** | 4,463 | **95.94%** |
| `HAS_PAST_USA_WIND_12H` | **101,123** | 8,778 | **92.01%** |

### Overall Class Distribution (Model 3)

| Class Label | Class ID | Sample Count | % of Dataset | Physical Meteorological Definition |
| :--- | :---: | :--- | :--- | :--- |
| **`Stable`** | 1 | **45,842** | **41.71%** | Wind change ΔV_24h satisfies threshold boundary. |
| **`Strengthening`** | 2 | **36,367** | **33.09%** | Wind change ΔV_24h satisfies threshold boundary. |
| **`Weakening`** | 0 | **27,692** | **25.20%** | Wind change ΔV_24h satisfies threshold boundary. |

**Total Model 3 Dataset Rows:** **109,901** instances across **4,336 unique cyclones**.

### Class Distribution by Partition Split (Train / Val / Test)

| Partition Split | Total Samples | Strengthening Count (%) | Stable Count (%) | Weakening Count (%) | Class Ratio (Max/Min) |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`TRAIN`** | **84,531** | 27,973 (33.09%) | 35,671 (42.20%) | 20,887 (24.71%) | **1.71** |
| **`VAL`** | **12,758** | 4,283 (33.57%) | 4,962 (38.89%) | 3,513 (27.54%) | **1.41** |
| **`TEST`** | **12,612** | 4,111 (32.60%) | 5,209 (41.30%) | 3,292 (26.10%) | **1.58** |

### Basin Distribution (Model 3)

| Ocean Basin Code | Basin Name | Total Instances | % of Dataset |
| :--- | :--- | :--- | :--- |
| **`WP`** | Western North Pacific | 35,618 | 32.41% |
| **`SI`** | South Indian | 21,022 | 19.13% |
| **`EP`** | Eastern North Pacific | 20,656 | 18.80% |
| **`NA`** | North Atlantic | 17,815 | 16.21% |
| **`SP`** | South Pacific | 10,082 | 9.17% |
| **`NI`** | North Indian | 4,659 | 4.24% |
| **`SA`** | South Atlantic | 49 | 0.04% |

---

## 5. Season-Based Temporal Dataset Splitting & Data Leakage Verification

### Why Season-Based Splitting Differs from Calendar-Year Splitting
In tropical meteorology, cyclone lifecycles are governed by meteorological seasons rather than calendar years:
- **Northern Hemisphere Basins (NA, EP, WP, NI):** The tropical cyclone season roughly aligns with the calendar year (January 1 to December 31).
- **Southern Hemisphere Basins (SI, SP, SA):** The tropical cyclone season spans across calendar years, beginning July 1 of year $Y-1$ and concluding June 30 of year $Y$. For example, `SEASON == 2015` in the South Indian Ocean begins in July/October 2014 and concludes in May/June 2015.

Partitioning datasets strictly by `SEASON` at the cyclone level ensures that **no cyclone lifecycle is ever bisected across splits**, completely eliminating temporal autocorrelation and lookahead leakage.

### Comprehensive Partition Breakdown Across All Models

| Dataset Deliverable | Partition Split | Total Samples | Unique Cyclones (SIDs) | Min ISO_TIME | Max ISO_TIME | Inter-Split SID Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Intensity** | `TRAIN` | **270,694** | 3,344 | `1979-11-25 00:00:00` | `2015-01-01 00:00:00` | **0 (VERIFIED)** |
| **Model 1: Intensity** | `VAL` | **40,801** | 503 | `2014-11-15 00:00:00` | `2019-12-29 00:00:00` | **0 (VERIFIED)** |
| **Model 1: Intensity** | `TEST` | **40,438** | 519 | `2019-11-22 00:00:00` | `2025-11-01 00:00:00` | **0 (VERIFIED)** |
| **Model 2: Movement** | `TRAIN` | **421,827** | 4,883 | `1969-08-18 18:00:00` | `2015-01-01 06:00:00` | **0 (VERIFIED)** |
| **Model 2: Movement** | `VAL` | **46,857** | 542 | `2014-11-15 00:00:00` | `2019-12-29 00:00:00` | **0 (VERIFIED)** |
| **Model 2: Movement** | `TEST` | **47,402** | 581 | `2019-11-22 00:00:00` | `2026-04-28 12:00:00` | **0 (VERIFIED)** |
| **Model 3: Development** | `TRAIN` | **84,531** | 3,319 | `1979-11-25 00:00:00` | `2014-12-31 06:00:00` | **0 (VERIFIED)** |
| **Model 3: Development** | `VAL` | **12,758** | 500 | `2014-11-15 00:00:00` | `2019-12-28 06:00:00` | **0 (VERIFIED)** |
| **Model 3: Development** | `TEST` | **12,612** | 517 | `2019-11-22 00:00:00` | `2025-10-31 06:00:00` | **0 (VERIFIED)** |

> [!IMPORTANT]
> **Zero Cyclone Overlap Confirmation:** Automated assertion checks verified that:
> - `len(set(train_sids) & set(val_sids)) == 0`
> - `len(set(train_sids) & set(test_sids)) == 0`
> - `len(set(val_sids) & set(test_sids)) == 0`
> All observations belonging to a single cyclone ID (`SID`) are strictly contained in exactly one partition.

---

## 6. Missing-Value Statistics in Processed Files

All target fields, primary identifiers, and core features have **0 missing values** across all three Parquet files.

| Attribute Field | Model 1 Missing | Model 2 Missing | Model 3 Missing | Operational Handling / Imputation Note |
| :--- | :---: | :---: | :---: | :--- |
| `SID`, `SEASON`, `BASIN`, `ISO_TIME` | **0** | **0** | **0** | 100% complete core identifiers. |
| Target Fields (`TARGET_USA_WIND`, `TARGET_LAT`, displacements, `TARGET_HAVERSINE_DISTANCE_KM`, `CLASS_LABEL`, `CLASS_ID`) | **0** | **0** | **0** | 100% complete. Every row is a verified training/evaluation sample. |
| Current Position (`LAT`, `LON`, `LON_NORM`) | **0** | **0** | **0** | 100% complete core tracking coordinates. |
| Current Intensity (`CURRENT_USA_WIND`) | **0** | — | **0** | 100% complete original wind speed. |
| Lag Indicator Flags (`HAS_PAST_USA_WIND_*`) | **0** | — | **0** | 100% complete binary presence flags (1=available, 0=absent). |
| `DIST2LAND` | **0** | **0** | **0** | 100% complete distance to nearest land. |
| Past Lag Features ($t-6h, t-12h$) | ~7–12% | ~7–12% | ~7–12% | Missing legitimately only at cyclone genesis or agency transition fixes. Binary indicator flags allow neural nets / linear models to zero-fill without losing information. |

---

## 7. Remaining Limitations & Downstream Recommendations

1. **Consolidated Best-Track Positions:** `LAT` and `LON` represent post-analysis consensus best-track coordinates. The 6-hourly synoptic sampling eliminates exposure to NOAA's intermediate 3-hourly linear interpolations, but users must recognize that historical best tracks represent human consensus trajectories smoothed during post-season reanalysis.
2. **Geodesic Evaluation Benchmark:** Downstream movement models should train on Cartesian orthogonal displacements (`TARGET_EAST_DISPLACEMENT_KM`, `TARGET_NORTH_DISPLACEMENT_KM`) but evaluate overall displacement accuracy using spherical Haversine geodesic distance (`TARGET_HAVERSINE_DISTANCE_KM`).
3. **Basin Volume Asymmetry:** The Western North Pacific, Eastern Pacific, and North Atlantic comprise ~68% of the data. Downstream modeling should incorporate basin embeddings and report basin-stratified metrics to monitor performance in smaller basins (e.g. North Indian Ocean).
4. **Genesis Fix Lag Sparseness:** For a storm's initial observation ($t=0$), historical lag features ($t-6h, t-12h$) are legitimately absent. Models should leverage the provided `HAS_PAST_*` binary indicators alongside zero-filling or masked recurrent representations.

---
*Report generated automatically by `scripts/prepare_datasets.py`.*
