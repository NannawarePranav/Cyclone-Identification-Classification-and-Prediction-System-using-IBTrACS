# IBTrACS Modelling Cohort & Feature Feasibility Audit

> **Project:** Prediction of Different Tropical Cyclones Patterns  
> **Dataset:** NOAA International Best Track Archive for Climate Stewardship (IBTrACS) v04r01 Global  
> **Source CSV:** `raw/ibtracs.ALL.list.v04r01.csv`  
> **Audit Focus:** Modelling Cohorts, Feature Feasibility, and Target Formulation for Three Machine Learning Models  
> **Audit Status:** Analysis-Only Complete (No raw data modified; No processed datasets generated)

---

## Executive Summary

This report establishes the empirical foundation for selecting historical training cohorts, feature sets, and target definitions for our three proposed tropical cyclone machine learning models. 

Our investigation yields four major architectural conclusions:
1. **Historical Cohort Selection:** While track coordinates are 100% complete from 1842 to present, reliable wind speed measurements are heavily concentrated in the post-satellite era. **1980–Present** is the optimal training cohort for intensity and trend classification (yielding **262,945** valid wind fixes and **224,918** complete 24-hour training pairs), while track movement prediction can leverage data from **1970–Present** (**359,520** usable 6-hour trajectory pairs).
2. **Intensity Target Standardization:** Naively combining wind speeds from different reporting agencies is **scientifically invalid**. `USA_WIND` follows the US **1-minute** sustained wind standard (JTWC/NHC), whereas `WMO_WIND` and regional RSMCs adhere to **10-minute** (Tokyo, Réunion, BoM), **3-minute** (New Delhi), or **2-minute** (China CMA) averaging intervals. Physical gust-factor differences cause 1-minute winds to be systematically $\approx 12-14\%$ higher, producing artificial intensity jumps whenever a storm crosses agency boundaries. We recommend standardizing on **`USA_WIND`** as the primary target across all basins.
3. **Movement Prediction & Interpolation Artefacts:** IBTrACS v04 linearly interpolates 6-hourly best-track fixes into 3-hourly positions. Evaluating movement models on 3-hour lead times risks evaluating the dataset's linear interpolation formula rather than physical cyclone steering. Models must be formulated and benchmarked on **6h, 12h, and 24h** synoptic horizons. Furthermore, **10,658 fixes** in the South Pacific and Eastern Pacific record longitudes exceeding $180^\circ$ (up to $266.9^\circ$), necessitating strict Antimeridian coordinate normalization.
4. **Development Trend Class Formulation:** A strict zero-threshold ($\Delta V > 0, =0, <0$) severely under-represents the "Stable" class (only 14.4% of examples) and misclassifies sensor noise ($\pm 5\text{ kts}$) as true physical development. A **symmetric $\pm 10\text{ kts}$ 24-hour wind change threshold** produces an optimal meteorological and statistical balance: **31.7% Strengthening**, **44.5% Stable**, and **23.8% Weakening**, with an ideal max-to-min class ratio of **1.87**.

---

## 1. Historical Cohort Comparative Analysis

To prevent historical observation gaps and pre-satellite observational uncertainties from corrupting model training, we evaluated three candidate historical cohorts:
- **1970–Present:** Dawn of continuous geostationary and polar-orbiting meteorological satellites.
- **1980–Present:** Worldwide operationalization of the standardized Dvorak satellite intensity estimation technique and modern ATCF archiving.
- **2000–Present:** Modern multi-sensor satellite era (QuikSCAT/ASCAT scatterometry, passive microwave sounders, automated wind radii archiving).

### Comprehensive Historical Cohort Comparison Matrix

| Metric / Attribute | 1970–Present (Satellite Era) | 1980–Present (Dvorak Era) | 2000–Present (Multi-Sensor Era) |
| :--- | :--- | :--- | :--- |
| **Calendar Span** | 1970 – 2026/2027 (57 seasons) | 1980 – 2026/2027 (47 seasons) | 2000 – 2026/2027 (27 seasons) |
| **Total Track Observations** | **386,212** (52.88% of global) | **312,668** (42.81% of global) | **169,796** (23.25% of global) |
| **Unique Cyclones (`SID`)** | **6,280** (46.07% of global) | **5,019** (36.82% of global) | **2,795** (20.50% of global) |
| **Position Completeness (`LAT`/`LON`)** | **100.00%** (0 missing) | **100.00%** (0 missing) | **100.00%** (0 missing) |
| **`USA_WIND` Availability** | 303,773 fixes (**78.65%**) | 262,945 fixes (**84.10%**) | 144,467 fixes (**85.08%**) |
| **`WMO_WIND` Availability** | 136,554 fixes (**35.36%**) | 121,090 fixes (**38.73%**) | 71,660 fixes (**42.20%**) |
| **Any Wind Availability (`USA` $\cup$ `WMO`)** | 320,424 fixes (**82.97%**) | 275,657 fixes (**88.16%**) | 152,216 fixes (**89.65%**) |
| **Main-Track Observations (`main`)** | 374,705 fixes (97.02%) | 302,968 fixes (96.90%) | 160,379 fixes (94.45%) |
| **Main-Track Unique Cyclones** | 6,014 cyclones (95.76%) | 4,801 cyclones (95.66%) | 2,642 cyclones (94.53%) |
| **Usable Movement Pairs (6h horizon)** | **371,604** (359,520 main) | **301,299** (290,015 main) | **163,448** (153,141 main) |
| **Usable Movement Pairs (12h horizon)** | **359,123** (347,596 main) | **291,245** (280,424 main) | **157,772** (147,828 main) |
| **Usable Movement Pairs (24h horizon)** | **333,781** (323,346 main) | **271,088** (261,187 main) | **146,537** (137,317 main) |
| **Usable Intensity Pairs (`USA_WIND`, 6h)** | **291,426** pairs | **252,561** pairs | **138,528** pairs |
| **Usable Intensity Pairs (`USA_WIND`, 12h)** | **280,716** pairs | **243,353** pairs | **133,229** pairs |
| **Usable Intensity Pairs (`USA_WIND`, 24h)** | **259,022** pairs | **224,918** pairs | **122,756** pairs |
| **Structural Radii Availability (`USA_R34`)** | 76,701 fixes (19.86%) | 76,701 fixes (24.53%) | 73,812 fixes (**43.47%**) |

### Basin Breakdown Across Cohorts

| Basin Code | Basin Name | 1970–Present (Obs / SIDs) | 1980–Present (Obs / SIDs) | 2000–Present (Obs / SIDs) |
| :--- | :--- | :--- | :--- | :--- |
| **`WP`** | Western North Pacific | 129,852 / 1,933 | 104,618 / 1,540 | 51,551 / 825 |
| **`SI`** | South Indian Ocean | 83,778 / 1,134 | 67,697 / 922 | 37,299 / 495 |
| **`EP`** | Eastern North Pacific | 58,792 / 1,374 | 50,835 / 1,161 | 29,468 / 646 |
| **`NA`** | North Atlantic | 50,943 / 963 | 41,685 / 753 | 27,155 / 474 |
| **`SP`** | South Pacific Ocean | 42,241 / 706 | 32,079 / 550 | 15,350 / 284 |
| **`NI`** | North Indian Ocean | 20,487 / 611 | 15,635 / 473 | 8,854 / 266 |
| **`SA`** | South Atlantic Ocean | 119 / 3 | 119 / 3 | 119 / 3 |

### Recommended Historical Period

> [!IMPORTANT]
> **Cohort Recommendation by Model:**
> - **Model 1 (Intensity Prediction) & Model 3 (Development Trend):** **1980–Present** is recommended. Prior to 1980, wind estimation methods in the Southern Hemisphere and North Indian basins were fragmented and inconsistent. The 1980 boundary provides **262,945** high-quality wind observations and **224,918** complete 24-hour training pairs with an **84.1%** completeness rate.
> - **Model 2 (Movement Prediction):** **1970–Present** is recommended. Position data is 100% complete, and satellite imagery tracking since 1970 provides accurate center fixes, yielding **359,520** verified 6-hour main-track trajectory pairs.
> - **Advanced Structural Modelling (Sub-study):** If quadrant wind radii (`USA_R34`, `USA_R50`, `USA_R64`, `USA_RMW`) are required as predictive inputs, the cohort must be restricted to **2000–Present**, as radii records prior to 2000 are virtually nonexistent.

---

## 2. Model 1 Feasibility Audit: Cyclone Intensity Prediction

**Proposed Target:** Future Maximum Sustained Wind Speed ($V_{t+6h}, V_{t+12h}, V_{t+24h}$).

### 1. Availability of `USA_WIND` vs `WMO_WIND` by Basin

Our audit reveals vast disparities in reporting rates between US agencies (JTWC/NHC) and the official WMO consensus field:

| Ocean Basin | Total Basin Fixes | `USA_WIND` Count | `USA_WIND` % | `WMO_WIND` Count | `WMO_WIND` % | Overlapping Fixes (`Both`) | Any Wind (`Union`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Eastern Pacific (`EP`)** | 67,136 | 66,185 | **98.58%** | 31,687 | 47.20% | 31,678 (47.18%) | 66,194 (98.60%) |
| **North Atlantic (`NA`)** | 127,584 | 108,745 | **85.23%** | 55,805 | 43.74% | 55,797 (43.73%) | 108,753 (85.24%) |
| **Western Pacific (`WP`)** | 243,859 | 131,641 | **53.98%** | 28,766 | 11.80% | 28,234 (11.58%) | 132,173 (54.20%) |
| **South Pacific (`SP`)** | 68,724 | 30,202 | **43.95%** | 14,334 | 20.86% | 9,840 (14.32%) | 34,696 (50.49%) |
| **South Indian (`SI`)** | 165,114 | 56,413 | **34.17%** | 32,649 | 19.77% | 23,130 (14.01%) | 65,932 (39.93%) |
| **North Indian (`NI`)** | 57,880 | 12,594 | **21.76%** | 7,041 | 12.16% | 4,920 (8.50%) | 14,715 (25.42%) |
| **South Atlantic (`SA`)** | 119 | 119 | **100.00%** | 61 | 51.26% | 61 (51.26%) | 119 (100.00%) |

**Key Takeaways:**
1. In the Western Pacific (`WP`), `USA_WIND` provides **131,641** fixes, while `WMO_WIND` provides only **28,766** fixes. Furthermore, 98.2% of all `WMO_WIND` entries in the Western Pacific directly overlap with `USA_WIND`.
2. In the North Atlantic (`NA`) and Eastern Pacific (`EP`), `USA_WIND` covers nearly the entire active tracking history (85.2% and 98.6% across the entire 184-year catalog; over 99% post-1980).
3. `WMO_WIND` is missing in more than **76%** of all global records and more than **61%** of records post-1980.

### 2. Availability of Regional Agency Wind Fields

IBTrACS archives best tracks from individual Regional Specialized Meteorological Centers (RSMCs) and Tropical Cyclone Warning Centers (TCWCs):

| Regional Agency Column | Responsible Agency | Basin Coverage | Fix Count | Global % | Active Years |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TOKYO_WIND` | JMA (Japan Meteorological Agency) | Western North Pacific (`WP`) | 51,285 | 7.02% | 1977 – 2026 |
| `CMA_WIND` | CMA (China Meteorological Administration) | Western North Pacific (`WP`) | 133,126 | 18.23% | 1949 – 2025 |
| `HKO_WIND` | Hong Kong Observatory | Western North Pacific (`WP`) | 86,262 | 11.81% | 1961 – 2025 |
| `KMA_WIND` | Korea Meteorological Administration | Western North Pacific (`WP`) | 13,072 | 1.79% | 2015 – 2025 |
| `NEWDELHI_WIND` | IMD (India Meteorological Department) | North Indian (`NI`) | 9,467 | 1.30% | 1982 – 2025 |
| `REUNION_WIND` | Météo-France La Réunion | South Indian (`SI`) | 34,288 | 4.69% | 1977 – 2026 |
| `BOM_WIND` | Australian Bureau of Meteorology | South Indian (`SI`), South Pacific (`SP`) | 42,162 | 5.77% | 1967 – 2026 |
| `NADI_WIND` | Fiji Meteorological Service | South Pacific (`SP`) | 7,829 | 1.07% | 1994 – 2026 |
| `WELLINGTON_WIND` | MetService New Zealand | South Pacific (`SP`) | 23,470 | 3.21% | 1968 – 2026 |

Each regional agency field is populated exclusively within that agency's Area of Responsibility (AOR). None of these fields can be used as a standalone global target.

### 3. `WMO_AGENCY` Distribution & Intra-Storm Consistency

When `WMO_WIND` is populated, the `WMO_AGENCY` attribute indicates which center provided the official estimate:

| `WMO_AGENCY` Code | Fix Count | % of Populated | Primary Ocean Basin |
| :--- | :--- | :--- | :--- |
| *(Missing / Not Provided)* | 498,090 | 68.19% (of all fixes) | Global historical |
| `tokyo` | 70,782 | 30.47% | Western North Pacific |
| `hurdat_atl` | 55,330 | 23.82% | North Atlantic |
| `hurdat_epa` | 31,220 | 13.44% | Eastern North Pacific |
| `bom` | 30,074 | 12.94% | Southern Hemisphere / Australia |
| `reunion` | 27,978 | 12.04% | South-West Indian Ocean |
| `newdelhi` | 7,311 | 3.15% | Bay of Bengal & Arabian Sea |
| `wellington` | 4,966 | 2.14% | South-West Pacific |
| `nadi` | 3,605 | 1.55% | South Pacific |
| `atcf` / `cphc` | 1,060 | 0.46% | Central & North Pacific |

#### Intra-Storm Agency Switching Analysis
- **`WMO_AGENCY` Consistency:** Among storms with at least one WMO agency fix, **377 cyclones switch WMO agencies mid-lifecycle** (up to 3 distinct agencies for a single storm). For example, storms moving across $90^\circ\text{E}$ in the Indian Ocean switch from `bom` (Perth) to `reunion` (Météo-France), or across $160^\circ\text{E}$ from `bom` (Brisbane) to `nadi` (Fiji).
- **`USA_AGENCY` Consistency:** In contrast, **8,477 cyclones maintain 100% agency consistency** under `USA_AGENCY`, with only 176 storms transitioning between NHC and CPHC ($140^\circ\text{W}$) or CPHC and JTWC ($180^\circ$).

### 4. Continuous Wind Observation Run Lengths

Machine learning sequence models (LSTMs, GRUs, 1D-CNNs) require uninterrupted historical sequences. We audited the maximum consecutive run of non-null wind observations for every cyclone in the dataset:

| Minimum Consecutive Wind Run | `USA_WIND` Cyclone Count | `WMO_WIND` Cyclone Count |
| :--- | :--- | :--- |
| **$\ge 4$ consecutive fixes (12 hours)** | **7,293 cyclones** | 695 cyclones |
| **$\ge 8$ consecutive fixes (24 hours)** | **7,232 cyclones** | 483 cyclones |
| **$\ge 12$ consecutive fixes (36 hours)** | **7,073 cyclones** | 384 cyclones |
| **$\ge 16$ consecutive fixes (48 hours)** | **6,871 cyclones** | 314 cyclones |
| **$\ge 24$ consecutive fixes (72 hours)** | **6,240 cyclones** | 184 cyclones |
| **Median Maximum Run per Cyclone** | **17.0 fixes (~51 hours)** | **0.0 fixes** |
| **Mean Maximum Run per Cyclone** | **29.73 fixes (~89 hours)** | **1.46 fixes** |

> [!CAUTION]
> **Severe Sparseness in `WMO_WIND` Sequences:** In IBTrACS, `WMO_WIND` is populated almost exclusively at 6-hourly synoptic fixes and is left blank at 3-hourly intermediate interpolated points. As a result, only **483 cyclones** in history have 8 consecutive 3-hourly `WMO_WIND` fixes. In sharp contrast, **7,232 cyclones** have $\ge 8$ consecutive `USA_WIND` fixes, with a median continuous run of **17 fixes**.

### 5. Physical Risks of Combining Different Agency Wind Measurements

A naive data engineering approach might attempt to "fill" missing `WMO_WIND` values with `USA_WIND`, or vice versa. **This is fundamentally invalid due to differing physical measurement standards:**

```text
┌────────────────────────────────────────────────────────────────────────┐
│             GLOBAL TROPICAL CYCLONE WIND AVERAGING STANDARDS           │
├──────────────────┬─────────────────┬───────────────────────────────────┤
│ Standard         │ Averaging Time  │ Operational Agencies              │
├──────────────────┼─────────────────┼───────────────────────────────────┤
│ US Standard      │ 1-minute sustained│ NHC (Miami), JTWC (Hawaii), CPHC │
│ WMO Standard     │ 10-minute sustained│ RSMC Tokyo, La Réunion, BoM, Nadi│
│ IMD Standard     │ 3-minute sustained│ IMD New Delhi (North Indian Ocean)│
│ CMA Standard     │ 2-minute sustained│ CMA Beijing (China)               │
└──────────────────┴─────────────────┴───────────────────────────────────┘
```

#### Physical & Mathematical Divergence
By boundary-layer atmospheric turbulence theory (e.g. Harper et al. 2010, WMO-TD No. 1555), maximum sustained wind speed is inversely related to the averaging period. A 1-minute average captures smaller, more intense convective gust peaks than a 10-minute average:

$$V_{1\text{-min}} \approx 1.12\text{ to }1.15 \times V_{10\text{-min}}$$

Empirically, when comparing co-located fixes where both `USA_WIND` and `WMO_WIND` are reported:
- The mean difference $(V_{\text{USA}} - V_{\text{WMO}})$ is **$+3.23\text{ kts}$** across 108,378 overlapping fixes (1980+).
- `USA_WIND` is strictly higher in **33.3%** of fixes, equal in **52.5%** of fixes (frequently at lower tropical storm thresholds where rounding dominates), and lower in only **14.2%**.
- Standard deviation of the difference is **$9.71\text{ kts}$**, with differences reaching up to **$25-35\text{ kts}$** in Category 4–5 systems.

#### Machine Learning Failure Mode
If a model trains on a target that silently toggles between 1-minute and 10-minute agencies:
1. **False Intensification / Decay:** When a storm moves from an Australian BoM region (10-min) into a JTWC tracking region (1-min), the reported wind speed will abruptly jump by 10–15 knots within 3 hours. The ML model will attempt to fit meteorological features (SST, shear) to explain an artificial administrative measurement jump.
2. **Loss Function Corruption:** Mean Squared Error (MSE) or Huber loss will be heavily penalized by agency boundary steps rather than physical dynamics.

### Recommended Target & Feature Strategy for Model 1

1. **Target Selection:** Standardize exclusively on **`USA_WIND` (1-minute sustained wind, in knots)**.
   - Globally unified under JTWC (Western Pacific, Indian Ocean, Southern Hemisphere) and NHC/CPHC (Atlantic, Eastern & Central Pacific).
   - Provides **262,945 training observations** post-1980 with unbroken sequence continuity.
2. **Prediction Horizons:** Formulate targets as multi-horizon forecasts:
   - $V_{t+6h}$ (short-term persistence/trend)
   - $V_{t+12h}$ (intermediate diurnal/structural change)
   - $V_{t+24h}$ (primary operational synoptic horizon)
3. **Core Feature Inputs:**
   - Current & lag intensity: $V_t, V_{t-6h}, V_{t-12h}, \Delta V_{-6h}, \Delta V_{-12h}$
   - Kinematics: `STORM_SPEED`, `STORM_DIR`, forward acceleration
   - Proximity to land: `DIST2LAND`, `LANDFALL` (critical decay dampening factor)
   - Spatial/Climatological: `LAT`, `LON`, day of year ($\sin/\cos$ harmonic encoding), `SEASON`
   - Environmental/Structural proxies: `USA_SSHS` (Saffir-Simpson category), `USA_STATUS` (depression, storm, hurricane), and quadrant radii asymmetry indices (post-2000 subcohort).

---

## 3. Model 2 Feasibility Audit: Cyclone Movement Prediction

**Proposed Targets:** Future Latitude and Longitude ($\text{LAT}_{t+\Delta t}, \text{LON}_{t+\Delta t}$) at 6h, 12h, and 24h.

### 1. Track Types & Filtering Criteria

IBTrACS classifies storm track segments using the `TRACK_TYPE` attribute:

| `TRACK_TYPE` Value | Observation Count | Unique Cyclones | Operational Meaning & Usability |
| :--- | :--- | :--- | :--- |
| **`main`** | **710,515** (97.28%) | **13,170** | Primary cyclone vortex trajectory. **Mandatory training subset.** |
| `US-PROVISIONAL` | 6,353 (0.87%) | 92 | Real-time provisional fixes before postseason post-analysis. |
| `spur-split` | 4,790 (0.66%) | 133 | Secondary bifurcating vortex track (e.g. Fujiwhara interaction). Exclude. |
| `PROVISIONAL` | 4,241 (0.58%) | 77 | Provisional unverified real-time operational track. Exclude. |
| `spur-merge` | 2,871 (0.39%) | 89 | Merging secondary vortices. Exclude. |
| `spur-other` | 1,621 (0.22%) | 69 | Miscellaneous secondary branches. Exclude. |
| `US-PROVISIONAL_spur`| 25 (0.00%) | 1 | Provisional secondary spur. Exclude. |

> [!WARNING]
> Secondary spurs represent split centers or merging vortices. Feeding spur segments into a sequential track model introduces artificial coordinate jumps and breaks trajectory continuity. **Training must strictly filter for `TRACK_TYPE == 'main'`.**

### 2. Observation Counts per Cyclone (1970–Present Cohort)

For main-track cyclones from 1970 onwards (**6,014 cyclones**, **374,705 fixes**):
- **Minimum:** 1 fix
- **25th Percentile (Q1):** 33.25 fixes (~4.2 days)
- **Median:** **57.0 fixes (~7.1 days)**
- **Mean:** **62.11 fixes (~7.8 days)**
- **75th Percentile (Q3):** 83.0 fixes (~10.4 days)
- **Maximum:** 350 fixes (*TC Helinda:Pancho*, 32.25 days)

Over **85%** of cyclones have $\ge 24$ observations (3 days of history), providing sufficient sequence depth for multi-step recurrent neural networks (RNNs/LSTMs) or Transformer temporal attention mechanisms.

### 3. Implications of IBTrACS Linear Position Interpolation

A critical architectural feature of the IBTrACS v04 global dataset:
- Standard best-track data from operational meteorological agencies is generated at **6-hourly synoptic intervals** (00:00, 06:00, 12:00, 18:00 UTC).
- To create a regularized global product, the IBTrACS algorithm linearly interpolates storm centers to produce **3-hourly positions** (03:00, 09:00, 15:00, 21:00 UTC), flagged in the `IFLAG` column.

#### Operational Implications for Machine Learning
1. **The 3-Hour Horizon Trap:** If an ML model is trained to predict track position at $\Delta t = 3\text{h}$ using 3-hourly sampled inputs, half of the target ground-truth labels are **mathematical linear midpoints** between $t-3\text{h}$ and $t+3\text{h}$. The model will simply learn to emulate NOAA's linear interpolation code rather than atmospheric steering physics.
2. **6h, 12h, and 24h Horizons:** Evaluating on **6h, 12h, and 24h** prediction horizons aligns forecasts with true physical synoptic fixes (00, 06, 12, 18 UTC), guaranteeing that model performance is measured against genuine meteorological observations.

### 4. Longitude Nuance: Coordinates Exceeding $180^\circ$

Our audit identified **10,658 observations** across **642 cyclones** where `LON > 180.0°`:
- **Basins Affected:** **South Pacific (`SP`): 6,609 fixes**; **Eastern Pacific (`EP`): 4,049 fixes**.
- **Distribution across Eras:** Pre-1970 (1,775), 1970–1999 (4,932), 2000–Present (3,951).
- **Maximum Longitude Recorded:** **$266.9^\circ\text{E}$** (in the Gulf of Mexico / East Pacific transition).

```text
       178°E    179°E     180°    -179°W   -178°W   (Standard -180 to +180)
─────────┼────────┼────────┼────────┼────────┼────────►
       178°E    179°E    180°     181°E    182°E    (IBTrACS 0 to 360 convention)
```

#### Hazard for Spatial Loss Functions
In standard GIS systems, $-179.0^\circ\text{W}$ and $+181.0^\circ\text{E}$ represent the identical geographical meridian. However, an unnormalized loss function (e.g. Mean Squared Error on raw longitude) will compute:

$$\text{Loss} = (181.0 - (-179.0))^2 = 360^2 = 129,600$$

This numerical catastrophe causes exploding gradients and destabilizes training whenever a cyclone nears the Antimeridian.

### 5. Usable Training Pairs for Movement Prediction

A training observation at timestamp $t$ is defined as "usable" for horizon $\Delta t$ if and only if the cyclone has a valid, continuous track observation at $t + \Delta t$ within the same storm ID:

| Prediction Horizon ($\Delta t$) | Global Usable Pairs | 1970–Present Usable Pairs | 1970–Present Main-Track Pairs | 1980–Present Main-Track Pairs |
| :--- | :--- | :--- | :--- | :--- |
| **6-Hour Horizon ($t+6\text{h}$)** | 684,958 | 371,604 | **359,520 pairs** | 290,015 pairs |
| **12-Hour Horizon ($t+12\text{h}$)** | 669,810 | 359,123 | **347,596 pairs** | 280,424 pairs |
| **24-Hour Horizon ($t+24\text{h}$)** | 615,297 | 333,781 | **323,346 pairs** | 261,187 pairs |

### Recommended Feature & Architecture Strategy for Model 2

1. **Cohort & Filter:** Cohort **1970–Present**, filtered strictly for `TRACK_TYPE == 'main'`.
2. **Antimeridian Coordinate Normalization:**
   - Preprocess all longitude coordinates into standard signed degrees:
     $$\text{LON}_{\text{norm}} = ((\text{LON} + 180) \pmod{360}) - 180$$
   - For sequential inputs crossing the $180^\circ$ boundary, implement **track unwrapping** so trajectory sequences remain spatially continuous without artificial $360^\circ$ jumps.
3. **Cartesian Displacement Formulation:** Rather than predicting raw spherical coordinates directly, formulate targets as **displacement vectors in kilometers**:
   $$\Delta x = R_{\text{earth}} \cdot (\lambda_{t+\Delta t} - \lambda_t) \cdot \cos\left(\frac{\phi_{t+\Delta t} + \phi_t}{2}\right)$$
   $$\Delta y = R_{\text{earth}} \cdot (\phi_{t+\Delta t} - \phi_t)$$
4. **Predictor Features:**
   - Past trajectory coordinates: $\text{LAT}_t, \text{LON}_t, \Delta x_{-6h}, \Delta y_{-6h}, \Delta x_{-12h}, \Delta y_{-12h}$
   - Translation components: $u = -V_{\text{storm}} \sin(\theta)$, $v = -V_{\text{storm}} \cos(\theta)$
   - Current intensity: `USA_WIND` (proxy for vertical depth of steering flow interaction)
   - Domain encoding: Ocean basin one-hot encoding, distance to land (`DIST2LAND`).

---

## 4. Model 3 Feasibility Audit: Cyclone Development Trend Classification

**Objective:** Classify whether a tropical cyclone will **Strengthen**, remain **Stable**, or **Weaken** over the subsequent 24 hours.

**Label Source:** Future 24-hour wind-speed change:

$$\Delta V_{24h} = V_{t+24h} - V_t$$

### 1. Data Availability across 24-Hour Windows

Evaluating complete 24-hour forward wind-speed pairs using `USA_WIND`:
- **Global All-Time Pairs:** **344,728 valid pairs** across **7,226 unique cyclones**.
- **1980–Present Cohort:** **224,918 valid pairs** across **4,500+ unique cyclones**.
- **2000–Present Cohort:** **122,756 valid pairs** across **2,600+ unique cyclones**.

### 2. Empirical Distribution of 24-Hour Wind Changes ($\Delta V_{24h}$)

Summary statistics for $\Delta V_{24h}$ across the recommended 1980–Present cohort (**224,918 pairs**):
- **Minimum:** $-160\text{ kts}$ (catastrophic landfall / post-tropical collapse)
- **25th Percentile (Q1):** $-8.0\text{ kts}$
- **Median:** **$0.0\text{ kts}$**
- **Mean:** **$+1.02\text{ kts}$**
- **75th Percentile (Q3):** $+10.0\text{ kts}$
- **Maximum:** $+105\text{ kts}$ (extreme hyper-rapid intensification)
- **Standard Deviation:** $17.71\text{ kts}$
- **Rapid Intensification (RI: $\Delta V_{24h} \ge +30\text{ kts}$):** 18,080 cases (**5.24%**)
- **Rapid Weakening (RW: $\Delta V_{24h} \le -30\text{ kts}$):** 18,833 cases (**5.46%**)

### 3. Quantitative Class Distribution under Candidate Thresholds

To establish a mathematically balanced and meteorologically defensible classification scheme, we tested four candidate threshold formulations on the 1980–Present cohort:

```text
Candidate A (Strict Zero):       Strengthening (> 0)  | Stable (== 0)      | Weakening (< 0)
Candidate B (±5 kts):            Strengthening (≥ +5) | Stable (-5 to +5)  | Weakening (≤ -5)
Candidate C (±10 kts):           Strengthening (≥+10) | Stable (-10 to +10)| Weakening (≤ -10)
Candidate D (±15 kts):           Strengthening (≥+15) | Stable (-15 to +15)| Weakening (≤ -15)
```

#### Empirical Class Distribution Matrix (1980–Present, $N = 224,918$)

| Threshold Candidate | Strengthening Count (%) | Stable Count (%) | Weakening Count (%) | Class Ratio (Max / Min) | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate A (Strict 0 kts)** | 108,751 (**48.35%**) | 32,428 (**14.42%**) | 83,739 (**37.23%**) | **3.35** | **Severely Imbalanced.** Stable class under-represented. |
| **Candidate B ($\pm 5\text{ kts}$)** | 102,213 (**45.44%**) | 44,770 (**19.91%**) | 77,935 (**34.65%**) | **2.28** | **Sub-optimal.** Noise-dominated boundary. |
| **Candidate C ($\pm 10\text{ kts}$)** | **71,209 (31.66%)** | **100,100 (44.51%)** | **53,609 (23.83%)** | **1.87** | **Optimal Balance & Meteorological Fidelity.** |
| **Candidate D ($\pm 15\text{ kts}$)** | 47,553 (**21.14%**) | 140,661 (**62.54%**) | 36,704 (**16.32%**) | **3.83** | **Severely Imbalanced.** Stable class dominates excessively. |

#### Empirical Comparison Across Cohorts for Candidate C ($\pm 10\text{ kts}$)

| Cohort | Total Pairs | Strengthening ($\ge +10\text{ kts}$) | Stable ($-10 < \Delta V < +10\text{ kts}$) | Weakening ($\le -10\text{ kts}$) | Max/Min Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Global All-Time** | 344,728 | 106,240 (30.82%) | 155,431 (45.09%) | 83,057 (24.09%) | **1.87** |
| **1980–Present** | 224,918 | 71,209 (31.66%) | 100,100 (44.51%) | 53,609 (23.83%) | **1.87** |
| **2000–Present** | 122,756 | 39,614 (32.27%) | 52,792 (43.01%) | 30,350 (24.72%) | **1.74** |

### 4. Meteorological & Mathematical Rationale for Candidate C ($\pm 10\text{ kts}$)

1. **Observational Uncertainty & Dvorak Resolution:** Operational satellite intensity estimates (Dvorak technique) are quantized in increments of 0.5 T-numbers, corresponding to roughly $\pm 5\text{ kts}$ of wind speed error. Furthermore, best-track winds are reported rounded to the nearest 5 knots. Under Candidate A, a single 5-knot wobble is classified as "Strengthening" or "Weakening", training the model on instrument noise. A $\pm 10\text{ kts}$ window corresponds to two discrete reporting bins ($1.0$ T-number), ensuring that classified trends represent genuine physical intensity changes.
2. **Statistical Parity:** With Candidate C, all three classes are well represented (**31.7% / 44.5% / 23.8%**). No class drops below 23%, and the ratio between the majority and minority class is just **1.87**, completely avoiding pathological minority collapse without requiring synthetic oversampling (SMOTE).
3. **Operational Relevance:** In tropical cyclone forecasting, a wind change of $< 10\text{ kts}$ in 24 hours is universally categorized as a steady/neutral intensity state.

### 5. Data Quality Limitations & Covariate Roles

- **Agency Consistency:** All 24h pairs must be computed strictly using a single agency stream (`USA_WIND`). If $V_t$ and $V_{t+24h}$ were taken from different agency columns, the resulting $\Delta V_{24h}$ would reflect averaging period discrepancies rather than physical intensity evolution.
- **Landfall Interaction:** Cyclones encountering land undergo severe frictionally forced filling ($\Delta V \ll -10\text{ kts}$). The model must ingest `DIST2LAND` and `LANDFALL` flags as features so it can differentiate between thermodynamic maritime decay and mechanical topographic destruction.
- **Role of the `NATURE` Column:** As specified in project instructions, `NATURE` (`TS`, `ET`, `DS`, `MX`, `SS`) is **not the prediction target**. Rather, it provides an invaluable categorical feature: extratropical transitions (`ET`) and subtropical systems (`SS`) exhibit asymmetric thermal and wind structures that strongly modulate 24-hour intensity changes.

---

## 5. Comprehensive Synthesis & Implementation Roadmap

### Summary of Recommendations for All Three Models

| Model | Recommended Cohort | Recommended Target | Target Units / Classes | Primary Input Features |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1: Intensity Prediction** | **1980–Present** (5,019 cyclones) | `USA_WIND` at $t+6\text{h}, t+12\text{h}, t+24\text{h}$ | Continuous (knots, 1-min sustained) | Current & lag winds ($V_t, V_{-6h}, V_{-12h}$), speed/direction, `DIST2LAND`, `USA_SSHS`, temporal harmonics |
| **Model 2: Movement Prediction** | **1970–Present** (6,014 main-track cyclones) | Future displacement ($\Delta x, \Delta y$) at 6h, 12h, 24h | Continuous (kilometers, Cartesian) | Past coordinates ($\text{LAT}, \text{LON}_{\text{norm}}$), displacement history, $u/v$ velocity, `USA_WIND`, `DIST2LAND` |
| **Model 3: Development Trend** | **1980–Present** (4,500+ cyclones) | 24-hour wind change class ($\Delta V_{24h}$) | 3-Class Categorical:<br>• **Strengthening** ($\ge +10\text{ kts}$)<br>• **Stable** ($-10 < \Delta V < +10\text{ kts}$)<br>• **Weakening** ($\le -10\text{ kts}$) | $\Delta V_{-6h}, \Delta V_{-12h}$, current intensity $V_t$, `DIST2LAND`, `NATURE` (categorical), translation speed, latitude |

### Critical Data Preparation Directives for Subsequent Phases

1. **Parser Protocol:** Always load the raw CSV using:
   ```python
   pd.read_csv(
       'raw/ibtracs.ALL.list.v04r01.csv',
       skiprows=[1],
       na_values=['', ' '],
       keep_default_na=False,
       low_memory=False
   )
   ```
2. **Coordinate Normalization Pipeline:**
   ```python
   # Convert longitude to [-180, +180] degrees
   df['LON_NORM'] = ((df['LON'] + 180.0) % 360.0) - 180.0
   ```
3. **Track Filtering:** Enforce `df['TRACK_TYPE'] == 'main'` for movement sequence construction.
4. **Target Isolation:** Never blend `USA_WIND` and `WMO_WIND` or apply arbitrary conversion scalars.

---
*Report completed and verified under Task 2 requirements. End of audit.*
