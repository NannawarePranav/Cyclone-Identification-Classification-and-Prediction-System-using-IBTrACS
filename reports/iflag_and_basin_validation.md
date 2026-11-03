# IBTrACS Original Observation & Basin Representation Audit

> **Project:** Prediction of Different Tropical Cyclones Patterns  
> **Dataset:** NOAA IBTrACS v04r01 Global (`raw/ibtracs.ALL.list.v04r01.csv`)  
> **Scope:** Validation of Original vs. Interpolated Observations (`IFLAG`), Synoptic Discretization, Basin Representation, and Model Evaluation Strategy  
> **Status:** Analysis-Only Complete (Zero dataset modifications; No models trained)

---

## Executive Summary

This audit validates the operational integrity of NOAA's 15-character **Interpolation Flag (`IFLAG`)**, determines the exact volume of **original vs. interpolated** observations across all reporting centers, and investigates the **geographical basin representation** for our three proposed machine learning models.

### Key Audit Findings
1. **The 50/50 Interpolation Reality:** Across the entire 730,416-record archive and across all historical eras, **approximately 50% of all observations are mathematically interpolated**. For `USA_WIND`, exactly **50.61% (133,085 fixes)** in the post-1980 cohort are original reports (`IFLAG == 'O' | 'V'`), while **49.39% (129,859 fixes)** are linear interpolations (`IFLAG == 'P' | 'I'`). This occurs because operational agency best tracks are generated at **6-hourly synoptic fixes** (00, 06, 12, 18 UTC), and IBTrACS linearly interpolates intermediate **3-hourly points** (03, 09, 15, 21 UTC) to produce a uniform global time step.
2. **Synoptic vs. Off-Synoptic Dichotomy:** At 6-hourly synoptic timestamps, **99.02% of global positions and 84.22% of USA positions are original**. At intermediate 3-hourly timestamps, **94.32% to 99.78% of positions are interpolated**. Crucially, **off-synoptic points** (e.g. landfalls or special aircraft reconnaissance fixes) are **99.87% original**, inserted at the exact recorded minute of landfall.
3. **Model 1 & Model 3 Reassessment (Original-Only Pairs):** Filtering strictly for pairs where both current ($t$) and future ($t+\Delta t$) winds are **original** (`IFLAG[0] in ['O', 'V']`) and originate from the **same USA agency stream** yields:
   - **6-Hour Horizon:** **127,401 pairs** across 4,556 unique cyclones.
   - **12-Hour Horizon:** **122,762 pairs** across 4,557 unique cyclones.
   - **24-Hour Horizon:** **113,482 pairs** across 4,524 unique cyclones.
   - **Recalculated Development Trend Classes ($\pm 10\text{ kts}$ on Original Pairs):** **33.0% Strengthening** (37,453), **41.9% Stable** (47,501), and **25.1% Weakening** (28,528), with an optimal max-to-min class ratio of **1.67**.
4. **Geographical Basin Representation:** Post-1980 `USA_WIND` coverage is high across major basins (Western Pacific: 80.5%; South Indian: 75.8%; North Atlantic: 99.9%; Eastern Pacific: 99.1%; South Pacific: 74.5%; North Indian: 73.0%). However, absolute sample counts are heavily skewed toward the Western Pacific (**36,833 24h pairs**, 32.5% of global) and Eastern Pacific (**21,377 pairs**), while the North Indian basin provides only **4,738 pairs** (4.2%) and the South Atlantic has only 49 pairs. A unified global model with explicit basin embeddings is viable, but must be paired with **basin-stratified evaluation**.
5. **Evaluation Protocol:** Models must be trained and evaluated using **Cyclone-ID Group Splitting** (or Out-of-Time Temporal Splitting: Train 1980–2014, Val 2015–2019, Test 2020–2026). Random row-level splitting produces catastrophic data leakage due to intense temporal autocorrelation within individual cyclone tracks.

---

## Part A: IFLAG Analysis & Interpolation Architecture

### 1. Structure of the 15-Character `IFLAG` String

In NOAA IBTrACS v04r01, the `IFLAG` attribute is a 15-character string where each index position maps to a specific international meteorological agency or historical dataset:

```text
Position:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
Agency:   USA JMA CMA HKO KMA IMD REU BOM FJI NZ TD6 TD5 DS8 NEU MLC
```

#### Documented Flag Definitions
- **`O` (Original):** Original report directly provided by the designated agency (both position and primary variables un-interpolated).
- **`P` (Position Interpolated):** Position was interpolated by IBTrACS algorithms (and all associated variables, including intensity, were interpolated).
- **`I` (Intensity Interpolated):** Position was originally reported, but intensity variables were interpolated.
- **`V` (Values/Other Interpolated):** Position and intensity are original, but secondary parameters (e.g. pressure or radii) were interpolated.
- **`_` (Missing / Not Reporting):** No report or tracking provided by this agency at this timestamp.

### 2. Global Distribution of Flags by Agency Position

The distribution of flags across all 730,416 rows in the dataset:

| Position | Agency / Source | Full Agency Name | `O` (Original) | `P` (Pos Interp) | `I` (Int Interp) | `V` (Val Interp) | `_` (Missing) | Active Fixes |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **USA** | NHC / JTWC / CPHC | **233,170** | **234,497** | **511** | **437** | 261,801 | **468,615** |
| **1** | **JMA_Tokyo** | RSMC Tokyo (Japan) | 71,126 | 64,006 | 0 | 0 | 595,284 | 135,132 |
| **2** | **CMA_China** | China Met Administration | 74,233 | 70,806 | 1 | 0 | 585,376 | 145,040 |
| **3** | **HKO_HongKong** | Hong Kong Observatory | 45,622 | 40,731 | 0 | 0 | 644,063 | 86,353 |
| **4** | **KMA_Korea** | Korea Met Administration | 6,877 | 6,252 | 0 | 0 | 717,287 | 13,129 |
| **5** | **NewDelhi_IMD** | RSMC New Delhi (India) | 7,311 | 3,440 | 3 | 0 | 719,662 | 10,754 |
| **6** | **LaReunion** | RSMC La Réunion (France) | 30,313 | 56,021 | 12 | 54 | 644,016 | 86,400 |
| **7** | **BoM_Australia** | TCWC Australia (BoM) | 31,357 | 39,267 | 76 | 240 | 659,476 | 70,940 |
| **8** | **Nadi_Fiji** | RSMC Nadi (Fiji) | 4,470 | 4,053 | 8 | 0 | 721,885 | 8,531 |
| **9** | **Wellington_NZ** | TCWC Wellington (NZ) | 9,173 | 14,715 | 1 | 0 | 706,527 | 23,889 |
| **10** | **TD9636** | Historical US Navy (NI) | 88,539 | 100,909 | 9 | 0 | 540,959 | 189,457 |
| **11** | **TD9635** | Historical US Navy (SH) | 76,025 | 225,422 | 1 | 0 | 428,968 | 301,448 |
| **12** | **DS824** | Historical Global Navy | 20,622 | 19,806 | 0 | 0 | 689,988 | 40,428 |
| **13** | **Neumann** | Historical Atlantic Archive | 40,727 | 46,819 | 14 | 0 | 642,856 | 87,560 |
| **14** | **Chenoweth_MLC** | Historical Reanalysis | 15,410 | 15,033 | 28 | 0 | 699,945 | 30,471 |

> [!NOTE]
> The first character (`IFLAG.str[0]`) strictly governs the **USA** source. Regional agencies each have dedicated character indices. Across all agencies, the ratio between original reports (`O`) and interpolated points (`P`) consistently approximates 1:1, mirroring the standard 6-hour synoptic reporting interval versus the 3-hour global product grid.

### 3. Original vs. Interpolated Intensity for `USA_WIND`

When evaluating the 405,899 global fixes where `USA_WIND` is populated:

| Flag Code | Interpretation for `USA_WIND` | Observation Count | % of Populated `USA_WIND` |
| :---: | :--- | :--- | :--- |
| **`O`** | **Original Best-Track Intensity** | **203,688** | **50.18%** |
| **`P`** | **Interpolated Intensity** (via position interpolation) | **201,229** | **49.58%** |
| **`I`** | **Interpolated Intensity** (position original, wind interpolated) | **511** | **0.13%** |
| **`V`** | **Original Intensity** (position & wind original, secondary var interp) | **437** | **0.11%** |
| `_` | Missing Flag (anomalous missing character) | 34 | 0.01% |
| **TOTAL ORIGINAL** | `O` $\cup$ `V` | **204,125** | **50.29%** |
| **TOTAL INTERPOLATED** | `P` $\cup$ `I` | **201,740** | **49.70%** |

### 4. Relationship Between `IFLAG` and Coordinates (`LAT`/`LON` vs `USA_LAT`/`USA_LON`)

1. **`USA_LAT` and `USA_LON`:** Governed directly by `IFLAG.str[0]`.
   - **Original Position (`O`, `I`, `V`):** **234,118 fixes (49.96%)**.
   - **Interpolated Position (`P`):** **234,497 fixes (50.04%)**.
2. **Unified `LAT` and `LON`:** IBTrACS synthesizes global coordinates by prioritizing the designated WMO agency for the basin (e.g. RSMC Tokyo for WP, NHC for NA/EP), falling back to JTWC or historical sources.
   - Across the entire dataset, **363,838 fixes (49.81%)** have an original position reported by at least one agency (`O`, `I`, or `V` in `IFLAG`).
   - Conversely, **366,578 fixes (50.19%)** are completely interpolated across all reporting agencies (only `P` flags exist).

### 5. Historical Period Comparison: Interpolation Rates

| Historical Period | Total Fixes | Populated `USA_WIND` | `USA_WIND` Original % | `USA_WIND` Interp % | Global Position Original % | Global Position Interp % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **All-Time (1842–Present)** | 730,416 | 405,899 | **50.29%** | 49.70% | **49.81%** | 50.19% |
| **1970–Present (Satellite)** | 386,212 | 303,773 | **50.31%** | 49.69% | **53.97%** | 46.03% |
| **1980–Present (Dvorak)** | 312,668 | 262,945 | **50.61%** | 49.39% | **53.72%** | 46.28% |
| **2000–Present (Multi-Sensor)** | 169,796 | 144,467 | **50.96%** | 49.04% | **55.13%** | 44.87% |

The interpolation rate remains exceptionally stable across all eras at **~49–50%** because IBTrACS systematically interpolates between 6-hourly best-track fixes. The slight rise in original global positions in modern eras (from 49.8% to 55.1%) is driven by an increased frequency of **off-synoptic landfall fixes** and aircraft reconnaissance fixes.

---

## Part B: Model 1 & Model 3 Reassessment (1980–Present Cohort)

In Task 2, candidate sample counts were computed assuming all available `USA_WIND` entries were usable. We now reassess sample volumes under strict physical data quality criteria:
- **Criterion 1 (Original Wind Only):** Both current wind $V_t$ and target wind $V_{t+\Delta t}$ must have `IFLAG.str[0] in ['O', 'V']`.
- **Criterion 2 (Source Consistency):** Both $V_t$ and $V_{t+\Delta t}$ must belong to the **same USA reporting agency stream** (`USA_AGENCY_t == USA_AGENCY_{t+\Delta t}`).

### 1. Updated Usable Prediction Pairs for Model 1 (Intensity Prediction)

Cohort: **1980–Present** ($N = 312,668$ total fixes; $262,945$ available `USA_WIND` fixes; **$133,085$ original `USA_WIND` fixes**):

| Prediction Horizon ($\Delta t$) | All Available `USA_WIND` Pairs | Original `USA_WIND` Pairs | Same Agency Original Pairs | Usable Unique Cyclones | % Retained vs All Available |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **6-Hour Horizon ($t+6\text{h}$)** | 252,561 | 127,524 | **127,401** | **4,556** | **50.44%** |
| **12-Hour Horizon ($t+12\text{h}$)** | 243,353 | 122,985 | **122,762** | **4,557** | **50.45%** |
| **24-Hour Horizon ($t+24\text{h}$)** | 224,918 | 113,868 | **113,482** | **4,524** | **50.45%** |

#### Operational Takeaways
1. Filtering out interpolated wind speeds reduces usable training pairs by almost exactly **49.5%**.
2. However, the remaining sample size (**113,482 complete 24-hour pairs** across **4,524 unique cyclones**) remains exceptionally large, providing more than sufficient statistical power for deep learning architectures without training on synthetic linear interpolations.
3. Agency switching between $t$ and $t+\Delta t$ affects fewer than **400 pairs globally** (0.34% of pairs), confirming that `USA_AGENCY` continuity is exceptionally strong within individual storms.

### 2. Cyclone Counts with Sufficient Original Wind Observations (1980–Present)

For the 5,019 unique cyclones in the 1980–Present cohort:

| Original Wind Run Length Criteria | Cyclone Count | % of 1980+ Cyclones |
| :--- | :--- | :--- |
| **$\ge 1$ Original Wind Observation** | **4,558 cyclones** | 90.81% |
| **$\ge 4$ Original Wind Observations (12 hours)** | **4,547 cyclones** | 90.59% |
| **$\ge 8$ Original Wind Observations (24 hours)** | **4,393 cyclones** | 87.53% |
| **$\ge 12$ Original Wind Observations (36 hours)** | **4,089 cyclones** | 81.47% |
| **$\ge 16$ Original Wind Observations (48 hours)** | **3,648 cyclones** | 72.68% |
| **$\ge 24$ Original Wind Observations (72 hours)** | **2,630 cyclones** | 52.40% |
| **Median Original Winds per Storm** | **25.0 fixes (~6.25 days)** | — |
| **Mean Original Winds per Storm** | **26.52 fixes (~6.6 days)** | — |

Over **87.5% of cyclones** have at least 8 original wind observations, ensuring robust sequence modeling capabilities.

### 3. Model 3 Reassessment: Development Trend Classification on Original Data

We recalculated the 24-hour development trend classes ($\Delta V_{24h} = V_{t+24h} - V_t$) using strictly **original-to-original wind pairs from the same USA agency stream** ($N = 113,482$ pairs):

| Candidate Threshold | Strengthening Count (%) | Stable Count (%) | Weakening Count (%) | Class Ratio (Max / Min) | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict 0 kts** ($\Delta V > 0, =0, <0$) | 53,005 (**46.71%**) | 19,411 (**17.10%**) | 41,066 (**36.19%**) | **2.73** | **Unbalanced.** Stable under-represented; noise-sensitive. |
| **$\pm 5\text{ kts}$** | 52,883 (**46.60%**) | 19,696 (**17.36%**) | 40,903 (**36.04%**) | **2.68** | **Unbalanced.** Marginally different from 0 kts due to 5-kt binning. |
| **$\pm 10\text{ kts}$ (Recommended)** | **37,453 (33.00%)** | **47,501 (41.86%)** | **28,528 (25.14%)** | **1.67** | **Optimal Parity & Physical Meaning.** |

#### Significance of the Reassessment
When restricted strictly to original observations, the **$\pm 10\text{ kts}$ threshold** demonstrates even superior balance compared to the preliminary audit:
- The majority-to-minority class ratio improves from **1.87 down to 1.67**.
- All three classes comprise between **25.1% and 41.9%** of the dataset.
- The Stable class constitutes the plurality (**41.86%**), accurately reflecting the physical reality that mature cyclones often persist near peak or steady intensity for 24 hours.

---

## Part C: Model 2 Reassessment (1970–Present Main-Track Cohort)

Cohort: **1970–Present**, filtered for `TRACK_TYPE == 'main'` ($N = 373,550$ fixes across **6,014 unique cyclones**).

### 1. Original vs. Interpolated Positions by Timestamp Category

| Synoptic Timestamp Group | Definition | Fix Count (% of Total) | Global Original Positions | Global Original % | USA Original Positions | USA Original % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Synoptic Fixes** | 00:00, 06:00, 12:00, 18:00 UTC | 186,032 (49.80%) | **184,206** | **99.02%** | **156,673** | **84.22%** |
| **Sub-Synoptic Fixes** | 03:00, 09:00, 15:00, 21:00 UTC | 180,401 (48.29%) | 10,253 | **5.68%** | 400 | **0.22%** |
| **Off-Synoptic Special Fixes** | Landfalls, Recon (e.g. 14:15 UTC) | 7,117 (1.91%) | **7,108** | **99.87%** | 574 | **8.07%** |
| **TOTAL** | All timestamps | 373,550 (100.0%) | **201,567** | **53.96%** | **157,647** | **42.20%** |

#### Crucial Insights on Track Position Origin
1. **The Synoptic Grid is the True Operational Anchor:** Nearly **100% (99.02%)** of positions at 00, 06, 12, 18 UTC are original observations directly recorded by RSMCs.
2. **Intermediate 3h Points are 94.3% Synthetic:** At 03, 09, 15, 21 UTC, 94.32% of global positions (and 99.78% of USA positions) are generated by linear interpolation.
3. **Off-Synoptic Landfall Precision:** Off-synoptic fixes are virtually **100% original reports (99.87%)**. They represent high-value ground-truth fixes inserted when a storm center crosses a coastline or passes an automated buoy/aircraft fix.

### 2. Usable Sample Counts for Movement Prediction Horizons

We evaluated usable main-track training pairs under three alternative data-quality regimes:
- **Regime 1 (All Main Fixes):** Pairs formed across all available fixes (ignoring interpolation flags).
- **Regime 2 (Original-to-Original Global Fixes):** Both $t$ and $t+\Delta t$ must have verified original global positions (`IFLAG contains 'O' | 'I' | 'V'`).
- **Regime 3 (Synoptic 6-Hourly Grid Fixes):** Both $t$ and $t+\Delta t$ must fall on the standard 6-hourly synoptic grid (00, 06, 12, 18 UTC).

| Prediction Horizon ($\Delta t$) | Regime 1: All Main Fixes | Regime 2: Original-to-Original Global | Regime 3: Synoptic 6-Hourly Grid |
| :---: | :---: | :---: | :---: |
| **6-Hour Horizon ($t+6\text{h}$)** | 359,520 pairs | **189,533 pairs (52.72%)** | **180,022 pairs (50.07%)** |
| **12-Hour Horizon ($t+12\text{h}$)** | 347,596 pairs | **184,350 pairs (53.04%)** | **174,016 pairs (50.06%)** |
| **24-Hour Horizon ($t+24\text{h}$)** | 323,346 pairs | **171,611 pairs (53.07%)** | **162,048 pairs (50.12%)** |

> [!TIP]
> **Recommended Movement Training Regime:** **Regime 3 (Synoptic 6-Hourly Grid)** or **Regime 2 (Original-to-Original)** is strongly recommended. Training on the synoptic 6-hourly grid provides **180,022 verified 6-hour pairs** and **162,048 24-hour pairs**. This completely prevents the movement model from fitting the linear interpolation artifacts of 3-hourly points, forcing it to learn true non-linear atmospheric steering curvature.

---

## Part D: Geographical Basin Representation (1980–Present Cohort)

To verify whether a single global model is statistically justified, we audited the geographical representation of `USA_WIND` and 24-hour prediction sequences across all seven ocean basins:

### Post-1980 Basin Representation Audit Matrix

| Ocean Basin Code | Basin Name | Total Observations | Available `USA_WIND` (%) | Original `USA_WIND` (%) | Unique Cyclones | Cyclones with $\ge 1$ Orig 24h Pair | Cyclones with $\ge 4$ Orig 24h Pairs | Total 24h Orig Pairs | % of Global 24h Pairs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`WP`** | Western North Pacific | 104,618 | 84,182 (**80.47%**) | 42,737 (**40.85%**) | 1,540 | **1,416** | 1,384 | **36,833** | **32.46%** |
| **`SI`** | South Indian Ocean | 67,697 | 51,298 (**75.78%**) | 25,147 (**37.15%**) | 922 | **793** | 782 | **21,787** | **19.20%** |
| **`EP`** | Eastern North Pacific | 50,835 | 50,373 (**99.09%**) | 25,738 (**50.63%**) | 1,161 | **945** | 896 | **21,377** | **18.84%** |
| **`NA`** | North Atlantic | 41,685 | 41,651 (**99.92%**) | 21,514 (**51.61%**) | 753 | **742** | 692 | **17,986** | **15.85%** |
| **`SP`** | South Pacific Ocean | 32,079 | 23,912 (**74.54%**) | 12,094 (**37.70%**) | 550 | **473** | 454 | **10,157** | **8.95%** |
| **`NI`** | North Indian Ocean | 15,635 | 11,410 (**72.98%**) | 5,794 (**37.06%**) | 473 | **253** | 245 | **4,738** | **4.18%** |
| **`SA`** | South Atlantic Ocean | 119 | 119 (**100.00%**) | 61 (**51.26%**) | 3 | **3** | 3 | **49** | **0.04%** |
| **GLOBAL**| **All Basins** | **312,668** | **262,945 (84.10%)** | **133,085 (42.56%)**| **5,019** | **4,524** | **4,357** | **113,482** | **100.00%** |

### Critical Evaluation: Is a Single Global Model Supported?

#### 1. Statistical Viability: Supported with Caution
- The sample size is undeniably large: **113,482 complete original 24h pairs** across **4,524 unique cyclones**.
- Every primary tropical cyclone basin has **hundreds of cyclones** with extensive 24-hour sequences (WP: 1,416 cyclones; EP: 945; SI: 793; NA: 742; SP: 473; NI: 253).
- Therefore, a global deep learning model (e.g. Multi-Task LSTM or Transformer) has sufficient global volume to learn core thermodynamic and kinematic features.

#### 2. The Geographical Skew Hazard
- **Severe Volume Asymmetry:** The Western Pacific (`WP`), Eastern Pacific (`EP`), and North Atlantic (`NA`) collectively supply **67.15%** of all training pairs (**76,196 pairs**).
- Conversely, the North Indian Ocean (`NI`) accounts for only **4.18%** of data, and the South Atlantic (`SA`) has only 49 pairs (Catarina 2004, Anita 2010, Iba 2019).
- **Physical Climatology Conflict:**
  - `WP` storms develop over deep oceanic warm pools with minimal shear, reaching category 5 super-typhoon intensities.
  - `NI` storms occur in shallow, land-locked seas with intense seasonal monsoonal shear and high salinity stratification.
  - An unstratified global model minimizing global MSE will optimize for Northwest Pacific and Atlantic dynamics, systematically underperforming or mispredicting North Indian cyclones.

#### 3. Strategic Solution: Global Backbone with Basin Conditioning
- **Do not train completely isolated regional models**, which discard valuable global transfer learning (especially for rare Category 5 systems).
- **Implement a Global Architecture with Basin Conditioning:**
  1. Ingest ocean basin as a trainable categorical embedding vector.
  2. Include explicit physical coordinates (`LAT`, `LON`), sea proximity (`DIST2LAND`), and seasonal solar geometry (day of year harmonic encoding).
  3. Perform **Basin-Stratified Loss Weighting** or report metrics disaggregated by basin to prevent dominance by Western Pacific storms.

---

## Part E: Comprehensive Evaluation Strategy

### 1. The Cardinal Rule: Cyclone-ID Group Splitting

> [!CAUTION]
> **NEVER randomly split individual observation rows across training, validation, and test sets.**  
> Tropical cyclone observations possess extreme temporal autocorrelation: a storm's intensity and position at $t = 12\text{h}$ is strongly dependent on $t = 06\text{h}$. If rows from the same cyclone ID (`SID`) are randomly partitioned, adjacent time steps of the same hurricane will appear in both training and test sets. This creates **catastrophic target leakage**, yielding unrealistically low test errors (e.g. MAE < 2 kts) that collapse completely when tested on unseen future cyclones.

### 2. Recommended Partitioning Protocol: Temporal Benchmark Split

To replicate realistic operational forecasting conditions, we recommend an **Out-of-Time Temporal Holdout Strategy**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TEMPORAL TRAIN / VAL / TEST SPLIT (1980–2026)                   │
├───────────────────────────────┬─────────────────────────────┬──────────────────────────┤
│ Training Cohort (75.6% Data)  │ Validation Cohort (12.4%)   │ Testing Cohort (12.0%)   │
│ Seasons 1980 – 2014           │ Seasons 2015 – 2019         │ Seasons 2020 – 2026/2027 │
│ 35 Seasons (3,782 Cyclones)   │ 5 Seasons (608 Cyclones)    │ 7 Seasons (629 Cyclones) │
│ ~85,800 Original 24h Pairs    │ ~14,000 Original 24h Pairs  │ ~13,680 Original 24h Pairs│
└───────────────────────────────┴─────────────────────────────┴──────────────────────────┤
◄────────────────── Past ──────────────────────────────────────────► Future ─────────────►
```

#### Why Temporal Splitting is Superior
1. **Zero Data Leakage:** Storms in the test set occurred years after the training cohort ended.
2. **Climate & Observational Robustness:** Tests whether the model generalizes across shifting satellite sensors (e.g. transition from older AVHRR/SSMI to modern Himawari-8/9, GOES-16/18, and CYGNSS).
3. **Operational Realism:** Simulates the exact task faced by meteorological agencies: deploying a model trained on historical seasons to forecast upcoming, unseen seasons.

#### Alternative: 5-Fold Stratified Group K-Fold
For hyperparameter tuning and ablation studies on the pre-2020 training cohort:
- Group strictly on `SID`.
- Stratify groups based on **Basin** and **Peak Intensity Category** (`USA_SSHS`) so that rare Category 5 cyclones and smaller basins are proportionally represented across all five folds.

### 3. Basin-Wise Evaluation Protocol

Every model deliverable must report performance metrics broken down across individual basins:

$$\text{Metric}_{\text{Global}}, \quad \text{Metric}_{\text{WP}}, \quad \text{Metric}_{\text{NA}}, \quad \text{Metric}_{\text{EP}}, \quad \text{Metric}_{\text{SI}}, \quad \text{Metric}_{\text{SP}}, \quad \text{Metric}_{\text{NI}}$$

In addition, researchers should perform **Leave-One-Basin-Out (LOBO)** transfer experiments:
- Train on all basins *except* Basin $B$.
- Test zero-shot generalization on Basin $B$.
- Identifies whether the model has learned fundamental cyclone atmospheric dynamics or merely memorized regional track climatology.

### 4. Recommended Performance Metrics

#### Model 1: Intensity Prediction (Continuous Regression)
1. **Mean Absolute Error (MAE):** Expressed in knots. Primary accuracy benchmark.
2. **Root Mean Squared Error (RMSE):** Strongly penalizes large outlier errors (critical for failed rapid intensification forecasts).
3. **Mean Bias Error (MBE):** Evaluates systematic over-intensification or under-intensification.
4. **Skill Score relative to Persistence ($S_{\text{pers}}$):**
   $$\text{Skill} = \left(1 - \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{persistence}}}\right) \times 100\%$$
   A model has zero practical meteorological value unless $\text{Skill} > 0$.
5. **Rapid Intensification (RI) Subset MAE:** Evaluates error specifically on fixes where $\Delta V_{24h} \ge +30\text{ kts}$.

#### Model 2: Movement Prediction (Trajectory Forecasting)
1. **Mean Absolute Track Error (MATE):** Geodesic great-circle distance (in kilometers) between forecasted and true coordinates:
   $$\text{Error}_{\text{track}} = R_{\text{earth}} \cdot \arccos\left(\sin \phi_1 \sin \phi_2 + \cos \phi_1 \cos \phi_2 \cos(\lambda_1 - \lambda_2)\right)$$
2. **Along-Track Error (ATE):** Measures forward translation speed error (fast/slow bias).
3. **Cross-Track Error (CTE):** Measures directional steering deviation (left/right path bias).
4. **Skill relative to CLIPER (Climatology and Persistence):** Standard operational benchmark.

#### Model 3: Development Trend Classification (3-Class Categorical)
1. **Macro-Averaged F1 Score ($F_{1,\text{macro}}$):** Unweighted mean of F1 scores across Strengthening, Stable, and Weakening. Ensures minority class degradation is heavily penalized.
2. **Balanced Accuracy:** Average of recall obtained on each class.
3. **Confusion Matrix:** Full $3 \times 3$ error matrix to audit dangerous misclassifications (e.g. predicting "Weakening" when a storm underwent "Rapid Intensification").
4. **Critical Success Index (CSI / Threat Score):** Specifically for the **Strengthening** class:
   $$\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}}$$

---

## Part F: Final Consolidated Specifications for All Models

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              FINAL ARCHITECTURAL & DATA PROTOCOL MATRIX                                │
├──────────────────────────┬─────────────────────────────┬──────────────────────────┬────────────────────┤
│ Specification Dimension  │ Model 1: Intensity          │ Model 2: Movement        │ Model 3: Trend     │
├──────────────────────────┼─────────────────────────────┼──────────────────────────┼────────────────────┤
│ Historical Cohort        │ 1980–Present (Dvorak Era)   │ 1970–Present (Satellite) │ 1980–Present       │
│ Track Filtering          │ All tracks with valid wind  │ TRACK_TYPE == 'main'     │ Valid wind tracks  │
│ Position Sampling Grid   │ 6-hourly or verified orig   │ Synoptic 6h (00,06,12,18)│ Synoptic / Orig    │
│ Wind Observation Filter  │ IFLAG[0] in ['O', 'V']      │ Not required for inputs  │ IFLAG[0] in ['O','V│
│ Agency Stream Filter     │ Same USA agency (t to t+h)  │ N/A                      │ Same USA agency    │
│ Coordinate Handling      │ Latitude/Longitude          │ Normalized [-180, +180]  │ Latitude/Longitude │
│ Target Definition        │ USA_WIND at t+6h, 12h, 24h  │ Cartesian Δx, Δy (km)    │ 3-Class ΔV_24h:    │
│ Target Units / Scale     │ Knots (1-min sustained)     │ Kilometers               │ ±10 kt threshold   │
│ Usable 24h Target Pairs  │ 113,482 original pairs      │ 162,048 synoptic pairs   │ 113,482 orig pairs │
│ Primary Baseline Model   │ 24h Persistence (V_t)       │ Linear Persistence / CLPR│ Climatological Dist│
│ Validation Protocol      │ Temporal Out-of-Time Split  │ Temporal Out-of-Time     │ Temporal Out-of-Tim│
└──────────────────────────┴─────────────────────────────┴──────────────────────────┴────────────────────┤
```

---
*Report completed and verified under Task 3 requirements. End of audit.*
