# IBTrACS Global Tropical Cyclone Dataset Inspection Report

> **Dataset Version:** IBTrACS v04r01 (ALL)
> **Inspection Date:** 2026-11-02 19:40:21 UTC
> **Source File:** `raw/ibtracs.ALL.list.v04r01.csv`
> **Total Observations:** 730,416 rows | **Total Features:** 174 columns
> **Temporal Coverage:** 1842-10-25 03:00:00 to 2026-11-01 00:00:00 (Seasons 1842 - 2027)

---

## Executive Summary

This report presents a comprehensive quality, statistical, and machine-learning feasibility inspection of the **NOAA International Best Track Archive for Climate Stewardship (IBTrACS) v04r01** global dataset.

- **Scale & Completeness:** The dataset comprises **730,416 storm fixes** across **13,631 unique tropical cyclones** spanning 184 years (1842 to 2026/2027).
- **Geographic Coverage:** Full global coverage across seven primary ocean basins: Western Pacific (`WP`), South Indian (`SI`), North Atlantic (`NA`), South Pacific (`SP`), Eastern Pacific (`EP`), North Indian (`NI`), and South Atlantic (`SA`). Core tracking coordinates (`LAT`, `LON`) are **100% complete** with zero missing values.
- **Data Architecture:** IBTrACS v04 uses 3-hourly interpolated tracks. **93.40%** of consecutive fixes occur at exactly 3-hour intervals, with zero duplicate `(SID, ISO_TIME)` pairs.
- **Critical Implementation Finding:** The second row of the raw CSV contains column unit definitions and must be skipped. Furthermore, default CSV parsers that treat string `'NA'` as `NaN` unintentionally corrupt the North Atlantic basin and subbasin codes. Explicit NA handling (`keep_default_na=False`) is required.
- **Meteorological Availability:** While position data is complete, wind speed and minimum central pressure completeness varies significantly across historical eras and regional agencies (`USA_WIND` is 55.57% complete, while `WMO_WIND` is 23.32% complete). Modern multi-sensor data (post-1980 / post-2000) provides the highest fidelity for machine learning.

---

## A. Dataset Overview

### 1. Dimensionality & Format

- **Total Records (Rows):** 730,416
- **Total Attributes (Columns):** 174
- **Temporal Span:** `1842-10-25 03:00:00` to `2026-11-01 00:00:00`
- **Season Span:** 1842 to 2027 *(Note: Southern Hemisphere seasons extend into early the following calendar year, hence season 2027 covers late 2026 storms such as TC WATI)*.

### 2. Data Types Distribution

| Data Type | Column Count | Description |
| :--- | :--- | :--- |
| `float64` | 151 | Attribute fields conforming to float64 |
| `str` | 19 | Attribute fields conforming to str |
| `int64` | 4 | Attribute fields conforming to int64 |

### 3. Column Categorization & Schema Architecture

The 174 columns are logically partitioned into seven operational categories:

| Category | Example Columns | Operational Description |
| :--- | :--- | :--- |
| **Storm Identifiers & Metadata** | `SID`, `SEASON`, `NUMBER`, `BASIN`, `SUBBASIN`, `NAME`, `ISO_TIME`, `NATURE`, `TRACK_TYPE` | Primary key identification, genesis season, ocean basin, storm name, synoptic timestamp, and lifecycle classification. |
| **Unified Spatial & Kinematic** | `LAT`, `LON`, `DIST2LAND`, `LANDFALL`, `STORM_SPEED`, `STORM_DIR`, `IFLAG` | Official IBTrACS interpolated storm center positions, translation speed, bearing, distance to nearest landmass, and interpolation quality flags. |
| **WMO Consensus Intensity** | `WMO_WIND`, `WMO_PRES`, `WMO_AGENCY` | World Meteorological Organization official designated agency consensus maximum sustained wind speed (kts) and central pressure (mb). |
| **USA / JTWC / NHC Data** | `USA_AGENCY`, `USA_ATCF_ID`, `USA_LAT`, `USA_LON`, `USA_STATUS`, `USA_WIND`, `USA_PRES`, `USA_SSHS` | Automated Tropical Cyclone Forecasting (ATCF) tracks from NHC and JTWC, including Saffir-Simpson Hurricane Scale (`USA_SSHS`). |
| **Storm Structure & Radii** | `USA_R34_NE/SE/SW/NW`, `USA_R50_*`, `USA_R64_*`, `USA_POCI`, `USA_ROCI`, `USA_RMW`, `USA_EYE` | Quadrant-specific wind radii for 34-knot, 50-knot, and 64-knot winds, radius of maximum wind (RMW), radius and pressure of outermost closed isobar (ROCI/POCI), and eye diameter. |
| **Regional Specialized Meteorological Centers (RSMCs)** | `TOKYO_*`, `CMA_*`, `HKO_*`, `KMA_*`, `NEWDELHI_*`, `REUNION_*`, `BOM_*`, `NADI_*`, `WELLINGTON_*` | Basin-specific national agency tracks (JMA Tokyo, China CMA, Hong Kong HKO, Korea KMA, IMD New Delhi, Météo-France La Réunion, Australian BoM, Fiji Nadi, MetService Wellington). |
| **Historical & Special Archives** | `DS824_*`, `TD9636_*`, `TD9635_*`, `NEUMANN_*`, `MLC_*`, `USA_GUST`, `BOM_GUST` | Pre-satellite digitizations, historical reconnaissance archives, gust estimates, and sea-height radii. |

### 4. First 10 Records (Representative Attributes)

| SID | Season | Basin | Subbasin | Name | ISO Time | Nature | Lat (°N) | Lon (°E) | WMO Wind | WMO Pres | Track Type | Dist2Land (km) | Storm Speed (kts) | Storm Dir (°) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1842298N11080` | 1842 | NI | BB | UNNAMED | `1842-10-25 03:00:00` | NR | 10.9 | 80.3 | — | — | main | 43 | 9 | 265 |
| `1842298N11080` | 1842 | NI | BB | UNNAMED | `1842-10-25 06:00:00` | NR | 10.9 | 79.8 | — | — | main | 0 | 9 | 265 |
| `1842298N11080` | 1842 | NI | BB | UNNAMED | `1842-10-25 09:00:00` | NR | 10.8 | 79.4 | — | — | main | 0 | 9 | 265 |
| `1842298N11080` | 1842 | NI | BB | UNNAMED | `1842-10-25 12:00:00` | NR | 10.8 | 78.9 | — | — | main | 0 | 9 | 265 |
| `1842298N11080` | 1842 | NI | BB | UNNAMED | `1842-10-25 15:00:00` | NR | 10.8 | 78.4 | — | — | main | 0 | 9 | 270 |
| `1842298N11080` | 1842 | NI | AS | UNNAMED | `1842-10-25 18:00:00` | NR | 10.8 | 77.9 | — | — | main | 0 | 10 | 270 |
| `1842298N11080` | 1842 | NI | AS | UNNAMED | `1842-10-25 21:00:00` | NR | 10.8 | 77.4 | — | — | main | 0 | 10 | 270 |
| `1842298N11080` | 1842 | NI | AS | UNNAMED | `1842-10-26 00:00:00` | NR | 10.8 | 76.9 | — | — | main | 0 | 10 | 270 |
| `1842298N11080` | 1842 | NI | AS | UNNAMED | `1842-10-26 03:00:00` | NR | 10.8 | 76.4 | — | — | main | 0 | 11 | 270 |
| `1842298N11080` | 1842 | NI | AS | UNNAMED | `1842-10-26 06:00:00` | NR | 10.8 | 75.8 | — | — | main | 10 | 11 | 270 |

---

## B. Cyclone Statistics

### 1. Cyclone Identifiers & Naming Distribution

- **Total Unique Cyclones (SIDs):** **13,631**
- **Total Unique Name Strings:** **1,913**
- **Officially Named Storms:** **5,763** (42.3% of all storms)
- **Unnamed / Historical Storms:** **7,868** (57.7% of all storms)

> [!NOTE]
> Formal tropical cyclone naming began in the Western Pacific and North Atlantic during the mid-20th century (c. 1945–1953). Consequently, pre-1950 storms and smaller southern hemisphere systems are cataloged as `UNNAMED`.

### 2. Geographic Basin Breakdown

IBTrACS tracks cyclones across the globe. Because tropical cyclones can traverse ocean basin boundaries (e.g., Eastern Pacific to Western Pacific, or South Indian to South Pacific), both **Genesis Basin** (genesis point) and **Basins Touched** (all basins traversed) are presented below:

| Ocean Basin Code | Basin Name | Genesis Cyclones | Cyclones Touched | Total Fixes (Observations) | % of Global Fixes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`WP`** | Western North Pacific | 4,191 | 4,257 | 243,859 | 33.39% |
| **`SI`** | South Indian Ocean | 2,778 | 2,889 | 165,114 | 22.61% |
| **`NA`** | North Atlantic | 2,306 | 2,312 | 127,584 | 17.47% |
| **`SP`** | South Pacific Ocean | 1,201 | 1,258 | 68,724 | 9.41% |
| **`EP`** | Eastern North Pacific | 1,382 | 1,734 | 67,136 | 9.19% |
| **`NI`** | North Indian Ocean (Arabian Sea & Bay of Bengal) | 1,770 | 1,860 | 57,880 | 7.92% |
| **`SA`** | South Atlantic Ocean | 3 | 3 | 119 | 0.02% |

### 3. Observations per Cyclone Distribution

| Metric | Observations per Cyclone | Equivalent Duration (3-hr fixes) |
| :--- | :--- | :--- |
| **Minimum** | 1 fix | 0.0 to 3.0 hours |
| **25th Percentile (Q1)** | 28.0 fixes | ~3.5 days |
| **Median (Q2)** | **46.0 fixes** | **~5.75 days** |
| **Mean** | **53.58 fixes** | **~6.7 days** |
| **75th Percentile (Q3)** | 73.0 fixes | ~9.1 days |
| **Maximum** | **350 fixes** | **~38.0 days** |
| **Standard Deviation** | 34.21 fixes | ~4.3 days |

### 4. Top 20 Cyclones with the Most Observations (Longest-Lived Storms)

| Rank | Storm ID (`SID`) | Name | Season | Basin | Fix Count | Start Date | End Date | Duration (Days) | Peak Wind (kts) | Min Pres (mb) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `1997013S08101` | **HELINDA:PANCHO** | 1997 | `SI` | **350** | `1997-01-13` | `1997-02-14` | 32.25 | 125 | 915 |
| 2 | `1972005S10152` | **CARLOTTA** | 1972 | `SP` | **308** | `1972-01-04` | `1972-01-26` | 21.25 | 65 | 945 |
| 3 | `2023036S12117` | **FREDDY** | 2023 | `SI` | **305** | `2023-02-04` | `2023-03-14` | 38.0 | 140 | 923 |
| 4 | `1899215N12329` | **UNNAMED** | 1899 | `NA` | **264** | `1899-08-03` | `1899-09-04` | 32.75 | 130 | 930 |
| 5 | `1979042S05170` | **KERRY** | 1979 | `SP` | **262** | `1979-02-11` | `1979-03-05` | 22.5 | 105 | 945 |
| 6 | `2000032S11116` | **ELINE:LEONE** | 2000 | `SI` | **259** | `2000-02-01` | `2000-02-29` | 28.33 | 115 | 930 |
| 7 | `1973060S08161` | **MADGE** | 1973 | `SP` | **257** | `1973-02-28` | `1973-03-18` | 17.33 | 80 | 952 |
| 8 | `1998002S14151` | **KATRINA** | 1998 | `SP` | **255** | `1998-01-01` | `1998-02-02` | 31.75 | 90 | 940 |
| 9 | `1994222N11267` | **JOHN** | 1994 | `EP` | **249** | `1994-08-10` | `1994-09-10` | 31.0 | 150 | 929 |
| 10 | `2026254N11252` | **NOLO** | 2026 | `EP` | **249** | `2026-09-10` | `2026-10-11` | 31.0 | 135 | 929 |
| 11 | `1971249N26289` | **GINGER** | 1971 | `NA` | **244** | `1971-09-06` | `1971-10-06` | 30.25 | 100 | 959 |
| 12 | `1996357S10136` | **PHIL** | 1997 | `SP` | **240** | `1996-12-22` | `1997-01-16` | 25.25 | 85 | 975 |
| 13 | `1955056S14175` | **UNNAMED** | 1955 | `SP` | **236** | `1955-02-24` | `1955-03-18` | 22.46 | — | 965 |
| 14 | `1993360S04171` | **REWA** | 1994 | `SP` | **231** | `1993-12-26` | `1994-01-23` | 28.5 | 125 | 920 |
| 15 | `1990309N08167` | **PAGE** | 1990 | `WP` | **227** | `1990-11-05` | `1990-12-03` | 28.25 | 140 | 910 |
| 16 | `1972102S11140` | **FAITH** | 1972 | `SP` | **220** | `1972-04-10` | `1972-04-24` | 13.75 | 35 | 990 |
| 17 | `2008033S11083` | **HONDO** | 2008 | `SI` | **217** | `2008-02-02` | `2008-02-29` | 27.0 | 130 | 926 |
| 18 | `2015229N08212` | **KILO** | 2015 | `EP` | **215** | `2015-08-17` | `2015-09-13` | 26.75 | 120 | 940 |
| 19 | `1997062S16158` | **JUSTIN** | 1997 | `SP` | **213** | `1997-03-03` | `1997-03-29` | 26.25 | 90 | 955 |
| 20 | `1993221N12216` | **KEONI** | 1993 | `EP` | **211** | `1993-08-09` | `1993-09-04` | 26.25 | 115 | 940 |

### 5. Storm Nature Distribution

| Code | Nature Classification | Fix Count | Percentage |
| :--- | :--- | :--- | :--- |
| `TS` | Tropical Storm / Cyclone | 554,183 | 75.87% |
| `NR` | Not Reported / Historical | 85,319 | 11.68% |
| `ET` | Extratropical Transition / Cyclone | 30,638 | 4.19% |
| `DS` | Disturbance / Tropical Depression | 29,700 | 4.07% |
| `MX` | Mixed / Transitioning Phase | 27,201 | 3.72% |
| `SS` | Subtropical Storm | 3,375 | 0.46% |

---

## C. Missing Value Analysis

### 1. Essential Attributes Completeness

| Feature Name | Data Type | Present Count | Missing Count | Missing % | Completeness Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SID` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `SEASON` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `NUMBER` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `BASIN` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `SUBBASIN` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `NAME` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `ISO_TIME` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `NATURE` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `LAT` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `LON` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `TRACK_TYPE` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `DIST2LAND` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `LANDFALL` | — | 716,785 | 13,631 | 1.87% | Excellent (<5%) |
| `IFLAG` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `STORM_SPEED` | — | 730,336 | 80 | 0.01% | Excellent (<5%) |
| `STORM_DIR` | — | 730,336 | 80 | 0.01% | Excellent (<5%) |
| `WMO_WIND` | — | 170,343 | 560,073 | 76.68% | Sparseness (>50%) |
| `WMO_PRES` | — | 174,031 | 556,385 | 76.17% | Sparseness (>50%) |
| `WMO_AGENCY` | — | 232,326 | 498,090 | 68.19% | Sparseness (>50%) |
| `USA_AGENCY` | — | 234,153 | 496,263 | 67.94% | Sparseness (>50%) |
| `USA_ATCF_ID` | — | 472,502 | 257,914 | 35.31% | Moderate (5-50%) |
| `USA_LAT` | — | 468,650 | 261,766 | 35.84% | Moderate (5-50%) |
| `USA_LON` | — | 468,650 | 261,766 | 35.84% | Moderate (5-50%) |
| `USA_STATUS` | — | 263,489 | 466,927 | 63.93% | Sparseness (>50%) |
| `USA_WIND` | — | 405,899 | 324,517 | 44.43% | Moderate (5-50%) |
| `USA_PRES` | — | 170,938 | 559,478 | 76.60% | Sparseness (>50%) |
| `USA_SSHS` | — | 730,416 | 0 | 0.00% | Complete (100%) |
| `USA_R34_NE` | — | 76,701 | 653,715 | 89.50% | Sparseness (>50%) |
| `USA_R50_NE` | — | 39,109 | 691,307 | 94.65% | Sparseness (>50%) |
| `USA_R64_NE` | — | 24,274 | 706,142 | 96.68% | Sparseness (>50%) |
| `USA_POCI` | — | 116,998 | 613,418 | 83.98% | Sparseness (>50%) |
| `USA_ROCI` | — | 116,575 | 613,841 | 84.04% | Sparseness (>50%) |
| `USA_RMW` | — | 116,606 | 613,810 | 84.04% | Sparseness (>50%) |
| `USA_EYE` | — | 4,123 | 726,293 | 99.44% | Sparseness (>50%) |
| `TOKYO_WIND` | — | 51,285 | 679,131 | 92.98% | Sparseness (>50%) |
| `TOKYO_PRES` | — | 135,074 | 595,342 | 81.51% | Sparseness (>50%) |
| `CMA_WIND` | — | 133,126 | 597,290 | 81.77% | Sparseness (>50%) |
| `CMA_PRES` | — | 144,910 | 585,506 | 80.16% | Sparseness (>50%) |
| `NEWDELHI_WIND` | — | 9,467 | 720,949 | 98.70% | Sparseness (>50%) |
| `NEWDELHI_PRES` | — | 8,605 | 721,811 | 98.82% | Sparseness (>50%) |
| `REUNION_WIND` | — | 34,288 | 696,128 | 95.31% | Sparseness (>50%) |
| `REUNION_PRES` | — | 33,626 | 696,790 | 95.40% | Sparseness (>50%) |
| `BOM_WIND` | — | 42,162 | 688,254 | 94.23% | Sparseness (>50%) |
| `BOM_PRES` | — | 55,193 | 675,223 | 92.44% | Sparseness (>50%) |

### 2. Detailed Spatial & Meteorological Completeness Matrix

| Variable Group | Attribute | Present Count | Missing Count | Missing % | Operational Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Coordinates** | `LAT` | 730,416 | 0 | 0.000% | 100% complete across all 730k fixes. |
| **Core Coordinates** | `LON` | 730,416 | 0 | 0.000% | 100% complete across all 730k fixes. |
| **WMO Consensus** | `WMO_WIND` | 170,343 | 560,073 | 76.68% | Only populated when official WMO agency consensus is available. |
| **WMO Consensus** | `WMO_PRES` | 174,031 | 556,385 | 76.17% | Central pressure not consistently archived by all regional WMO centers. |
| **USA (NHC/JTWC)** | `USA_WIND` | 405,899 | 324,517 | 44.43% | Populated for all ATCF tracked basins (NA, EP, WP, NI, SH). |
| **USA (NHC/JTWC)** | `USA_PRES` | 170,938 | 559,478 | 76.60% | Central pressure reported primarily in Atlantic, East Pacific, and modern recon. |
| **Combined Union** | `ANY_WIND` | 422,582 | 307,834 | 42.15% | Either `USA_WIND` or `WMO_WIND` populated (57.85% coverage). |
| **Combined Union** | `ANY_PRES` | 262,491 | 467,925 | 64.06% | Either `USA_PRES` or `WMO_PRES` populated (35.94% coverage). |

### 3. Agency Sparseness Analysis

National agency columns (e.g. `TOKYO_*`, `CMA_*`, `NEWDELHI_*`, `REUNION_*`, `BOM_*`) exhibit high overall missing percentages (80% to 98% missing in the global file) **by design**:
- Regional specialized agencies only track storms within their designated area of responsibility (e.g., `TOKYO` tracks Western North Pacific storms; `REUNION` tracks South Indian storms).
- Therefore, regional agency columns must be evaluated **per basin**, not globally.

---

## D. Meteorological Analysis

### 1. Summary Statistics for Key Meteorological & Physical Fields

| Variable | Count | Min | 25% (Q1) | Median | Mean | 75% (Q3) | 95% | 99% | Max | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `WMO_WIND_kts` | 170,343 | 3.0 | 30.0 | **45.0** | **49.8** | 65.0 | 95.0 | 120.0 | 185.0 | 23.82 |
| `WMO_PRES_mb` | 174,031 | 870.0 | 980.0 | **995.0** | **988.21** | 1002.0 | 1008.0 | 1012.0 | 1024.0 | 20.19 |
| `USA_WIND_kts` | 405,899 | 5.0 | 30.0 | **43.0** | **50.24** | 65.0 | 108.0 | 133.0 | 185.0 | 27.59 |
| `USA_PRES_mb` | 170,938 | 872.0 | 983.0 | **998.0** | **990.25** | 1005.0 | 1010.0 | 1013.0 | 1024.0 | 20.7 |
| `STORM_SPEED_kts` | 730,336 | 0.0 | 5.0 | **9.0** | **9.86** | 12.0 | 22.0 | 35.0 | 148.0 | 6.66 |
| `STORM_DIR_deg` | 730,336 | 0.0 | 120.0 | **255.0** | **209.54** | 290.0 | 335.0 | 355.0 | 360.0 | 106.16 |
| `LAT_deg` | 730,416 | -68.5 | -13.0 | **15.0** | **8.82** | 22.5 | 37.6 | 49.4 | 83.0 | 20.49 |
| `LON_deg` | 730,416 | -179.8 | -38.6 | **90.5** | **59.85** | 132.7 | 165.4 | 183.6 | 266.9 | 96.05 |
| `DIST2LAND_km` | 730,416 | 0.0 | 164.0 | **484.0** | **703.87** | 1129.0 | 2004.0 | 2491.0 | 4843.0 | 665.55 |

### 2. Coordinate System Nuances & Longitude Range

- **Latitude Span:** `-68.5°` to `83.0°` (covering genesis in deep tropics ~5°N/S to extratropical decay in polar waters).
- **Longitude Span:** `-179.8°` to `266.9°`.
- **Observations with `LON > 180°`:** **10,658 fixes (1.46%)**.

> [!WARNING]
> In the South Pacific (`SP`) and Eastern Pacific (`EP`), historical best tracks occasionally record longitude using the 0° to 360° convention (e.g. 185°E = 175°W). Any spatial machine learning model or GIS pipeline must unwrap or map longitudes strictly into $[-180°, +180°]$ to avoid 360-degree boundary discontinuities.

### 3. Historical Eras & Observation Evolution

Tropical cyclone observation density has evolved drastically across five distinct technological epochs:

| Observation Epoch | Calendar Years | Total Fixes | % of Dataset | Observational Character |
| :--- | :--- | :--- | :--- | :--- |
| **Pre-1900 (Early Historical / Ship Log Era)** | — | 77,556 | 10.62% | Best tracks reconstructed from ship logs, coastal stations, early aircraft, or satellite sensors. |
| **1900-1944 (Early 20th Century / Surface Networks)** | — | 129,859 | 17.78% | Best tracks reconstructed from ship logs, coastal stations, early aircraft, or satellite sensors. |
| **1945-1969 (Aircraft Reconnaissance Era)** | — | 136,789 | 18.73% | Best tracks reconstructed from ship logs, coastal stations, early aircraft, or satellite sensors. |
| **1970-1999 (Early Satellite Era / Dvorak Technique)** | — | 216,416 | 29.63% | Best tracks reconstructed from ship logs, coastal stations, early aircraft, or satellite sensors. |
| **2000-Present (Modern Multi-Sensor Satellite Era)** | — | 169,796 | 23.25% | Best tracks reconstructed from ship logs, coastal stations, early aircraft, or satellite sensors. |

### 4. Peak Activity Years

- **Top 5 Seasons by Observation Count:**
  - **1997:** 9,360 observations
  - **1971:** 8,676 observations
  - **1972:** 8,601 observations
  - **1996:** 8,496 observations
  - **1992:** 8,449 observations
- **Top 5 Seasons by Cyclone Count:**
  - **1970:** 143 unique cyclones

  - **1971:** 143 unique cyclones

  - **1975:** 140 unique cyclones

  - **1974:** 137 unique cyclones

  - **1967:** 135 unique cyclones

---

## E. Temporal Analysis

### 1. Observation Interval Distribution (Time Delta $\Delta t$)

IBTrACS v04 provides regularized, linearly interpolated positions to support climate and statistical analyses.

| Interval Duration | Fix Count | Percentage | Operational Context |
| :--- | :--- | :--- | :--- |
| **Exactly 3.0 Hours** | **669,500** | **93.40%** | Standard IBTrACS synoptic / interpolated product time step. |
| **Exactly 1.0 Hour** | **23,301** | **3.25%** | Special fixes: aircraft reconnaissance fixes, rapid changes, or landfall points. |
| **Exactly 2.0 Hours** | **22,460** | **3.13%** | Intermediate observational adjustments. |
| **Sub-hourly (< 1 Hour)** | **836** | **0.12%** | Exact landfall timestamp records (e.g. 15-minute or 1-minute resolution). |
| **Exactly 6.0 Hours** | **0** | **0.00%** | Gaps interpolated to 3h by IBTrACS v04 algorithm. |
| **Other Irregular** | **1,524** | **0.21%** | Off-hour observations. |
| **Zero Interval ($\Delta t = 0$)** | **0** | **0.00%** | Zero duplicate timestamps for any cyclone. |
| **Negative Interval ($\Delta t < 0$)** | **0** | **0.00%** | Timestamps are strictly monotonic in chronological order. |

### 2. Diurnal Cycle & Synoptic Hours Distribution

- **Standard 6-Hourly Synoptic Fixes (00, 06, 12, 18 UTC):** **357,610 fixes (48.96%)**
- **Intermediate 3-Hourly Synoptic Fixes (03, 09, 15, 21 UTC):** **347,526 fixes (47.58%)**
- **Off-Synoptic Special Fixes (Landfalls / Recon):** **25,280 fixes (3.46%)**

### 3. Duplicate Identification (`SID` + `ISO_TIME`)

- **Total Duplicate Records:** **0**
- **Unique Duplicate Pairs:** **0**
- **Integrity Result:** The composite primary key `(SID, ISO_TIME)` is **strictly unique** across all 730,416 rows in the dataset.

---

## F. Machine Learning Model Feasibility

We evaluate the feasibility of developing three predictive models using solely or primarily the IBTrACS v04r01 dataset:

### Model 1: Cyclone Intensity Prediction (Regression / Rapid Intensification)

**Objective:** Predict future maximum sustained wind speed ($V_{t+\Delta t}$) or classify Rapid Intensification (RI: $\Delta V \ge 30\text{ kts}$ in 24 hours).

- **Feasibility Assessment:** **High Feasibility (Modern Era $\ge 1980$)**.
- **Suitable Input Columns:**
  - Historical/Current Wind Speed: `USA_WIND`, `WMO_WIND`, regional agency winds (`TOKYO_WIND`, etc.).
  - Central Pressure: `USA_PRES`, `WMO_PRES`.
  - Spatial & Geographic Context: `LAT`, `LON`, `DIST2LAND`, `LANDFALL` (cyclones weaken drastically over land).
  - Kinematic Motion: `STORM_SPEED`, `STORM_DIR`.
  - Storm Structure (post-2000): `USA_R34_NE/SE/SW/NW`, `USA_R50_*`, `USA_R64_*`, `USA_RMW`, `USA_POCI`, `USA_ROCI`.
  - Seasonal / Climatological Temporal: `SEASON`, day of year, elapsed duration from genesis.
- **Potential Prediction Targets:**
  - Continuous: $V_{t+6h}, V_{t+12h}, V_{t+24h}$ or $\Delta V_{24h}$.
  - Categorical: Rapid Intensification (RI binary flag), Saffir-Simpson Hurricane Scale classification (`USA_SSHS`).
- **Missing Data Limitations:**
  - `WMO_WIND` is missing in 76.68% of fixes; `USA_WIND` is missing in 44.43% (principally pre-1970 and older Indian/South Pacific basins).
  - Wind radii columns (`USA_R34`, etc.) have ~85-95% missing values globally because systematic automated radii recording began only around 2001.
- **Required Preprocessing:**
  1. **Temporal Filtering:** Restrict training to the satellite era (e.g. $\ge 1980$ or $\ge 2000$ if structural radii are utilized).
  2. **Agency Consensus Synthesis:** Impute/synthesize a single unified intensity target by prioritizing `USA_WIND` $\to$ `WMO_WIND` $\to$ regional RSMC winds.
  3. **Temporal Regularization:** Filter out off-synoptic landfall fixes to produce a uniform 3-hourly or 6-hourly time series.
  4. **Feature Engineering:** Calculate intensity tendency ($\Delta V_{-6h}, \Delta V_{-12h}$), acceleration, pressure deficit ($P_{oci} - P_{min}$), and land proximity decay flags.

### Model 2: Cyclone Movement & Trajectory Prediction (Track Forecasting)

**Objective:** Forecast cyclone future spatial coordinates $(\text{LAT}_{t+\Delta t}, \text{LON}_{t+\Delta t})$ at lead times of 6h, 12h, 24h, and 48h.

- **Feasibility Assessment:** **High Feasibility for Short-Term (6h–24h) Extrapolation / Autoregression; Moderate Feasibility for Extended Lead Times (>24h)** without external environmental steering grids.
- **Suitable Input Columns:**
  - Historical Trajectory: `LAT`, `LON`, lag coordinates ($t-3h, t-6h, t-12h, t-24h$).
  - Kinematics: `STORM_SPEED`, `STORM_DIR`, forward velocity components ($u, v$).
  - Intensity Coupling: `USA_WIND`, `USA_PRES` (deeper, more intense cyclones are steered by higher atmospheric layers).
  - Geographic Domain: `BASIN`, `SUBBASIN`, `DIST2LAND`.
- **Potential Prediction Targets:**
  - Direct coordinates: $\text{LAT}_{t+\Delta t}, \text{LON}_{t+\Delta t}$.
  - Displacement vectors: $\Delta x, \Delta y$ (in kilometers or spherical degrees).
- **Missing Data Limitations:**
  - `LAT` and `LON` have **0.00% missing data** (100% complete across all 730k rows). `STORM_SPEED` and `STORM_DIR` are 99.99% complete.
  - *Limitation:* IBTrACS contains purely storm-centric observations. It does not contain background atmospheric steering fields (e.g. 500 hPa geopotential height, 850–200 hPa vertical wind shear, steering flow ridges). Pure IBTrACS track models behave analogously to statistical-climatological models (e.g. CLIPER) or LSTM track autoencoders.
- **Required Preprocessing:**
  1. **Antimeridian Longitude Normalization:** Correct all $LON > 180°$ entries into standard $[-180°, +180°]$ coordinates, and implement track unwrapping across the 180° meridian.
  2. **Track Filtering:** Restrict sequence training to `TRACK_TYPE == 'main'` to eliminate secondary or merging spurs.
  3. **Displacement Conversion:** Project spherical coordinates into Cartesian distances (km) via Haversine / geodesic formulas.

### Model 3: Cyclone Development Pattern Classification (Lifecycle & Transition)

**Objective:** Classify storm lifecycle stages (e.g. Tropical `TS`, Extratropical Transition `ET`, Subtropical `SS`, Decay) and identify canonical track archetypes (e.g. straight-moving, recurving, looping).

- **Feasibility Assessment:** **High Feasibility**.
- **Suitable Input Columns:**
  - Lifecycle Stage & Nature: `NATURE` (`TS`, `ET`, `DS`, `MX`, `SS`, `NR`), `TRACK_TYPE`.
  - Intensity Stage: `USA_STATUS` (`TD`, `TS`, `HU`, `EX`, `LO`, `DB`), `USA_SSHS`.
  - Trajectory & Dynamics: `LAT`, `LON`, `STORM_SPEED`, `STORM_DIR`, poleward speed component ($v_y$).
  - Structural Asymmetry: Wind radii quadrants (`USA_R34_NE`, `SE`, `SW`, `NW`) — expansion of poleward quadrants is a hallmark of Extratropical Transition.
- **Potential Prediction Targets:**
  - Multiclass classification of storm state `NATURE` at $t+\Delta t$.
  - Binary Extratropical Transition (ET) onset detection (`TS` $\to$ `ET`).
  - Unsupervised clustering of global track patterns into canonical regional archetypes.
- **Missing Data Limitations:**
  - `NATURE` is 100% complete, but historical pre-1970 records predominantly label storms with generic `NR` or `TS`.
  - `USA_STATUS` is missing in 63.93% of rows (present mostly in US agency basins).
- **Required Preprocessing:**
  1. **Temporal Horizon:** Restrict classification models to $\ge 1970$ (satellite era) where physical transitions were accurately identified.
  2. **Event Encoding:** Construct ground truth transition labels identifying the exact timestamp when a cyclone transitions from warm-core (`TS`) to cold-core (`ET`).
  3. **Kinematic Feature Engineering:** Derive poleward acceleration, track curvature, translation speed changes, and quadrant wind radii asymmetry ratios.

---

## G. Recommendations for Downstream Phases

1. **Strict CSV Parsing Parameters:** Always invoke `skiprows=[1]`, `na_values=['', ' ']`, and `keep_default_na=False` when loading IBTrACS CSVs.
2. **Coordinate Normalization Pipeline:** Enforce automatic conversion for any longitude exceeding 180° prior to spatial feature extraction or plotting.
3. **Era-Appropriate Cohort Selection:** For machine learning training:
   - **Track Modeling:** Use 1970–Present (high position accuracy from geostationary satellites).
   - **Intensity Modeling:** Use 1980–Present (standardized Dvorak technique and reconnaissance).
   - **Structure / Wind Radii Modeling:** Use 2001–Present (automated scatterometer and multi-platform radii).
4. **Consensus Target Creation:** Synthesize target columns (`INTENSITY_WIND`, `CENTRAL_PRES`) from multiple reporting agencies with documented basin-level priority.

---
*Report generated automatically by `scripts/inspect_dataset.py`.*
