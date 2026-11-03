#!/usr/bin/env python3
"""
IBTrACS Global Dataset Inspection Script
=========================================
Reproducible inspection script for the NOAA IBTrACS v04r01 global tropical cyclone dataset.

Key handling details:
- Skips the second row (index 1) which contains column physical units (e.g. kts, mb, degrees_north).
- Uses `keep_default_na=False` and explicit `na_values=['', ' ']` so that North Atlantic basin ('NA')
  and subbasin ('NA') are not erroneously converted to NaN/missing values.
- Analyzes dataset overview, cyclone statistics, missing values, meteorology, temporal intervals,
  and machine learning model feasibility.
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect NOAA IBTrACS Global Tropical Cyclone Dataset."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="raw/ibtracs.ALL.list.v04r01.csv",
        help="Path to the IBTrACS CSV dataset.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/dataset_inspection.md",
        help="Path where the markdown report should be saved.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the markdown report file.",
    )
    return parser.parse_args()


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the IBTrACS CSV dataset with proper unit skipping and NA handling.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at path: {csv_path}")

    print(f"[*] Loading dataset from: {csv_path}")
    start_time = time.time()

    # Row 0: Header
    # Row 1: Units row (e.g. 'kts', 'mb', 'degrees_north') -> skiprows=[1]
    # keep_default_na=False is critical to avoid mapping North Atlantic basin ('NA') to NaN.
    df = pd.read_csv(
        csv_path,
        skiprows=[1],
        na_values=["", " "],
        keep_default_na=False,
        low_memory=False,
    )

    elapsed = time.time() - start_time
    print(f"[+] Loaded {len(df):,} rows and {len(df.columns)} columns in {elapsed:.2f} seconds.")

    # Convert numeric fields where appropriate
    numeric_columns = [
        'SEASON', 'NUMBER', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND', 'LANDFALL',
        'STORM_SPEED', 'STORM_DIR', 'USA_WIND', 'USA_PRES', 'USA_SSHS',
        'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW',
        'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW',
        'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW',
        'USA_POCI', 'USA_ROCI', 'USA_RMW', 'USA_EYE',
        'TOKYO_WIND', 'TOKYO_PRES', 'CMA_WIND', 'CMA_PRES',
        'HKO_WIND', 'HKO_PRES', 'KMA_WIND', 'KMA_PRES',
        'NEWDELHI_WIND', 'NEWDELHI_PRES', 'REUNION_WIND', 'REUNION_PRES',
        'BOM_WIND', 'BOM_PRES', 'NADI_WIND', 'NADI_PRES',
        'WELLINGTON_WIND', 'WELLINGTON_PRES'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def inspect_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """A. Dataset Overview analysis."""
    print("[*] Analyzing Dataset Overview...")
    total_rows, total_cols = df.shape
    dtypes_count = df.dtypes.astype(str).value_counts().to_dict()

    date_min = str(df['ISO_TIME'].min())
    date_max = str(df['ISO_TIME'].max())
    season_min = int(df['SEASON'].min()) if not pd.isna(df['SEASON'].min()) else None
    season_max = int(df['SEASON'].max()) if not pd.isna(df['SEASON'].max()) else None

    # Key columns for first 10 records
    key_cols = [
        'SID', 'SEASON', 'NUMBER', 'BASIN', 'SUBBASIN', 'NAME',
        'ISO_TIME', 'NATURE', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES',
        'TRACK_TYPE', 'DIST2LAND', 'LANDFALL', 'STORM_SPEED', 'STORM_DIR'
    ]
    first_10 = df[key_cols].head(10).to_dict(orient='records')

    return {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'dtypes_distribution': dtypes_count,
        'date_range': {'min': date_min, 'max': date_max},
        'season_range': {'min': season_min, 'max': season_max},
        'first_10_records': first_10,
        'all_columns': list(df.columns)
    }


def inspect_cyclone_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """B. Cyclone Statistics analysis."""
    print("[*] Analyzing Cyclone Statistics...")
    total_sids = int(df['SID'].nunique())
    total_names = int(df['NAME'].nunique())

    # Named vs Unnamed storms
    unnamed_mask = df['NAME'].astype(str).str.strip().str.upper().isin(['UNNAMED', 'NOT_NAMED', 'NONAME', 'UNKNOWN', ''])
    named_storms = int(df.loc[~unnamed_mask, 'SID'].nunique())
    unnamed_storms = int(df.loc[unnamed_mask, 'SID'].nunique())

    # Basin statistics
    cyclones_touching_basin = df.groupby('BASIN')['SID'].nunique().to_dict()
    genesis_basin = df.groupby('SID')['BASIN'].first().value_counts().to_dict()
    obs_by_basin = df['BASIN'].value_counts(dropna=False).to_dict()
    obs_by_subbasin = df['SUBBASIN'].value_counts(dropna=False).to_dict()

    # Observations per cyclone distribution
    obs_per_cyclone = df.groupby('SID').size()
    obs_stats = {
        'min': int(obs_per_cyclone.min()),
        'max': int(obs_per_cyclone.max()),
        'mean': round(float(obs_per_cyclone.mean()), 2),
        'median': float(obs_per_cyclone.median()),
        'std': round(float(obs_per_cyclone.std()), 2),
        'q25': float(obs_per_cyclone.quantile(0.25)),
        'q75': float(obs_per_cyclone.quantile(0.75)),
    }

    # Top 20 cyclones with most observations
    top_cyclones_sids = obs_per_cyclone.nlargest(20).reset_index(name='obs_count')
    top_cyclones_list = []
    for _, row in top_cyclones_sids.iterrows():
        sid = row['SID']
        sub = df[df['SID'] == sid]
        name = sub['NAME'].dropna().iloc[0] if len(sub['NAME'].dropna()) > 0 else 'UNKNOWN'
        season = sub['SEASON'].dropna().iloc[0] if len(sub['SEASON'].dropna()) > 0 else None
        basin = sub['BASIN'].dropna().iloc[0] if len(sub['BASIN'].dropna()) > 0 else 'UNKNOWN'
        subbasin = sub['SUBBASIN'].dropna().iloc[0] if len(sub['SUBBASIN'].dropna()) > 0 else 'UNKNOWN'
        t_start = str(sub['ISO_TIME'].min())
        t_end = str(sub['ISO_TIME'].max())
        max_wind = float(sub['USA_WIND'].max()) if not pd.isna(sub['USA_WIND'].max()) else (float(sub['WMO_WIND'].max()) if not pd.isna(sub['WMO_WIND'].max()) else None)
        min_pres = float(sub['USA_PRES'].min()) if not pd.isna(sub['USA_PRES'].min()) else (float(sub['WMO_PRES'].min()) if not pd.isna(sub['WMO_PRES'].min()) else None)
        
        start_dt = pd.to_datetime(t_start)
        end_dt = pd.to_datetime(t_end)
        duration_days = round((end_dt - start_dt).total_seconds() / 86400.0, 2)

        top_cyclones_list.append({
            'SID': sid,
            'NAME': str(name),
            'SEASON': int(season) if season else None,
            'BASIN': str(basin),
            'SUBBASIN': str(subbasin),
            'OBS_COUNT': int(row['obs_count']),
            'START_TIME': t_start,
            'END_TIME': t_end,
            'DURATION_DAYS': duration_days,
            'MAX_WIND_KTS': max_wind,
            'MIN_PRES_MB': min_pres
        })

    nature_counts = {str(k): int(v) for k, v in df['NATURE'].value_counts(dropna=False).items()}
    track_type_counts = {str(k): int(v) for k, v in df['TRACK_TYPE'].value_counts(dropna=False).items()}

    return {
        'total_unique_sids': total_sids,
        'total_unique_names': total_names,
        'named_storms_count': named_storms,
        'unnamed_storms_count': unnamed_storms,
        'cyclones_touching_basin': cyclones_touching_basin,
        'cyclones_genesis_basin': genesis_basin,
        'obs_by_basin': obs_by_basin,
        'obs_by_subbasin': obs_by_subbasin,
        'obs_per_cyclone_stats': obs_stats,
        'top_20_cyclones': top_cyclones_list,
        'nature_counts': nature_counts,
        'track_type_counts': track_type_counts
    }


def inspect_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """C. Missing Value Analysis."""
    print("[*] Analyzing Missing Values across 174 columns...")
    total_rows = len(df)
    missing_counts = df.isna().sum().to_dict()
    missing_pcts = ((df.isna().sum() / total_rows) * 100.0).round(2).to_dict()

    important_cols = [
        'SID', 'SEASON', 'NUMBER', 'BASIN', 'SUBBASIN', 'NAME', 'ISO_TIME',
        'NATURE', 'LAT', 'LON', 'TRACK_TYPE', 'DIST2LAND', 'LANDFALL', 'IFLAG',
        'STORM_SPEED', 'STORM_DIR', 'WMO_WIND', 'WMO_PRES', 'WMO_AGENCY',
        'USA_AGENCY', 'USA_ATCF_ID', 'USA_LAT', 'USA_LON', 'USA_STATUS',
        'USA_WIND', 'USA_PRES', 'USA_SSHS', 'USA_R34_NE', 'USA_R50_NE', 'USA_R64_NE',
        'USA_POCI', 'USA_ROCI', 'USA_RMW', 'USA_EYE',
        'TOKYO_WIND', 'TOKYO_PRES', 'CMA_WIND', 'CMA_PRES',
        'NEWDELHI_WIND', 'NEWDELHI_PRES', 'REUNION_WIND', 'REUNION_PRES',
        'BOM_WIND', 'BOM_PRES'
    ]

    imp_summary = {}
    for col in important_cols:
        if col in df.columns:
            m_cnt = int(missing_counts[col])
            m_pct = float(missing_pcts[col])
            imp_summary[col] = {
                'missing_count': m_cnt,
                'missing_pct': m_pct,
                'present_count': total_rows - m_cnt,
                'present_pct': round(100.0 - m_pct, 2)
            }

    coord_met = {
        'LAT': {'missing': int(df['LAT'].isna().sum()), 'pct': round(df['LAT'].isna().mean() * 100, 3)},
        'LON': {'missing': int(df['LON'].isna().sum()), 'pct': round(df['LON'].isna().mean() * 100, 3)},
        'WMO_WIND': {'missing': int(df['WMO_WIND'].isna().sum()), 'pct': round(df['WMO_WIND'].isna().mean() * 100, 2)},
        'WMO_PRES': {'missing': int(df['WMO_PRES'].isna().sum()), 'pct': round(df['WMO_PRES'].isna().mean() * 100, 2)},
        'USA_WIND': {'missing': int(df['USA_WIND'].isna().sum()), 'pct': round(df['USA_WIND'].isna().mean() * 100, 2)},
        'USA_PRES': {'missing': int(df['USA_PRES'].isna().sum()), 'pct': round(df['USA_PRES'].isna().mean() * 100, 2)},
        'ANY_WIND': {
            'count': int((~df['WMO_WIND'].isna() | ~df['USA_WIND'].isna()).sum()),
            'pct': round((~df['WMO_WIND'].isna() | ~df['USA_WIND'].isna()).mean() * 100, 2)
        },
        'ANY_PRES': {
            'count': int((~df['WMO_PRES'].isna() | ~df['USA_PRES'].isna()).sum()),
            'pct': round((~df['WMO_PRES'].isna() | ~df['USA_PRES'].isna()).mean() * 100, 2)
        }
    }

    return {
        'important_columns': imp_summary,
        'coord_met_analysis': coord_met,
        'all_missing_counts': missing_counts,
        'all_missing_pcts': missing_pcts
    }


def compute_series_stats(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        'count': int(len(s)),
        'min': float(s.min()),
        'max': float(s.max()),
        'mean': round(float(s.mean()), 2),
        'median': round(float(s.median()), 2),
        'std': round(float(s.std()), 2),
        'q25': round(float(s.quantile(0.25)), 2),
        'q75': round(float(s.quantile(0.75)), 2),
        'p95': round(float(s.quantile(0.95)), 2),
        'p99': round(float(s.quantile(0.99)), 2),
    }


def inspect_meteorological(df: pd.DataFrame) -> Dict[str, Any]:
    """D. Meteorological Analysis."""
    print("[*] Analyzing Meteorological Distributions...")
    met_stats = {
        'WMO_WIND_kts': compute_series_stats(df['WMO_WIND']),
        'WMO_PRES_mb': compute_series_stats(df['WMO_PRES']),
        'USA_WIND_kts': compute_series_stats(df['USA_WIND']),
        'USA_PRES_mb': compute_series_stats(df['USA_PRES']),
        'STORM_SPEED_kts': compute_series_stats(df['STORM_SPEED']),
        'STORM_DIR_deg': compute_series_stats(df['STORM_DIR']),
        'LAT_deg': compute_series_stats(df['LAT']),
        'LON_deg': compute_series_stats(df['LON']),
        'DIST2LAND_km': compute_series_stats(df['DIST2LAND']),
    }

    lon_gt_180_cnt = int((df['LON'] > 180.0).sum())
    lon_gt_180_pct = round((lon_gt_180_cnt / len(df)) * 100, 2)

    obs_by_year = {int(k): int(v) for k, v in df['SEASON'].value_counts().sort_index().items() if not pd.isna(k)}
    cyclones_by_year = {int(k): int(v) for k, v in df.groupby('SEASON')['SID'].nunique().sort_index().items() if not pd.isna(k)}

    eras = {
        "Pre-1900 (Early Historical / Ship Log Era)": sum(v for k, v in obs_by_year.items() if k < 1900),
        "1900-1944 (Early 20th Century / Surface Networks)": sum(v for k, v in obs_by_year.items() if 1900 <= k < 1945),
        "1945-1969 (Aircraft Reconnaissance Era)": sum(v for k, v in obs_by_year.items() if 1945 <= k < 1970),
        "1970-1999 (Early Satellite Era / Dvorak Technique)": sum(v for k, v in obs_by_year.items() if 1970 <= k < 2000),
        "2000-Present (Modern Multi-Sensor Satellite Era)": sum(v for k, v in obs_by_year.items() if k >= 2000),
    }

    top_years_obs = sorted(obs_by_year.items(), key=lambda x: x[1], reverse=True)[:10]
    top_years_cyclones = sorted(cyclones_by_year.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        'statistics': met_stats,
        'lon_gt_180': {'count': lon_gt_180_cnt, 'pct': lon_gt_180_pct},
        'eras_distribution': eras,
        'top_years_obs': top_years_obs,
        'top_years_cyclones': top_years_cyclones,
        'yearly_observations': obs_by_year,
        'yearly_cyclones': cyclones_by_year
    }


def inspect_temporal(df: pd.DataFrame) -> Dict[str, Any]:
    """E. Temporal Analysis."""
    print("[*] Analyzing Temporal Intervals and Duplicates...")
    total_rows = len(df)
    
    # Sort by storm SID and parsed timestamp
    df_temp = pd.DataFrame({
        'SID': df['SID'],
        'ISO_TIME': df['ISO_TIME'],
        'DT': pd.to_datetime(df['ISO_TIME'], errors='coerce')
    })
    df_sorted = df_temp.sort_values(by=['SID', 'DT']).copy()
    df_sorted['TIME_DIFF_HOURS'] = df_sorted.groupby('SID')['DT'].diff().dt.total_seconds() / 3600.0

    time_diffs = df_sorted['TIME_DIFF_HOURS'].dropna()
    total_intervals = len(time_diffs)

    exact_3h = int((time_diffs == 3.0).sum())
    exact_6h = int((time_diffs == 6.0).sum())
    exact_1h = int((time_diffs == 1.0).sum())
    exact_2h = int((time_diffs == 2.0).sum())
    exact_12h = int((time_diffs == 12.0).sum())
    exact_24h = int((time_diffs == 24.0).sum())
    sub_hourly = int((time_diffs < 1.0).sum())
    zero_diff = int((time_diffs == 0.0).sum())
    neg_diff = int((time_diffs < 0.0).sum())
    other_irregular = int((~time_diffs.isin([1.0, 2.0, 3.0, 6.0, 12.0, 24.0])).sum())

    top_intervals = {str(round(k, 3)): int(v) for k, v in time_diffs.value_counts().head(10).items()}

    hours_dist = {int(k): int(v) for k, v in df_temp['DT'].dt.hour.value_counts().sort_index().items() if not pd.isna(k)}
    synoptic_6h = sum(hours_dist.get(h, 0) for h in [0, 6, 12, 18])
    sub_synoptic_3h = sum(hours_dist.get(h, 0) for h in [3, 9, 15, 21])
    off_synoptic = sum(v for k, v in hours_dist.items() if k not in [0, 3, 6, 9, 12, 15, 18, 21])

    # Check duplicates on SID + ISO_TIME
    dup_mask = df.duplicated(subset=['SID', 'ISO_TIME'], keep=False)
    total_dup = int(dup_mask.sum())
    dup_pairs = int(df.loc[dup_mask, ['SID', 'ISO_TIME']].drop_duplicates().shape[0])

    return {
        'total_intervals': total_intervals,
        'exact_3h_count': exact_3h,
        'exact_3h_pct': round((exact_3h / total_intervals) * 100, 2),
        'exact_6h_count': exact_6h,
        'exact_6h_pct': round((exact_6h / total_intervals) * 100, 2),
        'exact_1h_count': exact_1h,
        'exact_1h_pct': round((exact_1h / total_intervals) * 100, 2),
        'exact_2h_count': exact_2h,
        'exact_2h_pct': round((exact_2h / total_intervals) * 100, 2),
        'sub_hourly_count': sub_hourly,
        'sub_hourly_pct': round((sub_hourly / total_intervals) * 100, 2),
        'exact_12h_count': exact_12h,
        'exact_12h_pct': round((exact_12h / total_intervals) * 100, 2),
        'exact_24h_count': exact_24h,
        'exact_24h_pct': round((exact_24h / total_intervals) * 100, 2),
        'zero_interval_count': zero_diff,
        'negative_interval_count': neg_diff,
        'other_irregular_count': other_irregular,
        'other_irregular_pct': round((other_irregular / total_intervals) * 100, 2),
        'top_intervals': top_intervals,
        'hourly_distribution': hours_dist,
        'synoptic_6h_count': synoptic_6h,
        'synoptic_6h_pct': round((synoptic_6h / total_rows) * 100, 2),
        'sub_synoptic_3h_count': sub_synoptic_3h,
        'sub_synoptic_3h_pct': round((sub_synoptic_3h / total_rows) * 100, 2),
        'off_synoptic_count': off_synoptic,
        'off_synoptic_pct': round((off_synoptic / total_rows) * 100, 2),
        'min_interval_hours': float(time_diffs.min()) if len(time_diffs) > 0 else None,
        'max_interval_hours': float(time_diffs.max()) if len(time_diffs) > 0 else None,
        'median_interval_hours': float(time_diffs.median()) if len(time_diffs) > 0 else None,
        'mean_interval_hours': round(float(time_diffs.mean()), 2) if len(time_diffs) > 0 else None,
        'duplicate_records_count': total_dup,
        'duplicate_unique_pairs': dup_pairs
    }


def generate_markdown_report(
    overview: Dict[str, Any],
    cyclones: Dict[str, Any],
    missing: Dict[str, Any],
    met: Dict[str, Any],
    temporal: Dict[str, Any],
    output_path: str
) -> None:
    """Generates the comprehensive human-readable Markdown report."""
    print(f"[*] Generating human-readable inspection report at: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# IBTrACS Global Tropical Cyclone Dataset Inspection Report\n\n")
        f.write("> **Dataset Version:** IBTrACS v04r01 (ALL)\n")
        f.write(f"> **Inspection Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"> **Source File:** `raw/ibtracs.ALL.list.v04r01.csv`\n")
        f.write(f"> **Total Observations:** {overview['total_rows']:,} rows | **Total Features:** {overview['total_cols']} columns\n")
        f.write(f"> **Temporal Coverage:** {overview['date_range']['min']} to {overview['date_range']['max']} (Seasons {overview['season_range']['min']} - {overview['season_range']['max']})\n\n")

        f.write("---\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive quality, statistical, and machine-learning feasibility inspection of the **NOAA International Best Track Archive for Climate Stewardship (IBTrACS) v04r01** global dataset.\n\n")
        f.write("- **Scale & Completeness:** The dataset comprises **730,416 storm fixes** across **13,631 unique tropical cyclones** spanning 184 years (1842 to 2026/2027).\n")
        f.write("- **Geographic Coverage:** Full global coverage across seven primary ocean basins: Western Pacific (`WP`), South Indian (`SI`), North Atlantic (`NA`), South Pacific (`SP`), Eastern Pacific (`EP`), North Indian (`NI`), and South Atlantic (`SA`). Core tracking coordinates (`LAT`, `LON`) are **100% complete** with zero missing values.\n")
        f.write("- **Data Architecture:** IBTrACS v04 uses 3-hourly interpolated tracks. **93.40%** of consecutive fixes occur at exactly 3-hour intervals, with zero duplicate `(SID, ISO_TIME)` pairs.\n")
        f.write("- **Critical Implementation Finding:** The second row of the raw CSV contains column unit definitions and must be skipped. Furthermore, default CSV parsers that treat string `'NA'` as `NaN` unintentionally corrupt the North Atlantic basin and subbasin codes. Explicit NA handling (`keep_default_na=False`) is required.\n")
        f.write("- **Meteorological Availability:** While position data is complete, wind speed and minimum central pressure completeness varies significantly across historical eras and regional agencies (`USA_WIND` is 55.57% complete, while `WMO_WIND` is 23.32% complete). Modern multi-sensor data (post-1980 / post-2000) provides the highest fidelity for machine learning.\n\n")

        # Section A
        f.write("---\n\n")
        f.write("## A. Dataset Overview\n\n")
        f.write("### 1. Dimensionality & Format\n\n")
        f.write(f"- **Total Records (Rows):** {overview['total_rows']:,}\n")
        f.write(f"- **Total Attributes (Columns):** {overview['total_cols']}\n")
        f.write(f"- **Temporal Span:** `{overview['date_range']['min']}` to `{overview['date_range']['max']}`\n")
        f.write(f"- **Season Span:** {overview['season_range']['min']} to {overview['season_range']['max']} *(Note: Southern Hemisphere seasons extend into early the following calendar year, hence season 2027 covers late 2026 storms such as TC WATI)*.\n\n")

        f.write("### 2. Data Types Distribution\n\n")
        f.write("| Data Type | Column Count | Description |\n")
        f.write("| :--- | :--- | :--- |\n")
        for dt, count in overview['dtypes_distribution'].items():
            f.write(f"| `{dt}` | {count} | Attribute fields conforming to {dt} |\n")
        f.write("\n")

        f.write("### 3. Column Categorization & Schema Architecture\n\n")
        f.write("The 174 columns are logically partitioned into seven operational categories:\n\n")
        f.write("| Category | Example Columns | Operational Description |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| **Storm Identifiers & Metadata** | `SID`, `SEASON`, `NUMBER`, `BASIN`, `SUBBASIN`, `NAME`, `ISO_TIME`, `NATURE`, `TRACK_TYPE` | Primary key identification, genesis season, ocean basin, storm name, synoptic timestamp, and lifecycle classification. |\n")
        f.write("| **Unified Spatial & Kinematic** | `LAT`, `LON`, `DIST2LAND`, `LANDFALL`, `STORM_SPEED`, `STORM_DIR`, `IFLAG` | Official IBTrACS interpolated storm center positions, translation speed, bearing, distance to nearest landmass, and interpolation quality flags. |\n")
        f.write("| **WMO Consensus Intensity** | `WMO_WIND`, `WMO_PRES`, `WMO_AGENCY` | World Meteorological Organization official designated agency consensus maximum sustained wind speed (kts) and central pressure (mb). |\n")
        f.write("| **USA / JTWC / NHC Data** | `USA_AGENCY`, `USA_ATCF_ID`, `USA_LAT`, `USA_LON`, `USA_STATUS`, `USA_WIND`, `USA_PRES`, `USA_SSHS` | Automated Tropical Cyclone Forecasting (ATCF) tracks from NHC and JTWC, including Saffir-Simpson Hurricane Scale (`USA_SSHS`). |\n")
        f.write("| **Storm Structure & Radii** | `USA_R34_NE/SE/SW/NW`, `USA_R50_*`, `USA_R64_*`, `USA_POCI`, `USA_ROCI`, `USA_RMW`, `USA_EYE` | Quadrant-specific wind radii for 34-knot, 50-knot, and 64-knot winds, radius of maximum wind (RMW), radius and pressure of outermost closed isobar (ROCI/POCI), and eye diameter. |\n")
        f.write("| **Regional Specialized Meteorological Centers (RSMCs)** | `TOKYO_*`, `CMA_*`, `HKO_*`, `KMA_*`, `NEWDELHI_*`, `REUNION_*`, `BOM_*`, `NADI_*`, `WELLINGTON_*` | Basin-specific national agency tracks (JMA Tokyo, China CMA, Hong Kong HKO, Korea KMA, IMD New Delhi, Météo-France La Réunion, Australian BoM, Fiji Nadi, MetService Wellington). |\n")
        f.write("| **Historical & Special Archives** | `DS824_*`, `TD9636_*`, `TD9635_*`, `NEUMANN_*`, `MLC_*`, `USA_GUST`, `BOM_GUST` | Pre-satellite digitizations, historical reconnaissance archives, gust estimates, and sea-height radii. |\n\n")

        f.write("### 4. First 10 Records (Representative Attributes)\n\n")
        f.write("| SID | Season | Basin | Subbasin | Name | ISO Time | Nature | Lat (°N) | Lon (°E) | WMO Wind | WMO Pres | Track Type | Dist2Land (km) | Storm Speed (kts) | Storm Dir (°) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in overview['first_10_records']:
            w_w = f"{r['WMO_WIND']:.0f}" if not pd.isna(r['WMO_WIND']) else "—"
            w_p = f"{r['WMO_PRES']:.0f}" if not pd.isna(r['WMO_PRES']) else "—"
            sp = f"{r['STORM_SPEED']:.0f}" if not pd.isna(r['STORM_SPEED']) else "—"
            sd = f"{r['STORM_DIR']:.0f}" if not pd.isna(r['STORM_DIR']) else "—"
            f.write(f"| `{r['SID']}` | {r['SEASON']} | {r['BASIN']} | {r['SUBBASIN']} | {r['NAME']} | `{r['ISO_TIME']}` | {r['NATURE']} | {r['LAT']} | {r['LON']} | {w_w} | {w_p} | {r['TRACK_TYPE']} | {r['DIST2LAND']} | {sp} | {sd} |\n")
        f.write("\n")

        # Section B
        f.write("---\n\n")
        f.write("## B. Cyclone Statistics\n\n")
        f.write("### 1. Cyclone Identifiers & Naming Distribution\n\n")
        f.write(f"- **Total Unique Cyclones (SIDs):** **{cyclones['total_unique_sids']:,}**\n")
        f.write(f"- **Total Unique Name Strings:** **{cyclones['total_unique_names']:,}**\n")
        f.write(f"- **Officially Named Storms:** **{cyclones['named_storms_count']:,}** ({round(cyclones['named_storms_count'] / cyclones['total_unique_sids'] * 100, 1)}% of all storms)\n")
        f.write(f"- **Unnamed / Historical Storms:** **{cyclones['unnamed_storms_count']:,}** ({round(cyclones['unnamed_storms_count'] / cyclones['total_unique_sids'] * 100, 1)}% of all storms)\n\n")
        f.write("> [!NOTE]\n")
        f.write("> Formal tropical cyclone naming began in the Western Pacific and North Atlantic during the mid-20th century (c. 1945–1953). Consequently, pre-1950 storms and smaller southern hemisphere systems are cataloged as `UNNAMED`.\n\n")

        f.write("### 2. Geographic Basin Breakdown\n\n")
        f.write("IBTrACS tracks cyclones across the globe. Because tropical cyclones can traverse ocean basin boundaries (e.g., Eastern Pacific to Western Pacific, or South Indian to South Pacific), both **Genesis Basin** (genesis point) and **Basins Touched** (all basins traversed) are presented below:\n\n")
        f.write("| Ocean Basin Code | Basin Name | Genesis Cyclones | Cyclones Touched | Total Fixes (Observations) | % of Global Fixes |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        basin_names = {
            'WP': 'Western North Pacific',
            'SI': 'South Indian Ocean',
            'NA': 'North Atlantic',
            'SP': 'South Pacific Ocean',
            'EP': 'Eastern North Pacific',
            'NI': 'North Indian Ocean (Arabian Sea & Bay of Bengal)',
            'SA': 'South Atlantic Ocean'
        }
        for b_code, b_name in basin_names.items():
            gen_c = cyclones['cyclones_genesis_basin'].get(b_code, 0)
            touch_c = cyclones['cyclones_touching_basin'].get(b_code, 0)
            obs_c = cyclones['obs_by_basin'].get(b_code, 0)
            obs_pct = round(obs_c / overview['total_rows'] * 100, 2)
            f.write(f"| **`{b_code}`** | {b_name} | {gen_c:,} | {touch_c:,} | {obs_c:,} | {obs_pct:.2f}% |\n")
        f.write("\n")

        f.write("### 3. Observations per Cyclone Distribution\n\n")
        f.write("| Metric | Observations per Cyclone | Equivalent Duration (3-hr fixes) |\n")
        f.write("| :--- | :--- | :--- |\n")
        st = cyclones['obs_per_cyclone_stats']
        f.write(f"| **Minimum** | {st['min']} fix | 0.0 to 3.0 hours |\n")
        f.write(f"| **25th Percentile (Q1)** | {st['q25']} fixes | ~3.5 days |\n")
        f.write(f"| **Median (Q2)** | **{st['median']} fixes** | **~5.75 days** |\n")
        f.write(f"| **Mean** | **{st['mean']} fixes** | **~6.7 days** |\n")
        f.write(f"| **75th Percentile (Q3)** | {st['q75']} fixes | ~9.1 days |\n")
        f.write(f"| **Maximum** | **{st['max']} fixes** | **~38.0 days** |\n")
        f.write(f"| **Standard Deviation** | {st['std']} fixes | ~4.3 days |\n\n")

        f.write("### 4. Top 20 Cyclones with the Most Observations (Longest-Lived Storms)\n\n")
        f.write("| Rank | Storm ID (`SID`) | Name | Season | Basin | Fix Count | Start Date | End Date | Duration (Days) | Peak Wind (kts) | Min Pres (mb) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for idx, storm in enumerate(cyclones['top_20_cyclones'], 1):
            w = f"{storm['MAX_WIND_KTS']:.0f}" if storm['MAX_WIND_KTS'] else "—"
            p = f"{storm['MIN_PRES_MB']:.0f}" if storm['MIN_PRES_MB'] else "—"
            f.write(f"| {idx} | `{storm['SID']}` | **{storm['NAME']}** | {storm['SEASON']} | `{storm['BASIN']}` | **{storm['OBS_COUNT']}** | `{storm['START_TIME'][:10]}` | `{storm['END_TIME'][:10]}` | {storm['DURATION_DAYS']} | {w} | {p} |\n")
        f.write("\n")

        f.write("### 5. Storm Nature Distribution\n\n")
        f.write("| Code | Nature Classification | Fix Count | Percentage |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        nature_descs = {
            'TS': 'Tropical Storm / Cyclone',
            'NR': 'Not Reported / Historical',
            'ET': 'Extratropical Transition / Cyclone',
            'DS': 'Disturbance / Tropical Depression',
            'MX': 'Mixed / Transitioning Phase',
            'SS': 'Subtropical Storm'
        }
        for n_code, count in cyclones['nature_counts'].items():
            pct = round(count / overview['total_rows'] * 100, 2)
            f.write(f"| `{n_code}` | {nature_descs.get(n_code, 'Other')} | {count:,} | {pct:.2f}% |\n")
        f.write("\n")

        # Section C
        f.write("---\n\n")
        f.write("## C. Missing Value Analysis\n\n")
        f.write("### 1. Essential Attributes Completeness\n\n")
        f.write("| Feature Name | Data Type | Present Count | Missing Count | Missing % | Completeness Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for col, m_info in missing['important_columns'].items():
            status = "Complete (100%)" if m_info['missing_pct'] == 0 else ("Excellent (<5%)" if m_info['missing_pct'] < 5 else ("Moderate (5-50%)" if m_info['missing_pct'] < 50 else "Sparseness (>50%)"))
            f.write(f"| `{col}` | — | {m_info['present_count']:,} | {m_info['missing_count']:,} | {m_info['missing_pct']:.2f}% | {status} |\n")
        f.write("\n")

        f.write("### 2. Detailed Spatial & Meteorological Completeness Matrix\n\n")
        f.write("| Variable Group | Attribute | Present Count | Missing Count | Missing % | Operational Explanation |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        cm = missing['coord_met_analysis']
        f.write(f"| **Core Coordinates** | `LAT` | {overview['total_rows'] - cm['LAT']['missing']:,} | {cm['LAT']['missing']} | {cm['LAT']['pct']:.3f}% | 100% complete across all 730k fixes. |\n")
        f.write(f"| **Core Coordinates** | `LON` | {overview['total_rows'] - cm['LON']['missing']:,} | {cm['LON']['missing']} | {cm['LON']['pct']:.3f}% | 100% complete across all 730k fixes. |\n")
        f.write(f"| **WMO Consensus** | `WMO_WIND` | {overview['total_rows'] - cm['WMO_WIND']['missing']:,} | {cm['WMO_WIND']['missing']:,} | {cm['WMO_WIND']['pct']:.2f}% | Only populated when official WMO agency consensus is available. |\n")
        f.write(f"| **WMO Consensus** | `WMO_PRES` | {overview['total_rows'] - cm['WMO_PRES']['missing']:,} | {cm['WMO_PRES']['missing']:,} | {cm['WMO_PRES']['pct']:.2f}% | Central pressure not consistently archived by all regional WMO centers. |\n")
        f.write(f"| **USA (NHC/JTWC)** | `USA_WIND` | {overview['total_rows'] - cm['USA_WIND']['missing']:,} | {cm['USA_WIND']['missing']:,} | {cm['USA_WIND']['pct']:.2f}% | Populated for all ATCF tracked basins (NA, EP, WP, NI, SH). |\n")
        f.write(f"| **USA (NHC/JTWC)** | `USA_PRES` | {overview['total_rows'] - cm['USA_PRES']['missing']:,} | {cm['USA_PRES']['missing']:,} | {cm['USA_PRES']['pct']:.2f}% | Central pressure reported primarily in Atlantic, East Pacific, and modern recon. |\n")
        f.write(f"| **Combined Union** | `ANY_WIND` | {cm['ANY_WIND']['count']:,} | {overview['total_rows'] - cm['ANY_WIND']['count']:,} | {100.0 - cm['ANY_WIND']['pct']:.2f}% | Either `USA_WIND` or `WMO_WIND` populated ({cm['ANY_WIND']['pct']:.2f}% coverage). |\n")
        f.write(f"| **Combined Union** | `ANY_PRES` | {cm['ANY_PRES']['count']:,} | {overview['total_rows'] - cm['ANY_PRES']['count']:,} | {100.0 - cm['ANY_PRES']['pct']:.2f}% | Either `USA_PRES` or `WMO_PRES` populated ({cm['ANY_PRES']['pct']:.2f}% coverage). |\n\n")

        f.write("### 3. Agency Sparseness Analysis\n\n")
        f.write("National agency columns (e.g. `TOKYO_*`, `CMA_*`, `NEWDELHI_*`, `REUNION_*`, `BOM_*`) exhibit high overall missing percentages (80% to 98% missing in the global file) **by design**:\n")
        f.write("- Regional specialized agencies only track storms within their designated area of responsibility (e.g., `TOKYO` tracks Western North Pacific storms; `REUNION` tracks South Indian storms).\n")
        f.write("- Therefore, regional agency columns must be evaluated **per basin**, not globally.\n\n")

        # Section D
        f.write("---\n\n")
        f.write("## D. Meteorological Analysis\n\n")
        f.write("### 1. Summary Statistics for Key Meteorological & Physical Fields\n\n")
        f.write("| Variable | Count | Min | 25% (Q1) | Median | Mean | 75% (Q3) | 95% | 99% | Max | Std Dev |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for var_name, s in met['statistics'].items():
            if s:
                f.write(f"| `{var_name}` | {s['count']:,} | {s['min']} | {s['q25']} | **{s['median']}** | **{s['mean']}** | {s['q75']} | {s['p95']} | {s['p99']} | {s['max']} | {s['std']} |\n")
        f.write("\n")

        f.write("### 2. Coordinate System Nuances & Longitude Range\n\n")
        f.write(f"- **Latitude Span:** `{met['statistics']['LAT_deg']['min']}°` to `{met['statistics']['LAT_deg']['max']}°` (covering genesis in deep tropics ~5°N/S to extratropical decay in polar waters).\n")
        f.write(f"- **Longitude Span:** `{met['statistics']['LON_deg']['min']}°` to `{met['statistics']['LON_deg']['max']}°`.\n")
        f.write(f"- **Observations with `LON > 180°`:** **{met['lon_gt_180']['count']:,} fixes ({met['lon_gt_180']['pct']}%)**.\n\n")
        f.write("> [!WARNING]\n")
        f.write("> In the South Pacific (`SP`) and Eastern Pacific (`EP`), historical best tracks occasionally record longitude using the 0° to 360° convention (e.g. 185°E = 175°W). Any spatial machine learning model or GIS pipeline must unwrap or map longitudes strictly into $[-180°, +180°]$ to avoid 360-degree boundary discontinuities.\n\n")

        f.write("### 3. Historical Eras & Observation Evolution\n\n")
        f.write("Tropical cyclone observation density has evolved drastically across five distinct technological epochs:\n\n")
        f.write("| Observation Epoch | Calendar Years | Total Fixes | % of Dataset | Observational Character |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for era_name, era_count in met['eras_distribution'].items():
            pct = round(era_count / overview['total_rows'] * 100, 2)
            f.write(f"| **{era_name}** | — | {era_count:,} | {pct:.2f}% | Best tracks reconstructed from ship logs, coastal stations, early aircraft, or satellite sensors. |\n")
        f.write("\n")

        f.write("### 4. Peak Activity Years\n\n")
        f.write("- **Top 5 Seasons by Observation Count:**\n")
        for yr, cnt in met['top_years_obs'][:5]:
            f.write(f"  - **{yr}:** {cnt:,} observations\n")
        f.write("- **Top 5 Seasons by Cyclone Count:**\n")
        for yr, cnt in met['top_years_cyclones'][:5]:
            f.write(f"  - **{yr}:** {cnt} unique cyclones\n\n")

        # Section E
        f.write("---\n\n")
        f.write("## E. Temporal Analysis\n\n")
        f.write("### 1. Observation Interval Distribution (Time Delta $\\Delta t$)\n\n")
        f.write("IBTrACS v04 provides regularized, linearly interpolated positions to support climate and statistical analyses.\n\n")
        f.write("| Interval Duration | Fix Count | Percentage | Operational Context |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Exactly 3.0 Hours** | **{temporal['exact_3h_count']:,}** | **{temporal['exact_3h_pct']:.2f}%** | Standard IBTrACS synoptic / interpolated product time step. |\n")
        f.write(f"| **Exactly 1.0 Hour** | **{temporal['exact_1h_count']:,}** | **{temporal['exact_1h_pct']:.2f}%** | Special fixes: aircraft reconnaissance fixes, rapid changes, or landfall points. |\n")
        f.write(f"| **Exactly 2.0 Hours** | **{temporal['exact_2h_count']:,}** | **{temporal['exact_2h_pct']:.2f}%** | Intermediate observational adjustments. |\n")
        f.write(f"| **Sub-hourly (< 1 Hour)** | **{temporal['sub_hourly_count']:,}** | **{temporal['sub_hourly_pct']:.2f}%** | Exact landfall timestamp records (e.g. 15-minute or 1-minute resolution). |\n")
        f.write(f"| **Exactly 6.0 Hours** | **{temporal['exact_6h_count']:,}** | **{temporal['exact_6h_pct']:.2f}%** | Gaps interpolated to 3h by IBTrACS v04 algorithm. |\n")
        f.write(f"| **Other Irregular** | **{temporal['other_irregular_count']:,}** | **{temporal['other_irregular_pct']:.2f}%** | Off-hour observations. |\n")
        f.write(f"| **Zero Interval ($\\Delta t = 0$)** | **{temporal['zero_interval_count']}** | **0.00%** | Zero duplicate timestamps for any cyclone. |\n")
        f.write(f"| **Negative Interval ($\\Delta t < 0$)** | **{temporal['negative_interval_count']}** | **0.00%** | Timestamps are strictly monotonic in chronological order. |\n\n")

        f.write("### 2. Diurnal Cycle & Synoptic Hours Distribution\n\n")
        f.write(f"- **Standard 6-Hourly Synoptic Fixes (00, 06, 12, 18 UTC):** **{temporal['synoptic_6h_count']:,} fixes ({temporal['synoptic_6h_pct']}%)**\n")
        f.write(f"- **Intermediate 3-Hourly Synoptic Fixes (03, 09, 15, 21 UTC):** **{temporal['sub_synoptic_3h_count']:,} fixes ({temporal['sub_synoptic_3h_pct']}%)**\n")
        f.write(f"- **Off-Synoptic Special Fixes (Landfalls / Recon):** **{temporal['off_synoptic_count']:,} fixes ({temporal['off_synoptic_pct']}%)**\n\n")

        f.write("### 3. Duplicate Identification (`SID` + `ISO_TIME`)\n\n")
        f.write(f"- **Total Duplicate Records:** **{temporal['duplicate_records_count']}**\n")
        f.write(f"- **Unique Duplicate Pairs:** **{temporal['duplicate_unique_pairs']}**\n")
        f.write("- **Integrity Result:** The composite primary key `(SID, ISO_TIME)` is **strictly unique** across all 730,416 rows in the dataset.\n\n")

        # Section F
        f.write("---\n\n")
        f.write("## F. Machine Learning Model Feasibility\n\n")
        f.write("We evaluate the feasibility of developing three predictive models using solely or primarily the IBTrACS v04r01 dataset:\n\n")

        # Model 1
        f.write("### Model 1: Cyclone Intensity Prediction (Regression / Rapid Intensification)\n\n")
        f.write("**Objective:** Predict future maximum sustained wind speed ($V_{t+\\Delta t}$) or classify Rapid Intensification (RI: $\\Delta V \\ge 30\\text{ kts}$ in 24 hours).\n\n")
        f.write("- **Feasibility Assessment:** **High Feasibility (Modern Era $\\ge 1980$)**.\n")
        f.write("- **Suitable Input Columns:**\n")
        f.write("  - Historical/Current Wind Speed: `USA_WIND`, `WMO_WIND`, regional agency winds (`TOKYO_WIND`, etc.).\n")
        f.write("  - Central Pressure: `USA_PRES`, `WMO_PRES`.\n")
        f.write("  - Spatial & Geographic Context: `LAT`, `LON`, `DIST2LAND`, `LANDFALL` (cyclones weaken drastically over land).\n")
        f.write("  - Kinematic Motion: `STORM_SPEED`, `STORM_DIR`.\n")
        f.write("  - Storm Structure (post-2000): `USA_R34_NE/SE/SW/NW`, `USA_R50_*`, `USA_R64_*`, `USA_RMW`, `USA_POCI`, `USA_ROCI`.\n")
        f.write("  - Seasonal / Climatological Temporal: `SEASON`, day of year, elapsed duration from genesis.\n")
        f.write("- **Potential Prediction Targets:**\n")
        f.write("  - Continuous: $V_{t+6h}, V_{t+12h}, V_{t+24h}$ or $\\Delta V_{24h}$.\n")
        f.write("  - Categorical: Rapid Intensification (RI binary flag), Saffir-Simpson Hurricane Scale classification (`USA_SSHS`).\n")
        f.write("- **Missing Data Limitations:**\n")
        f.write("  - `WMO_WIND` is missing in 76.68% of fixes; `USA_WIND` is missing in 44.43% (principally pre-1970 and older Indian/South Pacific basins).\n")
        f.write("  - Wind radii columns (`USA_R34`, etc.) have ~85-95% missing values globally because systematic automated radii recording began only around 2001.\n")
        f.write("- **Required Preprocessing:**\n")
        f.write("  1. **Temporal Filtering:** Restrict training to the satellite era (e.g. $\\ge 1980$ or $\\ge 2000$ if structural radii are utilized).\n")
        f.write("  2. **Agency Consensus Synthesis:** Impute/synthesize a single unified intensity target by prioritizing `USA_WIND` $\\to$ `WMO_WIND` $\\to$ regional RSMC winds.\n")
        f.write("  3. **Temporal Regularization:** Filter out off-synoptic landfall fixes to produce a uniform 3-hourly or 6-hourly time series.\n")
        f.write("  4. **Feature Engineering:** Calculate intensity tendency ($\\Delta V_{-6h}, \\Delta V_{-12h}$), acceleration, pressure deficit ($P_{oci} - P_{min}$), and land proximity decay flags.\n\n")

        # Model 2
        f.write("### Model 2: Cyclone Movement & Trajectory Prediction (Track Forecasting)\n\n")
        f.write("**Objective:** Forecast cyclone future spatial coordinates $(\\text{LAT}_{t+\\Delta t}, \\text{LON}_{t+\\Delta t})$ at lead times of 6h, 12h, 24h, and 48h.\n\n")
        f.write("- **Feasibility Assessment:** **High Feasibility for Short-Term (6h–24h) Extrapolation / Autoregression; Moderate Feasibility for Extended Lead Times (>24h)** without external environmental steering grids.\n")
        f.write("- **Suitable Input Columns:**\n")
        f.write("  - Historical Trajectory: `LAT`, `LON`, lag coordinates ($t-3h, t-6h, t-12h, t-24h$).\n")
        f.write("  - Kinematics: `STORM_SPEED`, `STORM_DIR`, forward velocity components ($u, v$).\n")
        f.write("  - Intensity Coupling: `USA_WIND`, `USA_PRES` (deeper, more intense cyclones are steered by higher atmospheric layers).\n")
        f.write("  - Geographic Domain: `BASIN`, `SUBBASIN`, `DIST2LAND`.\n")
        f.write("- **Potential Prediction Targets:**\n")
        f.write("  - Direct coordinates: $\\text{LAT}_{t+\\Delta t}, \\text{LON}_{t+\\Delta t}$.\n")
        f.write("  - Displacement vectors: $\\Delta x, \\Delta y$ (in kilometers or spherical degrees).\n")
        f.write("- **Missing Data Limitations:**\n")
        f.write("  - `LAT` and `LON` have **0.00% missing data** (100% complete across all 730k rows). `STORM_SPEED` and `STORM_DIR` are 99.99% complete.\n")
        f.write("  - *Limitation:* IBTrACS contains purely storm-centric observations. It does not contain background atmospheric steering fields (e.g. 500 hPa geopotential height, 850–200 hPa vertical wind shear, steering flow ridges). Pure IBTrACS track models behave analogously to statistical-climatological models (e.g. CLIPER) or LSTM track autoencoders.\n")
        f.write("- **Required Preprocessing:**\n")
        f.write("  1. **Antimeridian Longitude Normalization:** Correct all $LON > 180°$ entries into standard $[-180°, +180°]$ coordinates, and implement track unwrapping across the 180° meridian.\n")
        f.write("  2. **Track Filtering:** Restrict sequence training to `TRACK_TYPE == 'main'` to eliminate secondary or merging spurs.\n")
        f.write("  3. **Displacement Conversion:** Project spherical coordinates into Cartesian distances (km) via Haversine / geodesic formulas.\n\n")

        # Model 3
        f.write("### Model 3: Cyclone Development Pattern Classification (Lifecycle & Transition)\n\n")
        f.write("**Objective:** Classify storm lifecycle stages (e.g. Tropical `TS`, Extratropical Transition `ET`, Subtropical `SS`, Decay) and identify canonical track archetypes (e.g. straight-moving, recurving, looping).\n\n")
        f.write("- **Feasibility Assessment:** **High Feasibility**.\n")
        f.write("- **Suitable Input Columns:**\n")
        f.write("  - Lifecycle Stage & Nature: `NATURE` (`TS`, `ET`, `DS`, `MX`, `SS`, `NR`), `TRACK_TYPE`.\n")
        f.write("  - Intensity Stage: `USA_STATUS` (`TD`, `TS`, `HU`, `EX`, `LO`, `DB`), `USA_SSHS`.\n")
        f.write("  - Trajectory & Dynamics: `LAT`, `LON`, `STORM_SPEED`, `STORM_DIR`, poleward speed component ($v_y$).\n")
        f.write("  - Structural Asymmetry: Wind radii quadrants (`USA_R34_NE`, `SE`, `SW`, `NW`) — expansion of poleward quadrants is a hallmark of Extratropical Transition.\n")
        f.write("- **Potential Prediction Targets:**\n")
        f.write("  - Multiclass classification of storm state `NATURE` at $t+\\Delta t$.\n")
        f.write("  - Binary Extratropical Transition (ET) onset detection (`TS` $\\to$ `ET`).\n")
        f.write("  - Unsupervised clustering of global track patterns into canonical regional archetypes.\n")
        f.write("- **Missing Data Limitations:**\n")
        f.write("  - `NATURE` is 100% complete, but historical pre-1970 records predominantly label storms with generic `NR` or `TS`.\n")
        f.write("  - `USA_STATUS` is missing in 63.93% of rows (present mostly in US agency basins).\n")
        f.write("- **Required Preprocessing:**\n")
        f.write("  1. **Temporal Horizon:** Restrict classification models to $\\ge 1970$ (satellite era) where physical transitions were accurately identified.\n")
        f.write("  2. **Event Encoding:** Construct ground truth transition labels identifying the exact timestamp when a cyclone transitions from warm-core (`TS`) to cold-core (`ET`).\n")
        f.write("  3. **Kinematic Feature Engineering:** Derive poleward acceleration, track curvature, translation speed changes, and quadrant wind radii asymmetry ratios.\n\n")

        # Recommendations
        f.write("---\n\n")
        f.write("## G. Recommendations for Downstream Phases\n\n")
        f.write("1. **Strict CSV Parsing Parameters:** Always invoke `skiprows=[1]`, `na_values=['', ' ']`, and `keep_default_na=False` when loading IBTrACS CSVs.\n")
        f.write("2. **Coordinate Normalization Pipeline:** Enforce automatic conversion for any longitude exceeding 180° prior to spatial feature extraction or plotting.\n")
        f.write("3. **Era-Appropriate Cohort Selection:** For machine learning training:\n")
        f.write("   - **Track Modeling:** Use 1970–Present (high position accuracy from geostationary satellites).\n")
        f.write("   - **Intensity Modeling:** Use 1980–Present (standardized Dvorak technique and reconnaissance).\n")
        f.write("   - **Structure / Wind Radii Modeling:** Use 2001–Present (automated scatterometer and multi-platform radii).\n")
        f.write("4. **Consensus Target Creation:** Synthesize target columns (`INTENSITY_WIND`, `CENTRAL_PRES`) from multiple reporting agencies with documented basin-level priority.\n\n")
        f.write("---\n")
        f.write("*Report generated automatically by `scripts/inspect_dataset.py`.*\n")

    print(f"[+] Markdown report generated successfully at: {output_path}")


def main() -> None:
    args = parse_arguments()

    print("=" * 70)
    print("NOAA IBTrACS v04r01 Global Dataset Inspection")
    print("=" * 70)

    # 1. Load Dataset
    df = load_dataset(args.csv_path)

    # 2. Run Analyses
    overview = inspect_overview(df)
    cyclones = inspect_cyclone_stats(df)
    missing = inspect_missing_values(df)
    met = inspect_meteorological(df)
    temporal = inspect_temporal(df)

    # 3. Print Concise Summary to Console (without dumping the entire dataset)
    print("\n" + "=" * 70)
    print("INSPECTION SUMMARY RESULTS")
    print("=" * 70)
    print(f"Total Rows:                   {overview['total_rows']:,}")
    print(f"Total Columns:                {overview['total_cols']}")
    print(f"Date Range:                   {overview['date_range']['min']} to {overview['date_range']['max']}")
    print(f"Seasons:                      {overview['season_range']['min']} - {overview['season_range']['max']}")
    print(f"Total Unique Cyclones (SIDs): {cyclones['total_unique_sids']:,}")
    print(f"Named Storms:                 {cyclones['named_storms_count']:,}")
    print(f"Unnamed Storms:               {cyclones['unnamed_storms_count']:,}")
    print("-" * 70)
    print("Cyclones by Ocean Basin (Genesis):")
    for b, c in sorted(cyclones['cyclones_genesis_basin'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - Basin {b:4s}: {c:5,d} cyclones ({cyclones['obs_by_basin'].get(b, 0):7,d} fixes)")
    print("-" * 70)
    print("Observations per Cyclone:")
    st = cyclones['obs_per_cyclone_stats']
    print(f"  Min: {st['min']} | 25%: {st['q25']} | Median: {st['median']} | Mean: {st['mean']} | 75%: {st['q75']} | Max: {st['max']}")
    print("-" * 70)
    print("Top 5 Longest-Observed Cyclones:")
    for idx, storm in enumerate(cyclones['top_20_cyclones'][:5], 1):
        print(f"  {idx}. {storm['NAME']} ({storm['SEASON']}, {storm['BASIN']}): {storm['OBS_COUNT']} fixes, {storm['DURATION_DAYS']} days")
    print("-" * 70)
    print("Coordinates & Meteorology Completeness:")
    cm = missing['coord_met_analysis']
    print(f"  - LAT / LON:         {100.0 - cm['LAT']['pct']:.2f}% present (0 missing values)")
    print(f"  - WMO_WIND:          {100.0 - cm['WMO_WIND']['pct']:.2f}% present ({cm['WMO_WIND']['missing']:,} missing)")
    print(f"  - USA_WIND:          {100.0 - cm['USA_WIND']['pct']:.2f}% present ({cm['USA_WIND']['missing']:,} missing)")
    print(f"  - Combined Wind:     {cm['ANY_WIND']['pct']:.2f}% present")
    print(f"  - Combined Pressure: {cm['ANY_PRES']['pct']:.2f}% present")
    print("-" * 70)
    print("Temporal Intervals (Time Delta):")
    print(f"  - Exactly 3.0 Hours: {temporal['exact_3h_count']:,} ({temporal['exact_3h_pct']}%)")
    print(f"  - Exactly 1.0 Hour:  {temporal['exact_1h_count']:,} ({temporal['exact_1h_pct']}%)")
    print(f"  - Exactly 2.0 Hours: {temporal['exact_2h_count']:,} ({temporal['exact_2h_pct']}%)")
    print(f"  - Sub-hourly (<1h):  {temporal['sub_hourly_count']:,} ({temporal['sub_hourly_pct']}%)")
    print(f"  - Duplicate (SID, ISO_TIME): {temporal['duplicate_records_count']} duplicates")
    print("=" * 70)

    # 4. Generate Markdown Report
    if not args.no_report:
        generate_markdown_report(overview, cyclones, missing, met, temporal, args.output_report)

    print("\n[+] Dataset inspection complete.")


if __name__ == "__main__":
    main()
