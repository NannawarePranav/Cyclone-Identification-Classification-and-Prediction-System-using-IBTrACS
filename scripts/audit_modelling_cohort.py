import os
import json
import time
import pandas as pd
import numpy as np

CSV_PATH = "raw/ibtracs.ALL.list.v04r01.csv"
OUT_JSON = "reports/audit_data.json"

print(f"[*] Loading IBTrACS CSV from {CSV_PATH}...")
start_time = time.time()

# Remember skiprows=[1], keep_default_na=False, na_values=['', ' ']
df = pd.read_csv(
    CSV_PATH,
    skiprows=[1],
    na_values=["", " "],
    keep_default_na=False,
    low_memory=False
)

print(f"[+] Loaded {len(df):,} rows in {time.time() - start_time:.2f}s")

# Convert key numeric columns
numeric_cols = [
    'SEASON', 'LAT', 'LON', 'USA_WIND', 'USA_PRES', 'WMO_WIND', 'WMO_PRES',
    'STORM_SPEED', 'STORM_DIR', 'DIST2LAND', 'LANDFALL',
    'TOKYO_WIND', 'TOKYO_PRES', 'CMA_WIND', 'CMA_PRES',
    'HKO_WIND', 'HKO_PRES', 'KMA_WIND', 'KMA_PRES',
    'NEWDELHI_WIND', 'NEWDELHI_PRES', 'REUNION_WIND', 'REUNION_PRES',
    'BOM_WIND', 'BOM_PRES', 'NADI_WIND', 'NADI_PRES',
    'WELLINGTON_WIND', 'WELLINGTON_PRES'
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Parse datetime
df['DT'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df['YEAR'] = df['DT'].dt.year

# Sort chronologically by storm and timestamp
df = df.sort_values(by=['SID', 'DT']).reset_index(drop=True)

# -------------------------------------------------------------
# 1. MODEL 1: INTENSITY PREDICTION INVESTIGATION
# -------------------------------------------------------------
print("[*] Auditing Model 1: Cyclone Intensity Prediction...")

# Availability of USA_WIND and WMO_WIND by basin and year
basins = ['WP', 'SI', 'NA', 'SP', 'EP', 'NI', 'SA']
basin_wind_audit = {}
for b in basins:
    sub = df[df['BASIN'] == b]
    n_obs = len(sub)
    n_sids = sub['SID'].nunique()
    usa_w_cnt = sub['USA_WIND'].notna().sum()
    wmo_w_cnt = sub['WMO_WIND'].notna().sum()
    any_w_cnt = ((sub['USA_WIND'].notna()) | (sub['WMO_WIND'].notna())).sum()
    both_w_cnt = ((sub['USA_WIND'].notna()) & (sub['WMO_WIND'].notna())).sum()
    
    basin_wind_audit[b] = {
        'total_obs': int(n_obs),
        'unique_sids': int(n_sids),
        'usa_wind_obs': int(usa_w_cnt),
        'usa_wind_pct': round(usa_w_cnt / n_obs * 100, 2) if n_obs > 0 else 0,
        'wmo_wind_obs': int(wmo_w_cnt),
        'wmo_wind_pct': round(wmo_w_cnt / n_obs * 100, 2) if n_obs > 0 else 0,
        'any_wind_obs': int(any_w_cnt),
        'any_wind_pct': round(any_w_cnt / n_obs * 100, 2) if n_obs > 0 else 0,
        'both_wind_obs': int(both_w_cnt),
        'both_wind_pct': round(both_w_cnt / n_obs * 100, 2) if n_obs > 0 else 0,
    }

# Regional agency wind fields availability
regional_wind_cols = [
    'TOKYO_WIND', 'CMA_WIND', 'HKO_WIND', 'KMA_WIND',
    'NEWDELHI_WIND', 'REUNION_WIND', 'BOM_WIND',
    'NADI_WIND', 'WELLINGTON_WIND'
]
reg_wind_audit = {}
for col in regional_wind_cols:
    cnt = df[col].notna().sum()
    reg_wind_audit[col] = {
        'present_count': int(cnt),
        'pct_of_global': round(cnt / len(df) * 100, 2),
        'min_year': int(df.loc[df[col].notna(), 'YEAR'].min()) if cnt > 0 else None,
        'max_year': int(df.loc[df[col].notna(), 'YEAR'].max()) if cnt > 0 else None,
        'basins_active': df.loc[df[col].notna(), 'BASIN'].value_counts().to_dict()
    }

# WMO_AGENCY distribution
wmo_agency_counts = df['WMO_AGENCY'].value_counts(dropna=False).to_dict()
wmo_agency_dist = {str(k): int(v) for k, v in wmo_agency_counts.items()}

# Wind reporting source consistency throughout individual storms
# How many storms have multiple distinct WMO_AGENCY values (excluding NaN)?
def check_agency_consistency(series):
    agencies = set(series.dropna())
    return len(agencies)

storms_wmo_agencies = df.groupby('SID')['WMO_AGENCY'].apply(check_agency_consistency)
storm_wmo_agency_consistency = {
    'storms_with_0_wmo_agencies': int((storms_wmo_agencies == 0).sum()),
    'storms_with_1_wmo_agency': int((storms_wmo_agencies == 1).sum()),
    'storms_with_multi_wmo_agencies': int((storms_wmo_agencies > 1).sum()),
    'max_wmo_agencies_in_single_storm': int(storms_wmo_agencies.max())
}

# USA_AGENCY consistency
storms_usa_agencies = df.groupby('SID')['USA_AGENCY'].apply(check_agency_consistency)
storm_usa_agency_consistency = {
    'storms_with_0_usa_agencies': int((storms_usa_agencies == 0).sum()),
    'storms_with_1_usa_agency': int((storms_usa_agencies == 1).sum()),
    'storms_with_multi_usa_agencies': int((storms_usa_agencies > 1).sum()),
    'max_usa_agencies_in_single_storm': int(storms_usa_agencies.max())
}

# Number of cyclones with sufficient continuous wind observations
# For each storm, what is the maximum consecutive run of non-null USA_WIND, WMO_WIND?
def max_consecutive_valid(series):
    is_valid = series.notna()
    if not is_valid.any():
        return 0
    # Group consecutive Trues
    blocks = (~is_valid).cumsum()
    valid_blocks = blocks[is_valid]
    if len(valid_blocks) == 0:
        return 0
    return int(valid_blocks.value_counts().max())

print("[*] Computing max consecutive wind observations per storm...")
consec_usa_wind = df.groupby('SID')['USA_WIND'].apply(max_consecutive_valid)
consec_wmo_wind = df.groupby('SID')['WMO_WIND'].apply(max_consecutive_valid)

def get_run_threshold_counts(series):
    return {
        'at_least_4_fixes (12h)': int((series >= 4).sum()),
        'at_least_8_fixes (24h)': int((series >= 8).sum()),
        'at_least_12_fixes (36h)': int((series >= 12).sum()),
        'at_least_16_fixes (48h)': int((series >= 16).sum()),
        'at_least_24_fixes (72h)': int((series >= 24).sum()),
        'median_max_run': float(series.median()),
        'mean_max_run': round(float(series.mean()), 2)
    }

consec_usa_summary = get_run_threshold_counts(consec_usa_wind)
consec_wmo_summary = get_run_threshold_counts(consec_wmo_wind)

# Compare wind coverage for 1980-present and 2000-present
df_1980 = df[df['SEASON'] >= 1980]
df_2000 = df[df['SEASON'] >= 2000]

def cohort_wind_profile(sub_df, name):
    n_obs = len(sub_df)
    n_sids = sub_df['SID'].nunique()
    usa_w = sub_df['USA_WIND'].notna().sum()
    wmo_w = sub_df['WMO_WIND'].notna().sum()
    any_w = ((sub_df['USA_WIND'].notna()) | (sub_df['WMO_WIND'].notna())).sum()
    both_w = ((sub_df['USA_WIND'].notna()) & (sub_df['WMO_WIND'].notna())).sum()
    
    # Difference between USA_WIND and WMO_WIND where both exist
    both_mask = sub_df['USA_WIND'].notna() & sub_df['WMO_WIND'].notna()
    diff = sub_df.loc[both_mask, 'USA_WIND'] - sub_df.loc[both_mask, 'WMO_WIND']
    
    return {
        'cohort_name': name,
        'total_obs': int(n_obs),
        'unique_sids': int(n_sids),
        'usa_wind_obs': int(usa_w),
        'usa_wind_pct': round(usa_w / n_obs * 100, 2),
        'wmo_wind_obs': int(wmo_w),
        'wmo_wind_pct': round(wmo_w / n_obs * 100, 2),
        'any_wind_obs': int(any_w),
        'any_wind_pct': round(any_w / n_obs * 100, 2),
        'both_wind_obs': int(both_w),
        'both_wind_pct': round(both_w / n_obs * 100, 2),
        'usa_minus_wmo_mean_diff': round(float(diff.mean()), 2) if len(diff) > 0 else None,
        'usa_minus_wmo_median_diff': round(float(diff.median()), 2) if len(diff) > 0 else None,
        'usa_minus_wmo_std_diff': round(float(diff.std()), 2) if len(diff) > 0 else None,
        'usa_minus_wmo_q25': round(float(diff.quantile(0.25)), 2) if len(diff) > 0 else None,
        'usa_minus_wmo_q75': round(float(diff.quantile(0.75)), 2) if len(diff) > 0 else None,
        'usa_higher_pct': round((diff > 0).mean() * 100, 2) if len(diff) > 0 else None,
        'usa_equal_pct': round((diff == 0).mean() * 100, 2) if len(diff) > 0 else None,
        'usa_lower_pct': round((diff < 0).mean() * 100, 2) if len(diff) > 0 else None,
    }

wind_profile_1980 = cohort_wind_profile(df_1980, "1980-Present")
wind_profile_2000 = cohort_wind_profile(df_2000, "2000-Present")

# -------------------------------------------------------------
# 2. MODEL 2: MOVEMENT PREDICTION INVESTIGATION
# -------------------------------------------------------------
print("[*] Auditing Model 2: Cyclone Movement Prediction...")

# Track type distribution
track_type_obs = df['TRACK_TYPE'].value_counts(dropna=False).to_dict()
track_type_sids = df.groupby('TRACK_TYPE')['SID'].nunique().to_dict()
track_types_audit = {
    str(k): {'observations': int(track_type_obs[k]), 'unique_cyclones': int(track_type_sids[k])}
    for k in track_type_obs
}

# Unique main-track cyclones
main_df = df[df['TRACK_TYPE'] == 'main']
total_main_sids = int(main_df['SID'].nunique())
total_main_obs = int(len(main_df))

# Cyclones from 1970 onwards
df_1970 = df[df['SEASON'] >= 1970]
main_1970 = df[(df['SEASON'] >= 1970) & (df['TRACK_TYPE'] == 'main')]
cyclones_1970_total = int(df_1970['SID'].nunique())
cyclones_1970_main = int(main_1970['SID'].nunique())

# Observation counts per cyclone for 1970+ main track
obs_per_main_1970 = main_1970.groupby('SID').size()
obs_stats_main_1970 = {
    'count_storms': int(len(obs_per_main_1970)),
    'min': int(obs_per_main_1970.min()),
    'max': int(obs_per_main_1970.max()),
    'mean': round(float(obs_per_main_1970.mean()), 2),
    'median': float(obs_per_main_1970.median()),
    'std': round(float(obs_per_main_1970.std()), 2),
    'q25': float(obs_per_main_1970.quantile(0.25)),
    'q75': float(obs_per_main_1970.quantile(0.75)),
}

# Longitude > 180 analysis
lon_gt_180_df = df[df['LON'] > 180.0]
lon_gt_180_total = int(len(lon_gt_180_df))
lon_gt_180_sids = int(lon_gt_180_df['SID'].nunique())
lon_gt_180_basins = lon_gt_180_df['BASIN'].value_counts().to_dict()
lon_gt_180_eras = {
    'pre_1970': int((lon_gt_180_df['SEASON'] < 1970).sum()),
    '1970_1999': int(((lon_gt_180_df['SEASON'] >= 1970) & (lon_gt_180_df['SEASON'] < 2000)).sum()),
    '2000_present': int((lon_gt_180_df['SEASON'] >= 2000).sum()),
}

# Usable training examples for movement prediction horizons: 6h, 12h, 24h
# For an observation at time t, is there an observation at t + delta_t in the same cyclone?
print("[*] Computing usable sequence pairs for prediction horizons (6h, 12h, 24h)...")

# We create target shifts. Because time intervals can vary (mostly 3h, some 1h, 2h),
# the rigorous way is to match (SID, DT + pd.Timedelta(hours=H)).
# Let's verify how many pairs match exactly DT + 6h, DT + 12h, DT + 24h:
# We will do an efficient self-merge on SID and DT!
df_keys = df[['SID', 'DT', 'SEASON', 'BASIN', 'TRACK_TYPE', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'DIST2LAND', 'LANDFALL']].copy()
df_keys['DT_STR'] = df_keys['DT'].astype(str)

# Pre-index for quick lookup or merge
# Let's create shifted timestamps
for h in [6, 12, 24]:
    df_keys[f'DT_plus_{h}h'] = df_keys['DT'] + pd.Timedelta(hours=h)

# Match for each horizon
horizon_stats = {}
for h in [6, 12, 24]:
    # Join df_keys with itself on SID and DT == DT_plus_Xh
    merged = pd.merge(
        df_keys,
        df_keys[['SID', 'DT', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'TRACK_TYPE']],
        left_on=['SID', f'DT_plus_{h}h'],
        right_on=['SID', 'DT'],
        suffixes=('', f'_target_{h}h')
    )
    
    # Total valid movement pairs globally
    global_movement_pairs = len(merged)
    # Main track only (both t and t+h are main track)
    main_movement_pairs = int(((merged['TRACK_TYPE'] == 'main') & (merged[f'TRACK_TYPE_target_{h}h'] == 'main')).sum())
    # 1970+
    m_1970 = merged[merged['SEASON'] >= 1970]
    m_1970_main = int(((m_1970['TRACK_TYPE'] == 'main') & (m_1970[f'TRACK_TYPE_target_{h}h'] == 'main')).sum())
    # 1980+
    m_1980 = merged[merged['SEASON'] >= 1980]
    m_1980_main = int(((m_1980['TRACK_TYPE'] == 'main') & (m_1980[f'TRACK_TYPE_target_{h}h'] == 'main')).sum())
    # 2000+
    m_2000 = merged[merged['SEASON'] >= 2000]
    m_2000_main = int(((m_2000['TRACK_TYPE'] == 'main') & (m_2000[f'TRACK_TYPE_target_{h}h'] == 'main')).sum())

    # Intensity pairs (where USA_WIND is valid at both t and t+h)
    usa_intensity_global = int((merged['USA_WIND'].notna() & merged[f'USA_WIND_target_{h}h'].notna()).sum())
    usa_intensity_1980 = int((m_1980['USA_WIND'].notna() & m_1980[f'USA_WIND_target_{h}h'].notna()).sum())
    usa_intensity_2000 = int((m_2000['USA_WIND'].notna() & m_2000[f'USA_WIND_target_{h}h'].notna()).sum())

    # WMO intensity pairs
    wmo_intensity_global = int((merged['WMO_WIND'].notna() & merged[f'WMO_WIND_target_{h}h'].notna()).sum())
    wmo_intensity_1980 = int((m_1980['WMO_WIND'].notna() & m_1980[f'WMO_WIND_target_{h}h'].notna()).sum())
    wmo_intensity_2000 = int((m_2000['WMO_WIND'].notna() & m_2000[f'WMO_WIND_target_{h}h'].notna()).sum())

    horizon_stats[f"{h}h"] = {
        'horizon_hours': h,
        'global_movement_pairs': global_movement_pairs,
        'global_main_movement_pairs': main_movement_pairs,
        'movement_1970_total': len(m_1970),
        'movement_1970_main': m_1970_main,
        'movement_1980_total': len(m_1980),
        'movement_1980_main': m_1980_main,
        'movement_2000_total': len(m_2000),
        'movement_2000_main': m_2000_main,
        'usa_intensity_global': usa_intensity_global,
        'usa_intensity_1980': usa_intensity_1980,
        'usa_intensity_2000': usa_intensity_2000,
        'wmo_intensity_global': wmo_intensity_global,
        'wmo_intensity_1980': wmo_intensity_1980,
        'wmo_intensity_2000': wmo_intensity_2000,
    }

# -------------------------------------------------------------
# 3. MODEL 3: CYCLONE DEVELOPMENT TREND CLASSIFICATION
# -------------------------------------------------------------
print("[*] Auditing Model 3: Cyclone Development Trend Classification (24h wind change)...")

# We analyze 24h pairs using USA_WIND
merged_24h = pd.merge(
    df_keys,
    df_keys[['SID', 'DT', 'USA_WIND', 'WMO_WIND', 'DIST2LAND', 'LANDFALL', 'TRACK_TYPE']],
    left_on=['SID', 'DT_plus_24h'],
    right_on=['SID', 'DT'],
    suffixes=('', '_target_24h')
)

# Filter for valid USA_WIND at t and t+24h
valid_24h_usa = merged_24h[merged_24h['USA_WIND'].notna() & merged_24h['USA_WIND_target_24h'].notna()].copy()
valid_24h_usa['DELTA_V'] = valid_24h_usa['USA_WIND_target_24h'] - valid_24h_usa['USA_WIND']

# Candidate Thresholds evaluation
# Candidate A: Symmetric +/- 5 kts
# Candidate B: Symmetric +/- 10 kts
# Candidate C: Strict zero (+ >0, 0, < -0)
# Candidate D: Symmetric +/- 15 kts
def evaluate_classes(delta_series, t_val):
    if t_val == 0:
        strengthening = (delta_series > 0).sum()
        stable = (delta_series == 0).sum()
        weakening = (delta_series < 0).sum()
    else:
        strengthening = (delta_series >= t_val).sum()
        stable = ((delta_series > -t_val) & (delta_series < t_val)).sum()
        weakening = (delta_series <= -t_val).sum()
    total = len(delta_series)
    return {
        'threshold': f"+/- {t_val} kts" if t_val > 0 else "Strict 0 kts",
        'strengthening_count': int(strengthening),
        'strengthening_pct': round(strengthening / total * 100, 2),
        'stable_count': int(stable),
        'stable_pct': round(stable / total * 100, 2),
        'weakening_count': int(weakening),
        'weakening_pct': round(weakening / total * 100, 2),
        'ratio_max_to_min': round(max(strengthening, stable, weakening) / min(strengthening, stable, weakening), 2) if min(strengthening, stable, weakening) > 0 else None
    }

# Evaluate on full dataset, 1980+, and 2000+
thresholds = [0, 5, 10, 15]
trend_audit = {
    'total_pairs_all_time': len(valid_24h_usa),
    'cyclones_with_at_least_1_pair': int(valid_24h_usa['SID'].nunique()),
    'delta_v_stats': {
        'min': float(valid_24h_usa['DELTA_V'].min()),
        'max': float(valid_24h_usa['DELTA_V'].max()),
        'mean': round(float(valid_24h_usa['DELTA_V'].mean()), 2),
        'median': float(valid_24h_usa['DELTA_V'].median()),
        'std': round(float(valid_24h_usa['DELTA_V'].std()), 2),
        'q25': float(valid_24h_usa['DELTA_V'].quantile(0.25)),
        'q75': float(valid_24h_usa['DELTA_V'].quantile(0.75)),
        'ri_count (>=30 kts)': int((valid_24h_usa['DELTA_V'] >= 30).sum()),
        'ri_pct': round((valid_24h_usa['DELTA_V'] >= 30).mean() * 100, 2),
        'rw_count (<=-30 kts)': int((valid_24h_usa['DELTA_V'] <= -30).sum()),
        'rw_pct': round((valid_24h_usa['DELTA_V'] <= -30).mean() * 100, 2),
    },
    'candidates_global': [evaluate_classes(valid_24h_usa['DELTA_V'], t) for t in thresholds],
    'candidates_1980': [evaluate_classes(valid_24h_usa.loc[valid_24h_usa['SEASON'] >= 1980, 'DELTA_V'], t) for t in thresholds],
    'candidates_2000': [evaluate_classes(valid_24h_usa.loc[valid_24h_usa['SEASON'] >= 2000, 'DELTA_V'], t) for t in thresholds],
}

# Also test on WMO_WIND where available for comparison
valid_24h_wmo = merged_24h[merged_24h['WMO_WIND'].notna() & merged_24h['WMO_WIND_target_24h'].notna()].copy()
valid_24h_wmo['DELTA_V'] = valid_24h_wmo['WMO_WIND_target_24h'] - valid_24h_wmo['WMO_WIND']
trend_audit['wmo_total_pairs'] = len(valid_24h_wmo)
trend_audit['wmo_candidates_1980'] = [evaluate_classes(valid_24h_wmo.loc[valid_24h_wmo['SEASON'] >= 1980, 'DELTA_V'], t) for t in thresholds]

# -------------------------------------------------------------
# 4. HISTORICAL COHORT COMPARISON: 1970+, 1980+, 2000+
# -------------------------------------------------------------
print("[*] Auditing Historical Cohorts (1970+, 1980+, 2000+)...")

cohorts_summary = {}
for min_year in [1970, 1980, 2000]:
    c_df = df[df['SEASON'] >= min_year]
    total_obs = len(c_df)
    unique_sids = c_df['SID'].nunique()
    
    # Basins breakdown
    basin_counts = c_df['BASIN'].value_counts().to_dict()
    basin_sids = c_df.groupby('BASIN')['SID'].nunique().to_dict()
    
    # Wind availability
    usa_w_cnt = int(c_df['USA_WIND'].notna().sum())
    wmo_w_cnt = int(c_df['WMO_WIND'].notna().sum())
    any_w_cnt = int(((c_df['USA_WIND'].notna()) | (c_df['WMO_WIND'].notna())).sum())
    
    # Position availability
    lat_cnt = int(c_df['LAT'].notna().sum())
    lon_cnt = int(c_df['LON'].notna().sum())

    cohorts_summary[f"{min_year}_present"] = {
        'start_year': min_year,
        'total_observations': total_obs,
        'unique_cyclones': unique_sids,
        'observations_by_basin': basin_counts,
        'cyclones_by_basin': basin_sids,
        'usa_wind_count': usa_w_cnt,
        'usa_wind_pct': round(usa_w_cnt / total_obs * 100, 2),
        'wmo_wind_count': wmo_w_cnt,
        'wmo_wind_pct': round(wmo_w_cnt / total_obs * 100, 2),
        'any_wind_count': any_w_cnt,
        'any_wind_pct': round(any_w_cnt / total_obs * 100, 2),
        'lat_availability_pct': round(lat_cnt / total_obs * 100, 2),
        'lon_availability_pct': round(lon_cnt / total_obs * 100, 2),
        'movement_pairs_6h': horizon_stats['6h'][f'movement_{min_year}_total'],
        'movement_pairs_12h': horizon_stats['12h'][f'movement_{min_year}_total'],
        'movement_pairs_24h': horizon_stats['24h'][f'movement_{min_year}_total'],
        'movement_main_pairs_6h': horizon_stats['6h'][f'movement_{min_year}_main'],
        'movement_main_pairs_12h': horizon_stats['12h'][f'movement_{min_year}_main'],
        'movement_main_pairs_24h': horizon_stats['24h'][f'movement_{min_year}_main'],
        'usa_intensity_pairs_6h': horizon_stats['6h'][f'usa_intensity_{min_year}'] if min_year >= 1980 else None,
        'usa_intensity_pairs_12h': horizon_stats['12h'][f'usa_intensity_{min_year}'] if min_year >= 1980 else None,
        'usa_intensity_pairs_24h': horizon_stats['24h'][f'usa_intensity_{min_year}'] if min_year >= 1980 else None,
    }

# Also calculate 1970 intensity pairs for completeness
for h in [6, 12, 24]:
    m_1970 = pd.merge(
        df_keys[df_keys['SEASON'] >= 1970],
        df_keys[df_keys['SEASON'] >= 1970][['SID', 'DT', 'USA_WIND', 'WMO_WIND']],
        left_on=['SID', f'DT_plus_{h}h'],
        right_on=['SID', 'DT']
    )
    cohorts_summary['1970_present'][f'usa_intensity_pairs_{h}h'] = int((m_1970['USA_WIND_x'].notna() & m_1970['USA_WIND_y'].notna()).sum())
    cohorts_summary['1970_present'][f'wmo_intensity_pairs_{h}h'] = int((m_1970['WMO_WIND_x'].notna() & m_1970['WMO_WIND_y'].notna()).sum())

# Compile complete results dictionary
audit_results = {
    'model_1_intensity': {
        'basin_wind_audit': basin_wind_audit,
        'regional_wind_audit': reg_wind_audit,
        'wmo_agency_distribution': wmo_agency_dist,
        'storm_wmo_agency_consistency': storm_wmo_agency_consistency,
        'storm_usa_agency_consistency': storm_usa_agency_consistency,
        'consecutive_usa_wind_runs': consec_usa_summary,
        'consecutive_wmo_wind_runs': consec_wmo_summary,
        'wind_profile_1980': wind_profile_1980,
        'wind_profile_2000': wind_profile_2000,
    },
    'model_2_movement': {
        'track_types': track_types_audit,
        'main_track_cyclones_total': total_main_sids,
        'main_track_obs_total': total_main_obs,
        'cyclones_1970_total': cyclones_1970_total,
        'cyclones_1970_main': cyclones_1970_main,
        'obs_stats_main_1970': obs_stats_main_1970,
        'lon_gt_180': {
            'total_obs': lon_gt_180_total,
            'unique_sids': lon_gt_180_sids,
            'basins': lon_gt_180_basins,
            'eras': lon_gt_180_eras
        },
        'horizon_usable_pairs': horizon_stats
    },
    'model_3_trend': trend_audit,
    'cohort_comparison': cohorts_summary
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)

print(f"[+] Audit data successfully written to {OUT_JSON}.")
