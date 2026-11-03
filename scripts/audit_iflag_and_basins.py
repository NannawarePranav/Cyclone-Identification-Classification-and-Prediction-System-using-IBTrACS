import os
import json
import time
import pandas as pd
import numpy as np

CSV_PATH = "raw/ibtracs.ALL.list.v04r01.csv"
OUT_JSON = "reports/iflag_basin_audit_data.json"

print(f"[*] Loading IBTrACS CSV from {CSV_PATH}...")
start_time = time.time()

df = pd.read_csv(
    CSV_PATH,
    skiprows=[1],
    na_values=["", " "],
    keep_default_na=False,
    low_memory=False
)
print(f"[+] Loaded {len(df):,} rows in {time.time() - start_time:.2f}s")

# Numeric conversion
for col in ['SEASON', 'LAT', 'LON', 'USA_LAT', 'USA_LON', 'USA_WIND', 'USA_PRES', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND', 'LANDFALL', 'STORM_SPEED', 'STORM_DIR']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Parse datetime
df['DT'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df['HOUR'] = df['DT'].dt.hour
df['MINUTE'] = df['DT'].dt.minute

# Sort chronologically by storm and timestamp
df = df.sort_values(by=['SID', 'DT']).reset_index(drop=True)

# -------------------------------------------------------------
# PART A: IFLAG ANALYSIS
# -------------------------------------------------------------
print("[*] Part A: Analyzing IFLAG field...")

# NOAA IBTrACS agency mapping for each of the 15 characters
agency_names = [
    "USA", "JMA_Tokyo", "CMA_China", "HKO_HongKong", "KMA_Korea",
    "NewDelhi_IMD", "LaReunion_France", "BoM_Australia", "Nadi_Fiji", "Wellington_NZ",
    "TD9636_Historical", "TD9635_Historical", "DS824_Historical", "Neumann_Historical", "Chenoweth_MLC"
]

# Character distribution for each agency position
agency_flag_dist = {}
for i, agency in enumerate(agency_names):
    chars = df['IFLAG'].str[i].value_counts().to_dict()
    agency_flag_dist[agency] = {str(k): int(v) for k, v in chars.items()}

# First character (USA) distribution
usa_flag_dist = agency_flag_dist["USA"]

# Number of USA_WIND values with original vs interpolated intensity
usa_wind_notna = df[df['USA_WIND'].notna()]
usa_wind_flag_counts = usa_wind_notna['IFLAG'].str[0].value_counts().to_dict()

# Flag definitions for USA_WIND:
# O, V = Original intensity
# P, I = Interpolated intensity
orig_usa_wind_mask = df['USA_WIND'].notna() & df['IFLAG'].str[0].isin(['O', 'V'])
interp_usa_wind_mask = df['USA_WIND'].notna() & df['IFLAG'].str[0].isin(['P', 'I'])

# Position analysis for USA_LAT / USA_LON
# For position: O, I, V = Original position; P = Interpolated position
usa_lat_notna = df[df['USA_LAT'].notna()]
usa_pos_flag_counts = usa_lat_notna['IFLAG'].str[0].value_counts().to_dict()
orig_usa_pos_mask = df['USA_LAT'].notna() & df['IFLAG'].str[0].isin(['O', 'I', 'V'])
interp_usa_pos_mask = df['USA_LAT'].notna() & df['IFLAG'].str[0].isin(['P'])

# Percentage of observations marked as interpolated in each historical period
print("[*] Vectorized evaluation of IFLAG characters...")
def get_interp_stats(sub_df, name):
    n = len(sub_df)
    n_usa_w = sub_df['USA_WIND'].notna().sum()
    n_usa_orig_w = (sub_df['USA_WIND'].notna() & sub_df['IFLAG'].str[0].isin(['O', 'V'])).sum()
    n_usa_interp_w = (sub_df['USA_WIND'].notna() & sub_df['IFLAG'].str[0].isin(['P', 'I'])).sum()
    
    n_usa_p = sub_df['USA_LAT'].notna().sum()
    n_usa_orig_p = (sub_df['USA_LAT'].notna() & sub_df['IFLAG'].str[0].isin(['O', 'I', 'V'])).sum()
    n_usa_interp_p = (sub_df['USA_LAT'].notna() & sub_df['IFLAG'].str[0].isin(['P'])).sum()

    has_any_orig_pos = sub_df['IFLAG'].apply(lambda s: any(c in ('O', 'I', 'V') for c in s))
    
    return {
        'period_name': name,
        'total_obs': int(n),
        'usa_wind_total': int(n_usa_w),
        'usa_wind_original': int(n_usa_orig_w),
        'usa_wind_original_pct': round(n_usa_orig_w / n_usa_w * 100, 2) if n_usa_w > 0 else 0,
        'usa_wind_interpolated': int(n_usa_interp_w),
        'usa_wind_interpolated_pct': round(n_usa_interp_w / n_usa_w * 100, 2) if n_usa_w > 0 else 0,
        'usa_pos_total': int(n_usa_p),
        'usa_pos_original': int(n_usa_orig_p),
        'usa_pos_original_pct': round(n_usa_orig_p / n_usa_p * 100, 2) if n_usa_p > 0 else 0,
        'usa_pos_interpolated': int(n_usa_interp_p),
        'usa_pos_interpolated_pct': round(n_usa_interp_p / n_usa_p * 100, 2) if n_usa_p > 0 else 0,
        'global_pos_any_orig': int(has_any_orig_pos.sum()),
        'global_pos_any_orig_pct': round(has_any_orig_pos.mean() * 100, 2),
        'global_pos_all_interp': int((~has_any_orig_pos).sum()),
        'global_pos_all_interp_pct': round((~has_any_orig_pos).mean() * 100, 2),
    }

periods_interp_summary = {
    'all_time': get_interp_stats(df, "All-Time (1842-Present)"),
    '1970_present': get_interp_stats(df[df['SEASON'] >= 1970], "1970-Present"),
    '1980_present': get_interp_stats(df[df['SEASON'] >= 1980], "1980-Present"),
    '2000_present': get_interp_stats(df[df['SEASON'] >= 2000], "2000-Present"),
}

# -------------------------------------------------------------
# PART B: MODEL 1 & MODEL 3 REASSESSMENT (1980-PRESENT)
# -------------------------------------------------------------
print("[*] Part B: Reassessing Model 1 & Model 3 for 1980-Present with Original Winds...")

df_1980 = df[df['SEASON'] >= 1980].copy()
df_1980['USA_WIND_ORIG'] = df_1980['USA_WIND'].where(df_1980['IFLAG'].str[0].isin(['O', 'V']), np.nan)
df_1980['USA_POS_ORIG'] = df_1980['LAT'].where(df_1980['IFLAG'].str[0].isin(['O', 'I', 'V']), np.nan)

total_1980_obs = len(df_1980)
total_1980_usa_wind = int(df_1980['USA_WIND'].notna().sum())
orig_1980_usa_wind = int(df_1980['USA_WIND_ORIG'].notna().sum())

model1_pairs = {}
df_1980_keys = df_1980[['SID', 'DT', 'USA_AGENCY', 'USA_WIND', 'USA_WIND_ORIG', 'BASIN', 'DIST2LAND', 'LANDFALL']].copy()

for h in [6, 12, 24]:
    df_1980_keys[f'DT_plus_{h}h'] = df_1980_keys['DT'] + pd.Timedelta(hours=h)

for h in [6, 12, 24]:
    merged_h = pd.merge(
        df_1980_keys,
        df_1980_keys[['SID', 'DT', 'USA_AGENCY', 'USA_WIND', 'USA_WIND_ORIG']],
        left_on=['SID', f'DT_plus_{h}h'],
        right_on=['SID', 'DT'],
        suffixes=('', f'_target_{h}h')
    )
    
    any_wind_pairs = len(merged_h[merged_h['USA_WIND'].notna() & merged_h[f'USA_WIND_target_{h}h'].notna()])
    orig_wind_pairs = merged_h[merged_h['USA_WIND_ORIG'].notna() & merged_h[f'USA_WIND_ORIG_target_{h}h'].notna()]
    n_orig_pairs = len(orig_wind_pairs)
    same_agency_orig_pairs = orig_wind_pairs[orig_wind_pairs['USA_AGENCY'] == orig_wind_pairs[f'USA_AGENCY_target_{h}h']]
    n_same_agency_pairs = len(same_agency_orig_pairs)
    sids_with_orig_pairs = same_agency_orig_pairs['SID'].nunique()

    model1_pairs[f"{h}h"] = {
        'horizon_hours': h,
        'any_usa_wind_pairs': int(any_wind_pairs),
        'orig_usa_wind_pairs': int(n_orig_pairs),
        'same_agency_orig_pairs': int(n_same_agency_pairs),
        'unique_cyclones_with_pairs': int(sids_with_orig_pairs),
    }

orig_wind_per_storm = df_1980.groupby('SID')['USA_WIND_ORIG'].apply(lambda s: s.notna().sum())
cyclones_orig_wind_summary = {
    'total_cyclones_1980': int(df_1980['SID'].nunique()),
    'cyclones_with_ge_1_orig_wind': int((orig_wind_per_storm >= 1).sum()),
    'cyclones_with_ge_4_orig_wind (12h)': int((orig_wind_per_storm >= 4).sum()),
    'cyclones_with_ge_8_orig_wind (24h)': int((orig_wind_per_storm >= 8).sum()),
    'cyclones_with_ge_12_orig_wind (36h)': int((orig_wind_per_storm >= 12).sum()),
    'cyclones_with_ge_16_orig_wind (48h)': int((orig_wind_per_storm >= 16).sum()),
    'cyclones_with_ge_24_orig_wind (72h)': int((orig_wind_per_storm >= 24).sum()),
    'median_orig_winds_per_storm': float(orig_wind_per_storm.median()),
    'mean_orig_winds_per_storm': round(float(orig_wind_per_storm.mean()), 2),
}

# Model 3: Trend classification on Original-Only pairs from same USA agency
print("[*] Recalculating Model 3 Trend classes on original-only pairs...")
pairs_24h_orig = pd.merge(
    df_1980_keys[df_1980_keys['USA_WIND_ORIG'].notna()],
    df_1980_keys[df_1980_keys['USA_WIND_ORIG'].notna()][['SID', 'DT', 'USA_AGENCY', 'USA_WIND_ORIG', 'DIST2LAND', 'LANDFALL']],
    left_on=['SID', 'DT_plus_24h'],
    right_on=['SID', 'DT'],
    suffixes=('', '_target_24h')
)
pairs_24h_orig_same_agency = pairs_24h_orig[pairs_24h_orig['USA_AGENCY'] == pairs_24h_orig['USA_AGENCY_target_24h']].copy()
pairs_24h_orig_same_agency['DELTA_V'] = pairs_24h_orig_same_agency['USA_WIND_ORIG_target_24h'] - pairs_24h_orig_same_agency['USA_WIND_ORIG']

total_orig_24h_pairs = len(pairs_24h_orig_same_agency)
delta_v_orig = pairs_24h_orig_same_agency['DELTA_V']

def classify_delta_v(series, t_val=10):
    strengthening = (series >= t_val).sum()
    stable = ((series > -t_val) & (series < t_val)).sum()
    weakening = (series <= -t_val).sum()
    total = len(series)
    return {
        'threshold': f"+/- {t_val} kts",
        'total_pairs': int(total),
        'strengthening_count': int(strengthening),
        'strengthening_pct': round(strengthening / total * 100, 2),
        'stable_count': int(stable),
        'stable_pct': round(stable / total * 100, 2),
        'weakening_count': int(weakening),
        'weakening_pct': round(weakening / total * 100, 2),
        'ratio_max_to_min': round(max(strengthening, stable, weakening) / min(strengthening, stable, weakening), 2)
    }

trend_classes_orig_10kt = classify_delta_v(delta_v_orig, 10)
trend_classes_orig_5kt = classify_delta_v(delta_v_orig, 5)
trend_classes_orig_0kt = {
    'threshold': 'Strict 0 kts',
    'total_pairs': int(len(delta_v_orig)),
    'strengthening_count': int((delta_v_orig > 0).sum()),
    'strengthening_pct': round((delta_v_orig > 0).mean() * 100, 2),
    'stable_count': int((delta_v_orig == 0).sum()),
    'stable_pct': round((delta_v_orig == 0).mean() * 100, 2),
    'weakening_count': int((delta_v_orig < 0).sum()),
    'weakening_pct': round((delta_v_orig < 0).mean() * 100, 2),
    'ratio_max_to_min': round(max((delta_v_orig > 0).sum(), (delta_v_orig == 0).sum(), (delta_v_orig < 0).sum()) / min((delta_v_orig > 0).sum(), (delta_v_orig == 0).sum(), (delta_v_orig < 0).sum()), 2)
}

# -------------------------------------------------------------
# PART C: MODEL 2 REASSESSMENT (1970-PRESENT MAIN-TRACK)
# -------------------------------------------------------------
print("[*] Part C: Reassessing Model 2 Movement for 1970-Present Main-Track...")

df_1970_main = df[(df['SEASON'] >= 1970) & (df['TRACK_TYPE'] == 'main')].copy()
n_1970_main_obs = len(df_1970_main)
n_1970_main_sids = df_1970_main['SID'].nunique()

df_1970_main['USA_POS_IS_ORIG'] = df_1970_main['IFLAG'].str[0].isin(['O', 'I', 'V'])
df_1970_main['GLOBAL_POS_IS_ORIG'] = df_1970_main['IFLAG'].apply(lambda s: any(c in ('O', 'I', 'V') for c in s))

usa_pos_orig_cnt = int(df_1970_main['USA_POS_IS_ORIG'].sum())
global_pos_orig_cnt = int(df_1970_main['GLOBAL_POS_IS_ORIG'].sum())

is_synoptic_6h = df_1970_main['HOUR'].isin([0, 6, 12, 18]) & (df_1970_main['MINUTE'] == 0)
is_synoptic_3h = df_1970_main['HOUR'].isin([3, 9, 15, 21]) & (df_1970_main['MINUTE'] == 0)
is_off_synoptic = ~is_synoptic_6h & ~is_synoptic_3h

synoptic_breakdown = {
    'synoptic_6h': {
        'count': int(is_synoptic_6h.sum()),
        'pct_of_total': round(is_synoptic_6h.mean() * 100, 2),
        'global_pos_orig_count': int((is_synoptic_6h & df_1970_main['GLOBAL_POS_IS_ORIG']).sum()),
        'global_pos_orig_pct': round((is_synoptic_6h & df_1970_main['GLOBAL_POS_IS_ORIG']).sum() / is_synoptic_6h.sum() * 100, 2),
        'usa_pos_orig_count': int((is_synoptic_6h & df_1970_main['USA_POS_IS_ORIG']).sum()),
        'usa_pos_orig_pct': round((is_synoptic_6h & df_1970_main['USA_POS_IS_ORIG']).sum() / is_synoptic_6h.sum() * 100, 2),
    },
    'sub_synoptic_3h': {
        'count': int(is_synoptic_3h.sum()),
        'pct_of_total': round(is_synoptic_3h.mean() * 100, 2),
        'global_pos_orig_count': int((is_synoptic_3h & df_1970_main['GLOBAL_POS_IS_ORIG']).sum()),
        'global_pos_orig_pct': round((is_synoptic_3h & df_1970_main['GLOBAL_POS_IS_ORIG']).sum() / is_synoptic_3h.sum() * 100, 2),
        'usa_pos_orig_count': int((is_synoptic_3h & df_1970_main['USA_POS_IS_ORIG']).sum()),
        'usa_pos_orig_pct': round((is_synoptic_3h & df_1970_main['USA_POS_IS_ORIG']).sum() / is_synoptic_3h.sum() * 100, 2),
    },
    'off_synoptic_special': {
        'count': int(is_off_synoptic.sum()),
        'pct_of_total': round(is_off_synoptic.mean() * 100, 2),
        'global_pos_orig_count': int((is_off_synoptic & df_1970_main['GLOBAL_POS_IS_ORIG']).sum()),
        'global_pos_orig_pct': round((is_off_synoptic & df_1970_main['GLOBAL_POS_IS_ORIG']).sum() / is_off_synoptic.sum() * 100, 2),
        'usa_pos_orig_count': int((is_off_synoptic & df_1970_main['USA_POS_IS_ORIG']).sum()),
        'usa_pos_orig_pct': round((is_off_synoptic & df_1970_main['USA_POS_IS_ORIG']).sum() / is_off_synoptic.sum() * 100, 2),
    }
}

df_1970_main_keys = df_1970_main[['SID', 'DT', 'LAT', 'LON', 'USA_POS_IS_ORIG', 'GLOBAL_POS_IS_ORIG']].copy()
df_1970_main_keys['IS_SYNOPTIC_6H'] = is_synoptic_6h

for h in [6, 12, 24]:
    df_1970_main_keys[f'DT_plus_{h}h'] = df_1970_main_keys['DT'] + pd.Timedelta(hours=h)

movement_reassessment = {}
for h in [6, 12, 24]:
    m_h = pd.merge(
        df_1970_main_keys,
        df_1970_main_keys[['SID', 'DT', 'LAT', 'LON', 'USA_POS_IS_ORIG', 'GLOBAL_POS_IS_ORIG', 'IS_SYNOPTIC_6H']],
        left_on=['SID', f'DT_plus_{h}h'],
        right_on=['SID', 'DT'],
        suffixes=('', f'_target_{h}h')
    )
    total_pairs = len(m_h)
    orig_global_pairs = len(m_h[m_h['GLOBAL_POS_IS_ORIG'] & m_h[f'GLOBAL_POS_IS_ORIG_target_{h}h']])
    orig_usa_pairs = len(m_h[m_h['USA_POS_IS_ORIG'] & m_h[f'USA_POS_IS_ORIG_target_{h}h']])
    synoptic_6h_pairs = len(m_h[m_h['IS_SYNOPTIC_6H'] & m_h[f'IS_SYNOPTIC_6H_target_{h}h']])

    movement_reassessment[f"{h}h"] = {
        'horizon_hours': h,
        'total_main_pairs': total_pairs,
        'orig_global_pos_pairs': int(orig_global_pairs),
        'orig_global_pos_pairs_pct': round(orig_global_pairs / total_pairs * 100, 2),
        'orig_usa_pos_pairs': int(orig_usa_pairs),
        'orig_usa_pos_pairs_pct': round(orig_usa_pairs / total_pairs * 100, 2),
        'synoptic_6h_pairs': int(synoptic_6h_pairs),
        'synoptic_6h_pairs_pct': round(synoptic_6h_pairs / total_pairs * 100, 2),
    }

# -------------------------------------------------------------
# PART D: GEOGRAPHICAL BASIN REPRESENTATION (1980-PRESENT)
# -------------------------------------------------------------
print("[*] Part D: Auditing geographical representation post-1980 by basin...")

basins_list = ['WP', 'SI', 'NA', 'SP', 'EP', 'NI', 'SA']
basin_rep_audit = {}

for b in basins_list:
    b_df = df_1980[df_1980['BASIN'] == b]
    n_obs = len(b_df)
    n_sids = b_df['SID'].nunique()
    
    usa_w_cnt = int(b_df['USA_WIND'].notna().sum())
    usa_w_pct = round(usa_w_cnt / n_obs * 100, 2) if n_obs > 0 else 0
    
    usa_orig_w_cnt = int((b_df['USA_WIND'].notna() & b_df['IFLAG'].str[0].isin(['O', 'V'])).sum())
    usa_orig_w_pct = round(usa_orig_w_cnt / n_obs * 100, 2) if n_obs > 0 else 0
    usa_orig_of_avail_pct = round(usa_orig_w_cnt / usa_w_cnt * 100, 2) if usa_w_cnt > 0 else 0

    b_keys = df_1980_keys[df_1980_keys['BASIN'] == b]
    b_24h_merged = pd.merge(
        b_keys[b_keys['USA_WIND_ORIG'].notna()],
        b_keys[b_keys['USA_WIND_ORIG'].notna()][['SID', 'DT', 'USA_AGENCY', 'USA_WIND_ORIG']],
        left_on=['SID', 'DT_plus_24h'],
        right_on=['SID', 'DT'],
        suffixes=('', '_target')
    )
    b_24h_same = b_24h_merged[b_24h_merged['USA_AGENCY'] == b_24h_merged['USA_AGENCY_target']]
    
    cyclones_with_ge_1_pair = int(b_24h_same['SID'].nunique())
    
    pairs_per_cyclone = b_24h_same.groupby('SID').size()
    cyclones_with_ge_4_pairs = int((pairs_per_cyclone >= 4).sum())
    cyclones_with_ge_8_pairs = int((pairs_per_cyclone >= 8).sum())
    
    total_24h_orig_pairs = len(b_24h_same)

    b_main_keys = df_1970_main_keys[df_1970_main['BASIN'] == b]
    b_mov_24h = pd.merge(
        b_main_keys,
        b_main_keys[['SID', 'DT', 'LAT', 'LON', 'GLOBAL_POS_IS_ORIG']],
        left_on=['SID', 'DT_plus_24h'],
        right_on=['SID', 'DT']
    )
    mov_pairs_24h = len(b_mov_24h)
    mov_sids_24h = int(b_mov_24h['SID'].nunique())

    basin_rep_audit[b] = {
        'basin_code': b,
        'total_observations': n_obs,
        'unique_cyclones': n_sids,
        'usa_wind_available_count': usa_w_cnt,
        'usa_wind_available_pct': usa_w_pct,
        'usa_wind_original_count': usa_orig_w_cnt,
        'usa_wind_original_pct_of_total': usa_orig_w_pct,
        'usa_wind_original_pct_of_available': usa_orig_of_avail_pct,
        'total_24h_orig_pairs': total_24h_orig_pairs,
        'cyclones_with_ge_1_orig_24h_pair': cyclones_with_ge_1_pair,
        'cyclones_with_ge_4_orig_24h_pairs': cyclones_with_ge_4_pairs,
        'cyclones_with_ge_8_orig_24h_pairs': cyclones_with_ge_8_pairs,
        'movement_24h_pairs_1970': mov_pairs_24h,
        'movement_24h_cyclones_1970': mov_sids_24h,
    }

results = {
    'part_a_iflag': {
        'agency_flag_distribution': agency_flag_dist,
        'usa_flag_distribution': usa_flag_dist,
        'usa_wind_flags': {str(k): int(v) for k, v in usa_wind_flag_counts.items()},
        'usa_pos_flags': {str(k): int(v) for k, v in usa_pos_flag_counts.items()},
        'periods_interpolation_stats': periods_interp_summary
    },
    'part_b_reassessment_intensity_trend': {
        'total_1980_obs': total_1980_obs,
        'available_usa_wind': total_1980_usa_wind,
        'original_usa_wind': orig_1980_usa_wind,
        'original_usa_wind_pct': round(orig_1980_usa_wind / total_1980_usa_wind * 100, 2),
        'model1_prediction_pairs': model1_pairs,
        'cyclones_orig_wind_summary': cyclones_orig_wind_summary,
        'model3_trend_recalculated': {
            'candidate_10kt': trend_classes_orig_10kt,
            'candidate_5kt': trend_classes_orig_5kt,
            'candidate_0kt': trend_classes_orig_0kt
        }
    },
    'part_c_reassessment_movement': {
        'total_1970_main_obs': n_1970_main_obs,
        'total_1970_main_cyclones': n_1970_main_sids,
        'usa_pos_original_count': usa_pos_orig_cnt,
        'usa_pos_original_pct': round(usa_pos_orig_cnt / n_1970_main_obs * 100, 2),
        'global_pos_original_count': global_pos_orig_cnt,
        'global_pos_original_pct': round(global_pos_orig_cnt / n_1970_main_obs * 100, 2),
        'synoptic_breakdown': synoptic_breakdown,
        'movement_reassessment_pairs': movement_reassessment
    },
    'part_d_geographical_representation': basin_rep_audit
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[+] Task 3 audit data written to {OUT_JSON}.")
