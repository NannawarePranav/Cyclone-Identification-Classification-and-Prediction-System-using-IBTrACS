"""
Cyclone Atlas — Data Loader and Historical Preprocessor
=======================================================
Implements cached, memory-efficient data loading for the processed Parquet
datasets, cyclone track indexing, physical observation deduplication, and
horizon-specific prediction sample extraction.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import streamlit as st

# Add src to sys.path for cyclone_ml imports
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cyclone_ml import compute_cyclical_features


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed')


@st.cache_data(show_spinner="Loading intensity dataset...")
def load_intensity_dataset() -> pd.DataFrame:
    """Loads and caches the clean processed Model 1 intensity dataset."""
    filepath = os.path.join(DATA_DIR, "model1_intensity.parquet")
    if not os.path.exists(filepath):
        st.error(f"Missing required dataset: {filepath}")
        st.stop()
    df = pd.read_parquet(filepath)
    # Ensure numerical types for features
    num_cols = [
        'CURRENT_USA_WIND', 'PAST_USA_WIND_6H', 'PAST_USA_WIND_12H',
        'HAS_PAST_USA_WIND_6H', 'HAS_PAST_USA_WIND_12H',
        'WIND_CHANGE_PAST_6H', 'WIND_CHANGE_PAST_12H',
        'LAT', 'LON_NORM', 'DIST2LAND', 'MONTH', 'HOUR'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float64)
    if 'SPLIT' in df.columns:
        df['SPLIT'] = df['SPLIT'].astype(str).str.upper()
    return compute_cyclical_features(df)


@st.cache_data(show_spinner="Loading movement dataset...")
def load_movement_dataset() -> pd.DataFrame:
    """Loads and caches the clean processed Model 2 movement dataset."""
    filepath = os.path.join(DATA_DIR, "model2_movement.parquet")
    if not os.path.exists(filepath):
        st.error(f"Missing required dataset: {filepath}")
        st.stop()
    df = pd.read_parquet(filepath)
    num_cols = [
        'LAT', 'LON_NORM', 'PAST_EAST_DISP_6H_KM', 'PAST_NORTH_DISP_6H_KM',
        'PAST_EAST_DISP_12H_KM', 'PAST_NORTH_DISP_12H_KM', 'DIST2LAND', 'MONTH', 'HOUR'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float64)
    if 'SPLIT' in df.columns:
        df['SPLIT'] = df['SPLIT'].astype(str).str.upper()
    return compute_cyclical_features(df)


def load_development_dataset() -> pd.DataFrame:
    """Loads and caches the clean processed Model 3 development dataset."""
    enriched = os.path.join(DATA_DIR, "model3_development_enriched.parquet")
    filepath = enriched if os.path.exists(enriched) else os.path.join(DATA_DIR, "model3_development.parquet")
    if not os.path.exists(filepath):
        st.error(f"Missing required dataset: {filepath}")
        st.stop()
    return _load_development_version(filepath, os.stat(filepath).st_mtime_ns)


@st.cache_data(show_spinner="Loading development dataset...")
def _load_development_version(filepath, version):
    df = pd.read_parquet(filepath)
    num_cols = [
        'CURRENT_USA_WIND', 'PAST_USA_WIND_6H', 'PAST_USA_WIND_12H',
        'HAS_PAST_USA_WIND_6H', 'HAS_PAST_USA_WIND_12H',
        'WIND_CHANGE_PAST_6H', 'WIND_CHANGE_PAST_12H',
        'LAT', 'LON_NORM', 'DIST2LAND', 'MONTH', 'HOUR'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float64)
    if 'SPLIT' in df.columns:
        df['SPLIT'] = df['SPLIT'].astype(str).str.upper()
    return compute_cyclical_features(df)


@st.cache_data(show_spinner=False)
def load_cyclone_names() -> Dict[str, str]:
    path = os.path.join(DATA_DIR, 'cyclone_names.csv')
    if not os.path.exists(path):
        return {}
    return pd.read_csv(path, keep_default_na=False).set_index('SID')['NAME'].to_dict()


@st.cache_data(show_spinner=False)
def get_global_summary_stats() -> Dict[str, Any]:
    """Computes global statistical summaries for the Overview page."""
    df_mov = load_movement_dataset()
    df_int = load_intensity_dataset()

    # Deduplicate observations to get actual physical observation count
    dedup_mov = df_mov.drop_duplicates(subset=['SID', 'ISO_TIME'])

    total_storms = int(dedup_mov['SID'].nunique())
    total_obs = int(len(dedup_mov))
    
    # Model 2 covers 1970–present; Model 1 covers 1980–present
    mov_season_min = int(df_mov['SEASON'].min())
    mov_season_max = int(df_mov['SEASON'].max())
    int_season_min = int(df_int['SEASON'].min())
    int_season_max = int(df_int['SEASON'].max())

    basin_counts = dedup_mov['BASIN'].value_counts().to_dict()
    split_counts = dedup_mov['SPLIT'].value_counts().to_dict()

    # Intensity coverage stats
    max_wind_observed = float(df_int['CURRENT_USA_WIND'].max())
    mean_wind_observed = float(df_int['CURRENT_USA_WIND'].mean())

    return {
        'total_storms': total_storms,
        'total_observations': total_obs,
        'season_range': f"{mov_season_min} – {mov_season_max}",
        'movement_season_range': f"{mov_season_min} – {mov_season_max}",
        'intensity_season_range': f"{int_season_min} – {int_season_max}",
        'basin_distribution': basin_counts,
        'split_distribution': split_counts,
        'max_wind_kts': max_wind_observed,
        'mean_wind_kts': round(mean_wind_observed, 1)
    }


@st.cache_data(show_spinner=False)
def get_cyclone_catalog(dataset_type: str = 'movement') -> pd.DataFrame:
    """
    Builds a catalog of unique cyclones with season, basin, physical observation counts,
    and temporal span for fast user selection.
    """
    if dataset_type == 'intensity':
        df = load_intensity_dataset()
    elif dataset_type == 'development':
        df = load_development_dataset()
    else:
        df = load_movement_dataset()

    # Deduplicate by SID and ISO_TIME to get physical records
    dedup = df.drop_duplicates(subset=['SID', 'ISO_TIME'])

    # Aggregate by SID
    agg_dict = {
        'SEASON': 'first',
        'BASIN': 'first',
        'NATURE': 'first',
        'ISO_TIME': ['min', 'max', 'count'],
        'SPLIT': 'first'
    }
    if 'CURRENT_USA_WIND' in dedup.columns:
        agg_dict['CURRENT_USA_WIND'] = 'max'

    catalog = dedup.groupby('SID').agg(agg_dict)
    catalog.columns = ['_'.join(c).strip('_') for c in catalog.columns]

    catalog = catalog.reset_index()
    rename_cols = {
        'SEASON_first': 'SEASON',
        'BASIN_first': 'BASIN',
        'NATURE_first': 'NATURE',
        'ISO_TIME_min': 'START_TIME',
        'ISO_TIME_max': 'END_TIME',
        'ISO_TIME_count': 'OBS_COUNT',
        'SPLIT_first': 'SPLIT'
    }
    if 'CURRENT_USA_WIND_max' in catalog.columns:
        rename_cols['CURRENT_USA_WIND_max'] = 'PEAK_WIND_KTS'

    catalog = catalog.rename(columns=rename_cols)
    catalog['NAME'] = catalog.SID.map(load_cyclone_names()).fillna('Unnamed cyclone')
    catalog['SEASON'] = catalog['SEASON'].astype(int)
    catalog = catalog.sort_values(by=['SEASON', 'START_TIME'], ascending=[False, False])
    return catalog


def get_physical_track_observations(sid: str, dataset_type: str = 'movement') -> pd.DataFrame:
    """
    Returns strictly unique physical track observations for an SID from the specified dataset,
    sorted chronologically by ISO_TIME. Exactly one record per (SID, ISO_TIME).
    """
    if dataset_type == 'intensity':
        df = load_intensity_dataset()
    elif dataset_type == 'development':
        df = load_development_dataset()
    else:
        df = load_movement_dataset()

    cyclone_df = df[df['SID'] == sid].copy()
    cyclone_df['ISO_TIME'] = pd.to_datetime(cyclone_df['ISO_TIME'])

    # Deduplicate by (SID, ISO_TIME) to isolate physical track observations
    physical_df = cyclone_df.drop_duplicates(subset=['SID', 'ISO_TIME']).sort_values('ISO_TIME').reset_index(drop=True)

    # Validate that no duplicate (SID, ISO_TIME) records exist
    assert not physical_df.duplicated(subset=['SID', 'ISO_TIME']).any(), (
        f"Duplicate (SID, ISO_TIME) physical observation sequence detected for SID: {sid}"
    )
    return physical_df


def get_observations_for_cyclone(sid: str, dataset_type: str = 'movement') -> pd.DataFrame:
    """
    Convenience wrapper returning unique physical track observations for an SID.
    Guarantees no repeated horizon-expanded samples.
    """
    return get_physical_track_observations(sid, dataset_type=dataset_type)


def get_horizon_targets_for_observation(
    sid: str,
    iso_time: Any,
    dataset_type: str = 'intensity'
) -> Dict[int, Dict[str, Any]]:
    """
    Retrieves the actual future ground-truth target samples for each horizon (+6h, +12h, +24h)
    corresponding to a specific physical observation (SID, ISO_TIME).
    
    Returns:
        Dict mapping horizon (int: 6, 12, 24) to a dictionary of target fields.
        If a horizon target is unavailable (e.g. beyond storm lifecycle), it is omitted.
    """
    if dataset_type == 'intensity':
        df = load_intensity_dataset()
    elif dataset_type == 'development':
        df = load_development_dataset()
    else:
        df = load_movement_dataset()

    target_time = pd.to_datetime(iso_time)

    # Match exact physical observation
    matches = df[(df['SID'] == sid) & (pd.to_datetime(df['ISO_TIME']) == target_time)]
    targets_by_horizon = {}
    if matches.empty:
        return targets_by_horizon

    if dataset_type == 'intensity':
        for _, row in matches.iterrows():
            if 'PRED_HORIZON_HOURS' in row and pd.notna(row['PRED_HORIZON_HOURS']):
                h = int(row['PRED_HORIZON_HOURS'])
                target_w = row.get('TARGET_USA_WIND', None)
                if pd.notna(target_w):
                    targets_by_horizon[h] = {
                        'target_wind_kts': float(target_w),
                        'target_wind_kmh': round(float(target_w) * 1.852, 1),
                        'horizon_hours': h
                    }
    elif dataset_type == 'movement':
        for _, row in matches.iterrows():
            if 'PRED_HORIZON_HOURS' in row and pd.notna(row['PRED_HORIZON_HOURS']):
                h = int(row['PRED_HORIZON_HOURS'])
                t_lat = row.get('TARGET_LAT', None)
                t_lon = row.get('TARGET_LON_NORM', row.get('TARGET_LON', None))
                dx = row.get('TARGET_EAST_DISPLACEMENT_KM', None)
                dy = row.get('TARGET_NORTH_DISPLACEMENT_KM', None)
                hav_d = row.get('TARGET_HAVERSINE_DISTANCE_KM', None)

                if pd.notna(t_lat) and pd.notna(t_lon):
                    targets_by_horizon[h] = {
                        'target_lat': float(t_lat),
                        'target_lon': float(t_lon),
                        'target_dx_km': float(dx) if pd.notna(dx) else None,
                        'target_dy_km': float(dy) if pd.notna(dy) else None,
                        'target_haversine_distance_km': float(hav_d) if pd.notna(hav_d) else None,
                        'horizon_hours': h
                    }
    elif dataset_type == 'development':
        row = matches.iloc[0]
        targets_by_horizon[24] = {
            'target_class': row.get('CLASS_LABEL', None),
            'target_wind_change_24h': row.get('WIND_CHANGE_24H', None),
            'target_wind_24h': row.get('TARGET_USA_WIND_24H', None),
            'horizon_hours': 24
        }

    return targets_by_horizon
