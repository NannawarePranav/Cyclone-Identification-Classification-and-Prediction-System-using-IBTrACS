"""
Cyclone Atlas — Data Loader and Historical Preprocessor
=======================================================
Implements cached, memory-efficient data loading for the processed Parquet
datasets, cyclone track indexing, and chronological trajectory extraction.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
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


@st.cache_data(show_spinner="Loading development dataset...")
def load_development_dataset() -> pd.DataFrame:
    """Loads and caches the clean processed Model 3 development dataset."""
    filepath = os.path.join(DATA_DIR, "model3_development.parquet")
    if not os.path.exists(filepath):
        st.error(f"Missing required dataset: {filepath}")
        st.stop()
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
def get_global_summary_stats() -> Dict[str, Any]:
    """Computes global statistical summaries for the Overview page."""
    df_mov = load_movement_dataset()
    df_int = load_intensity_dataset()

    # Deduplicate observations to get actual physical observation count
    dedup_mov = df_mov.drop_duplicates(subset=['SID', 'ISO_TIME'])

    total_storms = int(dedup_mov['SID'].nunique())
    total_obs = int(len(dedup_mov))
    season_min = int(dedup_mov['SEASON'].min())
    season_max = int(dedup_mov['SEASON'].max())

    basin_counts = dedup_mov['BASIN'].value_counts().to_dict()
    split_counts = dedup_mov['SPLIT'].value_counts().to_dict()

    # Intensity coverage stats
    max_wind_observed = float(df_int['CURRENT_USA_WIND'].max())
    mean_wind_observed = float(df_int['CURRENT_USA_WIND'].mean())

    return {
        'total_storms': total_storms,
        'total_observations': total_obs,
        'season_range': f"{season_min} – {season_max}",
        'basin_distribution': basin_counts,
        'split_distribution': split_counts,
        'max_wind_kts': max_wind_observed,
        'mean_wind_kts': round(mean_wind_observed, 1)
    }


@st.cache_data(show_spinner=False)
def get_cyclone_catalog(dataset_type: str = 'movement') -> pd.DataFrame:
    """
    Builds a catalog of unique cyclones with season, basin, observation counts,
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
    catalog['SEASON'] = catalog['SEASON'].astype(int)
    catalog = catalog.sort_values(by=['SEASON', 'START_TIME'], ascending=[False, False])
    return catalog


def get_observations_for_cyclone(sid: str, dataset_type: str = 'movement') -> pd.DataFrame:
    """Returns chronological observations for an SID from the specified dataset."""
    if dataset_type == 'intensity':
        df = load_intensity_dataset()
    elif dataset_type == 'development':
        df = load_development_dataset()
    else:
        df = load_movement_dataset()

    cyclone_df = df[df['SID'] == sid].copy()
    cyclone_df['ISO_TIME'] = pd.to_datetime(cyclone_df['ISO_TIME'])
    return cyclone_df.sort_values('ISO_TIME')
