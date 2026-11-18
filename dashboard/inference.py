"""
Cyclone Atlas — Machine Learning Inference Service
==================================================
Decoupled inference engine that accepts raw observation Series/DataFrames,
validates feature schemas, invokes the loaded XGBoost pipelines, and performs
geodetic coordinate reconstruction.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from dashboard.model_loader import (
    load_intensity_model,
    load_movement_model,
    load_development_model
)
from cyclone_ml import (
    reconstruct_coordinates,
    calculate_haversine_error_km,
    compute_cyclical_features
)

# Feature Specifications
M1_NUM_FEATURES = [
    'CURRENT_USA_WIND', 'PAST_USA_WIND_6H', 'PAST_USA_WIND_12H',
    'HAS_PAST_USA_WIND_6H', 'HAS_PAST_USA_WIND_12H',
    'WIND_CHANGE_PAST_6H', 'WIND_CHANGE_PAST_12H',
    'LAT', 'LON_NORM', 'DIST2LAND',
    'MONTH_SIN', 'MONTH_COS', 'HOUR_SIN', 'HOUR_COS'
]
M1_CAT_FEATURES = ['BASIN', 'NATURE']

M2_NUM_FEATURES = [
    'LAT', 'LON_NORM',
    'PAST_EAST_DISP_6H_KM', 'PAST_NORTH_DISP_6H_KM',
    'PAST_EAST_DISP_12H_KM', 'PAST_NORTH_DISP_12H_KM',
    'DIST2LAND',
    'MONTH_SIN', 'MONTH_COS', 'HOUR_SIN', 'HOUR_COS'
]
M2_CAT_FEATURES = ['BASIN', 'NATURE']

M3_NUM_FEATURES = M1_NUM_FEATURES
M3_CAT_FEATURES = M1_CAT_FEATURES


def _prepare_single_row_df(obs: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Converts a Series or single-row DataFrame and ensures cyclical features."""
    if isinstance(obs, pd.Series):
        df = pd.DataFrame([obs])
    else:
        df = obs.iloc[[0]].copy()
    return compute_cyclical_features(df)


def run_intensity_forecast(obs: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Executes intensity forecast across +6h, +12h, and +24h horizons.

    Returns:
        Dict containing current wind and predicted winds in knots and km/h.
    """
    row_df = _prepare_single_row_df(obs)
    for col in M1_NUM_FEATURES:
        if col in row_df.columns:
            row_df[col] = row_df[col].astype(np.float64)

    X_input = row_df[M1_NUM_FEATURES + M1_CAT_FEATURES]
    current_wind = float(row_df['CURRENT_USA_WIND'].values[0])

    m6 = load_intensity_model(6)
    m12 = load_intensity_model(12)
    m24 = load_intensity_model(24)

    p6 = float(m6.predict(X_input)[0])
    p12 = float(m12.predict(X_input)[0])
    p24 = float(m24.predict(X_input)[0])

    # Convert to km/h (1 knot = 1.852 km/h)
    return {
        'current_kts': round(current_wind, 1),
        'current_kmh': round(current_wind * 1.852, 1),
        'predictions_kts': {
            6: round(p6, 1),
            12: round(p12, 1),
            24: round(p24, 1)
        },
        'predictions_kmh': {
            6: round(p6 * 1.852, 1),
            12: round(p12 * 1.852, 1),
            24: round(p24 * 1.852, 1)
        },
        'delta_kts': {
            6: round(p6 - current_wind, 1),
            12: round(p12 - current_wind, 1),
            24: round(p24 - current_wind, 1)
        }
    }


def run_movement_forecast(obs: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Executes movement trajectory forecast across +6h, +12h, and +24h horizons
    and reconstructs future coordinates via spherical geodesy.
    """
    row_df = _prepare_single_row_df(obs)
    for col in M2_NUM_FEATURES:
        if col in row_df.columns:
            row_df[col] = row_df[col].astype(np.float64)

    X_input = row_df[M2_NUM_FEATURES + M2_CAT_FEATURES]

    curr_lat = float(row_df['LAT'].values[0])
    curr_lon_norm = float(row_df['LON_NORM'].values[0])

    results = {}
    for h in [6, 12, 24]:
        model = load_movement_model(h)
        pred_disp = model.predict(X_input)  # shape (1, 2) -> [dx, dy]
        dx_km = float(pred_disp[0, 0])
        dy_km = float(pred_disp[0, 1])

        # Reconstruct coordinates using verified geodesy utilities
        lat_arr = np.array([curr_lat], dtype=np.float64)
        lon_arr = np.array([curr_lon_norm], dtype=np.float64)
        dx_arr = np.array([dx_km], dtype=np.float64)
        dy_arr = np.array([dy_km], dtype=np.float64)

        lat_pred_arr, lon_pred_arr = reconstruct_coordinates(lat_arr, lon_arr, dx_arr, dy_arr)
        lat_p = float(lat_pred_arr[0])
        lon_p = float(lon_pred_arr[0])

        # Calculate Great-Circle displacement distance from current eye to predicted point
        hav_disp = calculate_haversine_error_km(lat_arr, lon_arr, lat_pred_arr, lon_pred_arr)
        hav_dist_km = float(hav_disp[0])
        euc_dist_km = float(np.sqrt(dx_km**2 + dy_km**2))

        results[h] = {
            'dx_km': round(dx_km, 1),
            'dy_km': round(dy_km, 1),
            'lat_pred': round(lat_p, 3),
            'lon_pred': round(lon_p, 3),
            'displacement_distance_km': round(euc_dist_km, 1),
            'haversine_distance_km': round(hav_dist_km, 1)
        }

    return {
        'current_lat': round(curr_lat, 3),
        'current_lon': round(curr_lon_norm, 3),
        'horizons': results
    }


def run_development_forecast(obs: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Executes 24-hour intensification pattern classification using Model 3.
    """
    row_df = _prepare_single_row_df(obs)
    for col in M3_NUM_FEATURES:
        if col in row_df.columns:
            row_df[col] = row_df[col].astype(np.float64)

    current_wind = float(row_df['CURRENT_USA_WIND'].values[0])

    model = load_development_model()
    features = list(model.feature_names_in_)
    missing = set(features) - set(row_df.columns)
    if missing:
        raise ValueError(f"Development input is missing required features: {sorted(missing)}")
    X_input = row_df[features]
    pred_label = str(model.predict(X_input)[0])

    # Class probabilities
    probs_raw = model.predict_proba(X_input)[0]
    clf_classes = list(model.named_steps['classifier'].classes_)

    prob_dict = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(clf_classes, probs_raw)
    }

    return {
        'predicted_class': pred_label,
        'probabilities': prob_dict,
        'current_wind_kts': round(current_wind, 1),
        'classes': clf_classes
    }
