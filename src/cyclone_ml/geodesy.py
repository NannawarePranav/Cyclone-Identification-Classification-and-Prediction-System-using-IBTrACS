"""
Cyclone ML Geodetic and Spatial Utilities
=========================================
Provides spherical coordinate reconstruction and geodesic distance calculations.
"""

from typing import Tuple
import numpy as np

EARTH_RADIUS_KM = 6371.0


def reconstruct_coordinates(
    lat_curr: np.ndarray,
    lon_curr_norm: np.ndarray,
    dx_km: np.ndarray,
    dy_km: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs predicted geographic coordinates (LAT, LON_NORM) from
    orthogonal eastward (dx) and northward (dy) displacements in kilometers.

    Parameters:
        lat_curr: Current cyclone latitude in degrees [-90, 90].
        lon_curr_norm: Current cyclone longitude normalized to [-180, 180].
        dx_km: Predicted eastward displacement in kilometers.
        dy_km: Predicted northward displacement in kilometers.

    Returns:
        (lat_pred, lon_pred): Reconstructed geographic coordinates.
    """
    # 1. Delta Latitude in degrees
    dlat_deg = dy_km * 180.0 / (np.pi * EARTH_RADIUS_KM)
    lat_pred = np.clip(lat_curr + dlat_deg, -90.0, 90.0)

    # 2. Midpoint latitude for mean curvature
    mean_lat_rad = np.radians((lat_curr + lat_pred) / 2.0)
    cos_mean_lat = np.maximum(np.cos(mean_lat_rad), 1e-6)

    # 3. Delta Longitude in degrees
    dlon_deg = dx_km * 180.0 / (np.pi * EARTH_RADIUS_KM * cos_mean_lat)
    lon_pred = ((lon_curr_norm + dlon_deg + 180.0) % 360.0) - 180.0

    return lat_pred, lon_pred


def calculate_haversine_error_km(
    lat_true: np.ndarray,
    lon_true_norm: np.ndarray,
    lat_pred: np.ndarray,
    lon_pred_norm: np.ndarray
) -> np.ndarray:
    """
    Computes exact spherical great-circle geodesic error in kilometers
    between predicted and true geographic coordinates.
    """
    dlat_rad = np.radians(lat_true - lat_pred)
    dlon_deg = ((lon_true_norm - lon_pred_norm + 180.0) % 360.0) - 180.0
    dlon_rad = np.radians(dlon_deg)

    lat1_rad = np.radians(lat_pred)
    lat2_rad = np.radians(lat_true)

    a = (np.sin(dlat_rad / 2.0) ** 2 +
         np.cos(lat1_rad) * np.cos(lat2_rad) * (np.sin(dlon_rad / 2.0) ** 2))
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c
