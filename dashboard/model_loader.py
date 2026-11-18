"""
Cyclone Atlas — Model Loader & Serialization Interface
======================================================
Implements lazy-loading, resource-cached model deserialization for all seven
trained XGBoost pipelines using @st.cache_resource.
"""

import os
import sys
from typing import Dict, Any, Optional
import joblib
import streamlit as st

# Ensure src/ package is accessible before deserializing custom classes
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cyclone_ml import CycloneClassifier


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost'))


def load_intensity_model(horizon_hours: int):
    """
    Loads and caches an XGBoost Intensity regression pipeline (+6h, +12h, or +24h).
    """
    filename = f"m1_xgboost_{horizon_hours}h.joblib"
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return _load_version(path, os.stat(path).st_mtime_ns, os.stat(path).st_size)


def load_movement_model(horizon_hours: int):
    """
    Loads and caches an XGBoost Movement multi-output regression pipeline (+6h, +12h, or +24h).
    """
    filename = f"m2_xgboost_{horizon_hours}h.joblib"
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return _load_version(path, os.stat(path).st_mtime_ns, os.stat(path).st_size)


def load_development_model():
    """
    Loads and caches the XGBoost Development Pattern multi-class classification pipeline (+24h).
    """
    filename = "m3_xgboost.joblib"
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return _load_version(path, os.stat(path).st_mtime_ns, os.stat(path).st_size)


@st.cache_resource(show_spinner=False)
def _load_version(path, modified_ns, size):
    """Invalidate the serving cache when a model file is replaced."""
    return joblib.load(path)


def check_all_models_present() -> Dict[str, bool]:
    """Inspects the disk to verify that all seven required XGBoost artifacts exist."""
    required = {
        'Intensity +6h': os.path.join(MODELS_DIR, "m1_xgboost_6h.joblib"),
        'Intensity +12h': os.path.join(MODELS_DIR, "m1_xgboost_12h.joblib"),
        'Intensity +24h': os.path.join(MODELS_DIR, "m1_xgboost_24h.joblib"),
        'Movement +6h': os.path.join(MODELS_DIR, "m2_xgboost_6h.joblib"),
        'Movement +12h': os.path.join(MODELS_DIR, "m2_xgboost_12h.joblib"),
        'Movement +24h': os.path.join(MODELS_DIR, "m2_xgboost_24h.joblib"),
        'Development +24h': os.path.join(MODELS_DIR, "m3_xgboost.joblib"),
    }
    return {k: os.path.exists(path) for k, path in required.items()}
