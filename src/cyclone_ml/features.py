"""
Cyclone ML Feature Engineering Utilities
========================================
Provides cyclical date/time encodings and feature extraction helpers.
"""

import numpy as np
import pandas as pd


def compute_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sine/cosine cyclical representations for Month (1-12) and Hour (0-23).
    """
    df = df.copy()
    if 'MONTH' in df.columns:
        df['MONTH_SIN'] = np.sin(2.0 * np.pi * df['MONTH'] / 12.0)
        df['MONTH_COS'] = np.cos(2.0 * np.pi * df['MONTH'] / 12.0)
    if 'HOUR' in df.columns:
        df['HOUR_SIN'] = np.sin(2.0 * np.pi * df['HOUR'] / 24.0)
        df['HOUR_COS'] = np.cos(2.0 * np.pi * df['HOUR'] / 24.0)
    return df
