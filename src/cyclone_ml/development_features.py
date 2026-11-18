"""Past-only development features, joined at exact observation times and agency."""
import numpy as np
import pandas as pd

BASE_NUM = [
    'CURRENT_USA_WIND', 'PAST_USA_WIND_6H', 'PAST_USA_WIND_12H',
    'HAS_PAST_USA_WIND_6H', 'HAS_PAST_USA_WIND_12H',
    'WIND_CHANGE_PAST_6H', 'WIND_CHANGE_PAST_12H', 'LAT', 'LON_NORM',
    'DIST2LAND', 'MONTH_SIN', 'MONTH_COS', 'HOUR_SIN', 'HOUR_COS',
]
HISTORY_NUM = ['STORM_AGE_HOURS', 'WIND_ACCELERATION_6H', 'LAND_APPROACH_6H']
for hours in (24, 36, 48):
    HISTORY_NUM += [f'PAST_USA_WIND_{hours}H', f'HAS_PAST_USA_WIND_{hours}H',
                    f'WIND_CHANGE_PAST_{hours}H']
PRESSURE_NUM = ['CURRENT_USA_PRES', 'HAS_CURRENT_USA_PRES']
for hours in (6, 12, 24):
    PRESSURE_NUM += [f'PAST_USA_PRES_{hours}H', f'HAS_PAST_USA_PRES_{hours}H',
                     f'PRESSURE_CHANGE_PAST_{hours}H']
CAT = ['BASIN', 'NATURE']


def enrich_development(frame, raw):
    """Preserve cohort/targets; no target or future record enters any feature."""
    out = frame.copy()
    raw = raw.loc[raw.TRACK_TYPE.eq('main')].copy()
    raw['DT'] = pd.to_datetime(raw.ISO_TIME, utc=True)
    now = pd.to_datetime(out.ISO_TIME, utc=True)
    for col in ('USA_WIND', 'USA_PRES', 'DIST2LAND'):
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
        raw.loc[raw[col] < 0, col] = np.nan
    raw.loc[raw.USA_PRES.eq(0), 'USA_PRES'] = np.nan
    first = raw.groupby('SID').DT.min()
    out['STORM_AGE_HOURS'] = (now - out.SID.map(first)).dt.total_seconds() / 3600
    pool = raw.loc[raw.IFLAG.str[0].isin(['O', 'V']) & raw.USA_AGENCY.ne('')]
    pool = pool.drop_duplicates(['SID', 'USA_AGENCY', 'DT']).set_index(['SID', 'USA_AGENCY', 'DT'])

    def past(column, hours):
        keys = pd.MultiIndex.from_arrays([out.SID, out.USA_AGENCY, now - pd.Timedelta(hours=hours)],
                                         names=['SID', 'USA_AGENCY', 'DT'])
        return pool[column].reindex(keys).to_numpy(dtype=float)

    for hours in (24, 36, 48):
        lag = past('USA_WIND', hours)
        out[f'PAST_USA_WIND_{hours}H'] = lag
        out[f'HAS_PAST_USA_WIND_{hours}H'] = np.isfinite(lag).astype(int)
        out[f'WIND_CHANGE_PAST_{hours}H'] = out.CURRENT_USA_WIND - lag
    out['WIND_ACCELERATION_6H'] = out.CURRENT_USA_WIND - 2*out.PAST_USA_WIND_6H + out.PAST_USA_WIND_12H
    out['LAND_APPROACH_6H'] = past('DIST2LAND', 6) - out.DIST2LAND
    out['CURRENT_USA_PRES'] = past('USA_PRES', 0)
    out['HAS_CURRENT_USA_PRES'] = out.CURRENT_USA_PRES.notna().astype(int)
    for hours in (6, 12, 24):
        lag = past('USA_PRES', hours)
        out[f'PAST_USA_PRES_{hours}H'] = lag
        out[f'HAS_PAST_USA_PRES_{hours}H'] = np.isfinite(lag).astype(int)
        out[f'PRESSURE_CHANGE_PAST_{hours}H'] = out.CURRENT_USA_PRES - lag
    return out
