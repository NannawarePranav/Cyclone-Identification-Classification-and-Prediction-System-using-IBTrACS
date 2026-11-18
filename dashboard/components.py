"""
Cyclone Atlas — UI Components
Standardized, dark-ocean themed reusable visual components.
"""

from typing import Optional, Tuple, Dict, Any
import streamlit as st
import pandas as pd
from dashboard.config import COLORS, ICONS, BASIN_NAMES, NATURE_NAMES, APP_DISCLAIMER
from dashboard.data_loader import get_cyclone_catalog, get_observations_for_cyclone


def render_header(title: str, subtitle: Optional[str] = None, page_badge: Optional[str] = None) -> None:
    """Renders standardized top-of-page header with dark ocean aesthetic."""
    badge_html = f'<span class="kpi-badge">{page_badge}</span>' if page_badge else ""
    sub_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""

    header_html = (
        f'<div style="margin-bottom: 1.5rem; border-bottom: 1px solid {COLORS["border_dark"]}; padding-bottom: 1.25rem;">'
        f'<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">'
        f'<h1 class="page-title" style="margin: 0;">{title}</h1>'
        f'{badge_html}'
        f'</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.html(header_html)


def render_pipeline_diagram():
    """Readable process diagram without external assets or JavaScript."""
    steps = [
        ('01 · Data', 'NOAA IBTrACS cyclone observations'),
        ('02 · Features', 'Wind history, position, pressure and time'),
        ('03 · Models', 'Intensity · Movement · Development'),
        ('04 · Predictions', 'Wind speed · Track · Development class'),
    ]
    cards = ''.join(
        f'<li style="flex:1 1 170px; list-style:none; padding:1rem; border:1px solid {COLORS["border"]}; '
        f'border-radius:6px; background:{COLORS["bg_card"]};">'
        f'<strong style="color:{COLORS["cyan"]};">{title}</strong><br>'
        f'<span style="font-size:.85rem;">{description}</span></li>' for title,description in steps)
    st.html(f'<ol aria-label="Prediction pipeline, read in numbered order" style="display:flex;flex-wrap:wrap;gap:.6rem;padding:0;">{cards}</ol>')
    st.caption('Data → Features → Three models → Predictions. Recorded future outcomes are used only for training labels and evaluation.')


def render_disclaimer() -> None:
    """Renders the required persistent research warning callout."""
    disclaimer_html = (
        f'<div class="disclaimer-callout">'
        f'<span style="display: inline-flex; align-items: center; vertical-align: middle; margin-right: 0.5rem;">'
        f'{ICONS["shield_alert"]}'
        f'</span>'
        f'<span><strong>About the data:</strong> {APP_DISCLAIMER}</span>'
        f'</div>'
    )
    st.html(disclaimer_html)


def render_metric_card(
    title: str,
    value: str,
    subvalue: Optional[str] = None,
    delta: Optional[str] = None,
    color_accent: str = COLORS['accent_cyan'],
    icon_key: Optional[str] = None,
    delta_color: Optional[str] = None
) -> None:
    """Renders a styled dark ocean KPI metric card."""
    icon_html = f'<span style="opacity: 0.85; margin-right: 0.4rem; display: inline-flex; align-items: center;">{ICONS.get(icon_key, "")}</span>' if icon_key else ""
    delta_html = f'<div style="font-size: 0.8rem; color: {delta_color or COLORS["accent_teal"]}; margin-top: 0.25rem; font-weight: 500;">{delta}</div>' if delta else ""
    sub_html = f'<div style="font-size: 0.75rem; color: {COLORS["text_muted"]}; margin-top: 0.35rem;">{subvalue}</div>' if subvalue else ""

    card_html = (
        f'<div class="kpi-card" style="border-top: 3px solid {color_accent};">'
        f'<div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: {COLORS["text_muted"]}; display: flex; align-items: center;">'
        f'{icon_html}{title}'
        f'</div>'
        f'<div style="font-size: 1.75rem; font-weight: 700; color: {COLORS["text_bright"]}; font-family: \'Space Grotesk\', \'Inter\', sans-serif; margin-top: 0.4rem; letter-spacing: -0.02em;">'
        f'{value}'
        f'</div>'
        f'{delta_html}'
        f'{sub_html}'
        f'</div>'
    )
    st.html(card_html)


def render_info_banner(text: str, level: str = "info") -> None:
    """Renders informative status or methodology banners."""
    border_color = COLORS['accent_cyan']
    bg_color = "rgba(0, 210, 211, 0.07)"
    icon = ICONS['info']

    if level == "warning":
        border_color = COLORS['accent_amber']
        bg_color = "rgba(245, 158, 11, 0.08)"
        icon = ICONS['alert_circle']
    elif level == "danger":
        border_color = COLORS['accent_rose']
        bg_color = "rgba(239, 68, 68, 0.08)"
        icon = ICONS['alert_circle']
    elif level == "success":
        border_color = COLORS['accent_teal']
        bg_color = "rgba(16, 185, 129, 0.08)"
        icon = ICONS['shield_alert']

    banner_html = (
        f'<div style="background: {bg_color}; border-left: 3px solid {border_color}; border-radius: 4px; padding: 0.75rem 1rem; margin: 1rem 0; font-size: 0.85rem; color: {COLORS["text_secondary"]}; display: flex; align-items: flex-start; gap: 0.6rem;">'
        f'<span style="flex-shrink: 0; margin-top: 0.15rem;">{icon}</span>'
        f'<div style="line-height: 1.5;">{text}</div>'
        f'</div>'
    )
    st.html(banner_html)


SELECTION_STATE_KEY = 'cyclone_selection'


def _remember_selection(field, widget_key):
    """Copy widget changes into durable state that survives page-widget cleanup."""
    selection = dict(st.session_state[SELECTION_STATE_KEY])
    selection[field] = st.session_state[widget_key]
    if field == 'sid':
        selection['timestamp'] = None
    elif field in {'basin', 'season', 'split', 'search'}:
        selection['filters_changed'] = True
    st.session_state[SELECTION_STATE_KEY] = selection


def render_cyclone_selector(
    dataset_type: str = 'intensity',
    key_prefix: str = 'sel',
    eligible_observations: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[pd.Series]]:
    """Share selection across models, replay and an optionally restricted demo cohort."""
    catalogs = {kind: get_cyclone_catalog(kind) for kind in ('intensity', 'movement', 'development')}
    catalog = catalogs[dataset_type]
    # Common widget bounds preserve filters even when model cohorts differ.
    min_season = min(int(c.SEASON.min()) for c in catalogs.values())
    max_season = max(int(c.SEASON.max()) for c in catalogs.values())
    basin_options = ['ALL'] + sorted(set().union(*(set(c.BASIN.dropna()) for c in catalogs.values())))
    if eligible_observations is not None:
        catalog = catalog[catalog.SID.isin(eligible_observations.SID)].copy()
        counts = eligible_observations.groupby('SID').ISO_TIME.nunique()
        catalog['OBS_COUNT'] = catalog.SID.map(counts)
    if SELECTION_STATE_KEY not in st.session_state:
        st.session_state[SELECTION_STATE_KEY] = {
            'basin': 'ALL', 'season': (max(min_season, 2020), max_season),
            'split': 'TEST', 'search': '', 'sid': None, 'timestamp': None,
        }
    selection = dict(st.session_state[SELECTION_STATE_KEY])
    filters_changed = selection.pop('filters_changed', False)
    st.session_state[SELECTION_STATE_KEY] = selection

    def widget_args(field, value):
        key = f'{key_prefix}_{field}'
        # Rehydrate before widget creation: changes were committed by on_change.
        st.session_state[key] = value
        return {'key': key, 'on_change': _remember_selection, 'args': (field, key)}

    with st.expander('Filter & Select Cyclone Observation', expanded=True):
        st.caption('Your filters, cyclone and observation carry across Models 1, 2, 3, Historical Cyclone Replay and the Guided Presentation Demo.')
        c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
        with c1:
            basin = st.selectbox('Oceanic Basin', basin_options,
                format_func=lambda b: f'{b} — {BASIN_NAMES.get(b,b)}' if b != 'ALL' else 'All Oceanic Basins',
                **widget_args('basin', selection['basin']))
        with c2:
            seasons = st.slider('Season Range (Years)', min_value=min_season, max_value=max_season,
                **widget_args('season', selection['season']))
        with c3:
            labels = {'ALL':'ALL', 'TEST':'TEST (2020+)', 'VAL':'VAL (2015–2019)',
                      'TRAIN':f"TRAIN ({int(catalogs[dataset_type].SEASON.min())}–2014)"}
            split = st.selectbox('Data Partition', list(labels), format_func=labels.get,
                **widget_args('split', selection['split']))
        search = st.text_input('Search cyclone name or ID', placeholder='Try AMPHAN, FANI or a storm ID',
            **widget_args('search', selection['search']))
        st.caption('Search respects the filters above. For older storms, widen the season range and choose ALL partitions.')
        filtered = catalog[catalog.SEASON.between(*seasons)]
        if basin != 'ALL':
            filtered = filtered[filtered.BASIN.eq(basin)]
        if split != 'ALL':
            filtered = filtered[filtered.SPLIT.eq(split)]
        if search.strip():
            filtered = filtered[filtered.NAME.str.contains(search.strip(),case=False,regex=False) |
                                filtered.SID.str.contains(search.strip(),case=False,regex=False)]
        if filtered.empty:
            st.warning('No cyclones match these filters on this page. Your shared selection is preserved; adjust the filters or switch back.')
            return None, None, None

        options = filtered.SID.tolist()
        desired_sid = selection['sid']
        if desired_sid is None or (filters_changed and desired_sid not in options):
            desired_sid = options[0]
            selection.update(sid=desired_sid, timestamp=None)
            st.session_state[SELECTION_STATE_KEY] = selection
        available = desired_sid in options
        if not available:
            st.info(f'The selected cyclone ({desired_sid}) is unavailable on this page with these filters. '
                    'Choose another cyclone to continue, or switch back to keep your original selection.')
        storm_column, obs_column = st.columns(2)
        lookup = filtered.set_index('SID')
        def format_storm(sid):
            r = lookup.loc[sid]
            return f"{r['NAME']} · {r['SEASON']} {r['BASIN']} · {sid} ({r['OBS_COUNT']} obs)"
        with storm_column:
            sid = st.selectbox(f'Cyclone (Total matching: {len(filtered):,})', options,
                index=None, placeholder='Choose an available cyclone', format_func=format_storm,
                **widget_args('sid', desired_sid if available else None))
        if sid is None:
            return None, None, None

        observations = (get_observations_for_cyclone(sid, dataset_type=dataset_type)
                        if eligible_observations is None else
                        eligible_observations[eligible_observations.SID.eq(sid)].sort_values('ISO_TIME').reset_index(drop=True))
        if observations.empty:
            st.warning('This cyclone has no usable observations on this page. Your selection is preserved.')
            return sid, None, None
        times = pd.to_datetime(observations.ISO_TIME, utc=True)
        assert not times.duplicated().any(), f'Duplicate observation timestamps for {sid}'
        timestamps = [t.isoformat() for t in times]
        wanted = selection['timestamp']
        fallback = False
        if wanted is None:
            chosen = timestamps[len(timestamps)//2]
            selection['timestamp'] = chosen
            st.session_state[SELECTION_STATE_KEY] = selection
        elif wanted in timestamps:
            chosen = wanted
        else:
            # Use the latest available observation at/before the requested time.
            earlier = [t for t in timestamps if pd.Timestamp(t) <= pd.Timestamp(wanted)]
            chosen = max(earlier, key=pd.Timestamp) if earlier else min(timestamps, key=pd.Timestamp)
            fallback = True
            # Keep the requested time in shared state so returning restores it.
        positions = {timestamp: i for i,timestamp in enumerate(timestamps)}
        def format_observation(timestamp):
            i = positions[timestamp]
            row = observations.iloc[i]
            wind = row.get('CURRENT_USA_WIND')
            wind_text = f' | {int(wind)} kt' if pd.notna(wind) else ''
            return f"Step {i+1}/{len(observations)}: {pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')} UTC (Lat {row.LAT:.1f}°, Lon {row.LON_NORM:.1f}°{wind_text})"
        with obs_column:
            timestamp = st.selectbox(f'Observation Timestamp ({len(observations)} available steps)', timestamps,
                format_func=format_observation, **widget_args('timestamp', chosen))
        if fallback:
            requested_text = pd.Timestamp(wanted).strftime('%Y-%m-%d %H:%M UTC')
            chosen_text = pd.Timestamp(chosen).strftime('%Y-%m-%d %H:%M UTC')
            st.info(f'{requested_text} is unavailable on this page. Showing {chosen_text} for the same cyclone. '
                    'Your original time is preserved when you switch back; choose a timestamp here to change it across models.')
        return sid, observations, observations.iloc[positions[timestamp]]


def render_observation_summary(row: pd.Series, dataset_type: str = 'intensity') -> None:
    """Renders physical status card of selected observation."""
    c1, c2, c3, c4, c5 = st.columns(5)

    basin_code = str(row.get('BASIN', 'NA'))
    basin_desc = BASIN_NAMES.get(basin_code, basin_code)
    nature_code = str(row.get('NATURE', 'TS'))
    nature_desc = NATURE_NAMES.get(nature_code, nature_code)
    
    with c1:
        render_metric_card(
            "Timestamp (UTC)",
            str(row.get('ISO_TIME', 'N/A')),
            f"Season {int(row.get('SEASON', 0))}",
            icon_key="clock"
        )
    with c2:
        render_metric_card(
            "Basin & Nature",
            basin_code,
            f"{basin_desc} • {nature_desc}",
            icon_key="globe"
        )
    with c3:
        lat = float(row.get('LAT', 0.0))
        lon = float(row.get('LON_NORM', row.get('LON', 0.0)))
        render_metric_card(
            "Coordinates",
            f"{lat:.2f}°, {lon:.2f}°",
            "WGS84 Normalized",
            icon_key="compass"
        )
    with c4:
        wind = row.get('CURRENT_USA_WIND', None)
        wind_disp = f"{int(wind)} kt" if pd.notna(wind) else "N/A"
        has_6h = bool(row.get('HAS_PAST_USA_WIND_6H', 0))
        past_6h_wind = row.get('PAST_USA_WIND_6H', None)
        delta_str = None
        if has_6h and pd.notna(past_6h_wind) and pd.notna(wind):
            diff = int(wind - past_6h_wind)
            delta_str = f"{diff:+d} kt in past 6h"

        render_metric_card(
            "Current Intensity",
            wind_disp,
            "1-min sustained (USA)",
            delta=delta_str,
            color_accent=COLORS['accent_cyan'],
            icon_key="wind"
        )
    with c5:
        dist = row.get('DIST2LAND', None)
        dist_str = f"{int(dist):,} km" if pd.notna(dist) else "N/A"
        split = str(row.get('SPLIT', 'TEST'))
        render_metric_card(
            "Distance to Coast",
            dist_str,
            f"Dataset Partition: {split}",
            color_accent=COLORS['accent_teal'],
            icon_key="activity"
        )
