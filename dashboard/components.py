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


def render_disclaimer() -> None:
    """Renders the required persistent research warning callout."""
    disclaimer_html = (
        f'<div class="disclaimer-callout">'
        f'<span style="display: inline-flex; align-items: center; vertical-align: middle; margin-right: 0.5rem;">'
        f'{ICONS["shield_alert"]}'
        f'</span>'
        f'<strong>RESEARCH & DEMONSTRATION NOTICE:</strong> {APP_DISCLAIMER}'
        f'</div>'
    )
    st.html(disclaimer_html)


def render_metric_card(
    title: str,
    value: str,
    subvalue: Optional[str] = None,
    delta: Optional[str] = None,
    color_accent: str = COLORS['accent_cyan'],
    icon_key: Optional[str] = None
) -> None:
    """Renders a styled dark ocean KPI metric card."""
    icon_html = f'<span style="opacity: 0.85; margin-right: 0.4rem; display: inline-flex; align-items: center;">{ICONS.get(icon_key, "")}</span>' if icon_key else ""
    delta_html = f'<div style="font-size: 0.8rem; color: {COLORS["accent_teal"]}; margin-top: 0.25rem; font-weight: 500;">{delta}</div>' if delta else ""
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


def render_cyclone_selector(
    dataset_type: str = 'intensity',
    key_prefix: str = 'sel'
) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Standardized, interactive cyclone observation selector.
    Filters by Basin, Season, Cyclone SID, and Observation Timestamp.
    Returns:
        (selected_sid, observations_dataframe, selected_observation_row)
    """
    with st.expander("Filter & Select Cyclone Observation", expanded=True):
        catalog = get_cyclone_catalog(dataset_type=dataset_type)

        col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

        # 1. Basin Filter
        available_basins = sorted(catalog['BASIN'].dropna().unique().tolist())
        basin_options = ['ALL'] + available_basins
        basin_labels = {b: (f"{b} — {BASIN_NAMES.get(b, b)}" if b != 'ALL' else "All Oceanic Basins") for b in basin_options}

        with col1:
            selected_basin = st.selectbox(
                "Oceanic Basin",
                options=basin_options,
                format_func=lambda b: basin_labels.get(b, b),
                key=f"{key_prefix}_basin"
            )

        # 2. Season Range Filter
        min_season = int(catalog['SEASON'].min())
        max_season = int(catalog['SEASON'].max())
        default_range = (max(min_season, 2020), max_season)

        with col2:
            season_range = st.slider(
                "Season Range (Years)",
                min_value=min_season,
                max_value=max_season,
                value=default_range,
                key=f"{key_prefix}_season"
            )

        # 3. Data Partition Filter (with dataset-specific temporal cohort)
        if dataset_type == 'movement':
            split_options = ['ALL', 'TEST (2020+)', 'VAL (2015–2019)', 'TRAIN (1970–2014)']
        else:
            split_options = ['ALL', 'TEST (2020+)', 'VAL (2015–2019)', 'TRAIN (1980–2014)']

        with col3:
            split_choice = st.selectbox(
                "Data Partition",
                options=split_options,
                index=1,  # Default to TEST
                key=f"{key_prefix}_split"
            )

        # Filter catalog
        filtered_catalog = catalog.copy()
        if selected_basin != 'ALL':
            filtered_catalog = filtered_catalog[filtered_catalog['BASIN'] == selected_basin]

        filtered_catalog = filtered_catalog[
            (filtered_catalog['SEASON'] >= season_range[0]) &
            (filtered_catalog['SEASON'] <= season_range[1])
        ]

        if split_choice.startswith('TEST'):
            filtered_catalog = filtered_catalog[filtered_catalog['SPLIT'] == 'TEST']
        elif split_choice.startswith('VAL'):
            filtered_catalog = filtered_catalog[filtered_catalog['SPLIT'] == 'VAL']
        elif split_choice.startswith('TRAIN'):
            filtered_catalog = filtered_catalog[filtered_catalog['SPLIT'] == 'TRAIN']

        if filtered_catalog.empty:
            st.warning("No cyclones match the selected filters. Please expand your criteria.")
            return None, None, None

        # 4. Cyclone SID Selection
        col_storm, col_obs = st.columns([2, 2])

        storm_options = filtered_catalog['SID'].tolist()
        def format_storm(sid: str) -> str:
            r = filtered_catalog[filtered_catalog['SID'] == sid].iloc[0]
            basin_short = r['BASIN']
            season = r['SEASON']
            obs = r['OBS_COUNT']
            peak = f"{int(r['PEAK_WIND_KTS'])} kt" if 'PEAK_WIND_KTS' in r and pd.notna(r['PEAK_WIND_KTS']) else ""
            peak_str = f" | Peak: {peak}" if peak else ""
            return f"{sid} — {season} {basin_short} ({obs} obs{peak_str})"

        with col_storm:
            selected_sid = st.selectbox(
                f"Cyclone Storm ID (Total matching: {len(filtered_catalog):,})",
                options=storm_options,
                format_func=format_storm,
                key=f"{key_prefix}_sid"
            )

        if not selected_sid:
            return None, None, None

        # 5. Load Unique Physical Track Observations for this Cyclone
        obs_df = get_observations_for_cyclone(selected_sid, dataset_type=dataset_type)
        if obs_df.empty:
            st.error(f"No records found for storm ID {selected_sid}")
            return selected_sid, None, None

        # Assert no duplicate physical timestamps exist
        assert not obs_df.duplicated(subset=['ISO_TIME']).any(), (
            f"Non-unique physical timestamps detected in track sequence for {selected_sid}"
        )

        def format_obs(idx: int) -> str:
            row = obs_df.iloc[idx]
            iso = str(row['ISO_TIME'])
            lat = row.get('LAT', 0.0)
            lon = row.get('LON_NORM', row.get('LON', 0.0))
            wind = row.get('CURRENT_USA_WIND', None)
            wind_str = f" | {int(wind)} kt" if pd.notna(wind) else ""
            return f"Step {idx+1}/{len(obs_df)}: {iso} (Lat {lat:.1f}°, Lon {lon:.1f}°{wind_str})"

        with col_obs:
            selected_idx = st.selectbox(
                f"Observation Timestamp ({len(obs_df)} available steps)",
                options=list(range(len(obs_df))),
                index=min(len(obs_df) // 2, len(obs_df) - 1),  # pick a middle step by default
                format_func=format_obs,
                key=f"{key_prefix}_obs_idx"
            )

        selected_row = obs_df.iloc[selected_idx]
        return selected_sid, obs_df, selected_row


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
