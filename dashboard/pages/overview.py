"""
Cyclone Atlas — Page 1: Overview & Global Activity
High-level meteorological intelligence, historical distributions, and system architecture.
"""

import streamlit as st
import pandas as pd
from dashboard.config import COLORS, ICONS
from dashboard.components import render_header, render_disclaimer, render_metric_card
from dashboard.data_loader import get_global_summary_stats, load_movement_dataset
from dashboard.charts import (
    create_global_activity_map,
    create_basin_distribution_chart,
    create_historical_trend_chart
)


def render_page() -> None:
    """Renders the Cyclone Atlas Overview page."""
    render_header(
        title="Cyclone Atlas — Global Activity Overview",
        subtitle="NOAA IBTrACS v04r01 (1980–2024) Global Meteorological Intelligence & Multitask AI Architecture",
        page_badge="IBTrACS v04r01"
    )
    render_disclaimer()

    # 1. Global KPI Metrics
    stats = get_global_summary_stats()
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card(
            title="Total Historical Storms",
            value=f"{stats['total_storms']:,}",
            subvalue=f"Temporal Range: {stats['season_range']}",
            icon_key="compass",
            color_accent=COLORS['accent_cyan']
        )
    with c2:
        render_metric_card(
            title="Verified Observations",
            value=f"{stats['total_observations']:,}",
            subvalue="Deduplicated 6-hourly physical track steps",
            icon_key="activity",
            color_accent=COLORS['accent_teal']
        )
    with c3:
        test_obs = stats['split_distribution'].get('TEST', 0)
        render_metric_card(
            title="Evaluation Cohort (TEST)",
            value=f"{test_obs:,}",
            subvalue="Independent 2020–2024 test split",
            icon_key="shield_alert",
            color_accent=COLORS['accent_amber']
        )
    with c4:
        render_metric_card(
            title="Peak Observed Wind",
            value=f"{int(stats['max_wind_kts'])} kt",
            subvalue=f"Global mean: {stats['mean_wind_kts']} kt",
            icon_key="wind",
            color_accent=COLORS['accent_purple']
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Global Geospatial Activity Map
    st.markdown(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['globe']}</span>
            Global Tropical Cyclone Trajectories & Basin Climatology
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Rendering global trajectory map..."):
        df_mov = load_movement_dataset()
        map_fig = create_global_activity_map(df_mov, sample_storms=250)
        st.plotly_chart(map_fig, use_container_width=True)

    # 3. Climatological Distribution Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(
            f"""
            <div class="card-header">
                <span style="color: {COLORS['accent_teal']};">{ICONS['bar_chart']}</span>
                Observation Frequency by Oceanic Basin
            </div>
            """,
            unsafe_allow_html=True
        )
        basin_fig = create_basin_distribution_chart(stats['basin_distribution'])
        st.plotly_chart(basin_fig, use_container_width=True)

    with col_chart2:
        st.markdown(
            f"""
            <div class="card-header">
                <span style="color: {COLORS['accent_amber']};">{ICONS['calendar']}</span>
                Historical Annual Cyclone Activity (1980–2024)
            </div>
            """,
            unsafe_allow_html=True
        )
        trend_fig = create_historical_trend_chart(df_mov)
        st.plotly_chart(trend_fig, use_container_width=True)

    # 4. Multi-Task Machine Learning Architecture Breakdown
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['cpu']}</span>
            Three-Task Machine Learning Architecture
        </div>
        """,
        unsafe_allow_html=True
    )

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid {COLORS['accent_cyan']};">
                <div style="font-size: 0.85rem; font-weight: 700; color: {COLORS['accent_cyan']}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
                    {ICONS['wind']} MODEL 1: INTENSITY FORECASTING
                </div>
                <div style="font-size: 0.82rem; color: {COLORS['text_secondary']}; line-height: 1.55;">
                    Direct multi-horizon regression models predicting 1-minute maximum sustained wind speed at <strong>+6h, +12h, and +24h</strong>.
                </div>
                <hr style="border: none; border-top: 1px solid {COLORS['border_dark']}; margin: 0.75rem 0;">
                <div style="font-size: 0.75rem; color: {COLORS['text_muted']};">
                    <strong>Features:</strong> Current USA wind, verified lags (t-6h, t-12h), wind trends, latitude, distance to land, month/hour cyclics.<br>
                    <strong>Algorithms:</strong> Linear Regression, Random Forest, Hist-Gradient XGBoost.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid {COLORS['accent_teal']};">
                <div style="font-size: 0.85rem; font-weight: 700; color: {COLORS['accent_teal']}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
                    {ICONS['navigation']} MODEL 2: TRAJECTORY TRACKING
                </div>
                <div style="font-size: 0.82rem; color: {COLORS['text_secondary']}; line-height: 1.55;">
                    MultiOutput regressors predicting zonal (East-West) and meridional (North-South) spherical displacement in kilometers at <strong>+6h, +12h, and +24h</strong>.
                </div>
                <hr style="border: none; border-top: 1px solid {COLORS['border_dark']}; margin: 0.75rem 0;">
                <div style="font-size: 0.75rem; color: {COLORS['text_muted']};">
                    <strong>Reconstruction:</strong> Geodesic destination projection with strict [-180, +180] antimeridian wrapping and [-90, +90] clamping.<br>
                    <strong>Validation metric:</strong> Mean Haversine track error in km.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid {COLORS['accent_amber']};">
                <div style="font-size: 0.85rem; font-weight: 700; color: {COLORS['accent_amber']}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
                    {ICONS['activity']} MODEL 3: DEVELOPMENT REGIME
                </div>
                <div style="font-size: 0.82rem; color: {COLORS['text_secondary']}; line-height: 1.55;">
                    Multiclass classification predicting tropical cyclone evolutionary state over the subsequent 24-hour window: <strong>Strengthening, Stable, or Weakening</strong>.
                </div>
                <hr style="border: none; border-top: 1px solid {COLORS['border_dark']}; margin: 0.75rem 0;">
                <div style="font-size: 0.75rem; color: {COLORS['text_muted']};">
                    <strong>Thresholds:</strong> Strengthening (Δ ≥ +10 kt), Weakening (Δ ≤ -10 kt), Stable (|Δ| < 10 kt).<br>
                    <strong>Validation metrics:</strong> Accuracy (0.753), Balanced Accuracy (0.741), Macro F1 (0.746).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
