"""
Cyclone Atlas — Page 1: Overview & Global Activity
High-level meteorological intelligence, historical distributions, and system architecture.
"""

import streamlit as st
import pandas as pd
from dashboard.config import COLORS, ICONS, APP_NAME
from dashboard.components import render_header, render_disclaimer, render_metric_card, render_pipeline_diagram
from dashboard.benchmarks import load_benchmarks
from dashboard.data_loader import get_global_summary_stats, load_movement_dataset
from dashboard.charts import (
    create_global_activity_map,
    create_basin_distribution_chart,
    create_historical_trend_chart
)


def render_page() -> None:
    """Renders the Cyclone Atlas Overview page."""
    # 1. Global KPI Metrics
    stats = get_global_summary_stats()

    render_header(
        title=APP_NAME,
        subtitle="How strong will a cyclone become, where will it move, and how will it develop?",
        page_badge="IBTrACS v04r01"
    )
    render_disclaimer()
    with st.expander('How predictions work'):
        render_pipeline_diagram()

    st.markdown("**Explore the three prediction tasks**")
    tasks = [
        ("Model 1 · Intensity", "Predict wind speed at +6, +12 and +24 hours.", "Intensity Forecast (Model 1)"),
        ("Model 2 · Movement", "Predict the next positions on the cyclone track.", "Movement & Trajectory (Model 2)"),
        ("Model 3 · Development", "Classify strengthening, stable or weakening at +24 hours.", "Development Classification (Model 3)"),
    ]
    def open_task(page):
        st.session_state["navigation"] = page

    for column, (label, description, page) in zip(st.columns(3), tasks):
        with column:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.caption(description)
                st.button("Open " + label.split(" · ")[1], key=page, on_click=open_task,
                          args=(page,), width="stretch")

    st.caption("Demo flow: select a cyclone → choose an observation → generate a prediction → compare with the recorded outcome.")

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
            subvalue="Deduplicated physical track steps",
            icon_key="activity",
            color_accent=COLORS['accent_teal']
        )
    with c3:
        test_obs = stats['split_distribution'].get('TEST', 0)
        render_metric_card(
            title="Evaluation Cohort (TEST)",
            value=f"{test_obs:,}",
            subvalue="Independent 2020+ test split",
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

    st.html("<div style='margin-top: 1.5rem;'></div>")

    # 2. Global Geospatial Activity Map
    st.html(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['globe']}</span>
            Global Tropical Cyclone Trajectories & Basin Climatology
        </div>
        """
    )

    with st.spinner("Rendering global trajectory map..."):
        df_mov = load_movement_dataset()
        map_fig = create_global_activity_map(df_mov, sample_storms=250)
        st.plotly_chart(map_fig, width="stretch")

    # 3. Climatological Distribution Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.html(
            f"""
            <div class="card-header">
                <span style="color: {COLORS['accent_teal']};">{ICONS['bar_chart']}</span>
                Observation Frequency by Oceanic Basin
            </div>
            """
        )
        basin_fig = create_basin_distribution_chart(stats['basin_distribution'])
        st.plotly_chart(basin_fig, width="stretch")

    with col_chart2:
        st.html(
            f"""
            <div class="card-header">
                <span style="color: {COLORS['accent_amber']};">{ICONS['calendar']}</span>
                Historical Annual Cyclone Activity ({stats['season_range']})
            </div>
            """
        )
        trend_fig = create_historical_trend_chart(df_mov)
        st.plotly_chart(trend_fig, width="stretch")

    # 4. Multi-Task Machine Learning Architecture Breakdown
    st.html("<div style='margin-top: 1.5rem;'></div>")
    st.html(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['cpu']}</span>
            Three-Task Machine Learning Architecture
        </div>
        """
    )

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.html(
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
                    <strong>Temporal Cohort:</strong> TRAIN 1980–2014 | VAL 2015–2019 | TEST 2020+.<br>
                    <strong>Features:</strong> Current USA wind, verified lags (t-6h, t-12h), wind trends, latitude, distance to land, month/hour cyclics.<br>
                    <strong>Algorithms:</strong> Linear Regression, Random Forest, Hist-Gradient XGBoost.
                </div>
            </div>
            """
        )

    with col_m2:
        st.html(
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
                    <strong>Temporal Cohort:</strong> TRAIN 1970–2014 (earlier historical cohort) | VAL 2015–2019 | TEST 2020+.<br>
                    <strong>Reconstruction:</strong> Geodesic destination projection with strict [-180, +180] antimeridian wrapping and [-90, +90] clamping.<br>
                    <strong>Validation metric:</strong> Mean Haversine track error in km.
                </div>
            </div>
            """
        )

    with col_m3:
        development_metrics = next(r for r in load_benchmarks()['records'] if r['task'] == 3)
        st.html(
            f"""
            <div class="kpi-card" style="border-top: 3px solid {COLORS['accent_amber']};">
                <div style="font-size: 0.85rem; font-weight: 700; color: {COLORS['accent_amber']}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
                    {ICONS['activity']} MODEL 3: DEVELOPMENT CLASSIFICATION
                </div>
                <div style="font-size: 0.82rem; color: {COLORS['text_secondary']}; line-height: 1.55;">
                    Multiclass classification predicting tropical cyclone evolutionary state over the subsequent 24-hour window: <strong>Strengthening, Stable, or Weakening</strong>.
                </div>
                <hr style="border: none; border-top: 1px solid {COLORS['border_dark']}; margin: 0.75rem 0;">
                <div style="font-size: 0.75rem; color: {COLORS['text_muted']};">
                    <strong>Temporal Cohort:</strong> TRAIN 1980–2014 | VAL 2015–2019 | TEST 2020+.<br>
                    <strong>Thresholds:</strong> Strengthening (Δ ≥ +10 kt), Weakening (Δ ≤ -10 kt), Stable (|Δ| < 10 kt).<br>
                    <strong>Test Performance:</strong> Accuracy ({development_metrics['value']:.2f}%), Macro F1 ({development_metrics['macro_f1']:.3f}).
                </div>
            </div>
            """
        )
