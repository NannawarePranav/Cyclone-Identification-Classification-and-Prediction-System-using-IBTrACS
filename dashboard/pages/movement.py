"""
Cyclone Atlas — Page 3: Movement & Trajectory Forecast
Multi-horizon (+6h, +12h, +24h) trajectory displacement regression and spherical geodetic reconstruction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from dashboard.config import COLORS, ICONS
from dashboard.components import (
    render_header,
    render_disclaimer,
    render_metric_card,
    render_cyclone_selector,
    render_observation_summary
)
from dashboard.inference import run_movement_forecast, calculate_haversine_error_km
from dashboard.data_loader import get_horizon_targets_for_observation
from dashboard.charts import create_trajectory_map


def render_page() -> None:
    """Renders the Cyclone Atlas Movement & Trajectory page."""
    render_header(
        title="Cyclone Movement & Trajectory Forecast",
        subtitle="Multi-horizon XGBoost MultiOutput regression with spherical geodesic coordinate reconstruction",
        page_badge="Model 2 — MultiOutput Geodesy"
    )
    render_disclaimer()

    # 1. Observation Selector
    sid, obs_df, selected_row = render_cyclone_selector(dataset_type='movement', key_prefix='mov')
    if selected_row is None:
        return

    # 2. Selected Observation Summary
    st.markdown(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['compass']}</span>
            Selected Storm State (t = 0)
        </div>
        """,
        unsafe_allow_html=True
    )
    render_observation_summary(selected_row, dataset_type='movement')

    # 3. Forecast Trigger Button
    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 3])
    with c1:
        run_btn = st.button("Generate Trajectory Forecast", type="primary", width="stretch")
    with c2:
        split_badge = selected_row.get('SPLIT', 'UNKNOWN')
        st.caption(f"Storm SID: `{selected_row.get('SID', 'N/A')}` | Split: **{split_badge}** | Geodesy: Spherical Haversine ($R = 6371$ km)")

    session_key = f"mov_forecast_{selected_row.get('SID', '')}_{selected_row.get('ISO_TIME', '')}"
    if run_btn or session_key in st.session_state:
        st.session_state[session_key] = True

        with st.spinner("Executing Model 2 MultiOutput XGBoost inference across +6h, +12h, and +24h horizons..."):
            forecast = run_movement_forecast(selected_row)

        curr_lat = forecast['current_lat']
        curr_lon = forecast['current_lon']
        horizons_data = forecast['horizons']

        # 4. Multi-Horizon KPI Cards
        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card-header">
                <span style="color: {COLORS['accent_teal']};">{ICONS['navigation']}</span>
                Reconstructed Geodetic Coordinates & Displacement
            </div>
            """,
            unsafe_allow_html=True
        )

        # Retrieve actual ground truth targets across all available horizons
        horizon_targets = get_horizon_targets_for_observation(
            sid=str(selected_row['SID']),
            iso_time=selected_row['ISO_TIME'],
            dataset_type='movement'
        )

        kp1, kp2, kp3 = st.columns(3)
        cols_h = [(6, kp1, COLORS['accent_cyan']), (12, kp2, COLORS['accent_teal']), (24, kp3, COLORS['accent_amber'])]

        for h, col, color in cols_h:
            h_info = horizons_data[h]
            lat_p = h_info['lat_pred']
            lon_p = h_info['lon_pred']
            dist_km = h_info['haversine_distance_km']

            if h in horizon_targets:
                gt_lat = horizon_targets[h]['target_lat']
                gt_lon = horizon_targets[h]['target_lon']
                err_km = float(calculate_haversine_error_km(
                    np.array([gt_lat]), np.array([gt_lon]),
                    np.array([lat_p]), np.array([lon_p])
                )[0])
                kpi_sub = f"GT: {gt_lat:.2f}°, {gt_lon:.2f}° | Track Error: {err_km:.1f} km"
            else:
                kpi_sub = f"Predicted Travel: {dist_km:.1f} km from eye (dx: {h_info['dx_km']:+.0f} km, dy: {h_info['dy_km']:+.0f} km)"

            with col:
                render_metric_card(
                    title=f"+{h}h Reconstructed Position",
                    value=f"{lat_p:.2f}°, {lon_p:.2f}°",
                    subvalue=kpi_sub,
                    delta=f"{dist_km:.0f} km predicted travel from eye",
                    color_accent=color,
                    icon_key="compass"
                )

        # 5. Interactive Trajectory Map
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # Build past track slice
        curr_time = pd.to_datetime(selected_row['ISO_TIME'])
        obs_times = pd.to_datetime(obs_df['ISO_TIME'])
        past_track = obs_df[obs_times <= curr_time].copy()

        map_fig = create_trajectory_map(
            curr_lat=curr_lat,
            curr_lon=curr_lon,
            forecast_results=horizons_data,
            past_track=past_track
        )
        st.plotly_chart(map_fig, width="stretch")

        # 6. Detailed Tabular Coordinate Forecast
        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card-header">
                <span style="color: {COLORS['accent_cyan']};">{ICONS['compass']}</span>
                Coordinate Reconstruction & Geodetic Error Analysis
            </div>
            """,
            unsafe_allow_html=True
        )

        table_rows = []
        # Current row (t=0)
        table_rows.append({
            'Lead Horizon': 't=0 (Current Eye)',
            'Predicted Position': f"{curr_lat:.3f}°, {curr_lon:.3f}°",
            'Predicted dx (East-West)': '0.0 km',
            'Predicted dy (North-South)': '0.0 km',
            'Predicted Travel Distance from Eye': '0.0 km',
            'Actual Future Position (Ground Truth)': f"{curr_lat:.3f}°, {curr_lon:.3f}° (Observed)",
            'Forecast Track Error (km)': '0.0 km (Reference)'
        })

        # Future horizons
        for h in [6, 12, 24]:
            h_info = horizons_data[h]
            lat_p = h_info['lat_pred']
            lon_p = h_info['lon_pred']

            gt_str = "Unavailable (Lifecycle ends)"
            err_str = "N/A"
            if h in horizon_targets:
                gt_lat = horizon_targets[h]['target_lat']
                gt_lon = horizon_targets[h]['target_lon']
                gt_str = f"{gt_lat:.3f}°, {gt_lon:.3f}°"
                err_km = float(calculate_haversine_error_km(
                    np.array([gt_lat]), np.array([gt_lon]),
                    np.array([lat_p]), np.array([lon_p])
                )[0])
                err_str = f"{err_km:.1f} km"

            table_rows.append({
                'Lead Horizon': f"+{h}h Forecast",
                'Predicted Position': f"{lat_p:.3f}°, {lon_p:.3f}°",
                'Predicted dx (East-West)': f"{h_info['dx_km']:+.1f} km",
                'Predicted dy (North-South)': f"{h_info['dy_km']:+.1f} km",
                'Predicted Travel Distance from Eye': f"{h_info['haversine_distance_km']:.1f} km",
                'Actual Future Position (Ground Truth)': gt_str,
                'Forecast Track Error (km)': err_str
            })

        table_df = pd.DataFrame(table_rows)
        st.dataframe(table_df, width="stretch", hide_index=True)

        # 7. Model Features Inspection
        with st.expander("Inspect Model 2 Input Features", expanded=False):
            m2_features = [
                'LAT', 'LON_NORM', 'DIST2LAND', 'CURRENT_TRANSLATION_SPEED_KMH',
                'CURRENT_HEADING_DEG', 'PAST_EAST_DISP_6H_KM', 'PAST_NORTH_DISP_6H_KM',
                'PAST_EAST_DISP_12H_KM', 'PAST_NORTH_DISP_12H_KM', 'BASIN', 'NATURE'
            ]
            f_vals = {}
            for f in m2_features:
                if f in selected_row:
                    v = selected_row[f]
                    f_vals[f] = [f"{v:.3f}" if isinstance(v, float) else str(v)]
            st.dataframe(pd.DataFrame(f_vals), width="stretch", hide_index=True)
            st.caption("Coordinates reconstructed via `src.cyclone_ml.geodesy.reconstruct_coordinates` preserving antimeridian continuity.")
