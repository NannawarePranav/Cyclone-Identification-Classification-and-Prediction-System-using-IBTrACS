"""
Cyclone Atlas — Page 2: Intensity Forecast
Multi-horizon (+6h, +12h, +24h) maximum sustained wind speed regression.
"""

import streamlit as st
import pandas as pd
from dashboard.config import COLORS, ICONS
from dashboard.components import (
    render_header,
    render_disclaimer,
    render_metric_card,
    render_cyclone_selector,
    render_observation_summary,
    render_info_banner
)
from dashboard.inference import run_intensity_forecast
from dashboard.data_loader import get_horizon_targets_for_observation
from dashboard.charts import create_intensity_forecast_chart
from dashboard.error_display import format_error


def render_page() -> None:
    """Renders the Cyclone Atlas Intensity Forecasting page."""
    render_header(
        title="Cyclone Intensity Forecasting",
        subtitle="Direct multi-horizon XGBoost regression for 1-minute maximum sustained wind speed (+6h, +12h, +24h)",
        page_badge="Model 1 — XGBoost Regressors"
    )
    render_disclaimer()

    # 1. Observation Selector
    sid, obs_df, selected_row = render_cyclone_selector(dataset_type='intensity', key_prefix='int')
    if selected_row is None:
        return

    # 2. Selected Observation Summary
    st.html(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['wind']}</span>
            Selected Storm State (t = 0)
        </div>
        """
    )
    render_observation_summary(selected_row, dataset_type='intensity')

    # 3. Forecast Execution Controls
    st.html("<div style='margin-top: 1.25rem;'></div>")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 2])

    with ctrl_col1:
        unit = st.radio(
            "Wind Speed Units",
            options=["knots", "km/h"],
            index=0,
            horizontal=True,
            key="int_unit_toggle"
        )

    with ctrl_col2:
        st.write("")  # spacing
        st.write("")
        run_btn = st.button("Generate Intensity Forecast", type="primary", width="stretch")

    with ctrl_col3:
        st.write("")
        st.write("")
        split_badge = selected_row.get('SPLIT', 'UNKNOWN')
        st.caption(f"Observation Partition: **{split_badge}** | Storm SID: `{selected_row.get('SID', 'N/A')}`")

    # If button pressed or already cached in session state
    session_key = f"int_forecast_{selected_row.get('SID', '')}_{selected_row.get('ISO_TIME', '')}"
    if run_btn or session_key in st.session_state:
        st.session_state[session_key] = True

        with st.spinner("Executing Model 1 XGBoost inference across +6h, +12h, and +24h horizons..."):
            forecast = run_intensity_forecast(selected_row)

        st.html("<div style='margin-top: 1.25rem;'></div>")
        st.html(
            f"""
            <div class="card-header">
                <span style="color: {COLORS['accent_teal']};">{ICONS['activity']}</span>
                Multi-Horizon Wind Speed Predictions
            </div>
            """
        )

        # 4. Multi-horizon KPI Cards
        kp1, kp2, kp3 = st.columns(3)
        unit_suffix = "kt" if unit == "knots" else "km/h"
        pred_dict = forecast['predictions_kts'] if unit == "knots" else forecast['predictions_kmh']
        curr_val = forecast['current_kts'] if unit == "knots" else forecast['current_kmh']

        horizons = [(6, kp1, COLORS['accent_cyan']), (12, kp2, COLORS['accent_teal']), (24, kp3, COLORS['accent_amber'])]

        # Retrieve actual ground truth targets across all available horizons
        horizon_targets = get_horizon_targets_for_observation(
            sid=str(selected_row['SID']),
            iso_time=selected_row['ISO_TIME'],
            dataset_type='intensity'
        )

        for h, col, color in horizons:
            pred_v = pred_dict[h]
            diff = pred_v - curr_val
            delta_str = f"{diff:+0.1f} {unit_suffix} from t=0"

            # Check if matching horizon target sample exists
            if h in horizon_targets:
                gt_val = horizon_targets[h]['target_wind_kts'] if unit == "knots" else horizon_targets[h]['target_wind_kmh']
                abs_err = abs(pred_v - gt_val)
                subvalue = f"Ground Truth: {gt_val:.1f} {unit_suffix} | Absolute Error: {format_error(abs_err, gt_val, unit_suffix)}"
            else:
                subvalue = f"Ground Truth: Unavailable (+{h}h beyond recorded track)"

            with col:
                render_metric_card(
                    title=f"+{h}h Forecast Horizon",
                    value=f"{pred_v:.1f} {unit_suffix}",
                    subvalue=subvalue,
                    delta=delta_str,
                    color_accent=color,
                    icon_key="wind"
                )

        st.caption("Error % = absolute error ÷ observed future wind speed × 100. N/A% when the observed wind is zero or unavailable.")

        # 5. Forecast Trajectory Plot
        st.html("<div style='margin-top: 1.5rem;'></div>")

        past_winds = None
        has_6h = bool(selected_row.get('HAS_PAST_USA_WIND_6H', 0))
        has_12h = bool(selected_row.get('HAS_PAST_USA_WIND_12H', 0))
        if has_6h or has_12h:
            past_winds = {}
            if has_6h and pd.notna(selected_row.get('PAST_USA_WIND_6H', None)):
                past_winds['t-6h'] = float(selected_row['PAST_USA_WIND_6H'])
            if has_12h and pd.notna(selected_row.get('PAST_USA_WIND_12H', None)):
                past_winds['t-12h'] = float(selected_row['PAST_USA_WIND_12H'])

        fig_int = create_intensity_forecast_chart(
            current_wind=forecast['current_kts'],
            predictions=forecast['predictions_kts'],
            past_winds=past_winds,
            unit=unit
        )
        st.plotly_chart(fig_int, width="stretch", theme=None)

        # 6. Model Inputs Inspection Accordion
        with st.expander("Inspect Model Input Features & Preprocessing State", expanded=False):
            feature_cols = [
                'CURRENT_USA_WIND', 'PAST_USA_WIND_6H', 'PAST_USA_WIND_12H',
                'HAS_PAST_USA_WIND_6H', 'HAS_PAST_USA_WIND_12H',
                'WIND_CHANGE_PAST_6H', 'WIND_CHANGE_PAST_12H',
                'LAT', 'LON_NORM', 'DIST2LAND', 'BASIN', 'NATURE', 'USA_AGENCY'
            ]
            feature_data = {}
            for col in feature_cols:
                if col in selected_row:
                    val = selected_row[col]
                    if isinstance(val, float):
                        val = f"{val:.3f}" if not pd.isna(val) else "NaN"
                    feature_data[col] = [val]

            f_df = pd.DataFrame(feature_data)
            st.dataframe(f_df, width="stretch", hide_index=True)
            st.caption("All features strictly adhere to `reports/model_feature_manifest.md` with verified zero target leakage.")
