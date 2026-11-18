"""
Cyclone Atlas — Page 4: Development Pattern Classification
Multiclass classification predicting 24-hour cyclone intensification regimes:
Strengthening, Stable, or Weakening.
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
from dashboard.inference import run_development_forecast
from dashboard.model_loader import load_development_model
from dashboard.charts import create_development_probability_chart


def render_page() -> None:
    """Renders the Cyclone Atlas Development Pattern Classification page."""
    render_header(
        title="Development Classification",
        subtitle="Will the cyclone strengthen, remain stable, or weaken over the next 24 hours?",
        page_badge="Model 3 · +24 hours"
    )
    render_disclaimer()
    st.caption(
        "Based on the change in sustained wind over 24 hours: "
        "Strengthening ≥ +10 kt · Stable between −10 and +10 kt · Weakening ≤ −10 kt."
    )

    # 1. Observation Selector
    sid, obs_df, selected_row = render_cyclone_selector(dataset_type='development', key_prefix='dev')
    if selected_row is None:
        return

    # 2. Selected Observation Summary
    st.html(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['activity']}</span>
            Selected Storm State (t = 0)
        </div>
        """
    )
    render_observation_summary(selected_row, dataset_type='development')

    # 4. Classification Trigger Controls
    st.html("<div style='margin-top: 1rem;'></div>")
    c1, c2 = st.columns([1.5, 3])
    with c1:
        run_btn = st.button("Classify Development", type="primary", width="stretch")
    with c2:
        split_badge = selected_row.get('SPLIT', 'UNKNOWN')
        st.caption(f"Storm SID: `{selected_row.get('SID', 'N/A')}` | Split: **{split_badge}** | Horizon: **+24 Hours**")

    session_key = f"dev_forecast_{selected_row.get('SID', '')}_{selected_row.get('ISO_TIME', '')}"
    if run_btn or session_key in st.session_state:
        st.session_state[session_key] = True

        with st.spinner("Executing Model 3 CycloneClassifier inference..."):
            forecast = run_development_forecast(selected_row)

        pred_class = forecast['predicted_class']
        probs = forecast['probabilities']
        winning_prob = probs.get(pred_class, 0.0) * 100

        # Class color accent
        class_color = COLORS['accent_cyan']
        if pred_class == 'Strengthening':
            class_color = COLORS['accent_amber']
        elif pred_class == 'Weakening':
            class_color = COLORS['accent_teal']

        st.html("<div style='margin-top: 1.25rem;'></div>")
        st.html(
            f"""
            <div class="card-header">
                <span style="color: {class_color};">{ICONS['activity']}</span>
                Predicted Development · Next 24 Hours
            </div>
            """
        )

        # 5. Result Metric Cards
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            render_metric_card(
                title="Predicted Development (+24h)",
                value=pred_class,
                subvalue="Based on the selected observation and past wind history",
                color_accent=class_color,
                icon_key="activity"
            )

        with col_res2:
            render_metric_card(
                title="Model Score",
                value=f"{winning_prob:.1f}%",
                subvalue="Uncalibrated probability for the predicted class",
                color_accent=COLORS['accent_cyan'],
                icon_key="shield_alert"
            )

        with col_res3:
            gt_label = selected_row.get('CLASS_LABEL', None)
            delta_wind = selected_row.get('WIND_CHANGE_24H', None)
            gt_str = str(gt_label) if pd.notna(gt_label) else "Unavailable"
            sub_str = f"Actual 24h Δ: {delta_wind:+0.0f} kt" if pd.notna(delta_wind) else "Recorded track ends"

            gt_color = COLORS['text_muted']
            if gt_label == pred_class:
                gt_color = COLORS['accent_teal']
            elif pd.notna(gt_label):
                gt_color = COLORS['accent_rose']

            render_metric_card(
                title="Observed Development",
                value=gt_str,
                subvalue=sub_str,
                delta="Prediction Matches" if gt_label == pred_class else ("Misclassified" if pd.notna(gt_label) else None),
                delta_color=gt_color,
                color_accent=gt_color,
                icon_key="compass"
            )

        # 6. Probability Distribution Plot
        st.html("<div style='margin-top: 1.5rem;'></div>")
        st.html(
            f"""
            <div class="card-header">
                <span style="color: {COLORS['accent_cyan']};">{ICONS['bar_chart']}</span>
                Model Scores for Each Development Class
            </div>
            """
        )
        prob_fig = create_development_probability_chart(probs)
        st.plotly_chart(prob_fig, width="stretch")
        st.caption("Scores are uncalibrated model probabilities, not measured chances of a weather event.")

        # 7. Physical Regime Definitions & Manifest Reference
        st.html("<div style='margin-top: 1.25rem;'></div>")
        col_def1, col_def2, col_def3 = st.columns(3)

        with col_def1:
            st.html(
                f"""
                <div class="kpi-card" style="border-top: 2px solid {COLORS['accent_amber']};">
                    <div style="font-size: 0.8rem; font-weight: 700; color: {COLORS['accent_amber']}; margin-bottom: 0.35rem;">
                        STRENGTHENING
                    </div>
                    <div style="font-size: 0.78rem; color: {COLORS['text_secondary']};">
                        Defined as a 24-hour maximum sustained wind speed increase of <strong>Δ ≥ +10 kt</strong>.
                    </div>
                </div>
                """
            )

        with col_def2:
            st.html(
                f"""
                <div class="kpi-card" style="border-top: 2px solid {COLORS['accent_cyan']};">
                    <div style="font-size: 0.8rem; font-weight: 700; color: {COLORS['accent_cyan']}; margin-bottom: 0.35rem;">
                        STABLE
                    </div>
                    <div style="font-size: 0.78rem; color: {COLORS['text_secondary']};">
                        Defined as steady-state intensity with 24-hour wind variation <strong>|Δ| &lt; 10 kt</strong>.
                    </div>
                </div>
                """
            )

        with col_def3:
            st.html(
                f"""
                <div class="kpi-card" style="border-top: 2px solid {COLORS['accent_teal']};">
                    <div style="font-size: 0.8rem; font-weight: 700; color: {COLORS['accent_teal']}; margin-bottom: 0.35rem;">
                        WEAKENING
                    </div>
                    <div style="font-size: 0.78rem; color: {COLORS['text_secondary']};">
                        Defined as a 24-hour wind decrease of <strong>Δ ≤ -10 kt</strong>.
                    </div>
                </div>
                """
            )

        # 8. Feature Inputs Inspector
        with st.expander("Inspect Model 3 Input Features", expanded=False):
            m3_features = list(load_development_model().feature_names_in_)
            f_vals = {}
            for f in m3_features:
                if f in selected_row:
                    v = selected_row[f]
                    f_vals[f] = [f"{v:.3f}" if isinstance(v, float) else str(v)]
            st.dataframe(pd.DataFrame(f_vals), width="stretch", hide_index=True)
            st.caption("Target definition: `reports/model_feature_manifest.md` | Model artifact: `models/xgboost/m3_xgboost.joblib`")

        st.divider()
        st.caption(
            "- The prediction uses wind history, location and the selected history/pressure features. "
            "The recorded outcome is shown only for comparison, never as a model input.\n"
            "- Class scores are uncalibrated: a 70% score does not establish a measured 70% likelihood.\n"
            "- The ±10-knot change thresholds over 24 hours define the three development classes."
        )
