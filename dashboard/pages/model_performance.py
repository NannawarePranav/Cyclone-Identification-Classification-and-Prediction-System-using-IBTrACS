"""
Cyclone Atlas — Page 6: Model Performance & Academic Benchmarks
Rigorous empirical evaluation and head-to-head scoreboards on the 2020+ test partition.
"""

import streamlit as st
import pandas as pd
from dashboard.config import COLORS, ICONS
from dashboard.components import render_header, render_disclaimer, render_metric_card
from dashboard.charts import create_model_performance_charts


def render_page() -> None:
    """Renders the Cyclone Atlas Model Performance page."""
    render_header(
        title="Model Performance & Academic Benchmarks",
        subtitle="Empirical evaluation on the untouched 2020–present independent test set across baselines and tuned XGBoost models",
        page_badge="Verified Test Results (2020+)"
    )
    render_disclaimer()

    # 1. Headline Scoreboard KPIs
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        render_metric_card(
            title="Model 1: 24h Wind MAE",
            value="9.58 kt",
            subvalue="+27.98% skill over persistence (13.31 kt)",
            delta="+4.30% gain over RF",
            icon_key="wind",
            color_accent=COLORS['accent_cyan']
        )
    with k2:
        render_metric_card(
            title="Model 2: 24h Track Error",
            value="181.6 km",
            subvalue="+12.87% skill over const-velocity (208.4 km)",
            delta="+1.35% gain over RF",
            icon_key="navigation",
            color_accent=COLORS['accent_teal']
        )
    with k3:
        render_metric_card(
            title="Model 3: Accuracy / F1",
            value="63.42%",
            subvalue="Macro-F1: 0.638 (vs Majority: 41.30%)",
            delta="+22.12% skill over majority",
            icon_key="activity",
            color_accent=COLORS['accent_amber']
        )
    with k4:
        render_metric_card(
            title="Temporal Generalization",
            value="Zero Leakage",
            subvalue="TRAIN (≤2014) • VAL (2015–19) • TEST (2020+)",
            icon_key="shield_alert",
            color_accent=COLORS['accent_purple']
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Master Head-to-Head Benchmark Table
    st.markdown(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['bar_chart']}</span>
            Master Benchmark Scoreboard (Untouched Test Set: 2020–Present)
        </div>
        """,
        unsafe_allow_html=True
    )

    master_benchmarks = [
        {
            "Prediction Task": "Model 1: Intensity (+6h)",
            "Evaluation Metric": "Wind MAE (kts)",
            "Simple Baseline": "Persistence: 4.02 kts",
            "Random Forest (Task 6)": "3.52 kts",
            "XGBoost (Task 7)": "3.43 kts",
            "Skill vs Simple": "+14.83%",
            "Skill vs RF": "+2.56%",
            "Status": "Superior"
        },
        {
            "Prediction Task": "Model 1: Intensity (+12h)",
            "Evaluation Metric": "Wind MAE (kts)",
            "Simple Baseline": "Persistence: 7.50 kts",
            "Random Forest (Task 6)": "5.92 kts",
            "XGBoost (Task 7)": "5.71 kts",
            "Skill vs Simple": "+23.93%",
            "Skill vs RF": "+3.55%",
            "Status": "Superior"
        },
        {
            "Prediction Task": "Model 1: Intensity (+24h)",
            "Evaluation Metric": "Wind MAE (kts)",
            "Simple Baseline": "Persistence: 13.31 kts",
            "Random Forest (Task 6)": "10.01 kts",
            "XGBoost (Task 7)": "9.58 kts",
            "Skill vs Simple": "+27.98%",
            "Skill vs RF": "+4.30%",
            "Status": "Superior"
        },
        {
            "Prediction Task": "Model 2: Movement (+6h)",
            "Evaluation Metric": "Haversine Error (km)",
            "Simple Baseline": "Const-Vel: 38.8 km",
            "Random Forest (Task 6)": "37.0 km",
            "XGBoost (Task 7)": "36.6 km",
            "Skill vs Simple": "+5.61%",
            "Skill vs RF": "+0.97%",
            "Status": "Superior"
        },
        {
            "Prediction Task": "Model 2: Movement (+12h)",
            "Evaluation Metric": "Haversine Error (km)",
            "Simple Baseline": "Const-Vel: 87.2 km",
            "Random Forest (Task 6)": "80.5 km",
            "XGBoost (Task 7)": "79.6 km",
            "Skill vs Simple": "+8.71%",
            "Skill vs RF": "+1.20%",
            "Status": "Superior"
        },
        {
            "Prediction Task": "Model 2: Movement (+24h)",
            "Evaluation Metric": "Haversine Error (km)",
            "Simple Baseline": "Const-Vel: 208.4 km",
            "Random Forest (Task 6)": "184.1 km",
            "XGBoost (Task 7)": "181.6 km",
            "Skill vs Simple": "+12.87%",
            "Skill vs RF": "+1.35%",
            "Status": "Superior"
        },
        {
            "Prediction Task": "Model 3: Development (+24h)",
            "Evaluation Metric": "Accuracy (Macro-F1)",
            "Simple Baseline": "Majority: 41.30%",
            "Random Forest (Task 6)": "62.71% (0.631)",
            "XGBoost (Task 7)": "63.42% (0.638)",
            "Skill vs Simple": "+22.12%",
            "Skill vs RF": "+0.71% Acc",
            "Status": "Superior"
        }
    ]
    st.dataframe(pd.DataFrame(master_benchmarks), use_container_width=True, hide_index=True)

    # 3. Model Benchmark Visualizations
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    fig_m1, fig_m2, fig_m3 = create_model_performance_charts()

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.plotly_chart(fig_m1, use_container_width=True)
    with col_v2:
        st.plotly_chart(fig_m2, use_container_width=True)

    st.plotly_chart(fig_m3, use_container_width=True)

    # 4. Basin-Wise Breakdown & Generalization Insights
    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
    with st.expander("Basin-Wise Generalization Analysis (Test Set, 2020–Present)", expanded=True):
        st.markdown(
            """
            Across all oceanic basins, **XGBoost demonstrates consistent, non-degrading skill** over both simple climatological baselines and Random Forests:
            - **Eastern Pacific (EP):** Highest relative intensity skill (+33.88% at +24h, MAE 8.40 kt vs Persistence 12.71 kt), driven by steady steering flows and strong sea surface temperature gradients.
            - **North Atlantic (NA):** Intensity MAE reaches 9.01 kt (+23.28% skill vs persistence) and 6h track error decreases to 35.8 km.
            - **North Indian (NI):** Characterized by high variability due to seasonal monsoons and land boundaries, yet XGBoost improves 24h intensity from 15.25 kt persistence to 10.05 kt (+34.10% skill).
            - **Western Pacific (WP):** The most active basin worldwide; 24h track displacement error is reduced from 215.3 km (constant-velocity) to 184.2 km.
            """
        )

        basin_summary_data = [
            {"Basin": "EP — Eastern North Pacific", "Test Obs": "2,430", "Pers 24h MAE": "12.71 kt", "RF 24h MAE": "8.75 kt", "XGB 24h MAE": "8.40 kt", "Skill vs Pers": "+33.88%"},
            {"Basin": "NA — North Atlantic", "Test Obs": "2,971", "Pers 24h MAE": "11.75 kt", "RF 24h MAE": "9.55 kt", "XGB 24h MAE": "9.01 kt", "Skill vs Pers": "+23.28%"},
            {"Basin": "NI — North Indian", "Test Obs": "540", "Pers 24h MAE": "15.25 kt", "RF 24h MAE": "10.73 kt", "XGB 24h MAE": "10.05 kt", "Skill vs Pers": "+34.10%"},
            {"Basin": "SI — South Indian", "Test Obs": "2,488", "Pers 24h MAE": "13.68 kt", "RF 24h MAE": "10.66 kt", "XGB 24h MAE": "10.22 kt", "Skill vs Pers": "+25.29%"},
            {"Basin": "SP — South Pacific", "Test Obs": "942", "Pers 24h MAE": "13.82 kt", "RF 24h MAE": "10.82 kt", "XGB 24h MAE": "10.41 kt", "Skill vs Pers": "+24.67%"},
            {"Basin": "WP — Western North Pacific", "Test Obs": "2,912", "Pers 24h MAE": "15.20 kt", "RF 24h MAE": "11.23 kt", "XGB 24h MAE": "10.79 kt", "Skill vs Pers": "+29.01%"}
        ]
        st.dataframe(pd.DataFrame(basin_summary_data), use_container_width=True, hide_index=True)

    # 5. Methodological Integrity & Academic Rigor
    with st.expander("Methodological Safeguards & Data Governance", expanded=False):
        st.markdown(
            """
            1. **Zero Information Leakage:** Models were fitted solely using pre-forecast historical features. Target values (`TARGET_*`), future coordinates, and storm identifiers (`SID`, `ISO_TIME`) were strictly isolated.
            2. **Temporal Split Discipline:** Observation cohorts were partitioned strictly chronologically:
               - **TRAIN:** 1980 – 2014 (35 years)
               - **VAL:** 2015 – 2019 (5 years, used for candidate parameter selection)
               - **TEST:** 2020 – 2024 (Untouched 5-year independent test set)
            3. **Spherical Geodesy:** Movement trajectory models use true spherical trigonometric transformations with antimeridian discontinuity handling.
            4. **Artifact Portability:** All 18 model pipelines are self-contained with scikit-learn compatible custom transformers located in `src/cyclone_ml/estimators.py`.
            """
        )
