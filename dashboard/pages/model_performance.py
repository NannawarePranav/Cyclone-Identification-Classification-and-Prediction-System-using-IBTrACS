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

    st.html("<div style='margin-top: 1.5rem;'></div>")

    # 2. Master Head-to-Head Benchmark Table
    st.html(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['bar_chart']}</span>
            Master Benchmark Scoreboard (Untouched Test Set: 2020–Present)
        </div>
        """
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
    st.dataframe(pd.DataFrame(master_benchmarks), width="stretch", hide_index=True)

    # 3. Model Benchmark Visualizations
    st.html("<div style='margin-top: 1.5rem;'></div>")
    fig_m1, fig_m2, fig_m3 = create_model_performance_charts()

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.plotly_chart(fig_m1, width="stretch")
    with col_v2:
        st.plotly_chart(fig_m2, width="stretch")

    st.plotly_chart(fig_m3, width="stretch")

    # 4. Basin-Wise Breakdown & Generalization Insights
    st.html("<div style='margin-top: 1.25rem;'></div>")
    with st.expander("Model 1: +24h Intensity Performance by Basin (Test Set: 2020–Present)", expanded=True):
        st.markdown(
            """
            Across all evaluated oceanic basins, **XGBoost demonstrates consistent, positive skill** relative to persistence benchmarks and outperforms Random Forest baselines:
            - **Eastern Pacific (EP):** 24h intensity MAE of 8.40 kt (+33.88% skill over 12.71 kt persistence).
            - **North Atlantic (NA):** 24h intensity MAE of 9.01 kt (+23.28% skill over 11.75 kt persistence).
            - **North Indian (NI):** 24h intensity MAE of 10.05 kt (+34.10% skill over 15.25 kt persistence).
            - **South Indian (SI):** 24h intensity MAE of 10.22 kt (+20.36% skill over 12.83 kt persistence).
            - **South Pacific (SP):** 24h intensity MAE of 10.00 kt (+26.60% skill over 13.62 kt persistence).
            - **Western Pacific (WP):** 24h intensity MAE of 10.32 kt (+32.46% skill over 15.29 kt persistence).
            """
        )

        basin_summary_data = [
            {"Basin": "EP — Eastern North Pacific", "Test Obs": "2,430", "Pers 24h MAE": "12.71 kt", "RF 24h MAE": "8.75 kt", "XGB 24h MAE": "8.40 kt", "Skill vs Pers": "+33.88%"},
            {"Basin": "NA — North Atlantic", "Test Obs": "2,971", "Pers 24h MAE": "11.75 kt", "RF 24h MAE": "9.55 kt", "XGB 24h MAE": "9.01 kt", "Skill vs Pers": "+23.28%"},
            {"Basin": "NI — North Indian", "Test Obs": "540", "Pers 24h MAE": "15.25 kt", "RF 24h MAE": "10.73 kt", "XGB 24h MAE": "10.05 kt", "Skill vs Pers": "+34.10%"},
            {"Basin": "SI — South Indian", "Test Obs": "2,636", "Pers 24h MAE": "12.83 kt", "RF 24h MAE": "10.50 kt", "XGB 24h MAE": "10.22 kt", "Skill vs Pers": "+20.36%"},
            {"Basin": "SP — South Pacific", "Test Obs": "1,000", "Pers 24h MAE": "13.62 kt", "RF 24h MAE": "10.68 kt", "XGB 24h MAE": "10.00 kt", "Skill vs Pers": "+26.60%"},
            {"Basin": "WP — Western North Pacific", "Test Obs": "3,035", "Pers 24h MAE": "15.29 kt", "RF 24h MAE": "10.70 kt", "XGB 24h MAE": "10.32 kt", "Skill vs Pers": "+32.46%"}
        ]
        st.dataframe(pd.DataFrame(basin_summary_data), width="stretch", hide_index=True)

    # 5. Methodological Integrity & Academic Rigor
    with st.expander("Methodological Safeguards & Data Governance", expanded=False):
        st.markdown(
            r"""
            1. **Zero Information Leakage:** Models were fitted solely using pre-forecast historical features. Target values (`TARGET_*`), future coordinates, and storm identifiers (`SID`, `ISO_TIME`) were strictly isolated.
            2. **Authoritative Temporal Split Discipline:** Observation cohorts were partitioned strictly chronologically at the cyclone level:
               - **Model 1 (Intensity):** TRAIN $\le 2014$ (1980–2014) • VAL 2015–2019 • TEST 2020 onward
               - **Model 2 (Movement):** TRAIN $\le 2014$ (1970–2014, earlier historical cohort) • VAL 2015–2019 • TEST 2020 onward
               - **Model 3 (Development):** TRAIN $\le 2014$ (1980–2014) • VAL 2015–2019 • TEST 2020 onward
            3. **Spherical Geodesy:** Movement trajectory models use true spherical trigonometric transformations with antimeridian discontinuity handling.
            4. **Artifact Portability:** All 18 model pipelines are self-contained with scikit-learn compatible custom estimators located in `src/cyclone_ml/estimators.py`.
            """
        )
