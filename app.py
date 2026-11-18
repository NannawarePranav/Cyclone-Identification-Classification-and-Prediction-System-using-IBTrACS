"""
Cyclone Atlas — Web Intelligence Dashboard
Main Entry Point and Multitask AI Application.

Usage:
    streamlit run app.py
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from dashboard.config import APP_NAME

# Configure page settings before any other Streamlit commands
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dashboard.config import apply_custom_css, COLORS, ICONS, APP_NAME, APP_SUBTITLE, APP_DISCLAIMER
from dashboard.components import SELECTION_STATE_KEY
from dashboard.data_loader import load_cyclone_names
from dashboard.model_loader import check_all_models_present
import dashboard.pages.overview as pg_overview
import dashboard.pages.intensity as pg_intensity
import dashboard.pages.movement as pg_movement
import dashboard.pages.development as pg_development
import dashboard.pages.historical_replay as pg_replay
import dashboard.pages.model_performance as pg_performance
import dashboard.pages.guided_demo as pg_demo


def main():
    # 1. Apply Dark Ocean Scientific CSS
    apply_custom_css()

    # 2. Build Sidebar Navigation
    with st.sidebar:
        # App Branding Header
        st.html(
            f"""
            <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid {COLORS['border_dark']}; margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="background: rgba(0, 210, 211, 0.12); border: 1px solid {COLORS['accent_cyan']}; border-radius: 6px; padding: 6px 8px; display: inline-flex; align-items: center;">
                        <span style="color: {COLORS['accent_cyan']};">{ICONS['globe']}</span>
                    </div>
                    <div>
                        <div style="font-size: 1.05rem; line-height: 1.4; font-weight: 700; color: {COLORS['text_bright']}; font-family: 'Fira Sans', sans-serif; letter-spacing: -0.02em;">
                            {APP_NAME}
                        </div>
                        <div style="font-size: 0.72rem; color: {COLORS['text_muted']};">
                            {APP_SUBTITLE}
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        # Page Selection Menu
        nav_options = [
            "Overview & Global Activity",
            "Intensity Forecast (Model 1)",
            "Movement & Trajectory (Model 2)",
            "Development Classification (Model 3)",
            "Historical Cyclone Replay",
            "Model Performance & Benchmarks",
            "Guided Presentation Demo"
        ]

        nav_icons = {
            "Overview & Global Activity": "globe",
            "Intensity Forecast (Model 1)": "wind",
            "Movement & Trajectory (Model 2)": "navigation",
            "Development Classification (Model 3)": "activity",
            "Historical Cyclone Replay": "play",
            "Model Performance & Benchmarks": "bar_chart"
        }

        selected_page = st.radio(
            "Navigation",
            options=nav_options,
            index=0,
            key="navigation",
            label_visibility="collapsed"
        )
        shared_selection_panel = st.empty()

        st.html("<div style='margin-top: 1.5rem;'></div>")

        # System Health & Models Status Card
        models_status = check_all_models_present()
        models_ok = all(models_status.values())
        missing_models = [k for k, v in models_status.items() if not v]
        status_color = COLORS['accent_teal'] if models_ok else COLORS['accent_amber']
        status_text = "All 7 Model Artifacts Present" if models_ok else f"Missing: {len(missing_models)}"

        st.html(
            f"""
            <div class="kpi-card" style="padding: 0.85rem; border-top: 2px solid {status_color}; margin-bottom: 1.25rem;">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: {COLORS['text_muted']}; margin-bottom: 0.4rem; display: flex; align-items: center; justify-content: space-between;">
                    <span>System Status</span>
                    <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: {status_color};"></span>
                </div>
                <div style="font-size: 0.82rem; font-weight: 600; color: {COLORS['text_bright']};">
                    {status_text}
                </div>
                <div style="font-size: 0.72rem; color: {COLORS['text_dim']}; margin-top: 0.25rem;">
                    NOAA IBTrACS v04r01 Archive<br>
                    3 Verified Task Datasets
                </div>
            </div>
            """
        )

        # Persistent Sidebar Research Callout
        st.html(
            f"""
            <div style="font-size: 0.72rem; color: {COLORS['text_dim']}; line-height: 1.45; border-top: 1px solid {COLORS['border_dark']}; padding-top: 1rem;">
                <strong style="color: {COLORS['text_muted']};">BTech Project · Semester VII</strong><br>
                Three prediction tasks powered by NOAA IBTrACS data and XGBoost.
            </div>
            """
        )

    # 3. Route to Selected Page
    if selected_page == "Overview & Global Activity":
        pg_overview.render_page()
    elif selected_page == "Intensity Forecast (Model 1)":
        pg_intensity.render_page()
    elif selected_page == "Movement & Trajectory (Model 2)":
        pg_movement.render_page()
    elif selected_page == "Development Classification (Model 3)":
        pg_development.render_page()
    elif selected_page == "Historical Cyclone Replay":
        pg_replay.render_page()
    elif selected_page == "Model Performance & Benchmarks":
        pg_performance.render_page()
    elif selected_page == "Guided Presentation Demo":
        pg_demo.render_page()

    # Populate after the page so newly selected defaults appear immediately.
    with shared_selection_panel.container():
        with st.expander('Shared cyclone selection', expanded=True):
            selection = st.session_state.get(SELECTION_STATE_KEY, {})
            sid = selection.get('sid')
            if sid:
                st.write(f"**{load_cyclone_names().get(sid, sid)}**")
                st.caption(sid)
                timestamp = selection.get('timestamp')
                st.caption(timestamp.replace('T', ' ').replace('+00:00', ' UTC') if timestamp else 'Choose an observation')
                season = selection['season']
                st.caption(f"{selection['basin']} basin · {season[0]}–{season[1]} · {selection['split']}")
                if selection.get('search'):
                    st.caption('Search: ' + selection['search'])
            else:
                st.caption('Select a cyclone on any model, replay or guided demo page.')
            st.caption('Shared by all three models, Historical Replay and Guided Demo. Your selection is retained on every page.')
            if selected_page in ('Overview & Global Activity', 'Model Performance & Benchmarks'):
                st.caption('This page shows global results; the shared selection does not filter its totals.')


if __name__ == "__main__":
    main()
