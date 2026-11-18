"""
Cyclone Atlas — Page 5: Historical Cyclone Replay
Interactive chronological step-through and lifecycle visualization of historical storm tracks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.config import COLORS, ICONS, BASIN_NAMES
from dashboard.components import (
    render_header, render_disclaimer, render_metric_card,
    render_cyclone_selector, SELECTION_STATE_KEY,
)
from dashboard.charts import create_replay_map, apply_dark_layout
from dashboard.inference import calculate_haversine_error_km


def _select_replay_step(timestamps, step=None):
    """Commit playback actions before widgets render, keeping all pages in sync."""
    if step is None:
        step = st.session_state['replay_step']
    selection = dict(st.session_state[SELECTION_STATE_KEY])
    selection['timestamp'] = timestamps[step]
    st.session_state[SELECTION_STATE_KEY] = selection


def render_page() -> None:
    """Renders the Cyclone Atlas Historical Replay page."""
    render_header(
        title="Historical Cyclone Lifecycle Replay",
        subtitle="Chronological track animation, intensity evolution, and physical trajectory analysis",
        page_badge="Lifecycle Replay Engine"
    )
    render_disclaimer()

    # Use the same filters, storm and observation as the three model pages.
    selected_sid, track_df, selected_row = render_cyclone_selector('movement', 'replay')
    if selected_row is None:
        return
    track_df = track_df.copy()

    # Ensure strictly sorted chronological physical observations with no duplicate timestamps
    track_df['ISO_TIME'] = pd.to_datetime(track_df['ISO_TIME'])
    track_df = track_df.sort_values('ISO_TIME').reset_index(drop=True)
    assert not track_df.duplicated(subset=['ISO_TIME']).any(), (
        f"Duplicate timestamps found in physical replay track for SID: {selected_sid}"
    )

    total_steps = len(track_df)

    # Calculate cumulative Great-Circle distance traveled
    cumulative_km = 0.0
    if total_steps > 1:
        lats = track_df['LAT'].values
        lons = track_df['LON_NORM'].values
        step_distances = calculate_haversine_error_km(lats[:-1], lons[:-1], lats[1:], lons[1:])
        cumulative_km = float(np.sum(step_distances))

    t_start = track_df['ISO_TIME'].iloc[0]
    t_end = track_df['ISO_TIME'].iloc[-1]
    actual_elapsed_seconds = (t_end - t_start).total_seconds()
    actual_elapsed_hours = max(0.0, actual_elapsed_seconds / 3600.0)

    if total_steps <= 1 or actual_elapsed_hours == 0:
        duration_display = "Single Obs (0h)"
        avg_speed_display = "N/A"
    else:
        days = actual_elapsed_hours / 24.0
        duration_display = f"{actual_elapsed_hours:.0f} Hours ({days:.1f} Days)"
        avg_speed = cumulative_km / actual_elapsed_hours
        avg_speed_display = f"{avg_speed:.1f} km/h"

    start_iso = t_start.strftime('%Y-%m-%d %H:%M')
    end_iso = t_end.strftime('%Y-%m-%d %H:%M')
    max_wind = float(track_df['CURRENT_USA_WIND'].max()) if 'CURRENT_USA_WIND' in track_df and not track_df['CURRENT_USA_WIND'].dropna().empty else 0.0

    # 3. Lifecycle KPI Summary
    st.html("<div style='margin-top: 1rem;'></div>")
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        render_metric_card(
            title="Storm Duration",
            value=duration_display,
            subvalue=f"{total_steps} physical observations ({start_iso[:10]} to {end_iso[:10]})",
            icon_key="clock",
            color_accent=COLORS['accent_cyan']
        )
    with k2:
        render_metric_card(
            title="Total Track Length",
            value=f"{cumulative_km:,.0f} km",
            subvalue=f"Average translation speed: {avg_speed_display}",
            icon_key="compass",
            color_accent=COLORS['accent_teal']
        )
    with k3:
        render_metric_card(
            title="Peak Sustained Wind",
            value=f"{int(max_wind)} kt" if max_wind > 0 else "N/A",
            subvalue=f"Temporal Span: {start_iso[:10]} to {end_iso[:10]}",
            icon_key="wind",
            color_accent=COLORS['accent_amber']
        )
    with k4:
        basin_code = track_df.iloc[0]['BASIN']
        render_metric_card(
            title="Basin & Partition",
            value=f"{basin_code}",
            subvalue=f"{BASIN_NAMES.get(basin_code, basin_code)} • {track_df.iloc[0]['SPLIT']}",
            icon_key="globe",
            color_accent=COLORS['accent_purple']
        )

    # 4. Interactive Step Slider & Playback Controls
    st.html("<div style='margin-top: 1.5rem;'></div>")
    st.html(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_cyan']};">{ICONS['play']}</span>
            Chronological Track Step-Through
        </div>
        """
    )

    timestamps = pd.to_datetime(track_df['ISO_TIME'], utc=True).map(lambda t: t.isoformat()).tolist()
    selected_time = pd.to_datetime(selected_row['ISO_TIME'], utc=True).isoformat()
    current_step = timestamps.index(selected_time)
    st.session_state['replay_step'] = current_step
    ctrl_col, slider_col = st.columns([1, 4])

    with ctrl_col:
        st.write("")
        b_prev, b_next = st.columns(2)
        with b_prev:
            st.button("< Prev", width="stretch", disabled=current_step == 0,
                      on_click=_select_replay_step, args=(timestamps, max(0, current_step - 1)))
        with b_next:
            st.button("Next >", width="stretch", disabled=current_step == total_steps - 1,
                      on_click=_select_replay_step, args=(timestamps, min(total_steps - 1, current_step + 1)))

    with slider_col:
        if total_steps > 1:
            st.slider("Observation Step (Chronological)", min_value=0, max_value=total_steps - 1,
                      format="Step %d", key='replay_step',
                      on_change=_select_replay_step, args=(timestamps,))
        else:
            st.caption("Single recorded observation (Step 0).")

    curr_point = track_df.iloc[current_step]

    # Current Step Snapshot
    st.caption(
        f"**Active Step {current_step + 1}/{total_steps}**: Timestamp `{curr_point['ISO_TIME']}` | "
        f"Position: **{curr_point['LAT']:.2f}°, {curr_point['LON_NORM']:.2f}°** | "
        f"Wind: **{curr_point.get('CURRENT_USA_WIND', 'N/A')} kt** | "
        f"Distance to Coast: **{curr_point.get('DIST2LAND', 'N/A')} km**"
    )

    # 5. Interactive Replay Map
    st.html("<div style='margin-top: 0.75rem;'></div>")
    replay_fig = create_replay_map(track_df, current_step)
    st.plotly_chart(replay_fig, width="stretch", theme=None)

    # 6. Storm Intensity Profile Evolution Chart
    st.html("<div style='margin-top: 1.25rem;'></div>")
    st.html(
        f"""
        <div class="card-header">
            <span style="color: {COLORS['accent_teal']};">{ICONS['wind']}</span>
            Lifecycle Intensity Profile (1-Minute Sustained Wind Speed)
        </div>
        """
    )

    if 'CURRENT_USA_WIND' in track_df and not track_df['CURRENT_USA_WIND'].dropna().empty:
        wind_fig = go.Figure()

        # Full intensity curve
        wind_fig.add_trace(go.Scatter(
            x=list(range(1, total_steps + 1)),
            y=track_df['CURRENT_USA_WIND'],
            mode='lines+markers',
            name="1-Min Sustained Wind (kt)",
            line=dict(color=COLORS['cyan'], width=2),
            marker=dict(size=5, color=COLORS['teal'])
        ))

        # Active Step marker
        wind_fig.add_trace(go.Scatter(
            x=[current_step + 1],
            y=[curr_point.get('CURRENT_USA_WIND', 0)],
            mode='markers+text',
            name="Selected Step",
            marker=dict(size=14, color=COLORS['accent_amber'], line=dict(color='#FFFFFF', width=2)),
            text=[f"{curr_point.get('CURRENT_USA_WIND', 0):.0f} kt"],
            textposition="top center",
            textfont=dict(family="Fira Code", color=COLORS['accent_amber'], size=12)
        ))

        apply_dark_layout(
            wind_fig,
            height=280,
            xaxis=dict(title="Track Step Number", tickmode='linear', dtick=max(1, total_steps // 10)),
            yaxis=dict(title="USA Sustained Wind (knots)"),
            showlegend=True
        )
        st.plotly_chart(wind_fig, width="stretch", theme=None)

    # 7. Complete Historical Track Table
    with st.expander(f"Inspect Complete Historical Track Data ({total_steps} Observations)", expanded=False):
        display_cols = ['ISO_TIME', 'LAT', 'LON_NORM', 'CURRENT_USA_WIND', 'DIST2LAND', 'NATURE', 'BASIN', 'SPLIT']
        available_display_cols = [c for c in display_cols if c in track_df.columns]
        st.dataframe(track_df[available_display_cols], width="stretch")
