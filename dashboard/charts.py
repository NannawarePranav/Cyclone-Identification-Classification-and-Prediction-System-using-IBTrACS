"""
Cyclone Atlas — Plotly Visualizations & Dark Ocean Chart Builders
Academic, dark-ocean styled visualizations adhering strictly to the design system.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dashboard.config import COLORS, BASIN_NAMES

# Shared Plotly Dark Layout Tokens
DARK_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color=COLORS['text_secondary'], size=12),
    margin=dict(l=40, r=20, t=35, b=40),
    xaxis=dict(
        gridcolor=COLORS['grid_line'],
        zerolinecolor=COLORS['border_dark'],
        tickfont=dict(family="Fira Code, monospace", size=11, color=COLORS['text_muted'])
    ),
    yaxis=dict(
        gridcolor=COLORS['grid_line'],
        zerolinecolor=COLORS['border_dark'],
        tickfont=dict(family="Fira Code, monospace", size=11, color=COLORS['text_muted'])
    ),
    legend=dict(
        bgcolor="rgba(15, 26, 48, 0.8)",
        bordercolor=COLORS['border_dark'],
        borderwidth=1,
        font=dict(size=11, color=COLORS['text_secondary'])
    )
)


def apply_dark_layout(fig: go.Figure, **kwargs) -> go.Figure:
    """Safely merges default dark theme layout tokens with custom overrides."""
    merged = {**DARK_PLOT_LAYOUT, **kwargs}
    fig.update_layout(**merged)
    return fig


def create_global_activity_map(df_movement: pd.DataFrame, sample_storms: int = 250) -> go.Figure:
    """
    Renders high-performance global activity map with sampled storm paths.
    """
    # Sample unique storms stratified or recent
    recent_sids = df_movement['SID'].unique()
    if len(recent_sids) > sample_storms:
        # Pick modern sample
        sampled_sids = recent_sids[-sample_storms:]
        sample_df = df_movement[df_movement['SID'].isin(sampled_sids)]
    else:
        sample_df = df_movement

    fig = go.Figure()

    # Base track lines by basin
    basin_colors = {
        'NA': COLORS['cyan'],
        'WP': COLORS['teal'],
        'EP': COLORS['amber'],
        'NI': '#A78BFA',  # purple
        'SI': '#F472B6',  # pink
        'SP': '#38BDF8',  # light blue
        'SA': '#F87171'   # light red
    }

    for basin, group in sample_df.groupby('BASIN'):
        color = basin_colors.get(basin, COLORS['cyan'])
        basin_name = BASIN_NAMES.get(basin, basin)

        # Build paths with None breaks between storms for fast single-trace rendering
        lons: List[Optional[float]] = []
        lats: List[Optional[float]] = []
        for _, storm in group.groupby('SID'):
            lons.extend(storm['LON_NORM'].tolist())
            lons.append(None)
            lats.extend(storm['LAT'].tolist())
            lats.append(None)

        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(width=1.2, color=color),
            opacity=0.7,
            name=f"{basin} ({basin_name})",
            hoverinfo='none'
        ))

    fig.update_geos(
        projection_type="equirectangular",
        showland=True,
        landcolor="#1A2846",
        showocean=True,
        oceancolor="#080E1E",
        showcoastlines=True,
        coastlinecolor="#2D3E66",
        showcountries=True,
        countrycolor="#1D2A48",
        bgcolor="rgba(0,0,0,0)"
    )

    apply_dark_layout(fig,
        height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_basin_distribution_chart(basin_counts: Dict[str, int]) -> go.Figure:
    """
    Renders clean horizontal bar chart of observation frequency by basin.
    """
    basins = list(basin_counts.keys())
    counts = list(basin_counts.values())
    labels = [f"{b} ({BASIN_NAMES.get(b, b)})" for b in basins]

    # Sort descending
    sorted_pairs = sorted(zip(labels, counts), key=lambda x: x[1])
    sorted_labels = [p[0] for p in sorted_pairs]
    sorted_counts = [p[1] for p in sorted_pairs]

    fig = go.Figure(go.Bar(
        x=sorted_counts,
        y=sorted_labels,
        orientation='h',
        marker=dict(
            color=COLORS['cyan'],
            line=dict(color=COLORS['teal'], width=1)
        ),
        text=[f"{c:,}" for c in sorted_counts],
        textposition='outside',
        textfont=dict(family="Fira Code, monospace", size=10, color=COLORS['text_muted'])
    ))

    apply_dark_layout(fig,
        height=320,
        xaxis=dict(title="Physical Track Observations", gridcolor=COLORS['grid_line']),
        yaxis=dict(title="", tickfont=dict(size=11, color=COLORS['text_secondary']))
    )
    return fig


def create_historical_trend_chart(df_movement: pd.DataFrame) -> go.Figure:
    """
    Renders annual observation count line and bar plot.
    """
    # Count unique storms per season
    dedup = df_movement.drop_duplicates(subset=['SID'])
    annual_counts = dedup.groupby('SEASON')['SID'].count().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=annual_counts['SEASON'],
        y=annual_counts['SID'],
        name="Storm Count",
        marker_color="rgba(0, 210, 211, 0.35)",
        marker_line=dict(color=COLORS['cyan'], width=1)
    ))

    # Rolling 5-year average
    rolling_avg = annual_counts['SID'].rolling(5, min_periods=1, center=True).mean()
    fig.add_trace(go.Scatter(
        x=annual_counts['SEASON'],
        y=rolling_avg,
        mode='lines',
        name="5-Yr Rolling Mean",
        line=dict(color=COLORS['amber'], width=2.5)
    ))

    apply_dark_layout(fig,
        height=320,
        xaxis=dict(title="Season (Year)", dtick=5),
        yaxis=dict(title="Annual Tropical Cyclones"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
    )
    return fig


def create_intensity_forecast_chart(
    current_wind: float,
    predictions: Dict[int, float],
    past_winds: Optional[Dict[str, float]] = None,
    unit: str = "knots"
) -> go.Figure:
    """
    Renders time-series forecast plot for intensity (+6h, +12h, +24h).
    """
    mult = 1.852 if unit == "km/h" else 1.0
    u_label = "km/h" if unit == "km/h" else "knots"

    times = [0, 6, 12, 24]
    winds = [
        current_wind * mult,
        predictions[6] * mult,
        predictions[12] * mult,
        predictions[24] * mult
    ]

    fig = go.Figure()

    # Historical lead-in if available
    if past_winds:
        past_times = [-12, -6, 0]
        past_vals = [
            past_winds.get('t-12h', current_wind) * mult,
            past_winds.get('t-6h', current_wind) * mult,
            current_wind * mult
        ]
        fig.add_trace(go.Scatter(
            x=past_times,
            y=past_vals,
            mode='lines+markers',
            name="Past Observed",
            line=dict(color=COLORS['text_muted'], width=2, dash='dot'),
            marker=dict(size=7, color=COLORS['text_muted'])
        ))

    # Forecast Trajectory
    fig.add_trace(go.Scatter(
        x=times,
        y=winds,
        mode='lines',
        name="Forecast Trend",
        line=dict(color=COLORS['cyan'], width=2.5, dash='dash')
    ))

    # Current Point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[winds[0]],
        mode='markers+text',
        name="Current Eye (t=0)",
        marker=dict(size=12, color='#FFFFFF', line=dict(color=COLORS['cyan'], width=2)),
        text=[f"t=0h: {winds[0]:.1f} {u_label}"],
        textposition="top center",
        textfont=dict(family="Fira Code, monospace", color="#FFFFFF", size=11)
    ))

    # Forecast Points with distinct horizon colors
    colors_h = {6: COLORS['cyan'], 12: COLORS['teal'], 24: COLORS['amber']}
    for h, w in zip([6, 12, 24], winds[1:]):
        fig.add_trace(go.Scatter(
            x=[h],
            y=[w],
            mode='markers+text',
            name=f"+{h}h Forecast",
            marker=dict(size=10, color=colors_h[h]),
            text=[f"+{h}h: {w:.1f} {u_label}"],
            textposition="top center",
            textfont=dict(family="Fira Code, monospace", color=colors_h[h], size=11)
        ))

    apply_dark_layout(fig,
        height=380,
        xaxis=dict(
            title="Forecast Lead Time (Hours from t=0)",
            tickvals=[-12, -6, 0, 6, 12, 24] if past_winds else [0, 6, 12, 24],
            ticktext=["t-12h", "t-6h", "t=0", "+6h", "+12h", "+24h"] if past_winds else ["t=0", "+6h", "+12h", "+24h"]
        ),
        yaxis=dict(title=f"1-Min Sustained Wind ({u_label})"),
        showlegend=True
    )
    return fig


def create_trajectory_map(
    curr_lat: float,
    curr_lon: float,
    forecast_results: Dict[int, Dict[str, Any]],
    past_track: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Renders interactive Plotly map showing current location and three independently
    predicted future points (+6h, +12h, +24h) with reconstructed coordinates.
    """
    fig = go.Figure()

    # Past observed track if provided
    if past_track is not None and len(past_track) > 0:
        fig.add_trace(go.Scattergeo(
            lon=past_track['LON_NORM'],
            lat=past_track['LAT'],
            mode='lines+markers',
            name="Past Observed Track",
            line=dict(color=COLORS['text_muted'], width=1.5, dash='dot'),
            marker=dict(size=4, color=COLORS['text_muted']),
            hoverinfo='text',
            text=[f"Obs: {t}<br>Lat: {la:.2f}°, Lon: {lo:.2f}°"
                  for t, la, lo in zip(past_track['ISO_TIME'], past_track['LAT'], past_track['LON_NORM'])]
        ))

    # Forecast points line
    f_lons = [curr_lon] + [forecast_results[h]['lon_pred'] for h in [6, 12, 24]]
    f_lats = [curr_lat] + [forecast_results[h]['lat_pred'] for h in [6, 12, 24]]

    fig.add_trace(go.Scattergeo(
        lon=f_lons,
        lat=f_lats,
        mode='lines',
        name="Predicted Track Trajectory",
        line=dict(color=COLORS['cyan'], width=2, dash='dash')
    ))

    # Current Location marker
    fig.add_trace(go.Scattergeo(
        lon=[curr_lon],
        lat=[curr_lat],
        mode='markers+text',
        name="Current Eye Position (t=0)",
        marker=dict(size=14, color='#FFFFFF', line=dict(color=COLORS['cyan'], width=2)),
        text=["Current Eye"],
        textposition="top center",
        textfont=dict(family="Fira Code, monospace", color="#FFFFFF", size=11),
        hoverinfo='text',
        hovertext=f"Current Eye (t=0)<br>Lat: {curr_lat:.3f}°<br>Lon: {curr_lon:.3f}°"
    ))

    # Individual Horizon Points
    colors_h = {6: COLORS['cyan'], 12: COLORS['teal'], 24: COLORS['amber']}
    for h in [6, 12, 24]:
        info = forecast_results[h]
        fig.add_trace(go.Scattergeo(
            lon=[info['lon_pred']],
            lat=[info['lat_pred']],
            mode='markers+text',
            name=f"+{h}h Forecast ({info['haversine_distance_km']:.0f} km)",
            marker=dict(size=11, color=colors_h[h], line=dict(color='#080E1E', width=1)),
            text=[f"+{h}h"],
            textposition="top right",
            textfont=dict(family="Fira Code, monospace", color=colors_h[h], size=11),
            hoverinfo='text',
            hovertext=(f"+{h}h Forecast<br>Lat: {info['lat_pred']:.3f}°<br>Lon: {info['lon_pred']:.3f}°"
                       f"<br>dx: {info['dx_km']:+.1f} km<br>dy: {info['dy_km']:+.1f} km"
                       f"<br>Distance: {info['haversine_distance_km']:.1f} km")
        ))

    # Map bounds centered around cyclone
    all_lats = f_lats + [curr_lat]
    all_lons = f_lons + [curr_lon]
    center_lat = float(np.mean(all_lats))
    center_lon = float(np.mean(all_lons))

    fig.update_geos(
        center=dict(lat=center_lat, lon=center_lon),
        projection_type="equirectangular",
        projection_scale=3.5,
        showland=True,
        landcolor="#1A2846",
        showocean=True,
        oceancolor="#080E1E",
        showcoastlines=True,
        coastlinecolor="#2D3E66",
        showcountries=True,
        countrycolor="#1D2A48",
        bgcolor="rgba(0,0,0,0)"
    )
    apply_dark_layout(fig,
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def create_development_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Renders horizontal stacked / sorted probability bar chart for Model 3 classification.
    """
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]

    palette = {
        'Strengthening': COLORS['accent_amber'],
        'Stable': COLORS['accent_cyan'],
        'Weakening': COLORS['accent_teal']
    }
    bar_colors = [palette.get(c, COLORS['cyan']) for c in classes]

    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color=COLORS['border_dark'], width=1)
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition='inside',
        textfont=dict(family="Fira Code, monospace", size=13, color="#080E1E")
    ))

    apply_dark_layout(fig,
        height=260,
        xaxis=dict(title="Model Output Probability (%)", range=[0, 105]),
        yaxis=dict(title="", tickfont=dict(size=12, color=COLORS['text_bright']))
    )
    return fig


def create_replay_map(
    track_df: pd.DataFrame,
    current_step: int
) -> go.Figure:
    """
    Renders cyclone track replay with historical steps, current step indicator, and remaining path.
    """
    fig = go.Figure()

    # Full track outline (faint)
    fig.add_trace(go.Scattergeo(
        lon=track_df['LON_NORM'],
        lat=track_df['LAT'],
        mode='lines',
        name="Complete Storm Lifecycle",
        line=dict(color="rgba(100, 116, 139, 0.4)", width=2, dash='dot'),
        hoverinfo='none'
    ))

    # Past path up to current step
    past_steps = track_df.iloc[:current_step + 1]
    fig.add_trace(go.Scattergeo(
        lon=past_steps['LON_NORM'],
        lat=past_steps['LAT'],
        mode='lines+markers',
        name="Observed Elapsed Path",
        line=dict(color=COLORS['cyan'], width=3),
        marker=dict(size=5, color=COLORS['cyan']),
        hoverinfo='text',
        text=[f"Step: {i+1}<br>Time: {t}<br>Wind: {w} kt<br>Lat: {la:.2f}°, Lon: {lo:.2f}°"
              for i, (t, w, la, lo) in enumerate(zip(
                  past_steps['ISO_TIME'],
                  past_steps.get('CURRENT_USA_WIND', [0]*len(past_steps)),
                  past_steps['LAT'],
                  past_steps['LON_NORM']
              ))]
    ))

    # Current Active Eye
    curr_point = track_df.iloc[current_step]
    fig.add_trace(go.Scattergeo(
        lon=[curr_point['LON_NORM']],
        lat=[curr_point['LAT']],
        mode='markers+text',
        name=f"Eye at Step {current_step + 1}",
        marker=dict(size=16, color=COLORS['accent_amber'], line=dict(color='#FFFFFF', width=2)),
        text=[f"Step {current_step + 1}"],
        textposition="top center",
        textfont=dict(family="Fira Code, monospace", color=COLORS['accent_amber'], size=12),
        hoverinfo='text',
        hovertext=(f"Active Eye (Step {current_step + 1})<br>"
                   f"Time: {curr_point['ISO_TIME']}<br>"
                   f"Wind: {curr_point.get('CURRENT_USA_WIND', 'N/A')} kt<br>"
                   f"Lat: {curr_point['LAT']:.2f}°, Lon: {curr_point['LON_NORM']:.2f}°")
    ))

    center_lat = float(track_df['LAT'].mean())
    center_lon = float(track_df['LON_NORM'].mean())

    fig.update_geos(
        center=dict(lat=center_lat, lon=center_lon),
        projection_type="equirectangular",
        projection_scale=2.8,
        showland=True,
        landcolor="#1A2846",
        showocean=True,
        oceancolor="#080E1E",
        showcoastlines=True,
        coastlinecolor="#2D3E66",
        showcountries=True,
        countrycolor="#1D2A48",
        bgcolor="rgba(0,0,0,0)"
    )

    apply_dark_layout(fig,
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def create_model_performance_charts() -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    Generates academic benchmark comparison figures from reported experiment results
    (Persistence / Constant-Velocity / Random Forest Baselines vs. XGBoost on 2020+ Test Set).
    """
    # 1. Model 1 MAE across horizons
    horizons = ["+6h", "+12h", "+24h"]
    pers_mae = [4.02, 7.50, 13.31]
    rf_mae = [3.52, 5.92, 10.01]
    xgb_mae = [3.43, 5.71, 9.58]

    fig_m1 = go.Figure()
    fig_m1.add_trace(go.Bar(
        x=horizons,
        y=pers_mae,
        name="Persistence Baseline",
        marker_color="rgba(100, 116, 139, 0.45)"
    ))
    fig_m1.add_trace(go.Bar(
        x=horizons,
        y=rf_mae,
        name="Random Forest (Task 6)",
        marker_color="rgba(148, 163, 184, 0.75)"
    ))
    fig_m1.add_trace(go.Bar(
        x=horizons,
        y=xgb_mae,
        name="XGBRegressor (Task 7)",
        marker_color=COLORS['cyan'],
        text=[f"{v:.2f} kt" for v in xgb_mae],
        textposition='outside',
        textfont=dict(family="Fira Code", size=10)
    ))
    apply_dark_layout(fig_m1,
        height=320,
        title="Model 1: Mean Absolute Error (MAE in kt) on Independent Test Set (2020+)",
        barmode='group',
        yaxis=dict(title="MAE (knots)")
    )

    # 2. Model 2 Haversine Error across horizons
    cv_hav = [38.8, 87.2, 208.4]
    rf_hav = [37.0, 80.5, 184.1]
    xgb_hav = [36.6, 79.6, 181.6]

    fig_m2 = go.Figure()
    fig_m2.add_trace(go.Bar(
        x=horizons,
        y=cv_hav,
        name="Constant-Velocity Baseline",
        marker_color="rgba(100, 116, 139, 0.45)"
    ))
    fig_m2.add_trace(go.Bar(
        x=horizons,
        y=rf_hav,
        name="Random Forest MultiOutput",
        marker_color="rgba(148, 163, 184, 0.75)"
    ))
    fig_m2.add_trace(go.Bar(
        x=horizons,
        y=xgb_hav,
        name="XGBoost MultiOutput",
        marker_color=COLORS['teal'],
        text=[f"{v:.1f} km" for v in xgb_hav],
        textposition='outside',
        textfont=dict(family="Fira Code", size=10)
    ))
    apply_dark_layout(fig_m2,
        height=320,
        title="Model 2: Mean Haversine Track Error (km) on Independent Test Set (2020+)",
        barmode='group',
        yaxis=dict(title="Track Error (km)")
    )

    # 3. Model 3 Classification Metrics
    metrics = ["Overall Accuracy (%)", "Macro F1 Score (x100)"]
    maj_cls = [41.3, 20.0]
    rf_cls = [62.71, 63.1]
    xgb_cls = [63.42, 63.8]

    fig_m3 = go.Figure()
    fig_m3.add_trace(go.Bar(
        x=metrics,
        y=maj_cls,
        name="Majority Class Baseline",
        marker_color="rgba(100, 116, 139, 0.45)"
    ))
    fig_m3.add_trace(go.Bar(
        x=metrics,
        y=rf_cls,
        name="Random Forest Classifier",
        marker_color="rgba(148, 163, 184, 0.75)"
    ))
    fig_m3.add_trace(go.Bar(
        x=metrics,
        y=xgb_cls,
        name="CycloneClassifier (XGBoost)",
        marker_color=COLORS['amber'],
        text=[f"{v:.1f}" for v in xgb_cls],
        textposition='outside',
        textfont=dict(family="Fira Code", size=10)
    ))
    apply_dark_layout(fig_m3,
        height=320,
        title="Model 3: Development Pattern Classification (2020+ Test Set)",
        barmode='group',
        yaxis=dict(title="Percentage / Score", range=[0, 75])
    )

    return fig_m1, fig_m2, fig_m3
