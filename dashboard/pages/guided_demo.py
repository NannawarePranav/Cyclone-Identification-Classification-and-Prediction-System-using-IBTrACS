"""An outcome-blind, staged demonstration using a shared test-storm observation."""
import json
import numpy as np
import pandas as pd
import streamlit as st
from dashboard.components import render_header, render_disclaimer, render_pipeline_diagram, render_cyclone_selector
from dashboard.data_loader import (load_intensity_dataset, load_movement_dataset, load_development_dataset,
    load_cyclone_names, get_horizon_targets_for_observation)
from dashboard.inference import run_intensity_forecast, run_movement_forecast, run_development_forecast
from dashboard.charts import create_development_probability_chart, create_trajectory_map
from cyclone_ml.geodesy import calculate_haversine_error_km


@st.cache_data(show_spinner=False)
def demo_observations():
    keys=['SID','ISO_TIME']
    common=None
    for data in (load_intensity_dataset(),load_movement_dataset()):
        counts=data[data.SPLIT.eq('TEST')].groupby(keys).PRED_HORIZON_HOURS.nunique()
        valid=counts[counts.eq(3)].reset_index()[keys]
        common=valid if common is None else common.merge(valid,on=keys,validate='one_to_one')
    dev=load_development_dataset()
    return dev[dev.SPLIT.eq('TEST')].merge(common,on=keys,validate='one_to_one').sort_values(keys)


def render_page():
    render_header('Guided Presentation Demo','One cyclone, one observation, three prediction tasks. Reveal the recorded outcome after predicting.','Classroom walkthrough')
    render_disclaimer()
    with st.expander('Explain the system',expanded=True):
        render_pipeline_diagram()
    data=demo_observations()
    names=load_cyclone_names()
    if data.empty:
        st.warning('No common test observations are available across all three tasks.')
        return
    st.caption('The demo requires TEST observations with all three forecast horizons available for every task. Shared filters apply here too; use TEST or ALL partitions and seasons from 2020 onward.')
    sid, rows, row = render_cyclone_selector('development', 'demo', eligible_observations=data)
    if row is None:
        return
    st.caption('This cyclone is in TEST and was excluded from training and validation. The shared observation is used when eligible; an unavailable timestamp is explained above.')
    step=st.radio('Presentation step',['1 · Observe','2 · Predict','3 · Reveal outcome','4 · Discuss'],
                  horizontal=True,key=f'demo_step_{sid}_{row.ISO_TIME}')
    st.write(f"**Starting state:** {row.CURRENT_USA_WIND:.0f} kt sustained wind at {row.LAT:.1f}°, {row.LON_NORM:.1f}°. All displayed times are UTC.")
    if step.startswith('1'):
        st.info('Ask the audience: will this cyclone strengthen, stay stable, or weaken? Then choose “2 · Predict”.')
        return
    movement=load_movement_dataset()
    move_row=movement[movement.SID.eq(sid) & movement.ISO_TIME.eq(row.ISO_TIME)].iloc[0]
    intensity=run_intensity_forecast(row)
    track=run_movement_forecast(move_row)
    development=run_development_forecast(row)
    st.subheader('The three predictions')
    for column,title,value in zip(st.columns(3),['Wind at +24h','Position at +24h','Development at +24h'],
        [f"{intensity['predictions_kts'][24]:.1f} kt",f"{track['horizons'][24]['lat_pred']:.2f}°, {track['horizons'][24]['lon_pred']:.2f}°",development['predicted_class']]):
        with column: st.metric(title,value)
    st.plotly_chart(create_development_probability_chart(development['probabilities']),width='stretch')
    st.caption('Class scores are uncalibrated model probabilities.')
    reveal=step.startswith(('3','4'))
    # Show only pre-forecast track points until the user explicitly reveals the outcome.
    past=movement[movement.SID.eq(sid) & (movement.ISO_TIME<=row.ISO_TIME)].drop_duplicates('ISO_TIME').sort_values('ISO_TIME').tail(8)
    targets=get_horizon_targets_for_observation(sid,row.ISO_TIME,'movement') if reveal else None
    map_figure=create_trajectory_map(track['current_lat'],track['current_lon'],track['horizons'],past_track=past)
    if reveal:
        map_figure.add_scattergeo(lat=[targets[h]['target_lat'] for h in (6,12,24)],
            lon=[targets[h]['target_lon'] for h in (6,12,24)],mode='lines+markers',
            name='Recorded future track',line=dict(color='#F1F5F9',dash='dash'),
            text=['+6h actual','+12h actual','+24h actual'])
    st.plotly_chart(map_figure,width='stretch')
    if reveal:
        intensity_targets=get_horizon_targets_for_observation(sid,row.ISO_TIME,'intensity')
        comparison=[]
        for h in (6,12,24):
            pred=track['horizons'][h]; actual=targets[h]
            error=calculate_haversine_error_km(np.array([actual['target_lat']]),np.array([actual['target_lon']]),
                np.array([pred['lat_pred']]),np.array([pred['lon_pred']]))[0]
            wind=intensity_targets[h]['target_wind_kts']
            comparison.append({'Horizon':f'+{h}h','Predicted wind (kt)':intensity['predictions_kts'][h],
                'Recorded wind (kt)':wind,'Wind error (kt)':round(abs(wind-intensity['predictions_kts'][h]),1),
                'Track error (km)':round(float(error),1)})
        st.dataframe(pd.DataFrame(comparison),hide_index=True,width='stretch')
        st.write(f"**Actual development:** {row.CLASS_LABEL} ({row.WIND_CHANGE_24H:+.0f} kt in 24 hours). "
                 f"**Prediction:** {development['predicted_class']}.")
        if step.startswith('4'):
            st.info('Discuss both matches and mistakes. One storm illustrates behavior; the full test-set confusion matrix measures performance. These tabular features do not include sea-surface temperature or environmental wind shear.')
        st.download_button('Download this demo result',json.dumps({'storm_id':sid,'name':names.get(sid),
            'observation_utc':str(row.ISO_TIME),'intensity':intensity,'movement':track,'development':development,
            'actual_development':row.CLASS_LABEL,'comparison':comparison},indent=2),
            file_name=f'demo_{sid}.json',mime='application/json')
