"""Artifact-verified performance and plain-language error analysis."""
import numpy as np
import pandas as pd
import streamlit as st
from dashboard.components import render_header, render_disclaimer, render_metric_card
from dashboard.benchmarks import load_benchmarks
from dashboard.charts import create_model_performance_charts, create_development_confusion


def render_page():
    render_header('Model Performance & Benchmarks',
        'Results from the saved models on the 2020+ test seasons. Lower error is better; higher accuracy and F1 are better.',
        'Saved evaluation results')
    render_disclaimer()
    results = load_benchmarks()
    records = results['records']
    final = {r['task']:r for r in records if r['horizon']==24}
    for col,task in zip(st.columns(3),(1,2,3)):
        r = final[task]
        with col:
            render_metric_card(f"Model {task} · +24h {r['metric']}", f"{r['value']:.2f} {r['unit']}",
                f"Baseline: {r['baseline']:.2f} {r['unit']}" +
                (f" · Macro-F1: {r['macro_f1']:.3f}" if task==3 else ''))
    st.caption('Generated (UTC): '+results['generated_utc']+' · Results refresh automatically when saved models or datasets change.')
    rows=[]
    for r in records:
        rows.append({'Task':f"Model {r['task']} · +{r['horizon']}h",'Metric':r['metric'],
            'Baseline':f"{r['baseline']:.2f} {r['unit']} ({r['baseline_name']})",
            'Current model':f"{r['value']:.2f} {r['unit']}",
            'Improvement':f"{r['value']-r['baseline']:+.2f} percentage points" if r['task']==3 else f"{r['skill_percent']:.2f}% lower error"})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    dev=results['development']
    if 'previous_model' in dev:
        old=dev['previous_model']
        st.info(f"Model 3 before → after: {old['accuracy']*100:.2f}% → {final[3]['value']:.2f}% accuracy; "
                f"macro-F1 {old['macro_f1']:.3f} → {final[3]['macro_f1']:.3f}. Selection used validation data.")
    tabs=st.tabs(['Error analysis · Model 3','Compare all models','Regional results','Evaluation method'])
    with tabs[0]:
        st.subheader('Where does Model 3 get confused?')
        mode=st.radio('Matrix display',['Row percentages','Counts'],horizontal=True)
        st.plotly_chart(create_development_confusion(dev,mode=='Row percentages'),width='stretch')
        st.caption('Rows are actual classes; columns are predictions. The diagonal contains correct predictions. Percentages sum to 100% within each row, apart from rounding.')
        matrix=np.asarray(dev['confusion_matrix'])
        mistakes=matrix.copy(); np.fill_diagonal(mistakes,0)
        i,j=np.unravel_index(mistakes.argmax(),mistakes.shape)
        st.write(f"Largest error group: **{int(matrix[i,j]):,} actual {dev['labels'][i]} observations** were predicted as **{dev['labels'][j]}**.")
        cls=[]
        for label in dev['labels']:
            r=dev['classification_report'][label]
            cls.append({'Class':label,'Precision':round(r['precision'],3),'Recall':round(r['recall'],3),
                        'F1':round(r['f1-score'],3),'Observations':int(r['support'])})
        st.dataframe(pd.DataFrame(cls),hide_index=True,width='stretch')
        st.caption('Recall: fraction of an actual class found correctly. Precision: fraction of predictions for a class that are correct. Macro-F1 gives each class equal weight.')
    with tabs[1]:
        for chart in create_model_performance_charts(results):
            st.plotly_chart(chart,width='stretch')
    with tabs[2]:
        task=st.selectbox('Regional task',[1,2,3],format_func=lambda n:f'Model {n} · +24h')
        regional=[]
        for r in results['basins']:
            if r['task']!=task: continue
            row={'Basin':r['basin'],'Observations':r['count']}
            if task==3: row.update({'Accuracy (%)':round(100*r['accuracy'],2),'Macro-F1':round(r['macro_f1'],3)})
            else: row.update({'Model error':round(r['value'],2),'Baseline error':round(r['baseline'],2),'Unit':'kt' if task==1 else 'km'})
            regional.append(row)
        st.dataframe(pd.DataFrame(regional),hide_index=True,width='stretch')
    with tabs[3]:
        st.write('Training uses seasons through 2014; validation uses 2015–2019; test uses 2020 onward. Cyclone IDs are disjoint across partitions. Season boundaries may cross calendar years.')
        st.write('Development experiments use 2015–2016 for early stopping and 2017–2019 for model selection. Test results are calculated after selection. This test set was reported previously, so it is not a new blind holdout.')
        st.write('The simple baselines hold wind constant, continue recent track velocity, or always predict the most common training class. These results are from recorded observations, not an operational real-time trial.')
        st.write('Model and dataset SHA-256 fingerprints are stored with the scores. Replacing an artifact invalidates the cached results.')
        st.json(results['split_checks'])
    st.download_button('Download evaluation results',data=__import__('json').dumps(results,indent=2),
                       file_name='cyclone_benchmark_results.json',mime='application/json')
