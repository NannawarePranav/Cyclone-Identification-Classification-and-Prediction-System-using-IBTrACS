"""Reproducible validation-only selection; preserve incumbent and evaluate test once."""
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime, timezone
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cyclone_ml import compute_cyclical_features, CycloneClassifier
from cyclone_ml.development_features import enrich_development, BASE_NUM, HISTORY_NUM, PRESSURE_NUM, CAT


def scores(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=24)
    args = parser.parse_args()
    output = ROOT / 'models/experiments/development_v2'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'selection.json').exists():
        raise SystemExit('Completed experiment already exists; preserve it and use a new experiment directory for another study.')
    raw = pd.read_csv(ROOT/'raw/ibtracs.ALL.list.v04r01.csv', skiprows=[1], keep_default_na=False,
        usecols=['SID','NAME','ISO_TIME','USA_AGENCY','USA_WIND','USA_PRES','DIST2LAND','IFLAG','TRACK_TYPE'], low_memory=False)
    for col in ['SID','NAME','USA_AGENCY','IFLAG','TRACK_TYPE']:
        raw[col] = raw[col].str.strip()
    names = raw.loc[~raw.NAME.isin(['', 'UNNAMED', 'NOT_NAMED']) & raw.TRACK_TYPE.eq('main'), ['SID','NAME']]
    names = names.drop_duplicates('SID')
    names.to_csv(ROOT/'processed/cyclone_names.csv',index=False)
    base = pd.read_parquet(ROOT/'processed/model3_development.parquet')
    df = enrich_development(base, raw)
    df.to_parquet(ROOT/'processed/model3_development_enriched.parquet',index=False)
    df = compute_cyclical_features(df)
    train = df[df.SPLIT.eq('train')]
    val = df[df.SPLIT.eq('val')]
    stop = val.SEASON.le(2016).to_numpy()
    select = ~stop
    classes = np.array(sorted(train.CLASS_LABEL.unique()))
    ytrain = pd.Categorical(train.CLASS_LABEL, categories=classes).codes
    ystop = pd.Categorical(val.loc[stop,'CLASS_LABEL'], categories=classes).codes
    incumbent = joblib.load(ROOT/'models/xgboost/m3_xgboost.joblib')
    pred_inc = incumbent.predict(val)
    baseline = {'validation': scores(val.CLASS_LABEL,pred_inc),
                'selection': scores(val.loc[select,'CLASS_LABEL'], pred_inc[select])}
    def yearly(pred):
        return {str(year): scores(val.loc[val.SEASON.eq(year),'CLASS_LABEL'],pred[val.SEASON.eq(year)])
                for year in sorted(val.SEASON.unique())}
    baseline['yearly'] = yearly(pred_inc)
    baseline_mean = np.mean([baseline['yearly'][str(y)]['macro_f1'] for y in (2017,2018,2019)])
    print('Incumbent:', json.dumps(baseline), flush=True)
    coverage = {split: {col: float(g[col].notna().mean()) for col in ['CURRENT_USA_PRES','PAST_USA_WIND_24H','PAST_USA_WIND_48H']}
                for split,g in df.groupby('SPLIT')}
    print('Feature availability:',json.dumps(coverage),flush=True)
    configs = list(ParameterSampler({
        'max_depth':[3,4,5,6,7], 'learning_rate':[0.025,0.05,0.08],
        'min_child_weight':[1,5,12], 'subsample':[0.8,1.0],
        'colsample_bytree':[0.75,1.0], 'reg_lambda':[5.,15.,30.],
    }, n_iter=args.trials, random_state=2026))
    groups = {'base':BASE_NUM, 'history':BASE_NUM+HISTORY_NUM,
              'history_pressure':BASE_NUM+HISTORY_NUM+PRESSURE_NUM}
    candidates = []
    best = None
    best_score = baseline_mean
    # Same anchor configuration in each group is a controlled feature ablation.
    trials = [(g,dict(max_depth=5,learning_rate=.03,min_child_weight=1,subsample=1.,colsample_bytree=.8,reg_lambda=10.)) for g in groups]
    trials += [(list(groups)[i % 3], config) for i,config in enumerate(configs)]
    for index,(group,params) in enumerate(trials):
        features = groups[group]+CAT
        pre = ColumnTransformer([('num','passthrough',groups[group]),
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
        xt = pre.fit_transform(train[features])
        xv = pre.transform(val[features])
        estimator = XGBClassifier(**params,n_estimators=1600,early_stopping_rounds=60,
            tree_method='hist',eval_metric='mlogloss',n_jobs=4,random_state=42)
        estimator.fit(xt,ytrain,eval_set=[(xv[stop],ystop)],verbose=False)
        wrapper = CycloneClassifier(**params,n_estimators=estimator.best_iteration+1)
        wrapper.classes_ = classes
        wrapper.model_ = estimator
        pipeline = Pipeline([('preprocessor',pre),('classifier',wrapper)])
        pred = pipeline.predict(val)
        per_year = yearly(pred)
        mean_f1 = float(np.mean([per_year[str(y)]['macro_f1'] for y in (2017,2018,2019)]))
        record = {'trial':index+1,'features':group,'parameters':params,'trees':estimator.best_iteration+1,
            'selection':scores(val.loc[select,'CLASS_LABEL'],pred[select]),
            'validation':scores(val.CLASS_LABEL,pred),'yearly':per_year,'selection_year_mean_f1':mean_f1}
        candidates.append(record)
        print(json.dumps(record),flush=True)
        eligible = (record['selection']['accuracy'] >= baseline['selection']['accuracy'] and
                    record['validation']['macro_f1'] > baseline['validation']['macro_f1'] and
                    record['validation']['accuracy'] >= baseline['validation']['accuracy'])
        if eligible and mean_f1 > best_score + 0.002:
            best_score = mean_f1
            best = record
            joblib.dump(pipeline,output/'candidate.joblib')
        (output/'trials.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    # Lock selection before accessing test labels or predictions.
    selection = {'created_utc':datetime.now(timezone.utc).isoformat(), 'incumbent':baseline,
        'selected':best,'coverage':coverage,'trial_count':len(candidates),
        'protocol':'TRAIN <=2014; early stopping VAL 2015-2016; selection mean yearly macro-F1 2017-2019, with non-decreasing validation accuracy; TEST evaluated only after selection. A candidate replaces the current leader only with >0.002 additional mean yearly macro-F1; validation accuracy must not decrease.',
        'test_caveat':'The existing test set was previously reported; this is a follow-up evaluation, not a new blind holdout.'}
    (output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    if best:
        archive = ROOT/'models/archive/pre_development_v2'
        archive.mkdir(parents=True,exist_ok=True)
        old = archive/'m3_xgboost.joblib'
        if not old.exists():
            shutil.copy2(ROOT/'models/xgboost/m3_xgboost.joblib',old)
        shutil.copy2(output/'candidate.joblib',ROOT/'models/xgboost/m3_xgboost.joblib')
    active = joblib.load(ROOT/'models/xgboost/m3_xgboost.joblib')
    test = df[df.SPLIT.eq('test')]
    selection['test'] = scores(test.CLASS_LABEL,active.predict(test))
    selection['incumbent_test'] = scores(test.CLASS_LABEL,incumbent.predict(test))
    (output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('FINAL:',json.dumps(selection),flush=True)


if __name__ == '__main__':
    main()
