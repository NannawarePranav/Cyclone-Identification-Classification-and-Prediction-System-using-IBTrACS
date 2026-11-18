"""Compute serving-model benchmarks with artifact and dataset provenance."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from .features import compute_cyclical_features
from .geodesy import reconstruct_coordinates, calculate_haversine_error_km

LABELS = ['Strengthening', 'Stable', 'Weakening']


def development_path(root):
    enriched = root/'processed/model3_development_enriched.parquet'
    return enriched if enriched.exists() else root/'processed/model3_development.parquet'


def inputs(root):
    paths = [*sorted((root/'models/xgboost').glob('*.joblib')),
             root/'processed/model1_intensity.parquet', root/'processed/model2_movement.parquet', development_path(root)]
    previous = root/'models/archive/pre_development_v2/m3_xgboost.joblib'
    if previous.exists():
        paths.append(previous)
    return paths


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def classification_metrics(y, predictions):
    return {'accuracy':float(accuracy_score(y,predictions)),
            'macro_f1':float(f1_score(y,predictions,average='macro',zero_division=0))}


def evaluate_all(root):
    root = Path(root)
    result = {'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),
              'artifacts':{str(p.relative_to(root)).replace('\\','/'):fingerprint(p) for p in inputs(root)},
              'records':[], 'basins':[], 'split_checks':{}}
    for task,name in [(1,'intensity'),(2,'movement'),(3,'development')]:
        path = development_path(root) if task == 3 else root/f'processed/model{task}_{name}.parquet'
        data = compute_cyclical_features(pd.read_parquet(path))
        storm_sets = {split:set(g.SID) for split,g in data.groupby('SPLIT')}
        disjoint = all(not (storm_sets[a]&storm_sets[b]) for a,b in [('train','val'),('train','test'),('val','test')])
        if not disjoint:
            raise ValueError(f'Model {task}: storm IDs overlap between partitions')
        result['split_checks'][str(task)] = {'disjoint_storm_ids':disjoint,'test_storms':len(storm_sets['test'])}
        test = data[data.SPLIT.eq('test')]
        for horizon in ([6,12,24] if task != 3 else [24]):
            rows = test[test.PRED_HORIZON_HOURS.eq(horizon)] if task != 3 else test
            suffix = f'_{horizon}h' if task != 3 else ''
            model = joblib.load(root/f'models/xgboost/m{task}_xgboost{suffix}.joblib')
            predictions = model.predict(rows)
            record = {'task':task,'horizon':horizon,'count':len(rows),'model':'XGBoost'}
            if task == 1:
                errors = np.abs(predictions-rows.TARGET_USA_WIND.to_numpy())
                simple_errors = np.abs(rows.CURRENT_USA_WIND-rows.TARGET_USA_WIND).to_numpy()
                record.update(metric='Wind MAE',unit='kt',value=float(errors.mean()),
                              baseline_name='Persistence',baseline=float(simple_errors.mean()))
            elif task == 2:
                lat,lon = reconstruct_coordinates(rows.LAT.to_numpy(),rows.LON_NORM.to_numpy(),predictions[:,0],predictions[:,1])
                errors = calculate_haversine_error_km(rows.TARGET_LAT.to_numpy(),rows.TARGET_LON_NORM.to_numpy(),lat,lon)
                dx = rows.PAST_EAST_DISP_6H_KM.fillna(0).to_numpy()*horizon/6
                dy = rows.PAST_NORTH_DISP_6H_KM.fillna(0).to_numpy()*horizon/6
                bl_lat,bl_lon = reconstruct_coordinates(rows.LAT.to_numpy(),rows.LON_NORM.to_numpy(),dx,dy)
                simple_errors = calculate_haversine_error_km(rows.TARGET_LAT.to_numpy(),rows.TARGET_LON_NORM.to_numpy(),bl_lat,bl_lon)
                record.update(metric='Mean track error',unit='km',value=float(errors.mean()),
                              baseline_name='Constant velocity',baseline=float(simple_errors.mean()))
            else:
                majority = data.loc[data.SPLIT.eq('train'),'CLASS_LABEL'].mode().iloc[0]
                baseline_pred = np.repeat(majority,len(rows))
                stats = classification_metrics(rows.CLASS_LABEL,predictions)
                base_stats = classification_metrics(rows.CLASS_LABEL,baseline_pred)
                record.update(metric='Accuracy',unit='%',value=stats['accuracy']*100,
                              baseline_name=f'Always {majority}',baseline=base_stats['accuracy']*100,
                              macro_f1=stats['macro_f1'],baseline_macro_f1=base_stats['macro_f1'])
                result['development'] = {'labels':LABELS,
                    'confusion_matrix':confusion_matrix(rows.CLASS_LABEL,predictions,labels=LABELS).tolist(),
                    'classification_report':classification_report(rows.CLASS_LABEL,predictions,labels=LABELS,output_dict=True,zero_division=0),
                    'features':model.feature_names_in_.tolist()}
                for basin in sorted(rows.BASIN.unique()):
                    mask = rows.BASIN.eq(basin).to_numpy()
                    result['basins'].append({'task':task,'basin':basin,'count':int(mask.sum()),
                        **classification_metrics(rows.loc[mask,'CLASS_LABEL'],predictions[mask])})
                previous = root/'models/archive/pre_development_v2/m3_xgboost.joblib'
                if previous.exists():
                    result['development']['previous_model'] = classification_metrics(rows.CLASS_LABEL,joblib.load(previous).predict(rows))
            if task != 3:
                record['skill_percent'] = float(100*(1-errors.mean()/simple_errors.mean()))
                if horizon == 24:
                    for basin in sorted(rows.BASIN.unique()):
                        mask = rows.BASIN.eq(basin).to_numpy()
                        result['basins'].append({'task':task,'basin':basin,'count':int(mask.sum()),
                            'value':float(errors[mask].mean()),'baseline':float(simple_errors[mask].mean())})
            result['records'].append(record)
    path = root/'reports/benchmark_results.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2),encoding='utf-8')
    temporary.replace(path)
    return result
