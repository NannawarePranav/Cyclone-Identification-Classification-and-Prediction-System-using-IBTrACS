"""Regression checks for temporal isolation, serving parity and result provenance."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from cyclone_ml.development_features import enrich_development, HISTORY_NUM, PRESSURE_NUM
from cyclone_ml.benchmarks import fingerprint, inputs
from dashboard.data_loader import load_development_dataset, load_cyclone_names
from dashboard.inference import run_development_forecast
from dashboard.benchmarks import load_benchmarks
from dashboard.pages.guided_demo import demo_observations


class UpgradeTests(unittest.TestCase):
    def fixture(self):
        raw=pd.DataFrame({'SID':['storm']*4,'ISO_TIME':['2020-01-01 00:00:00','2020-01-01 12:00:00','2020-01-02 00:00:00','2020-01-03 00:00:00'],
            'USA_AGENCY':['a']*4,'USA_WIND':[20,25,40,100],'USA_PRES':[1000,990,980,900],
            'DIST2LAND':[100,90,80,10],'TRACK_TYPE':['main']*4,'IFLAG':['O']*4})
        row=pd.DataFrame({'SID':['storm'],'ISO_TIME':['2020-01-02 00:00:00'],'USA_AGENCY':['a'],
            'CURRENT_USA_WIND':[40.],'PAST_USA_WIND_6H':[35.],'PAST_USA_WIND_12H':[25.],'DIST2LAND':[80.]})
        return row,raw

    def test_future_records_do_not_change_features(self):
        row,raw=self.fixture()
        before=enrich_development(row,raw)
        raw.loc[3,['USA_WIND','USA_PRES','DIST2LAND']]=[200,800,0]
        after=enrich_development(row,raw)
        pd.testing.assert_frame_equal(before,after)
        self.assertEqual(before.PAST_USA_WIND_24H.iloc[0],20)
        self.assertEqual(before.PRESSURE_CHANGE_PAST_24H.iloc[0],-20)
        self.assertEqual(before.WIND_ACCELERATION_6H.iloc[0],-5)

    def test_exact_time_agency_and_quality(self):
        row,raw=self.fixture()
        raw.loc[0,'USA_AGENCY']='other'
        out=enrich_development(row,raw)
        self.assertTrue(pd.isna(out.PAST_USA_WIND_24H.iloc[0]))
        raw.loc[0,'USA_AGENCY']='a'; raw.loc[0,'IFLAG']='P'
        self.assertTrue(pd.isna(enrich_development(row,raw).PAST_USA_WIND_24H.iloc[0]))
        raw.loc[0,'IFLAG']='O'; raw.loc[0,'ISO_TIME']='2020-01-01 01:00:00'
        self.assertTrue(pd.isna(enrich_development(row,raw).PAST_USA_WIND_24H.iloc[0]))

    def test_cohort_and_targets_preserved(self):
        before=pd.read_parquet(ROOT/'processed/model3_development.parquet')
        after=pd.read_parquet(ROOT/'processed/model3_development_enriched.parquet')
        pd.testing.assert_frame_equal(before,after[before.columns])
        self.assertTrue(set(HISTORY_NUM+PRESSURE_NUM)<=set(after.columns))

    def test_serving_and_target_isolation(self):
        data=load_development_dataset()
        model=joblib.load(ROOT/'models/xgboost/m3_xgboost.joblib')
        forbidden={'SID','ISO_TIME','SPLIT','CLASS_LABEL','CLASS_ID','WIND_CHANGE_24H'}
        self.assertFalse(set(model.feature_names_in_)&forbidden)
        self.assertFalse(any(f.startswith('TARGET_') for f in model.feature_names_in_))
        for row in [data[data.SPLIT.eq('TEST')].iloc[0],data[data.SPLIT.eq('TEST')].iloc[-1]]:
            expected=str(model.predict(pd.DataFrame([row]))[0])
            changed=row.copy(); changed['CLASS_LABEL']='corrupted'; changed['TARGET_USA_WIND_24H']=999
            self.assertEqual(run_development_forecast(changed)['predicted_class'],expected)

    def test_provenance_and_matrix_totals(self):
        result=load_benchmarks()
        for path in inputs(ROOT):
            key=str(path.relative_to(ROOT)).replace('\\','/')
            self.assertEqual(result['artifacts'][key],fingerprint(path))
        matrix=np.asarray(result['development']['confusion_matrix'])
        record=next(r for r in result['records'] if r['task']==3)
        self.assertEqual(int(matrix.sum()),record['count'])
        self.assertAlmostEqual(100*matrix.trace()/matrix.sum(),record['value'])

    def test_named_unseen_demo(self):
        data=demo_observations()
        self.assertEqual(set(data.SPLIT),{'TEST'})
        original=pd.read_parquet(ROOT/'processed/model3_development.parquet')
        self.assertFalse(set(data.SID)&set(original[original.SPLIT.ne('test')].SID))
        self.assertIn('AMPHAN',set(load_cyclone_names().values()))

    def test_results_refresh_after_artifact_change(self):
        import dashboard.benchmarks as benchmarks
        with patch.object(benchmarks,'fingerprint',return_value='changed-artifact'), \
             patch.object(benchmarks,'evaluate_all',return_value={'refreshed':True}) as refresh:
            benchmarks._load_verified.clear()
            result=benchmarks._load_verified(('synthetic-version',),0)
            self.assertEqual(result,{'refreshed':True})
            refresh.assert_called_once_with(ROOT)
            benchmarks._load_verified.clear()


if __name__=='__main__':
    unittest.main()
