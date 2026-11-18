"""Exercise actual Streamlit widget lifecycle across model-page switches."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dashboard.components import SELECTION_STATE_KEY

APP = '''
import streamlit as st
from dashboard.components import render_cyclone_selector
page = st.radio('Page', ['intensity','movement','development','replay','demo','other'], key='page')
if page == 'replay':
    from dashboard.pages.historical_replay import render_page
    render_page()
elif page == 'demo':
    from scripts.test_shared_selection import observations
    render_cyclone_selector('development', 'demo', eligible_observations=observations('A','development').iloc[:1])
elif page != 'other':
    sid, observations, row = render_cyclone_selector(page, page)
    if row is not None:
        st.session_state['displayed'] = (sid, str(row.ISO_TIME))
else:
    st.write('Another page without selection widgets')
'''


def catalog(kind):
    entries = [('A','AMPHAN','NI',2020,'TEST'), ('B','BETA','NI',2020,'TEST'),
               ('D','DELTA','NA',2010,'TRAIN')]
    if kind=='movement':
        entries += [('C','MOVEMENT ONLY','NI',2020,'TEST'), ('OLD','OLD','NA',1975,'TRAIN')]
    return pd.DataFrame([{'SID':sid,'NAME':name,'BASIN':basin,'SEASON':season,
                          'SPLIT':split,'OBS_COUNT':3} for sid,name,basin,season,split in entries])


def observations(sid, dataset_type):
    hours = [0,6,12,18] if dataset_type=='movement' else [0,12,18]
    if dataset_type=='development':
        hours = [12,18]  # Missing both the earliest and an intermediate observation.
    return pd.DataFrame([{'SID':sid,'ISO_TIME':pd.Timestamp('2020-05-17')+pd.Timedelta(hours=h),
                          'LAT':12.,'LON_NORM':86.,'CURRENT_USA_WIND':40.,
                          'BASIN':'NI','SPLIT':'TEST'} for h in hours])


class SharedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch('dashboard.components.get_cyclone_catalog', side_effect=catalog),
                        patch('dashboard.components.get_observations_for_cyclone', side_effect=observations)]
        for mock in self.patches:
            mock.start(); self.addCleanup(mock.stop)
        self.app = AppTest.from_string(APP,default_timeout=20).run()
        self.assertFalse(self.app.exception)

    def go(self,page):
        self.app.radio(key='page').set_value(page).run()
        self.assertFalse(self.app.exception)

    def timestamp(self,page,hour):
        timestamp=(pd.Timestamp('2020-05-17',tz='UTC')+pd.Timedelta(hours=hour)).isoformat()
        self.app.selectbox(key=f'{page}_timestamp').set_value(timestamp).run()
        self.assertFalse(self.app.exception)
        return timestamp

    def test_filters_and_exact_time_across_all_models(self):
        self.app.selectbox(key='intensity_basin').set_value('NI').run()
        self.app.text_input(key='intensity_search').set_value('BETA').run()
        chosen=self.timestamp('intensity',18)
        for page in ['movement','development','intensity']:
            self.go(page)
            self.assertEqual(self.app.selectbox(key=f'{page}_sid').value,'B')
            self.assertEqual(self.app.selectbox(key=f'{page}_timestamp').value,chosen)
            self.assertEqual(self.app.selectbox(key=f'{page}_basin').value,'NI')
            self.assertEqual(self.app.text_input(key=f'{page}_search').value,'BETA')
            self.assertEqual(tuple(self.app.slider(key=f'{page}_season').value),(2020,2020))

    def test_latest_edit_wins_and_survives_other_page(self):
        self.go('movement')
        self.app.selectbox(key='movement_sid').set_value('B').run()
        chosen=self.timestamp('movement',18)
        self.go('other'); self.go('intensity')
        self.assertEqual(self.app.selectbox(key='intensity_sid').value,'B')
        self.assertEqual(self.app.selectbox(key='intensity_timestamp').value,chosen)

    def test_replay_controls_and_models_share_timestamp(self):
        self.app.text_input(key='intensity_search').set_value('BETA').run()
        chosen = self.timestamp('intensity',12)
        self.go('replay')
        self.assertEqual(self.app.selectbox(key='replay_timestamp').value,chosen)
        self.assertEqual(self.app.text_input(key='replay_search').value,'BETA')
        self.assertEqual(self.app.slider(key='replay_step').value,2)
        next(b for b in self.app.button if b.label=='Next >').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.slider(key='replay_step').value,3)
        self.go('development')
        self.assertEqual(pd.Timestamp(self.app.selectbox(key='development_timestamp').value).hour,18)
        self.go('replay')
        next(b for b in self.app.button if b.label=='< Prev').click().run()
        self.assertEqual(self.app.slider(key='replay_step').value,2)
        self.app.slider(key='replay_step').set_value(0).run()
        self.assertFalse(self.app.exception)
        self.go('intensity')
        self.assertEqual(pd.Timestamp(self.app.selectbox(key='intensity_timestamp').value).hour,0)
        self.timestamp('intensity',18)
        self.go('replay')
        self.assertEqual(self.app.slider(key='replay_step').value,3)

    def test_single_observation_replay(self):
        with patch('dashboard.components.get_observations_for_cyclone',
                   side_effect=lambda sid,dataset_type: observations(sid,dataset_type).iloc[:1]):
            self.go('replay')
            self.assertTrue(next(b for b in self.app.button if b.label=='Next >').disabled)
            self.assertTrue(next(b for b in self.app.button if b.label=='< Prev').disabled)

    def test_restricted_demo_preserves_unavailable_selection(self):
        chosen = self.timestamp('intensity',18)
        self.go('demo')
        self.assertEqual(pd.Timestamp(self.app.selectbox(key='demo_timestamp').value).hour,12)
        self.assertEqual(self.app.session_state[SELECTION_STATE_KEY]['timestamp'],chosen)
        self.go('intensity')
        self.assertEqual(self.app.selectbox(key='intensity_timestamp').value,chosen)
        self.app.selectbox(key='intensity_sid').set_value('B').run()
        self.go('demo')
        self.assertIsNone(self.app.selectbox(key='demo_sid').value)
        self.assertEqual(self.app.session_state[SELECTION_STATE_KEY]['sid'],'B')

    def test_restricted_demo_empty_filters_can_recover(self):
        self.go('demo')
        self.app.text_input(key='demo_search').set_value('no-match').run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.warning)
        self.app.text_input(key='demo_search').set_value('AMPHAN').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.selectbox(key='demo_sid').value,'A')

    def test_missing_time_uses_previous_and_restores_requested_time(self):
        self.go('movement'); requested=self.timestamp('movement',6)
        self.go('intensity')
        self.assertEqual(pd.Timestamp(self.app.selectbox(key='intensity_timestamp').value).hour,0)
        self.assertTrue(self.app.info)
        self.assertEqual(self.app.session_state[SELECTION_STATE_KEY]['timestamp'],requested)
        self.go('movement')
        self.assertEqual(self.app.selectbox(key='movement_timestamp').value,requested)

    def test_no_earlier_observation_and_explicit_override(self):
        requested=self.timestamp('intensity',0)
        self.go('development')
        self.assertEqual(pd.Timestamp(self.app.selectbox(key='development_timestamp').value).hour,12)
        self.assertEqual(self.app.session_state[SELECTION_STATE_KEY]['timestamp'],requested)
        updated=self.timestamp('development',18)
        self.go('intensity')
        self.assertEqual(self.app.selectbox(key='intensity_timestamp').value,updated)

    def test_unavailable_storm_is_not_silently_replaced(self):
        self.go('movement')
        self.app.selectbox(key='movement_sid').set_value('C').run()
        self.go('intensity')
        self.assertIsNone(self.app.selectbox(key='intensity_sid').value)
        self.assertTrue(self.app.info)
        self.assertEqual(self.app.session_state[SELECTION_STATE_KEY]['sid'],'C')
        self.go('movement')
        self.assertEqual(self.app.selectbox(key='movement_sid').value,'C')

    def test_partition_codes_survive_different_cohort_labels(self):
        self.app.slider(key='intensity_season').set_range(1975,2010).run()
        self.app.selectbox(key='intensity_split').set_value('TRAIN').run()
        self.go('movement')
        self.assertEqual(self.app.selectbox(key='movement_split').value,'TRAIN')
        self.assertEqual(tuple(self.app.slider(key='movement_season').value),(1975,2010))
        self.assertEqual(self.app.selectbox(key='movement_sid').value,'D')

    def test_empty_search_persists_and_recovers(self):
        self.app.text_input(key='intensity_search').set_value('no-such-storm').run()
        self.go('movement')
        self.assertTrue(self.app.warning)
        self.assertEqual(self.app.text_input(key='movement_search').value,'no-such-storm')
        self.app.text_input(key='movement_search').set_value('AMPHAN').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.selectbox(key='movement_sid').value,'A')


if __name__=='__main__':
    unittest.main()
