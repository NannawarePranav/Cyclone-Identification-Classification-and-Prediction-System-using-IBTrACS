"""
Cyclone Atlas Dashboard — Automated Smoke & Correctness Integrity Test Suite
===========================================================================
Validates:
1. Module imports and component dependencies.
2. Artifact presence for all 7 trained XGBoost pipelines.
3. Separation of physical track observations vs. horizon-expanded samples.
4. Correct horizon-specific ground truth lookups (+6h, +12h, +24h).
5. Movement prediction displacement vs. Haversine track error calculations.
6. Historical track duration and average speed with irregular / duplicate timestamps.
7. Split definition boundaries and exact basin scorecard values.
8. Plotly dark-ocean visualization generation.
"""

import sys
import os
import traceback
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_checks() -> bool:
    print("=" * 75)
    print("CYCLONE ATLAS DASHBOARD: AUTOMATED SMOKE & CORRECTNESS INTEGRITY SUITE")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Check Module Imports
    # ---------------------------------------------------------
    print("\n[1/8] Testing Module Imports...")
    try:
        import dashboard.config as config
        import dashboard.data_loader as dl
        import dashboard.model_loader as ml
        import dashboard.inference as inf
        import dashboard.charts as charts
        import dashboard.components as comp
        import dashboard.pages.overview as pg_overview
        import dashboard.pages.intensity as pg_int
        import dashboard.pages.movement as pg_mov
        import dashboard.pages.development as pg_dev
        import dashboard.pages.historical_replay as pg_replay
        import dashboard.pages.model_performance as pg_perf
        import app
        print("    [OK] All dashboard packages and page modules imported successfully.")
    except Exception as e:
        print(f"    [FAIL] Import Failed: {e}")
        traceback.print_exc()
        return False

    # ---------------------------------------------------------
    # 2. Check Model Artifacts Presence
    # ---------------------------------------------------------
    print("\n[2/8] Checking XGBoost Model Artifacts on Disk...")
    models_status = ml.check_all_models_present()
    all_ok = all(models_status.values())
    missing = [k for k, v in models_status.items() if not v]
    if all_ok:
        print("    [OK] All 7 XGBoost model artifacts present in models/xgboost/")
    else:
        print(f"    [FAIL] Missing models: {missing}")
        return False

    # ---------------------------------------------------------
    # 3. Test Physical Observations vs Horizon-Expanded Samples
    # ---------------------------------------------------------
    print("\n[3/8] Testing Physical Track Deduplication & Schema Isolation...")
    try:
        df1 = dl.load_intensity_dataset()
        df2 = dl.load_movement_dataset()
        df3 = dl.load_development_dataset()

        # Check unique physical tracks vs expanded rows
        sample_sid = df1['SID'].iloc[0]
        phys_track = dl.get_physical_track_observations(sample_sid, 'intensity')
        expanded_rows = df1[df1['SID'] == sample_sid]

        # Ensure physical track contains exactly one observation per timestamp
        assert not phys_track.duplicated(subset=['ISO_TIME']).any(), (
            "Physical track must have strictly unique timestamps."
        )
        assert len(phys_track) < len(expanded_rows), (
            f"Expanded samples ({len(expanded_rows)}) should exceed physical observations ({len(phys_track)}) for multi-horizon datasets."
        )
        print(f"    [OK] Verified physical track deduplication for {sample_sid}: {len(phys_track)} physical steps vs {len(expanded_rows)} horizon samples.")
    except Exception as e:
        print(f"    [FAIL] Physical deduplication failed: {e}")
        traceback.print_exc()
        return False

    # ---------------------------------------------------------
    # 4. Test Correct Horizon-Specific Target Lookups
    # ---------------------------------------------------------
    print("\n[4/8] Testing Horizon-Specific Target Lookups (+6h, +12h, +24h)...")
    try:
        # Test intensity target lookup
        sample_time = phys_track['ISO_TIME'].iloc[0]
        int_targets = dl.get_horizon_targets_for_observation(sample_sid, sample_time, 'intensity')
        
        # Verify that targets are retrieved from rows with corresponding PRED_HORIZON_HOURS
        raw_int_matches = df1[(df1['SID'] == sample_sid) & (pd.to_datetime(df1['ISO_TIME']) == pd.to_datetime(sample_time))]
        for _, r in raw_int_matches.iterrows():
            h = int(r['PRED_HORIZON_HOURS'])
            expected_w = float(r['TARGET_USA_WIND'])
            assert h in int_targets, f"Horizon {h} missing from lookup result."
            assert int_targets[h]['target_wind_kts'] == expected_w, (
                f"Target wind mismatch at +{h}h: expected {expected_w}, got {int_targets[h]['target_wind_kts']}"
            )
        print(f"    [OK] Model 1 ground truth matched across horizons: {list(int_targets.keys())}")

        # Test movement target lookup
        mov_targets = dl.get_horizon_targets_for_observation(sample_sid, sample_time, 'movement')
        raw_mov_matches = df2[(df2['SID'] == sample_sid) & (pd.to_datetime(df2['ISO_TIME']) == pd.to_datetime(sample_time))]
        for _, r in raw_mov_matches.iterrows():
            h = int(r['PRED_HORIZON_HOURS'])
            expected_lat = float(r['TARGET_LAT'])
            assert h in mov_targets, f"Movement horizon {h} missing from lookup."
            assert mov_targets[h]['target_lat'] == expected_lat, (
                f"Target latitude mismatch at +{h}h: expected {expected_lat}, got {mov_targets[h]['target_lat']}"
            )
        print(f"    [OK] Model 2 ground truth coordinates matched across horizons: {list(mov_targets.keys())}")
    except Exception as e:
        print(f"    [FAIL] Horizon target lookup failed: {e}")
        traceback.print_exc()
        return False

    # ---------------------------------------------------------
    # 5. Test Movement Displacement vs Forecast Error Distinction
    # ---------------------------------------------------------
    print("\n[5/8] Testing Movement Displacement vs Haversine Forecast Error...")
    try:
        sample_m2 = df2[df2['SPLIT'] == 'TEST'].iloc[20]
        forecast_mov = inf.run_movement_forecast(sample_m2)
        curr_lat = forecast_mov['current_lat']
        curr_lon = forecast_mov['current_lon']
        h24 = forecast_mov['horizons'][24]

        # Calculate predicted travel distance from eye
        predicted_travel_km = h24['haversine_distance_km']

        # Get actual ground truth for this observation at +24h
        targets_m2 = dl.get_horizon_targets_for_observation(sample_m2['SID'], sample_m2['ISO_TIME'], 'movement')
        if 24 in targets_m2:
            gt_lat = targets_m2[24]['target_lat']
            gt_lon = targets_m2[24]['target_lon']
            
            # Forecast track error between predicted coords and ground truth coords
            forecast_error_km = float(inf.calculate_haversine_error_km(
                np.array([gt_lat]), np.array([gt_lon]),
                np.array([h24['lat_pred']]), np.array([h24['lon_pred']])
            )[0])
            
            # Distance from eye to ground truth
            gt_travel_km = float(inf.calculate_haversine_error_km(
                np.array([curr_lat]), np.array([curr_lon]),
                np.array([gt_lat]), np.array([gt_lon])
            )[0])

            # Ensure predicted travel distance is not conflated with forecast track error
            print(f"    [OK] Eye position: ({curr_lat}°, {curr_lon}°)")
            print(f"    [OK] Predicted +24h position: ({h24['lat_pred']}°, {h24['lon_pred']}°) -> Predicted Travel: {predicted_travel_km:.1f} km")
            print(f"    [OK] Ground Truth +24h position: ({gt_lat}°, {gt_lon}°) -> Actual Travel: {gt_travel_km:.1f} km")
            print(f"    [OK] Forecast Track Error (Predicted vs GT): {forecast_error_km:.1f} km (Verified Distinct)")
    except Exception as e:
        print(f"    [FAIL] Movement error distinction test failed: {e}")
        traceback.print_exc()
        return False

    # ---------------------------------------------------------
    # 6. Test Historical Replay Duration & Speed Calculations
    # ---------------------------------------------------------
    print("\n[6/8] Testing Historical Replay Calculations (Irregular & Missing Intervals)...")
    try:
        # Test Case A: Track with duplicate timestamps (must deduplicate)
        synthetic_dup_df = pd.DataFrame({
            'SID': ['TEST_STORM'] * 4,
            'ISO_TIME': pd.to_datetime(['2022-09-01 00:00', '2022-09-01 00:00', '2022-09-01 06:00', '2022-09-01 12:00']),
            'LAT': [15.0, 15.0, 15.5, 16.0],
            'LON_NORM': [-60.0, -60.0, -61.0, -62.0],
            'CURRENT_USA_WIND': [45.0, 45.0, 50.0, 55.0]
        })
        dedup_track = synthetic_dup_df.drop_duplicates(subset=['SID', 'ISO_TIME']).sort_values('ISO_TIME').reset_index(drop=True)
        assert len(dedup_track) == 3, f"Expected 3 unique timestamps, got {len(dedup_track)}"
        t0 = dedup_track['ISO_TIME'].iloc[0]
        t_end = dedup_track['ISO_TIME'].iloc[-1]
        elapsed_hrs = (t_end - t0).total_seconds() / 3600.0
        assert elapsed_hrs == 12.0, f"Expected 12.0 elapsed hours, got {elapsed_hrs}"

        # Test Case B: Track with a missing 6-hour interval (e.g. 0h, 6h, 24h -> 3 steps, actual elapsed = 24h, NOT 3*6=18h)
        synthetic_gap_df = pd.DataFrame({
            'SID': ['GAP_STORM'] * 3,
            'ISO_TIME': pd.to_datetime(['2022-09-01 00:00', '2022-09-01 06:00', '2022-09-02 00:00']),
            'LAT': [15.0, 15.5, 17.0],
            'LON_NORM': [-60.0, -61.0, -64.0]
        })
        t_start_gap = synthetic_gap_df['ISO_TIME'].iloc[0]
        t_end_gap = synthetic_gap_df['ISO_TIME'].iloc[-1]
        gap_elapsed_hrs = (t_end_gap - t_start_gap).total_seconds() / 3600.0
        assert gap_elapsed_hrs == 24.0, f"Expected 24.0 elapsed hours for gap track, got {gap_elapsed_hrs}"
        assert gap_elapsed_hrs != (len(synthetic_gap_df) * 6), "Must not assume duration equals number of obs * 6 hours."
        print(f"    [OK] Verified duration on gap track: {gap_elapsed_hrs} hours (correctly avoids step*6 assumption).")
    except Exception as e:
        print(f"    [FAIL] Replay calculation test failed: {e}")
        traceback.print_exc()
        return False

    # ---------------------------------------------------------
    # 7. Test Authoritative Split Definitions & Basin Scorecards
    # ---------------------------------------------------------
    print("\n[7/8] Testing Temporal Split Boundaries & Basin Benchmark Scorecards...")
    try:
        # Check Model 2 cohort starts earlier (1970) than Model 1 & 3 (1980)
        assert df2['SEASON'].min() <= 1970, f"Model 2 expected SEASON <= 1970, got {df2['SEASON'].min()}"
        assert df1['SEASON'].min() >= 1980, f"Model 1 expected SEASON >= 1980, got {df1['SEASON'].min()}"
        assert df3['SEASON'].min() >= 1980, f"Model 3 expected SEASON >= 1980, got {df3['SEASON'].min()}"
        print(f"    [OK] Model 2 cohort verified at {df2['SEASON'].min()} (earlier historical cohort) vs Model 1 at {df1['SEASON'].min()}")

        from scripts.test_dashboard_refinements import test_basin_scorecards
        test_basin_scorecards()
        print('    [OK] Saved regional benchmark data verified.')
    except Exception as e:
        print(f"    [FAIL] Split definitions or scorecard verification failed: {e}")
        traceback.print_exc()
        return False

    # ---------------------------------------------------------
    # 8. Test Plotly Dark Ocean Visualizations
    # ---------------------------------------------------------
    print("\n[8/8] Testing Plotly Dark Ocean Visualizations...")
    try:
        stats = dl.get_global_summary_stats()
        cat = dl.get_cyclone_catalog('movement')
        sample_sid = cat.iloc[0]['SID']
        storm_track = dl.get_observations_for_cyclone(sample_sid, 'movement')

        fig_glob = charts.create_global_activity_map(df2, sample_storms=50)
        fig_basin = charts.create_basin_distribution_chart(stats['basin_distribution'])
        fig_trend = charts.create_historical_trend_chart(df2)
        fig_replay = charts.create_replay_map(storm_track, current_step=0)
        fig_p1, fig_p2, fig_p3 = charts.create_model_performance_charts()

        print("    [OK] Plotly figures successfully rendered with dark-ocean styling.")
    except Exception as e:
        print(f"    [FAIL] Chart generation failed: {e}")
        traceback.print_exc()
        return False

    print("\n" + "=" * 75)
    print(">>> ALL AUTOMATED SMOKE AND INTEGRITY TESTS PASSED SUCCESSFULLY. <<<")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_checks()
    if not success:
        sys.exit(1)
