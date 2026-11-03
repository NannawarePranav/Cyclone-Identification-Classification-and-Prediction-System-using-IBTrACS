"""
Verification script for Cyclone Atlas Dashboard modules.
Tests imports, model loading, parquet data loading, inference pipeline, and chart rendering.
"""

import sys
import os
import traceback

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_checks():
    print("=" * 70)
    print("CYCLONE ATLAS DASHBOARD: INTEGRITY VERIFICATION SUITE")
    print("=" * 70)

    # 1. Check imports
    print("\n[1/6] Testing Module Imports...")
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
        print("    [OK] All modules imported successfully.")
    except Exception as e:
        print(f"    [FAIL] Import Failed: {e}")
        traceback.print_exc()
        return False

    # 2. Check model artifacts presence
    print("\n[2/6] Checking XGBoost Model Artifacts...")
    models_status = ml.check_all_models_present()
    all_ok = all(models_status.values())
    missing = [k for k, v in models_status.items() if not v]
    if all_ok:
        print("    [OK] All 7 XGBoost model artifacts found in models/xgboost/")
    else:
        print(f"    [FAIL] Missing models: {missing}")
        return False

    # 3. Test Parquet dataset loading
    print("\n[3/6] Testing Parquet Dataset Loaders...")
    try:
        df1 = dl.load_intensity_dataset()
        df2 = dl.load_movement_dataset()
        df3 = dl.load_development_dataset()
        print(f"    [OK] Model 1 Dataset: {len(df1):,} rows, {len(df1.columns)} columns")
        print(f"    [OK] Model 2 Dataset: {len(df2):,} rows, {len(df2.columns)} columns")
        print(f"    [OK] Model 3 Dataset: {len(df3):,} rows, {len(df3.columns)} columns")

        stats = dl.get_global_summary_stats()
        print(f"    [OK] Global Summary Stats: {stats['total_storms']} storms, {stats['total_observations']} observations")

        cat = dl.get_cyclone_catalog('movement')
        print(f"    [OK] Cyclone Catalog: {len(cat):,} unique storms indexed")
    except Exception as e:
        print(f"    [FAIL] Dataset loading failed: {e}")
        traceback.print_exc()
        return False

    # 4. Test Model Loading & Inference on sample row
    print("\n[4/6] Testing Model Inference Pipelines...")
    try:
        # Pick a sample row from test set for each model
        sample_m1 = df1[df1['SPLIT'] == 'TEST'].iloc[10]
        res_m1 = inf.run_intensity_forecast(sample_m1)
        print(f"    [OK] Model 1 Intensity Forecast Output:")
        print(f"        Current: {res_m1['current_kts']} kt | +6h: {res_m1['predictions_kts'][6]} kt | +12h: {res_m1['predictions_kts'][12]} kt | +24h: {res_m1['predictions_kts'][24]} kt")

        sample_m2 = df2[df2['SPLIT'] == 'TEST'].iloc[10]
        res_m2 = inf.run_movement_forecast(sample_m2)
        print(f"    [OK] Model 2 Movement Forecast Output:")
        print(f"        Current: ({res_m2['current_lat']}°, {res_m2['current_lon']}°)")
        print(f"        +6h: ({res_m2['horizons'][6]['lat_pred']}°, {res_m2['horizons'][6]['lon_pred']}°), disp: {res_m2['horizons'][6]['haversine_distance_km']} km")
        print(f"        +24h: ({res_m2['horizons'][24]['lat_pred']}°, {res_m2['horizons'][24]['lon_pred']}°), disp: {res_m2['horizons'][24]['haversine_distance_km']} km")

        sample_m3 = df3[df3['SPLIT'] == 'TEST'].iloc[10]
        res_m3 = inf.run_development_forecast(sample_m3)
        print(f"    [OK] Model 3 Development Forecast Output:")
        print(f"        Predicted Class: {res_m3['predicted_class']}")
        print(f"        Class Probabilities: {res_m3['probabilities']}")
    except Exception as e:
        print(f"    [FAIL] Inference Pipeline Failed: {e}")
        traceback.print_exc()
        return False

    # 5. Test Chart Generation
    print("\n[5/6] Testing Plotly Dark Ocean Visualizations...")
    try:
        fig_glob = charts.create_global_activity_map(df2, sample_storms=50)
        fig_basin = charts.create_basin_distribution_chart(stats['basin_distribution'])
        fig_trend = charts.create_historical_trend_chart(df2)
        fig_int = charts.create_intensity_forecast_chart(res_m1['current_kts'], res_m1['predictions_kts'])
        fig_traj = charts.create_trajectory_map(res_m2['current_lat'], res_m2['current_lon'], res_m2['horizons'])
        fig_dev = charts.create_development_probability_chart(res_m3['probabilities'])
        
        # Test replay map
        storm_sample_sid = cat.iloc[0]['SID']
        storm_track = dl.get_observations_for_cyclone(storm_sample_sid, 'movement')
        fig_replay = charts.create_replay_map(storm_track, current_step=0)

        fig_p1, fig_p2, fig_p3 = charts.create_model_performance_charts()
        print("    [OK] All 10 Plotly chart builders executed cleanly.")
    except Exception as e:
        print(f"    [FAIL] Chart Generation Failed: {e}")
        traceback.print_exc()
        return False

    print("\n[6/6] Overall Verification Result:")
    print("=" * 70)
    print(">>> ALL CHECKS PASSED SUCCESSFULLY. DASHBOARD READY FOR PRODUCTION. <<<")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_checks()
    if not success:
        sys.exit(1)
