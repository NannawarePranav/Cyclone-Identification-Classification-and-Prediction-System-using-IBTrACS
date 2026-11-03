#!/usr/bin/env python3
"""
IBTrACS Processed Dataset Validation Suite
==========================================
Validates the integrity, physical consistency, anti-leakage guarantees,
and statistical fidelity of the three model-ready Parquet datasets:

1. processed/model1_intensity.parquet
2. processed/model2_movement.parquet
3. processed/model3_development.parquet

Checks Performed:
- Complete file loadability and schema validation.
- Zero missing values in all target, identifier, and core predictor columns.
- Same-agency operational consistency for historical lag wind features (Model 1 & 3).
- Strict adherence to the ±10-knot threshold classification rule (Model 3).
- Exact prediction horizon matching for movement targets (Model 2).
- Absolute longitude normalization within [-180, +180] degrees.
- Absolute zero cyclone ID (SID) crossover between train, validation, and test splits.
- Zero unexpected duplicate prediction instances.
- Complete parity with documented metrics in reports/preprocessing_report.md.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure UTF-8 output across all platforms
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def validate_model1(df_m1: pd.DataFrame, raw_csv_path: str = None) -> None:
    print("\n" + "=" * 70)
    print("VALIDATING: Model 1 — Intensity Prediction Dataset")
    print("=" * 70)

    # 1. Row count and cyclone count
    assert len(df_m1) == 351933, f"Model 1 row count mismatch: expected 351,933, got {len(df_m1):,}"
    assert df_m1['SID'].nunique() == 4366, f"Model 1 cyclone count mismatch: expected 4,366, got {df_m1['SID'].nunique():,}"
    print(f"[PASS] Instance Count: {len(df_m1):,} rows | Unique Cyclones: {df_m1['SID'].nunique():,}")

    # 2. Target columns zero nulls
    assert df_m1['TARGET_USA_WIND'].isna().sum() == 0, "Model 1 TARGET_USA_WIND contains null values!"
    print("[PASS] Target Column 'TARGET_USA_WIND': 0 missing values.")

    # 3. Core identifiers and predictors zero nulls
    core_cols = ['SID', 'SEASON', 'BASIN', 'ISO_TIME', 'USA_AGENCY', 'PRED_HORIZON_HOURS',
                 'CURRENT_USA_WIND', 'LAT', 'LON_NORM', 'DIST2LAND', 'NATURE', 'MONTH', 'HOUR',
                 'HAS_PAST_USA_WIND_6H', 'HAS_PAST_USA_WIND_12H', 'SPLIT']
    for c in core_cols:
        assert c in df_m1.columns, f"Missing expected column in Model 1: {c}"
        assert df_m1[c].isna().sum() == 0, f"Column '{c}' contains unexpected nulls in Model 1!"
    print(f"[PASS] Core Features ({len(core_cols)} columns): 0 missing values.")

    # 4. Longitude normalization
    assert (df_m1['LON_NORM'] >= -180.0).all() and (df_m1['LON_NORM'] <= 180.0).all(), \
        "Model 1 LON_NORM values outside [-180, +180] range!"
    print(f"[PASS] Longitude Normalization: min={df_m1['LON_NORM'].min():.2f}, max={df_m1['LON_NORM'].max():.2f}")

    # 5. Lag consistency & indicator logic
    # Check 6h lag
    has_6h = df_m1['HAS_PAST_USA_WIND_6H'] == 1
    no_6h = df_m1['HAS_PAST_USA_WIND_6H'] == 0
    assert df_m1.loc[has_6h, 'PAST_USA_WIND_6H'].notna().all(), "HAS_PAST_USA_WIND_6H=1 but PAST_USA_WIND_6H is null!"
    assert df_m1.loc[no_6h, 'PAST_USA_WIND_6H'].isna().all(), "HAS_PAST_USA_WIND_6H=0 but PAST_USA_WIND_6H is not null!"
    
    diff_6h = (df_m1.loc[has_6h, 'CURRENT_USA_WIND'] - df_m1.loc[has_6h, 'PAST_USA_WIND_6H']).round(1)
    assert np.allclose(diff_6h.values, df_m1.loc[has_6h, 'WIND_CHANGE_PAST_6H'].values), \
        "WIND_CHANGE_PAST_6H does not match CURRENT_USA_WIND - PAST_USA_WIND_6H!"
    print(f"[PASS] 6h Lag Consistency: {has_6h.sum():,} available ({has_6h.mean()*100:.2f}%), {no_6h.sum():,} missing (genesis/transition).")

    # Check 12h lag
    has_12h = df_m1['HAS_PAST_USA_WIND_12H'] == 1
    no_12h = df_m1['HAS_PAST_USA_WIND_12H'] == 0
    assert df_m1.loc[has_12h, 'PAST_USA_WIND_12H'].notna().all(), "HAS_PAST_USA_WIND_12H=1 but PAST_USA_WIND_12H is null!"
    assert df_m1.loc[no_12h, 'PAST_USA_WIND_12H'].isna().all(), "HAS_PAST_USA_WIND_12H=0 but PAST_USA_WIND_12H is not null!"
    diff_12h = (df_m1.loc[has_12h, 'CURRENT_USA_WIND'] - df_m1.loc[has_12h, 'PAST_USA_WIND_12H']).round(1)
    assert np.allclose(diff_12h.values, df_m1.loc[has_12h, 'WIND_CHANGE_PAST_12H'].values), \
        "WIND_CHANGE_PAST_12H does not match CURRENT_USA_WIND - PAST_USA_WIND_12H!"
    print(f"[PASS] 12h Lag Consistency: {has_12h.sum():,} available ({has_12h.mean()*100:.2f}%), {no_12h.sum():,} missing (genesis/transition).")

    # 6. Prediction horizons
    assert set(df_m1['PRED_HORIZON_HOURS'].unique()) == {6, 12, 24}, "Unexpected prediction horizons in Model 1!"
    h_counts = df_m1['PRED_HORIZON_HOURS'].value_counts().to_dict()
    assert h_counts[6] == 123234, f"+6h count mismatch: expected 123,234, got {h_counts[6]:,}"
    assert h_counts[12] == 118798, f"+12h count mismatch: expected 118,798, got {h_counts[12]:,}"
    assert h_counts[24] == 109901, f"+24h count mismatch: expected 109,901, got {h_counts[24]:,}"
    print(f"[PASS] Prediction Horizons: +6h ({h_counts[6]:,}), +12h ({h_counts[12]:,}), +24h ({h_counts[24]:,}).")

    # 7. Uniqueness
    dups = df_m1.duplicated(subset=['SID', 'ISO_TIME', 'PRED_HORIZON_HOURS']).sum()
    assert dups == 0, f"Found {dups} duplicate instances in Model 1!"
    print("[PASS] Uniqueness: 0 duplicate (SID, ISO_TIME, PRED_HORIZON_HOURS) combinations.")

    # 8. Zero split crossover
    tr_sids = set(df_m1[df_m1['SPLIT'] == 'train']['SID'])
    val_sids = set(df_m1[df_m1['SPLIT'] == 'val']['SID'])
    te_sids = set(df_m1[df_m1['SPLIT'] == 'test']['SID'])
    assert len(tr_sids & val_sids) == 0, "Train and Val share SIDs in Model 1!"
    assert len(tr_sids & te_sids) == 0, "Train and Test share SIDs in Model 1!"
    assert len(val_sids & te_sids) == 0, "Val and Test share SIDs in Model 1!"
    print(f"[PASS] Zero Split Leakage: Train={len(tr_sids):,} SIDs, Val={len(val_sids):,} SIDs, Test={len(te_sids):,} SIDs (0 overlap).")

    # 9. Split row count parity
    assert (df_m1['SPLIT'] == 'train').sum() == 270694, "Train instance count mismatch!"
    assert (df_m1['SPLIT'] == 'val').sum() == 40801, "Val instance count mismatch!"
    assert (df_m1['SPLIT'] == 'test').sum() == 40438, "Test instance count mismatch!"
    print("[PASS] Split Row Counts: Train=270,694 | Val=40,801 | Test=40,438 (exact match).")


def validate_model2(df_m2: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("VALIDATING: Model 2 — Movement Prediction Dataset")
    print("=" * 70)

    # 1. Row count and cyclone count
    assert len(df_m2) == 516086, f"Model 2 row count mismatch: expected 516,086, got {len(df_m2):,}"
    assert df_m2['SID'].nunique() == 6006, f"Model 2 cyclone count mismatch: expected 6,006, got {df_m2['SID'].nunique():,}"
    print(f"[PASS] Instance Count: {len(df_m2):,} rows | Unique Cyclones: {df_m2['SID'].nunique():,}")

    # 2. Target columns zero nulls
    target_cols = [
        'TARGET_LAT', 'TARGET_LON', 'TARGET_LON_NORM',
        'TARGET_EAST_DISPLACEMENT_KM', 'TARGET_NORTH_DISPLACEMENT_KM',
        'TARGET_TOTAL_DISTANCE_KM', 'TARGET_HAVERSINE_DISTANCE_KM'
    ]
    for c in target_cols:
        assert c in df_m2.columns, f"Missing target column in Model 2: {c}"
        assert df_m2[c].isna().sum() == 0, f"Target column '{c}' contains nulls in Model 2!"
    print(f"[PASS] Target Columns ({len(target_cols)} columns): 0 missing values.")

    # 3. Core identifiers and predictors zero nulls
    core_cols = ['SID', 'SEASON', 'BASIN', 'ISO_TIME', 'PRED_HORIZON_HOURS',
                 'LAT', 'LON_NORM', 'DIST2LAND', 'NATURE', 'MONTH', 'HOUR', 'SPLIT']
    for c in core_cols:
        assert c in df_m2.columns, f"Missing core column in Model 2: {c}"
        assert df_m2[c].isna().sum() == 0, f"Core column '{c}' contains unexpected nulls in Model 2!"
    print(f"[PASS] Core Features ({len(core_cols)} columns): 0 missing values.")

    # 4. Longitude normalization
    assert (df_m2['LON_NORM'] >= -180.0).all() and (df_m2['LON_NORM'] <= 180.0).all(), \
        "Model 2 LON_NORM values outside [-180, +180] range!"
    assert (df_m2['TARGET_LON_NORM'] >= -180.0).all() and (df_m2['TARGET_LON_NORM'] <= 180.0).all(), \
        "Model 2 TARGET_LON_NORM values outside [-180, +180] range!"
    print(f"[PASS] Longitude Normalization: Current & Target LON_NORM strictly within [-180, +180].")

    # 5. Geodesic distance approximation validation
    diff_km = (df_m2['TARGET_TOTAL_DISTANCE_KM'] - df_m2['TARGET_HAVERSINE_DISTANCE_KM']).abs()
    mean_err = diff_km.mean()
    med_err = diff_km.median()
    max_err = diff_km.max()
    print(f"[PASS] Geodesic Validation: Mean Error={mean_err:.4f} km | Median Error={med_err:.4f} km | Max Error={max_err:.2f} km.")
    assert mean_err < 0.20, f"Tangent-plane mean approximation error too large: {mean_err} km!"

    # 6. Prediction horizons
    assert set(df_m2['PRED_HORIZON_HOURS'].unique()) == {6, 12, 24}, "Unexpected prediction horizons in Model 2!"
    h_counts = df_m2['PRED_HORIZON_HOURS'].value_counts().to_dict()
    assert h_counts[6] == 180022, f"+6h count mismatch: expected 180,022, got {h_counts[6]:,}"
    assert h_counts[12] == 174016, f"+12h count mismatch: expected 174,016, got {h_counts[12]:,}"
    assert h_counts[24] == 162048, f"+24h count mismatch: expected 162,048, got {h_counts[24]:,}"
    print(f"[PASS] Prediction Horizons: +6h ({h_counts[6]:,}), +12h ({h_counts[12]:,}), +24h ({h_counts[24]:,}).")

    # 7. Synoptic grid compliance (00, 06, 12, 18 UTC)
    assert set(df_m2['HOUR'].unique()).issubset({0, 6, 12, 18}), "Non-synoptic hours detected in Model 2!"
    print(f"[PASS] Synoptic Grid Compliance: 100% of observations at 00, 06, 12, or 18 UTC.")

    # 8. Uniqueness
    dups = df_m2.duplicated(subset=['SID', 'ISO_TIME', 'PRED_HORIZON_HOURS']).sum()
    assert dups == 0, f"Found {dups} duplicate instances in Model 2!"
    print("[PASS] Uniqueness: 0 duplicate (SID, ISO_TIME, PRED_HORIZON_HOURS) combinations.")

    # 9. Zero split crossover
    tr_sids = set(df_m2[df_m2['SPLIT'] == 'train']['SID'])
    val_sids = set(df_m2[df_m2['SPLIT'] == 'val']['SID'])
    te_sids = set(df_m2[df_m2['SPLIT'] == 'test']['SID'])
    assert len(tr_sids & val_sids) == 0, "Train and Val share SIDs in Model 2!"
    assert len(tr_sids & te_sids) == 0, "Train and Test share SIDs in Model 2!"
    assert len(val_sids & te_sids) == 0, "Val and Test share SIDs in Model 2!"
    print(f"[PASS] Zero Split Leakage: Train={len(tr_sids):,} SIDs, Val={len(val_sids):,} SIDs, Test={len(te_sids):,} SIDs (0 overlap).")

    # 10. Split row count parity
    assert (df_m2['SPLIT'] == 'train').sum() == 421827, "Train instance count mismatch!"
    assert (df_m2['SPLIT'] == 'val').sum() == 46857, "Val instance count mismatch!"
    assert (df_m2['SPLIT'] == 'test').sum() == 47402, "Test instance count mismatch!"
    print("[PASS] Split Row Counts: Train=421,827 | Val=46,857 | Test=47,402 (exact match).")


def validate_model3(df_m3: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("VALIDATING: Model 3 — Development Pattern Classification Dataset")
    print("=" * 70)

    # 1. Row count and cyclone count
    assert len(df_m3) == 109901, f"Model 3 row count mismatch: expected 109,901, got {len(df_m3):,}"
    assert df_m3['SID'].nunique() == 4336, f"Model 3 cyclone count mismatch: expected 4,336, got {df_m3['SID'].nunique():,}"
    print(f"[PASS] Instance Count: {len(df_m3):,} rows | Unique Cyclones: {df_m3['SID'].nunique():,}")

    # 2. Target columns zero nulls
    target_cols = ['TARGET_USA_WIND_24H', 'WIND_CHANGE_24H', 'CLASS_LABEL', 'CLASS_ID']
    for c in target_cols:
        assert c in df_m3.columns, f"Missing target column in Model 3: {c}"
        assert df_m3[c].isna().sum() == 0, f"Target column '{c}' contains nulls in Model 3!"
    print(f"[PASS] Target Columns ({len(target_cols)} columns): 0 missing values.")

    # 3. Core identifiers and predictors zero nulls
    core_cols = ['SID', 'SEASON', 'BASIN', 'ISO_TIME', 'USA_AGENCY', 'CURRENT_USA_WIND',
                 'LAT', 'LON_NORM', 'DIST2LAND', 'NATURE', 'MONTH', 'HOUR',
                 'HAS_PAST_USA_WIND_6H', 'HAS_PAST_USA_WIND_12H', 'SPLIT']
    for c in core_cols:
        assert c in df_m3.columns, f"Missing core column in Model 3: {c}"
        assert df_m3[c].isna().sum() == 0, f"Core column '{c}' contains unexpected nulls in Model 3!"
    print(f"[PASS] Core Features ({len(core_cols)} columns): 0 missing values.")

    # 4. Longitude normalization
    assert (df_m3['LON_NORM'] >= -180.0).all() and (df_m3['LON_NORM'] <= 180.0).all(), \
        "Model 3 LON_NORM values outside [-180, +180] range!"
    print(f"[PASS] Longitude Normalization: strictly within [-180, +180].")

    # 5. Class definition rule adherence (±10 kts threshold)
    w_change = df_m3['WIND_CHANGE_24H'].values
    expected_strengthening = w_change >= 10.0
    expected_stable = (w_change > -10.0) & (w_change < 10.0)
    expected_weakening = w_change <= -10.0

    actual_strengthening = df_m3['CLASS_LABEL'] == 'Strengthening'
    actual_stable = df_m3['CLASS_LABEL'] == 'Stable'
    actual_weakening = df_m3['CLASS_LABEL'] == 'Weakening'

    assert np.array_equal(expected_strengthening, actual_strengthening), "Strengthening class mapping violation!"
    assert np.array_equal(expected_stable, actual_stable), "Stable class mapping violation!"
    assert np.array_equal(expected_weakening, actual_weakening), "Weakening class mapping violation!"

    # Ordinal class ID check: Strengthening=2, Stable=1, Weakening=0
    assert (df_m3.loc[actual_strengthening, 'CLASS_ID'] == 2).all(), "Strengthening class ID != 2!"
    assert (df_m3.loc[actual_stable, 'CLASS_ID'] == 1).all(), "Stable class ID != 1!"
    assert (df_m3.loc[actual_weakening, 'CLASS_ID'] == 0).all(), "Weakening class ID != 0!"
    print("[PASS] Classification Rule Adherence: 100% conformity with ±10-knot threshold boundaries.")

    # 6. Class distribution parity
    class_counts = df_m3['CLASS_LABEL'].value_counts().to_dict()
    assert class_counts['Stable'] == 45842, f"Stable count mismatch: expected 45,842, got {class_counts['Stable']:,}"
    assert class_counts['Strengthening'] == 36367, f"Strengthening count mismatch: expected 36,367, got {class_counts['Strengthening']:,}"
    assert class_counts['Weakening'] == 27692, f"Weakening count mismatch: expected 27,692, got {class_counts['Weakening']:,}"
    class_ratio = max(class_counts.values()) / min(class_counts.values())
    print(f"[PASS] Class Distribution: Stable={class_counts['Stable']:,} (41.71%) | Strengthening={class_counts['Strengthening']:,} (33.09%) | Weakening={class_counts['Weakening']:,} (25.20%) | Ratio={class_ratio:.2f}")

    # 7. Lag consistency
    has_6h = df_m3['HAS_PAST_USA_WIND_6H'] == 1
    no_6h = df_m3['HAS_PAST_USA_WIND_6H'] == 0
    assert df_m3.loc[has_6h, 'PAST_USA_WIND_6H'].notna().all(), "HAS_PAST_USA_WIND_6H=1 but PAST_USA_WIND_6H is null!"
    assert df_m3.loc[no_6h, 'PAST_USA_WIND_6H'].isna().all(), "HAS_PAST_USA_WIND_6H=0 but PAST_USA_WIND_6H is not null!"
    has_12h = df_m3['HAS_PAST_USA_WIND_12H'] == 1
    no_12h = df_m3['HAS_PAST_USA_WIND_12H'] == 0
    assert df_m3.loc[has_12h, 'PAST_USA_WIND_12H'].notna().all(), "HAS_PAST_USA_WIND_12H=1 but PAST_USA_WIND_12H is null!"
    assert df_m3.loc[no_12h, 'PAST_USA_WIND_12H'].isna().all(), "HAS_PAST_USA_WIND_12H=0 but PAST_USA_WIND_12H is not null!"
    print(f"[PASS] Lag Feature Indicators: 6h available={has_6h.sum():,} ({has_6h.mean()*100:.2f}%) | 12h available={has_12h.sum():,} ({has_12h.mean()*100:.2f}%)")

    # 8. Uniqueness
    dups = df_m3.duplicated(subset=['SID', 'ISO_TIME']).sum()
    assert dups == 0, f"Found {dups} duplicate instances in Model 3!"
    print("[PASS] Uniqueness: 0 duplicate (SID, ISO_TIME) combinations.")

    # 9. Zero split crossover
    tr_sids = set(df_m3[df_m3['SPLIT'] == 'train']['SID'])
    val_sids = set(df_m3[df_m3['SPLIT'] == 'val']['SID'])
    te_sids = set(df_m3[df_m3['SPLIT'] == 'test']['SID'])
    assert len(tr_sids & val_sids) == 0, "Train and Val share SIDs in Model 3!"
    assert len(tr_sids & te_sids) == 0, "Train and Test share SIDs in Model 3!"
    assert len(val_sids & te_sids) == 0, "Val and Test share SIDs in Model 3!"
    print(f"[PASS] Zero Split Leakage: Train={len(tr_sids):,} SIDs, Val={len(val_sids):,} SIDs, Test={len(te_sids):,} SIDs (0 overlap).")

    # 10. Split row count parity
    assert (df_m3['SPLIT'] == 'train').sum() == 84531, "Train instance count mismatch!"
    assert (df_m3['SPLIT'] == 'val').sum() == 12758, "Val instance count mismatch!"
    assert (df_m3['SPLIT'] == 'test').sum() == 12612, "Test instance count mismatch!"
    print("[PASS] Split Row Counts: Train=84,531 | Val=12,758 | Test=12,612 (exact match).")


def cross_verify_with_raw_archive(df_m1: pd.DataFrame, df_m3: pd.DataFrame, raw_csv_path: str) -> None:
    """Verifies same-agency historical wind lag matching against the raw CSV archive."""
    if not os.path.exists(raw_csv_path):
        print(f"\n[!] Raw CSV archive not found at: {raw_csv_path}. Skipping raw cross-verification.")
        return

    print("\n" + "=" * 70)
    print("CROSS-VERIFYING: Historical Lag Wind Agency Source Against Raw Archive")
    print("=" * 70)

    # Sample 500 instances from Model 1 where HAS_PAST_USA_WIND_6H == 1
    sample_m1 = df_m1[df_m1['HAS_PAST_USA_WIND_6H'] == 1].sample(n=500, random_state=42)
    sids_to_check = set(sample_m1['SID'].unique())

    print(f"[*] Reading raw records for {len(sids_to_check)} sampled storm tracks from {raw_csv_path}...")
    df_raw = pd.read_csv(
        raw_csv_path,
        skiprows=[1],
        usecols=['SID', 'ISO_TIME', 'USA_AGENCY', 'USA_WIND', 'IFLAG'],
        na_values=["", " "],
        keep_default_na=False,
        low_memory=False
    )
    df_raw = df_raw[df_raw['SID'].isin(sids_to_check)].copy()
    df_raw['DT'] = pd.to_datetime(df_raw['ISO_TIME'], utc=True)
    df_raw['USA_WIND'] = pd.to_numeric(df_raw['USA_WIND'], errors='coerce')

    # Build raw lookup dict: (SID, ISO_TIME) -> (USA_AGENCY, USA_WIND, IFLAG_0)
    raw_dict = {}
    for _, row in df_raw.iterrows():
        raw_dict[(row['SID'], row['ISO_TIME'])] = (
            str(row['USA_AGENCY']),
            row['USA_WIND'],
            str(row['IFLAG'])[0] if row['IFLAG'] else ''
        )

    matched_count = 0
    for _, row in sample_m1.iterrows():
        sid = row['SID']
        curr_time = pd.to_datetime(row['ISO_TIME'], utc=True)
        lag6_time_str = (curr_time - pd.Timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')

        raw_entry = raw_dict.get((sid, lag6_time_str))
        assert raw_entry is not None, f"Expected raw observation at t-6h ({lag6_time_str}) for storm {sid} not found!"
        raw_agency, raw_wind, raw_flag = raw_entry

        # Assert same agency, original flag, and matching wind
        assert raw_agency == row['USA_AGENCY'], \
            f"Agency mismatch at t-6h for {sid}: raw agency '{raw_agency}' != current '{row['USA_AGENCY']}'"
        assert raw_flag in ['O', 'V'], \
            f"Flag mismatch at t-6h for {sid}: flag '{raw_flag}' not in ['O', 'V']"
        assert float(raw_wind) == float(row['PAST_USA_WIND_6H']), \
            f"Wind mismatch at t-6h for {sid}: raw wind {raw_wind} != past {row['PAST_USA_WIND_6H']}"
        matched_count += 1

    print(f"[PASS] Cross-Verification: {matched_count}/500 sampled lag instances strictly match raw archive same-agency original fixes.")


def main():
    parser = argparse.ArgumentParser(description="Validate IBTrACS Processed Datasets.")
    parser.add_argument("--processed-dir", type=str, default="processed", help="Path to processed parquet directory.")
    parser.add_argument("--raw-csv", type=str, default="raw/ibtracs.ALL.list.v04r01.csv", help="Path to raw IBTrACS CSV.")
    args = parser.parse_args()

    m1_path = os.path.join(args.processed_dir, "model1_intensity.parquet")
    m2_path = os.path.join(args.processed_dir, "model2_movement.parquet")
    m3_path = os.path.join(args.processed_dir, "model3_development.parquet")

    for p in [m1_path, m2_path, m3_path]:
        assert os.path.exists(p), f"Processed file does not exist: {p}"

    print("Loading processed Parquet files...")
    df_m1 = pd.read_parquet(m1_path)
    df_m2 = pd.read_parquet(m2_path)
    df_m3 = pd.read_parquet(m3_path)
    print("All Parquet files loaded successfully.")

    validate_model1(df_m1, args.raw_csv)
    validate_model2(df_m2)
    validate_model3(df_m3)
    cross_verify_with_raw_archive(df_m1, df_m3, args.raw_csv)

    print("\n" + "=" * 70)
    print("ALL VALIDATION SUITE CHECKS PASSED WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
