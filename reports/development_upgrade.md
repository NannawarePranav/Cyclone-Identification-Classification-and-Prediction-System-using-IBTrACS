# Development Classification and Dashboard Upgrade

Project: **Prediction of Different Tropical Cyclone Patterns**

## Measured outcome

| Metric | Previous model | Selected model |
|---|---:|---:|
| Test accuracy | 63.42% | 64.05% |
| Test macro-F1 | 0.638 | 0.646 |
| Validation accuracy (2015–2019) | 64.79% | 66.08% |
| Validation macro-F1 | 0.653 | 0.664 |

The test accuracy gain is 0.63 percentage points: 79 additional correct classifications on 12,612 observations. No claim of statistical significance is made; observations from the same cyclone are correlated.

## Experiment and selection protocol

TRAIN <=2014; early stopping VAL 2015-2016; selection mean yearly macro-F1 2017-2019, with non-decreasing validation accuracy; TEST evaluated only after selection. A candidate replaces the current leader only with >0.002 additional mean yearly macro-F1; validation accuracy must not decrease.

27 candidates: three controlled feature-set comparisons using a shared configuration, plus 24 seeded random configurations. Early stopping caps the search at 1,600 trees with 60 rounds of patience. The base-only candidate uses native missing-value handling, so it is not identical to the archived zero-imputed pipeline.

Selected trial: **3**, feature group **history_pressure**, **671 trees used for prediction**. The conservative 0.002 replacement margin retains this candidate over some slightly higher-scoring validation candidates; selection does not claim the absolute highest observed validation score.

Test results were computed after the selection record was written and were not used to choose or reverse the replacement. The test set was previously reported, so this is a follow-up evaluation rather than a new blind holdout.

## Features and data integrity

- Retains all original 109,901 development rows, target labels and splits.
- Adds exact 24/36/48-hour wind lags and availability flags, wind acceleration, observed storm age and past distance-to-land change.
- Adds current pressure and exact 6/12/24-hour pressure lags, changes and availability flags.
- Joins on storm ID, agency and exact current/past timestamp; excludes non-main tracks and interpolated/unverified source observations using the existing O/V flag convention.
- Uses native XGBoost missing-value handling. Pressure is available in 49.08% of training records and approximately 99.95% of validation records: this historical coverage shift remains a limitation.
- No sea-surface temperature, wind shear or satellite features have been added. Models 1 and 2 are unchanged.

## Reproducibility and rollback

- Active model: `models/xgboost/m3_xgboost.joblib`.
- Original model preserved: `models/archive/pre_development_v2/m3_xgboost.joblib`.
- Enhanced inputs: `processed/model3_development_enriched.parquet`; original dataset is preserved.
- Complete trial records and frozen selection: `models/experiments/development_v2/`.
- Reproduce using `python scripts/improve_development.py --trials 24` in a fresh copy with the original model and no existing experiment directory. The script refuses to overwrite a completed experiment.
- Refresh scores explicitly with `python scripts/export_benchmarks.py`. The dashboard also verifies SHA-256 hashes and regenerates results when a saved model or dataset changes.
- To roll back, restore the archived model to the active model path; enriched inputs remain backward-compatible because each pipeline selects its own feature schema. Scores and model caches refresh automatically.

## Dashboard delivery

- Names and IDs are searchable on all forecast selectors and historical replay; filters continue to apply.
- Guided demo uses the shared cyclone/observation selection, restricted to the held-out test partition with all forecast horizons available. If no selection exists, it uses the first eligible cyclone and middle eligible timestamp, independent of prediction success.
- Staged observation → prediction → recorded outcome → discussion, with actual-versus-predicted track and wind comparison.
- Numbered process diagram and downloadable demo results.
- Dynamic scoreboard, regional results, class precision/recall/F1, counts/row-percent confusion matrix and largest-error explanation.
- Previous Random Forest results remain archived in `reports/baseline_model_results.md` and `reports/xgboost_experiment_results.md`; the live dashboard compares current artifacts with freshly computed simple baselines.

## Validation trials

| Trial | Feature group | Trees | Validation accuracy | Selection-year mean F1 |
|---:|---|---:|---:|---:|
| 1 | base | 785 | 64.71% | 0.6404 |
| 2 | history | 990 | 65.39% | 0.6509 |
| 3 | history_pressure | 671 | 66.08% | 0.6595 |
| 4 | base | 333 | 64.56% | 0.6405 |
| 5 | history | 768 | 65.52% | 0.6557 |
| 6 | history_pressure | 506 | 66.22% | 0.6584 |
| 7 | base | 1162 | 64.33% | 0.6364 |
| 8 | history | 402 | 65.55% | 0.6508 |
| 9 | history_pressure | 520 | 66.42% | 0.6612 |
| 10 | base | 323 | 64.60% | 0.6426 |
| 11 | history | 788 | 65.50% | 0.6549 |
| 12 | history_pressure | 628 | 66.14% | 0.6582 |
| 13 | base | 1411 | 64.88% | 0.6443 |
| 14 | history | 658 | 65.75% | 0.6524 |
| 15 | history_pressure | 492 | 66.26% | 0.6600 |
| 16 | base | 1591 | 64.91% | 0.6441 |
| 17 | history | 869 | 65.47% | 0.6503 |
| 18 | history_pressure | 383 | 66.33% | 0.6600 |
| 19 | base | 288 | 64.42% | 0.6402 |
| 20 | history | 523 | 65.53% | 0.6540 |
| 21 | history_pressure | 494 | 66.00% | 0.6551 |
| 22 | base | 371 | 64.52% | 0.6392 |
| 23 | history | 967 | 65.62% | 0.6512 |
| 24 | history_pressure | 578 | 66.21% | 0.6599 |
| 25 | base | 256 | 64.42% | 0.6405 |
| 26 | history | 222 | 65.55% | 0.6496 |
| 27 | history_pressure | 567 | 66.17% | 0.6603 |

## Completed verification

- Existing dashboard integrity suite: passed.
- Existing seven dashboard refinement checks: passed after switching static-source checks to saved benchmark data.
- Seven new upgrade checks: passed, including future-record isolation, exact time/agency matching, preserved cohort, serving parity, artifact fingerprint refresh and unseen demo selection.
- All seven serving models loaded and produced valid predictions in the portability smoke check.
- Streamlit app-level checks passed for all seven pages, all prediction actions, name search, empty search results and all four guided-demo stages.
- Browser review verified the pipeline diagram, AMPHAN demo, outcome reveal, updated scoreboard and interactive confusion matrix. The local preview was restarted and checked on 18 November 2026.
- All changes remain local; no GitHub push was performed.
