# Baseline Machine Learning Models

This directory stores the serialized baseline models (Linear Regression, Logistic Regression, and Random Forest) trained in Task 6.

### Large File Notice
The Random Forest models for track displacement and intensity prediction range from 26 MB to 147 MB each (totaling ~600 MB). Because individual models exceed GitHub's 100 MB hard limit, the `.joblib` binary artifacts are excluded from git tracking via `.gitignore`.

### Reproducing Baseline Models
You can reproduce all baseline model artifacts locally at any time by executing:
```bash
python scripts/train_baselines.py --model all
```
Evaluation metrics and benchmark comparisons are documented in `reports/baseline_model_results.md`.
