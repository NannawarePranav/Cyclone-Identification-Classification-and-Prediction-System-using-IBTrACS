"""Refresh dashboard results from saved models: python scripts/export_benchmarks.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cyclone_ml.benchmarks import evaluate_all

if __name__ == '__main__':
    result = evaluate_all(ROOT)
    print(f"Exported {len(result['records'])} benchmarks to reports/benchmark_results.json")
