"""Load verified saved scores, automatically refreshing after artifact changes."""
import json
from pathlib import Path
import streamlit as st
from dashboard import model_loader  # Ensure the portable src package is importable.
from cyclone_ml.benchmarks import evaluate_all, inputs, fingerprint

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def _load_verified(versions, report_version):
    path = ROOT/'reports/benchmark_results.json'
    current = {str(p.relative_to(ROOT)).replace('\\','/'):fingerprint(p) for p in inputs(ROOT)}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('schema_version') == 1 and data.get('artifacts') == current:
                return data
        except (ValueError, KeyError):
            pass
    return evaluate_all(ROOT)


def load_benchmarks():
    versions = tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in inputs(ROOT))
    report = ROOT/'reports/benchmark_results.json'
    with st.spinner('Checking saved evaluation results…'):
        return _load_verified(versions,report.stat().st_mtime_ns if report.exists() else 0)
