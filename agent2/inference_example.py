"""
Example: load the exported 120-day exit predictor and run inference.

Drop this file (and the `model_export/` directory produced by
preliminary_analysis.py) into a separate Python project. The only runtime
dependency is `autogluon.tabular` (which pulls in lightgbm, pandas, numpy).

Install:
    pip install "autogluon.tabular[lightgbm]==1.5.0"

Layout assumed:
    your_project/
        inference_example.py
        model_export/
            metadata.json
            autogluon_predictor_120d/   # AutoGluon predictor folder
            lightgbm_booster_120d.txt   # (optional) raw LightGBM booster
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

EXPORT_DIR = Path(__file__).parent / 'model_export'
META = json.loads((EXPORT_DIR / 'metadata.json').read_text())
PREDICTOR_PATH = EXPORT_DIR / META['predictor_path']
MODEL_NAME = META['model_name']
THRESHOLD = META['recommended_threshold']
FEATURE_COLUMNS = META['feature_columns']

_predictor: TabularPredictor | None = None


def _load() -> TabularPredictor:
    global _predictor
    if _predictor is None:
        _predictor = TabularPredictor.load(str(PREDICTOR_PATH))
    return _predictor

def predict(features: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Run inference on a DataFrame with the same feature columns used at training.

    `features` must include all columns listed in `metadata.json -> feature_columns`.
    Missing values are allowed (the predictor imputes). Extra columns are ignored.

    Returns a DataFrame with `exit_120d_probability` and `exit_120d_prediction`.
    """
    predictor = TabularPredictor.load('autogluon_models/autogluon_cast_120d')
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature(s); first few: {missing[:5]}"
        )
    X = features[FEATURE_COLUMNS]
    proba = predictor.predict_proba(X, model=MODEL_NAME)
    print("HELLO HELLO" + str(proba))
    pos_col = proba.columns[-1]  # positive (=exit) class
    p = proba[pos_col].values
    thr = THRESHOLD if threshold is None else threshold
    return pd.DataFrame(
        {
            'exit_120d_probability': p,
            'exit_120d_prediction': (p >= thr).astype(int),
        },
        index=features.index,
    )


if __name__ == '__main__':
    print(f"Model:      {META['model_name']}")
    print(f"Trained on: {META.get('source_csv')}")
    print(f"Threshold:  {THRESHOLD:.4f} (F1-optimal on OOF train predictions)")
    print(f"Features:   {len(FEATURE_COLUMNS)}")

    # Smoke test: run on a single all-NaN row to confirm the pipeline loads.
    import numpy as np
    sample = pd.DataFrame([{c: np.nan for c in FEATURE_COLUMNS}])
    out = predict(sample)
    print("\nSample inference (all-NaN input):")
    print(out)
    
