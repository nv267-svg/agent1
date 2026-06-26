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
    predictor = _load()
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature(s); first few: {missing[:5]}"
        )
    X = features[FEATURE_COLUMNS]
    proba = predictor.predict_proba(X, model=META['model_name'])
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

def predict_one_cow(df_raw: pd.DataFrame, animal_id, lact: int, threshold: float | None = None) -> pd.DataFrame:
    """Predict 120-day exit probability for ONE cow-lactation."""
    cow_raw = df_raw[(df_raw['animal_id'] == animal_id) & (df_raw['lact'] == lact)]
    if cow_raw.empty:
        raise ValueError(f"No rows found for animal_id={animal_id!r}, lact={lact!r}")

    dim_min = META['feature_window']['dim_min']
    dim_max = META['feature_window']['dim_max']
    cow_early = cow_raw[(cow_raw['dim'] >= dim_min) & (cow_raw['dim'] <= dim_max)]
    if cow_early.empty:
        raise ValueError(f"No DIM {dim_min}-{dim_max} records for this cow-lactation.")

    mean_suffix = f"_mean_DIM{dim_min}_{dim_max}"
    std_suffix = f"_std_DIM{dim_min}_{dim_max}"

    row = {}
    for col in FEATURE_COLUMNS:
        if col.endswith(mean_suffix):
            raw_col = col[: -len(mean_suffix)]
            row[col] = cow_early[raw_col].mean() if raw_col in cow_early.columns else np.nan
        elif col.endswith(std_suffix):
            raw_col = col[: -len(std_suffix)]
            row[col] = cow_early[raw_col].std() if raw_col in cow_early.columns else np.nan
        elif col == 'parity':
            row[col] = cow_raw['lact'].iloc[0]
        elif col == 'pen_first':
            row[col] = cow_raw['pen'].iloc[0] if 'pen' in cow_raw.columns else np.nan
        elif col == 'scc_first':
            row[col] = cow_raw['scc'].iloc[0] if 'scc' in cow_raw.columns else np.nan
        elif col == 'n_early_records':
            row[col] = len(cow_early)
        else:
            row[col] = np.nan  

    cow_x = pd.DataFrame([row])
    cow_x.index = [f"{animal_id}_{lact}"]
    return predict(cow_x, threshold=threshold)

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
    df_raw = pd.read_csv('/Users/nandinivenkatesh/agent1/automl-export/query_results_20260430_200359.csv', low_memory=False)
    result = predict_one_cow(df_raw, animal_id=2708, lact=3)
    print(result)
