"""
build_features.py

Reusable feature-engineering logic extracted from preliminary_analysis.py
(STEP 2 + STEP 3). This lets you build the exact same feature columns the
model was trained on, either for:
  (a) the full training CSV (many cow-lactations), or
  (b) a single cow's freshly-queried daily records (for live inference)

Usage for inference:
    import pandas as pd
    from build_features import build_cow_lactation_features

    # df_raw: daily rows for ONE cow's current lactation, pulled from your DB.
    # Must contain at minimum: animal_id, lact, dim, date, pen, scc,
    # plus all the raw sensor/feed columns used at training time.
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    features_df = build_cow_lactation_features(
        df_raw,
        feature_dim_min=1,
        feature_dim_max=21,
        min_early_records=7,   # set to 0 if you want a row even with sparse data
    )
    # features_df has one row (indexed by animal_id, lact) ready to pass into
    # inference_example.predict() -- after selecting/reindexing to FEATURE_COLUMNS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Same exclude list as preliminary_analysis.py STEP 3 -- keep these in sync
# if the training script's exclude_cols ever changes.
EXCLUDE_COLS = [
    'date', 'animal_id', 'lact', 'dim', 'pen', 'cbrd', 'pta', 'gdpr',
    'fdat', 'cdat', 'ddat', 'rc', 'rpro', 'dcc', 'ddry', 't2000',
    'psirc', 'tbrd', 'allflex_tag_number', 'recipeNames',
]

# Same variability keyword list as STEP 3 -- keep in sync.
VARIABILITY_KEYWORDS = [
    'temp', 'act', 'rumination', 'daily_rumination', 'daily_activity',
    'daily_weight', 'health_index', 'scc', 'rum_index', 'water_intake',
]


def build_cow_lactation_features(
    df_raw: pd.DataFrame,
    feature_dim_min: int = 1,
    feature_dim_max: int = 21,
    min_early_records: int = 7,
) -> pd.DataFrame:
    """Build cow-lactation-level features from raw daily rows.

    Mirrors preliminary_analysis.py STEP 2 + STEP 3 exactly, so that features
    built here match training-time features column-for-column.

    Parameters
    ----------
    df_raw : daily-grain rows. Must have animal_id, lact, dim, date, pen, scc,
             plus whatever raw sensor/feed columns you want aggregated.
             Works for one cow-lactation or many at once.
    feature_dim_min, feature_dim_max : the DIM window to aggregate over
             (must match training: 1-21).
    min_early_records : drop cow-lactations with fewer than this many
             early-window records (set to 0 to keep a single sparse cow
             for inference rather than dropping her).

    Returns
    -------
    DataFrame indexed by (animal_id, lact), one row per cow-lactation,
    with all _mean_DIM*, _std_DIM*, parity, pen_first, scc_first,
    n_early_records columns -- same shape as cow_lact_features in training.
    """
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['animal_id', 'date']).reset_index(drop=True)

    # ---- STEP 2 logic ----
    df_lact = df[df['lact'].notna() & df['dim'].notna()].copy()
    df_lact['dim'] = df_lact['dim'].astype(int)
    df_lact['lact'] = df_lact['lact'].astype(int)

    cow_lact = df_lact.groupby(['animal_id', 'lact']).agg(
        min_dim=('dim', 'min'),
        max_dim=('dim', 'max'),
        n_records=('date', 'count'),
        first_date=('date', 'min'),
        last_date=('date', 'max'),
    ).reset_index()

    cow_lact['calving_date'] = cow_lact['first_date'] - pd.to_timedelta(cow_lact['min_dim'], unit='D')
    # Note: obs_window / days_observed aren't needed for inference (no labeling
    # step happens here), but kept for parity with training if useful upstream.
    dataset_end = df['date'].max()
    cow_lact['obs_window'] = (dataset_end - cow_lact['calving_date']).dt.days
    cow_lact['days_observed'] = (cow_lact['last_date'] - cow_lact['calving_date']).dt.days

        # ---- STEP 3 logic ----
    df_early = df_lact[(df_lact['dim'] >= feature_dim_min) & (df_lact['dim'] <= feature_dim_max)].copy()

    # Coerce candidate sensor columns to numeric first. An all-NULL column for
    # a given cow/window comes back from SQL as dtype 'object' (no values to
    # infer a numeric type from), which would otherwise get silently excluded
    # below -- this keeps it as a proper float (all-NaN) column instead, so
    # it's still present (just missing-valued) in the output, matching the
    # training-time column set.
    candidate_cols = [c for c in df_early.columns if c not in EXCLUDE_COLS]
    for c in candidate_cols:
        if df_early[c].dtype not in [np.float64, np.int64]:
            coerced = pd.to_numeric(df_early[c], errors='coerce')
            if coerced.notna().any() or df_early[c].isna().all():
                df_early[c] = coerced

    sensor_cols = [
        c for c in candidate_cols
        if df_early[c].dtype in [np.float64, np.int64]
    ]

    agg_features = df_early.groupby(['animal_id', 'lact'])[sensor_cols].mean()
    agg_features.columns = [f"{c}_mean_DIM{feature_dim_min}_{feature_dim_max}" for c in agg_features.columns]

    variability_cols = [c for c in sensor_cols if any(kw in c for kw in VARIABILITY_KEYWORDS)]
    if variability_cols:
        agg_std = df_early.groupby(['animal_id', 'lact'])[variability_cols].std()
        agg_std.columns = [f"{c}_std_DIM{feature_dim_min}_{feature_dim_max}" for c in agg_std.columns]
        agg_features = agg_features.join(agg_std)

    static_features = df_lact.groupby(['animal_id', 'lact']).agg(
        parity=('lact', 'first'),
        pen_first=('pen', 'first'),
        scc_first=('scc', 'first'),
    ).reset_index().set_index(['animal_id', 'lact'])

    early_record_count = df_early.groupby(['animal_id', 'lact']).size().rename('n_early_records')
    agg_features = agg_features.join(early_record_count)
    agg_features = agg_features.join(static_features)

    cow_lact_features = cow_lact.set_index(['animal_id', 'lact']).join(agg_features, how='inner')

    if min_early_records > 0:
        cow_lact_features = cow_lact_features[cow_lact_features['n_early_records'] >= min_early_records]

    return cow_lact_features