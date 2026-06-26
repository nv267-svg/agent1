"""
CAST Dairy Herd Preliminary Analysis Pipeline
Predicting early lactation exit using AutoGluon

Approach (informed by Shiang-Perez et al.):
- Reframe from daily-grain to cow-lactation level (one row per cow per lactation)
- Aggregate early post-calving sensor data (DIM 1-21) as predictive features
- Target: cow exits herd within N days of calving (N = 60, 90, 120)
- Use PR AUC as primary metric due to class imbalance
- AutoGluon for automated model selection and ensembling
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, precision_recall_curve
)

# Set up paths
WORKSPACE = Path(__file__).resolve().parent

# Auto-pick the most recent CSV produced by dbconnect.py
_CANDIDATE_CSVS = sorted(WORKSPACE.glob('query_results_*.csv'))
if not _CANDIDATE_CSVS:
    raise FileNotFoundError(f"No query_results_*.csv found in {WORKSPACE}. Run dbconnect.py first.")
CSV_FILE = _CANDIDATE_CSVS[-1]

# Tee stdout to a dated analysis file (in addition to printing to terminal)
RUN_STAMP = datetime.now().strftime('%Y%m%d')
OUTPUT_FILE = WORKSPACE / f'analysis_output_{RUN_STAMP}.txt'

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()

_OUTPUT_FH = open(OUTPUT_FILE, 'w', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _OUTPUT_FH)
print(f"[Output also being written to: {OUTPUT_FILE}]")
print(f"[Using CSV: {CSV_FILE.name}]")

# Early-lactation aggregation window (DIM range for feature engineering)
FEATURE_DIM_MIN = 1
FEATURE_DIM_MAX = 21

# Exit prediction windows (days post-calving)
EXIT_WINDOWS = [60, 90, 120]

print("=" * 80)
print("CAST DAIRY HERD — LACTATION EXIT PREDICTION")
print("=" * 80)
print(f"\nWorkspace: {WORKSPACE}")
print(f"Analysis Start: {datetime.now().isoformat()}")
print(f"Feature window: DIM {FEATURE_DIM_MIN}-{FEATURE_DIM_MAX}")
print(f"Exit windows: {EXIT_WINDOWS} days post-calving")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading CSV data...")
df = pd.read_csv(CSV_FILE, low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['animal_id', 'date']).reset_index(drop=True)
print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Unique cows: {df['animal_id'].nunique():,}")

dataset_end = df['date'].max()

# ============================================================================
# 2. IDENTIFY COW-LACTATIONS WITH VALID DIM DATA
# ============================================================================
print("\n[STEP 2] Building cow-lactation table...")

# Keep only rows with valid lactation/DIM info
df_lact = df[df['lact'].notna() & df['dim'].notna()].copy()
df_lact['dim'] = df_lact['dim'].astype(int)
df_lact['lact'] = df_lact['lact'].astype(int)
print(f"  Rows with valid lact + dim: {len(df_lact):,}")

# Build cow-lactation summary
cow_lact = df_lact.groupby(['animal_id', 'lact']).agg(
    min_dim=('dim', 'min'),
    max_dim=('dim', 'max'),
    n_records=('date', 'count'),
    first_date=('date', 'min'),
    last_date=('date', 'max'),
).reset_index()

# Estimate calving date: first_date - min_dim
cow_lact['calving_date'] = cow_lact['first_date'] - pd.to_timedelta(cow_lact['min_dim'], unit='D')
# Days from calving to dataset end (observation window available)
cow_lact['obs_window'] = (dataset_end - cow_lact['calving_date']).dt.days
# Days from calving to last observed record
cow_lact['days_observed'] = (cow_lact['last_date'] - cow_lact['calving_date']).dt.days

print(f"  Unique cow-lactations: {len(cow_lact):,}")
print(f"  Parity distribution:")
for p in sorted(cow_lact['lact'].unique()):
    n = (cow_lact['lact'] == p).sum()
    print(f"    Parity {p}: {n} ({n/len(cow_lact)*100:.1f}%)")

# ============================================================================
# 3. AGGREGATE EARLY-LACTATION FEATURES (DIM 1-21)
# ============================================================================
print(f"\n[STEP 3] Aggregating sensor features from DIM {FEATURE_DIM_MIN}-{FEATURE_DIM_MAX}...")

# Filter to the early-lactation window
df_early = df_lact[(df_lact['dim'] >= FEATURE_DIM_MIN) & (df_lact['dim'] <= FEATURE_DIM_MAX)].copy()
print(f"  Records in DIM {FEATURE_DIM_MIN}-{FEATURE_DIM_MAX}: {len(df_early):,}")

# Identify numeric sensor/production columns (exclude identifiers and date fields)
exclude_cols = ['date', 'animal_id', 'lact', 'dim', 'pen', 'cbrd', 'pta', 'gdpr',
                'fdat', 'cdat', 'ddat', 'rc', 'rpro', 'dcc', 'ddry', 't2000',
                'psirc', 'tbrd', 'allflex_tag_number', 'recipeNames']
sensor_cols = [c for c in df_early.columns
               if c not in exclude_cols
               and df_early[c].dtype in [np.float64, np.int64]]

print(f"  Sensor/numeric columns for aggregation: {len(sensor_cols)}")

# Aggregate: mean per cow-lactation over the early DIM window
agg_features = df_early.groupby(['animal_id', 'lact'])[sensor_cols].mean()
agg_features.columns = [f"{c}_mean_DIM{FEATURE_DIM_MIN}_{FEATURE_DIM_MAX}" for c in agg_features.columns]

# Also compute std for key sensor columns (captures variability/instability)
variability_cols = [c for c in sensor_cols if any(kw in c for kw in
                    ['temp', 'act', 'rumination', 'daily_rumination', 'daily_activity',
                     'daily_weight', 'health_index', 'scc', 'rum_index', 'water_intake'])]
if variability_cols:
    agg_std = df_early.groupby(['animal_id', 'lact'])[variability_cols].std()
    agg_std.columns = [f"{c}_std_DIM{FEATURE_DIM_MIN}_{FEATURE_DIM_MAX}" for c in agg_std.columns]
    agg_features = agg_features.join(agg_std)

# Add static cow-lactation features (parity, first DIM values)
static_features = df_lact.groupby(['animal_id', 'lact']).agg(
    parity=('lact', 'first'),
    pen_first=('pen', 'first'),
    scc_first=('scc', 'first'),
).reset_index().set_index(['animal_id', 'lact'])

# Count of early records (data completeness proxy)
early_record_count = df_early.groupby(['animal_id', 'lact']).size().rename('n_early_records')
agg_features = agg_features.join(early_record_count)
agg_features = agg_features.join(static_features)

# Merge with cow-lactation table
cow_lact_features = cow_lact.set_index(['animal_id', 'lact']).join(agg_features, how='inner')
print(f"  Cow-lactations with early features: {len(cow_lact_features):,}")
print(f"  Total features: {len(agg_features.columns)}")

# Drop cow-lactations with too few early records (need at least 7 days of data)
min_early_records = 7
cow_lact_features = cow_lact_features[cow_lact_features['n_early_records'] >= min_early_records]
print(f"  After requiring >= {min_early_records} early records: {len(cow_lact_features):,}")

# ============================================================================
# 4. DEFINE TARGETS AND REPORT CLASS BALANCE
# ============================================================================
print("\n[STEP 4] Defining exit targets for each window...")

for window in EXIT_WINDOWS:
    # A cow "survived" if she has records beyond N days post-calving
    # A cow "exited" if her last record is before N days AND we have enough obs window
    # Right-censored if obs_window < N (can't determine outcome)
    survived = cow_lact_features['days_observed'] >= window
    exited = (cow_lact_features['days_observed'] < window) & (cow_lact_features['obs_window'] >= window)
    censored = cow_lact_features['obs_window'] < window

    target = np.where(survived, 0, np.where(exited, 1, np.nan))
    cow_lact_features[f'exit_{window}d'] = target

    n_survived = survived.sum()
    n_exited = exited.sum()
    n_censored = censored.sum()
    total_labeled = n_survived + n_exited
    exit_rate = n_exited / total_labeled * 100 if total_labeled > 0 else 0

    print(f"\n  {window}-day window:")
    print(f"    Survived (stay >= {window}d): {n_survived:,}")
    print(f"    Exited (left < {window}d):    {n_exited:,}")
    print(f"    Right-censored (excluded):   {n_censored:,}")
    print(f"    Exit rate: {exit_rate:.1f}% ({n_exited}/{total_labeled})")

# ============================================================================
# 5. ANOMALY AND DATA QUALITY SUMMARY
# ============================================================================
print("\n[STEP 5] Data quality summary...")

# Feature coverage
feature_cols = [c for c in cow_lact_features.columns
                if c.endswith(f'_DIM{FEATURE_DIM_MIN}_{FEATURE_DIM_MAX}')
                or c in ['parity', 'pen_first', 'scc_first', 'n_early_records']]

coverage = cow_lact_features[feature_cols].notna().mean().sort_values(ascending=False)
print(f"  Feature coverage (% non-null across cow-lactations):")
print(f"    >= 80%: {(coverage >= 0.8).sum()} features")
print(f"    50-80%: {((coverage >= 0.5) & (coverage < 0.8)).sum()} features")
print(f"    < 50%:  {(coverage < 0.5).sum()} features (will be dropped)")

# Keep only features with >= 50% coverage
good_features = coverage[coverage >= 0.5].index.tolist()
print(f"  Features retained: {len(good_features)}")

# ============================================================================
# 6. AUTOGLUON MODELING FOR EACH EXIT WINDOW
# ============================================================================
print("\n[STEP 6] Running AutoGluon for each exit window...")

from autogluon.tabular import TabularPredictor


def tune_threshold_from_pr(y_true, y_prob):
    """Pick decision thresholds from the PR curve on the given (true, prob) pairs.

    Returns a dict with F1-optimal, F2-optimal (recall-favored), and Youden's-J
    optimal thresholds. F1 is used as the primary tuned threshold; F2 is reported
    for stakeholders who care more about catching exits at the cost of precision.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds of length n-1 vs precision/recall length n
    p = precision[:-1]
    r = recall[:-1]
    t = thresholds
    # F-beta = (1+b^2) * P*R / (b^2*P + R)
    def fbeta(beta):
        b2 = beta * beta
        denom = (b2 * p + r)
        with np.errstate(divide='ignore', invalid='ignore'):
            f = np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)
        return f
    f1 = fbeta(1.0)
    f2 = fbeta(2.0)
    # Youden's J via ROC, but here we approximate from PR-derived predictions
    # by sweeping the same thresholds and computing TPR - FPR.
    j_scores = []
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    for thr in t:
        pred = (y_prob >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr = tp / pos if pos else 0.0
        fpr = fp / neg if neg else 0.0
        j_scores.append(tpr - fpr)
    j_scores = np.array(j_scores) if len(j_scores) else np.array([0.0])
    out = {
        'f1_opt': {
            'threshold': float(t[int(np.nanargmax(f1))]) if len(t) else 0.5,
            'f1': float(np.nanmax(f1)) if len(f1) else 0.0,
        },
        'f2_opt': {
            'threshold': float(t[int(np.nanargmax(f2))]) if len(t) else 0.5,
            'f2': float(np.nanmax(f2)) if len(f2) else 0.0,
        },
        'youden_opt': {
            'threshold': float(t[int(np.nanargmax(j_scores))]) if len(t) else 0.5,
            'j': float(np.nanmax(j_scores)) if len(j_scores) else 0.0,
        },
    }
    return out


def report_at_threshold(label, y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    print(f"    [{label} | thr={threshold:.4f}]")
    print(f"      Precision: {precision_score(y_true, pred, zero_division=0):.4f}  "
          f"Recall: {recall_score(y_true, pred, zero_division=0):.4f}  "
          f"F1: {f1_score(y_true, pred, zero_division=0):.4f}  "
          f"Acc: {accuracy_score(y_true, pred):.4f}")
    print(f"      Confusion (TN/FP/FN/TP): "
          f"{cm[0,0]} / {cm[0,1]} / {cm[1,0]} / {cm[1,1]}")


# Persistent location for the model artifacts we want to ship
EXPORT_DIR = WORKSPACE / 'model_export'
EXPORT_DIR.mkdir(exist_ok=True)

# Track the chosen 120-day predictor + threshold so we can export at the end
_export_state = {}

for window in EXIT_WINDOWS:
    target_col = f'exit_{window}d'
    print(f"\n{'='*70}")
    print(f"  MODELING: {window}-DAY EXIT PREDICTION")
    print(f"{'='*70}")

    # Filter to labeled rows (exclude censored)
    df_mod = cow_lact_features[cow_lact_features[target_col].notna()].copy()
    df_mod[target_col] = df_mod[target_col].astype(int)

    if df_mod[target_col].sum() < 10:
        print(f"  Skipping — only {df_mod[target_col].sum()} exit cases (need >= 10)")
        continue

    # Prepare feature matrix
    model_data = df_mod[good_features + [target_col]].copy()

    # Time-aware split: use calving_date for temporal ordering
    # Train on earlier calvings, test on later ones
    calving_dates = df_mod['calving_date']
    cutoff = calving_dates.quantile(0.75)
    train_mask = calving_dates <= cutoff
    test_mask = calving_dates > cutoff

    train_data = model_data[train_mask].reset_index(drop=True)
    test_data = model_data[test_mask].reset_index(drop=True)

    train_exits = train_data[target_col].sum()
    test_exits = test_data[target_col].sum()

    print(f"  Train: {len(train_data)} rows, {train_exits} exits ({train_exits/len(train_data)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} rows, {test_exits} exits ({test_exits/len(test_data)*100:.1f}%)")

    if test_exits < 3:
        print(f"  Skipping — only {test_exits} exit cases in test set")
        continue

    # Use a persistent (non-/tmp) path so artifacts survive reboots and can be exported
    model_path = str(WORKSPACE / 'autogluon_models' / f'autogluon_cast_{window}d')
    shutil.rmtree(model_path, ignore_errors=True)

    # Train AutoGluon
    predictor = TabularPredictor(
        label=target_col,
        problem_type='binary',
        eval_metric='average_precision',  # PR AUC — per Shiang-Perez recommendation
        path=model_path,
        verbosity=0
    )

    print(f"  Training AutoGluon (180s time limit, eval=PR AUC)...")
    predictor.fit(
        train_data=train_data,
        time_limit=180,
        presets='good_quality'
    )

    # Evaluate on test set
    y_test = test_data[target_col]
    y_pred_prob = predictor.predict_proba(test_data)
    if hasattr(y_pred_prob, 'values'):
        y_pred_prob = y_pred_prob.values
    y_prob_pos = y_pred_prob[:, 1]
    y_pred = (y_prob_pos > 0.5).astype(int)

    # ---- Threshold tuning on out-of-fold predictions (no test-set leakage) ----
    # Try the bagged version of the leaderboard winner first (BAG models have OOF);
    # fall back to the FULL or generic OOF if needed.
    tuned_thresholds = None
    try:
        oof_proba_df = predictor.predict_proba_oof()
        oof_pos = oof_proba_df.iloc[:, -1].values
        y_train = train_data[target_col].values
        if len(oof_pos) == len(y_train) and y_train.sum() > 0:
            tuned_thresholds = tune_threshold_from_pr(y_train, oof_pos)
            print(f"\n  Tuned thresholds (from PR curve on OOF train predictions):")
            print(f"    F1-optimal:     thr={tuned_thresholds['f1_opt']['threshold']:.4f}  "
                  f"F1={tuned_thresholds['f1_opt']['f1']:.4f}")
            print(f"    F2-optimal:     thr={tuned_thresholds['f2_opt']['threshold']:.4f}  "
                  f"F2={tuned_thresholds['f2_opt']['f2']:.4f}  (favors recall)")
            print(f"    Youden-optimal: thr={tuned_thresholds['youden_opt']['threshold']:.4f}  "
                  f"J={tuned_thresholds['youden_opt']['j']:.4f}")
    except Exception as e:
        print(f"  [Threshold tuning skipped: {e}]")

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob_pos)
    except ValueError:
        auc = float('nan')
    pr_auc = average_precision_score(y_test, y_prob_pos)

    # Naive baseline: always predict "stay"
    naive_acc = 1 - y_test.mean()

    print(f"\n  Results ({window}-day exit):")
    print(f"    Accuracy:   {acc:.4f}  (naive baseline: {naive_acc:.4f})")
    print(f"    Precision:  {prec:.4f}")
    print(f"    Recall:     {rec:.4f}")
    print(f"    F1-Score:   {f1:.4f}")
    print(f"    ROC-AUC:    {auc:.4f}")
    print(f"    PR AUC:     {pr_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {cm[0,0]:,}")
    print(f"    False Positives: {cm[0,1]:,}")
    print(f"    False Negatives: {cm[1,0]:,}")
    print(f"    True Positives:  {cm[1,1]:,}")

    # Test-set metrics at the tuned thresholds (sanity check)
    if tuned_thresholds is not None:
        print(f"\n  Test-set metrics at tuned thresholds:")
        report_at_threshold('default', y_test, y_prob_pos, 0.5)
        report_at_threshold('F1-tuned', y_test, y_prob_pos,
                            tuned_thresholds['f1_opt']['threshold'])
        report_at_threshold('F2-tuned', y_test, y_prob_pos,
                            tuned_thresholds['f2_opt']['threshold'])
        report_at_threshold('Youden-tuned', y_test, y_prob_pos,
                            tuned_thresholds['youden_opt']['threshold'])

    # Capture state for the 120-day window so we can export the LightGBM winner
    if window == 120:
        _export_state['predictor'] = predictor
        _export_state['model_path'] = model_path
        _export_state['feature_cols'] = list(model_data.columns.drop(target_col))
        _export_state['target_col'] = target_col
        _export_state['tuned_thresholds'] = tuned_thresholds
        _export_state['test_metrics_default'] = {
            'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
            'roc_auc': float(auc) if not np.isnan(auc) else None,
            'pr_auc': float(pr_auc), 'accuracy': float(acc),
        }
        _export_state['train_size'] = int(len(train_data))
        _export_state['train_exits'] = int(train_exits)
        _export_state['test_size'] = int(len(test_data))
        _export_state['test_exits'] = int(test_exits)

    # Leaderboard
    print(f"\n  Model Leaderboard (top 5):")
    lb = predictor.leaderboard(test_data)
    score_col = 'score_test' if 'score_test' in lb.columns else ('score_val' if 'score_val' in lb.columns else None)
    for rank, (_, row) in enumerate(lb.head(5).iterrows(), 1):
        score = row[score_col] if score_col else row.iloc[-1]
        print(f"    {rank}. {row['model']:35s}: {score:.4f}")

    # Feature importance
    print(f"\n  Feature Importance (top 15):")
    try:
        feature_imp = predictor.feature_importance(test_data)
        for rank, (feat, row) in enumerate(feature_imp.head(15).iterrows(), 1):
            imp_val = row['importance'] if 'importance' in feature_imp.columns else row.iloc[0]
            print(f"    {rank:2d}. {feat:50s}: {imp_val:8.4f}")
    except Exception as e:
        print(f"    [Skipped: {e}]")

# ============================================================================
# 6b. EXPORT BEST 120-DAY MODEL (LightGBM_BAG_L1_FULL) FOR INFERENCE
# ============================================================================
print("\n[STEP 6b] Exporting LightGBM_BAG_L1_FULL (120-day window) for inference...")

EXPORT_MODEL_NAME = 'LightGBM_BAG_L1_FULL'

if not _export_state:
    print("  [Skipped: 120-day window did not train]")
else:
    predictor = _export_state['predictor']
    leaderboard = predictor.leaderboard(silent=True)
    available_models = set(leaderboard['model'].tolist())
    if EXPORT_MODEL_NAME not in available_models:
        # Fall back to whichever LightGBM variant won; otherwise top model
        lgbm_models = [m for m in leaderboard['model'].tolist() if 'LightGBM' in m]
        chosen = lgbm_models[0] if lgbm_models else leaderboard['model'].iloc[0]
        print(f"  [Notice] {EXPORT_MODEL_NAME} not present; exporting {chosen} instead.")
        EXPORT_MODEL_NAME = chosen

    deploy_path = EXPORT_DIR / 'autogluon_predictor_120d'
    if deploy_path.exists():
        shutil.rmtree(deploy_path)

    # Clone the predictor, then strip everything except the chosen model
    try:
        predictor.clone(path=str(deploy_path), return_clone=False)
    except TypeError:
        # Older AutoGluon: positional arg
        predictor.clone(str(deploy_path))

    deploy = TabularPredictor.load(str(deploy_path))
    try:
        deploy.delete_models(models_to_keep=EXPORT_MODEL_NAME, dry_run=False)
    except Exception as e:
        print(f"  [Warning] delete_models failed ({e}); export will include all models.")
    try:
        deploy.save_space()
    except Exception as e:
        print(f"  [Warning] save_space failed: {e}")

    # Persist metadata: feature names, threshold, training context
    meta = {
        'exported_at': datetime.now().isoformat(),
        'window_days': 120,
        'target_col': _export_state['target_col'],
        'model_name': EXPORT_MODEL_NAME,
        'predictor_path': 'autogluon_predictor_120d',
        'feature_columns': _export_state['feature_cols'],
        'feature_window': {'dim_min': FEATURE_DIM_MIN, 'dim_max': FEATURE_DIM_MAX},
        'tuned_thresholds': _export_state['tuned_thresholds'],
        'recommended_threshold': (
            _export_state['tuned_thresholds']['f1_opt']['threshold']
            if _export_state['tuned_thresholds'] else 0.5
        ),
        'test_metrics_default_threshold': _export_state['test_metrics_default'],
        'train_size': _export_state['train_size'],
        'train_exits': _export_state['train_exits'],
        'test_size': _export_state['test_size'],
        'test_exits': _export_state['test_exits'],
        'autogluon_version': getattr(__import__('autogluon.tabular', fromlist=['__version__']),
                                     '__version__', 'unknown'),
        'source_csv': CSV_FILE.name,
    }
    (EXPORT_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2))

    # Best-effort: also dump the raw LightGBM Booster as a .txt for users who
    # want to bypass AutoGluon. NOTE: using the raw booster requires replicating
    # AutoGluon's preprocessing, which is non-trivial — prefer the predictor.
    try:
        ag_lgbm = deploy._trainer.load_model(EXPORT_MODEL_NAME)
        booster = getattr(ag_lgbm, 'model', None)
        if booster is not None and hasattr(booster, 'save_model'):
            booster.save_model(str(EXPORT_DIR / 'lightgbm_booster_120d.txt'))
            print(f"  Raw LightGBM booster saved: {EXPORT_DIR / 'lightgbm_booster_120d.txt'}")
    except Exception as e:
        print(f"  [Notice] Could not extract raw LightGBM booster: {e}")

    print(f"  Predictor exported to: {deploy_path}")
    print(f"  Metadata: {EXPORT_DIR / 'metadata.json'}")
    print(f"  Recommended threshold: {meta['recommended_threshold']:.4f}")
    print(f"  See inference_example.py for how to load and predict.")


# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
APPROACH (informed by Shiang-Perez et al.):
  - Reframed from daily-grain to cow-lactation level
  - Aggregated early post-calving sensor data (DIM {FEATURE_DIM_MIN}-{FEATURE_DIM_MAX})
  - {len(good_features)} features retained (>= 50% non-null)
  - Targets: herd exit within {EXIT_WINDOWS} days of calving
  - Evaluation: PR AUC prioritized (class-imbalance aware)
  - Right-censored cow-lactations excluded from each window

DATA SUMMARY:
  - {len(df):,} daily records from {df['animal_id'].nunique():,} cows
  - {len(cow_lact_features):,} cow-lactations with early-DIM features
  - Date range: {df['date'].min().date()} to {df['date'].max().date()}

NEXT STEPS:
  a) Threshold tuning: optimize decision threshold per window using PR curve
  b) Feature engineering: add DIM 1-7 and DIM 8-14 sub-windows (per paper)
  c) Add health event counts if available (mastitis, metritis, etc.)
  d) Stratify by parity group (1, 2, 3+) for parity-specific models
  e) External validation with additional herd data
""")

print(f"Analysis End: {datetime.now().isoformat()}")
print("=" * 80)
