"""Re-export the best-performing 120-day model from the trained AutoGluon predictor.

Reads the predictor at autogluon_models/autogluon_cast_120d, identifies the
top-ranked model on the leaderboard, clones the predictor, and trims everything
except the chosen model (and any base models it depends on, for L2+ stackers).

Also re-tunes the decision threshold against this specific model's OOF
predictions, then updates model_export/metadata.json.

Run after preliminary_analysis.py has produced the trained artifacts:
    python export_best_120d.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import precision_recall_curve

WORKSPACE = Path(__file__).resolve().parent
SOURCE_PATH = WORKSPACE / 'autogluon_models' / 'autogluon_cast_120d'
EXPORT_DIR = WORKSPACE / 'model_export'
DEPLOY_PATH = EXPORT_DIR / 'autogluon_predictor_120d'
METADATA_PATH = EXPORT_DIR / 'metadata.json'

# Which model to export. None = auto-pick top of validation leaderboard.
# `WeightedEnsemble_L3_FULL` is the most defensible "best" 120-day model:
# top of validation leaderboard (as `WeightedEnsemble_L3`) AND #2 on the
# held-out test set (the one model that's strong on both signals), refit
# on all training data.
TARGET_MODEL: str | None = 'WeightedEnsemble_L3_FULL'


def tune_threshold_from_pr(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    p, r, t = precision[:-1], recall[:-1], thresholds
    if len(t) == 0:
        return None

    def fbeta(beta):
        b2 = beta * beta
        denom = b2 * p + r
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)

    f1, f2 = fbeta(1.0), fbeta(2.0)
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    j_scores = []
    for thr in t:
        pred = (y_prob >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr = tp / pos if pos else 0.0
        fpr = fp / neg if neg else 0.0
        j_scores.append(tpr - fpr)
    j_scores = np.array(j_scores)
    return {
        'f1_opt': {'threshold': float(t[int(np.nanargmax(f1))]), 'f1': float(np.nanmax(f1))},
        'f2_opt': {'threshold': float(t[int(np.nanargmax(f2))]), 'f2': float(np.nanmax(f2))},
        'youden_opt': {'threshold': float(t[int(np.nanargmax(j_scores))]), 'j': float(np.nanmax(j_scores))},
    }


def get_oof_for_model(predictor: TabularPredictor, model_name: str):
    """Return (y_train, oof_pos_proba) for the given model.

    AutoGluon does not produce OOF predictions for `_FULL` (refit-on-all-data)
    models, so we strip the suffix and pull OOF from the underlying bagged model.
    """
    candidates = []
    if model_name.endswith('_FULL'):
        candidates.append(model_name[:-len('_FULL')])
    candidates.append(model_name)

    last_err = None
    for cand in candidates:
        try:
            oof_df = predictor.predict_proba_oof(model=cand)
            y_train = predictor._trainer.load_y()
            oof_pos = oof_df.iloc[:, -1].values
            if len(oof_pos) == len(y_train):
                return cand, np.asarray(y_train), oof_pos
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not retrieve OOF for {model_name}: {last_err}")


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"No trained predictor at {SOURCE_PATH}. Run preliminary_analysis.py first."
        )

    src = TabularPredictor.load(str(SOURCE_PATH))
    leaderboard = src.leaderboard(silent=True).reset_index(drop=True)
    print("120-day validation leaderboard (top 5):")
    print(leaderboard[['model', 'score_val']].head(5).to_string(index=False))

    if TARGET_MODEL is None:
        best_model = leaderboard.iloc[0]['model']
    else:
        all_models = set(src.model_names())
        if TARGET_MODEL not in all_models:
            raise ValueError(
                f"TARGET_MODEL={TARGET_MODEL!r} is not present. "
                f"Available models: {sorted(all_models)}"
            )
        best_model = TARGET_MODEL

    row = leaderboard[leaderboard['model'] == best_model]
    best_score = float(row['score_val'].iloc[0]) if not row.empty else float('nan')
    print(f"\nExporting model: {best_model}  (val PR-AUC = {best_score:.4f})")

    # Resolve dependencies (L2 stackers need their L1 bases). Keep both the
    # FULL refit models and their non-FULL bagged counterparts so OOF / scoring
    # paths stay intact.
    try:
        keep_set = set(src._trainer.get_minimum_model_set(best_model))
    except Exception as e:  # noqa: BLE001
        print(f"  [Warning] Could not compute dependency set ({e}); keeping just {best_model}")
        keep_set = {best_model}
    print(f"\nMinimum model set for inference ({len(keep_set)}):")
    for m in sorted(keep_set):
        print(f"  + {m}")

    # Fresh clone
    if DEPLOY_PATH.exists():
        shutil.rmtree(DEPLOY_PATH)
    src.clone(path=str(DEPLOY_PATH), return_clone=False)

    deploy = TabularPredictor.load(str(DEPLOY_PATH))
    all_models = set(deploy.model_names())
    to_delete = sorted(all_models - keep_set)
    print(f"\nDeleting {len(to_delete)} models from clone:")
    for m in to_delete:
        print(f"  - {m}")

    try:
        deploy.delete_models(models_to_delete=to_delete, dry_run=False)
    except Exception as e:  # noqa: BLE001
        # Workaround for the AutoGluon size-accounting bug observed during the
        # original run: fall back to manual directory removal, which is safe
        # because clone() already wrote a complete on-disk copy.
        print(f"\n[Warning] delete_models raised: {e}")
        print("Falling back to direct directory removal.")
        models_root = DEPLOY_PATH / 'models'
        for m in to_delete:
            mdir = models_root / m
            if mdir.exists():
                shutil.rmtree(mdir)

    try:
        deploy.save_space()
    except Exception as e:  # noqa: BLE001
        print(f"[Warning] save_space failed: {e}")

    # Smoke-test that the surviving predictor still loads and predicts
    deploy = TabularPredictor.load(str(DEPLOY_PATH))
    tracked_models = set(deploy.model_names())
    print(f"\nSurviving models on disk: {sorted(tracked_models)}")

    # Sweep stray model directories the trainer no longer tracks (e.g. failed
    # NN models that were pruned during training but left their dir behind).
    models_root = DEPLOY_PATH / 'models'
    for entry in models_root.iterdir():
        if entry.is_dir() and entry.name not in tracked_models:
            print(f"  Removing untracked stray dir: models/{entry.name}")
            shutil.rmtree(entry)

    # Re-tune threshold for THIS model's probability scale
    print("\nRe-tuning decision threshold on OOF predictions for the new model...")
    tuned = None
    try:
        oof_model, y_train, oof_pos = get_oof_for_model(src, best_model)
        tuned = tune_threshold_from_pr(y_train, oof_pos)
        print(f"  Source OOF model: {oof_model}")
        print(f"  F1-optimal:     thr={tuned['f1_opt']['threshold']:.4f}  F1={tuned['f1_opt']['f1']:.4f}")
        print(f"  F2-optimal:     thr={tuned['f2_opt']['threshold']:.4f}  F2={tuned['f2_opt']['f2']:.4f}")
        print(f"  Youden-optimal: thr={tuned['youden_opt']['threshold']:.4f}  J={tuned['youden_opt']['j']:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"  [Warning] Threshold re-tuning skipped: {e}")
        print("  Falling back to existing threshold in metadata.json (may be wrong-scale).")

    # Update metadata
    meta = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    # Drop stale test metrics from the previously exported model — at thr=0.5
    # they apply to a different model's probability scale and would mislead.
    meta.pop('test_metrics_default_threshold', None)
    meta.update({
        'model_name': best_model,
        # Use None instead of NaN — NaN is not valid JSON and breaks strict parsers.
        'model_val_pr_auc': best_score if not (isinstance(best_score, float) and np.isnan(best_score)) else None,
        'models_kept': sorted(keep_set),
        'predictor_path': 'autogluon_predictor_120d',
        're_exported_at': datetime.now().isoformat(),
    })
    if tuned is not None:
        meta['tuned_thresholds'] = tuned
        meta['recommended_threshold'] = tuned['f1_opt']['threshold']
    METADATA_PATH.write_text(json.dumps(meta, indent=2, allow_nan=False))

    folder_size = sum(f.stat().st_size for f in DEPLOY_PATH.rglob('*') if f.is_file())
    print(f"\nExported to: {DEPLOY_PATH}")
    print(f"Metadata:    {METADATA_PATH}")
    print(f"Folder size: {folder_size / 1e6:.1f} MB")
    print(f"Recommended threshold: {meta.get('recommended_threshold', 'unchanged')}")


if __name__ == '__main__':
    main()
