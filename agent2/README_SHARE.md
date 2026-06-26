# CAST Lactation-Exit Prediction — reproducible bundle

Predicts whether a dairy cow exits the herd within N days of calving (60/90/120-day
windows) from early-lactation (DIM 1–21) sensor features, using AutoGluon.

## Contents
- `preliminary_analysis.py` — end-to-end pipeline: load CSV → build cow-lactation
  table → aggregate DIM 1–21 features → train AutoGluon per window → export the
  120-day model to `model_export/`.
- `export_best_120d.py` — (optional) re-export the "best" 120-day model
  (`WeightedEnsemble_L3_FULL`) from the trained predictor and re-tune the threshold.
  Run *after* `preliminary_analysis.py`.
- `inference_example.py` — load the exported predictor from `model_export/` and run
  inference on a feature DataFrame.
- `query_results_20260430_200359.csv` — the training input (latest pull).
- `model_export/` — a pre-built export so `inference_example.py` runs immediately,
  without retraining first.
- `requirements.txt` — dependencies (unpinned; developed against autogluon 1.5.0).

## Steps to reproduce
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Train + export:  `python preliminary_analysis.py`
   - Auto-picks the most recent `query_results_*.csv` in this folder.
   - Writes the trained predictor to `autogluon_models/` and the deploy artifact to
     `model_export/`.
4. (optional) `python export_best_120d.py` to ship the WeightedEnsemble instead.
5. Inference smoke test:  `python inference_example.py`

## Notes
- The `WORKSPACE` path in the scripts has been changed to be script-relative
  (`Path(__file__).resolve().parent`), so the bundle runs from any directory.
- Data generation (`dbconnect.py`) is **not** included — it needs DB credentials and
  network access. The provided CSV is the materialized query output, which is all you
  need to reproduce training.
