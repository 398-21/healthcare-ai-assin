# SOFA trajectory forecasting — `modeling/`

Forecast the rolling SOFA score at a 6-hour cadence over ICU hours 24–48.
**Full problem statement, decision log (D1–D22), roadmap, TRIPOD+AI checklist and
experiment tracker:** `../SOFA_forecast_project_plan.docx`.
**Plain-language overview to share with non-technical teammates:**
`../SOFA_forecasting_team_briefing.docx`.

## Task (one line)

At decision time `T ∈ {24, 30, 36, 42} h` predict the trailing-24 h SOFA over
`[T−18, T+6)` from `icu_records[0, T)` only. Six organ axes, summed. Cardiovascular is 0/1
(no vasopressor data). `outcomes.csv` is never a feature (D6). **Five model families are
compared head-to-head** (D19–D22 + the original primary, D12–D14/D17) rather than tuning one
family in isolation.

## Layout

| path | what |
| --- | --- |
| `config.py` | paths, horizons, window geometry, scoring + split constants — single source of truth |
| `data_prep/parsing.py` | `parse_record` / `parse_all` + doc §7 cleaning |
| `data_prep/scoring.py` | canonical rule-based SOFA scorers + `score_window()` |
| `data_prep/splits.py` | `make_splits()` — record-level stratified 70/15/15, seed 42 |
| `data_prep/targets.py` | `build_targets()` — rolling 6-axis SOFA + delta + deterioration flag |
| `data_prep/features.py` | `build_features()` — expanding-window features at each origin `T` |
| `data_prep/qc.py` | data-quality audit + `qc_report.md` |
| `models/common.py` | shared plumbing: `AXES`, `feature_columns()`, `axis_frame()`, `deterioration_frame()` (D17: measured-rows-only) |
| `models/ordinal.py` | **family 0 (D19)** — ordinal regression via Frank & Hall cumulative binary decomposition; predicts the absolute score directly |
| `models/linear.py` | **family 1 (D20)** — Ridge / class-weighted LogisticRegression baseline, same delta target as histgb |
| `models/histgb.py` | **family 2, PRIMARY (D12–D14/D17)** — six pooled HistGB axis regressors (delta target) + one pooled deterioration classifier; keeps the original un-suffixed names for backward compatibility |
| `models/catboost_model.py` | **family 3 (D21, optional dep.)** — native categoricals + ordered boosting; guarded import, not installed here |
| `models/lightgbm_model.py` | **family 4 (D22, optional dep.)** — leaf-wise GBDT; guarded import, not installed here |
| `evaluate.py` | bootstrap-CI metrics (`regression_report`, `classification_comparison`), calibration, subgroup breakdown, permutation importance (D16), no-skill reference (D18) — scores predictions, never fits anything |
| `sofa_forecasting.ipynb` | **the single canonical notebook**: preprocessing through evaluation, one document, TRIPOD+AI-tagged section headers, all 5 families |
| `outputs/` | generated: `splits.csv`, `targets.csv`, `features.csv.gz`, `records_clean.joblib`, `qc_report.md`, and (once trained) `models/<family>/*.joblib`, `model_spec.json` |

## Status

- **Sections 1–9 of `sofa_forecasting.ipynb` (data prep through model-development methods):
  executed, verified 2026-09-16.** All leakage/QC checks pass; `T=42` target reproduces
  `SOFA_D2` exactly; the `[0,24)h` rule SOFA vs `outcomes.SOFA` reference check reproduces
  r=0.933, bias=−0.862 as before; every family's hyperparameter grid prints correctly,
  including CatBoost/LightGBM's (guarded imports work without the packages installed).
- **Sections 10–14 (model training for all 5 families, results, interpretation,
  artefact-saving): code complete but deliberately NOT executed in this environment.**
  Training is being run on a separate device — `pip install catboost lightgbm` there first
  to get all 5 families; re-run the notebook top-to-bottom to populate everything from
  "10 Model training" onward.
- `phase1_data.ipynb` is retired (git history keeps it recoverable); its content is folded
  into `sofa_forecasting.ipynb` sections 1–7. The original single-file `models.py` is now
  the `models/` package (D15's notebook-consolidation logic extended to the model code once
  a second family was added).

## Reproduce

```bash
cd modeling
jupyter nbconvert --to notebook --execute --inplace sofa_forecasting.ipynb
```

or from Python, mirroring what the notebook does:

```python
from modeling.data_prep import parsing, splits, targets, features
from modeling import models, evaluate

records = parsing.parse_all()
sp = splits.make_splits()
tg = targets.build_targets(records)
ft = features.build_features(records)

print("installed families:", models.AVAILABLE_FAMILIES)

ordinal_models, *_ = models.ordinal.train_axis_models_ordinal(ft, tg, sp)
linear_models, *_ = models.linear.train_axis_models_linear(ft, tg, sp)
axis_models, axis_params, axis_grids = models.train_axis_models(ft, tg, sp)   # primary (histgb)
det_model, det_params, det_grid = models.train_deterioration_model(ft, tg, sp)
preds = models.predict_axis_scores(axis_models, ft)

if models.catboost_model.CATBOOST_AVAILABLE:
    catboost_models, *_ = models.catboost_model.train_axis_models_catboost(ft, tg, sp)
if models.lightgbm_model.LIGHTGBM_AVAILABLE:
    lightgbm_models, *_ = models.lightgbm_model.train_axis_models_lightgbm(ft, tg, sp)
```
