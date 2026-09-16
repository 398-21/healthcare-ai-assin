# SOFA trajectory forecasting — `modeling/`

Forecast the rolling SOFA score at a 6-hour cadence over ICU hours 24–48.
**Full problem statement, decision log (D1–D17), roadmap, TRIPOD+AI checklist and
experiment tracker:** `../SOFA_forecast_project_plan.docx`.

## Task (one line)

At decision time `T ∈ {24, 30, 36, 42} h` predict the trailing-24 h SOFA over
`[T−18, T+6)` from `icu_records[0, T)` only. Six pooled organ-axis models (D12), summed.
Cardiovascular is 0/1 (no vasopressor data). `outcomes.csv` is never a feature (D6).

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
| `models.py` | **model development** (D12–D14, D17): six pooled axis regressors (delta target, measured-rows-only) + one pooled deterioration classifier, small fixed hyperparameter grid selected on the external validation split |
| `evaluate.py` | bootstrap-CI metrics, calibration, subgroup breakdown, permutation importance (D16) — scores predictions, never fits anything |
| `sofa_forecasting.ipynb` | **the single canonical notebook**: preprocessing through evaluation, one document, TRIPOD+AI-tagged section headers |
| `outputs/` | generated: `splits.csv`, `targets.csv`, `features.csv.gz`, `records_clean.joblib`, `qc_report.md`, and (once trained) `models/*.joblib`, `model_spec.json` |

## Status

- **Sections 1–9 of `sofa_forecasting.ipynb` (data prep through EDA/baselines): executed,
  verified 2026-09-16.** All leakage/QC checks pass; `T=42` target reproduces `SOFA_D2`
  exactly; the `[0,24)h` rule SOFA vs `outcomes.SOFA` reference check reproduces r=0.933,
  bias=−0.862 as before.
- **Sections 10–14 (model training, results, interpretation, artefact-saving): code
  complete but deliberately NOT executed in this environment.** Training is being run on a
  separate device — re-run the notebook top-to-bottom there to populate everything from
  "10 Model training" onward.
- `phase1_data.ipynb` is retired (git history keeps it recoverable); its content is folded
  into `sofa_forecasting.ipynb` sections 1–7.

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

axis_models, axis_params, axis_grids = models.train_axis_models(ft, tg, sp)
det_model, det_params, det_grid = models.train_deterioration_model(ft, tg, sp)
preds = models.predict_axis_scores(axis_models, ft)
```
