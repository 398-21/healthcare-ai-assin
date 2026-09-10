# SOFA trajectory forecasting — `modeling/`

Forecast the rolling SOFA score at a 6-hour cadence over ICU hours 24–48.
**Full problem statement, decision log, roadmap and experiment tracker:**
`../SOFA_forecast_project_plan.docx`.

## Task (one line)

At decision time `T ∈ {24, 30, 36, 42} h` predict the trailing-24 h SOFA over
`[T−18, T+6)` from `icu_records[0, T)` only. Six organ axes, summed. Cardiovascular
is 0/1 (no vasopressor data). `outcomes.csv` is never a feature.

## Layout

| path | what |
| --- | --- |
| `config.py` | paths, horizons, window geometry, scoring + split constants — single source of truth |
| `data_prep/parsing.py` | `parse_record` / `parse_all` + doc §7 cleaning |
| `data_prep/scoring.py` | canonical rule-based SOFA scorers + `score_window()` (mirrors `sofa_calculation.ipynb`) |
| `data_prep/splits.py` | `make_splits()` — record-level stratified 70/15/15, seed 42 |
| `data_prep/targets.py` | `build_targets()` — rolling 6-axis SOFA + delta + deterioration flag |
| `data_prep/features.py` | `build_features()` — expanding-window features at each origin `T` |
| `data_prep/qc.py` | data-quality audit + `qc_report.md` |
| `phase1_data.ipynb` | **Phase 1 runner** — orchestrates the above, renders QC + leakage checks |
| `outputs/` | generated: `splits.csv`, `targets.csv`, `features.csv.gz`, `records_clean.joblib`, `qc_report.md` |

## Reproduce Phase 1

```bash
cd modeling
jupyter nbconvert --to notebook --execute --inplace phase1_data.ipynb
```

or from Python:

```python
from modeling.data_prep import parsing, splits, targets, features
records = parsing.parse_all()
splits.make_splits().to_csv("outputs/splits.csv", index=False)
targets.build_targets(records).to_csv("outputs/targets.csv", index=False)
features.build_features(records).to_csv("outputs/features.csv.gz", index=False)
```

## Status

- **Phase 1 (data prep & split): DONE 2026-09-10.** 48,000 `(record × horizon)` rows,


  407 features; all leakage / QC checks pass; the `T=42` target reproduces the
  notebook's `SOFA_D2` exactly.

- **Phase 2 (EDA + baselines): next.** Persistence baseline (`SOFA_hat = SOFA_now`)
  is the bar every model must beat.
