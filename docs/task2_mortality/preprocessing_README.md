# Task 2 preprocessing

This guide describes the active implementation in `src/task2_mortality/`. The raw data under `data/` is not modified. See [`DATA_README.md`](DATA_README.md) for the full cohort, feature, and output details.

## Cohort and prediction setup

Task 2 uses all 12,000 released ICU stays. The 167 records with `Length_of_stay = -1` are retained because length of stay is neither the label nor a predictor. Measurements in `[0, 48 h)` predict recorded `In-hospital_death`. The split is stratified by outcome, fixed at 70/15/15 with seed `20260907` (8,400 / 1,800 / 1,800 stays). Task 1 has a separate split.

## Active cleaning and feature steps

- Invalid times, blank or unknown parameters, invalid/non-finite values, and observations after 48:00 are excluded from feature summaries. Exact 48:00 observations are excluded from the primary features and saved separately for sensitivity analysis (234 affected stays). These row-level exclusions do not remove stays from the cohort.
- The `-1` sentinel and measurements outside the ranges in `src/shared/physiology.py` are treated as missing; values are not clipped. Before range checks, pH values above 14 are divided by 100 and heights below 100 cm are multiplied by 100.
- Same-time continuous observations are aggregated by median, `MechVent` by maximum, and `Urine` by sum. Duplicate and static-descriptor conflicts are audited.
- Base features summarize 37 dynamic variables over three windows. The pipeline then adds 42 derived clinical features. Train medians are used for exported imputed matrices; entirely missing training columns are filled with zero and flagged. Model pipelines fit their own preprocessing on training folds.
- `SAPS-I`, `SOFA`, `Length_of_stay`, and `Survival` are not model predictors. SAPS-I and SOFA are retained as bedside-score comparators, with `-1` converted to missing and availability recorded.

## Run from the repository root

```powershell
python -m src.task2_mortality.preprocess
python -m src.task2_mortality.verify_outputs
python -m src.task2_mortality.validate_phase3_outputs
```

The notebook defaults to loading existing artifacts; set its preprocessing rebuild switch only when intentionally regenerating them.
