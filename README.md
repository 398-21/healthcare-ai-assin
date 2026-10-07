# Predicting ICU trajectories and outcomes from the first 48 hours

MD6117 Machine Learning for Healthcare AI, group project.

Everything we ran is in one notebook, **[`MD6117_group_project.ipynb`](MD6117_group_project.ipynb)**. It covers data cleaning, preprocessing, training, evaluation, tables, figures and discussion for both tasks. The `src/` modules it imports hold the reusable code: SOFA scorers, feature builders, model families and metrics.

## Dataset

The data is the PhysioNet/Computing in Cardiology Challenge 2012 release: **12,000 adult ICU stays** of at least 48 h from cardiac, medical, surgical and trauma ICUs.

* Each stay is one CSV of time-stamped observations (`Time, Parameter, Value`) covering the first 48 h after ICU admission.
* There are 6 admission descriptors (age, gender, height, ICU type, weight) and 36 time-varying vitals and labs, sampled irregularly. `-1` means missing.
* `outcomes.csv` holds one row per stay: SAPS-I, SOFA, length of stay, survival, and in-hospital death (14.2 %).

The full variable list is in [`data/README.md`](data/README.md).

Two properties of the data shape both tasks:

* **No medication data.** There are no vasopressor doses, so the SOFA cardiovascular sub-score computed from these records can only be 0 or 1.
* **Informative missingness.** Many labs are measured in only a minority of stays, and *whether* a test was ordered is itself a signal.

## The two tasks

| | Task 1: SOFA trajectory forecasting | Task 2: in-hospital mortality |
|---|---|---|
| Question | At hour *T* ∈ {24, 30, 36, 42}, what will the patient's SOFA score be 6 h later? | At hour 48, will the patient die before hospital discharge? |
| Unit | stay × decision time (48,000 rows) | stay (11,833; 167 stays with negative length of stay excluded) |
| Target | six organ sub-scores (0–4) and their total, from our own rule engine; plus P(SOFA rises ≥ 2) | `In-hospital_death` |
| Baseline to beat | persistence ("no change") | ladder: random → one variable → SAPS-I / SOFA → 8-feature logistic regression |
| Models | ordinal decomposition, Ridge/logistic, HistGradientBoosting, CatBoost, LightGBM (+ an absolute-target ablation) | logistic regression, XGBoost, CatBoost |
| Split | 70/15/15 by stay, stratified, seed 42 | 70/15/15 by stay, stratified, seed 20260907 |

**Headline results (test set, 95 % bootstrap CI).**

* **Task 1.** TODO
* **Task 2.** TODO

Hyperparameters and decision thresholds were chosen on the validation split only, and the test split was scored once per declared model.

## Repository layout

```
MD6117_group_project.ipynb   the notebook: both tasks, end to end (executed, outputs included)
requirements.txt             pinned package versions used to execute the notebook
data/
  README.md                  dataset description (course-provided)
  outcomes.csv
  icu_records/               12,000 raw <RecordID>.csv files (not in git, see "How to run")
src/
  task1_sofa/                config, data_prep/ (parsing, SOFA scoring, targets, features, splits, QC),
                             models/ (5 families + ablation), evaluate.py (bootstrap metrics)
  task2_mortality/           preprocess.py + config.json (raw -> 1,474 features), verify_outputs.py,
                             validate_phase3_outputs.py, data.py (loader), metrics.py, tests/
preprocessed/                pre-processing outputs (the notebook loads these, or rebuilds them)
  task1_sofa/                cleaned records, targets, 407-feature matrix, split, QC report
  task2_mortality/           unimputed + train-median-imputed feature matrices, labels/splits,
                             feature dictionary, QC tables, SHA-256 manifest
trained_models/
  task1_sofa/                <family>/axis_<organ>.joblib, deteriorate_24h.joblib, model_spec.json
  task2_mortality/           L4_xgboost_best.json (selected), L4_catboost_best.cbm
results/
  task1_sofa/                figures/ and result tables written by the notebook
  task2_mortality/           figures/, result tables, hyperparameter-search logs
docs/                        project plan, TRIPOD+AI checklist, reports and process notes for each task
```

## How to run

1. **Python 3.13** (other 3.11+ versions should work).
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
   pip install -r requirements.txt
   ```
   Keep `scikit-learn==1.7.2` if you load the saved task 1 models: they are pickled with that version.
2. **Data.** Copy the course release's `icu_records/` folder into `data/icu_records/`, so that `data/icu_records/132539.csv` exists. `data/outcomes.csv` is already in the repo.
3. **Run the notebook from the repository root.**
   ```bash
   jupyter notebook MD6117_group_project.ipynb      # then Run All
   # or headless:
   jupyter nbconvert --to notebook --execute --inplace MD6117_group_project.ipynb
   ```

Three switches in the first code cell control what is recomputed:

| Switch | Default | Effect |
|---|---|---|
| `REBUILD_PREPROCESSING` | `False` | `True` rebuilds `preprocessed/` from the raw records (task 1 ~2 min, task 2 ~8 min). `False` loads the saved outputs. |
| `RETRAIN_TASK1` | `True` | Refits all task 1 models on CPU (TODO min) and saves them to `trained_models/task1_sofa/`. `False` loads them. |
| `RETRAIN_TASK2_SEARCH` | `False` | `True` reruns the 50-candidate × 5-fold searches for XGBoost and CatBoost and overwrites the saved models. A GPU is recommended: they took 13 and 89 min on a 24 GB GPU. `False` loads the selected models. Task 2's logistic regressions and its CPU ablations always run. |

With the defaults, a full run takes about TODO min on a laptop CPU.

The task 2 preprocessing also has unit tests and a standalone verifier:

```bash
python -m unittest discover -s src/task2_mortality/tests -t .
python -m src.task2_mortality.verify_outputs
```

## Reproducibility notes

* **Seeds and splits.** All randomness is seeded. Both splits are frozen and written to `preprocessed/`.
* **Task 2 features.** The task 2 feature matrices were frozen on 2026-09-21 (SHA-256 manifest in `preprocessed/task2_mortality/manifest.sha256`). Rebuilding them from the raw records reproduces the same feature values.
* **Task 2 GBDT models.** The selected XGBoost and CatBoost models were trained on a GPU. Retraining on a CPU gives very slightly different trees, so the saved models are loaded by default to reproduce the reported test numbers exactly.
* **Outcome columns.** `outcomes.csv` is never a model input. Task 1 uses its SOFA only as an external check on our rule engine; task 2 uses SAPS-I and SOFA only as the bedside-score baselines.
