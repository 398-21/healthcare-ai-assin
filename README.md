# Predicting ICU trajectories and outcomes from the first 48 hours

MD6117 Machine Learning for Healthcare AI, group project.

Everything we ran is in one notebook, **[`MD6117_group_project.ipynb`](MD6117_group_project.ipynb)**. It covers data cleaning, pre-processing, feature engineering and selection, training, evaluation, tables, figures and discussion for both tasks. The `src/` modules it imports hold the reusable code: parsers, SOFA scorers, feature builders, pipeline transformers, model families and metrics.

## Dataset

The data is the PhysioNet/Computing in Cardiology Challenge 2012 release: **12,000 adult ICU stays** of at least 48 h from coronary-care, cardiac-surgery, medical and surgical ICUs.

* Each stay is one CSV of time-stamped observations (`Time, Parameter, Value`) covering the first 48 h after ICU admission.
* There are 6 admission descriptors (age, gender, height, ICU type, weight) and 36 time-varying vitals and labs, sampled irregularly. `-1` means missing.
* `outcomes.csv` holds one row per stay: SAPS-I, SOFA, length of stay, survival, and in-hospital death (14.2 %).

The full variable list is in [`data/README.md`](data/README.md).

Two properties of the data shape both tasks:

* **No medication data.** There are no vasopressor doses, so the SOFA cardiovascular sub-score computed from these records can only be 0 or 1.
* **Informative missingness.** Many labs are measured in only a minority of stays, and *whether* a test was ordered is itself a signal.

## The two tasks

|                  | **Task 1: SOFA Trajectory Forecasting**                                                                                                                                                                                                                                               | **Task 2: In-Hospital Mortality Prediction**                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Task focus       | *6-hour-ahead forecast of the rolling 24-hour rule-based proxy-SOFA score*                                                                                                                                                                                                                | *In-hospital mortality prediction from the first 48 hours after ICU admission*                               |
| Objective        | Test whether models improve on persistence when forecasting the rolling 24-h proxy-SOFA score ending at T+6 h; examine differences across organs and formulations. The secondary label marks a ≥2-point rise between overlapping rolling scores, not an acute event in the next 6 h. | Use the first 48 h of measurements to predict recorded in-hospital death; compare models with SAPS-I and SOFA. |
| Question         | At T = 24, 30, 36, or 42 h, what will the proxy-SOFA score over [T-18, T+6) be, using only measurements before T?                                                                                                                                                                           | Can measurements from [0, 48 h) predict death before hospital discharge, including after ICU discharge?        |
| Unit             | stay x decision time (48,000 rows)                                                                                                                                                                                                                                                          | stay (12,000; all released stays retained)                                                                     |
| Target           | six rule-engine proxy-SOFA sub-scores and `sofa_24h_tplus6`; plus `sofa_rise_ge2_tplus6` for a ≥2-point rise from `SOFA_now`                                                                                                                                                   | `In-hospital_death`                                                                                          |
| Baseline to beat | persistence ("no change")                                                                                                                                                                                                                                                                   | ladder: random → one variable → SAPS-I / SOFA → 8-feature logistic regression                               |
| Models           | ordinal decomposition, Ridge/logistic, HistGradientBoosting, CatBoost, LightGBM (+ an absolute-target ablation)                                                                                                                                                                             | logistic regression, XGBoost, CatBoost                                                                         |
| Split            | 70/15/15 by stay, stratified, seed 42                                                                                                                                                                                                                                                       | 70/15/15 by stay, stratified, seed 20260907                                                                    |

Task 1 naming: for brevity, `sofa_*` targets and outputs below mean the project rule-engine proxy-SOFA score; they are not the full clinical SOFA. `outcomes.SOFA` is kept as a separate reference.

**Headline results (test set, 95 % bootstrap CI).**

* **Task 1.** The ordinal model forecasts the rolling 24-h proxy-SOFA score ending 6 h ahead with mean MAE **0.568**, against **0.726** for persistence. It beats persistence at all 4 horizons, with separated CIs at 24 h and 30 h, and the gain concentrates in the neurological score. The rolling 24-h proxy-SOFA ≥2-point rise classifier reaches AUROC 0.828 [0.804, 0.850] and AUPRC 0.315 [0.264, 0.366], 7× the no-skill level.
* **Task 2.** CatBoost on the 1,163 features retained by the train-fitted filter reaches **AUROC 0.864 [0.842, 0.884]** and **AUPRC 0.570 [0.519, 0.625]** on the 1,800-stay test split. The bedside scores reach AUROC 0.652 (SAPS-I) and 0.629 (SOFA); the 8-feature L3-revised logistic model reaches 0.815.

Every hyperparameter, threshold, feature-selection and class-imbalance choice was made on train, by CV on train, or on validation. The test split was scored once per declared model. The full tuning record is in [`results/hyperparameter_tuning_log.md`](results/hyperparameter_tuning_log.md).

## Data pre-processing, feature engineering and selection (summary)

* **Split first.** Each task splits by stay before any statistic is computed. Every imputation value, scaler, encoder, filter and resampler is fitted on training rows only, inside a scikit-learn / imbalanced-learn `Pipeline`, and the fitted pipeline is what is saved.
* **Cleaning and outliers.** The `-1` sentinel and empty rows are removed. Physiologically implausible values become missing (never clipped), using one shared rule set (`src/shared/physiology.py`); pH and height unit errors are corrected first. Task 1's SOFA *target* keeps the published methodology cleaning unchanged.
* **Feature engineering.**
  * *Task 1:* expanding-window summaries per variable (level, extremes, variability, 12-h trend, measurement count and recency), clinical signals (MAP < 70 / < 65 burden, urine mL/kg/h, P/F ratio, sedation flag) and rule-based SOFA sub-scores.
  * *Task 2:* 37 variables × 3 windows × 12 statistics, plus merged blood pressure and BMI, plus clinically derived ratios (P/F, shock index, BUN/creatinine, urine mL/kg/h) and our own organ sub-scores.
* **Missing data.** Tree models route NaN natively. Linear models median-impute and get missing-indicator columns. Outcome missingness is excluded, never zero-filled.
* **Encoding and scaling.** ICU type is one-hot encoded for the linear models and a native categorical in CatBoost. Linear models are standardised; tree models need no scaling.
* **Feature selection.**
  * A train-fitted filter in every pipeline drops constant, ≥ 99 %-missing and near-duplicate (|r| ≥ 0.98) columns.
  * Task 2 adds clinical-prior selection (L3), a summary-statistic ablation (L3-revised), and embedded SHAP top-*k* selection chosen by nested 5-fold CV.
* **Class imbalance.** No weighting, class weights, SMOTE and random under-sampling are compared without leakage: resampling happens inside training folds only. The simplest strategy within one standard error of the best is kept.
* **Reproducibility.** One seed (`SEED = 42`, `src/shared/reproducibility.py`) drives every estimator, split, sampler, CV fold and bootstrap; task 2's frozen split uses 20260907. LightGBM runs in deterministic mode.

## Repository layout

```
MD6117_group_project.ipynb   the notebook: both tasks, end to end (executed, outputs included)
requirements.txt             pinned package versions used to execute the notebook
data/
  README.md                  dataset description (course-provided)
  outcomes.csv
  icu_records/               12,000 raw <RecordID>.csv files (not in git, see "How to run")
src/
  shared/                    physiology.py (plausibility rules), preprocessing.py (FeatureFilter,
                             ColumnSelector, linear pre-processor), reproducibility.py (SEED)
  task1_sofa/                config, data_prep/ (parsing, SOFA scoring, targets, features, splits, QC),
                             models/ (5 families + ablation, all as pipelines), evaluate.py (bootstrap metrics)
  task2_mortality/           preprocess.py + config.json (raw -> 1,474 features), derived_features.py,
                             modeling.py (pipelines, imbalance CV, nested top-k selection), verify_outputs.py,
                             validate_phase3_outputs.py, data.py, metrics.py, tests/
preprocessed/                pre-processing outputs (the notebook loads these, or rebuilds them)
  task1_sofa/                cleaned records, targets, 407-feature matrix, split, day-level SOFA, extended SOFA audit table, QC report
  task2_mortality/           feature matrices, derived features, labels/splits, feature dictionary,
                             QC tables, SHA-256 manifest
trained_models/              fitted pipelines (pre-processing + model saved together)
  task1_sofa/                <family>/axis_<organ>.joblib, sofa_rise_ge2_tplus6.joblib, model_spec.json
  task2_mortality/           L4_xgboost_pipeline.joblib (+ booster .json), L4_catboost_pipeline.joblib,
                             L4_selected_features.json
results/
  hyperparameter_tuning_log.md   every hyperparameter configuration tried in both tasks, with the
                                 validation / CV score it was selected on (written by the notebook)
  task1_sofa/                figures/ and result tables written by the notebook
  task2_mortality/           figures/, result tables, experiment logs (imbalance CV, all 50 search
                             candidates with per-fold scores, nested CV)
docs/                        project plan, TRIPOD+AI checklist, reports, process notes and SOFA scoring audit for each task
```

## How to run

1. **Python 3.11 or newer** (this run was checked with Python 3.12; Python 3.13 is also supported).
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
   pip install -r requirements.txt
   ```

   Keep `scikit-learn==1.7.2` if you load the saved pipelines: they are pickled with that version. In VS Code, select this environment as the notebook kernel.
2. **Data.** Copy the course release's `icu_records/` folder into `data/icu_records/`, so that `data/icu_records/132539.csv` exists. `data/outcomes.csv` is already in the repo.
3. **Run the notebook from the repository root.**
   ```bash
   jupyter notebook MD6117_group_project.ipynb      # then Run All
   # or headless:
   jupyter nbconvert --to notebook --execute --inplace MD6117_group_project.ipynb
   ```

Three switches in the first code cell control what is recomputed:

| Switch                    | As submitted | Effect                                                                                                                                                                                                                    |
| ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `REBUILD_PREPROCESSING` | `False`    | `True` rebuilds `preprocessed/` from the raw records (task 1 ~4 min, task 2 ~10 min).                                                                                                                                 |
| `RETRAIN_TASK1`         | `False`    | `True` refits every task 1 pipeline (~1 h 40 min, CPU). `False` loads the saved pipelines.                                                                                                                            |
| `RETRAIN_TASK2` | `False` | `True` reruns task 2's imbalance CV, the 50 x 5-fold XGBoost search, nested-CV feature selection and final fits (about 40-50 min). `False` loads the logged results and saved pipelines. |

The saved task 1 and task 2 pipelines were fitted by the notebook's own cells. With the submitted defaults, the notebook loads them and the logged Task 2 search results; it still recomputes evaluation tables, plots and SHAP explanations. Set a retrain switch to `True` only when you intend to refit that task.

The task 2 preprocessing also has unit tests and a standalone verifier:

```bash
python -m unittest discover -s src/task2_mortality/tests -t .
python -m src.task2_mortality.verify_outputs
```

## Reproducibility notes

* **Seeds and splits.** Both splits are frozen and written to `preprocessed/`.
* **GPU vs CPU.** The XGBoost search was configured with `device=cuda`, but this run had no visible GPU and XGBoost fell back to CPU. Load the saved pipelines (`RETRAIN_TASK2 = False`) to reproduce the reported test numbers without another search or fit.
* **Task 2 features.** The task 2 features were checked to rebuild byte-for-byte from the raw records. A SHA-256 manifest is in `preprocessed/task2_mortality/manifest.sha256`.
* **Outcome columns.** `outcomes.csv` is never a model input. Task 1 uses its SOFA only as an external check on our rule engine; task 2 uses SAPS-I and SOFA only as the bedside-score baselines.
