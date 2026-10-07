# In-Hospital Mortality Prediction from 48 Hours of ICU Data

**Course**: MD6117 Machine Learning for Healthcare AI
**Task**: In-hospital death prediction (任务 1 of the group project)
**Data**: PhysioNet/CinC Challenge 2012, all 12,000 stays pooled, 11,833 after exclusions
**Date**: 2026-09-25
**Scope**: This report covers the modelling half of task 1. Preprocessing was done by a teammate and is described in `../preprocessing/README.md`. Task 2 (prolonged length of stay) is a separate deliverable.

Every number in this report is read from a file under `results/`. The file is named at the end of each table.

---

## 1. Summary

A gradient-boosted tree model (XGBoost, 1,474 features) predicts in-hospital death from the first 48 hours of an ICU stay with **test AUROC 0.885 [0.865, 0.903]** and **AUPRC 0.591 [0.535, 0.649]** (n = 1,775, prevalence 0.144). The bedside scores on the same patients reach AUROC 0.640 (SAPS-I) and 0.623 (SOFA). An eight-feature logistic regression with well-chosen summary statistics reaches 0.821, so most of the distance from the bedside scores is closed by feature engineering rather than by model complexity; the full GBDT adds a further 0.064, which is still statistically separated.

Three findings carry the discussion:

1. **Feature breadth, not nonlinearity, does most of the work.** Fifty features recover 98.6% of the full model's validation AUROC; at eight features a tree model and a linear model are within noise of each other.
2. **Which summary statistic you take matters as much as which variable.** The last recorded GCS in the window (AUROC 0.757 alone on train+validation) beats the minimum GCS (0.589) by a wide margin, because minimum GCS saturates on sedation and procedures. Swapping that one statistic in the logistic model raised test AUROC from 0.742 to 0.821.
3. **The model is well calibrated and generalises across ICU types.** Brier 0.086; subgroup AUROC ranges 0.837 (Medical) to 0.955 (Cardiac surgery recovery), with no subgroup below 0.83.

At the clinically chosen operating point (validation recall 0.80), the model flags 29% of patients and 38.7% of those flagged die, against a base rate of 14.4%.

---

## 2. Data and protocol

### 2.1 Cohort

| | Rows | Deaths | Prevalence |
|---|---|---|---|
| Train | 8,283 | 1,195 | 0.1443 |
| Validation | 1,775 | 256 | 0.1442 |
| Test | 1,775 | 256 | 0.1442 |
| Total | 11,833 | 1,707 | 0.1443 |

Splits are 70/15/15, stratified by outcome, seed 20260907, fixed before any modelling. 167 stays with negative length of stay were excluded upstream; all were survivors, which is why prevalence is 0.1443 and not the 0.142 quoted in the original plan. Source: `../experiment/data/split_summary.csv`.

### 2.2 Features

1,474 features per stay, built by the preprocessing team and frozen on 2026-09-21: 37 dynamic parameters × 3 windows (0–24 h, 24–48 h, 0–48 h) × 12 statistics (measured, count, first, last, min, max, mean, median, std, slope, delta, time span), plus merged blood-pressure streams, static descriptors, and record-level counters. Missing values are kept as NaN for tree models; the logistic models use train-median imputation. SAPS-I, SOFA, length of stay and survival time are excluded from the matrix.

### 2.3 Evaluation rules

- **Test set scored once per declared model.** All hyperparameter search, model selection and threshold choice used train + validation only.
- **Primary metric AUPRC, secondary AUROC**, both with 1,000-resample stratified bootstrap 95% CIs from one shared `metrics.py`.
- **Operating point**: threshold chosen on validation for recall ≈ 0.80, then applied to test. Missing a death costs more than an unnecessary review.
- **min(Se, PPV)** reported for comparability with the 2012 challenge.
- **Bedside scores** are evaluated on the patients for whom they were computable (SAPS-I n = 1,728, SOFA n = 1,733). 47 and 42 test patients respectively have no score; those patients have roughly twice the average mortality, so treating the sentinel as a low score would invert them. The model is also scored on those same subsets for the head-to-head.
- **Post-hoc analyses** (calibration, subgroups, SHAP, case explanations) run on the frozen test prediction vector. They were listed before being computed and involve no further choices.

---

## 3. Baseline ladder

**Table 1. Test-set performance, all rungs.** Se/PPV/Sp at the validation recall-0.80 threshold. min(Se, PPV) for L0/L1 uses the same threshold rule and is reported for completeness only.

| Rung | Model | Features | AUROC [95% CI] | AUPRC [95% CI] | Se | PPV | Sp | min(Se,PPV) | n |
|---|---|---|---|---|---|---|---|---|---|
| L0 | Random | 0 | 0.474 [0.436, 0.513] | 0.135 [0.124, 0.151] | — | — | — | 0.147 | 1,775 |
| L1 | Age | 1 | 0.627 [0.590, 0.667] | 0.224 [0.196, 0.265] | — | — | — | 0.164 | 1,775 |
| L1 | GCS min | 1 | 0.553 [0.517, 0.586] | 0.155 [0.143, 0.168] | — | — | — | 0.144 | 1,775 |
| L1 | BUN max | 1 | 0.675 [0.635, 0.710] | 0.265 [0.231, 0.310] | — | — | — | 0.185 | 1,775 |
| L2 | SAPS-I | score | 0.640 | 0.213 | 0.819 | 0.171 | 0.348 | 0.171 | 1,728 |
| L2 | SOFA | score | 0.623 | 0.253 | 0.820 | 0.171 | 0.345 | 0.171 | 1,733 |
| L3 | Logistic, 8 features (clinical-prior statistics) | 8 | 0.742 [0.708, 0.774] | 0.395 [0.343, 0.455] | 0.797 | 0.220 | 0.524 | 0.220 | 1,775 |
| **L3-revised** | **Logistic, 8 features (GCS-last, urine-total)** | 8 | **0.821 [0.793, 0.846]** | **0.500 [0.445, 0.556]** | 0.840 | 0.297 | 0.666 | 0.297 | 1,775 |
| **L4** | **XGBoost, 1,474 features** | 1,474 | **0.885 [0.865, 0.903]** | **0.591 [0.535, 0.649]** | 0.785 | 0.387 | 0.791 | 0.387 | 1,775 |

Sources: `L0_random_baseline.json`, `L1_*.json`, `L2_clinical_scores_analysis.json`, `L3_threshold_analysis.json`, `L3_logistic_regression_8features.json`, `L3_revised_test_evaluation.json`, `L4_test_evaluation.json`, `L4_calibration_and_subgroup.json` (L0/L1 min(Se,PPV)).

**Reading the ladder.**
- A single lab value (BUN max, 0.675) already matches or beats both bedside scores. The scores were designed for population stratification, not individual prediction, and each takes only a few dozen distinct values across 1,775 patients.
- L3 and L3-revised use the same eight parameters and the same recipe (L2 logistic, C = 1, balanced weights). They differ only in two summary statistics (§5). That swap is worth 0.079 AUROC.
- L4 over L3-revised: +0.064 AUROC, +0.091 AUPRC, CIs disjoint.

### 3.1 Head-to-head with the bedside scores on their own patients

| Comparison | Baseline AUROC [CI] | L4 AUROC [CI] | Gain | CIs overlap |
|---|---|---|---|---|
| vs SAPS-I (n = 1,728) | 0.640 [0.601, 0.676] | 0.886 [0.864, 0.905] | +0.246 | No |
| vs SOFA (n = 1,733) | 0.623 [0.586, 0.660] | 0.890 [0.870, 0.909] | +0.267 | No |

Source: `L4_supplementary_evaluation.json`.

---

## 4. The L4 model

### 4.1 Training and selection

Two GBDT implementations were trained on the unimputed matrix with native NaN handling and `scale_pos_weight` for class imbalance (no resampling). Hyperparameters were chosen by 5-fold stratified CV on the training set, 50 random candidates each. LightGBM was attempted but could not use the GPU on the available machine and was dropped before producing a result.

| Model | CV AUROC | Validation AUROC | Validation AUPRC | Training time |
|---|---|---|---|---|
| **XGBoost** (selected) | 0.871 | **0.866** | **0.538** | 13 min (GPU) |
| CatBoost | 0.867 | 0.860 | 0.505 | 89 min (GPU) |

XGBoost selected on validation AUROC. Chosen configuration: `max_depth=7, learning_rate=0.05, n_estimators=300, min_child_weight=3, subsample=0.9, colsample_bytree=0.9, gamma=0, max_bin=127`. Sources: `L4_xgboost_training.json`, `L4_catboost_training.json`, `L4_model_selection.json`.

### 4.2 Capacity and generalisation

| Model | Trees used | Train AUROC | Validation AUROC | Gap |
|---|---|---|---|---|
| XGBoost | 300 of 300 | 1.000 | 0.866 | 0.134 |
| CatBoost | 224 (early-stopped) | 0.957 | 0.860 | 0.097 |
| L3 logistic (reference) | — | 0.733 | 0.722 | 0.012 |

XGBoost fits the training set exactly. This is a capacity property of boosted trees on 8,283 rows, not a defect: the model performs consistently on two independent hold-outs (validation 0.866, test 0.885, difference within the ±0.02 CI half-width), and the train–validation gap is similar for CatBoost, which does not interpolate. An ablation confirmed that adding early stopping to XGBoost's final fit stops at 160 rounds and changes validation AUROC by 0.0002. Sources: `L4_overfit_comparison.json`, `L4_xgboost_es_ablation.json`.

The right wording for the report is: *the model has enough capacity to fit the training set exactly and performs consistently on two independent held-out sets.* "Overfitting" would imply degraded generalisation, which was not observed.

### 4.3 How many features are needed

Features ranked by mean |SHAP| on the **training** set, then XGBoost refit on the top k with early stopping on validation. Validation metrics only.

| Features | Validation AUROC | Validation AUPRC | Share of full-model AUROC |
|---|---|---|---|
| 8 | 0.813 | 0.422 | 93.5% |
| 20 | 0.835 | 0.452 | 96.1% |
| 50 | 0.857 | 0.491 | 98.6% |
| 100 | 0.861 | 0.506 | 99.0% |
| 200 | 0.862 | 0.530 | 99.1% |
| 500 | 0.867 | 0.517 | 99.8% |
| 1,474 | 0.869 | 0.532 | 100% |

Two things follow. Fifty features carry nearly all of the signal, so a compact model is the natural deployment candidate. And XGBoost on its own top eight (0.813) is within noise of the eight-feature logistic model (L3-revised, validation 0.808): at that scale the trees add essentially nothing over a linear fit, so the L3→L4 gap is bought with breadth. Source: `L4_feature_count_ablation.json`.

### 4.4 Calibration

Brier score **0.086** on test. The ten-bin reliability curve tracks the diagonal up to about 0.35 predicted probability and then falls below it: patients predicted at 0.65 die at a rate of 0.54, and patients predicted at 0.95 die at 0.83. The model is slightly over-confident at the high end, which is expected with `scale_pos_weight` inflating positive-class scores. Ranking (AUROC) is unaffected; if calibrated probabilities are needed for communication with families, isotonic or Platt scaling fitted on validation would correct it. Source: `L4_calibration_and_subgroup.json`; figure `results/figures/L4_calibration_curve.png`.

### 4.5 Subgroups by ICU type

| ICU type | n | Deaths | Prevalence | AUROC [95% CI] |
|---|---|---|---|---|
| Coronary | 262 | 37 | 0.141 | 0.854 [0.789, 0.909] |
| Cardiac surgery recovery | 370 | 19 | 0.051 | 0.955 [0.913, 0.986] |
| Medical | 646 | 120 | 0.186 | 0.837 [0.799, 0.871] |
| Surgical | 497 | 80 | 0.161 | 0.897 [0.860, 0.929] |
| All | 1,775 | 256 | 0.144 | 0.885 [0.865, 0.903] |

No subgroup drops below 0.83. The Medical ICU, the largest and most heterogeneous, is hardest; the cardiac-surgery unit, with only 19 deaths, is easiest and has the widest CI. `static_icutype_2` is the 11th most important SHAP feature, so the model uses unit type directly. Source: `L4_subgroup_by_icutype.json`.

---

## 5. What the model learned

### 5.1 Global feature importance

Mean |SHAP| on the test set, XGBoost. Top 10 of 1,474:

| Rank | Feature | \|SHAP\| | Reading |
|---|---|---|---|
| 1 | gcs__0_48h__last | 0.694 | Last recorded consciousness level in the window |
| 2 | static_age_years | 0.258 | Age |
| 3 | urine__0_48h__total | 0.141 | Total urine output, 48 h |
| 4 | wbc__0_48h__slope_per_hour | 0.099 | Trend in white-cell count |
| 5 | gcs__24_48h__last | 0.097 | Last GCS, second day |
| 6 | bun__0_48h__min | 0.088 | Best BUN in the window |
| 7 | mechvent__0_48h__time_span_hours | 0.085 | Hours between first and last ventilator record |
| 8 | urine__24_48h__total | 0.083 | Urine output, second day |
| 9 | gcs__24_48h__median | 0.082 | Median GCS, second day |
| 10 | bun__24_48h__last | 0.079 | Last BUN, second day |

GCS-last is 2.7× the next feature. Seven of the top twenty are GCS statistics; four are renal (BUN, urine); two are trends (`slope`); two are measurement-behaviour features (`time_span_hours`). CatBoost, trained independently, puts the same three features at ranks 1–3 and shares 12 of the top 20, so the ranking is not an artefact of one implementation. Sources: `L4_feature_importance.json`, `L4_shap_validation_comparison.json`; figure `results/figures/L4_shap_summary.png`.

Leakage was checked: record-length proxies (`time_span_hours`, parameter counts) have single-variable AUROC 0.50–0.62; GCS charting span is identical for deaths and survivors (median 44.5 h vs 44.0 h); no ID or outcome-derived column is in the matrix.

### 5.2 Last GCS versus minimum GCS

The original L3 used `gcs__0_48h__min` because the SOFA neurological component uses the worst GCS in the window. Single-variable screen on train + validation (test untouched):

| GCS statistic, 0–48 h | AUROC |
|---|---|
| last | 0.757 |
| median | 0.722 |
| mean | 0.707 |
| max | 0.686 |
| slope / delta | 0.666 / 0.665 |
| **min** | **0.589** |

Almost every ICU patient records at least one low GCS in 48 hours from sedation, intubation or a procedure, so `min` saturates near the floor and cannot separate treatment-induced lows from prognostic ones. `last` reads consciousness after those effects have worn off. Refitting the L3 recipe with `min → last` and `urine min → urine total` raised validation AUROC from 0.722 to 0.808; that feature set was then declared as L3-revised and scored on test once (0.821). Source: `L3_gcs_statistic_ablation.json`, `L3_revised_test_evaluation.json`.

This is the project's clearest example of the "measurement behaviour versus physiology" question the plan asked about: the clinical prior identified the right variable and the wrong summary of it.

### 5.3 Three patients

SHAP waterfalls on the test set for one correct high-risk call, one false alarm, and one missed death. Figures in `results/figures/L4_shap_waterfall_*.png`; contributions in `L4_case_examples.json`.

**Correct high-risk call (RecordID 103032, predicted 0.997, died).** Last GCS 7 (+0.91 log-odds), last lactate 11.6 mmol/L (+0.33), second-day urine only 414 mL (+0.32), platelets 25 (+0.23), mean systolic BP 97 (+0.21). Every major organ system is contributing in the same direction; the model has no doubt and neither would a clinician.

**False alarm (RecordID 100571, predicted 0.976, survived).** Last GCS 6 (+1.74), age 80 (+0.23), lactate 3.0 (+0.23). Urine output 5.5 L pulls the other way (−0.17) but cannot offset the GCS. A deeply sedated or post-ictal 80-year-old with otherwise adequate perfusion looks like this; the model is reading depressed consciousness as prognostic when here it probably was not. This is the failure mode of the GCS-last feature and the case for pairing it with a less timing-dependent summary such as the median.

**Missed death (RecordID 102169, predicted 0.004, died).** Last GCS 8 pushes risk up (+0.48), but 7.2 L urine, age 53, a falling white count, lactate 0.6 and a short ventilation span all push it down, and the remaining 1,460 features contribute a further −4.5 log-odds. On the numbers available at 48 h this patient looked like a recovering one. Deaths of this kind, with normal perfusion markers, are the residual that no 48-hour model will catch.

---

## 6. Comparison with the literature

The 2012 challenge released the same 12,000 stays as set A (4,000, outcomes public) and sets B and C (outcomes withheld). Its official metric was min(Se, PPV); the SAPS-I reference scored 0.313 and the winning entry 0.535 on sets B + C. Post-challenge AUROC reports on this data, most of them training on set A only, cluster at 0.84–0.85 (CinC 2014 preprocessing study 0.848; a 2026 XGBoost baseline 0.840 / AUPRC 0.456).

Our L4 sits at 0.885. The two most likely reasons are that we train on 8,283 rows (all three sets pooled and re-split) rather than 4,000, and that we use 1,474 engineered features with window-level trends. Those are legitimate differences, but they also mean our number is not directly comparable to the challenge-era results, and the report should say "consistent with or above published results on this dataset" rather than "beats the literature".

On the challenge metric, our test value at the recall-0.80 threshold is 0.387. A validation-only sweep finds the metric-optimal threshold at 0.345, giving min(Se, PPV) = 0.518 with Se 0.52 and PPV 0.52. That would have been competitive with the 2012 winner, but it halves sensitivity, which is the trade the project's clinical framing rejects. The test set was not re-scored at that threshold. Source: `L4_challenge_metric_validation.json`.

---

## 7. Clinical reading of the operating point

At the recall-0.80 threshold on test (n = 1,775):

| | Flagged | Not flagged |
|---|---|---|
| Died (256) | 201 | 55 |
| Survived (1,519) | 318 | 1,201 |

The model flags 519 patients (29%). Of those, 201 die: a 38.7% event rate against a 14.4% base rate, a 2.7-fold enrichment. It misses 55 of 256 deaths. SAPS-I at its own recall-0.80 threshold flags 1,167 of 1,728 patients (68%) to catch a similar number of deaths, so the model delivers the same sensitivity with less than half the alert burden.

---

## 8. Limitations

1. **Single cohort, one split, no external validation.** All results come from PhysioNet 2012. `gcs__0_48h__last` in particular depends on charting cadence and needs testing at another site.
2. **Probabilities are over-confident above ~0.35** (§4.4). Fine for ranking and alerting; recalibrate before quoting risks to families.
3. **Two L3 coefficients have the wrong sign** (creatinine, platelets), attributable to collinearity with BUN and weak signal. Left as is because fixing them would be a third test-set look.
4. **The full 1,474-feature model is the research benchmark, not the deployment candidate.** A 50–100 feature version keeps 98.6–99% of validation AUROC and is far easier to audit.
5. **The outcome is in-hospital death, not ICU death**, so deaths after ICU discharge count as positives; survival time after 48 h is not modelled.
6. **Fairness by sex or age band was not assessed.** Age is the second most important feature.

---

## 9. Reproducing this work

```
code/
├── src/data.py, src/metrics.py          shared loader and metrics
└── baselines/run_L*.py                  one script per rung and per analysis
models/L4_xgboost_best.json, L4_catboost_best.cbm
results/*.json, results/figures/*.png
```

Training scripts (`run_L4_xgboost_gpu.py`, `run_L4_catboost_gpu.py`) need a GPU and took 13 and 89 minutes. Everything else runs on CPU in under three minutes from the saved models. Environment: `code/environment_gpu.yml`; Python 3.11, xgboost 3.2, catboost 1.2, scikit-learn 1.2, pandas 2.1, numpy 1.26. Data: `../experiment/data/` (frozen snapshot, SHA-256 manifest included).
