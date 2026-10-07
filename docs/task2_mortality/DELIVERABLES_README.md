# In-Hospital Mortality Prediction — Deliverables

MD6117 Machine Learning for Healthcare AI, group project, task 1 (院内死亡预测), modelling half.
PhysioNet/CinC 2012 data, 11,833 ICU stays, 1,474 features, one frozen 70/15/15 split.

**Headline**: XGBoost reaches test AUROC **0.885 [0.865, 0.903]**, AUPRC **0.591 [0.535, 0.649]**; the bedside scores reach 0.640 (SAPS-I) and 0.623 (SOFA); an eight-feature logistic regression reaches 0.821. Every number in these documents is read from a file in `results/`.

---

## Read in this order

| | File | What it is | Time |
|---|---|---|---|
| 1 | `FINAL_REPORT.md` | The report. Ladder table, model, calibration, subgroups, SHAP, three patient cases, limitations. | 25 min |
| 2 | `PRESENTATION_MATERIALS.md` | Slide-ready tables, figure list, slide order, expected questions. | 10 min |
| 3 | `data_snapshot/DATA_README.md` | Where the data lives, what the 1,474 columns are, the traps. | 5 min |
| 4 | `process_docs/` | Stage-by-stage working notes with self-critique and lineage. Read if you want to know *why* a decision was made. | 1–2 h |

That is the whole reading list. Everything else here is code, results, or models.

---

## Folder layout

```
DELIVERABLES/
├── README.md                     this file
├── FINAL_REPORT.md               the report
├── PRESENTATION_MATERIALS.md     slides and Q&A
│
├── code/
│   ├── src/data.py               loader: splits, feature list, labels
│   ├── src/metrics.py            AUROC/AUPRC, bootstrap CI, threshold at recall 0.80, min(Se,PPV)
│   ├── baselines/                one script per rung and per analysis (24 files, list below)
│   ├── environment_gpu.yml       conda spec (xgboost 3.2, catboost 1.2, sklearn 1.2, pandas 2.1, numpy 1.26)
│   └── requirements.txt          pip freeze of the environment used
│
├── models/
│   ├── L4_xgboost_best.json      selected model (1.9 MB)
│   └── L4_catboost_best.cbm      runner-up, kept for the cross-model SHAP comparison (1.0 MB)
│
├── results/                      24 JSON files + figures/ (list below)
├── data_snapshot/DATA_README.md  pointer to experiment/data/; no data copied here
├── process_docs/                 stage1, stage2 v2, stage3 v2
└── trash_bin/                    superseded documents; see WHY_THESE_ARE_HERE.md inside. Do not cite.
```

---

## What each script does and what it writes

Ladder rungs (each scores test once):

| Script | Writes | Rung |
|---|---|---|
| `run_L0_random.py` | `L0_random_baseline.json` | L0 |
| `run_L1_static_age_years.py`, `run_L1_gcs__0_48h__min.py`, `run_L1_bun__0_48h__max.py` | `L1_*.json` | L1 |
| `run_L2_clinical_scores.py` | `L2_clinical_scores_analysis.json` | L2, incl. the `-1` sentinel analysis |
| `run_L3_logistic_regression.py` | `L3_logistic_regression_8features.json` | L3 |
| `run_L3_threshold_analysis.py` | `L3_threshold_analysis.json` | L3 + L2 operating points |
| `run_L3_revised_test_evaluation.py` | `L3_revised_test_evaluation.json` | L3-revised (pre-declared, scored once) |
| `run_L4_xgboost_gpu.py` | `L4_xgboost_training.json`, `models/L4_xgboost_best.json` | L4 training, GPU, 13 min |
| `run_L4_catboost_gpu.py` | `L4_catboost_training.json`, `models/L4_catboost_best.cbm` | L4 training, GPU, 89 min |
| `run_L4_model_selection.py` | `L4_model_selection.json` | picks XGBoost on validation AUROC |
| `run_L4_test_evaluation_gpu.py` | `L4_test_evaluation.json` | **L4 test numbers (the headline)** |

Analyses on frozen predictions or on train + validation only (no new test looks):

| Script | Writes | Answers |
|---|---|---|
| `run_L3_supplementary_evaluation.py` | `L3_supplementary_evaluation.json` | L3 train/val/test, fair subset vs L2 |
| `run_L3_gcs_statistic_ablation.py` | `L3_gcs_statistic_ablation.json` | GCS min → last, urine min → total (validation) |
| `run_L4_supplementary_evaluation.py` | `L4_supplementary_evaluation.json` | L4 vs SAPS-I/SOFA on their subsets; SHAP re-categorised |
| `run_L4_overfit_comparison.py` | `L4_overfit_comparison.json` | train/val for XGBoost and CatBoost |
| `run_L4_xgboost_es_ablation.py` | `L4_xgboost_es_ablation.json` | does early stopping change anything (no) |
| `run_L4_feature_count_ablation.py` | `L4_feature_count_ablation.json` | top-k features, validation AUROC |
| `run_L4_shap_analysis_gpu.py` | `L4_feature_importance.json`, `figures/L4_shap_summary.png` | global SHAP on test |
| `run_L4_shap_validation_comparison.py` | `L4_shap_validation_comparison.json` | XGBoost vs CatBoost SHAP agreement |
| `run_L4_calibration_and_subgroup.py` | `L4_calibration_and_subgroup.json`, `figures/L4_calibration_curve.png` | Brier, reliability curve, L0/L1 min(Se,PPV) |
| `run_L4_subgroup_by_icutype.py` | `L4_subgroup_by_icutype.json` | AUROC by all four ICU types |
| `run_L4_challenge_metric_validation.py` | `L4_challenge_metric_validation.json` | min(Se,PPV) optimal point, validation only |
| `run_L4_case_examples.py` | `L4_case_examples.json`, `figures/L4_shap_waterfall_{TP,FP,FN}_*.png` | three patient explanations |

Also in `results/`: `L4_catboost_search_checkpoint.json` (hyperparameter search log, resumable).

---

## Rules this work followed

- Test set scored once per declared model. L4: once. L3: once. L3-revised: once, with its feature set fixed in the script before first run. Every hyperparameter, threshold and feature choice came from train + validation.
- Calibration, subgroups, SHAP and case explanations were computed on the frozen test prediction vector after all choices were made, and were listed before being run.
- Feature matrix frozen 2026-09-21; not modified.
- SAPS-I and SOFA are evaluated on the patients for whom they exist (n = 1,728 / 1,733). The model is scored on those same subsets for the head-to-head.
- One shared `metrics.py`; AUPRC via `average_precision_score`, never trapezoidal.

## Models

XGBoost was selected (validation AUROC 0.866 vs CatBoost 0.860). LightGBM was attempted but had no GPU support on the training machine and produced no result; it is not part of the ladder.

## Not in scope

Task 2 (prolonged length of stay) and its two open questions (LOS threshold, competing-risk handling) belong to the other half of the group. Both halves use the same split and the same `metrics.py`.

## Reproduce

```bash
conda env create -f code/environment_gpu.yml && conda activate ml_healthcare_gpu
cd ../experiment            # scripts expect experiment/ as working directory
python baselines/run_L4_test_evaluation_gpu.py     # re-derives the headline from the saved model, ~1 min
```

Full retraining needs a GPU: `run_L4_xgboost_gpu.py` (13 min), `run_L4_catboost_gpu.py` (89 min).
