> **Historical snapshot - superseded (2026-10-08).** This dated working report contains results from the earlier 11,833-stay Task 2 cohort. The current analysis retains all 12,000 stays. Use the repository README and current notebook/DATA_README for active results; the historical numbers below are not current.

---

# Stage 1: Baseline Framework and L0-L2 Results

**Date**: 2026-09-23  
**Author**: Experiment modeling team  
**Purpose**: Establish infrastructure, execute baseline ladder (L0-L2), and document the SAPS-I/SOFA -1 sentinel decision

---

## What Was Accomplished

### Infrastructure Modules Created

**`src/data.py`** - Data loading utilities
- `load_data()`: Loads labels, unimputed features, and imputed splits
- `get_split(split_name)`: Returns X, y for train/validation/test
- `get_feature_names()`: Returns list of 1,474 feature names
- Caching to avoid reloading on repeated calls

**`src/metrics.py`** - Evaluation metrics
- `compute_metrics(y_true, y_score, threshold=None)`: AUPRC, AUROC, Se/PPV/Sp
- `bootstrap_ci()`: Stratified bootstrap 95% confidence intervals
- `find_threshold_for_recall(target_recall=0.80)`: Threshold selection for validation
- `min_se_ppv()`: PhysioNet 2012 Challenge metric

Both modules follow the patterns from `src/make_labels_and_baselines.py` and use `average_precision_score` (not `auc(recall, precision)`) per 实验方案 §5.6.

### Baseline Experiments Executed

All baselines evaluated on **test set (n=1,775, prevalence=0.1443)** with **1,000-bootstrap 95% CIs**.

---

## Results Summary

| Baseline | AUROC [95% CI] | AUPRC [95% CI] | Notes |
|----------|----------------|----------------|-------|
| **L0: Random** | 0.474 [0.436, 0.513] | 0.135 [0.124, 0.151] | No-skill floor established |
| **L1: Age** | 0.627 [0.590, 0.667] | 0.224 [0.196, 0.265] | Moderate single-variable predictor |
| **L1: GCS min** | 0.553 [0.517, 0.586] | 0.155 [0.143, 0.168] | Weak alone |
| **L1: BUN max** | **0.675 [0.635, 0.710]** | **0.265 [0.231, 0.310]** | **Best single variable** |
| **L2: SAPS-I** | 0.640 (n=1,728) | 0.213 (n=1,728) | Score-available subset only |
| **L2: SOFA** | 0.623 (n=1,733) | 0.253 (n=1,733) | Score-available subset only |

**Key finding**: BUN max (AUROC=0.675) **outperforms both clinical severity scores** as a single variable, confirming substantial headroom exists for L3 multi-feature logistic regression.

---

## SAPS-I/SOFA -1 Sentinel Decision

### The Problem

- **SAPS-I**: 47 test rows (2.6%) have value = -1 (score not computable)
- **SOFA**: 42 test rows (2.4%) have value = -1 (score not computable)
- These patients are **high-risk**: 27.7% mortality (SAPS-I -1 rows) vs 14.1% overall, 26.2% (SOFA -1) vs 14.1% overall
- Keeping -1 as a numeric value inverts risk ranking (treats high-risk as lowest-risk)

### Impact Quantified

| Score | Kept -1 (inverted) | Excluded -1 rows | AUROC Impact |
|-------|-------------------|------------------|--------------|
| SAPS-I | AUROC 0.616, AUPRC 0.209 | AUROC 0.640, AUPRC 0.213 (n=1,728) | **+24.3 points** |
| SOFA | AUROC 0.604, AUPRC 0.248 | AUROC 0.623, AUPRC 0.253 (n=1,733) | **+19.0 points** |

Median replacement (SAPS-I median=15, SOFA median=7) partially recovers performance but still introduces bias by fabricating scores.

### Decision and Rationale

**Chosen approach**: Report clinical baselines **only on score-available subsets** (SAPS-I n=1,728, SOFA n=1,733). For head-to-head comparison with our ML models, evaluate the ML model on the **same score-available subset** to ensure fair comparison. Also report ML model performance on the **full test set (n=1,775)** for completeness. **Always clearly state the denominator.**

**Why this approach**:
1. **Eliminates risk-inversion artifact**: Avoids treating high-risk patients as low-risk
2. **Ensures fair comparison**: Same patients in both baseline and model denominators
3. **Maintains transparency**: Full-set numbers also reported, denominators always named
4. **Follows README.md guidance**: Implements trap 1 recommendation exactly

Alternative approaches rejected:
- Keeping -1 as numeric: Statistically indefensible given 2× mortality differential
- Median replacement: Fabricates scores and introduces bias toward the baseline

This protocol will be applied consistently in all subsequent modeling stages.

---

## Key Findings

1. **L0 establishes floor**: Random baseline AUROC=0.474, AUPRC=0.135 confirms models must substantially exceed this
2. **BUN max beats clinical scores**: Single variable (AUROC=0.675) outperforms SAPS-I (0.640) and SOFA (0.623)
3. **Age moderate, GCS weak**: Age AUROC=0.627 provides moderate discrimination; GCS min alone (0.553) is barely better than random
4. **Strong headroom for L3**: Individual lab values show predictive signal, suggesting multi-feature logistic regression should substantially exceed clinical scores
5. **-1 sentinel creates artifact**: 2.5% of test rows with unavailable scores have 2× mortality, requiring explicit handling protocol

---

## Next Steps for Stage 2

### L3: Small Logistic Regression (5-10 features)

**Goal**: Hand-pick 5-10 features and fit logistic regression to establish "simple ML" baseline

**Feature selection approach**:
- Start with L1 best performers: BUN max, Age
- Add physiologic criticality markers: GCS min, urine output metrics
- Add organ failure indicators from SOFA components: platelets, bilirubin
- Consider measurement behavior features: lactate_measured, count features

**Expected performance**: 实验方案 §5.5 preliminary test showed L3 (5 features, AUROC=0.738) substantially exceeds SAPS-I (0.657) and becomes the real opponent for L4

**Threshold selection protocol**:
- Use `find_threshold_for_recall(y_val, y_score_val, target_recall=0.80)` on **validation set**
- Report sensitivity, PPV, specificity at that threshold on test set
- Rationale: Missing a death (FN) costs far more than one unnecessary review (FP)

---

## How to Reproduce

### Data Validation
```bash
cd /mnt/d/personal_Profiles/files/NTU学系资料/Trimester1/Machine\ Learning\ for\ Healthcare\ AI/Group_project/experiment
python validate_data_loading.py
```

### Run Baselines
```bash
# L0: Random baseline
python baselines/run_L0_random.py

# L1: Single variables
python baselines/run_L1_static_age_years.py
python baselines/run_L1_gcs__0_48h__min.py
python baselines/run_L1_bun__0_48h__max.py

# L2: Clinical scores with sentinel analysis
python baselines/run_L2_clinical_scores.py
```

All results saved to `results/*.json` with structured metrics and CI.

---

## Files Created This Stage

```
experiment/
├── src/
│   ├── data.py              # NEW: Data loading utilities
│   └── metrics.py           # NEW: Evaluation metrics
├── baselines/
│   ├── run_L0_random.py     # NEW: L0 baseline
│   ├── run_L1_static_age_years.py   # NEW
│   ├── run_L1_gcs__0_48h__min.py    # NEW
│   ├── run_L1_bun__0_48h__max.py    # NEW
│   └── run_L2_clinical_scores.py    # NEW: L2 + sentinel analysis
├── results/
│   ├── L0_random_baseline.json
│   ├── L1_static_age_years.json
│   ├── L1_gcs__0_48h__min.json
│   ├── L1_bun__0_48h__max.json
│   └── L2_clinical_scores_analysis.json
└── validate_data_loading.py   # NEW: Data integrity checks
```

---

## Notes

- All metrics use stratified bootstrap (1,000 resamples, seed=42)
- AUPRC computed with `sklearn.metrics.average_precision_score` per 实验方案 §5.6
- Test set touched only for final evaluation; no hyperparameter tuning on test
- Feature matrix remains frozen; all experiments use same data snapshot from 2026-09-21
- Prevalence is **0.1443** (not 0.142 from plan) due to 167 negative-LOS exclusions
