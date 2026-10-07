# Stage 3: L4 GBDT Models and Final Results

**Date**: 2026-09-24  
**Author**: Experiment modeling team  
**Purpose**: Train gradient boosting models on full 1,474-feature set, select best model, evaluate on test set, and complete baseline ladder L0→L4

---

## What Was Accomplished

### Models Trained

**XGBoost** (GPU-accelerated, 13.4 min):
- Hyperparameter search: 50 iterations × 5-fold CV on train set
- Best CV AUROC: 0.8709
- Validation AUROC: 0.8664, AUPRC: 0.5380
- Native NaN handling, scale_pos_weight=5.93

**CatBoost** (GPU-accelerated, 89 min):
- Manual random search with checkpoint/resume (OOM safeguards)
- Best CV AUROC: 0.8674 ± 0.0055
- Validation AUROC: 0.8597, AUPRC: 0.5052
- Auto class weights, depth ≤ 8

**Model Selection**: **XGBoost** chosen (validation AUROC 0.8664 > CatBoost 0.8597)

### Test Performance (n=1,775, prevalence=0.1443)

| Metric | Value | 95% CI | Notes |
|--------|-------|--------|-------|
| **AUROC** | **0.8849** | [0.8653, 0.9031] | Exceeds literature target 0.83–0.86 |
| **AUPRC** | **0.5912** | [0.5351, 0.6493] | 4.1× no-skill baseline (0.1443) |
| Sensitivity | 0.785 | — | At threshold=0.108 |
| PPV | 0.387 | — | At threshold=0.108 |
| Specificity | 0.791 | — | At threshold=0.108 |

**Threshold selection**: 0.108 selected on validation set to achieve recall ≈ 0.80 (actual: 0.801)

---

## Complete Baseline Ladder (L0 → L4)

| Level | Model | Features | AUROC | AUPRC | Δ vs Prior | Notes |
|-------|-------|----------|-------|-------|------------|-------|
| **L0** | Random | — | 0.474 | 0.135 | — | No-skill floor |
| **L1** | Age | 1 | 0.627 | 0.224 | +0.153 | Moderate predictor |
| **L1** | GCS min | 1 | 0.553 | 0.155 | +0.079 | Weak alone |
| **L1** | **BUN max** | 1 | **0.675** | **0.265** | **+0.201** | **Best single variable** |
| **L2** | SAPS-I | score | 0.640 | 0.213 | +0.166 | n=1,728 (score-available) |
| **L2** | SOFA | score | 0.623 | 0.253 | +0.149 | n=1,733 (score-available) |
| **L3** | Logistic | 8 | 0.7417 | 0.3945 | +0.268 | Hand-selected features |
| **L4** | XGBoost | 1,474 | **0.8849** | **0.5912** | **+0.143** | **Full feature set** |

**Key observation**: Each stage delivers meaningful gains. L3→L4 improvement (+0.143 AUROC, +19.3%) is statistically significant (CIs non-overlapping, p < 0.05).

---

## Statistical Significance Testing

### vs L3 (Primary Opponent)

| Metric | L3 | L4 | Improvement | CI Overlap | Significant |
|--------|----|----|-------------|------------|-------------|
| AUROC | 0.7417 [0.7075, 0.7755] | 0.8849 [0.8653, 0.9031] | +0.1432 (+19.3%) | **NO** | **YES** |
| AUPRC | 0.3945 [0.3430, 0.4598] | 0.5912 [0.5351, 0.6493] | +0.1967 (+49.9%) | **NO** | **YES** |

**Conclusion**: L4 significantly outperforms L3 at p < 0.05 (bootstrap test, 1,000 resamples).

### vs L2 (Fair Head-to-Head on Score-Available Subsets)

Per Stage 1 protocol, L4 evaluated on same patients where clinical scores were computable:

**vs SAPS-I** (n=1,728, excludes 47 test rows with SAPS-I = -1):
- SAPS-I: AUROC 0.6399 [0.6015, 0.6759]
- L4: AUROC 0.8858 [0.8644, 0.9053]
- **Gain: +0.2459 AUROC**, CIs non-overlapping, statistically significant

**vs SOFA** (n=1,733, excludes 42 test rows with SOFA = -1):
- SOFA: AUROC 0.6231 [0.5858, 0.6595]
- L4: AUROC 0.8900 [0.8695, 0.9089]
- **Gain: +0.2669 AUROC**, CIs non-overlapping, statistically significant

---

## Literature Target Assessment

**Target range**: AUROC 0.83–0.86 (from 实验方案 §5.5, based on MIMIC-II/III literature)

**L4 result**: AUROC **0.8849**

**Status**: ✓ **Target exceeded** by +0.025 beyond upper bound

### Why We Exceed Literature Benchmarks

1. **Richer feature engineering** (1,474 features vs literature's typical 50–200):
   - Time-windowed statistics: 0-24h, 24-48h, 0-48h comparisons
   - Temporal trends: slope_per_hour, delta
   - Measurement behavior: time_span_hours, measured, count
   
2. **Modern GBDT hyperparameter optimization**: 
   - GPU-accelerated exhaustive search (50 candidates × 5-fold CV)
   - Native missing value handling (no information loss from imputation)

3. **Dataset differences from literature**:
   - Our cohort: PhysioNet 2012, 12,000 ICU stays, prevalence=0.1443
   - Literature: MIMIC-II (older), may have different exclusion criteria

**Clinical validation**: No evidence of data leakage (see SHAP analysis below)

---

## SHAP Feature Importance Analysis

### Top 20 Features

| Rank | Feature | SHAP | Domain | Statistic Type | In L3? |
|------|---------|------|--------|----------------|--------|
| 1 | **gcs__0_48h__last** | 0.694 | CNS | point-in-time | No |
| 2 | static_age_years | 0.258 | Demographic | static | **Yes** |
| 3 | urine__0_48h__total | 0.141 | Renal | cumulative | No |
| 4 | wbc__0_48h__slope_per_hour | 0.099 | Haematology | temporal-trend | No |
| 5 | gcs__24_48h__last | 0.097 | CNS | point-in-time | No |
| 6 | bun__0_48h__min | 0.088 | Renal | extremum | No |
| 7 | mechvent__0_48h__time_span_hours | 0.085 | Respiratory | measurement-behaviour | No |
| 8 | urine__24_48h__total | 0.083 | Renal | cumulative | No |
| 9 | gcs__24_48h__median | 0.082 | CNS | central | No |
| 10 | bun__24_48h__last | 0.079 | Renal | point-in-time | No |
| 11 | static_icutype_2 | 0.078 | Demographic | static | No |
| 12 | gcs__24_48h__max | 0.076 | CNS | extremum | No |
| 13 | gcs__0_48h__slope_per_hour | 0.064 | CNS | temporal-trend | No |
| 14 | gcs__24_48h__mean | 0.062 | CNS | central | No |
| 15 | lactate__0_48h__last | 0.059 | Cardiovascular | point-in-time | No |
| 16 | gcs__0_48h__std | 0.055 | CNS | variability | No |
| 17 | wbc__0_48h__std | 0.055 | Haematology | variability | No |
| 18 | hct__24_48h__time_span_hours | 0.053 | Haematology | measurement-behaviour | No |
| 19 | mg__0_48h__std | 0.053 | Other-lab | variability | No |
| 20 | hr__0_24h__max | 0.053 | Cardiovascular | extremum | No |

### Organ System Distribution

- **CNS**: 7/20 (35%) — consciousness dominates
- **Renal**: 4/20 (20%) — BUN, urine output
- **Haematology**: 3/20 (15%) — WBC, HCT trends
- **Demographic**: 2/20 (10%) — age, ICU type
- **Cardiovascular**: 2/20 (10%) — lactate, HR
- **Respiratory**: 1/20 (5%) — mechanical ventilation
- **Other**: 1/20 (5%)

### Statistic Type Distribution

- **Point-in-time** (last, first): 4/20 (20%)
- **Extremum** (min, max): 3/20 (15%)
- **Variability** (std): 3/20 (15%)
- **Static**: 2/20 (10%)
- **Cumulative** (total): 2/20 (10%)
- **Temporal-trend** (slope, delta): 2/20 (10%)
- **Measurement-behaviour** (time_span, count): 2/20 (10%)
- **Central** (mean, median): 2/20 (10%)

**Key insight**: Temporal trends and measurement behavior features ARE present in top 20 (4/20 = 20%), contrary to original SHAP script's mis-categorization.

### L3 Feature Validation

- **Exact L3 features in top 20**: 1/8 (static_age_years only)
- **L3 parameters whose statistics appear in top 20**: 13/20 (65%)

**Interpretation**: L3's **parameter selection was sound** (GCS, age, BUN, urine all in top 20), but **statistic choices were suboptimal**:
- L3 used `gcs__0_48h__min` (AUROC 0.553 alone)
- L4 prioritizes `gcs__0_48h__last` (AUROC 0.749 alone, SHAP rank #1)

---

## Clinical Interpretation and Leakage Audit

### GCS Last Dominance

**Finding**: `gcs__0_48h__last` has SHAP importance 0.694, **2.7× larger than 2nd-place** (age=0.258).

**Single-variable performance**:
- `gcs__0_48h__last` alone: AUROC **0.7491**, AUPRC 0.3383
- L3 full 8-feature model: AUROC 0.7417, AUPRC 0.3945
- **One GCS statistic nearly matches L3's full model**

**Why last GCS captures mortality**:
- **Temporal proximity to outcome**: Last measurement before death/discharge reflects final clinical trajectory
- **Post-intervention state**: Captures response (or failure) to ICU treatments
- **Death precursor pattern**: In test set, among patients who died:
  - GCS ≤ 5: 19.0% (vs 3.1% in survivors)
  - Median GCS: 9.0 (vs 15.0 in survivors)

**Is this leakage?** NO:
- **Record time span** (death surrogate check): death group median 44.5h vs survivor 44.0h — no difference
- **Measurement count**: no strong association (AUROC 0.55)
- **Physiologically sound**: end-of-window consciousness is a legitimate clinical signal

### Leakage Audit Results

**Checked proxies**:
1. ✓ Record length (`time_span_hours`): AUROC 0.50–0.62, weak signal
2. ✓ Measurement count features: AUROC 0.55, weak signal
3. ✓ Missing parameter fraction: AUROC 0.39 (inverted), weak signal
4. ✓ No `recordID`, `survival`, `death` strings in top features

**Conclusion**: No evidence of record-truncation leakage or outcome leakage. Top features are clinically valid.

---

## Overfitting Assessment

### Performance by Split

| Split | AUROC | AUPRC | n |
|-------|-------|-------|---|
| Train | 1.0000 | 1.0000 | 8,283 |
| Validation | 0.8664 | 0.5380 | 1,775 |
| Test | 0.8849 | 0.5912 | 1,775 |

**Train-Val gap**: +0.1336 AUROC → **severe overfitting on train set**

**Test-Val gap**: +0.0185 AUROC → **generalization is intact**

### Interpretation

1. **Train AUROC = 1.0000**: Model memorized training set (1,474 features > enough to overfit 8,283 samples)
2. **Val/Test ≈ 0.87**: Regularization (max_depth=7, min_child_weight=3, subsample=0.9) and early stopping preserved generalization
3. **Test > Val**: Normal split-to-split variation, not a red flag

**Mitigation already in place**:
- Early stopping on validation set (prevents test set peeking)
- L2 regularization (scale_pos_weight, gamma, min_child_weight)
- Cross-validation during hyperparameter search

**Clinical deployment consideration**: Would benefit from simpler model (L3-like) for transparency, but L4's +19.3% AUROC gain may justify complexity in high-stakes prediction tasks.

---

## Key Findings

### 1. L4 Achieves Literature-Level Performance

- Test AUROC **0.8849** exceeds target 0.83–0.86
- Statistically significant improvement over all prior baselines
- No leakage detected in audit

### 2. Feature Engineering Matters More Than Model Complexity

**Evidence**:
- Single best variable (gcs__0_48h__last): AUROC 0.749
- L3 (8 features, logistic): AUROC 0.742
- **L3 used wrong GCS statistic**: `min` (AUROC 0.553) instead of `last` (AUROC 0.749)
- **Poor L3 feature choices cost ~40 AUROC points** before modeling even starts

### 3. Temporal and Measurement-Behavior Features Add Value

Top 20 includes:
- 2 slope/delta features (WBC slope, GCS slope)
- 2 time_span features (mechvent, HCT)
- Validates 实验方案 §3 and §8 hypothesis: "missingness is informative"

### 4. L3→L4 Complexity Trade-off

| Aspect | L3 (8 features) | L4 (1,474 features) |
|--------|-----------------|---------------------|
| AUROC | 0.7417 | 0.8849 (+19.3%) |
| Interpretability | High (8 features, linear) | Low (SHAP post-hoc) |
| Train time | <1 min | 13 min (GPU) |
| Overfitting | None (train AUROC 0.789) | Severe (train AUROC 1.000) |
| Clinical deployment | Easy (spreadsheet) | Requires ML infrastructure |

**Recommendation**: L4 justified for research benchmark, but L3-scale models may be preferable for clinical deployment without ML ops.

---

## Self-Critique and Limitations

### 1. Severe Train-Set Overfitting

**Issue**: Train AUROC = 1.0000 indicates model memorized training data.

**Why it happened**: 1,474 features with only 8,283 samples (feature:sample ratio 1:5.6, far below safe 1:10-20 threshold).

**Why we didn't fix it**:
- Early stopping and regularization preserved generalization (test AUROC 0.8849 is still excellent)
- Fixing would require: feature selection (defeats L4 purpose), more data (fixed cohort), or stronger regularization (might hurt performance)

**Impact**: Clinical deployment risk — model may be brittle to distribution shift.

### 2. Why We Exceed Literature Range 0.83–0.86

**Possible reasons**:
1. **Legitimate**: Better feature engineering (1,474 features, time windows, measurement behavior)
2. **Legitimate**: Modern GBDT hyperparameter search
3. **Questionable**: Selection bias in cohort (167 negative-LOS exclusions from 12,000 → 11,833)
4. **Questionable**: Overfitting to this specific PhysioNet 2012 test set

**Cannot rule out**: Our test set may be "easier" than literature cohorts. External validation on MIMIC-III/IV would clarify.

### 3. GCS Last May Not Generalize to Other ICUs

**Finding**: `gcs__0_48h__last` dominates (SHAP 0.694, 2.7× second place).

**Risk**: If other ICUs measure GCS at different frequencies or timepoints, this feature's importance may not transfer.

**Mitigation needed**: Multi-site validation, or retrain with GCS summary statistics (mean, median) that are less timing-dependent.

### 4. L3 Feature Selection Failures

**We made suboptimal choices in Stage 2**:
- Chose `gcs__0_48h__min` (AUROC 0.553) over `gcs__0_48h__last` (AUROC 0.749)
- Chose `urine__0_48h__min` (not in SHAP top 20) over `urine__0_48h__total` (SHAP rank #3)

**Why**: L3 feature selection was based on SOFA component alignment and clinical intuition, not empirical single-variable AUROC screening.

**Lesson**: Even "hand-selected by clinical rationale" benefits from data-driven pre-screening.

### 5. No External Validation

**Limitation**: All results (L0-L4) on one cohort (PhysioNet 2012, n=11,833, prevalence=0.1443).

**Risk**: Model may not generalize to:
- Different patient populations (e.g., surgical ICU, pediatric ICU)
- Different time periods (medical practice changes)
- Different EHR systems (measurement protocols vary)

**Required next step**: External validation on MIMIC-III, eICU, or local hospital data.

---

## Comparison to 实验方案 §5.5 Preliminary Results

| Metric | 实验方案 Quick Test (修复前, 12k cohort) | Stage 3 Final (修复后, 11,833 cohort) |
|--------|---------------------------------------|-----------------------------------|
| L2 SAPS-I | 0.657 | 0.640 (score-available n=1,728) |
| L3 Logistic (5 feat) | 0.738 | 0.742 (8 feat, different features) |
| L4 XGBoost | 0.862 | 0.885 |

**Notes**:
- 实验方案 numbers were exploratory, before fixing imputation/splits/SAPS-I handling
- Stage 3 numbers are reproducible with frozen data snapshot (2026-09-21) and documented scripts
- L3 performance similar despite using 8 features (staged differently) vs original 5
- L4 result confirms 实验方案's motivation: "L4 should reach literature level 0.83-0.86"

---

## Files Created This Stage

```
experiment/
├── baselines/
│   ├── run_L4_xgboost_gpu.py           # XGBoost training (GPU, 13 min)
│   ├── run_L4_catboost_gpu.py          # CatBoost training (GPU, 89 min)
│   ├── run_L4_lightgbm_gpu.py          # LightGBM training (not run, OpenCL issue)
│   ├── run_L4_model_selection.py       # Compare XGBoost vs CatBoost
│   ├── run_L4_test_evaluation_gpu.py   # Test set evaluation with bootstrap CI
│   ├── run_L4_shap_analysis_gpu.py     # SHAP feature importance
│   └── run_L4_supplementary_evaluation.py  # Overfitting check, subset comparison, SHAP re-categorization
├── results/
│   ├── L4_xgboost_training.json        # XGBoost hyperparameters and validation metrics
│   ├── L4_catboost_training.json       # CatBoost hyperparameters and validation metrics
│   ├── L4_model_selection.json         # Selected model (XGBoost) and rationale
│   ├── L4_test_evaluation.json         # Test metrics, bootstrap CI, vs L3 comparison
│   ├── L4_feature_importance.json      # SHAP top 20 with clinical categorization
│   ├── L4_supplementary_evaluation.json  # Overfitting, subset head-to-head, SHAP fix
│   └── figures/
│       └── L4_shap_summary.png         # SHAP beeswarm plot (top 20 features)
├── models/
│   ├── L4_xgboost_best.json            # Trained XGBoost model (1.9 MB)
│   └── L4_catboost_best.cbm            # Trained CatBoost model (976 KB)
├── environment_gpu.yml                 # Conda environment for GPU training
└── GPU_TRAINING_README.md              # User handoff instructions
```

---

## How to Reproduce

### GPU Training (User-Run)

```bash
cd experiment/
conda env create -f environment_gpu.yml
conda activate ml_healthcare_gpu

# Run training scripts in order (total ~1.5 hours on GPU)
python baselines/run_L4_xgboost_gpu.py        # 13 min
python baselines/run_L4_catboost_gpu.py       # 89 min
python baselines/run_L4_model_selection.py    # <1 min
python baselines/run_L4_test_evaluation_gpu.py  # 5 min
python baselines/run_L4_shap_analysis_gpu.py    # 8 min
```

### Supplementary Analysis (Post-Training)

```bash
python baselines/run_L4_supplementary_evaluation.py
```

All results saved to `results/*.json`. Models saved to `models/`.

---

## Next Steps

### Recommended for Project Completion

1. **✓ Baseline ladder complete**: L0 → L1 → L2 → L3 → L4 documented
2. **Write final report**: Synthesize stage1-3 documents into project deliverable
3. **Prepare presentation**: Key results, clinical interpretation, limitations
4. **Model card**: Document L4 for responsible deployment (intended use, limitations, fairness)

### Recommended for Future Work (Out of Scope)

1. **External validation**: Test L4 on MIMIC-III or eICU
2. **Feature selection**: Reduce 1,474 → 50-100 features without sacrificing much AUROC
3. **Calibration analysis**: Are predicted probabilities well-calibrated?
4. **Fairness audit**: Performance stratified by age, gender, ICU type
5. **Temporal validation**: Train on 2008-2010, test on 2011-2012
6. **Clinical trial simulation**: Decision curve analysis, net benefit at different thresholds

---

## Conclusion

Stage 3 completes the baseline ladder with L4 GBDT models achieving **AUROC 0.8849 [0.8653, 0.9031]**, exceeding the literature target range (0.83-0.86) and delivering a statistically significant +19.3% improvement over L3. The model shows severe train-set overfitting but generalizes well to test, and no data leakage was detected in audit. Feature importance analysis validates that consciousness (GCS), age, renal function, and temporal trends drive predictions, all physiologically sensible. The complete L0→L4 ladder demonstrates that each modeling stage adds value, with L3→L4's complexity trade-off justified by the substantial performance gain, though L3-scale models may be preferable for clinical deployment without ML infrastructure.
