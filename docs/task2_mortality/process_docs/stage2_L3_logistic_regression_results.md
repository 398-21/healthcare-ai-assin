> **Historical snapshot - superseded (2026-10-08).** This dated working report contains results from the earlier 11,833-stay Task 2 cohort. The current analysis retains all 12,000 stays. Use the repository README and current notebook/DATA_README for active results; the historical numbers below are not current.

---

# Stage 2: L3 Logistic Regression Baseline Results

**Date**: 2026-09-24  
**Author**: Experiment modeling team  
**Purpose**: Hand-select 5-10 clinically meaningful features for L3 baseline and establish the real opponent for L4

---

## Executive Summary

L3 logistic regression (8 features) achieved **AUROC=0.7417 [0.7075, 0.7755], AUPRC=0.3945 [0.3430, 0.4598]** on test set (n=1,775, prevalence=0.1443).

**Key finding**: L3 substantially outperforms both clinical scores (SAPS-I AUROC=0.640, SOFA AUROC=0.623) and is **the real opponent L4 must beat**, not the clinical baselines. Per 实验方案 §5.5: "如果最终 GBDT 相对 L3 增益很小，整套复杂管线就无法辩护。"

---

## Feature Selection Rationale

### Design Principles

1. **Clinical meaning drives selection**, not pure statistical screening
2. **Multi-organ coverage** aligned with SOFA domains
3. **Include L1 best performers** where clinically appropriate
4. **Measurement behavior features** for interpretability (缺失即信息, 实验方案 §3)

### Selected Features (n=8)

| # | Feature | Domain | SOFA Alignment | Coef | Direction |
|---|---------|--------|----------------|------|-----------|
| 1 | `bun__0_48h__max` | Renal | Renal component | +0.562 | ↑ BUN → ↑ risk |
| 2 | `static_age_years` | Demographic | Risk stratification | +0.324 | ↑ Age → ↑ risk |
| 3 | `urine__0_48h__min` | Renal | Renal component | −0.312 | ↓ Urine → ↑ risk |
| 4 | `lactate__0_48h__max` | Cardiovascular/Perfusion | Cardiovascular proxy | +0.302 | ↑ Lactate → ↑ risk |
| 5 | `gcs__0_48h__min` | CNS | Neurological component | −0.258 | ↓ GCS → ↑ risk |
| 6 | `lactate__0_48h__measured` | Measurement behavior | Physician decision | +0.207 | Measured → ↑ risk |
| 7 | `creatinine__0_48h__max` | Renal | Renal component | −0.176 | ↑ Creat → ↑ risk* |
| 8 | `platelets__0_48h__min` | Coagulation | Coagulation component | +0.069 | ↓ Platelet → ↑ risk* |

*Note: Negative coefficients for creatinine and positive for platelets contradict clinical expectation. This may indicate collinearity with BUN/other renal markers or class-balancing effects. Will require SHAP analysis in L4 stage.

### Organ System Coverage

- **Renal** (3 features): BUN max, urine min, creatinine max
- **Demographic** (1 feature): Age
- **CNS** (1 feature): GCS min
- **Cardiovascular/Perfusion** (1 feature): Lactate max
- **Measurement behavior** (1 feature): Lactate measured
- **Coagulation** (1 feature): Platelets min

**Coverage: 6 domains, 5 of 6 SOFA components** (missing: liver, respiratory)

### Clinical Justification by Feature

1. **BUN max**: Best L1 single variable (AUROC=0.675). BUN elevation indicates renal dysfunction and catabolic state. Strongest coefficient (+0.562) confirms importance.

2. **Age**: Strong demographic predictor (L1 AUROC=0.627). Captures physiologic reserve. Second strongest coefficient (+0.324).

3. **Urine min**: Minimum urine captures worst oliguria episode. Oliguria/anuria is critical mortality signal (实验方案 §3: "Urine=0 是真实的少尿/无尿，不是缺失"). Negative coefficient (−0.312) correct: lower urine → higher risk.

4. **Lactate max**: Peak lactate indicates tissue hypoperfusion and anaerobic metabolism. Gold standard shock marker. Positive coefficient (+0.302) correct.

5. **GCS min**: Core ICU neurological assessment, directly used in SOFA. Negative coefficient (−0.258) correct: lower consciousness → higher risk.

6. **Lactate measured**: Measurement behavior feature. Whether lactate was ordered reflects physician concern for shock (实验方案 §8: "缺失即信息"). Positive coefficient (+0.207) confirms this hypothesis.

7. **Creatinine max**: Standard AKI marker, complements BUN. Negative coefficient (−0.176) unexpected, likely collinearity effect.

8. **Platelets min**: SOFA coagulation component. Thrombocytopenia indicates consumption/DIC. Positive coefficient (+0.069) unexpected, weakest predictor.

---

## Results

### Performance Across Splits (Overfitting Check)

| Split | AUROC [95% CI] | AUPRC [95% CI] | Notes |
|-------|----------------|----------------|-------|
| **Train** | 0.7334 [0.7192, 0.7489] | 0.3340 [0.3133, 0.3595] | Training set |
| **Validation** | 0.7217 [0.6883, 0.7524] | 0.3063 [0.2678, 0.3525] | For model selection |
| **Test** | 0.7417 [0.7075, 0.7740] | 0.3945 [0.3430, 0.4546] | Final evaluation |

**Overfitting assessment**: Train-Val AUROC gap = 0.0117 (< 0.05 threshold). **No severe overfitting detected.** Test performance slightly exceeds validation, which is within normal split variation.

### Comparison to Baseline Ladder

| Level | Model | AUROC | AUPRC | Δ AUROC from L3 |
|-------|-------|-------|-------|-----------------|
| L0 | Random | 0.474 | 0.135 | −0.268 |
| L1 | Age only | 0.627 | 0.224 | −0.115 |
| L1 | GCS min only | 0.553 | 0.155 | −0.189 |
| L1 | BUN max only | 0.675 | 0.265 | −0.067 |
| L2 | SAPS-I | 0.640 | 0.213 | −0.102 |
| L2 | SOFA | 0.623 | 0.253 | −0.119 |
| **L3** | **8-feature LR** | **0.7417** | **0.3945** | — |

**Key insights**:
- L3 beats SAPS-I by **+0.102 AUROC** (16% relative improvement)
- L3 beats SOFA by **+0.119 AUROC** (19% relative improvement)
- L3 beats best single variable (BUN) by **+0.067 AUROC** (10% relative improvement)
- AUPRC improvement even more dramatic: +0.142 vs SAPS-I (67% relative), +0.129 vs BUN (49% relative)

### Threshold-Based Performance

Threshold selected on validation set for **target recall=0.80**:
- **Threshold**: 0.4204
- **Test set at this threshold**:
  - Sensitivity: 0.7969 (captured 80% of deaths)
  - PPV: 0.2201 (22% of alarms are true deaths)
  - Specificity: 0.5240
- **PhysioNet 2012 Challenge metric** min(Se, PPV): **0.2201**

---

## Model Details

- **Algorithm**: Logistic Regression with L2 regularization
- **Hyperparameters**: C=1.0, class_weight='balanced', solver='lbfgs'
- **Preprocessing**: Median imputation (from training set), standardization
- **Missingness**: All 8 features had 0% missing in imputed data (confirmed data quality)
- **Validation AUROC**: 0.7217 (test=0.7417 shows good generalization, no overfitting)

---

## Self-Reflection and Issues

### What Worked

1. **Clinical-first selection paid off**: L3 substantially exceeds clinical scores despite using only 8 features vs SAPS-I's 14.

2. **L1 guidance was valuable**: Starting with L1 best performers (BUN, Age) gave strong foundation. BUN has highest coefficient, Age second highest.

3. **Measurement behavior hypothesis confirmed**: `lactate__0_48h__measured` has positive coefficient (+0.207), supporting "缺失即信息" — doctors order lactate when they suspect shock, making the measurement itself a risk signal.

4. **Multi-organ coverage achieved**: 6 domains covered, 5 of 6 SOFA components represented.

### Unexpected Findings Requiring Investigation

1. **Creatinine coefficient sign wrong** (−0.176): Higher creatinine should increase risk, not decrease. Likely explanations:
   - **Collinearity with BUN**: Both measure renal function, model may be treating them as redundant
   - **Class weighting artifact**: `class_weight='balanced'` may have introduced unexpected interactions
   - **Recommendation**: Check correlation matrix, consider dropping one renal marker

2. **Platelets coefficient sign wrong** (+0.069): Lower platelets should increase risk. Coefficient also weakest in magnitude. Likely explanations:
   - **Coagulation less important in this cohort**: Unlike sepsis-specific datasets, general ICU may have weaker platelet-mortality link
   - **Mild thrombocytopenia protective?**: Possible survivor bias (very sick patients die before labs drawn)
   - **Recommendation**: Check distribution, consider replacing with bilirubin (liver) for better SOFA coverage

3. **No respiratory/liver features**: SOFA respiratory (PaO2/FiO2) and liver (bilirubin) components not selected. May be limiting performance.

### Comparison to 实验方案 Preliminary Results

实验方案 §5.5 reported preliminary L3 (5 features,粗特征, 未调参) achieved AUROC=0.738 on old test set. Current results:
- **8 features, proper imputation/scaling** → AUROC=0.7417
- Comparable performance with cleaner methodology
- Difference likely due to different test split and proper preprocessing

### L3 as the Real Opponent

Per 实验方案 §5.5: "L3（5 特征逻辑回归，0.738）才是真正要打败的对手，不是 SAPS-I。如果最终 GBDT 只到 0.75，说明整套复杂管线相对 5 个特征几乎无增益，复杂度无法辩护。"

**L4 target**: Must exceed **AUROC=0.74** to justify complexity. Literature reports 0.83–0.86 on this dataset, suggesting substantial headroom exists.

---

## Next Steps for Stage 3 (L4 GBDT)

### Feature Engineering

1. **Add respiratory component**: FiO2 max, MechVent ever, respiratory rate
2. **Add liver component**: Bilirubin max (for complete SOFA coverage)
3. **Add temporal features**: 24h vs 48h window differences, trends (delta, slope)
4. **Add derived clinical features**:
   - PaO2/FiO2 ratio (SOFA respiratory)
   - BUN/Creatinine ratio
   - Shock index (HR/SysABP)

### Model

1. **XGBoost or LightGBM**: Use full 1,474-feature set
2. **Hyperparameter tuning**: Grid search on validation set
3. **Feature importance**: SHAP analysis to validate clinical sensibility

### Success Criteria

- **Minimum acceptable**: AUROC > 0.75 (exceed L3 by meaningful margin)
- **Good performance**: AUROC > 0.80
- **Literature-competitive**: AUROC 0.83–0.86

### Validation

- SHAP top features must align with clinical knowledge
- Check for unexpected features (data leakage signals)
- Sub-group analysis by ICUType

---

## Files Generated This Stage

```
experiment/
├── baselines/
│   ├── run_L3_logistic_regression.py           # NEW: L3 training script
│   ├── run_L3_threshold_analysis.py            # NEW: Threshold selection
│   └── run_L3_supplementary_evaluation.py      # NEW: Phase 3 补充评估
├── results/
│   ├── L3_logistic_regression_8features.json   # NEW: Full results with CI
│   ├── L3_threshold_analysis.json              # NEW: Operating point metrics
│   └── L3_supplementary_evaluation.json        # NEW: Validation performance + statistical tests
└── stage2_L3_logistic_regression_results.md    # THIS FILE
```

**Note on supplementary evaluation**: Workflow phase 3 "Evaluate Performance" was inadvertently skipped. `run_L3_supplementary_evaluation.py` was created to fill gaps: validation performance (overfitting check), structured L2 comparison, and statistical significance tests. Key findings confirmed no overfitting and statistically significant improvements over clinical baselines.

---

## Reproducibility

```bash
cd /path/to/Group_project/experiment

# Main L3 training and evaluation
python baselines/run_L3_logistic_regression.py

# Threshold analysis (recall≈0.80 on validation)
python baselines/run_L3_threshold_analysis.py

# Supplementary evaluation (validation performance, statistical tests)
python baselines/run_L3_supplementary_evaluation.py
```

Outputs: 
- `results/L3_logistic_regression_8features.json`
- `results/L3_threshold_analysis.json`
- `results/L3_supplementary_evaluation.json`

---

**Conclusion**: L3 baseline (8 features, AUROC=0.7417) is solid and clinically interpretable. It substantially beats clinical scores and establishes a meaningful bar for L4. The combination of renal markers (BUN, creatinine, urine), age, neurological status (GCS), perfusion (lactate), and measurement behavior creates a simple yet effective mortality predictor. Next stage must demonstrate that full feature set + GBDT complexity adds meaningful value beyond this interpretable baseline.
