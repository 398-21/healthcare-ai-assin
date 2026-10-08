> **Historical snapshot - superseded (2026-10-08).** This dated working report contains results from the earlier 11,833-stay Task 2 cohort. The current analysis retains all 12,000 stays. Use the repository README and current notebook/DATA_README for active results; the historical numbers below are not current.

---

# Stage 2 (v2): L3 Logistic Regression — Original Rung Plus a Pre-Declared Revised Rung

**Date**: 2026-09-25
**Supersedes**: `stage2_L3_logistic_regression_results.md` (v1, 2026-09-24). v1 is retained unchanged; everything in it still holds. This revision adds one experiment and the reasoning behind it.
**Test-set policy**: the original L3 test numbers are carried over verbatim. The revised rung was scored on test **once**, with its feature set fixed in the script before first execution.

---

## Lineage

| Item | v1 | v2 |
|---|---|---|
| L3 rung (8 features, `gcs__0_48h__min`, `urine__0_48h__min`) | Defined, scored on test once: AUROC 0.7417 | **Unchanged.** Still the official L3 rung. |
| L3 AUROC 95% CI | 0.7755 in the summary, 0.7740 in the split table (inconsistent) | 0.7740 throughout, reproducible with `metrics.bootstrap_ci` seed 42. JSON corrected with a provenance note. |
| Feature-statistic choice | Clinical rationale only | Same choice kept for the original rung. A train+validation screen (§2) shows two statistics were mis-specified. |
| L3-revised rung | — | **New.** Same recipe, two statistics swapped, pre-declared, test scored once (§3). |

---

## 1. Why a revised rung, and why now

Stage 3's SHAP put `gcs__0_48h__last` first among 1,474 features and `gcs__0_48h__min`, the statistic L3 chose, at rank 326. That SHAP was computed on the test set, so it cannot be fed back into feature selection. The question stage 3 left open was whether the same conclusion could be reached without the test set, and if so, what it would do to the ladder.

It matters for the project's central argument. 实验方案 §5.5 says L3, not the clinical scores, is the opponent L4 must beat, and that if GBDT gains little over a handful of features the whole pipeline is indefensible. That argument is only as strong as L3 is. If L3 was weakened by a mis-chosen summary statistic, the L3→L4 gap of 0.143 overstates what the 1,474-feature model buys, and a reviewer who spots the GCS issue can dismiss the comparison. A properly specified L3 makes the L4 gain smaller but credible.

## 2. Evidence that never touches the test set

### 2.1 Single-variable screen, train + validation (n = 10,058; 1,451 deaths)

All twelve statistics of GCS over 0–48 h, AUROC as a lone predictor:

| Statistic | AUROC | | Statistic | AUROC |
|---|---|---|---|---|
| **last** | **0.757** | | **min (L3 chose)** | **0.589** |
| median | 0.722 | | time_span_hours | 0.543 |
| mean | 0.707 | | first | 0.536 |
| max | 0.686 | | count | 0.532 |
| slope_per_hour | 0.666 | | std | 0.523 |
| delta | 0.665 | | measured | 0.501 |

`last` leads by 0.035 over the next statistic and by 0.168 over `min`. Stage 1's L1 rung had already reported `gcs__0_48h__min` at 0.553 on test and called it weak, so this signal was on record before stage 2 selected it.

### 2.2 Model-level ablation, validation metrics only

Identical L3 recipe (L2 logistic, C = 1, `class_weight='balanced'`, train-median imputation, `StandardScaler` fit on train), swapping one or two statistics:

| Variant | Train AUROC | Val AUROC | Val AUPRC | Δ val AUROC |
|---|---|---|---|---|
| A. Original L3 | 0.733 | 0.7217 | 0.3063 | — |
| B. gcs min → last | 0.813 | 0.7985 | 0.4225 | +0.077 |
| C. gcs last added, min kept (9 features) | 0.822 | 0.7990 | 0.4159 | +0.077 |
| D. urine min → total | 0.747 | 0.7430 | 0.3369 | +0.021 |
| **E. B + D** | **0.819** | **0.8081** | **0.4374** | **+0.086** |

Two things to note. The GCS swap moves the model by +0.077, less than half its single-variable margin, because the other seven features overlap with it. And C ≈ B: once `last` is in, `min` adds nothing, so `min` is not carrying independent information that `last` misses.

Source: `results/L3_gcs_statistic_ablation.json`, `baselines/run_L3_gcs_statistic_ablation.py`.

### 2.3 Why `last` beats `min` clinically

Nearly every ICU patient posts at least one low GCS in 48 hours: sedation, intubation and procedures all depress it. `min` therefore saturates near the floor and cannot tell treatment-induced lows from prognostic ones. `last` reads consciousness at the end of the window, after those effects have worn off. That is the difference between a measurement artefact and a physiological signal, which is the thread 实验方案 §3 and §8 asked the project to pull on. The SOFA convention of using worst-in-window GCS was designed for bedside scoring on a 24-hour cycle, not for a 48-hour retrospective feature.

## 3. L3-revised: declared, then scored once

### 3.1 Declaration

Feature set is variant E, copied verbatim into `baselines/run_L3_revised_test_evaluation.py` before that script was first run. No search takes place in the script. Recipe identical to L3. Threshold chosen on validation at recall 0.80.

| # | Feature | Change from L3 |
|---|---|---|
| 1 | bun__0_48h__max | — |
| 2 | static_age_years | — |
| 3 | **gcs__0_48h__last** | was gcs__0_48h__min |
| 4 | **urine__0_48h__total** | was urine__0_48h__min |
| 5 | lactate__0_48h__max | — |
| 6 | platelets__0_48h__min | — |
| 7 | creatinine__0_48h__max | — |
| 8 | lactate__0_48h__measured | — |

### 3.2 Results

| Split | AUROC | AUPRC |
|---|---|---|
| Train | 0.8192 | 0.4697 |
| Validation | 0.8081 | 0.4374 |
| **Test** (n = 1,775) | **0.8207 [0.7932, 0.8459]** | **0.5002 [0.4447, 0.5560]** |

Train–validation gap 0.011: no capacity concern. Validation matches the ablation exactly (0.8081), confirming the script reproduces variant E.

Operating point (threshold 0.395 from validation, recall achieved 0.801): test sensitivity 0.840, PPV 0.297, specificity 0.666, min(Se, PPV) 0.297.

### 3.3 Standardised coefficients

| Feature | Coef | Sign as expected? |
|---|---|---|
| gcs__0_48h__last | −0.916 | Yes (lower GCS → higher risk) |
| bun__0_48h__max | +0.526 | Yes |
| urine__0_48h__total | −0.370 | Yes (less urine → higher risk) |
| static_age_years | +0.331 | Yes |
| creatinine__0_48h__max | −0.238 | No (collinear with BUN, as in v1) |
| lactate__0_48h__max | +0.147 | Yes |
| lactate__0_48h__measured | +0.053 | Yes (ordered → concern), now small |
| platelets__0_48h__min | +0.023 | No (weak, as in v1) |

GCS-last is now the largest coefficient in the linear model, as it is the top SHAP feature in the GBDT. Two things moved relative to v1's original L3: the creatinine sign anomaly got slightly larger (−0.176 → −0.238), and the two lactate features shrank (max 0.302 → 0.147; measured 0.207 → 0.053). Plausible reading: GCS-last and urine-total now carry perfusion information that lactate was proxying for. Not tested; noted for the report.

Source: `results/L3_revised_test_evaluation.json`.

## 4. What this does to the ladder

| Rung | Test AUROC [95% CI] | vs L3 original | vs L4 |
|---|---|---|---|
| L3 original | 0.7417 [0.7075, 0.7740] | — | −0.143 |
| **L3-revised** | **0.8207 [0.7932, 0.8459]** | **+0.079, CIs disjoint** | **−0.064, CIs disjoint** |
| L4 XGBoost | 0.8849 [0.8653, 0.9031] | +0.143 | — |

- **L4 still wins, and the win is still significant.** The gap shrinks from 0.143 to 0.064, but the CIs do not overlap. A 1,474-feature GBDT is worth about six AUROC points over the best eight-feature logistic model we know how to build.
- **L3-revised lands inside the 0.83–0.86 literature band's lower edge** on its own (CI upper bound 0.846). Eight well-chosen features and a linear model are competitive with published GBDT results on this dataset that trained on 4,000 rows.
- **The project's argument gets stronger, not weaker.** "L4 beats a mis-specified L3 by 0.14" invites the reviewer to fix L3 for you. "L4 beats a properly screened L3 by 0.06, still significant" survives that challenge.
- **AUPRC tells the same story**: 0.395 → 0.500 → 0.591.

## 5. Should the group adopt L3-revised as the official L3?

Both rungs are reported; the choice is which one the main table leads with. Recommendation: **lead with L3-revised, keep L3 original in the table as "L3 (clinical-prior statistics)"** with one sentence explaining the swap. Reasons:

1. It reflects the method the group would actually defend: clinical rationale to pick parameters, a train+validation screen to pick statistics.
2. Dropping the original hides a genuine finding about clinical priors versus data signal, which is discussion material.
3. It costs one pre-declared test look, which is exactly what 实验方案 §4.3 permits when the model is defined before scoring.

What it does **not** license: any further tweaking of L3-revised. The script is frozen. If someone wants a third variant, it needs its own declaration and its own single test look, and the group should ask whether the ladder needs it.

## 6. Self-critique

1. **The screen in §2.1 should have been run in stage 2.** It is free, it touches no test data, and stage 1 had already flagged `gcs min` as weak. "Clinical meaning over statistical screening" was the instruction, but the two are not in conflict: use clinical meaning to choose parameters, use the screen to choose the summary. That is the method rule this stage adds.
2. **Creatinine and platelets still have the wrong sign.** Small coefficients, likely collinearity with BUN and low signal respectively. Not fixed here because fixing them would be a third variant; noted for the report's limitations.
3. **The urine swap (`min` → `total`) is a smaller effect (+0.021 on validation)** and was bundled with the GCS swap because both were identified by the same screen. If a reviewer asks, the GCS swap alone (variant B) accounts for 0.077 of the 0.086.
4. **Stage 3 v1 called the original choice a "failure" and quoted "~20 AUROC points."** Both wrong. The choice followed instructions and the figure was a single-variable, test-set number. Corrected in stage 3 v2 §5 and here.

## 7. Files added in this revision

```
baselines/run_L3_revised_test_evaluation.py     # declared feature set, test scored once
results/L3_revised_test_evaluation.json         # results, coefficients, comparisons
```

Reused from stage 3 v2 (train+validation only): `baselines/run_L3_gcs_statistic_ablation.py`, `results/L3_gcs_statistic_ablation.json`.

Unchanged: all stage-2 v1 files, including `results/L3_logistic_regression_8features.json` except for the CI correction noted in the lineage table.

### Reproduce

```bash
cd experiment/
python baselines/run_L3_gcs_statistic_ablation.py        # §2.2, validation only
python baselines/run_L3_revised_test_evaluation.py        # §3, deterministic; re-running reproduces the same test numbers
```
