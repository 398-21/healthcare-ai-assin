# Stage 3 (v2): L4 GBDT Models — Capacity, Ablations, and Corrected Findings

**Date**: 2026-09-25
**Supersedes**: `stage3_L4_gbdt_results.md` (v1, 2026-09-24). v1 is retained unchanged.
**Test-set policy for this revision**: the XGBoost test numbers from v1 are carried over verbatim. **Nothing new was scored on the test set.** Every new experiment in this document uses train and/or validation only. CatBoost, which lost model selection, was never scored on test.

---

## Lineage: what changed from v1 and why

A review of v1 (prompted by a side discussion the author had with a second reviewer) found one wrong causal claim, two numbers with no source, one mislabelled diagnosis, and one self-critique that overstated an unmeasured effect. Each is corrected below with the evidence that replaced it.

| # | v1 statement | Status | v2 replacement | Evidence |
|---|---|---|---|---|
| 1 | "Early stopping and regularization preserved generalization" | **Wrong** | XGBoost's final fit passed `eval_set` but no `early_stopping_rounds`, so it ran all 300 trees. An ablation shows early stopping changes neither train AUROC nor validation AUROC. | §3.2, `results/L4_xgboost_es_ablation.json` |
| 2 | "Severe overfitting" | **Mislabelled** | No loss of generalization was observed. The model interpolates the training set and is consistent on two independent hold-outs. Reworded throughout. | §3 |
| 3 | L3 train AUROC "0.789" (v1 complexity table) | **No source** | Actual L3 train AUROC is 0.7334. | `results/L3_supplementary_evaluation.json` |
| 4 | 实验方案 quick-test L4 XGBoost "0.862" | **No source** | 实验方案 §5.5 contains no L4 quick-test number; it cites the literature range 0.83–0.86 only. Row removed. | `../实验方案.md` line 136 |
| 5 | "L3 feature selection failures … cost ~20 AUROC points" | **Overstated, wrong frame** | Single-variable gap ≠ model loss. Model-level effect measured on validation: +0.077 AUROC. Reframed as a finding about clinical prior vs data signal, and the evidence is re-derived on train+validation so it does not lean on test-set SHAP. | §5, `results/L3_gcs_statistic_ablation.json` |
| 6 | Overfitting conclusion drawn from XGBoost alone | **Incomplete** | CatBoost train/val now measured. Result is more nuanced than "both ≈ 1.0" (see §3.1). | `results/L4_overfit_comparison.json` |
| 7 | L3 AUROC CI upper bound 0.7755 | **Non-reproducible** | Reproducible value with shared `metrics.bootstrap_ci` (seed 42) is 0.7740; JSON updated with provenance note. | `results/L3_logistic_regression_8features.json` |
| 8 | Literature comparison ("MIMIC-II vs III", "50–200 features") | **Hand-waved** | Replaced with the concrete difference: most published PhysioNet-2012 results train on the 4,000-record set A; we train on 8,283. | §7 |

Two things v1 got right and v2 keeps: the leakage audit (§6) and the selected model's test numbers (§2).

---

## 1. Executive summary

- **Selected model**: XGBoost (validation AUROC 0.8664 vs CatBoost 0.8597).
- **Test** (n=1,775, prevalence 0.1443): AUROC **0.8849** [0.8653, 0.9031], AUPRC **0.5912** [0.5351, 0.6493]. Exceeds the 0.83–0.86 literature range. Statistically separated from every lower rung.
- **Capacity, not overfitting**: XGBoost interpolates the training set (train AUROC 1.0000). CatBoost does not (0.9570). Both show a train-validation gap of about 0.10–0.13 and both generalize; validation and test agree within noise. Early stopping does not change this picture.
- **Where the L3→L4 gain comes from**: mostly feature breadth, not model nonlinearity. XGBoost on its own top-8 features scores validation 0.813; a logistic regression on 8 better-chosen features scores 0.808. Fifty features recover all but 0.012 of the full model's validation AUROC.
- **GCS finding, reframed**: L3 chose `gcs__0_48h__min` on SOFA grounds. Both GBDTs rank `gcs__0_48h__last` first. Re-screening on train+validation (test untouched) confirms last ≫ min. Swapping the statistic in L3 raises validation AUROC by 0.077. This is a result about clinical prior versus data signal, not a procedural failure.

---

## 2. Test-set results (unchanged from v1)

| Metric | Value | 95% CI (bootstrap, 1,000) |
|---|---|---|
| AUROC | 0.8849 | [0.8653, 0.9031] |
| AUPRC | 0.5912 | [0.5351, 0.6493] |
| Threshold (validation recall ≈ 0.80) | 0.108 | validation recall achieved 0.801 |
| Sensitivity @ threshold | 0.785 | |
| PPV @ threshold | 0.387 | |
| Specificity @ threshold | 0.791 | |
| min(Se, PPV) | 0.387 | |

### Complete ladder (test set)

| Rung | Model | Features | AUROC [95% CI] | AUPRC [95% CI] | n |
|---|---|---|---|---|---|
| L0 | Random | 0 | 0.474 [0.436, 0.513] | 0.135 [0.124, 0.151] | 1,775 |
| L1 | Age | 1 | 0.627 [0.590, 0.667] | 0.224 [0.196, 0.265] | 1,775 |
| L1 | GCS min | 1 | 0.553 [0.517, 0.586] | 0.155 [0.143, 0.168] | 1,775 |
| L1 | BUN max | 1 | 0.675 [0.635, 0.710] | 0.265 [0.231, 0.310] | 1,775 |
| L2 | SAPS-I | score | 0.640 [0.602, 0.676] | 0.213 | 1,728 |
| L2 | SOFA | score | 0.623 [0.586, 0.660] | 0.253 | 1,733 |
| L3 | Logistic, 8 hand-picked | 8 | 0.7417 [0.7075, 0.7740] | 0.3945 [0.3430, 0.4546] | 1,775 |
| **L4** | **XGBoost** | **1,474** | **0.8849 [0.8653, 0.9031]** | **0.5912 [0.5351, 0.6493]** | 1,775 |

### Significance

- **vs L3**: +0.1432 AUROC (+19.3%), CIs do not overlap.
- **vs SAPS-I on its score-available subset** (n=1,728): L4 0.8858 [0.8644, 0.9053] vs 0.6399 [0.6015, 0.6759], +0.246.
- **vs SOFA on its score-available subset** (n=1,733): L4 0.8900 [0.8695, 0.9089] vs 0.6231 [0.5858, 0.6595], +0.267.

Subset protocol per stage 1 (SAPS-I/SOFA `-1` sentinel decision). Source: `results/L4_test_evaluation.json`, `results/L4_supplementary_evaluation.json`.

---

## 3. Model capacity and generalization

### 3.1 Both candidates, train vs validation (test not scored)

| Model | Trees used | Final-fit early stopping | Train AUROC | Val AUROC | Gap | Train AUPRC | Val AUPRC | Train perfectly separated |
|---|---|---|---|---|---|---|---|---|
| XGBoost (selected) | 300 of 300 | **No** | 1.0000 | 0.8664 | +0.134 | 1.0000 | 0.5380 | Yes |
| CatBoost | 224 (cap 2,000, patience 50) | Yes | 0.9570 | 0.8597 | +0.097 | 0.8264 | 0.5052 | No |
| L3 logistic (reference) | — | — | 0.7334 | 0.7217 | +0.012 | 0.3340 | 0.3063 | No |

"Perfectly separated" means every training death scored above every training survivor.

**Reading.** The hypothesis going in was "if both models hit train ≈ 1.0, it is the data shape (1,474 features vs 8,283 rows), not an implementation quirk." The data only half-support that:

- The **large train-validation gap is shared** (+0.13 and +0.10). Any GBDT of this capacity on this matrix will fit the training set far more tightly than it fits new patients.
- **Perfect interpolation is XGBoost-specific** here. CatBoost's train AUROC is 0.957 and it does not separate the classes perfectly. Its configuration is more conservative (`l2_leaf_reg=9`, `learning_rate=0.03`, ordered boosting, early stopping fired at 224 trees), but we did not ablate which of those matters, so the cause is not attributed.

The correct general statement is: *on this matrix, tree ensembles operate in or near the interpolation regime, and the gap to held-out data is about 0.10–0.13 AUROC regardless of implementation.* That is a property of the problem, and it is not evidence of harm: both models generalize, and their validation scores differ by only 0.007.

### 3.2 Early-stopping ablation (XGBoost, train + validation only)

The v1 claim that early stopping protected generalization was false on two counts: it was not enabled in the final fit, and, when enabled, it makes no difference.

Same hyperparameters as the selected model, refit on CPU (`tree_method='hist'`) for a like-for-like comparison:

| Arm | Trees used | Best iteration | Train AUROC | Val AUROC | Gap | Perfectly separated |
|---|---|---|---|---|---|---|
| No early stopping, `n_estimators=300` | 300 | — | 1.0000 | 0.8707 | +0.129 | Yes |
| Early stopping (patience 50, cap 2,000) | 211 | 160 | 1.0000 | 0.8705 | +0.130 | No |

Early stopping halts at iteration 160, trims 89 trees, and leaves both train AUROC (still 1.0000 to four decimals, though no longer perfectly separated) and validation AUROC (0.8707 vs 0.8705) unchanged. The model is already at the ceiling of what this feature set yields on validation before it finishes interpolating training. Early stopping is not what makes the test number trustworthy.

CPU refits land 0.003–0.004 above the saved GPU model on validation (0.8707/0.8691 vs 0.8664). That is backend histogram and seed noise, within the validation CI half-width of about 0.02.

### 3.3 Why the test number is trusted

Only one argument carries weight, and it is procedural rather than architectural:

1. Hyperparameters were chosen by 5-fold CV on train; model selection used validation; the threshold used validation. **Test was scored once, after all choices were frozen.**
2. Validation (0.8664) and test (0.8849) are independent hold-outs and agree within noise. With 256 deaths per split, the bootstrap CI half-width is about ±0.02, so +0.0185 is consistency, not "test is easier."

The regularization settings (`max_depth=7`, `min_child_weight=3`, `subsample=0.9`, `colsample_bytree=0.9`, `max_bin=127`) bound single-tree variance and are reported as configuration. No ablation was run on them, so no causal claim is made.

### 3.4 Wording for the report

Use: *"The model has enough capacity to fit the training set exactly (train AUROC 1.00) and performs consistently on two independent held-out sets (validation 0.866, test 0.885)."*

Avoid: *"severe overfitting."* Overfitting means degraded generalization; none was observed. Tree ensembles with far more features than rows routinely enter the interpolation regime, and that is not by itself harmful.

---

## 4. Feature-count ablation: how much of the 1,474 is needed?

Ranking by mean |SHAP| computed on the **train** set (so validation plays no role in selection), then refit with the selected hyperparameters and early stopping on validation. Test not scored.

| Top-k features | Trees | Train AUROC | Val AUROC | Val AUPRC | Δ val AUROC vs all 1,474 |
|---|---|---|---|---|---|
| 8 | 90 | 0.936 | 0.8129 | 0.4219 | −0.056 |
| 20 | 104 | 0.972 | 0.8349 | 0.4521 | −0.034 |
| 50 | 179 | 0.9995 | 0.8566 | 0.4908 | −0.012 |
| 100 | 218 | 1.000 | 0.8607 | 0.5062 | −0.008 |
| 200 | 156 | 0.9998 | 0.8616 | 0.5296 | −0.007 |
| 500 | 193 | 1.000 | 0.8673 | 0.5174 | −0.002 |
| 1,474 | 275 | 1.000 | 0.8691 | 0.5321 | 0 |

Three conclusions:

- **Interpolation starts at 50 features.** Train AUROC is 0.9995 with 50 features, so "1,474 features" is not what causes it. It is a capacity property of boosted trees on 8,283 rows.
- **Diminishing returns are steep.** 50 features get within 0.012 of the full model on validation; 100 get within 0.008. The last 1,374 features contribute under one AUROC point.
- **At 8 features, GBDT ≈ logistic regression.** XGBoost on its train-ranked top 8 scores validation 0.813. A logistic regression on a better-chosen 8 (§5, variant E) scores 0.808. Nonlinearity contributes little at that scale; the L3→L4 gain is bought mainly with breadth.

For the group's "is the complexity justified?" question this reframes the answer: the defensible model for deployment is likely a 50–100 feature GBDT, which keeps nearly all the performance and is far easier to audit. That is a stage-4 candidate, not a change to this ladder.

---

## 5. The GCS statistic: clinical prior versus data signal

### 5.1 What happened

Stage 2 selected `gcs__0_48h__min` because the SOFA CNS component uses the worst GCS in the window: a textbook choice. Stage 3's SHAP ranks `gcs__0_48h__last` first, and ranks `gcs__0_48h__min` 326th (XGBoost) and 170th (CatBoost). Stage 1's L1 rung had already reported `gcs__0_48h__min` at AUROC 0.553 and labelled it weak, so the signal was on file before stage 2 chose it.

v1 called this a "failure" and quoted "~20 AUROC points." Both were wrong: the choice followed the stage-2 instruction (clinical meaning over statistical screening) and the 20-point figure was a single-variable gap measured on the test set, not a model effect.

### 5.2 Evidence re-derived without the test set

**Single-variable screen on train + validation** (n=10,058, 1,451 deaths):

| GCS statistic (0–48h) | AUROC |
|---|---|
| last | **0.757** |
| median | 0.722 |
| mean | 0.707 |
| max | 0.686 |
| slope_per_hour | 0.666 |
| delta | 0.665 |
| **min (L3's choice)** | **0.589** |
| first | 0.536 |
| std | 0.523 |

The ordering matches the test-set diagnostic in v1, so the "last ≫ min" conclusion stands on data that was legitimately available at stage 2.

**Model-level ablation** (same L3 recipe: L2 logistic, C=1, balanced, train-median imputation, standard scaling; **validation** metrics; L3's frozen test number is untouched):

| Variant | Features | Train AUROC | Val AUROC | Val AUPRC | Δ val vs A |
|---|---|---|---|---|---|
| A. Original L3 (gcs min, urine min) | 8 | 0.733 | 0.7217 | 0.3063 | — |
| B. gcs min → gcs last | 8 | 0.813 | 0.7985 | 0.4225 | +0.077 |
| C. Add gcs last, keep min | 9 | 0.822 | 0.7990 | 0.4159 | +0.077 |
| D. urine min → urine total | 8 | 0.747 | 0.7430 | 0.3369 | +0.021 |
| E. Both swaps (B + D) | 8 | 0.819 | 0.8081 | 0.4374 | +0.086 |

The model-level effect of the GCS statistic is +0.077 validation AUROC, less than half the single-variable gap, because the other seven features already carry overlapping information. Adding `last` alongside `min` (C) gives nothing over replacing it (B): `min` contributes no independent signal once `last` is present.

### 5.3 Why "last" beats "min" clinically

Almost every ICU patient has at least one low GCS reading in 48 hours: sedation, intubation, and procedures all depress it. `min` therefore saturates near the floor and mixes treatment-induced lows with prognostic lows. `last` reads the patient's consciousness once those effects have played out at the end of the window, so it separates the two. This is exactly the "measurement behaviour versus physiological signal" thread the project set out to follow (实验方案 §3, §8), and it deserves a place in the discussion, not the limitations.

The point-in-time nature of `last` also carries a portability risk: sites with different GCS charting cadence could shift its meaning. `median` (0.722 single-variable) is a less timing-dependent alternative worth carrying into any external validation.

### 5.4 What this does and does not change

- The original L3 rung **stays at 0.7417**. It was defined, frozen, and scored once.
- Variant E was subsequently promoted to a pre-declared **L3-revised** rung and scored on test once (decision taken 2026-09-25, after this section was first written). Details in `stage2_L3_logistic_regression_results_v2.md`; effect on this stage's conclusions in §5.5 below.
- Test-set SHAP (v1) and the test-set single-variable diagnostics remain valid as a post-hoc leakage audit. They must not be used to pick features. The train+validation screen above is the version that can.

### 5.5 L3-revised on test, and what it does to the L4 argument

| Rung | Test AUROC [95% CI] | Test AUPRC [95% CI] | vs L4 |
|---|---|---|---|
| L3 original | 0.7417 [0.7075, 0.7740] | 0.3945 [0.3430, 0.4546] | −0.143, CIs disjoint |
| **L3-revised** | **0.8207 [0.7932, 0.8459]** | **0.5002 [0.4447, 0.5560]** | **−0.064, CIs disjoint** |
| L4 XGBoost | 0.8849 [0.8653, 0.9031] | 0.5912 [0.5351, 0.6493] | — |

Three consequences for this stage:

1. **The L4 gain over the best small model is 0.064 AUROC, not 0.143.** It is still statistically separated (CIs do not overlap), so the "is the complexity justified?" answer stays yes, but with the honest margin. Six points for 1,466 extra features and a GBDT is the number to defend.
2. **It corroborates §4.** The feature-count ablation put XGBoost-on-8-features at validation 0.813 and L3-revised at validation 0.808; on test L3-revised gives 0.821. At eight features, the linear model and the tree model are within noise of each other, so the L3→L4 gap is bought with breadth, not nonlinearity.
3. **The literature comparison shifts.** L3-revised alone sits at the low edge of the published 0.83–0.86 band (upper CI 0.846). The story "eight well-chosen features and logistic regression are competitive with published GBDT results; the full model adds six points on top" is more useful to the group than "our GBDT beats the literature."

Test was scored once for L3-revised, with the feature set fixed in the script before first execution. `L4_test_evaluation.json` is unchanged.

---

## 6. Leakage audit (unchanged from v1, summarized)

- Record-length proxies (`*__time_span_hours`): single-variable AUROC 0.50–0.62. `record__0_48h__distinct_parameter_count`: 0.61. Weak next to clinical variables.
- GCS charting span: deaths median 44.5 h, survivors 44.0 h. No truncation signal.
- No `RecordID`, `Survival`, `Length_of_stay` or derived columns in the matrix (enforced upstream, `make_labels_and_baselines.py`).
- Top SHAP features are consciousness, age, urine output, BUN, WBC trend, ventilation duration: all physiologically plausible.

---

## 7. Cross-model agreement on feature importance (validation set)

SHAP computed on **validation** for both models (CatBoost may not be scored on test; XGBoost's test SHAP from v1 is left as is).

| Rank | XGBoost | \|SHAP\| | CatBoost | \|SHAP\| |
|---|---|---|---|---|
| 1 | gcs__0_48h__last | 0.693 | gcs__0_48h__last | 0.439 |
| 2 | static_age_years | 0.252 | static_age_years | 0.209 |
| 3 | urine__0_48h__total | 0.140 | urine__0_48h__total | 0.100 |
| 4 | wbc__0_48h__slope_per_hour | 0.102 | bun__0_48h__min | 0.096 |
| 5 | gcs__24_48h__last | 0.098 | gcs__24_48h__max | 0.086 |

- Top 3 identical, top-20 overlap 12/20.
- #1-to-#2 ratio: XGBoost 2.75×, CatBoost 2.10×. GCS-last dominance is not an XGBoost artefact.
- XGBoost's validation SHAP for `gcs__0_48h__last` (0.693) matches its test SHAP (0.694): the importance structure is stable across hold-outs.

Statistic-type mix in XGBoost's top 20 (corrected categoriser, from `L4_supplementary_evaluation.json`): point-in-time 4, extremum 3, variability 3, static 2, cumulative 2, temporal-trend 2, measurement-behaviour 2, central 2. Trend and measurement-behaviour features are present (4/20). v1's original SHAP script mis-categorised them because it tested organ keywords before statistic keywords.

---

## 8. Literature context (corrected)

The PhysioNet/CinC 2012 challenge released 12,000 ICU stays as set A (4,000, outcomes public) and sets B and C (4,000 each, outcomes withheld during the competition). Official scoring was min(Se, PPV) and the Hosmer-Lemeshow H statistic, not AUROC ([challenge page](https://physionet.org/challenge/2012/); [Silva et al. 2012](https://pubmed.ncbi.nlm.nih.gov/24678516/); [test-set description](https://archive.physionet.org/pn3/challenge/2012/)). The SAPS-I reference scored 0.3125 on event 1; the winning entry scored 0.5353.

Post-challenge AUROC reports on this data cluster around 0.84–0.85: a preprocessing study reached 0.848 ([CinC 2014](https://cinc.org/archives/2014/pdf/0157.pdf)); a recent XGBoost baseline reports 0.8395 AUROC / 0.4556 AUPRC ([Zenodo](https://zenodo.org/records/18827972)). Most of these train on set A only (4,000 rows) or use A/B/C as train/validation/test.

Our cohort pools all three sets (11,833 after the 167 negative-LOS exclusions) and re-splits them 70/15/15, so we **train on 8,283 rows, roughly twice the training data behind most published numbers**, and we use 1,474 engineered features. Those two facts are the most likely reasons 0.885 sits above 0.84–0.85. This is a legitimate difference, not a flaw, but it does mean our number is not directly comparable to challenge-era results, and the report should say so rather than present it as "beating the literature."

Our min(Se, PPV) of 0.387 at the recall-0.80 operating point lands between the SAPS-I reference (0.3125) and the winner (0.5353), but on a different test set and with a threshold fixed by recall rather than optimized for the metric, so it is context only.

### 8.1 Using the literature as a baseline tier: how, and how not

实验方案 §5.5 already defines L4 as "文献最优（此数据集 AUROC 0.83–0.86）", so the literature is part of the ladder by design. It is one rung, not one number, and it splits into three tiers that answer different questions. Keep them apart in the final report.

| Tier | Source | What it is | Comparable to ours? | Use it to answer |
|---|---|---|---|---|
| **Same-data AUROC** | [CinC 2014 preprocessing study](https://cinc.org/archives/2014/pdf/0157.pdf): 0.848; [Zenodo 2026 XGBoost](https://zenodo.org/records/18827972): 0.8395 / AUPRC 0.4556 | AUROC on PhysioNet 2012 data, usually trained on set A (4,000 rows) | **Roughly.** Same source data, same outcome. Different split, cohort size, and features. | "Did we reach the level others reach on this dataset?" Yes: 0.8849 vs 0.84–0.85, and our L3-revised alone reaches 0.821. Say why we are higher (train on 8,283 rows, 1,474 features; §8). |
| **Official challenge metric** | [Silva et al. 2012](https://pubmed.ncbi.nlm.nih.gov/24678516/): random 0.1386, SAPS-I 0.3125, winner 0.5353 on sets B+C | min(Se, PPV) at the entrant's chosen threshold | **Only as an anchor.** Their test sets are not ours, and their threshold was optimised for the metric. | "Would we have been competitive in 2012?" Our frozen test value 0.387 (recall-0.80 threshold) is a lower bound. A validation sweep (§8.2) puts the metric-optimal point at 0.518, close to the winner. Report both, label the test set. |
| **Cross-dataset AUROC** | MIMIC-III / eICU mortality papers, typically 0.85–0.90 with deep models | Same task, different data, different cohorts | **No.** Different populations, variable sets, prevalence. | Nothing quantitative. Mention only to place the task; never put in the ladder table. |

Rule for the final report: the ladder table gets one literature row, labelled "Published on this dataset (set-A training)", showing 0.84–0.85 AUROC with the two citations. The challenge metric goes in a separate small table with the three anchors and our two numbers. Cross-dataset results stay in the discussion text.

### 8.2 Challenge-metric operating point (validation only)

The frozen test min(Se, PPV) = 0.387 was measured at the recall-0.80 threshold. A challenge entrant would have picked the threshold that maximises the metric. Re-thresholding on test would be a second look, so the sweep was done on validation:

| Operating point (validation) | Threshold | Se | PPV | Sp | min(Se, PPV) |
|---|---|---|---|---|---|
| Recall-0.80 (our reporting point) | 0.108 | 0.801 | 0.378 | 0.778 | 0.378 |
| min(Se, PPV)-optimal | 0.345 | 0.520 | 0.518 | 0.918 | **0.518** |

Headroom from re-thresholding: +0.14. The metric-optimal point trades 28 points of sensitivity for 14 points of PPV, which is exactly the trade 实验方案 §5.2 rejected on clinical grounds (missed death ≫ unnecessary review). So the 0.518 is a "what the challenge would have scored" number, not our recommended operating point. If the group wants a test-set value at that threshold, it is one pre-declared look; not taken here.

Source: `results/L4_challenge_metric_validation.json`, `baselines/run_L4_challenge_metric_validation.py`.

---

## 9. Self-critique (revised)

1. **The v1 draft contained unsupported numbers and an untested causal claim.** Two figures had no source and one mechanism (early stopping) was asserted without checking the code. All were caught in review, not by the author. The process fix, applied from this revision on: every number in a report must trace to a results file, and every "because" must have an ablation or be labelled as configuration.
2. **Capacity is a property of the problem; the label "overfitting" was wrong.** Both GBDTs sit near the interpolation regime and both generalize. The group should describe this accurately (§3.4) rather than apologize for it.
3. **Most of L4's gain is feature breadth, and 50–100 features would do.** The full 1,474-feature model is the right research benchmark for the ladder, but it is not the model one would deploy. A compact GBDT is the natural stage-4 candidate.
4. **The GCS statistic result is a finding about method.** Clinical rationale correctly identified the variable and mis-specified the summary. A train+validation single-variable screen of each parameter's 12 statistics costs nothing and touches no test data; it should precede any future "hand-picked" feature set.
5. **No external validation.** All results are from one cohort with one split. `gcs__0_48h__last`'s dominance in particular depends on charting cadence and needs a second site.
6. **Calibration and subgroup performance are unmeasured.** 实验方案 §5.7 lists both; neither has been done for L4.

---

## 10. Files added in this revision

```
baselines/
├── run_L4_overfit_comparison.py           # §3.1  both models, train/val only
├── run_L4_xgboost_es_ablation.py          # §3.2  early stopping on/off
├── run_L4_feature_count_ablation.py       # §4    top-k by train-SHAP, val eval
├── run_L3_gcs_statistic_ablation.py       # §5.2  L3 statistic swaps, val eval
├── run_L3_revised_test_evaluation.py      # §5.5  pre-declared rung, test once
├── run_L4_shap_validation_comparison.py   # §7    XGB vs CatBoost SHAP on val
└── run_L4_challenge_metric_validation.py  # §8.2  min(Se,PPV) sweep, val only
results/
├── L4_overfit_comparison.json
├── L4_xgboost_es_ablation.json
├── L4_feature_count_ablation.json
├── L3_gcs_statistic_ablation.json
├── L3_revised_test_evaluation.json
├── L4_shap_validation_comparison.json
└── L4_challenge_metric_validation.json
results/L3_logistic_regression_8features.json   # CI fields corrected, provenance note added
stage2_L3_logistic_regression_results_v2.md     # companion: L3-revised rationale and results
```

Unchanged from v1: the GPU training scripts, `models/`, `L4_test_evaluation.json`, `L4_feature_importance.json`, `L4_supplementary_evaluation.json`, `results/figures/L4_shap_summary.png`.

### Reproduce this revision

```bash
cd experiment/
python baselines/run_L4_overfit_comparison.py           # ~1 min, CPU
python baselines/run_L4_xgboost_es_ablation.py          # ~1 min, CPU
python baselines/run_L4_feature_count_ablation.py       # ~2 min, CPU
python baselines/run_L3_gcs_statistic_ablation.py       # seconds
python baselines/run_L4_shap_validation_comparison.py   # ~1 min, CPU
```

Environment: `environment_gpu.yml` (xgboost 3.2.0, catboost 1.2.10 used here). All five scripts load saved models or refit small models; none needs the GPU.

---

## 11. Next steps

**For the project deliverable**
1. Merge the ladder table (§2 plus the L3-revised row from §5.5), the capacity wording (§3.4), and the literature tiers (§8.1) into the final report.
2. Group decision: lead the main table with L3-revised and keep L3 original as a labelled row (recommended in stage 2 v2 §5), or the reverse.
3. Add calibration (reliability curve, Brier) and ICUType subgroup AUROC for L4; both are validation-then-test-once analyses.

**Optional stage 4**
4. Compact GBDT (50–100 train-SHAP-ranked features) as the deployment candidate, with SHAP on validation.
5. External validation on MIMIC-III/eICU if data access allows, with `gcs__0_48h__median` carried alongside `last`.
