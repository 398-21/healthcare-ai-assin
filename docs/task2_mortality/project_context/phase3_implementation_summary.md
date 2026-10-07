# Phase 3 Implementation Summary
## In Hospital Death Prediction - Priority 2 Warnings Resolved

**Date**: 2026-09-21  
**Scope**: Phase 3 (Priority 2) of `preprocessing_fix_plan.md`  
**Status**: ✅ **COMPLETE AND VERIFIED** — 18/18 unit tests, 66/66 output checks, independent verifier PASS

---

## Document Lineage

Fifth document in the preprocessing chain. Every earlier document is preserved unchanged.

| Order | Document | Date | Role |
|---|---|---|---|
| 1 | `preprocessing_validation_report.md` | 09-20 | Independent validation: 4 critical issues, 5 warnings |
| 2 | `preprocessing_fix_plan.md` | 09-21 | Phased plan (Phase 1+2 = Priority 1, Phase 3 = Priority 2) |
| 3 | `phase1_2_implementation_summary.md` | 09-21 | First pass; pH still failing |
| 4 | `phase1_2_implementation_summary_v2.md` | 09-21 | pH resolved; all 4 critical fixes verified |
| 5 | `phase3_implementation_summary.md` (this file) | 09-21 | Priority 2 warnings W1–W5 resolved |

This document closes the fix plan. Nothing in it changes any Phase 1 or 2 conclusion; those fixes were re-verified as regression checks and all still hold.

---

## 1. Executive Summary

Phase 3 addressed the five Priority 2 warnings. Three were implemented as code, one was **deliberately not implemented** with a documented rationale, and one turned out to need only documentation.

Feature count went from **1,361 to 1,474** (+113). All additions are **additive**: not one existing column was renamed, altered, or removed, so any model result from the Phase 1+2 matrix remains comparable and an ablation study is possible.

| ID | Warning | Decision | Outcome |
|---|---|---|---|
| W1 | Invasive/non-invasive BP not merged | **Implemented** | 3 merged streams + arterial-line indicator; coverage 70.0% → 98.6% |
| W2 | Weight time-zero duplication | **Deliberately skipped** | The overlap is a useful alias, not a defect. Rationale below and in code |
| W3 | Height < 100 cm not flagged | Already fixed in Phase 2 | Height range now 121.9–210.8 cm |
| W4 | "trend" vs `slope_per_hour` naming | **Documentation only** | `delta` already *was* the required trend; now stated in the feature dictionary |
| W5 | BMI not computed | **Implemented** | `static_bmi` + missing indicator, 52.2% coverage |

---

## 2. W1 — Merged Blood Pressure (implemented)

### Why it mattered

Measured on the current cohort, the three-way split is real and large:

| Group (0–48h) | Patients | Share |
|---|---|---|
| Both invasive and non-invasive | 6,949 | 58.7% |
| Non-invasive only | 3,384 | 28.6% |
| Invasive only | 1,329 | 11.2% |
| Neither | 171 | 1.4% |

With the six source streams modelled separately, the 3,384 cuff-only patients have all 108 invasive columns empty. Tree models tolerate that, but median imputation hands those patients a fabricated arterial pressure. That is the concrete harm the merge removes.

### What was built

Three merged streams (`bp_sys`, `bp_dias`, `bp_mean`) × 3 windows × 12 statistics = 108 features, plus `bp__<window>__has_arterial_line` × 3 = **111 features**.

The preference is decided **per window**, not per patient. A patient whose arterial line is pulled at hour 30 uses invasive data in 0–24h and cuff data in 24–48h, which is the clinically correct reading of their record.

### Verified effect on coverage

| Window | Invasive only | Non-invasive only | **Merged** |
|---|---|---|---|
| 0–24h | 66.3% | 78.2% | **98.5%** |
| 24–48h | 65.0% | 67.0% | **98.3%** |
| 0–48h | 70.0% | 87.3% | **98.6%** |

### Design decisions worth defending in the report

- **Source columns retained unchanged.** Merged streams sit alongside the originals, which makes the merge an ablation-testable hypothesis rather than an irreversible transformation.
- **`has_arterial_line` is 1 if *any* invasive stream is present in the window**, not just SysABP. The three invasive streams are usually co-recorded but not always (0–24h: SysABP 7,848, DiasABP 7,847, MAP 7,850). Whether a patient has an arterial line is itself a severity signal, per 实验方案 §3.6.
- **Derived streams are excluded from record-level counters.** `record__<window>__distinct_parameter_count` and `missing_parameter_fraction` still describe only the 37 raw dynamic parameters, so those columns stay comparable with the Phase 1+2 matrix. Verified: max distinct count is 35, still ≤ 37.

Note that `sysabp__<window>__measured` already encoded arterial-line presence implicitly. The new indicator makes it explicit and window-correct; the larger win from W1 is the coverage jump and the removal of fabricated imputed pressures.

---

## 3. W5 — BMI (implemented)

`static_bmi = static_admission_weight_kg / (static_height_cm / 100)²`, plus `static_bmi_missing`. Two features.

- **Coverage 6,181 patients (52.2%)**, bounded by Height, which is present for only 52.3%. Weight is present for 91.8%.
- **Range 10.0 to 99.6 kg/m²**, all physiologically attainable in an ICU population after Phase 2 range validation.
- NaN whenever either input is missing, and guarded against non-positive height so no division-by-zero can produce an infinity.

Rationale: gradient-boosted trees approximate ratios poorly from their components, and BMI is an established ICU mortality covariate. Cost was minutes.

---

## 4. W2 — Weight time-zero dedup (deliberately NOT implemented)

The fix plan proposed removing the 00:00 Weight descriptor from the dynamic Weight series because it also appears in `static_admission_weight_kg`. **I did not do this, and recommend against it.**

The overlap is an **alias, not a double measurement**. Removing the 00:00 anchor would strip the admission baseline out of `weight__*__delta` and `weight__*__slope_per_hour`. Weight change across 48 hours is a fluid-balance signal with genuine prognostic meaning in critical care, and it is only interpretable relative to admission weight. The "fix" would destroy information to remove a cosmetic redundancy.

This is recorded in three places so a reader cannot miss it: the feature dictionary note on `static_admission_weight_kg`, the README boundary section, and here.

If the team disagrees, the change is roughly ten minutes of work. It should be a conscious group decision, not a silent cleanup.

---

## 5. W4 — Trend naming (documentation only)

实验方案 §4.1 asks for a "(末值−首值) trend". The pipeline already computed exactly that, under the name `delta`; `slope_per_hour` is a separate, more sophisticated least-squares trend. Nothing was missing. The feature dictionary note now states this explicitly, so a reader checking the plan against the schema will not conclude the statistic is absent.

---

## 6. Repairs found while working (not in the original plan)

Three scripts had **hardcoded constants that Phase 1 had already invalidated**. These were latent failures, not caused by Phase 3, and would have produced either a crash or a false-confidence report the first time anyone ran them.

| File | Was | Problem |
|---|---|---|
| `verify_outputs.py` | `== 12000`, `{8400, 1800, 1800}`, `== 1361`, `== 235` | Would have **failed outright** on the Phase 1 cohort |
| `build_preprocessing_report_artifact.py` | QA evidence strings `"12,000 = 12,000"`, `"8,400 / 1,800 / 1,800"`, `"1,707 death"` | Would have printed **stale numbers next to PASS marks**, the worst kind of wrong |
| `build_mortality_notebook.py` | `== 12_000`, `== 1_361` in emitted cells | Generated notebook would have failed its own asserts |

All now derive their figures from `preprocessing_summary.json` at runtime. Two narrative strings in the artifact builder also asserted things that had become false — that all 12,000 stays were retained, and that physiological range rules were still future work — and were corrected.

`README.md` line 44 stated *"没有凭经验裁剪生理极值"*, the direct opposite of what Phase 2 implemented. Rewritten to document the real policy, including that thresholds come from clinical knowledge rather than training-set fitting and therefore carry no leakage risk.

**My first sweep was incomplete, and the notebook proved it.** The initial `grep` used comma-formatted and plain patterns, which missed Python underscore literals. Two further sites survived in the notebook builder's QA cells: `== 235` (boundary rows, now 232) and `{"train": 8_400, "validation": 1_800, "test": 1_800}`. The first notebook regeneration therefore **failed** with an `AssertionError` in cell 10. Both were patched and the notebook then regenerated cleanly at 16/16 PASS.

Worth recording as a process lesson: the failure was caught only because I actually executed the notebook rather than trusting the grep. A pattern-based sweep is evidence of absence only for the patterns you thought to write.

A final sweep using comma, underscore and bare-digit patterns confirms **no stale cohort or schema constant remains in any `.py` file**.

---

## 7. Verification

### Unit tests: 18/18 pass (7 pre-existing + 11 new)

New file `tests/test_phase3_derived_features.py` covers BMI normal/missing/zero-height cases, invasive preference, non-invasive fallback, **per-window** preference switching, neither-stream-present, source-column retention, counter isolation, and a **schema contract test**.

That last test is the most valuable one. The pipeline does `reindex(columns=expected_columns)`, so if `build_feature_dictionary()` and `aggregate_patient()` ever disagree, undeclared features are **silently dropped** and undelivered ones become **silently all-NaN**. Neither failure trips any downstream check. The test asserts set equality in both directions.

### Output validation: 66/66 pass

New script `validate_phase3_outputs.py`. Re-checks every Phase 1+2 guarantee as a regression suite, then adds Phase 3 checks.

Phase 1+2 regression, all holding:

| Check | Result |
|---|---|
| Cohort | 11,833 (167 negative-LOS excluded) |
| Split stratification | 14.43% / 14.42% / 14.42% |
| MechVent `ever` | {0, 1}, no NaN, all three windows |
| pH | [6.75, 7.72] |
| Temp / K / HR / BUN / DiasABP / NIDiasABP / Na / Glucose | all within clinical bounds |

Phase 3 checks include a recomputation of BMI from its inputs, and a **row-by-row assertion that each merged stream equals invasive-where-available-else-non-invasive**, with 0 mismatches across all three windows. Merged coverage is confirmed ≥ max(source coverages), and no derived column is entirely NaN.

### Independent end-to-end verifier: PASS

`verify_outputs.py` reloads every artifact from disk:

| Field | Value |
|---|---|
| Rows | 11,833 |
| Model features | 1,474 |
| Splits | 8,283 / 1,775 / 1,775 |
| Fill values recomputed from train only | 1,474 / 1,474 |
| Imputed cells re-verified | 17,441,842 |
| Output hashes verified | 12 |
| Imputed outputs all finite | true |
| Feature schema SHA256 | `4f8f9bee69f1a74e94415e724b6d71438ee0d352ffdfb97175b780b7196be514` |

Run: status PASS, 686.1 s, 232 boundary-sensitivity rows.

---

## 8. Reproduction

```bash
cd Group_project/preprocessing
python mortality_preprocess.py              # ~11.5 min, writes outputs/
python -m unittest discover -s tests -v     # 18 tests
python validate_phase3_outputs.py           # 66 checks
python verify_outputs.py                    # independent reload + hashes
```

Deterministic: `random_seed` 20260907, fixed split ratios, no wall-clock or RNG in feature construction. Same inputs reproduce the same schema hash.

---

## 9. Files changed

**Modified** (each with a `.bak_before_phase3` backup):

- `mortality_preprocess.py` — `derive_bmi()`, BMI in static block, merged-BP block in `aggregate_patient()`, dictionary rows for all 113 new features, W2 and W4 notes
- `config.json` — `bp_merge_groups`, `bp_merge_policy`, `derived_static_policy` (additive; original compact formatting preserved)
- `README.md` — corrected range-validation policy, new derived-features section, W2 rationale, W4 clarification
- `verify_outputs.py`, `build_preprocessing_report_artifact.py`, `build_mortality_notebook.py` — stale constants now summary-derived

**Added**: `tests/test_phase3_derived_features.py`, `validate_phase3_outputs.py`

**Regenerated** (also backed up as `*.bak_before_phase3`):

- `outputs/` — all 15 artifacts from the 686 s pipeline run, plus `independent_verification.json` from the verifier
- `mortality_preprocessing_report_artifact.json` — now 1,474 features / 11,833 stays
- `mortality_data_preprocessing.ipynb` — re-executed through a Jupyter kernel, 16/16 QA checks PASS

---

## 10. Known state and caveats

1. **All downstream artifacts are now current.** `mortality_preprocessing_report_artifact.json` and `mortality_data_preprocessing.ipynb` were both regenerated after the builder repairs and now report 1,474 features / 11,833 stays with zero stale constants. The notebook re-executed through a live Jupyter kernel: 12 cells, 0 errors, **16/16 QA checks PASS**. Backups of the previous versions are kept as `*.bak_before_phase3`.

2. **Range thresholds still want a clinical sign-off.** They come from standard reference ranges, not from a clinician on this team. The values are documented in `PHYSIOLOGICAL_RANGES` and the README for review.

3. **Boundary-sensitivity rows moved from 235 to 232.** Three of the originally affected stays were among the 167 negative-LOS exclusions. Expected, not a regression.

4. **The 167 excluded negative-LOS records were all survivors.** Death count is unchanged at exactly 1,707, while the denominator fell from 12,000 to 11,833. Prevalence therefore rose from 1,707/12,000 = 0.1422 to **1,707/11,833 = 0.1443**. Small, but it matters for two reasons: AUPRC must be read against 0.1443, not the 0.142 quoted in 实验方案 §1.2 and §5.5, and any L0 random-baseline figure carried over from the plan's quick tests is now slightly stale. Measured per split: train 0.14427, validation 0.14423, test 0.14423. Worth one line in the report, since a reviewer comparing against the plan will notice the difference.

4. **Two stays have no valid dynamic observations at all.** They are retained with missingness indicators rather than dropped, consistent with the "missingness is informative" principle.

5. **BMI coverage is capped at 52%** by Height availability. `static_bmi_missing` carries that signal, and whether a patient was measured at all may itself be informative.

---

## 11. Next step

The feature matrix is now **frozen at 1,474 features** and should stay frozen through the whole baseline ladder. 实验方案 §4.3 allows the test set to be touched exactly once; changing features mid-ladder would make L0–L4 mutually incomparable and force a full re-run.

Proceed to 实验方案 §5.5: L0 random, L1 single-variable, L2 SAPS-I/SOFA, L3 small logistic regression, L4 GBDT. Primary metric AUPRC against the 14.43% prevalence baseline, AUROC secondary, plus min(Sensitivity, PPV) for PhysioNet comparability. Use the unimputed matrix for tree models and the imputed splits for linear models.

Worth watching: whether the merged BP streams and BMI earn their place in feature importance. If they rank low, that is a reportable finding about this dataset, not a failure of the work.

---

**Document version**: 1.0  
**Session**: fix3 (branched from fix1_2_ph_debugging)
