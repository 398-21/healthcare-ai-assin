> **Historical snapshot - superseded (2026-10-08).** This document records the earlier 11,833-stay analysis that excluded 167 stays with `Length_of_stay = -1`.

---

# Experiment Data — In-hospital Death Prediction

Frozen data snapshot and working directory for the In-hospital Death modelling
experiments (实验方案 task 1).

**Snapshot taken**: 2026-09-21, after preprocessing Phase 1+2+3 all verified.
**Schema SHA256**: `4f8f9bee69f1a74e94415e724b6d71438ee0d352ffdfb97175b780b7196be514`

---

## Why this directory exists

`data/` is a **byte-identical frozen copy** of `preprocessing/outputs/`, verified
by SHA256 at copy time. It is deliberately a copy rather than a symlink or a
relative path into the preprocessing directory.

The preprocessing pipeline is owned by a teammate. If they re-run it — to fix
something, to add a feature, to try a different window — `preprocessing/outputs/`
changes underneath you, silently. Any experiment already run against the old
matrix becomes unreproducible, and 实验方案 §4.3 says the test set is touched
exactly once, so that is not recoverable by re-running.

Read from `experiment/data/`. Never read from `preprocessing/outputs/` in
modelling code.

If preprocessing is legitimately updated later, take a **new** snapshot
(`data_v2/`) rather than overwriting this one, and re-run the whole baseline
ladder against it so all rungs stay comparable.

---

## Directory layout

```
experiment/
├── README.md                  <- this file
├── HANDOFF.md                 <- what has been done, what comes next
├── NEW_SESSION_PROMPT.md      <- paste-ready opening prompt for a fresh session
├── data/                      <- frozen snapshot, treat as read-only
├── src/                       <- reusable code
├── notebooks/                 <- exploratory work
└── results/                   <- metrics, figures, saved models
```

---

## Cohort

| | Rows | Deaths | Prevalence |
|---|---|---|---|
| Train | 8,283 | 1,195 | 0.14427 |
| Validation | 1,775 | 256 | 0.14423 |
| Test | 1,775 | 256 | 0.14423 |
| **Total** | **11,833** | **1,707** | **0.1443** |

**Prevalence is 0.1443, not the 0.142 quoted in 实验方案 §1.2 and §5.5.** The 167
excluded negative-`Length_of_stay` records were all survivors, so the death count
stayed at 1,707 while the denominator fell from 12,000 to 11,833. AUPRC must be
read against 0.1443; an AUPRC of 0.30 is roughly 2.1× the no-skill floor, not 2.1×
of 0.142.

Splits are deterministic (seed 20260907), stratified by outcome, and disjoint by
RecordID. Use them as given — do not re-split, or cross-rung comparisons break.

---

## Files in `data/`

### Feature matrices

| File | Rows × Cols | Use for |
|---|---|---|
| `patient_features_unimputed.csv.gz` | 11,833 × 1,475 | **Tree models** (XGBoost / LightGBM). NaN preserved; let the model's native missing handling use it |
| `X_train_imputed.csv.gz` | 8,283 × 1,475 | Linear / distance-based models |
| `X_validation_imputed.csv.gz` | 1,775 × 1,475 | " |
| `X_test_imputed.csv.gz` | 1,775 × 1,475 | " — touch once, at the very end |

1,475 columns = `RecordID` + 1,474 model features. **Drop `RecordID` before fitting.**

Imputation is train-median only, computed on the 8,283 training rows and applied
unchanged to validation and test. No scaling was applied; fit any scaler inside
your training folds only.

The unimputed file contains 4,436,599 NaN cells (25.4% of the matrix), which is
expected — many lab tests are ordered for only a minority of patients, and that
missingness is itself informative.

### Labels and baseline scores

| File | Contents |
|---|---|
| `labels_and_splits.csv` | `RecordID`, `In-hospital_death`, `split` |
| `labels_and_baselines.csv` | the above **plus** `saps_i`, `saps_i_available`, `sofa`, `sofa_available` |

Use `labels_and_baselines.csv`. It is the only file you need for both y and the L2
clinical-score baseline, and it is built to contain no other outcome column.

**Do not merge `release/outcomes.csv` into anything.** It also holds
`Length_of_stay` and `Survival`, both of which leak the outcome. The builder
script `src/make_labels_and_baselines.py` does that merge once, safely, with an
explicit forbidden-column assertion.

### Documentation and provenance

| File | Contents |
|---|---|
| `feature_dictionary.csv` | one row per feature: source parameter, window, statistic, unit, missing semantics, notes |
| `feature_missingness.csv` | per-feature missing fraction and the train fill value used |
| `parameter_distributions.csv` | per-parameter min/p01/median/p99/max and rejected-value counts |
| `split_summary.csv` | rows / deaths / prevalence per split |
| `preprocessing_state.json` | exact column order, schema hash, all 1,474 fill values |
| `preprocessing_summary.json` | run provenance, audit counters, internal QA |
| `independent_verification.json` | independent reload check (hashes, recomputed fill values) |
| `source_manifest.json` | SHA256 of the raw inputs |
| `quality_flags_by_stay.csv` | per-stay data quality counters |
| `patient_features_48h_inclusive_affected_rows.csv.gz` | 232 boundary-sensitivity replacement rows (see below) |
| `manifest.sha256` | checksums of every file here; verify before a long run |

---

## Feature composition (1,474 model features)

| Group | Count |
|---|---|
| Raw dynamic parameters (37 params × 3 windows × 12 statistics) | 1,338 |
| Merged blood pressure, derived (3 streams × 3 windows × 12 stats) | 108 |
| Static descriptors incl. BMI and missing indicators | 16 |
| Record-level counters (3 per window) | 9 |
| Arterial-line indicator (1 per window) | 3 |

Windows are `0_24h`, `24_48h`, `0_48h` — 486 features each. The two sub-windows
exist to capture trajectory: compare a 24–48h statistic against its 0–24h
counterpart to separate deteriorating from improving patients.

The 12 statistics per parameter per window are `measured`, `count`, `first`,
`last`, `min`, `max`, `mean`, `median`, `std`, `slope_per_hour`, `delta`,
`time_span_hours`.

- **`delta`** = last − first. This *is* the "(末值−首值) trend" required by 实验方案 §4.1.
- **`slope_per_hour`** = least-squares slope over the window. A second, smoother trend.
- **`measured` / `count`** encode *whether and how often* a test was ordered. In this
  dataset those are often more predictive than the values themselves, and they are
  the natural material for the interpretability section.

### Derived features added in Phase 3

**Merged blood pressure** (`bp_sys`, `bp_dias`, `bp_mean`): per window, uses the
invasive arterial stream if that window has any invasive reading, else the
non-invasive cuff. Coverage rises from 70.0% (invasive alone) to 98.6%. The six
original source streams are retained unchanged, so you can ablate the merge.

**`bp__<window>__has_arterial_line`**: 1 if any invasive stream is present in that
window. Whether a clinician placed an arterial line is itself a severity signal.

**`static_bmi`** with `static_bmi_missing`: present for 6,181 patients (52.2%),
capped by Height availability.

Derived streams are excluded from the `record__*` counters, which still describe
only the 37 raw parameters.

---

## Traps

**1. `-1` in SAPS-I and SOFA means "not computable", not "score of -1".**

368 rows (3.11%) have `SAPS-I = -1`; 236 rows (1.99%) have `SOFA = -1`. Those
patients are *higher* risk than average (20.7% mortality for the SAPS-I group vs
14.43% cohort-wide), so leaving −1 in place as the lowest score inverts them and
depresses the baseline. Measured on the test set:

| Baseline | −1 kept as lowest | −1 rows excluded | n excluded |
|---|---|---|---|
| SAPS-I | AUROC 0.6156 / AUPRC 0.2091 | AUROC 0.6399 / AUPRC 0.2126 | 1,775 → 1,728 |
| SOFA | AUROC 0.6041 / AUPRC 0.2480 | AUROC 0.6231 / AUPRC 0.2527 | 1,775 → 1,733 |

`labels_and_baselines.csv` gives you NaN plus an `_available` flag so the choice
is explicit. **Recommendation**: report the baseline on the subset where the score
exists, and evaluate your model on that *same* subset for the head-to-head
comparison, while also reporting your model on the full test set. Say which
denominator each number uses. A baseline computed on 1,728 patients and a model
computed on 1,775 are not comparable.

**2. Clinical baselines must be recomputed on your exact test set.**

实验方案 §5.6 flags this and the numbers above confirm it. The plan's §5.5 quick
test reported SAPS-I AUROC 0.657 on n=1,800; the same score on this cohort's
1,775-row test set gives 0.6156 raw / 0.6399 cleaned. The cohort changed (167
exclusions) and split membership changed with it. **Do not quote the plan's L2
numbers in the report.** Recompute and cite your own.

**3. AUPRC must use `average_precision_score`.**

Not `auc(recall, precision)`. Trapezoidal interpolation on a PR curve is
optimistically biased, and the two disagree measurably when there are many ties —
which there are: SAPS-I takes only 32 distinct values and SOFA only 24 while
ranking 1,775 test patients. That tie density is itself worth discussing as a
reason discrete scores underperform continuous model probabilities.

**4. AUROC/AUPRC need only a ranking.** Pass the raw score; no probability
conversion, no monotone transform. Verified to change nothing.

**5. `record__*__missing_parameter_fraction` is a feature, not metadata.** It is a
legitimate severity proxy (sicker patients get more tests), but be ready to
discuss it if it lands high in feature importance — reviewers may read it as
leakage-adjacent even though it is computed purely from the 48-hour window.

**6. The boundary-sensitivity file is not extra training data.** The 232 rows in
`patient_features_48h_inclusive_affected_rows.csv.gz` are alternative versions of
232 stays computed with `Time = 48:00` included. The primary matrix excludes that
boundary. Use the file only for a sensitivity check; never concatenate it.

---

## Loading

```python
import pandas as pd

DATA = "experiment/data"

labels = pd.read_csv(f"{DATA}/labels_and_baselines.csv")
X_all = pd.read_csv(f"{DATA}/patient_features_unimputed.csv.gz")   # trees

df = labels.merge(X_all, on="RecordID", validate="one_to_one")
feature_cols = [c for c in X_all.columns if c != "RecordID"]       # 1,474

train = df[df["split"] == "train"]
valid = df[df["split"] == "validation"]
test  = df[df["split"] == "test"]                                  # touch once

X_tr, y_tr = train[feature_cols], train["In-hospital_death"]
```

For linear models load `X_{split}_imputed.csv.gz` instead and merge labels the
same way. Those files are already split, so do not filter by `split` again.

Verify the snapshot before a long run:

```bash
cd experiment/data && sha256sum -c manifest.sha256
```

---

## Environment

The snapshot was produced under Python 3.11.7, pandas 2.1.4, numpy 1.26.4. The
pipeline run took 686 s. Record whatever versions you use for the report's
reproducibility section.

---

## Reproducing the snapshot from scratch

```bash
cd Group_project/preprocessing
python mortality_preprocess.py              # ~11.5 min
python -m unittest discover -s tests -v     # 18 tests
python validate_phase3_outputs.py           # 66 checks
python verify_outputs.py                    # independent reload
```

Deterministic: fixed seed, no wall-clock or RNG in feature construction. Same
inputs reproduce the same schema hash. Then re-copy into `data/` and rebuild
`labels_and_baselines.csv` with `src/make_labels_and_baselines.py`.
