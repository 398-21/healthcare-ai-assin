> **Historical snapshot - superseded (2026-10-08).** This document records the earlier 11,833-stay analysis that excluded 167 stays with `Length_of_stay = -1`.

---

# Handoff — In-hospital Death Modelling

Written 2026-09-21 at the end of the preprocessing-validation work, for whoever
(or whichever session) picks up the modelling.

This document is **self-contained**. You do not need to read the four preprocessing
documents to start work. They are listed at the end if you need the history.

---

## Where things stand

The preprocessing pipeline was written by a teammate. It was independently
validated, four critical data-quality defects were found and fixed, five
lower-priority warnings were resolved, and everything was re-verified. **The
feature matrix is now frozen and ready to model against.**

Nothing has been modelled yet. Not one line of modelling code exists. You are
starting the modelling work from zero, against clean and documented data.

| | |
|---|---|
| Cohort | 11,833 ICU stays |
| Features | 1,474 (+ `RecordID`) |
| Splits | 8,283 / 1,775 / 1,775, stratified, deterministic (seed 20260907) |
| Prevalence | **0.1443** (1,707 deaths) |
| Data location | `Group_project/experiment/data/` — frozen, read-only |
| Verification | 18 unit tests, 66 output checks, independent reload: all pass |

Read `experiment/README.md` before writing code. It is the data dictionary and it
lists six traps that will otherwise cost you a rerun each.

---

## What was fixed, and why it matters to your results

Four defects were corrected. Each one would have quietly distorted a model.

**Mechanical ventilation was encoded as "unknown" instead of "not ventilated."**
When a patient had no `MechVent` record, the feature was NaN, so median imputation
assigned them a ventilation status. Absence of a record means the patient was not
on a ventilator — a clinical fact, not missing data. Now encoded 0/1.

**Physiologically impossible values were passing through.** Temperature of
−17.8 °C, pH of 735 (a misplaced decimal for 7.35), potassium of 22.9 mmol/L (a
fatal level), heart rate of 0. Range validation now rejects out-of-bounds values
as missing, with targeted corrections for the pH decimal error and a Height unit
error. 2,459 values were rejected in total. Without this, models would have been
fitting measurement artifacts.

**167 records had negative `Length_of_stay`.** Excluded, per 实验方案 §3.4. This is
the reason prevalence moved from 0.142 to 0.1443 — all 167 were survivors.

**Blood pressure was split across two parallel variables.** 28.6% of patients have
only cuff readings, 11.2% only arterial. Modelled separately, median imputation
handed the cuff-only patients a fabricated arterial pressure. Merged streams now
prefer invasive per window; coverage went from 70.0% to 98.6%.

Also added: BMI, an explicit arterial-line indicator. All additions are additive —
no existing column was renamed or removed.

---

## One decision deliberately left open

**Should `-1` in SAPS-I/SOFA be excluded or imputed when computing the L2 baseline?**

368 patients (3.11%) have no computable SAPS-I, 236 (1.99%) no SOFA. Left as a
numeric −1 they rank as lowest-risk, but they are actually *higher* risk (20.7%
mortality vs 14.43%). That inversion costs the baseline about 0.024 AUROC.

`labels_and_baselines.csv` gives NaN plus an `_available` flag rather than forcing
a choice. My recommendation is in README trap 1: report on the score-available
subset and evaluate your model on that same subset for the head-to-head, while
also reporting your model on the full test set, always naming the denominator.

This matters because the L2 rung is where clinical significance lives — "are we
better than the tool at the bedside today" — so the comparison has to be fair in
both directions.

---

## What to do next

Follow the baseline ladder in 实验方案 §5.5. The point of the ladder is that each
rung answers a different question; skipping one means you cannot answer that
question at the defence.

**L0 — no-skill floor.** AUPRC = 0.1443, AUROC = 0.5. One line of code. Establishes
that later numbers mean something.

**L1 — single variables.** Age alone, `gcs__0_48h__min` alone, `bun__0_48h__max`
alone. Answers: is the 1,474-feature pipeline worth building at all?

**L2 — clinical scores.** SAPS-I and SOFA, recomputed on your exact test set. This
is the rung with clinical meaning. Mind the −1 decision above.

**L3 — small logistic regression.** 5–10 hand-picked features. In the plan's rough
test this beat SAPS-I substantially, which makes it the real opponent for L4, not
the clinical scores.

**L4 — gradient boosting.** XGBoost / LightGBM on the unimputed matrix using native
NaN handling. Target range from the literature is AUROC 0.83–0.86.

Then interpretability: the project's narrative is *"bedside scores reach 0.64, we
reach 0.85 from the same 48 hours — where does the extra performance come from?"*
Candidate answers are temporal trends, measurement behaviour (the `measured` and
`count` features), and non-linear interactions. SHAP on the L4 model is the
natural tool.

### Metrics

Primary **AUPRC**, always printed next to the 0.1443 floor. Secondary **AUROC** for
literature comparability. Also report **min(Sensitivity, PPV)**, the official
PhysioNet 2012 metric, for direct comparison with the challenge entrants.

Pick the operating threshold on **validation** at recall ≈ 0.80 — missing a death
costs far more than one unnecessary review — then report sensitivity, PPV and
specificity at that threshold on test. Wrap test metrics in 95% bootstrap CIs
(1,000 resamples); with 256 test deaths the intervals are wide enough that small
differences between models will not be significant, and saying so is better than
over-claiming.

### The one hard rule

**Tune on train + validation only. Touch test once, at the very end.** 实验方案 §4.3.
Every threshold, every hyperparameter, every feature-selection decision comes from
validation. If you evaluate on test twice, the second number is not a test number.

Equally: **do not change the feature matrix mid-ladder.** If features change, every
completed rung is invalidated and the whole ladder must be rerun. If a change looks
necessary, snapshot to `data_v2/` and rerun everything.

---

## Suggested structure

```
experiment/
├── src/
│   ├── make_labels_and_baselines.py   # exists — the safe outcomes.csv merge
│   ├── data.py                        # loaders, split access, feature lists
│   ├── metrics.py                     # AUPRC/AUROC/min(Se,PPV), bootstrap CI, threshold picker
│   └── baselines.py                   # L0–L2
├── notebooks/                         # exploration
└── results/                           # metrics tables, figures, models
```

Write `metrics.py` first and have every rung call it. 实验方案 §5.6 warns that if
teammates compute AUPRC differently the numbers stop being comparable; one shared
implementation removes that risk.

---

## Things that will bite you

**Drop `RecordID` before fitting.** It is an ID in column 1 of every matrix. A tree
will happily split on it.

**Do not merge `release/outcomes.csv`.** It contains `Length_of_stay` and
`Survival`. Use `labels_and_baselines.csv`.

**The imputed files are already split.** Do not filter them by `split` again.

**The boundary file is not extra data.** 232 alternative rows for a sensitivity
check only. Never concatenate.

**Read from `experiment/data/`, never `preprocessing/outputs/`.** The teammate may
re-run their pipeline at any time.

**Class imbalance**: prefer `scale_pos_weight` / `class_weight` over resampling, and
if you do resample, resample inside training folds only. Never touch validation or
test distribution.

---

## Open items not blocking you

1. **Range thresholds lack clinical sign-off.** They come from standard reference
   ranges, not from a clinician on the team. Documented in
   `preprocessing/config.json` and the preprocessing README for review.
2. **The Weight time-zero overlap was left in place deliberately.** The 00:00
   descriptor also sits inside the Weight series so that 48-hour weight
   delta/slope keep an admission anchor. Reversible in ~10 minutes if the team
   disagrees; it should be a group decision.
3. **BMI covers only 52%** of patients, limited by Height availability.
4. **Two stays have no valid dynamic observations at all.** Retained with
   missingness indicators rather than dropped.

---

## History, if you need it

In `Group_project/`, in order. All frozen; do not edit them.

| Document | Role |
|---|---|
| `实验方案.md` | The group's experiment plan. §3 data traps, §4 pipeline, §5 metrics, §5.5 baseline ladder |
| `preprocessing_validation_report.md` | 09-20 independent validation: 4 critical issues, 5 warnings |
| `preprocessing_fix_plan.md` | Phased fix plan |
| `phase1_2_implementation_summary.md` | First fix pass; pH still failing |
| `phase1_2_implementation_summary_v2.md` | pH resolved; 4 critical fixes verified |
| `phase3_implementation_summary.md` | Warnings resolved; closes the fix plan |

Note that 实验方案 §5.5's quick-test numbers predate all the fixes and a 12,000-row
cohort. Treat them as motivation, not as baselines to cite.
