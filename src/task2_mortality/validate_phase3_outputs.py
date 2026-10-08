"""Validate preprocessing outputs after Phase 1+2+3.

Re-checks every Phase 1/2 guarantee (so Phase 3 cannot silently regress them)
and adds Phase 3 checks for the merged blood pressure streams and BMI.

Run:  python validate_phase3_outputs.py
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "preprocessed" / "task2_mortality"

failures: list[str] = []
checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {label}" + (f"   [{detail}]" if detail else ""))
        failures.append(label)


def main(out_dir: Path | str | None = None) -> int:
    """Print one PASS/FAIL line per check; raise RuntimeError if any check failed."""
    global OUT, checks
    if out_dir is not None:
        OUT = Path(out_dir)
    failures.clear()
    checks = 0
    labels = pd.read_csv(OUT / "labels_and_splits.csv")
    dist = pd.read_csv(OUT / "parameter_distributions.csv")
    fdict = pd.read_csv(OUT / "feature_dictionary.csv")
    with gzip.open(OUT / "patient_features_unimputed.csv.gz", "rt") as handle:
        feats = pd.read_csv(handle)
    summary = json.loads((OUT / "preprocessing_summary.json").read_text(encoding="utf-8"))
    state = json.loads((OUT / "preprocessing_state.json").read_text(encoding="utf-8"))

    print("=" * 78)
    print("PHASE 1 + 2 REGRESSION CHECKS")
    print("=" * 78)

    check("run status is PASS", summary["status"] == "PASS", summary["status"])
    check("cohort includes all 12,000 challenge stays", len(labels) == 12000, f"{len(labels)}")
    check("features rowcount matches labels", len(feats) == len(labels), f"{len(feats)}")

    prevalence = labels.groupby("split", observed=True)["In-hospital_death"].mean()
    spread = float(prevalence.max() - prevalence.min())
    check(
        "splits stratified within 0.1pp",
        spread < 0.001,
        ", ".join(f"{k}={v:.4f}" for k, v in prevalence.items()),
    )

    mech_cols = [c for c in feats.columns if c.startswith("mechvent__") and c.endswith("__ever")]
    check("MechVent ever columns exist", len(mech_cols) == 3, f"{len(mech_cols)} cols")
    for col in mech_cols:
        vals = set(feats[col].dropna().unique())
        check(
            f"{col} is 0/1 with no NaN",
            vals <= {0.0, 1.0} and not feats[col].isna().any(),
            f"values={sorted(vals)}",
        )

    ranges = {
        "pH": (6.5, 8.0), "Temp": (30.0, 44.0), "K": (1.5, 10.0),
        "HR": (20, 250), "BUN": (1, 300), "DiasABP": (20, 200),
        "NIDiasABP": (20, 200), "Na": (100, 180), "Glucose": (10, 2000),
    }
    for param, (lo, hi) in ranges.items():
        row = dist[dist["Parameter"] == param]
        if row.empty:
            check(f"{param} present in distributions", False)
            continue
        lo_actual = float(row["min"].iloc[0])
        hi_actual = float(row["max"].iloc[0])
        check(
            f"{param} within [{lo}, {hi}]",
            lo_actual >= lo and hi_actual <= hi,
            f"[{lo_actual:.2f}, {hi_actual:.2f}]",
        )

    print()
    print("=" * 78)
    print("PHASE 3 DERIVED FEATURE CHECKS")
    print("=" * 78)

    # --- BMI (W5) ---
    check("static_bmi column exists", "static_bmi" in feats.columns)
    check("static_bmi_missing column exists", "static_bmi_missing" in feats.columns)
    if "static_bmi" in feats.columns:
        bmi = feats["static_bmi"]
        present = bmi.notna()
        check(
            "static_bmi_missing agrees with static_bmi NaN pattern",
            bool((feats["static_bmi_missing"] == (~present).astype(float)).all()),
            f"present={int(present.sum())}",
        )
        check(
            "static_bmi recomputes from Height/Weight",
            bool(
                (
                    (bmi[present]
                     - feats.loc[present, "static_admission_weight_kg"]
                     / (feats.loc[present, "static_height_cm"] / 100) ** 2).abs()
                    < 0.01
                ).all()
            ),
        )
        check(
            "static_bmi present only when both inputs present",
            bool(
                (present == (feats["static_height_cm"].notna() & feats["static_admission_weight_kg"].notna())).all()
            ),
        )
        if present.any():
            check(
                "static_bmi values physiologically plausible",
                float(bmi[present].min()) >= 8 and float(bmi[present].max()) <= 120,
                f"[{bmi[present].min():.1f}, {bmi[present].max():.1f}]",
            )

    # --- merged BP (W1) ---
    for win in ("0_24h", "24_48h", "0_48h"):
        for stream in ("bp_sys", "bp_dias", "bp_mean"):
            check(f"{stream}__{win}__mean exists", f"{stream}__{win}__mean" in feats.columns)
        line_col = f"bp__{win}__has_arterial_line"
        check(f"{line_col} exists", line_col in feats.columns)
        if line_col not in feats.columns:
            continue

        line = feats[line_col]
        check(
            f"{line_col} is strictly 0/1 with no NaN",
            set(line.dropna().unique()) <= {0.0, 1.0} and not line.isna().any(),
            f"rate={line.mean():.3f}",
        )
        # The indicator is shared across all three groups: it is 1 when ANY invasive
        # stream has a reading in this window. The three streams are usually recorded
        # together but not always (e.g. MAP present while SysABP is absent), so this
        # must be checked against their union, not against SysABP alone.
        inv_any = (
            (feats[f"sysabp__{win}__measured"].fillna(0) == 1)
            | (feats[f"diasabp__{win}__measured"].fillna(0) == 1)
            | (feats[f"map__{win}__measured"].fillna(0) == 1)
        )
        check(
            f"{line_col} matches any-invasive availability",
            bool((line == inv_any.astype(float)).all()),
            f"any_invasive={int(inv_any.sum())}",
        )

        # merged stream must equal invasive where that group's own invasive stream
        # is available, else fall back to its non-invasive counterpart
        inv_measured = feats[f"sysabp__{win}__measured"].fillna(0) == 1
        merged = feats[f"bp_sys__{win}__mean"]
        inv = feats[f"sysabp__{win}__mean"]
        nin = feats[f"nisysabp__{win}__mean"]
        expected = inv.where(inv_measured, nin)
        both_nan = merged.isna() & expected.isna()
        agree = ((merged - expected).abs() < 0.01) | both_nan
        check(
            f"bp_sys__{win}__mean = invasive-preferred merge",
            bool(agree.all()),
            f"mismatches={int((~agree).sum())}",
        )

        # coverage must be at least as good as either source alone
        cov_merged = float((feats[f"bp_sys__{win}__measured"].fillna(0) == 1).mean())
        cov_inv = float(inv_measured.mean())
        cov_nin = float((feats[f"nisysabp__{win}__measured"].fillna(0) == 1).mean())
        check(
            f"bp_sys__{win} coverage >= max(source coverages)",
            cov_merged >= max(cov_inv, cov_nin) - 1e-9,
            f"merged={cov_merged:.3f} inv={cov_inv:.3f} ni={cov_nin:.3f}",
        )

    # --- source columns retained (additive, not replacing) ---
    for src in ("sysabp", "nisysabp", "diasabp", "nidiasabp", "map", "nimap"):
        check(f"source stream {src} retained", f"{src}__0_48h__mean" in feats.columns)

    # --- record-level counters must ignore derived streams ---
    check(
        "missing_parameter_fraction still based on 37 raw parameters",
        bool(
            (
                (feats["record__0_48h__missing_parameter_fraction"]
                 - (1 - feats["record__0_48h__distinct_parameter_count"] / 37)).abs() < 1e-6
            ).all()
        ),
    )
    check(
        "distinct_parameter_count never exceeds 37",
        int(feats["record__0_48h__distinct_parameter_count"].max()) <= 37,
        f"max={int(feats['record__0_48h__distinct_parameter_count'].max())}",
    )

    print()
    print("=" * 78)
    print("SCHEMA CONTRACT CHECKS")
    print("=" * 78)

    declared = fdict["feature_name"].tolist()
    check("feature dictionary has no duplicates", len(declared) == len(set(declared)))
    check(
        "unimputed columns exactly match dictionary order",
        feats.columns.tolist() == declared,
        f"{len(feats.columns)} vs {len(declared)}",
    )
    check(
        "state feature_columns matches dictionary minus RecordID",
        state["feature_columns"] == declared[1:],
        f"{len(state['feature_columns'])} cols",
    )
    check(
        "summary model_feature_columns agrees",
        int(summary["output"]["model_feature_columns"]) == len(declared) - 1,
        f"{summary['output']['model_feature_columns']}",
    )

    # no fully-empty derived columns (would signal a wiring bug)
    derived = [c for c in declared if c.startswith(("bp_sys", "bp_dias", "bp_mean", "bp__", "static_bmi"))]
    all_nan = [c for c in derived if feats[c].isna().all()]
    check(
        "no derived column is entirely NaN",
        not all_nan,
        f"{len(derived)} derived cols, empty={all_nan[:5]}",
    )

    # splits align with the new schema
    for split in ("train", "validation", "test"):
        with gzip.open(OUT / f"X_{split}_imputed.csv.gz", "rt") as handle:
            frame = pd.read_csv(handle, nrows=5)
        check(f"X_{split} columns match schema", frame.columns.tolist() == declared)
        with gzip.open(OUT / f"X_{split}_imputed.csv.gz", "rt") as handle:
            full = pd.read_csv(handle)
        check(f"X_{split} has no NaN after imputation", not full.isna().any().any(), f"{len(full)} rows")

    print()
    print("=" * 78)
    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED:")
        for name in failures:
            print(f"  - {name}")
        print("=" * 78)
        raise RuntimeError(f"{len(failures)} pre-processing check(s) failed: {failures}")
    print("ALL CHECKS PASSED")
    print("=" * 78)
    return checks


if __name__ == "__main__":
    try:
        main()
    except RuntimeError:
        sys.exit(1)
