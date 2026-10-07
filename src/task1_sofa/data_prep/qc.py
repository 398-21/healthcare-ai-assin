"""Data-quality audit + QC report for Phase 1.

`audit_raw` re-scans the raw record files (before cleaning) and reproduces the
methodology-doc section-7 counts exactly.
`write_qc_report` renders `preprocessed/task1_sofa/qc_report.md`.
"""
from __future__ import annotations
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

# doc section-7 reference counts (must match exactly)
DOC_REF = {
    "empty_param_rows": 2403, "empty_param_records": 1594, "MAP_eq_0": 86,
    "PaO2_eq_0": 3, "SaO2_eq_0": 2, "Weight_neg1": 1001, "Weight_le20": 353,
    "Weight_gt300": 1, "Height_lt100": 17, "Height_gt250": 11, "MechVent_rows": 91644,
}


def audit_raw(data_dir=None) -> dict:
    data_dir = Path(data_dir or config.DATA_DIR)
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    a = {k: 0 for k in DOC_REF if k != "empty_param_records"}
    empty_records = set()
    mechvent_values = set()
    last_t_min = []
    param_valid = {}

    for f in files:
        mx = 0
        with open(f, newline="") as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if len(row) != 3:
                    continue
                t_s, p, v_s = row
                try:
                    hh, mm = t_s.split(":")
                    t = int(hh) * 60 + int(mm)
                    v = float(v_s)
                except ValueError:
                    continue
                mx = max(mx, t)
                if p == "":
                    a["empty_param_rows"] += 1
                    empty_records.add(f)
                    continue
                if p == "MAP" and v == 0:
                    a["MAP_eq_0"] += 1
                if p == "PaO2" and v == 0:
                    a["PaO2_eq_0"] += 1
                if p == "SaO2" and v == 0:
                    a["SaO2_eq_0"] += 1
                if p == "Weight":
                    if v == -1:
                        a["Weight_neg1"] += 1
                    elif v <= 20:
                        a["Weight_le20"] += 1
                    elif v > 300:
                        a["Weight_gt300"] += 1
                if p == "Height":
                    if v != -1 and v < 100:
                        a["Height_lt100"] += 1
                    if v > 250:
                        a["Height_gt250"] += 1
                if p == "MechVent":
                    a["MechVent_rows"] += 1
                    mechvent_values.add(v)
                if _keep(p, v):
                    param_valid[p] = param_valid.get(p, 0) + 1
        last_t_min.append(mx)

    a["empty_param_records"] = len(empty_records)
    return {
        "counts": a,
        "mechvent_values": sorted(mechvent_values),
        "last_t_hours": np.array(last_t_min) / 60.0,
        "param_valid": param_valid,
        "n_files": len(files),
    }


def _keep(param, v):
    from .parsing import keep_value
    return keep_value(param, v)


def window_coverage(records: dict) -> pd.DataFrame:
    """% of records with >=1 valid value of each key var in each target window."""
    keyv = ["GCS", "HR", "Creatinine", "Platelets", "Bilirubin", "PaO2", "FiO2"]
    rows = []
    n = len(records)
    for T in config.HORIZONS_H:
        lo, hi = config.target_window_min(T)
        row = {"origin_h": T, "window": f"[{lo//60},{hi//60})h"}
        for p in keyv:
            c = sum(1 for _rid, (s, _d) in records.items()
                    if any(lo <= t < hi for t, _v in s.get(p, [])))
            row[p] = round(100 * c / n, 1)
        # combined MAP availability
        c = sum(1 for _rid, (s, _d) in records.items()
                if any(lo <= t < hi for t, _v in s.get("MAP", []) + s.get("NIMAP", [])))
        row["MAP|NIMAP"] = round(100 * c / n, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def write_qc_report(audit: dict, records: dict, targets: pd.DataFrame,
                    features: pd.DataFrame, splits: pd.DataFrame,
                    path=config.QC_REPORT_MD) -> str:
    a = audit["counts"]
    lt = audit["last_t_hours"]
    lines = ["# Phase 1 — data-quality report", ""]

    lines.append("## Section-7 implausible-value audit (computed | doc)")
    lines.append("")
    lines.append("| check | computed | doc | ok |")
    lines.append("| --- | --- | --- | --- |")
    for k, ref in DOC_REF.items():
        got = a[k]
        lines.append(f"| {k} | {got} | {ref} | {'yes' if got == ref else '**NO**'} |")
    lines.append("")
    lines.append(f"MechVent distinct values: {audit['mechvent_values']} "
                 f"(existence flag only — cardiovascular stays capped 0/1).")
    lines.append("")

    lines.append("## Record duration")
    lines.append("")
    lines.append(f"- last observation (h): min {lt.min():.1f}, p5 {np.percentile(lt,5):.1f}, "
                 f"median {np.percentile(lt,50):.1f}, max {lt.max():.1f}")
    for thr in (36, 42, 46, 47):
        lines.append(f"- records ending before {thr} h: {(lt < thr).sum()}")
    lines.append("")

    lines.append("## Target-window coverage (% of 12,000 records with >=1 valid value)")
    lines.append("")
    lines.append("```")
    lines.append(window_coverage(records).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Target table")
    lines.append("")
    g = targets.groupby("origin_h")
    summ = g[["sofa_total", "sofa_now", "sofa_delta", "deteriorate_24h",
              "cardio_instability", "target_window_empty"]].mean().round(3)
    lines.append("```")
    lines.append(summ.to_string())
    lines.append("```")
    lines.append("")
    lines.append(f"- rows: {len(targets)}  ({targets.RecordID.nunique()} records x "
                 f"{targets.origin_h.nunique()} horizons)")
    lines.append(f"- deterioration (SOFA rises >= {config.DETERIORATE_DELTA} in 24 h) base rate: "
                 f"{targets.deteriorate_24h.mean():.3f}")
    lines.append("")

    lines.append("## Feature table")
    lines.append("")
    lines.append(f"- shape: {features.shape}  ({features.shape[1]-2} feature columns)")
    nan_frac = features.drop(columns=["RecordID", "origin_h"]).isna().mean().sort_values()
    lines.append(f"- columns fully populated: {(nan_frac == 0).sum()}; "
                 f"median column NaN fraction: {nan_frac.median():.2f}")
    lines.append(f"- most-missing feature: {nan_frac.index[-1]} ({nan_frac.iloc[-1]:.2f})")
    lines.append("")

    lines.append("## Split")
    lines.append("")
    from .splits import split_summary
    lines.append("```")
    lines.append(split_summary(splits).to_string())
    lines.append("```")
    lines.append("")

    txt = "\n".join(lines)
    Path(path).write_text(txt, encoding="utf-8")
    return txt
