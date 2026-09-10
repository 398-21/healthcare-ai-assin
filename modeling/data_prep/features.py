"""Expanding-window features at each decision time T.

Every feature for (RecordID, T) is derived ONLY from measurements with t_min < T*60
(`config.feature_cutoff_min`). Nothing from `outcomes.csv`. The rule-SOFA feature
blocks are computed from `icu_records` and are therefore part of "our data".

Feature families (prefixes):
  v_<Param>_<stat>   per-variable summary over [0, T)
  e_<name>           engineered hemodynamic / respiratory / renal signals
  sofa0_24_<...>     rule SOFA over the fixed admission day [0, 24)h
  sofa_now_<...>     rule SOFA over the trailing 24 h [T-24, T)h  ( == SOFA_now )
  d_<name>           admission descriptors
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config
from .scoring import score_window, w_vals, w_tv, _nearest, SYSTEMS

FEATURE_VARS = [
    "GCS", "HR", "RespRate", "Temp", "Urine",
    "MAP", "NIMAP", "SysABP", "NISysABP", "DiasABP", "NIDiasABP",
    "Creatinine", "BUN", "Platelets", "WBC", "HCT", "K", "Na", "HCO3", "Glucose", "Mg",
    "PaO2", "FiO2", "PaCO2", "pH", "SaO2", "MechVent", "Lactate",
    "Bilirubin", "Albumin", "ALP", "ALT", "AST", "Cholesterol", "TroponinI", "TroponinT",
]

_STAT_KEYS = ("last", "first", "min", "max", "mean", "std", "count",
              "hrs_since_last", "hrs_coverage", "slope12h")


def _var_feats(tv, cutoff, prefix, out):
    if not tv:
        for k in _STAT_KEYS:
            out[f"{prefix}_{k}"] = 0.0 if k == "count" else np.nan
        return
    ts = np.fromiter((t for t, _ in tv), dtype=float, count=len(tv))
    vs = np.fromiter((v for _, v in tv), dtype=float, count=len(tv))
    out[f"{prefix}_last"] = vs[-1]
    out[f"{prefix}_first"] = vs[0]
    out[f"{prefix}_min"] = vs.min()
    out[f"{prefix}_max"] = vs.max()
    out[f"{prefix}_mean"] = vs.mean()
    out[f"{prefix}_std"] = vs.std() if len(vs) > 1 else 0.0
    out[f"{prefix}_count"] = float(len(vs))
    out[f"{prefix}_hrs_since_last"] = (cutoff - ts[-1]) / 60.0
    out[f"{prefix}_hrs_coverage"] = (ts[-1] - ts[0]) / 60.0
    m = ts >= (cutoff - 12 * 60)
    if m.sum() >= 2 and np.unique(ts[m]).size >= 2:
        out[f"{prefix}_slope12h"] = float(np.polyfit(ts[m] / 60.0, vs[m], 1)[0])
    else:
        out[f"{prefix}_slope12h"] = np.nan


def _sofa_block(series, lo, hi, prefix, out):
    sw = score_window(series, lo, hi)
    for a in SYSTEMS:
        out[f"{prefix}_{a}"] = np.nan if sw[a] is None else float(sw[a])
        out[f"{prefix}_{a}_present"] = int(sw[f"{a}_present"])
    out[f"{prefix}_total"] = float(sw["sofa_total"])
    out[f"{prefix}_n_present"] = int(sw["n_present"])


def _clip(series, cutoff):
    return {p: [(t, v) for (t, v) in tv if t < cutoff] for p, tv in series.items()}


def _cohort_weight_medians(records):
    wl = {0: [], 1: []}
    for _rid, (_s, d) in records.items():
        w, g = d.get("Weight"), d.get("Gender")
        if w and w > 20 and g in (0, 1):
            wl[int(g)].append(w)
    return {g: (float(np.median(v)) if v else 78.0) for g, v in wl.items()}


def build_features(records: dict[int, tuple[dict, dict]]) -> pd.DataFrame:
    wmed = _cohort_weight_medians(records)
    rows = []
    for rid, (series, desc) in records.items():
        for T in config.HORIZONS_H:
            cutoff = config.feature_cutoff_min(T)
            s = _clip(series, cutoff)
            out = {"RecordID": rid, "origin_h": T}

            for p in FEATURE_VARS:
                _var_feats(s.get(p, []), cutoff, f"v_{p}", out)

            # ---- engineered: hemodynamics ----
            maps = w_vals(s, "MAP", 0, cutoff) + w_vals(s, "NIMAP", 0, cutoff)
            out["e_map_min"] = min(maps) if maps else np.nan
            out["e_map_n"] = float(len(maps))
            out["e_map_frac_lt70"] = float(np.mean([x < 70 for x in maps])) if maps else np.nan
            out["e_map_frac_lt65"] = float(np.mean([x < 65 for x in maps])) if maps else np.nan

            # ---- engineered: neuro / sedation ----
            gcs = w_tv(s, "GCS", 0, cutoff)
            mv = w_tv(s, "MechVent", 0, cutoff)
            out["e_gcs_min"] = min(v for _, v in gcs) if gcs else np.nan
            out["e_sedation_suspected"] = float(
                sum(1 for tg, vg in gcs if vg <= 8 and _nearest(mv, tg) is not None))
            out["e_mechvent_any"] = float(bool(mv))
            out["e_mechvent_hrs_since"] = (cutoff - mv[-1][0]) / 60.0 if mv else np.nan

            # ---- engineered: renal / urine rate over last 24 h (mL/kg/h) ----
            u24 = w_vals(s, "Urine", cutoff - 1440, cutoff)
            w = desc.get("Weight")
            if not w or w <= 20:
                g = int(desc["Gender"]) if desc.get("Gender") in (0, 1) else 0
                w = wmed.get(g, 78.0)
                out["e_weight_imputed"] = 1.0
            else:
                out["e_weight_imputed"] = 0.0
            out["e_weight_used"] = float(w)
            out["e_urine_rate_24h"] = (sum(u24) / w / 24.0) if u24 else np.nan
            out["e_urine_sum_24h"] = float(sum(u24)) if u24 else np.nan

            # ---- engineered: respiratory P/F so far ----
            pao2, fio2 = w_tv(s, "PaO2", 0, cutoff), w_tv(s, "FiO2", 0, cutoff)
            pf = []
            for tp, vp in pao2:
                vf = _nearest(fio2, tp)
                if vf and vf > 0:
                    pf.append(vp / vf)
            out["e_pf_min"] = min(pf) if pf else np.nan
            out["e_pf_n"] = float(len(pf))

            # ---- rule SOFA blocks (computed from our data) ----
            _sofa_block(s, *config.FULL_DAY1_MIN, "sofa0_24", out)
            _sofa_block(s, *config.now_window_min(T), "sofa_now", out)

            # ---- descriptors ----
            out["d_age"] = desc.get("Age", np.nan)
            out["d_gender"] = desc.get("Gender", np.nan)
            out["d_icutype"] = desc.get("ICUType", np.nan)
            out["d_weight_adm"] = desc.get("Weight", np.nan)
            out["d_height"] = desc.get("Height", np.nan)

            rows.append(out)

    df = pd.DataFrame(rows).sort_values(["RecordID", "origin_h"]).reset_index(drop=True)
    return df
