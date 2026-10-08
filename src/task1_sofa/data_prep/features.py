"""Expanding-window features at each decision time T.

Every feature for (RecordID, T) is derived ONLY from measurements with t_min < T*60
(`config.feature_cutoff_min`). Nothing from `outcomes.csv`. The rule-SOFA feature
blocks are computed from `icu_records` and are therefore part of "our data".

Pre-processing applied here (all statistics fitted on TRAINING stays only):
  * physiological screening of model inputs (src/shared/physiology.py): implausible values
    (e.g. arterial pressure 0, temperature -17.8 C, a single 19,990 mL urine entry) become
    missing. The rule-SOFA blocks deliberately use the unscreened, methodology-cleaned series
    so `sofa_now_*` is computed exactly like the forecast target.
  * the sex-specific median weight used when admission weight is missing (urine-rate
    feature only, flagged by `e_weight_imputed`) is computed from training stays only.

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
from ...shared.physiology import screen_series, validate_physiological_value, MISSING
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


def _valid_desc(param, v):
    """Admission descriptor after the shared physiological screen (None if implausible)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    vv = validate_physiological_value(param, v)
    return None if vv == MISSING else vv


def _cohort_weight_medians(records):
    wl = {0: [], 1: []}
    for _rid, (_s, d) in records.items():
        w, g = _valid_desc("Weight", d.get("Weight")), d.get("Gender")
        if w is not None and g in (0, 1):
            wl[int(g)].append(w)
    return {g: (float(np.median(v)) if v else 78.0) for g, v in wl.items()}


def input_screening_audit(records) -> pd.DataFrame:
    """How many methodology-cleaned values the physiological screen removes, per variable."""
    removed, total = {}, {}
    for _rid, (series, _d) in records.items():
        _, rm = screen_series(series)
        for p, tv in series.items():
            total[p] = total.get(p, 0) + len(tv)
        for p, n in rm.items():
            removed[p] = removed.get(p, 0) + n
    df = pd.DataFrame({"values": pd.Series(total), "set to missing": pd.Series(removed)}).fillna(0).astype(int)
    df["%"] = 100 * df["set to missing"] / df["values"].where(df["values"] > 0)
    return df[df["set to missing"] > 0].sort_values("set to missing", ascending=False)


def build_features(records: dict[int, tuple[dict, dict]], train_ids) -> pd.DataFrame:
    """`train_ids`: RecordIDs of the training split, the only stays used to fit statistics."""
    train_ids = set(train_ids)
    wmed = _cohort_weight_medians({r: v for r, v in records.items() if r in train_ids})
    rows = []
    for rid, (series_raw, desc) in records.items():
        series, _ = screen_series(series_raw)
        for T in config.HORIZONS_H:
            cutoff = config.feature_cutoff_min(T)
            s = _clip(series, cutoff)              # screened: model inputs
            s_raw = _clip(series_raw, cutoff)      # methodology-cleaned: rule-SOFA blocks
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
            w = _valid_desc("Weight", desc.get("Weight"))
            if w is None:
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
            _sofa_block(s_raw, *config.FULL_DAY1_MIN, "sofa0_24", out)
            _sofa_block(s_raw, *config.now_window_min(T), "sofa_now", out)

            # ---- descriptors (screened) ----
            for name, param in (("d_age", "Age"), ("d_weight_adm", "Weight"), ("d_height", "Height")):
                v = _valid_desc(param, desc.get(param))
                out[name] = np.nan if v is None else v
            out["d_gender"] = desc.get("Gender", np.nan)
            out["d_icutype"] = desc.get("ICUType", np.nan)

            rows.append(out)

    df = pd.DataFrame(rows).sort_values(["RecordID", "origin_h"]).reset_index(drop=True)
    return df
