"""Task 2 feature engineering v2: clinically derived, cross-variable features.

`preprocess.py` summarises each variable on its own (37 variables x 3 windows x 12
statistics). Several of the strongest bedside signals are *combinations* of variables that a
tree model can only approximate with many splits, so they are built explicitly here, per
window (0-24 h, 24-48 h, 0-48 h; observations at exactly 48:00 excluded, as in the primary
features):

  pf_ratio      PaO2 / FiO2 (mmHg), each PaO2 paired with the nearest FiO2 within +-2 h --
                the oxygenation index behind the SOFA respiratory score (min, mean, last)
  shock_index   HR / systolic BP at the same minute, invasive pressure preferred over cuff --
                a haemodynamic-instability marker (max, mean, last)
  bun_cr_ratio  BUN / creatinine at the same draw -- pre-renal vs intrinsic renal injury (max, last)
  urine_ml_kg_h window urine total / admission weight / window hours -- the KDIGO oliguria unit

plus organ-dysfunction sub-scores from the project's rule-based SOFA engine
(src/task1_sofa/data_prep/scoring.py) for day 1 and day 2 and their change:

  organ_score__{0_24h|24_48h}__{resp,coag,liver,cardio,neuro,renal,total}, organ_score__delta_total

These are computed from the ICU records only -- never from outcomes.csv (whose SOFA/SAPS-I stay
excluded from X). Ratio inputs go through the shared physiological screen first; the organ
scores use the methodology-cleaned records exactly like task 1's targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..shared.physiology import MISSING, screen_series, validate_physiological_value
from ..task1_sofa.data_prep import parsing
from ..task1_sofa.data_prep.scoring import SYSTEMS, score_window

WINDOWS = {"0_24h": (0, 1440), "24_48h": (1440, 2880), "0_48h": (0, 2880)}
PAIR_WINDOW_MIN = 120


def _in(tv, lo, hi):
    return [(t, v) for t, v in tv if lo <= t < hi]


def _stats(values, which):
    if not values:
        return {k: np.nan for k in which}
    arr = np.asarray(values, dtype=float)
    fn = {"min": arr.min, "max": arr.max, "mean": arr.mean, "last": lambda: arr[-1]}
    return {k: float(fn[k]()) for k in which}


def _pf_pairs(pao2, fio2):
    out = []
    for tp, vp in pao2:
        cand = [(abs(tf - tp), tf, vf) for tf, vf in fio2 if abs(tf - tp) <= PAIR_WINDOW_MIN]
        if cand:
            vf = min(cand)[2]
            if vf > 0:
                out.append(vp / vf)
    return out


def _same_minute_ratio(num, den_primary, den_fallback=()):
    dp, df = dict(den_primary), dict(den_fallback)
    out = []
    for t, v in num:
        d = dp.get(t, df.get(t))
        if d:
            out.append(v / d)
    return out


def record_features(series_raw: dict, desc: dict) -> dict:
    s_raw = {p: [(t, v) for t, v in tv if t < 2880] for p, tv in series_raw.items()}
    s, _ = screen_series(s_raw)
    w = desc.get("Weight")
    w = None if w is None else validate_physiological_value("Weight", w)
    w = None if w in (None, MISSING) else w
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        g = lambda p: _in(s.get(p, []), lo, hi)  # noqa: E731
        out.update({f"pf_ratio__{name}__{k}": v for k, v in
                    _stats(_pf_pairs(g("PaO2"), g("FiO2")), ("min", "mean", "last")).items()})
        out.update({f"shock_index__{name}__{k}": v for k, v in
                    _stats(_same_minute_ratio(g("HR"), g("SysABP"), g("NISysABP")), ("max", "mean", "last")).items()})
        out.update({f"bun_cr_ratio__{name}__{k}": v for k, v in
                    _stats(_same_minute_ratio(g("BUN"), g("Creatinine")), ("max", "last")).items()})
        urine = g("Urine")
        out[f"urine_ml_kg_h__{name}"] = (sum(v for _, v in urine) / w / ((hi - lo) / 60)
                                         if urine and w else np.nan)
    for name in ("0_24h", "24_48h"):
        sw = score_window(s_raw, *WINDOWS[name])
        for a in SYSTEMS:
            out[f"organ_score__{name}__{a}"] = np.nan if sw[a] is None else float(sw[a])
        out[f"organ_score__{name}__total"] = float(sw["sofa_total"])
    out["organ_score__delta_total"] = out["organ_score__24_48h__total"] - out["organ_score__0_24h__total"]
    return out


def build_derived_features(record_ids, records: dict | None = None, data_dir=None) -> pd.DataFrame:
    """One row per RecordID. `records`: task-1-style parsed records {rid: (series, desc)}; when
    omitted, the raw files are parsed with task 1's methodology cleaning."""
    rows = []
    for rid in sorted(int(r) for r in record_ids):
        if records is not None:
            series, desc = records[rid]
        else:
            from pathlib import Path
            from ..task1_sofa import config as t1cfg
            series, desc = parsing.parse_record(Path(data_dir or t1cfg.DATA_DIR) / f"{rid}.csv")
        rows.append({"RecordID": rid, **record_features(series, desc)})
    return pd.DataFrame(rows)
