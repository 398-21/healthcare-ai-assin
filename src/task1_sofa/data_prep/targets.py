"""Forecast targets: the rolling trailing-24 h SOFA at each decision time.

For every (RecordID, T) with T in {24, 30, 36, 42}:
  * target window  = [T-18h, T+6h)   -> 6 rule sub-scores + total  (`*_score`, `sofa_total`)
  * now window      = [T-24h, T)      -> SOFA_now
  * delta            = target - now    (`*_delta`, `sofa_delta`)
  * deteriorate_24h  = sofa_delta >= config.DETERIORATE_DELTA
  * cardio_instability = min MAP (invasive U non-invasive) < 65 in the target window

Cardiovascular is 0/1 (no vasopressor data) — a known project limitation, present on
both the feature and target side.

Rows whose target window has no measurement at all (`target_window_empty == 1`) keep
NaN score columns and are excluded at evaluation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config
from .scoring import score_window, SYSTEMS


def build_targets(records: dict[int, tuple[dict, dict]]) -> pd.DataFrame:
    rows = []
    for rid, (series, _desc) in records.items():
        for T in config.HORIZONS_H:
            tlo, thi = config.target_window_min(T)
            nlo, nhi = config.now_window_min(T)
            tgt = score_window(series, tlo, thi)
            now = score_window(series, nlo, nhi)
            empty = tgt["n_present"] == 0

            row = {"RecordID": rid, "origin_h": T}
            for a in SYSTEMS:
                row[f"{a}_score"] = np.nan if (empty or tgt[a] is None) else float(tgt[a])
                row[f"{a}_present"] = int(tgt[f"{a}_present"])
                nd = None if (tgt[a] is None or now[a] is None) else tgt[a] - now[a]
                row[f"{a}_delta"] = np.nan if (empty or nd is None) else float(nd)

            row["sofa_total"] = np.nan if empty else float(tgt["sofa_total"])
            row["sofa_now"] = float(now["sofa_total"])
            row["sofa_delta"] = np.nan if empty else float(tgt["sofa_total"] - now["sofa_total"])
            row["deteriorate_24h"] = (
                np.nan if empty
                else int((tgt["sofa_total"] - now["sofa_total"]) >= config.DETERIORATE_DELTA))
            row["n_present_target"] = int(tgt["n_present"])
            row["n_present_now"] = int(now["n_present"])
            mm = tgt["min_map"]
            row["cardio_instability"] = np.nan if mm is None else int(mm < 65)
            row["target_window_empty"] = int(empty)
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["RecordID", "origin_h"]).reset_index(drop=True)
