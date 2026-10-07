"""Record-level train / val / test split.

Each ICU stay (RecordID) goes entirely to one split, so all four horizon-rows of a
record share its split. Stratified on a coarse SOFA-severity band x in-hospital death
(records with SOFA == -1 form their own band). Frozen by `config.SEED`.
No patient linkage exists in this dataset, so record-level is the finest grouping
available.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config

_SOFA_BINS = [-2, -0.5, 2, 5, 8, 11, 14, 100]
_SOFA_LABELS = ["na", "0-2", "3-5", "6-8", "9-11", "12-14", "15+"]


def _strat_key(oc: pd.DataFrame) -> pd.Series:
    band = pd.cut(oc["SOFA"].to_numpy(), bins=_SOFA_BINS, labels=_SOFA_LABELS).astype(str)
    death = oc["In-hospital_death"].astype(int).astype(str)
    return pd.Series(band, index=oc.index) + "|d" + death.values


def make_splits(ratios=config.SPLIT_RATIOS, seed: int = config.SEED,
                outcomes_csv=config.OUTCOMES_CSV) -> pd.DataFrame:
    oc = pd.read_csv(outcomes_csv).sort_values("RecordID").reset_index(drop=True)
    key = _strat_key(oc)
    idx = np.arange(len(oc))
    tr_frac, va_frac, te_frac = ratios

    tr_idx, rest_idx = train_test_split(
        idx, train_size=tr_frac, random_state=seed, stratify=key)
    va_idx, te_idx = train_test_split(
        rest_idx, train_size=va_frac / (va_frac + te_frac),
        random_state=seed, stratify=key.iloc[rest_idx])

    split = np.array(["train"] * len(oc), dtype=object)
    split[va_idx] = "val"
    split[te_idx] = "test"

    return pd.DataFrame({
        "RecordID": oc["RecordID"].to_numpy(),
        "split": split,
        "has_sofa": (oc["SOFA"] != -1).to_numpy(),
        "strat_key": key.to_numpy(),
    })


def split_summary(splits: pd.DataFrame, icutype: pd.Series,
                  outcomes_csv=config.OUTCOMES_CSV) -> pd.DataFrame:
    """Per-split balance table. `icutype` is the admission ICUType indexed by RecordID."""
    oc = pd.read_csv(outcomes_csv)[["RecordID", "SOFA", "SAPS-I", "In-hospital_death"]]
    icu = icutype.rename("ICUType").rename_axis("RecordID").reset_index()
    m = splits.merge(oc, on="RecordID").merge(icu, on="RecordID")
    m["sofa_lab"] = m.SOFA.where(m.SOFA != -1)

    rows = {}
    for name, d in m.groupby("split"):
        rows[name] = {
            "n": len(d),
            "pct": round(100 * len(d) / len(m), 1),
            "sofa_mean": round(d.sofa_lab.mean(), 2),
            "sofa_p25": d.sofa_lab.quantile(.25),
            "sofa_p75": d.sofa_lab.quantile(.75),
            "death_rate": round(d["In-hospital_death"].mean(), 3),
            "sofa_missing": int((d.SOFA == -1).sum()),
            **{f"icu{k}": round((d.ICUType == k).mean(), 3) for k in (1, 2, 3, 4)},
        }
    return pd.DataFrame(rows).T.loc[["train", "val", "test"]]
