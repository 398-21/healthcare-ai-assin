"""Shared plumbing for every model family: which columns are features, and how to
build a (train|val|test) design matrix for one axis's target column. Every family
module (`histgb.py`, `ordinal.py`, `linear.py`, `catboost_model.py`, `lightgbm_model.py`)
imports from here so the "what counts as a feature" and "which rows get supervision"
rules (D12, D17) stay in exactly one place.
"""
from __future__ import annotations
import pandas as pd

from ..data_prep.scoring import SYSTEMS

AXES = tuple(SYSTEMS)   # ("resp", "coag", "liver", "cardio", "neuro", "renal")


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Every column of features.csv.gz except the join key. `origin_h` IS a feature (D12)."""
    return [c for c in features.columns if c != "RecordID"]


def axis_frame(features: pd.DataFrame, targets: pd.DataFrame, splits: pd.DataFrame,
               axis: str, split_name: str, target_col: str):
    """Rows for one axis / one split / one target column, keeping only measured rows
    (D17): a row is used for supervision only if `target_col` is non-null for it --
    never a lenient zero-fill (that fill is for `sofa_total` reporting, not training).

    `target_col` is typically `f"{axis}_delta"` (histgb/linear/catboost/lightgbm, which
    all predict the change from SOFA_now) or `f"{axis}_score"` (ordinal, which predicts
    the absolute target-window score directly -- see ordinal.py).
    """
    df = (features.merge(targets[["RecordID", "origin_h", target_col]],
                         on=["RecordID", "origin_h"])
                  .merge(splits[["RecordID", "split"]], on="RecordID"))
    df = df[df.split == split_name].dropna(subset=[target_col])
    cols = feature_columns(features)
    return df, df[cols], df[target_col].to_numpy()


def deterioration_frame(features: pd.DataFrame, targets: pd.DataFrame, splits: pd.DataFrame):
    """Rows for the (non-ordinal) `deteriorate_24h` binary outcome, split into
    train/val/test -- shared by every family that fits a deterioration classifier."""
    df = (features.merge(targets[["RecordID", "origin_h", "deteriorate_24h"]],
                         on=["RecordID", "origin_h"])
                  .merge(splits[["RecordID", "split"]], on="RecordID")
                  .dropna(subset=["deteriorate_24h"]))
    cols = feature_columns(features)
    return df, cols
