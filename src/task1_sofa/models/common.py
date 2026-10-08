"""Shared plumbing for every model family: which columns are features, and how to
build a (train|val|test) design matrix for one axis's target column. Every family
module (`histgb.py`, `ordinal.py`, `linear.py`, `catboost_model.py`, `lightgbm_model.py`)
imports from here so the "what counts as a feature" and "which rows get supervision"
rules (D12, D17) stay in exactly one place.
"""
from __future__ import annotations
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .. import config
from ..data_prep.scoring import SYSTEMS
from ...shared.preprocessing import FeatureFilter

AXES = tuple(SYSTEMS)   # ("resp", "coag", "liver", "cardio", "neuro", "renal")

# Columns the filter must never drop: the decision time, the nominal descriptors (CatBoost
# categoricals / one-hot in the linear family) and each organ's SOFA_now anchor.
PROTECTED = ("origin_h", "d_icutype", "d_gender") + tuple(f"sofa_now_{a}" for a in AXES)
IMBALANCE_STRATEGIES = ("none", "class_weight", "smote", "undersample")


def feature_filter() -> FeatureFilter:
    return FeatureFilter(max_missing=0.99, corr_threshold=0.98, protected=PROTECTED)


def with_filter(estimator, *middle) -> Pipeline:
    """Filter (fitted on the model's own training rows) -> optional steps -> estimator."""
    return Pipeline([("filter", feature_filter()), *middle, ("model", estimator)])


def classifier_pipeline(make_estimator, params: dict, imbalance: str = "class_weight",
                        weight_key: str = "class_weight", weight_value="balanced") -> ImbPipeline:
    """A classifier pipeline under one class-imbalance strategy:

    * "class_weight" -- re-weight the loss (`weight_key=weight_value`), no resampling;
    * "none"         -- plain fit;
    * "smote"        -- median-impute, then SMOTE synthetic minority oversampling;
    * "undersample"  -- randomly drop majority-class rows.
    Resampling lives INSIDE the pipeline, so it only ever touches the rows the pipeline is
    being fitted on; validation and test rows are never resampled.
    """
    if imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError(imbalance)
    p = {k: v for k, v in params.items() if k != weight_key}
    if imbalance == "class_weight":
        p[weight_key] = weight_value
    steps = [("filter", feature_filter())]
    if imbalance == "smote":
        steps += [("impute", SimpleImputer(strategy="median")), ("sampler", SMOTE(random_state=config.SEED))]
    elif imbalance == "undersample":
        steps += [("sampler", RandomUnderSampler(random_state=config.SEED))]
    steps.append(("model", make_estimator(**p)))
    return ImbPipeline(steps)


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


def sofa_rise_frame(features: pd.DataFrame, targets: pd.DataFrame, splits: pd.DataFrame):
    """Rows for the (non-ordinal) `sofa_rise_ge2_tplus6` binary outcome, split into
    train/val/test -- shared by every family that fits a rolling 24-hour proxy-SOFA ≥2-point rise classifier."""
    df = (features.merge(targets[["RecordID", "origin_h", "sofa_rise_ge2_tplus6"]],
                         on=["RecordID", "origin_h"])
                  .merge(splits[["RecordID", "split"]], on="RecordID")
                  .dropna(subset=["sofa_rise_ge2_tplus6"]))
    cols = feature_columns(features)
    return df, cols
