"""Ablation, not a sixth headline family: HistGB predicting the ABSOLUTE target-window
score directly, instead of the delta from SOFA_now that histgb.py (and linear/catboost/
lightgbm) predict.

Motivation: the ordinal family (ordinal.py) is the only one of the five headline families
that predicts the absolute score directly -- the other four predict a delta and reconstruct
via SOFA_now + delta. Ordinal also has the lowest total-SOFA MAE at every horizon (see
SOFA_forecasting_research_report.docx). That leaves a genuine confound: is ordinal's
advantage coming from its ordinal cumulative-binary decomposition, or simply from predicting
the absolute target directly rather than through the delta trick? Those are different
scientific claims, and the five-family comparison alone cannot separate them, because
ordinal is the only family varying both things (formulation AND target) at once.

This module holds that confound's control condition constant: same base learner
(HistGradientBoostingRegressor), same grid, same six axes, same protocol as histgb.py's
primary family (D12/D13) -- the ONLY difference is the target column, f"{axis}_score"
(absolute) instead of f"{axis}_delta". Comparing HistGB+delta vs HistGB+absolute vs
Ordinal+absolute lets the target-formulation effect and the ordinal-decomposition effect be
read apart from each other.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .common import AXES, feature_columns, axis_frame
from .histgb import REGRESSOR_GRID, fit_grid_regressor


def train_axis_models_absolute(features: pd.DataFrame, targets: pd.DataFrame,
                               splits: pd.DataFrame, axes=AXES, grid=REGRESSOR_GRID):
    """Identical to histgb.train_axis_models, except the target column is f"{axis}_score"
    (the absolute target-window score, same measured-rows-only rule D17) instead of
    f"{axis}_delta". Returns (models, chosen_params, grid_tables), same shape as
    histgb.train_axis_models."""
    models, chosen, grids = {}, {}, {}
    for axis in axes:
        _, X_tr, y_tr = axis_frame(features, targets, splits, axis, "train", f"{axis}_score")
        _, X_val, y_val = axis_frame(features, targets, splits, axis, "val", f"{axis}_score")
        model, params, grid_df = fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid)
        models[axis], chosen[axis], grids[axis] = model, params, grid_df
    return models, chosen, grids


def predict_axis_scores_absolute(models: dict, features: pd.DataFrame) -> pd.DataFrame:
    """No SOFA_now anchor is needed to reconstruct a score here -- that is the entire point
    of this ablation -- just clip the raw prediction to [0,4]. The same has_now fallback-to-0
    gate used by every headline family (D24) is still applied, so this ablation is scored on
    the identical population as the five-family comparison, not a larger/easier one."""
    cols = feature_columns(features)
    out = features[["RecordID", "origin_h"]].copy()
    for axis, model in models.items():
        pred = np.clip(model.predict(features[cols]), 0, 4)
        now_axis = features[f"sofa_now_{axis}"].to_numpy()
        has_now = ~np.isnan(now_axis)
        out[f"{axis}_pred_abs"] = np.where(has_now, pred, 0.0)
    out["sofa_total_pred_abs"] = sum(out[f"{a}_pred_abs"] for a in models)
    return out
