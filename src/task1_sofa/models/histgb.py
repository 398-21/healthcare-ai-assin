"""Model family 1 of 5 (THE DEFAULT / PRIMARY family): pooled HistGradientBoosting
(TRIPOD+AI items 12b-12g, 13, 14, 15).

Six pooled axis regressors (D12, see SOFA_forecast_project_plan.docx decision log) predict
the per-axis DELTA (target-window score minus SOFA_now for that axis), trained on all four
horizons together with `origin_h` as an ordinary feature. Reconstruction adds SOFA_now back
and clips to [0, 4] (item 12g). One pooled rolling 24-hour proxy-SOFA ≥2-point rise classifier (D14) predicts
P(proxy-SOFA over [T-18,T+6) minus proxy-SOFA over [T-24,T) is >=2), `class_weight="balanced"` for the ~4 % positive rate (item 13).

Hyperparameters are chosen from a small fixed grid, scored on the *external* validation split
(D13) -- not HistGB's internal `early_stopping`, which would silently reuse training rows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

from .. import config
from .common import (AXES, feature_columns, axis_frame, sofa_rise_frame, with_filter,
                     classifier_pipeline)

# ----------------------------------------------------------------------- D13: fixed grids
# early_stopping=False is explicit: sklearn's default ("auto") switches early stopping ON
# above 10,000 rows and silently holds out 10 % of the TRAINING rows -- exactly what D13 rules out.
REGRESSOR_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": mln, "max_iter": 300, "early_stopping": False,
     "l2_regularization": 0.0, "random_state": config.SEED}
    for lr in (0.05, 0.1) for mln in (31, 63)
]
CLASSIFIER_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": mln, "max_iter": 300, "early_stopping": False,
     "l2_regularization": 0.0, "class_weight": "balanced", "random_state": config.SEED}
    for lr in (0.05, 0.1) for mln in (31, 63)
]


def fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid=REGRESSOR_GRID):
    """D13: fit every grid point on train, score MAE on the external validation split,
    return the best model + its params + the full grid's scores (for transparent reporting,
    item 12c)."""
    rows, best = [], None
    for params in grid:
        m = with_filter(HistGradientBoostingRegressor(**params))
        m.fit(X_tr, y_tr)
        val_mae = mean_absolute_error(y_val, m.predict(X_val))
        rows.append({**params, "val_mae": val_mae})
        if best is None or val_mae < best[1]:
            best = (m, val_mae, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_axis_models(features: pd.DataFrame, targets: pd.DataFrame, splits: pd.DataFrame,
                      axes=AXES, grid=REGRESSOR_GRID):
    """D12: one pooled (all-horizon) HistGradientBoostingRegressor per organ axis.

    Returns (models: {axis: fitted estimator}, chosen_params: {axis: dict},
             grid_tables: {axis: DataFrame of every grid point's val MAE}).
    """
    models, chosen, grids = {}, {}, {}
    for axis in axes:
        _, X_tr, y_tr = axis_frame(features, targets, splits, axis, "train", f"{axis}_delta")
        _, X_val, y_val = axis_frame(features, targets, splits, axis, "val", f"{axis}_delta")
        model, params, grid_df = fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid)
        models[axis], chosen[axis], grids[axis] = model, params, grid_df
    return models, chosen, grids


def fit_grid_classifier(X_tr, y_tr, X_val, y_val, grid=CLASSIFIER_GRID, imbalance="class_weight"):
    """As `fit_grid_regressor`, but selects on validation AUPRC (item 13: the positive class
    -- rolling 24-hour proxy-SOFA ≥2-point rise label -- is ~4 % of rows, so AUROC alone would be misleading). `imbalance`
    picks the class-imbalance strategy (see common.classifier_pipeline)."""
    rows, best = [], None
    for params in grid:
        m = classifier_pipeline(HistGradientBoostingClassifier, params, imbalance)
        m.fit(X_tr, y_tr)
        val_ap = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        rows.append({**params, "imbalance": imbalance, "val_auprc": val_ap})
        if best is None or val_ap > best[1]:
            best = (m, val_ap, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_sofa_rise_model(features: pd.DataFrame, targets: pd.DataFrame,
                              splits: pd.DataFrame, grid=CLASSIFIER_GRID, imbalance="class_weight"):
    """D14: one pooled classifier for `sofa_rise_ge2_tplus6` (proxy-SOFA over [T-18,T+6) minus proxy-SOFA over [T-24,T) is >=2)."""
    df, cols = sofa_rise_frame(features, targets, splits)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    model, params, grid_df = fit_grid_classifier(
        tr[cols], tr.sofa_rise_ge2_tplus6.astype(int),
        va[cols], va.sofa_rise_ge2_tplus6.astype(int), grid, imbalance)
    return model, params, grid_df


def predict_axis_scores(models: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Item 12g: how predictions are calculated at evaluation time.

    Absolute sub-score = SOFA_now(axis) [the `sofa_now_<axis>` feature, computed from our
    data over [T-24,T)] + predicted delta, clipped to [0, 4]. Total = sum of the six.

    D24: when `sofa_now_<axis>` is missing (that axis had no measurement in the last
    fully-observed 24 h -- true for 0.9-60% of rows depending on axis, see decision log),
    there is no anchor to add the delta to. The axis prediction falls back to 0, mirroring
    the lenient "missing axis -> 0" convention `sofa_total` itself uses (D10) -- NOT NaN.
    NaN would silently propagate into `sofa_24h_tplus6_pred` and, because only ~22% of rows have
    all six `sofa_now_<axis>` present simultaneously, would restrict every downstream
    comparison to that easier complete-case subset without anyone noticing (found only once
    this was actually run end-to-end, D24).
    """
    cols = feature_columns(features)
    out = features[["RecordID", "origin_h"]].copy()
    for axis, model in models.items():
        pred_delta = model.predict(features[cols])
        now_axis = features[f"sofa_now_{axis}"].to_numpy()
        has_now = ~np.isnan(now_axis)
        out[f"{axis}_pred_delta"] = pred_delta
        out[f"{axis}_pred"] = np.where(has_now, np.clip(now_axis + pred_delta, 0, 4), 0.0)
    out["sofa_24h_tplus6_pred"] = sum(out[f"{a}_pred"] for a in models)
    return out


def predict_sofa_rise(model, features: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(features)
    proba = model.predict_proba(features[cols])[:, 1]
    return pd.DataFrame({"RecordID": features.RecordID.values,
                         "origin_h": features.origin_h.values,
                         "sofa_rise_ge2_proba": proba})
