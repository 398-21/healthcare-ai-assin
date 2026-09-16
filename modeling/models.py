"""Model development for the SOFA-forecasting task (TRIPOD+AI items 12b-12g, 13, 14, 15).

Six pooled axis regressors (D12, see SOFA_forecast_project_plan.docx decision log) predict
the per-axis DELTA (target-window score minus SOFA_now for that axis), trained on all four
horizons together with `origin_h` as an ordinary feature. Reconstruction adds SOFA_now back
and clips to [0, 4] (item 12g). One pooled deterioration classifier (D14) predicts
P(SOFA rises >= 2 within 24 h), `class_weight="balanced"` for the ~4 % positive rate (item 13).

Hyperparameters are chosen from a small fixed grid, scored on the *external* validation split
(D13) -- not HistGB's internal `early_stopping`, which would silently reuse training rows.

IMPORTANT: this module is written and import-checked but **not executed** in this environment
-- model.fit() is deliberately never called here. Training happens on a separate device by
running the "Model training" section of `sofa_forecasting.ipynb`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

from . import config
from .data_prep.scoring import SYSTEMS

AXES = tuple(SYSTEMS)   # ("resp", "coag", "liver", "cardio", "neuro", "renal")

# ----------------------------------------------------------------------- D13: fixed grids
REGRESSOR_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": mln, "max_iter": 300,
     "l2_regularization": 0.0, "random_state": config.SEED}
    for lr in (0.05, 0.1) for mln in (31, 63)
]
CLASSIFIER_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": mln, "max_iter": 300,
     "l2_regularization": 0.0, "class_weight": "balanced", "random_state": config.SEED}
    for lr in (0.05, 0.1) for mln in (31, 63)
]


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Every column of features.csv.gz except the join key. `origin_h` IS a feature (D12)."""
    return [c for c in features.columns if c != "RecordID"]


def _axis_frame(features: pd.DataFrame, targets: pd.DataFrame, splits: pd.DataFrame,
                axis: str, split_name: str):
    """Rows for one axis / one split, keeping only measured-target rows (D17): a row is
    used for supervision only if that axis actually had a value in the target window --
    never the lenient zero-fill (that fill is for `sofa_total` reporting, not for teaching
    the model that 'missing' means 'normal')."""
    df = (features.merge(targets[["RecordID", "origin_h", f"{axis}_delta"]],
                         on=["RecordID", "origin_h"])
                  .merge(splits[["RecordID", "split"]], on="RecordID"))
    df = df[df.split == split_name].dropna(subset=[f"{axis}_delta"])
    cols = feature_columns(features)
    return df, df[cols], df[f"{axis}_delta"].to_numpy()


def fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid=REGRESSOR_GRID):
    """D13: fit every grid point on train, score MAE on the external validation split,
    return the best model + its params + the full grid's scores (for transparent reporting,
    item 12c)."""
    rows, best = [], None
    for params in grid:
        m = HistGradientBoostingRegressor(**params)
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
        _, X_tr, y_tr = _axis_frame(features, targets, splits, axis, "train")
        _, X_val, y_val = _axis_frame(features, targets, splits, axis, "val")
        model, params, grid_df = fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid)
        models[axis], chosen[axis], grids[axis] = model, params, grid_df
    return models, chosen, grids


def fit_grid_classifier(X_tr, y_tr, X_val, y_val, grid=CLASSIFIER_GRID):
    """As `fit_grid_regressor`, but selects on validation AUPRC (item 13: the positive class
    -- deterioration -- is ~4 % of rows, so AUROC alone would be misleading)."""
    rows, best = [], None
    for params in grid:
        m = HistGradientBoostingClassifier(**params)
        m.fit(X_tr, y_tr)
        val_ap = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        rows.append({**params, "val_auprc": val_ap})
        if best is None or val_ap > best[1]:
            best = (m, val_ap, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_deterioration_model(features: pd.DataFrame, targets: pd.DataFrame,
                              splits: pd.DataFrame, grid=CLASSIFIER_GRID):
    """D14: one pooled classifier for `deteriorate_24h` (SOFA rises >= 2 within 24 h)."""
    df = (features.merge(targets[["RecordID", "origin_h", "deteriorate_24h"]],
                         on=["RecordID", "origin_h"])
                  .merge(splits[["RecordID", "split"]], on="RecordID")
                  .dropna(subset=["deteriorate_24h"]))
    cols = feature_columns(features)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    model, params, grid_df = fit_grid_classifier(
        tr[cols], tr.deteriorate_24h.astype(int),
        va[cols], va.deteriorate_24h.astype(int), grid)
    return model, params, grid_df


def predict_axis_scores(models: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Item 12g: how predictions are calculated at evaluation time.

    Absolute sub-score = SOFA_now(axis) [the `sofa_now_<axis>` feature, computed from our
    data over [T-24,T)] + predicted delta, clipped to [0, 4]. Total = sum of the six.
    """
    cols = feature_columns(features)
    out = features[["RecordID", "origin_h"]].copy()
    for axis, model in models.items():
        pred_delta = model.predict(features[cols])
        now_axis = features[f"sofa_now_{axis}"].to_numpy()
        out[f"{axis}_pred_delta"] = pred_delta
        out[f"{axis}_pred"] = np.clip(now_axis + pred_delta, 0, 4)
    out["sofa_total_pred"] = sum(out[f"{a}_pred"] for a in models)
    return out


def predict_deterioration(model, features: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(features)
    proba = model.predict_proba(features[cols])[:, 1]
    return pd.DataFrame({"RecordID": features.RecordID.values,
                         "origin_h": features.origin_h.values,
                         "deteriorate_proba": proba})
