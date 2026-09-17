"""Model family 3 of 5: simple linear / logistic baselines (D20).

Not intended to be shipped -- intended to answer a real question: does the tree ensemble's
non-linearity and interaction-modelling actually buy anything over a plain regularized linear
model? Uses the SAME delta target and SAME feature set as `histgb.py` (predict
`{axis}_delta`, reconstruct via `sofa_now_<axis> + prediction`, clipped to [0,4]) so the
comparison isolates model family, holding the target framing constant -- change one variable
at a time.

Ridge / LogisticRegression cannot route missing values internally the way HistGB can, so this
module adds a median-impute + standardize step that the tree-based families don't need. That is
an honest, logged difference (D20): any performance gap partly reflects "native missingness
handling" as well as "linear vs non-linear", and the two effects are not separated by this
comparison alone.

IMPORTANT: written and import-checked, never executed here -- see histgb.py's module note.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from .common import AXES, feature_columns, axis_frame, deterioration_frame

RIDGE_GRID = [{"alpha": a} for a in (0.1, 1.0, 10.0)]
LOGISTIC_GRID = [{"C": c, "class_weight": "balanced", "max_iter": 2000} for c in (0.1, 1.0, 10.0)]


def _ridge_pipeline(**params):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        Ridge(random_state=config.SEED, **params))


def _logistic_pipeline(**params):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(random_state=config.SEED, **params))


def fit_grid_ridge(X_tr, y_tr, X_val, y_val, grid=RIDGE_GRID):
    """Same protocol as histgb.fit_grid_regressor: fit on train, select by validation MAE."""
    rows, best = [], None
    for params in grid:
        m = _ridge_pipeline(**params)
        m.fit(X_tr, y_tr)
        val_mae = mean_absolute_error(y_val, m.predict(X_val))
        rows.append({**params, "val_mae": val_mae})
        if best is None or val_mae < best[1]:
            best = (m, val_mae, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_axis_models_linear(features: pd.DataFrame, targets: pd.DataFrame,
                             splits: pd.DataFrame, axes=AXES, grid=RIDGE_GRID):
    """D20: one pooled (median-impute + standardize + Ridge) pipeline per axis, predicting
    the same `{axis}_delta` target as histgb.train_axis_models."""
    models, chosen, grids = {}, {}, {}
    for axis in axes:
        _, X_tr, y_tr = axis_frame(features, targets, splits, axis, "train", f"{axis}_delta")
        _, X_val, y_val = axis_frame(features, targets, splits, axis, "val", f"{axis}_delta")
        model, params, grid_df = fit_grid_ridge(X_tr, y_tr, X_val, y_val, grid)
        models[axis], chosen[axis], grids[axis] = model, params, grid_df
    return models, chosen, grids


def fit_grid_logistic(X_tr, y_tr, X_val, y_val, grid=LOGISTIC_GRID):
    """Same protocol as histgb.fit_grid_classifier: fit on train, select by validation AUPRC."""
    rows, best = [], None
    for params in grid:
        m = _logistic_pipeline(**params)
        m.fit(X_tr, y_tr)
        val_ap = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        rows.append({**params, "val_auprc": val_ap})
        if best is None or val_ap > best[1]:
            best = (m, val_ap, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_deterioration_model_linear(features: pd.DataFrame, targets: pd.DataFrame,
                                     splits: pd.DataFrame, grid=LOGISTIC_GRID):
    """D20: class-weighted logistic regression baseline for `deteriorate_24h`."""
    df, cols = deterioration_frame(features, targets, splits)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    return fit_grid_logistic(tr[cols], tr.deteriorate_24h.astype(int),
                             va[cols], va.deteriorate_24h.astype(int), grid)


def predict_axis_scores_linear(models: dict, features: pd.DataFrame) -> pd.DataFrame:
    """D24: falls back to 0 (not NaN) when `sofa_now_<axis>` is missing -- see
    histgb.predict_axis_scores's docstring for the full rationale."""
    cols = feature_columns(features)
    out = features[["RecordID", "origin_h"]].copy()
    for axis, model in models.items():
        pred_delta = model.predict(features[cols])
        now_axis = features[f"sofa_now_{axis}"].to_numpy()
        has_now = ~np.isnan(now_axis)
        out[f"{axis}_pred_delta_linear"] = pred_delta
        out[f"{axis}_pred_linear"] = np.where(has_now, np.clip(now_axis + pred_delta, 0, 4), 0.0)
    out["sofa_total_pred_linear"] = sum(out[f"{a}_pred_linear"] for a in models)
    return out


def predict_deterioration_linear(model, features: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(features)
    proba = model.predict_proba(features[cols])[:, 1]
    return pd.DataFrame({"RecordID": features.RecordID.values,
                         "origin_h": features.origin_h.values,
                         "deteriorate_proba_linear": proba})
