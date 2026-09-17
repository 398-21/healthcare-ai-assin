"""Model family 5 of 5: LightGBM (D22). Optional dependency -- NOT installed in this
development environment; install on the training device with `pip install lightgbm`.

A second opinion alongside CatBoost: another histogram-based GBDT, but leaf-wise (best-first)
tree growth rather than HistGB's/CatBoost's more level-wise growth, and native missing-value
routing like HistGB (no imputation needed, unlike linear.py). Leaf-wise growth can fit sharper
patterns with fewer trees, at a higher risk of overfitting the smaller axes (D17) -- exactly
the kind of trade-off this multi-family comparison is designed to surface empirically rather
than assume.

Import is guarded the same way as catboost_model.py: `import modeling.models` must succeed
without lightgbm installed; every train_*/fit_* function raises a clear ImportError if called
without it.

IMPORTANT: written and import-checked (against the guard, not against lightgbm itself, which
isn't installed here) but never executed in this environment -- see histgb.py's module note.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from .. import config
from .common import AXES, feature_columns, axis_frame, deterioration_frame

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

REGRESSOR_GRID = [
    {"learning_rate": lr, "num_leaves": nl, "n_estimators": 300,
     "random_state": config.SEED, "verbosity": -1}
    for lr in (0.05, 0.1) for nl in (31, 63)
]
CLASSIFIER_GRID = [{**p, "class_weight": "balanced"} for p in REGRESSOR_GRID]


def _require_lightgbm():
    if not LIGHTGBM_AVAILABLE:
        raise ImportError(
            "lightgbm is not installed in this environment. Run `pip install lightgbm` on "
            "the training device, then re-import modeling.models before calling any "
            "*_lightgbm function.")


def fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid=REGRESSOR_GRID):
    """Same protocol as histgb.fit_grid_regressor: fit on train, select by validation MAE."""
    _require_lightgbm()
    rows, best = [], None
    for params in grid:
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr)
        val_mae = mean_absolute_error(y_val, m.predict(X_val))
        rows.append({**params, "val_mae": val_mae})
        if best is None or val_mae < best[1]:
            best = (m, val_mae, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_axis_models_lightgbm(features: pd.DataFrame, targets: pd.DataFrame,
                               splits: pd.DataFrame, axes=AXES, grid=REGRESSOR_GRID):
    """D22: one pooled LGBMRegressor per axis, predicting the same `{axis}_delta` target
    as histgb.train_axis_models."""
    _require_lightgbm()
    models, chosen, grids = {}, {}, {}
    for axis in axes:
        _, X_tr, y_tr = axis_frame(features, targets, splits, axis, "train", f"{axis}_delta")
        _, X_val, y_val = axis_frame(features, targets, splits, axis, "val", f"{axis}_delta")
        model, params, grid_df = fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid)
        models[axis], chosen[axis], grids[axis] = model, params, grid_df
    return models, chosen, grids


def fit_grid_classifier(X_tr, y_tr, X_val, y_val, grid=CLASSIFIER_GRID):
    """Same protocol as histgb.fit_grid_classifier: fit on train, select by validation AUPRC."""
    _require_lightgbm()
    rows, best = [], None
    for params in grid:
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        val_ap = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        rows.append({**params, "val_auprc": val_ap})
        if best is None or val_ap > best[1]:
            best = (m, val_ap, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_deterioration_model_lightgbm(features: pd.DataFrame, targets: pd.DataFrame,
                                       splits: pd.DataFrame, grid=CLASSIFIER_GRID):
    _require_lightgbm()
    df, cols = deterioration_frame(features, targets, splits)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    return fit_grid_classifier(tr[cols], tr.deteriorate_24h.astype(int),
                               va[cols], va.deteriorate_24h.astype(int), grid)


def predict_axis_scores_lightgbm(models: dict, features: pd.DataFrame) -> pd.DataFrame:
    """D24: falls back to 0 (not NaN) when `sofa_now_<axis>` is missing -- see
    histgb.predict_axis_scores's docstring for the full rationale."""
    cols = feature_columns(features)
    out = features[["RecordID", "origin_h"]].copy()
    for axis, model in models.items():
        pred_delta = model.predict(features[cols])
        now_axis = features[f"sofa_now_{axis}"].to_numpy()
        has_now = ~np.isnan(now_axis)
        out[f"{axis}_pred_delta_lightgbm"] = pred_delta
        out[f"{axis}_pred_lightgbm"] = np.where(has_now, np.clip(now_axis + pred_delta, 0, 4), 0.0)
    out["sofa_total_pred_lightgbm"] = sum(out[f"{a}_pred_lightgbm"] for a in models)
    return out


def predict_deterioration_lightgbm(model, features: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(features)
    proba = model.predict_proba(features[cols])[:, 1]
    return pd.DataFrame({"RecordID": features.RecordID.values,
                         "origin_h": features.origin_h.values,
                         "deteriorate_proba_lightgbm": proba})
