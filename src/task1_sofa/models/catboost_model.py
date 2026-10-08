"""Model family 4 of 5: CatBoost (D21). Optional dependency (`pip install catboost`).

Two reasons this is worth a head-to-head comparison against `histgb.py` rather than just
tuning HistGB harder: (1) native categorical handling for `d_icutype` / `d_gender` /
`origin_h` instead of treating small-cardinality codes as raw numbers, and (2) "ordered
boosting" (CatBoost's default for datasets this size), designed specifically to reduce the
overfitting-from-target-leakage that plain (unordered) gradient boosting can suffer on
smaller training sets -- relevant here since some axes (e.g. liver, ~13,500 rows) train on
far fewer rows than others (D17).

Import is guarded: `import src.task1_sofa.models` must succeed on a machine without
catboost installed. Every train_*/fit_* function raises a clear ImportError if actually
called without the package -- never a silent no-op.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from .. import config
from .common import AXES, feature_columns, axis_frame, sofa_rise_frame, with_filter

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Small-cardinality columns passed to CatBoost as native categoricals rather than raw numbers.
CAT_FEATURE_NAMES = ["d_icutype", "d_gender", "origin_h"]

REGRESSOR_GRID = [
    {"learning_rate": lr, "depth": depth, "iterations": 300,
     "random_seed": config.SEED, "verbose": False,
     "allow_writing_files": False}       # no stray catboost_info/ training-log folders
    for lr in (0.05, 0.1) for depth in (4, 6)
]
CLASSIFIER_GRID = [{**p, "auto_class_weights": "Balanced"} for p in REGRESSOR_GRID]


def _require_catboost():
    if not CATBOOST_AVAILABLE:
        raise ImportError(
            "catboost is not installed in this environment. Run `pip install catboost`, "
            "then re-import src.task1_sofa.models before calling any "
            "*_catboost function.")


def _cat_feature_names(X: pd.DataFrame) -> list[str]:
    # passed by NAME: the FeatureFilter in front of CatBoost changes column positions
    return [c for c in CAT_FEATURE_NAMES if c in X.columns]


def _prepare_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost requires `cat_features` columns to be int or str -- NOT float/NaN. Passing
    the raw float64 columns straight out of pandas (e.g. `origin_h` = 24.0) raises
    `CatBoostError: ... real number values and NaN values should be converted to string`,
    even though the values are small integer codes. Missing values become their own
    explicit category ('-1') rather than being silently dropped. Applied identically at fit
    time and predict time (same function, same column set) so the model is never asked to
    score a representation it wasn't trained on."""
    X = X.copy()
    for c in CAT_FEATURE_NAMES:
        if c in X.columns:
            X[c] = X[c].fillna(-1).astype(int).astype(str)
    return X


def fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid=REGRESSOR_GRID):
    """Same protocol as histgb.fit_grid_regressor: fit on train, select by validation MAE."""
    _require_catboost()
    X_tr = _prepare_categoricals(X_tr)
    X_val = _prepare_categoricals(X_val)
    cat_names = _cat_feature_names(X_tr)
    rows, best = [], None
    for params in grid:
        m = with_filter(CatBoostRegressor(**params))
        m.fit(X_tr, y_tr, model__cat_features=cat_names)
        val_mae = mean_absolute_error(y_val, m.predict(X_val))
        rows.append({**params, "val_mae": val_mae})
        if best is None or val_mae < best[1]:
            best = (m, val_mae, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_axis_models_catboost(features: pd.DataFrame, targets: pd.DataFrame,
                               splits: pd.DataFrame, axes=AXES, grid=REGRESSOR_GRID):
    """D21: one pooled CatBoostRegressor per axis, predicting the same `{axis}_delta`
    target as histgb.train_axis_models, with d_icutype/d_gender/origin_h as native
    categoricals."""
    _require_catboost()
    models, chosen, grids = {}, {}, {}
    for axis in axes:
        _, X_tr, y_tr = axis_frame(features, targets, splits, axis, "train", f"{axis}_delta")
        _, X_val, y_val = axis_frame(features, targets, splits, axis, "val", f"{axis}_delta")
        model, params, grid_df = fit_grid_regressor(X_tr, y_tr, X_val, y_val, grid)
        models[axis], chosen[axis], grids[axis] = model, params, grid_df
    return models, chosen, grids


def fit_grid_classifier(X_tr, y_tr, X_val, y_val, grid=CLASSIFIER_GRID):
    """Same protocol as histgb.fit_grid_classifier: fit on train, select by validation AUPRC."""
    _require_catboost()
    X_tr = _prepare_categoricals(X_tr)
    X_val = _prepare_categoricals(X_val)
    cat_names = _cat_feature_names(X_tr)
    rows, best = [], None
    for params in grid:
        m = with_filter(CatBoostClassifier(**params))
        m.fit(X_tr, y_tr, model__cat_features=cat_names)
        val_ap = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        rows.append({**params, "val_auprc": val_ap})
        if best is None or val_ap > best[1]:
            best = (m, val_ap, params)
    return best[0], best[2], pd.DataFrame(rows)


def train_sofa_rise_model_catboost(features: pd.DataFrame, targets: pd.DataFrame,
                                       splits: pd.DataFrame, grid=CLASSIFIER_GRID):
    _require_catboost()
    df, cols = sofa_rise_frame(features, targets, splits)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    return fit_grid_classifier(tr[cols], tr.sofa_rise_ge2_tplus6.astype(int),
                               va[cols], va.sofa_rise_ge2_tplus6.astype(int), grid)


def predict_axis_scores_catboost(models: dict, features: pd.DataFrame) -> pd.DataFrame:
    """D24: falls back to 0 (not NaN) when `sofa_now_<axis>` is missing -- see
    histgb.predict_axis_scores's docstring for the full rationale."""
    cols = feature_columns(features)
    X = _prepare_categoricals(features[cols])
    out = features[["RecordID", "origin_h"]].copy()
    for axis, model in models.items():
        pred_delta = model.predict(X)
        now_axis = features[f"sofa_now_{axis}"].to_numpy()
        has_now = ~np.isnan(now_axis)
        out[f"{axis}_pred_delta_catboost"] = pred_delta
        out[f"{axis}_pred_catboost"] = np.where(has_now, np.clip(now_axis + pred_delta, 0, 4), 0.0)
    out["sofa_24h_tplus6_pred_catboost"] = sum(out[f"{a}_pred_catboost"] for a in models)
    return out


def predict_sofa_rise_catboost(model, features: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(features)
    X = _prepare_categoricals(features[cols])
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"RecordID": features.RecordID.values,
                         "origin_h": features.origin_h.values,
                         "sofa_rise_ge2_proba_catboost": proba})
