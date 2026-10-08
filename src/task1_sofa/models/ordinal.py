"""Model family 2 of 5: ordinal regression via cumulative binary decomposition (D19).

Each axis's target-window score is an ORDINAL variable in {0,1,2,3,4} -- "2 is worse than 1
but the gap to 3 isn't necessarily the same size" -- not a free continuous number. `histgb.py`
treats it as plain regression on the delta (predict a real number, clip to [0,4]); this module
instead uses the standard ordinal-classification decomposition (Frank & Hall, "A Simple
Approach to Ordinal Classification", ECML 2001; also Li & Lin 2007): for a K=5-level ordinal
outcome, train K-1=4 binary classifiers, one per threshold k in {0,1,2,3}, where classifier
C_k predicts P(score > k).

Reconstruction uses the tail-sum identity for a bounded non-negative integer variable:

    E[score] = P(score>0) + P(score>1) + P(score>2) + P(score>3)

which follows from writing score = sum_{k=0}^{3} 1[score > k] and taking expectations. This
lands automatically in [0, 4] with no clipping step, and the four P(score>k) values also give
the full predicted class distribution (P(score=0)=1-P(score>0), P(score=1)=P(score>0)-P(score>1),
etc.) for a calibration check if wanted. No monotonicity constraint is enforced across the four
independently-trained classifiers -- a known, minor simplification of the method, logged rather
than silently assumed away.

Unlike `histgb.py`, this predicts the ABSOLUTE target-window score directly (SOFA_now and the
other engineered "current state" features are just ordinary input features here, not a
separate delta target) -- the delta trick was introduced specifically to make continuous
regression easier, and isn't needed once the outcome is decomposed into binary questions the
input features can answer directly ("is this patient's score already above 2 right now").

Compared head-to-head against `histgb.py` in the notebook's Results section, not used as a
replacement -- D19's rationale is "is a decomposition that respects ordinality better than
plain regression", which is an empirical question this comparison is designed to answer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .. import config
from .common import AXES, feature_columns, axis_frame, with_filter

THRESHOLDS = (0, 1, 2, 3)   # 0-4 ordinal target -> 4 cumulative thresholds

# Same shape/values as histgb.REGRESSOR_GRID, so any performance difference in the
# Results comparison reflects the framing (ordinal vs regression), not a different search.
GRID = [
    {"learning_rate": lr, "max_leaf_nodes": mln, "max_iter": 300, "early_stopping": False,
     "l2_regularization": 0.0, "random_state": config.SEED}
    for lr in (0.05, 0.1) for mln in (31, 63)
]


def fit_grid_threshold_classifier(X_tr, y_tr, X_val, y_val, grid=GRID):
    """Select by validation AUROC. Each threshold question ('is the score already above k')
    is a much less imbalanced binary problem than `deteriorate_24h` (thresholds near the
    middle of the 0-4 range are often close to balanced), so AUROC -- not AUPRC -- is the
    natural selection metric here; unlike histgb.fit_grid_classifier's D14 rationale."""
    rows, best = [], None
    for params in grid:
        m = with_filter(HistGradientBoostingClassifier(**params))
        m.fit(X_tr, y_tr)
        if len(np.unique(y_val)) < 2:
            val_auc = np.nan
        else:
            val_auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        rows.append({**params, "val_auroc": val_auc})
        if best is None or (not np.isnan(val_auc) and (np.isnan(best[1]) or val_auc > best[1])):
            best = (m, val_auc, params)
    return best[0], best[2], pd.DataFrame(rows)


def _threshold_target(features, targets, splits, axis, split_name, k):
    """Same row-selection rule as common.axis_frame (D17: measured rows only), but the
    label is the binary indicator score > k rather than the raw score/delta."""
    df, X, y_score = axis_frame(features, targets, splits, axis, split_name, f"{axis}_score")
    y = (y_score > k).astype(int)
    return df, X, y


def train_axis_models_ordinal(features: pd.DataFrame, targets: pd.DataFrame,
                              splits: pd.DataFrame, axes=AXES, thresholds=THRESHOLDS,
                              grid=GRID):
    """D19: for each axis, one HistGradientBoostingClassifier per threshold k.

    Returns (models: {axis: {k: fitted estimator}}, chosen_params: {axis: {k: dict}},
             grid_tables: {axis: {k: DataFrame of every grid point's val AUROC}}).
    """
    models, chosen, grids = {}, {}, {}
    for axis in axes:
        models[axis], chosen[axis], grids[axis] = {}, {}, {}
        for k in thresholds:
            _, X_tr, y_tr = _threshold_target(features, targets, splits, axis, "train", k)
            _, X_val, y_val = _threshold_target(features, targets, splits, axis, "val", k)
            m, params, grid_df = fit_grid_threshold_classifier(X_tr, y_tr, X_val, y_val, grid)
            models[axis][k], chosen[axis][k], grids[axis][k] = m, params, grid_df
    return models, chosen, grids


def predict_axis_scores_ordinal(models: dict, features: pd.DataFrame,
                                thresholds=THRESHOLDS) -> pd.DataFrame:
    """Reconstruct via E[score] = sum_k P(score > k) -- lands in [0,4] automatically,
    no clipping needed (item 12g, ordinal variant).

    D24: unlike the delta families, these classifiers never naturally abstain -- they'll
    happily guess P(score>k) from the other ~400 features even when `sofa_now_<axis>`
    itself is missing (that axis wasn't measured in the last 24 h). For a *fair* comparison
    against the delta families (D24 gates them to predict 0 in that situation, matching the
    lenient "missing axis -> 0" convention `sofa_total` uses, D10), the same gate is applied
    here: an ungated guess would systematically inflate the total for exactly the
    sparsely-measured patients where the true lenient total scores that axis 0. This does
    forfeit a genuine capability (predicting from indirect evidence with no direct anchor)
    in the headline comparison; that capability is not lost, just not exercised here -- an
    ablation without the gate is one obvious follow-up (open items).
    """
    cols = feature_columns(features)
    out = features[["RecordID", "origin_h"]].copy()
    for axis, axis_models in models.items():
        surv = np.zeros(len(features))
        for k in thresholds:
            p = axis_models[k].predict_proba(features[cols])[:, 1]
            out[f"{axis}_p_gt{k}_ordinal"] = p
            surv = surv + p
        now_axis = features[f"sofa_now_{axis}"].to_numpy()
        has_now = ~np.isnan(now_axis)
        out[f"{axis}_pred_ordinal"] = np.where(has_now, surv, 0.0)
    out["sofa_total_pred_ordinal"] = sum(out[f"{a}_pred_ordinal"] for a in models)
    return out
