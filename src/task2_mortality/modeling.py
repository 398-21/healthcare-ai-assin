"""Task 2 model pipelines, class-imbalance strategies and embedded feature selection.

Every model is an (imbalanced-learn) `Pipeline`, so imputation, scaling and any resampling
are fitted on the rows the pipeline is trained on -- inside each cross-validation fold during
model development, never on validation or test rows.

Class-imbalance strategies (deaths are 14.4 % of stays):
  "none"         plain fit
  "class_weight" re-weight the loss (LR: class_weight="balanced"; XGBoost: scale_pos_weight)
  "smote"        median-impute, then SMOTE synthetic minority oversampling
  "undersample"  randomly drop survivors until classes balance
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from ..shared.preprocessing import ColumnSelector, FeatureFilter
from ..shared.reproducibility import SEED

IMBALANCE_STRATEGIES = ("none", "class_weight", "smote", "undersample")
SCORING = {"AUROC": "roc_auc", "AUPRC": "average_precision", "Brier": "neg_brier_score"}


def feature_filter() -> FeatureFilter:
    return FeatureFilter(max_missing=0.99, corr_threshold=0.98)


def _resampling_steps(imbalance: str) -> list:
    if imbalance == "smote":
        return [("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
                ("sampler", SMOTE(random_state=SEED))]
    if imbalance == "undersample":
        return [("sampler", RandomUnderSampler(random_state=SEED))]
    return []


def xgb_pipeline(params: dict, imbalance: str, spw: float, columns=None, use_filter=True,
                 device: str = "cpu") -> ImbPipeline:
    """[filter] -> [fixed column subset] -> [resampling] -> XGBoost (native NaN handling)."""
    if imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError(imbalance)
    p = dict(objective="binary:logistic", tree_method="hist", random_state=SEED, n_jobs=-1,
             device=device, **params)
    p["scale_pos_weight"] = spw if imbalance == "class_weight" else 1.0
    steps = [("filter", feature_filter())] if use_filter else []
    if columns is not None:
        steps.append(("select", ColumnSelector(list(columns))))
    steps += _resampling_steps(imbalance)
    steps.append(("model", xgb.XGBClassifier(**p)))
    return ImbPipeline(steps)


def logistic_pipeline(features, imbalance: str = "class_weight", C: float = 1.0) -> ImbPipeline:
    """Column subset -> median impute -> standardise -> [resampling] -> L2 logistic regression."""
    steps = [("select", ColumnSelector(list(features))),
             ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
             ("scale", StandardScaler().set_output(transform="pandas"))]
    if imbalance == "smote":
        steps.append(("sampler", SMOTE(random_state=SEED)))
    elif imbalance == "undersample":
        steps.append(("sampler", RandomUnderSampler(random_state=SEED)))
    steps.append(("model", LogisticRegression(C=C, max_iter=1000, random_state=SEED,
                                              class_weight="balanced" if imbalance == "class_weight" else None)))
    return ImbPipeline(steps)


def cv_folds(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def cv_evaluate(pipeline, X, y, n_splits: int = 5) -> dict:
    """Mean and SD over stratified folds of the training split (resampling stays inside folds)."""
    r = cross_validate(pipeline, X, y, cv=cv_folds(n_splits), scoring=SCORING, n_jobs=1)
    out = {}
    for name in SCORING:
        v = r[f"test_{name}"] * (-1 if name == "Brier" else 1)
        out[f"{name} mean"], out[f"{name} sd"] = float(v.mean()), float(v.std())
    return out


def imbalance_experiment(make_pipeline, X, y, strategies=IMBALANCE_STRATEGIES) -> pd.DataFrame:
    """`make_pipeline(strategy)` -> pipeline; returns one row of CV metrics per strategy."""
    return pd.DataFrame({s: cv_evaluate(make_pipeline(s), X, y) for s in strategies}).T


def shap_ranking(model: xgb.XGBClassifier, X: pd.DataFrame) -> list[str]:
    """Columns ordered by mean |SHAP| (exact TreeSHAP via XGBoost's pred_contribs)."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    order = np.argsort(-np.abs(contrib).mean(0))
    return [X.columns[i] for i in order]


def topk_nested_cv(X: pd.DataFrame, y, params: dict, imbalance: str, spw: float, ks,
                   device: str = "cpu", n_splits: int = 5) -> pd.DataFrame:
    """Embedded selection with nested ranking: in every fold the SHAP ranking is learned on the
    fold's training part only, then XGBoost is refitted on the top-k columns and scored on the
    fold's held-out part. Returns one row per (fold, k)."""
    y = np.asarray(y)
    rows = []
    for fold, (tr, va) in enumerate(cv_folds(n_splits).split(X, y)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        full = xgb_pipeline(params, imbalance, spw, use_filter=False, device=device).fit(Xtr, y[tr])
        ranked = shap_ranking(full.named_steps["model"], Xtr)
        for k in ks:
            cols = ranked[:k] if k else ranked
            m = xgb_pipeline(params, imbalance, spw, columns=cols, use_filter=False, device=device).fit(Xtr, y[tr])
            p = m.predict_proba(Xva)[:, 1]
            rows.append({"fold": fold, "k": len(cols), "AUROC": roc_auc_score(y[va], p),
                         "AUPRC": average_precision_score(y[va], p)})
    return pd.DataFrame(rows)


def one_se_choice(cv_table: pd.DataFrame, metric: str = "AUPRC") -> tuple[int, pd.DataFrame]:
    """Smallest k whose mean CV score is within one standard error of the best k's mean."""
    g = cv_table.groupby("k")[metric].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    best = g["mean"].idxmax()
    threshold = g.loc[best, "mean"] - g.loc[best, "se"]
    chosen = int(g.index[g["mean"] >= threshold].min())
    return chosen, g.assign(within_1se=g["mean"] >= threshold)
