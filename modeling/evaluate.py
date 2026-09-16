"""Evaluation utilities (TRIPOD+AI items 12d, 12e, 23a, 23b).

Nothing here fits a model -- these functions score predictions that `models.py` produced
(or, before any model exists, score the persistence / last-value-per-axis baselines).
All performance numbers are reported with a bootstrap confidence interval (item 12d: "how
heterogeneity in ... model performance was handled"; item 23a: "report model performance
estimates with confidence intervals").
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             mean_absolute_error, mean_squared_error, roc_auc_score)

from . import config


def _mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = 1000, seed: int = config.SEED,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for an arbitrary metric_fn(y_true, y_pred)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n": n}


def regression_report(df: pd.DataFrame, y_col: str, pred_cols: dict,
                      group_col: str = "origin_h", n_boot: int = 1000) -> pd.DataFrame:
    """MAE + RMSE with bootstrap CI, per group, for each column in `pred_cols`
    ({label: column_name}) -- e.g. {"persistence": "sofa_now", "model": "sofa_total_pred"}.
    Used both for the Phase-2 baselines (before any model exists) and for the final
    model-vs-baseline comparison in Results (items 12e, 23a).
    """
    rows = []
    for g, d in df.groupby(group_col):
        for label, col in pred_cols.items():
            dd = d.dropna(subset=[y_col, col])
            if dd.empty:
                continue
            mae = bootstrap_ci(dd[y_col], dd[col], _mae, n_boot)
            rmse = bootstrap_ci(dd[y_col], dd[col], _rmse, n_boot)
            rows.append({group_col: g, "series": label, "n": mae["n"],
                        "mae": mae["point"], "mae_lo": mae["ci_lo"], "mae_hi": mae["ci_hi"],
                        "rmse": rmse["point"], "rmse_lo": rmse["ci_lo"], "rmse_hi": rmse["ci_hi"]})
    return pd.DataFrame(rows)


def classification_report_ci(y_true, y_prob, n_boot: int = 1000) -> dict:
    """AUROC, AUPRC, Brier score, each with a bootstrap CI (items 12e, 23a)."""
    return {name: bootstrap_ci(np.asarray(y_true), np.asarray(y_prob), fn, n_boot)
            for name, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score),
                             ("brier", brier_score_loss))}


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs mean-predicted event rate per decile of predicted probability."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    return (df.groupby("bin", observed=True)
              .agg(n=("y", "size"), mean_pred=("p", "mean"), observed_rate=("y", "mean"))
              .reset_index())


def subgroup_report(df: pd.DataFrame, y_col: str, pred_col: str, group_col: str,
                    n_boot: int = 500, min_n: int = 20) -> pd.DataFrame:
    """MAE with CI within subgroups (e.g. `cardio_instability`, `ICUType`, admission-SOFA
    tier) -- item 23b: heterogeneity of performance across subgroups."""
    rows = []
    for g, d in df.groupby(group_col):
        d = d.dropna(subset=[y_col, pred_col])
        if len(d) < min_n:
            continue
        r = bootstrap_ci(d[y_col], d[pred_col], _mae, n_boot)
        rows.append({group_col: g, "n": r["n"], "mae": r["point"],
                    "mae_lo": r["ci_lo"], "mae_hi": r["ci_hi"]})
    return pd.DataFrame(rows)


def permutation_report(model, X: pd.DataFrame, y, n_repeats: int = 3, n_sample: int = 2000,
                       seed: int = config.SEED, scoring=None, top_k: int = 20) -> pd.DataFrame:
    """D16: permutation importance on a fixed random subsample (not the full validation set)
    -- a deliberate compute/precision trade-off, logged rather than silently applied."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    Xs = X.iloc[idx]
    ys = y.iloc[idx] if hasattr(y, "iloc") else np.asarray(y)[idx]
    r = permutation_importance(model, Xs, ys, n_repeats=n_repeats, random_state=seed,
                               scoring=scoring)
    imp = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean,
                        "importance_std": r.importances_std})
    return imp.sort_values("importance_mean", ascending=False).head(top_k).reset_index(drop=True)
