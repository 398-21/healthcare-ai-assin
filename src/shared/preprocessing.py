"""scikit-learn transformers shared by both tasks' model pipelines.

Every model in the project is a single `Pipeline` whose first step is a `FeatureFilter`, so
all column decisions are *fitted on the training rows only* and travel with the saved model
(no step ever sees validation or test data when it is fitted).

* `FeatureFilter`  -- unsupervised filter-type feature selection: drops columns that are
  constant, (almost) always missing, or near-duplicates of a more complete column.
* `ColumnSelector` -- keeps a fixed list of columns (used for embedded, model-based selection
  where the list is chosen beforehand by cross-validation on the training split).
* `linear_preprocessor` -- impute + missing-indicators + scale numeric columns and one-hot
  encode nominal ones, for models that cannot handle NaN or nominal codes themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureFilter(BaseEstimator, TransformerMixin):
    """Filter-type feature selection fitted on the training data only.

    A numeric column is dropped when, in the training rows,
      1. it has no variance (at most one distinct observed value), or
      2. at least `max_missing` of its values are missing, or
      3. its absolute Pearson correlation with an already-kept column is >= `corr_threshold`
         (columns are visited from most to least complete, so the more complete member of
         each near-duplicate pair is the one kept).
    Correlations are computed after median-filling, which can only lower the correlation of
    columns with different missingness patterns -- so the duplicate test is conservative.
    Non-numeric columns and the names in `protected` are always kept.

    The filter uses no labels, so it cannot leak outcome information.
    """

    def __init__(self, max_missing: float = 0.99, corr_threshold: float = 0.98, protected=()):
        self.max_missing = max_missing
        self.corr_threshold = corr_threshold
        self.protected = protected

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureFilter expects a pandas DataFrame")
        protected = set(self.protected)
        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in protected]
        Xn = X[numeric].astype(np.float64)
        miss = Xn.isna().mean()
        nunique = Xn.nunique(dropna=True)
        reasons = {}
        for c in numeric:
            if nunique[c] <= 1:
                reasons[c] = "constant"
            elif miss[c] >= self.max_missing:
                reasons[c] = f"missing >= {self.max_missing:.0%}"
        cand = [c for c in numeric if c not in reasons]
        if cand:
            order = sorted(range(len(cand)), key=lambda i: (miss[cand[i]], i))
            vals = Xn[cand].to_numpy()
            med = np.nanmedian(vals, axis=0)
            vals = np.where(np.isnan(vals), med, vals)
            sd = vals.std(0)
            vals = (vals - vals.mean(0)) / np.where(sd == 0, 1.0, sd)
            corr = np.abs(vals.T @ vals / len(vals))
            kept_idx: list[int] = []
            for i in order:
                if kept_idx and corr[i, kept_idx].max() >= self.corr_threshold:
                    j = kept_idx[int(np.argmax(corr[i, kept_idx]))]
                    reasons[cand[i]] = f"|r| >= {self.corr_threshold} with {cand[j]}"
                else:
                    kept_idx.append(i)
        self.drop_reasons_ = reasons
        self.keep_ = [c for c in X.columns if c not in reasons]
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_)

    def report(self) -> pd.DataFrame:
        """One row per dropped column with the reason, plus a summary count by reason."""
        df = pd.DataFrame({"feature": list(self.drop_reasons_), "reason": list(self.drop_reasons_.values())})
        df["rule"] = df.reason.str.split(" ").str[0].replace({"|r|": "near-duplicate", "missing": "mostly missing"})
        return df


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep a fixed, pre-chosen list of columns (embedded selection chosen by CV beforehand)."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"selected columns not in input: {missing[:5]}")
        return self

    def transform(self, X):
        return X[list(self.columns)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns)


class _Columns:
    """Picklable column selector for ColumnTransformer: the listed columns present in X
    (or, with `invert=True`, every other column)."""

    def __init__(self, columns, invert=False):
        self.columns, self.invert = list(columns), invert

    def __call__(self, X):
        if self.invert:
            return [c for c in X.columns if c not in self.columns]
        return [c for c in self.columns if c in X.columns]


def linear_preprocessor(nominal=()) -> ColumnTransformer:
    """For linear models: nominal columns -> most-frequent impute + one-hot; every other
    column -> median impute (+ a 0/1 missing indicator per column that had missing values
    in training) + standardisation. All statistics are fitted on the training data."""
    nominal = list(nominal)
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                             ("scale", StandardScaler())])
    steps = []
    if nominal:
        steps.append(("nominal", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                           ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                    sparse_output=False))]),
                      _Columns(nominal)))
    steps.append(("numeric", numeric_pipe, _Columns(nominal, invert=True)))
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


__all__ = ["FeatureFilter", "ColumnSelector", "linear_preprocessor", "make_column_selector"]
