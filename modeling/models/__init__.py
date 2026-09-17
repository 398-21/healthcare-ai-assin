"""Model development for the SOFA-forecasting task (TRIPOD+AI items 12b-12g, 13, 14, 15).

Five model families are provided, compared head-to-head in the notebook's Results section
against each other and the Section 8 baselines -- not because we expect to ship all five, but
because "does complexity/framing help, and by how much" is itself a result worth reporting
(professional practice; user-directed 2026-09-16, in this priority order):

  0. models.ordinal        -- D19: the six axes reframed as ORDINAL regression via
                               Frank & Hall (2001) cumulative binary decomposition, predicting
                               the ABSOLUTE target score directly (no delta trick).
  1. models.linear         -- D20: Ridge / class-weighted LogisticRegression baselines, same
                               delta target as (2), to isolate "does non-linearity help".
  2. models.histgb         -- D12/D13/D14/D17: six pooled HistGB regressors (predict the
                               DELTA) + one pooled HistGB deterioration classifier.
                               THE DEFAULT / PRIMARY family; un-suffixed prediction columns.
  3. models.catboost_model -- D21: CatBoost (optional dependency, guarded import).
  4. models.lightgbm_model -- D22: LightGBM (optional dependency, guarded import).

Every family exposes analogous function names (train_axis_models_<suffix>,
train_deterioration_model_<suffix>, predict_axis_scores_<suffix>) and writes its predicted
columns with that suffix (`_ordinal`, `_linear`, `_catboost`, `_lightgbm`), so the notebook can
merge every family's predictions into one wide table without name collisions. Family 2
(histgb) keeps its original UN-suffixed names (train_axis_models, train_deterioration_model,
predict_axis_scores, predict_deterioration, REGRESSOR_GRID, CLASSIFIER_GRID) for backward
compatibility with the already-executed cells of sofa_forecasting.ipynb.

IMPORTANT: as with every family module, nothing here is executed as part of the notebook
build in this environment -- model.fit() is deliberately never called. Training happens on a
separate device by running the notebook's "Model training" section.
"""
from .common import AXES, axis_frame, deterioration_frame, feature_columns
from .histgb import (
    CLASSIFIER_GRID,
    REGRESSOR_GRID,
    fit_grid_classifier,
    fit_grid_regressor,
    predict_axis_scores,
    predict_deterioration,
    train_axis_models,
    train_deterioration_model,
)
from . import catboost_model, histgb, lightgbm_model, linear, ordinal

FAMILIES = ("ordinal", "linear", "histgb", "catboost_model", "lightgbm_model")

AVAILABLE_FAMILIES = {
    "ordinal": True,
    "linear": True,
    "histgb": True,
    "catboost_model": catboost_model.CATBOOST_AVAILABLE,
    "lightgbm_model": lightgbm_model.LIGHTGBM_AVAILABLE,
}
