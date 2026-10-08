"""Write `results/hyperparameter_tuning_log.md`: the full hyperparameter-tuning record.

Every configuration that was tried is listed with the score it was selected on (validation or
cross-validation on the training split -- never test), and the selected one is marked. Built by
the notebook's final cell from the tuning records it saved, so the log always matches the run.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

VARIED = ("learning_rate", "max_leaf_nodes", "depth", "num_leaves", "alpha", "C")
AXIS_NAME = {"resp": "Respiratory", "coag": "Coagulation", "liver": "Liver",
             "cardio": "Cardiovascular", "neuro": "Neurological", "renal": "Renal"}


def _fmt(v, nd=4):
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.{nd}f}" if abs(v) < 1e5 else f"{v:.3g}"
    return str(v)


def md_table(df: pd.DataFrame, index: bool = True, nd: int = 4) -> str:
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + [_fmt(v, nd) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _label(p: dict) -> str:
    return ", ".join(f"{k}={p[k]:g}" if isinstance(p[k], (int, float)) else f"{k}={p[k]}"
                     for k in VARIED if k in p)


def _fixed(grid: list[dict]) -> str:
    keys = [k for k in grid[0] if k not in VARIED and all(g.get(k) == grid[0][k] for g in grid)]
    return ", ".join(f"`{k}={grid[0][k]}`" for k in keys) or "library defaults"


def _grid_rows(records: list[dict], metric: str, best: str):
    df = pd.DataFrame(records)
    df["configuration"] = df.apply(lambda r: _label(r.to_dict()), axis=1)
    valid = df[metric].dropna()
    pick = (valid.idxmin() if best == "min" else valid.idxmax()) if len(valid) else None
    df["selected"] = ["★" if i == pick else "" for i in df.index]
    return df[["configuration", metric, "selected"]]


def task1_section(spec: dict, spaces: dict[str, list[dict]]) -> str:
    out = [
        "## Task 1: 6-hour-ahead rolling proxy-SOFA forecast\n\n"
        "*Forecast the rolling 24-hour rule-based proxy-SOFA score.*",
        "",
        "**Protocol.** Each family has a fixed 4-point grid (Ridge and logistic regression have 3 points). "
        "Every grid point is fitted on the **training** split and scored on the **validation** split. "
        "Regressors are scored by validation MAE (lower is better). Ordinal threshold classifiers are scored by "
        "validation AUROC, and rolling 24-hour proxy-SOFA ≥2-point rise classifiers by validation AUPRC (higher is better). "
        "The best point is selected separately for each organ. No library-internal early stopping is used, "
        "and the test split plays no part.",
        "",
    ]
    out.append("### Search spaces")
    rows = []
    for name, grid in spaces.items():
        rows.append({"family": name, "configurations tried": "; ".join(_label(g) for g in grid),
                     "fixed": _fixed(grid)})
    out += [md_table(pd.DataFrame(rows), index=False), ""]

    metric_of = {"ordinal": ("val_auroc", "max"), "histgb_absolute": ("val_mae", "min")}
    for fam, entry in spec["families"].items():
        metric, best = metric_of.get(fam, ("val_mae", "min"))
        out.append(f"### `{fam}`: every grid point ({metric}; ★ = selected)")
        grids = entry["validation_grids"]
        if fam == "ordinal":
            for axis, by_k in grids.items():
                parts = []
                for k, recs in by_k.items():
                    t = _grid_rows(recs, metric, best)
                    t.insert(0, "threshold", f"P(score>{k})")
                    parts.append(t)
                out += [f"**{AXIS_NAME.get(axis, axis)}** (one selection per threshold classifier):",
                        md_table(pd.concat(parts), index=False), ""]
        else:
            parts = []
            for axis, recs in grids.items():
                t = _grid_rows(recs, metric, best)
                t.insert(0, "organ", AXIS_NAME.get(axis, axis))
                parts.append(t)
            out += [md_table(pd.concat(parts), index=False), ""]
        if "sofa_rise_grid" in entry:
            out.append(f"rolling 24-hour proxy-SOFA ≥2-point rise classifier (`{fam}`), validation AUPRC:")
            out += [md_table(_grid_rows(entry["sofa_rise_grid"], "val_auprc", "max"), index=False), ""]
        if "imbalance_experiment" in entry:
            out.append("**Class-imbalance strategy** (a pipeline hyperparameter), compared on validation. The "
                       "simplest strategy within one bootstrap SD of the best AUPRC is kept:")
            out += [md_table(pd.DataFrame(entry["imbalance_experiment"]).T), "",
                    f"Kept: **{entry['imbalance_strategy']}**.", ""]
    if "filter_ablation" in spec:
        out.append("### Feature filter (pipeline hyperparameters `max_missing=0.99`, `corr_threshold=0.98`)")
        out.append("Each organ's selected HistGB configuration was refitted without the filter and compared on "
                   "validation MAE:")
        out += [md_table(pd.DataFrame(spec["filter_ablation"]).T), ""]
    return "\n".join(out)


def task2_section(*, imb_lr: pd.DataFrame, imb_lr_kept: str, l3_variants: pd.DataFrame,
                  imb_xgb: pd.DataFrame, imb_xgb_kept: str, filter_check: pd.DataFrame,
                  xgb_space: dict, xgb_candidates: pd.DataFrame, xgb_best: dict, xgb_search_meta: dict,
                  topk: pd.DataFrame, k_chosen: int, n_filtered: int,
                  cb_space: dict, cb_candidates: pd.DataFrame, cb_params: dict, cb_trees: int,
                  selected_model: str) -> str:
    out = ["## Task 2: In-Hospital Mortality Prediction", "",
           "*Using measurements from the first 48 hours after ICU admission.*",
           "",
           "**Protocol.** All tuning uses the **training** split, by stratified 5-fold cross-validation "
           "(`StratifiedKFold(shuffle=True, random_state=42)`) unless a step says validation. Resampling, when "
           "compared, happens inside each training fold only. The primary score is AUPRC (average precision). "
           "The test split plays no part.",
           "",
           "### Logistic regression (L3, L3-revised)",
           "The recipe is **fixed, not tuned**: L2 penalty, `C=1`, `lbfgs`, `max_iter=1000`. The ladder's L3 rung "
           "is a pre-declared reference model, and tuning it would blur the comparison. What *was* chosen with "
           "data:",
           "",
           "*Class-imbalance strategy* (5-fold CV on train; simplest within one SE of the best AUPRC):",
           md_table(imb_lr), "", f"Kept: **{imb_lr_kept}**.", "",
           "*Feature set* (summary-statistic variants, scored on **validation**; variant E became L3-revised):",
           md_table(l3_variants), "",
           "### XGBoost (L4)",
           "*Class-imbalance strategy* (5-fold CV on train, filtered columns, starting configuration):",
           md_table(imb_xgb), "", f"Kept: **{imb_xgb_kept}**.", "",
           "*Feature filter check* (5-fold CV on train):",
           md_table(filter_check), "",
           f"*Hyperparameter search.* `RandomizedSearchCV`, **{xgb_search_meta.get('n_candidates', 50)} random "
           f"candidates × {xgb_search_meta.get('cv_folds', 5)}-fold CV** on the training split, scored by AUPRC, "
           f"`random_state=42`, on device `{xgb_search_meta.get('device')}` "
           f"({xgb_search_meta.get('minutes', float('nan')):.1f} min). Search space:",
           md_table(pd.DataFrame({"values": {k: str(v) for k, v in xgb_space.items()}})), "",
           f"All candidates, ranked by mean CV AUPRC (★ = selected; per-fold scores shown):",
           md_table(xgb_candidates, index=False), "",
           f"Selected: `{xgb_best}`.", "",
           f"*Embedded feature selection.* Nested 5-fold CV over *k*: the SHAP ranking is relearned inside every "
           f"training fold, then each candidate *k* is refitted and scored on the held-out fold. The smallest *k* "
           f"within one SE of the best is kept: **k = {k_chosen}** of {n_filtered} filtered features.",
           md_table(topk), "",
           "### CatBoost (L4 runner-up)",
           "Hyperparameters come from the original 50-candidate × 5-fold CV random search on the training split. "
           "That search scored by AUROC on the v1 feature set; its log is in "
           "`docs/task2_mortality/original_results/`. Search space:",
           md_table(pd.DataFrame({"values": {k: str(v) for k, v in cb_space.items()}})), "",
           "All candidates, ranked by mean CV AUROC (★ = selected):",
           md_table(cb_candidates, index=False), "",
           f"Selected: `{cb_params}`. In v2 it is refitted on the filtered v2 features with up to 2,000 trees and "
           f"early stopping on validation (patience 50), which stopped at **{cb_trees} trees**.", "",
           f"### Model selection",
           f"Validation AUPRC picks between the two final pipelines: **{selected_model}**."]
    return "\n".join(out)


def write_tuning_log(path: Path, task1: str, task2: str) -> Path:
    path = Path(path)
    header = ("# Hyperparameter tuning log\n\n"
              "Generated by the final cell of `MD6117_group_project.ipynb` from the tuning records saved during "
              "the run (`trained_models/task1_sofa/model_spec.json`, `results/task2_mortality/*`). Every "
              "configuration tried is listed with the score it was selected on, from validation or from "
              "cross-validation on the training split, **never from test**. All randomness is seeded "
              "(`SEED = 42`).\n")
    path.write_text(header + "\n" + task1 + "\n\n" + task2 + "\n", encoding="utf-8")
    return path
