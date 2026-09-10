"""Central configuration for the SOFA-forecasting project.

Single source of truth for paths, the forecasting task geometry, the rule-based
scoring constants (mirrors `sofa_calculation.ipynb`), and the train/val/test split.
"""
from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- paths
MODELING_DIR = Path(__file__).resolve().parent
PROJ_ROOT = MODELING_DIR.parent
DATA_DIR = PROJ_ROOT / "icu_records"
OUTCOMES_CSV = PROJ_ROOT / "outcomes.csv"
SOFA_SCORES_CSV = PROJ_ROOT / "sofa_scores.csv"          # rule-based scores from the notebook
OUT_DIR = MODELING_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RECORDS_CLEAN = OUT_DIR / "records_clean.joblib"
SPLITS_CSV = OUT_DIR / "splits.csv"
TARGETS_CSV = OUT_DIR / "targets.csv"
FEATURES_CSV = OUT_DIR / "features.csv.gz"    # ~400 cols x 48k rows — gzip (pandas reads/writes transparently)
QC_REPORT_MD = OUT_DIR / "qc_report.md"

SEED = 42

# ------------------------------------------------------------------- forecasting task
# Decision times T (hours after ICU admission). At each T we forecast 6 h ahead.
HORIZONS_H: tuple[int, ...] = (24, 30, 36, 42)
FORECAST_AHEAD_H = 6            # predict SOFA as of  T + 6
TARGET_WINDOW_H = 24           # target = trailing-24 h SOFA ending at T+6  ->  [T-18, T+6)
NOW_WINDOW_H = 24             # SOFA_now = trailing-24 h SOFA ending at T   ->  [T-24, T)
DETERIORATE_DELTA = 2         # "deterioration" = SOFA rises >= 2 within 24 h

def target_window_min(T: int) -> tuple[int, int]:
    """[T-18h, T+6h) in minutes — the window the forecast target is scored over."""
    end = (T + FORECAST_AHEAD_H) * 60
    return (end - TARGET_WINDOW_H * 60, end)

def now_window_min(T: int) -> tuple[int, int]:
    """[T-24h, T) in minutes — the last fully-observed 24 h, used for SOFA_now."""
    return ((T - NOW_WINDOW_H) * 60, T * 60)

def feature_cutoff_min(T: int) -> int:
    """Features at decision time T may only use measurements with t_min < this."""
    return T * 60

# ------------------------------------------------------- rule-based scoring constants
# (identical to sofa_calculation.ipynb / the methodology doc §3+§5)
PAIR_WINDOW_MIN = 120          # PaO2 pairs with the nearest FiO2 / MechVent within +-2 h
URINE_MIN_SPAN_MIN = 1200     # use the urine threshold only when its span >= 20 h
FULL_DAY1_MIN = (0, 1440)     # [0, 24) h  — the fixed admission-day SOFA block

# doc §7 physiologic-plausibility bounds (see data_prep/parsing.keep_value)
IMPLAUSIBLE = {
    "nonpositive": ("MAP", "NIMAP", "PaO2", "SaO2"),   # drop value <= 0
    "weight_max_drop": 20.0,                            # drop Weight <= 20 kg
    "height_lo": 100.0, "height_hi": 250.0,             # drop Height < 100 or > 250 cm
}
DESC_PARAMS = ("RecordID", "Age", "Gender", "Height", "ICUType", "Weight")

# --------------------------------------------------------------------------- split
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test
