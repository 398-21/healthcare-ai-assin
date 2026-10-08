"""Physiological plausibility rules shared by both tasks.

A recorded value outside these clinical bounds is treated as a measurement/entry artefact
and set to missing -- never clipped to the bound, which would pile fake values up at the
edge. Two unit errors are corrected first: pH recorded without its decimal point (735 ->
7.35) and height recorded in metres (1.7 -> 170 cm). The bounds are fixed from clinical
knowledge, not fitted to the data, so applying them cannot leak information between splits.

`Urine` keeps 0 (true anuria). Age starts at 15 because the dataset's own documentation
states ages run from 15 to 90 (90 = 90 or older).
"""
from __future__ import annotations

import math

PHYSIOLOGICAL_RANGES: dict[str, tuple[float, float]] = {
    # vital signs
    "HR": (20, 250), "Temp": (30.0, 44.0), "RespRate": (4, 60),
    # blood pressure (0 is a sensor dropout, not a pressure)
    "SysABP": (40, 280), "DiasABP": (20, 200), "MAP": (30, 200),
    "NISysABP": (40, 280), "NIDiasABP": (20, 200), "NIMAP": (30, 200),
    # blood gas
    "pH": (6.5, 8.0), "PaCO2": (10, 150), "PaO2": (20, 600), "SaO2": (50, 100), "FiO2": (0.21, 1.0),
    # electrolytes / chemistry
    "K": (1.5, 10.0), "Na": (100, 180), "Mg": (0.3, 6.0), "HCO3": (5, 60), "Glucose": (10, 2000),
    "Lactate": (0.1, 50),
    # renal
    "BUN": (1, 300), "Creatinine": (0.1, 25), "Urine": (0, 2000),
    # haematology
    "WBC": (0.1, 500), "HCT": (5.0, 75.0), "Platelets": (1, 2000),
    # liver / protein
    "Albumin": (0.5, 6.0), "Bilirubin": (0.1, 60), "AST": (1, 50000), "ALT": (1, 30000), "ALP": (10, 5000),
    # cardiac / lipids
    "TroponinI": (0, 500), "TroponinT": (0, 50), "Cholesterol": (20, 600),
    # neuro
    "GCS": (3, 15),
    # descriptors
    "Weight": (20, 300), "Height": (100, 250), "Age": (15, 120),
}

MISSING = -1


def validate_physiological_value(param: str, value: float) -> float:
    """Return the (unit-corrected) value, or MISSING (-1) if it is implausible."""
    if value == MISSING or value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING
    if param == "pH" and value > 14:          # decimal point dropped: 735 -> 7.35
        value = value / 100
    if param == "Height" and 0 < value < 100:   # recorded in metres: 1.7 -> 170 cm
        value = value * 100
    if param not in PHYSIOLOGICAL_RANGES:
        return value
    lo, hi = PHYSIOLOGICAL_RANGES[param]
    return value if lo <= value <= hi else MISSING


def screen_series(series: dict[str, list[tuple[int, float]]]) -> tuple[dict, dict]:
    """Apply `validate_physiological_value` to a {param: [(t_min, value)]} record.

    Returns (screened_series, n_removed_per_param)."""
    out, removed = {}, {}
    for p, tv in series.items():
        kept = []
        for t, v in tv:
            vv = validate_physiological_value(p, v)
            if vv == MISSING:
                removed[p] = removed.get(p, 0) + 1
            else:
                kept.append((t, vv))
        out[p] = kept
    return out, removed
