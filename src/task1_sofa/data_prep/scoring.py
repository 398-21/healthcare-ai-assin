"""Rule-based SOFA scoring over an arbitrary time window.

The six organ-system scorers of the methodology doc (section 3 table + section 5
dataset rules). The project notebook validates them against `outcomes.csv` SOFA and
the doc's worked examples before they are used to build any forecast target.

Every scorer takes the parsed `series` dict and a window `[lo, hi)` in minutes and
returns `(score | None, measured_in_window: bool)`.  `None` = the axis has no valid
value in the window (lenient totals count it as 0; strict totals treat it as missing).

Known limitation: cardiovascular can only reach 0/1 — the dataset has no vasopressor
drug/dose data, so SOFA cardiovascular levels 2-4 are structurally unreachable.
"""
from __future__ import annotations

from .. import config

PAIR_WINDOW_MIN = config.PAIR_WINDOW_MIN
URINE_MIN_SPAN_MIN = config.URINE_MIN_SPAN_MIN

AXES = ("resp", "coag", "liver", "cardio", "neuro", "renal")


# --------------------------------------------------------------------- window helpers
def w_vals(series, p, lo, hi):
    return [v for (t, v) in series.get(p, []) if lo <= t < hi]


def w_tv(series, p, lo, hi):
    return [(t, v) for (t, v) in series.get(p, []) if lo <= t < hi]


def _nearest(seq, t):
    """value in seq=[(t,v)] nearest to `t` within PAIR_WINDOW_MIN (ties -> earlier); else None."""
    cand = sorted(
        ((abs(tt - t), tt, vv) for (tt, vv) in seq if abs(tt - t) <= PAIR_WINDOW_MIN),
        key=lambda x: (x[0], x[1]),
    )
    return cand[0][2] if cand else None


# --------------------------------------------------------------------- organ scorers
def resp_score(series, lo, hi):
    pao2, fio2 = w_tv(series, "PaO2", lo, hi), w_tv(series, "FiO2", lo, hi)
    if not pao2 or not fio2:
        return None, False
    mv = w_tv(series, "MechVent", lo, hi)
    pairs = []
    for tp, vp in pao2:
        vf = _nearest(fio2, tp)
        if vf and vf > 0:
            pairs.append((vp / vf, _nearest(mv, tp) is not None))
    if not pairs:
        return None, False
    r, support = min(pairs, key=lambda x: x[0])
    if r >= 400:
        s = 0
    elif r >= 300:
        s = 1
    elif r >= 200:
        s = 2
    elif r >= 100:
        s = 3 if support else 2
    else:
        s = 4 if support else 2
    return s, True


def coag_score(series, lo, hi):
    v = w_vals(series, "Platelets", lo, hi)
    if not v:
        return None, False
    m = min(v)
    return (0 if m >= 150 else 1 if m >= 100 else 2 if m >= 50 else 3 if m >= 20 else 4), True


def _bili(v):
    return 0 if v < 1.2 else 1 if v < 2.0 else 2 if v < 6.0 else 3 if v < 12.0 else 4


def liver_score(series, lo, hi):
    v = w_vals(series, "Bilirubin", lo, hi)
    if v:
        return _bili(max(v)), True
    prior = [x for (t, x) in series.get("Bilirubin", []) if t < lo]
    if prior:                       # LOCF from before the window -> lenient total only
        return _bili(prior[-1]), False
    return None, False


def cardio_score(series, lo, hi):
    inv, ni = w_vals(series, "MAP", lo, hi), w_vals(series, "NIMAP", lo, hi)
    if inv:
        m = min(inv)
    elif ni:
        m = min(ni)
    else:
        return None, False
    return (0 if m >= 70 else 1), True          # 2-4 unreachable: no vasopressor data


def neuro_score(series, lo, hi):
    v = w_vals(series, "GCS", lo, hi)
    if not v:
        return None, False
    m = min(v)
    return (0 if m >= 15 else 1 if m >= 13 else 2 if m >= 10 else 3 if m >= 6 else 4), True


def renal_score(series, lo, hi):
    cr = w_vals(series, "Creatinine", lo, hi)
    cr_s = None
    if cr:
        c = max(cr)
        cr_s = 0 if c < 1.2 else 1 if c < 2.0 else 2 if c < 3.5 else 3 if c < 5.0 else 4
    ur = w_tv(series, "Urine", lo, hi)
    ur_s = None
    if ur:
        ts = [t for t, _ in ur]
        if max(ts) - min(ts) >= URINE_MIN_SPAN_MIN:     # urine represents ~a full 24 h
            tot = sum(v for _, v in ur)
            ur_s = 4 if tot < 200 else 3 if tot < 500 else 0
    avail = [s for s in (cr_s, ur_s) if s is not None]
    return (max(avail), True) if avail else (None, False)


SYSTEMS = {
    "resp": resp_score, "coag": coag_score, "liver": liver_score,
    "cardio": cardio_score, "neuro": neuro_score, "renal": renal_score,
}


# --------------------------------------------------------------------- aggregation
def score_window(series, lo, hi) -> dict:
    """Score all six axes over [lo, hi).

    Returns {axis: score|None, axis_present: bool, ...,
             sofa_total (lenient), n_present, min_map}.
    """
    out: dict = {}
    total = 0
    n_present = 0
    for name, fn in SYSTEMS.items():
        s, present = fn(series, lo, hi)
        out[name] = s
        out[f"{name}_present"] = present
        total += s if s is not None else 0
        n_present += int(present)
    out["sofa_total"] = total
    out["n_present"] = n_present
    maps = w_vals(series, "MAP", lo, hi) + w_vals(series, "NIMAP", lo, hi)
    out["min_map"] = min(maps) if maps else None
    return out
