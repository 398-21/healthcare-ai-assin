"""Parse and clean the raw PhysioNet 2012 record files.

Lifted verbatim (rules-wise) from `scratchpad/sofa_dev.py` / `sofa_calculation.ipynb`
so the modelling pipeline and the SOFA notebook clean the data identically.

Cleaning = methodology-doc section 7:
  * drop the -1 missing sentinel
  * drop rows with an empty `Parameter`
  * drop MAP / NIMAP / PaO2 / SaO2 <= 0    (waveform dropouts)
  * drop Weight <= 20 kg
  * drop Height < 100 or > 250 cm
"""
from __future__ import annotations
import csv
import glob
from pathlib import Path

from .. import config

_NONPOS = set(config.IMPLAUSIBLE["nonpositive"])
_W_DROP = config.IMPLAUSIBLE["weight_max_drop"]
_H_LO = config.IMPLAUSIBLE["height_lo"]
_H_HI = config.IMPLAUSIBLE["height_hi"]
_DESC = set(config.DESC_PARAMS)


def keep_value(param: str, v: float) -> bool:
    """doc section 7 physiologic-plausibility filter (on top of the -1 sentinel)."""
    if v == -1:
        return False
    if param in _NONPOS and v <= 0:
        return False
    if param == "Weight" and v <= _W_DROP:
        return False
    if param == "Height" and (v < _H_LO or v > _H_HI):
        return False
    return True


def parse_record(path: str | Path) -> tuple[dict[str, list[tuple[int, float]]], dict[str, float]]:
    """-> (series: {param: sorted [(t_min, value)]}, descriptors: {param: value}).

    `t_min` = minutes since ICU admission (HH*60 + MM; HH runs past 24).
    """
    series: dict[str, list[tuple[int, float]]] = {}
    desc: dict[str, float] = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        next(r, None)  # header
        for row in r:
            if len(row) != 3:
                continue
            t_s, p, v_s = row
            if p == "":  # doc section 7: drop label-less rows
                continue
            try:
                hh, mm = t_s.split(":")
                t = int(hh) * 60 + int(mm)
                v = float(v_s)
            except ValueError:
                continue
            if not keep_value(p, v):
                continue
            if p in _DESC and t == 0 and p not in desc:
                desc[p] = v
            series.setdefault(p, []).append((t, v))
    for p in series:
        series[p].sort()
    return series, desc


def parse_all(data_dir: str | Path | None = None) -> dict[int, tuple[dict, dict]]:
    """Parse every `<RecordID>.csv` under `data_dir` -> {RecordID: (series, desc)}."""
    data_dir = Path(data_dir or config.DATA_DIR)
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    out: dict[int, tuple[dict, dict]] = {}
    for f in files:
        series, desc = parse_record(f)
        rid = int(desc.get("RecordID", int(Path(f).stem)))
        out[rid] = (series, desc)
    return out
