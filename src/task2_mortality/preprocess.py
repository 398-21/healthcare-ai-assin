"""Task 2 pre-processing: raw ICU records -> one 1,474-feature row per stay.

Landmark task: use observations with 0 <= Time < 48:00 to predict final In-hospital_death.
Rules live in `config.json`; the raw `data/` folder is never modified. Outputs go to
`preprocessed/task2_mortality/` (or `output_dir` if given):

  patient_features_unimputed.csv.gz   NaN kept -- used by the tree models
  X_{train,validation,test}_imputed   train-median imputed -- used by the linear models
  labels_and_splits.csv               label + frozen stratified 70/15/15 split (seed 20260907)
  labels_and_baselines.csv            + SAPS-I / SOFA as *scores* (never features) for rung L2
  feature_dictionary.csv, preprocessing_state.json, ... (QC tables, hashes, audit summary)

Run from the repo root:  python -m src.task2_mortality.preprocess
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]          # repo root (src/task2_mortality/ -> ../..)
HERE = Path(__file__).resolve().parent
RELEASE_DIR = ROOT / "data"
RECORDS_DIR = RELEASE_DIR / "icu_records"
OUTCOMES_PATH = RELEASE_DIR / "outcomes.csv"
CONFIG_PATH = HERE / "config.json"
OUTPUT_DIR = ROOT / "preprocessed" / "task2_mortality"


def _rel(path: Path) -> str:
    """Path relative to the repo root when possible (keeps manifests machine-independent)."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


UNITS = {
    "Albumin": "g/dL", "ALP": "IU/L", "ALT": "IU/L", "AST": "IU/L",
    "Bilirubin": "mg/dL", "BUN": "mg/dL", "Cholesterol": "mg/dL",
    "Creatinine": "mg/dL", "DiasABP": "mmHg", "FiO2": "fraction",
    "GCS": "score", "Glucose": "mg/dL", "HCO3": "mmol/L", "HCT": "%",
    "HR": "bpm", "K": "mEq/L", "Lactate": "mmol/L", "MAP": "mmHg",
    "MechVent": "binary", "Mg": "mmol/L", "Na": "mEq/L",
    "NIDiasABP": "mmHg", "NIMAP": "mmHg", "NISysABP": "mmHg",
    "PaCO2": "mmHg", "PaO2": "mmHg", "pH": "pH", "Platelets": "cells/nL",
    "RespRate": "bpm", "SaO2": "%", "SysABP": "mmHg", "Temp": "degC",
    "TroponinI": "ug/L", "TroponinT": "ug/L", "Urine": "mL",
    "WBC": "cells/nL", "Weight": "kg",
}

# Physiological ranges (Phase 2): one shared definition for both tasks -- see
# src/shared/physiology.py. Out-of-range values become missing; pH and height unit
# errors are corrected first. Age now starts at 15, as the dataset documentation states.
from src.shared.physiology import PHYSIOLOGICAL_RANGES, validate_physiological_value  # noqa: E402,F401

QUALITY_COLUMNS = [
    "raw_rows", "invalid_time_rows", "empty_parameter_rows", "unknown_parameter_rows",
    "invalid_numeric_rows", "sentinel_rows", "negative_time_rows", "exact_48h_rows",
    "after_48h_rows", "exact_duplicate_rows", "conflicting_time_parameter_keys",
    "conflicting_time_parameter_extra_values", "primary_conflicting_time_parameter_keys",
    "primary_conflicting_time_parameter_extra_values", "static_conflict_parameters",
    "filename_recordid_mismatch", "valid_dynamic_rows_including_time0_weight",
    "valid_dynamic_rows_excluding_time0_weight",
]


@dataclass(frozen=True)
class Window:
    name: str
    start_minute: int
    end_minute: int


def parse_elapsed_minutes(raw: str) -> int | None:
    """Parse elapsed HH:MM without applying a 24-hour clock wrap."""
    try:
        hour_raw, minute_raw = raw.strip().split(":", 1)
        hour = int(hour_raw)
        minute = int(minute_raw)
    except (AttributeError, TypeError, ValueError):
        return None
    if hour < 0 or minute < 0 or minute >= 60:
        return None
    return hour * 60 + minute


def safe_float(raw: str | None) -> float | None:
    try:
        value = float(raw) if raw is not None else math.nan
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_parameter(parameter: str) -> str:
    return parameter.lower().replace("-", "_")


def feature_name(parameter: str, window: str, statistic: str) -> str:
    return f"{normalize_parameter(parameter)}__{window}__{statistic}"


def deterministic_stratified_split(
    record_ids: np.ndarray,
    labels: np.ndarray,
    seed: int,
    ratios: dict[str, float],
) -> dict[int, str]:
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to 1")
    rng = np.random.default_rng(seed)
    assignment: dict[int, str] = {}
    for label in sorted(np.unique(labels).tolist()):
        class_ids = np.sort(record_ids[labels == label].astype(np.int64))
        shuffled = class_ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * ratios["train"]))
        n_validation = int(round(n * ratios["validation"]))
        boundaries = (n_train, n_train + n_validation)
        for rid in shuffled[: boundaries[0]]:
            assignment[int(rid)] = "train"
        for rid in shuffled[boundaries[0] : boundaries[1]]:
            assignment[int(rid)] = "validation"
        for rid in shuffled[boundaries[1] :]:
            assignment[int(rid)] = "test"
    return assignment


def window_mask(times: np.ndarray, window: Window, include_48h: bool) -> np.ndarray:
    if include_48h and window.end_minute == 2880:
        return (times >= window.start_minute) & (times <= window.end_minute)
    return (times >= window.start_minute) & (times < window.end_minute)


def summarize_observations(
    observations: list[tuple[int, int, float]],
    window: Window,
    include_48h: bool = False,
    same_time_rule: str = "median",
) -> dict[str, float]:
    if not observations:
        return {"measured": 0.0, "count": 0.0}
    ordered = sorted(observations, key=lambda item: (item[0], item[1]))
    times_all = np.asarray([item[0] for item in ordered], dtype=np.float64)
    mask = window_mask(times_all, window, include_48h)
    if not bool(mask.any()):
        return {"measured": 0.0, "count": 0.0}
    selected = [item for item, keep in zip(ordered, mask.tolist()) if keep]
    grouped: dict[int, list[float]] = defaultdict(list)
    for elapsed, _row_index, value in selected:
        grouped[elapsed].append(value)
    collapsed: list[tuple[int, float]] = []
    for elapsed, same_time_values in sorted(grouped.items()):
        if same_time_rule == "sum":
            collapsed_value = float(np.sum(same_time_values))
        elif same_time_rule == "max":
            collapsed_value = float(np.max(same_time_values))
        else:
            collapsed_value = float(np.median(same_time_values))
        collapsed.append((elapsed, collapsed_value))
    times = np.asarray([item[0] for item in collapsed], dtype=np.float64)
    values = np.asarray([item[1] for item in collapsed], dtype=np.float64)
    first = float(values[0])
    last = float(values[-1])
    result = {
        "measured": 1.0,
        "count": float(len(selected)),
        "first": first,
        "last": last,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "delta": last - first,
        "time_span_hours": float((times[-1] - times[0]) / 60.0),
    }
    centered_time = times - float(np.mean(times))
    denominator = float(np.sum(centered_time * centered_time))
    if len(values) >= 2 and denominator > 0:
        centered_values = values - float(np.mean(values))
        result["slope_per_hour"] = float(np.sum(centered_time * centered_values) / denominator * 60.0)
    else:
        result["slope_per_hour"] = math.nan
    return result


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_feature_dictionary(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "feature_name": "RecordID", "source_parameter": "RecordID", "window": "none",
            "statistic": "join_key", "data_type": "integer", "unit": "none",
            "missing_behavior": "not allowed", "included_in_model": False,
            "notes": "Join key only; must be removed from X before model fitting.",
        },
        {
            "feature_name": "static_age_years", "source_parameter": "Age", "window": "00:00 descriptor",
            "statistic": "first_valid", "data_type": "continuous", "unit": "years",
            "missing_behavior": "training median", "included_in_model": True,
            "notes": "Age 90 represents age 90 or older per release README.",
        },
        {
            "feature_name": "static_height_cm", "source_parameter": "Height", "window": "00:00 descriptor",
            "statistic": "first_valid", "data_type": "continuous", "unit": "cm",
            "missing_behavior": "training median", "included_in_model": True,
            "notes": "No outlier clipping was applied.",
        },
        {
            "feature_name": "static_admission_weight_kg", "source_parameter": "Weight", "window": "00:00 descriptor",
            "statistic": "first_valid", "data_type": "continuous", "unit": "kg",
            "missing_behavior": "training median", "included_in_model": True,
            "notes": (
                "Separate from longitudinal Weight summaries. The 00:00 descriptor is intentionally also "
                "retained inside the Weight time series so that 48h weight delta/slope keep an admission "
                "anchor for fluid-balance interpretation; the overlap is an alias, not a double measurement."
            ),
        },
        {
            "feature_name": "static_bmi", "source_parameter": "Height,Weight", "window": "00:00 descriptor",
            "statistic": "derived_ratio", "data_type": "continuous", "unit": "kg/m^2",
            "missing_behavior": "training median", "included_in_model": True,
            "notes": "Weight / (Height/100)^2 from the 00:00 descriptors; missing when either input is missing.",
        },
    ]
    for name, source in [
        ("static_age_missing", "Age"), ("static_height_missing", "Height"),
        ("static_admission_weight_missing", "Weight"), ("static_bmi_missing", "Height,Weight"),
    ]:
        rows.append({
            "feature_name": name, "source_parameter": source, "window": "00:00 descriptor",
            "statistic": "missing_indicator", "data_type": "binary", "unit": "none",
            "missing_behavior": "always observed", "included_in_model": True,
            "notes": "1 means the valid descriptor was absent.",
        })
    for value in (0, 1):
        rows.append({
            "feature_name": f"static_gender_{value}", "source_parameter": "Gender",
            "window": "00:00 descriptor", "statistic": f"one_hot_{value}",
            "data_type": "binary", "unit": "none", "missing_behavior": "always observed",
            "included_in_model": True, "notes": "Fixed one-hot encoding from README categories.",
        })
    rows.append({
        "feature_name": "static_gender_missing", "source_parameter": "Gender",
        "window": "00:00 descriptor", "statistic": "missing_or_invalid_indicator",
        "data_type": "binary", "unit": "none", "missing_behavior": "always observed",
        "included_in_model": True, "notes": "1 means absent, -1, or outside {0,1}.",
    })
    for value in (1, 2, 3, 4):
        rows.append({
            "feature_name": f"static_icutype_{value}", "source_parameter": "ICUType",
            "window": "00:00 descriptor", "statistic": f"one_hot_{value}",
            "data_type": "binary", "unit": "none", "missing_behavior": "always observed",
            "included_in_model": True, "notes": "Fixed one-hot encoding from README categories.",
        })
    rows.append({
        "feature_name": "static_icutype_missing", "source_parameter": "ICUType",
        "window": "00:00 descriptor", "statistic": "missing_or_invalid_indicator",
        "data_type": "binary", "unit": "none", "missing_behavior": "always observed",
        "included_in_model": True, "notes": "1 means absent, -1, or outside {1,2,3,4}.",
    })

    for window in config["windows"]:
        window_name = window["name"]
        for parameter in config["dynamic_parameters"]:
            for statistic in config["summary_statistics"]:
                missing = "0 when absent" if statistic in {"measured", "count"} else "training median; measured/count retained"
                rows.append({
                    "feature_name": feature_name(parameter, window_name, statistic),
                    "source_parameter": parameter, "window": window_name,
                    "statistic": statistic, "data_type": "binary" if statistic == "measured" else "continuous",
                    "unit": UNITS.get(parameter, "unknown"), "missing_behavior": missing,
                    "included_in_model": True,
                    "notes": (
                        "Primary excludes Time=48:00. Same-time values use median; MechVent uses max and "
                        "Urine uses sum. Count is raw valid row count. Statistic 'delta' is last minus first, "
                        "i.e. the (末值−首值) trend required by 实验方案 §4.1; 'slope_per_hour' is the "
                        "least-squares trend per hour over the same window."
                    ),
                })
        for statistic, note in [
            ("valid_observation_count", "Number of finite, non-sentinel values across dynamic parameters."),
            ("distinct_parameter_count", "Number of dynamic parameters with at least one valid value."),
            ("missing_parameter_fraction", "1 - distinct_parameter_count / 37."),
        ]:
            rows.append({
                "feature_name": f"record__{window_name}__{statistic}", "source_parameter": "all_dynamic",
                "window": window_name, "statistic": statistic, "data_type": "continuous",
                "unit": "count" if statistic != "missing_parameter_fraction" else "fraction",
                "missing_behavior": "always observed", "included_in_model": True, "notes": note,
            })
        for parameter, statistic, unit, note in [
            ("Urine", "total", "mL", "Sum of recorded urine values; not extrapolated for unobserved intervals."),
            ("MechVent", "ever", "binary", "1 if any recorded MechVent value equals 1."),
        ]:
            rows.append({
                "feature_name": feature_name(parameter, window_name, statistic),
                "source_parameter": parameter, "window": window_name, "statistic": statistic,
                "data_type": "binary" if statistic == "ever" else "continuous", "unit": unit,
                "missing_behavior": "training median; measured/count retained", "included_in_model": True,
                "notes": note,
            })
        for group in config.get("bp_merge_groups", []):
            for statistic in config["summary_statistics"]:
                missing = "0 when absent" if statistic in {"measured", "count"} else "training median; measured/count retained"
                rows.append({
                    "feature_name": feature_name(group["name"], window_name, statistic),
                    "source_parameter": f"{group['invasive']}|{group['noninvasive']}",
                    "window": window_name, "statistic": statistic,
                    "data_type": "binary" if statistic == "measured" else "continuous",
                    "unit": group["unit"], "missing_behavior": missing, "included_in_model": True,
                    "notes": (
                        f"Derived merged blood pressure: uses {group['invasive']} when this window has at least "
                        f"one invasive reading, otherwise {group['noninvasive']}. Source parallel columns are "
                        "retained separately. Excluded from record-level distinct/total observation counters."
                    ),
                })
        if config.get("bp_merge_groups"):
            rows.append({
                "feature_name": feature_name("BP", window_name, "has_arterial_line"),
                "source_parameter": "SysABP|DiasABP|MAP", "window": window_name,
                "statistic": "has_arterial_line", "data_type": "binary", "unit": "none",
                "missing_behavior": "always observed", "included_in_model": True,
                "notes": "1 if any invasive arterial stream has a reading in this window; presence of an arterial line is itself a severity signal.",
            })
    return pd.DataFrame(rows)


def derive_bmi(height_cm: float, weight_kg: float) -> float:
    """BMI from the 00:00 descriptors; NaN when either input is unusable (Fix W5 - Phase 3)."""
    if math.isnan(height_cm) or math.isnan(weight_kg) or height_cm <= 0:
        return math.nan
    return weight_kg / ((height_cm / 100.0) ** 2)


def initial_static_features(static_values: dict[str, float]) -> dict[str, float]:
    age = static_values.get("Age", math.nan)
    height = static_values.get("Height", math.nan)
    weight = static_values.get("Weight", math.nan)
    gender = static_values.get("Gender", math.nan)
    icu_type = static_values.get("ICUType", math.nan)
    bmi = derive_bmi(height, weight)
    return {
        "static_age_years": age,
        "static_age_missing": float(math.isnan(age)),
        "static_height_cm": height,
        "static_height_missing": float(math.isnan(height)),
        "static_admission_weight_kg": weight,
        "static_admission_weight_missing": float(math.isnan(weight)),
        "static_bmi": bmi,
        "static_bmi_missing": float(math.isnan(bmi)),
        "static_gender_0": float(gender == 0),
        "static_gender_1": float(gender == 1),
        "static_gender_missing": float(gender not in {0, 1}),
        "static_icutype_1": float(icu_type == 1),
        "static_icutype_2": float(icu_type == 2),
        "static_icutype_3": float(icu_type == 3),
        "static_icutype_4": float(icu_type == 4),
        "static_icutype_missing": float(icu_type not in {1, 2, 3, 4}),
    }


def aggregate_patient(
    static_values: dict[str, float],
    observations: dict[str, list[tuple[int, int, float]]],
    config: dict[str, Any],
    include_48h: bool,
) -> dict[str, float]:
    features = initial_static_features(static_values)
    for raw_window in config["windows"]:
        window = Window(**raw_window)
        distinct = 0
        total = 0
        for parameter in config["dynamic_parameters"]:
            same_time_rule = "sum" if parameter == "Urine" else "max" if parameter == "MechVent" else "median"
            summary = summarize_observations(
                observations.get(parameter, []), window, include_48h, same_time_rule=same_time_rule
            )
            if summary["measured"] == 1:
                distinct += 1
                total += int(summary["count"])
            for statistic in config["summary_statistics"]:
                features[feature_name(parameter, window.name, statistic)] = summary.get(statistic, math.nan)

            parameter_observations = observations.get(parameter, [])
            if parameter in {"Urine", "MechVent"} and parameter_observations:
                ordered = sorted(parameter_observations, key=lambda item: (item[0], item[1]))
                times = np.asarray([item[0] for item in ordered], dtype=np.float64)
                mask = window_mask(times, window, include_48h)
                values = np.asarray([item[2] for item, keep in zip(ordered, mask.tolist()) if keep], dtype=np.float64)
            else:
                values = np.asarray([], dtype=np.float64)
            if parameter == "Urine":
                features[feature_name(parameter, window.name, "total")] = float(np.sum(values)) if values.size else math.nan
            elif parameter == "MechVent":
                features[feature_name(parameter, window.name, "ever")] = float(np.max(values)) if values.size else 0.0

        features[f"record__{window.name}__valid_observation_count"] = float(total)
        features[f"record__{window.name}__distinct_parameter_count"] = float(distinct)
        features[f"record__{window.name}__missing_parameter_fraction"] = float(
            1.0 - distinct / len(config["dynamic_parameters"])
        )

        # Merged blood pressure streams, invasive preferred (Fix W1 - Phase 3).
        # Derived only: does NOT contribute to the record-level counters above, which
        # describe the 37 raw dynamic parameters and must stay comparable across runs.
        arterial_line = 0.0
        for group in config.get("bp_merge_groups", []):
            invasive_summary = summarize_observations(
                observations.get(group["invasive"], []), window, include_48h
            )
            if invasive_summary["measured"] == 1:
                arterial_line = 1.0
                merged = invasive_summary
            else:
                merged = summarize_observations(
                    observations.get(group["noninvasive"], []), window, include_48h
                )
            for statistic in config["summary_statistics"]:
                features[feature_name(group["name"], window.name, statistic)] = merged.get(statistic, math.nan)
        if config.get("bp_merge_groups"):
            features[feature_name("BP", window.name, "has_arterial_line")] = arterial_line
    return features


def write_csv_gzip(frame: pd.DataFrame, path: Path, compresslevel: int) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=compresslevel, mtime=0, fileobj=raw_handle) as gzip_handle:
            frame.to_csv(gzip_handle, index=False, lineterminator="\n", float_format="%.8g")


def add_source_manifest(config: dict[str, Any], record_paths: list[Path]) -> dict[str, Any]:
    zip_path = ROOT / "release.zip"   # the course zip, if kept next to the repo
    directory_digest = hashlib.sha256()
    for path in record_paths:
        directory_digest.update(path.name.encode("utf-8"))
        directory_digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                directory_digest.update(chunk)
        directory_digest.update(b"\0")
    manifest = {
        "release_readme": {
            "path": _rel(RELEASE_DIR / "README.md"),
            "sha256": sha256_file(RELEASE_DIR / "README.md"),
        },
        "outcomes": {
            "path": _rel(OUTCOMES_PATH),
            "sha256": sha256_file(OUTCOMES_PATH),
        },
        "record_directory": {
            "path": _rel(RECORDS_DIR),
            "csv_count": len(record_paths),
            "total_bytes": int(sum(path.stat().st_size for path in record_paths)),
            "filename_size_manifest_sha256": sha256_json(
                [(path.name, path.stat().st_size) for path in record_paths]
            ),
            "ordered_filename_content_sha256": directory_digest.hexdigest(),
        },
        "release_zip": None,
        "config_sha256": sha256_file(CONFIG_PATH),
        "pipeline_sha256": sha256_file(Path(__file__).resolve()),
    }
    if zip_path.exists():
        manifest["release_zip"] = {
            "path": _rel(zip_path),
            "bytes": zip_path.stat().st_size,
            "sha256": sha256_file(zip_path),
        }
    return manifest


def main(output_dir: Path | str | None = None) -> dict[str, Any]:
    """Run the full pipeline; returns the run summary (also written as preprocessing_summary.json)."""
    started = time.perf_counter()
    config = load_config()
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    record_paths = sorted(RECORDS_DIR.glob("*.csv"), key=lambda path: int(path.stem))
    outcomes = pd.read_csv(OUTCOMES_PATH)
    expected_outcome_columns = {
        "RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"
    }
    if set(outcomes.columns) != expected_outcome_columns:
        raise ValueError(f"Unexpected outcome schema: {outcomes.columns.tolist()}")
    if outcomes["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID in outcomes.csv")
    if outcomes["In-hospital_death"].isna().any() or not set(outcomes["In-hospital_death"].unique()).issubset({0, 1}):
        raise ValueError("In-hospital_death must be complete and binary")

    # Validate IDs match before filtering (Fix 1.2)
    outcome_ids_before_filter = set(outcomes["RecordID"].astype(int))
    file_ids = {int(path.stem) for path in record_paths}
    if outcome_ids_before_filter != file_ids:
        raise ValueError("Record file IDs and outcome IDs differ")

    # Filter out negative Length_of_stay values (Fix 1.2)
    n_negative_los = (outcomes['Length_of_stay'] < 0).sum()
    if n_negative_los > 0:
        print(f"Excluding {n_negative_los} records with negative Length_of_stay")
        outcomes = outcomes[outcomes['Length_of_stay'] >= 0].copy()
        # Filter record_paths to only process valid RecordIDs
        valid_record_ids = set(outcomes["RecordID"].astype(int))
        record_paths = [path for path in record_paths if int(path.stem) in valid_record_ids]

    outcome_ids = set(outcomes["RecordID"].astype(int))

    allowed_parameters = set(config["static_parameters"]) | set(config["dynamic_parameters"]) | {"RecordID"}
    all_rows: list[dict[str, float | int]] = []
    sensitivity_rows: list[dict[str, float | int]] = []
    quality_rows: list[dict[str, float | int]] = []
    audit = Counter()
    parameter_valid_values: dict[str, list[float]] = defaultdict(list)
    parameter_valid_patients: dict[str, set[int]] = defaultdict(set)
    parameter_sentinel = Counter()
    parameter_unknown = Counter()

    for path_number, path in enumerate(record_paths, start=1):
        record_id = int(path.stem)
        observations: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
        static_values: dict[str, float] = {}
        static_values_at_zero: dict[str, set[float]] = defaultdict(set)
        exact_triples: set[tuple[int, str, float]] = set()
        same_time_values: dict[tuple[int, str], set[float]] = defaultdict(set)
        primary_same_time_values: dict[tuple[int, str], set[float]] = defaultdict(set)
        local = Counter()
        inside_record_ids: set[int] = set()

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["Time", "Parameter", "Value"]:
                raise ValueError(f"Unexpected record schema in {path.name}: {reader.fieldnames}")
            for row_index, row in enumerate(reader, start=1):
                local["raw_rows"] += 1
                elapsed = parse_elapsed_minutes(row.get("Time", ""))
                if elapsed is None:
                    local["invalid_time_rows"] += 1
                    continue
                parameter = (row.get("Parameter") or "").strip()
                if not parameter:
                    local["empty_parameter_rows"] += 1
                    continue
                if parameter not in allowed_parameters:
                    local["unknown_parameter_rows"] += 1
                    parameter_unknown[parameter] += 1
                    continue
                raw_value = safe_float(row.get("Value"))
                if raw_value is None:
                    local["invalid_numeric_rows"] += 1
                    continue
                # Apply physiological range validation
                value = validate_physiological_value(parameter, raw_value) if raw_value is not None else -1
                if value == config["missing_sentinel"]:
                    local["sentinel_rows"] += 1
                    parameter_sentinel[parameter] += 1
                    continue
                if elapsed < 0:
                    local["negative_time_rows"] += 1
                    continue
                if elapsed == 2880:
                    local["exact_48h_rows"] += 1
                elif elapsed > 2880:
                    local["after_48h_rows"] += 1
                    continue

                triple = (elapsed, parameter, value)
                if triple in exact_triples:
                    local["exact_duplicate_rows"] += 1
                exact_triples.add(triple)
                same_time_values[(elapsed, parameter)].add(value)
                if elapsed < 2880:
                    primary_same_time_values[(elapsed, parameter)].add(value)

                if parameter == "RecordID":
                    inside_record_ids.add(int(value))
                    continue
                if elapsed == 0 and parameter in config["static_parameters"]:
                    static_values_at_zero[parameter].add(value)
                    static_values.setdefault(parameter, value)
                if parameter in config["dynamic_parameters"] and elapsed <= 2880:
                    observations[parameter].append((elapsed, row_index, value))
                    local["valid_dynamic_rows_including_time0_weight"] += 1
                    if not (parameter == "Weight" and elapsed == 0):
                        local["valid_dynamic_rows_excluding_time0_weight"] += 1
                    if elapsed < 2880 and value != config["missing_sentinel"]:
                        # Only collect non-sentinel values for distribution statistics
                        parameter_valid_values[parameter].append(value)
                        parameter_valid_patients[parameter].add(record_id)

        local["conflicting_time_parameter_keys"] = sum(len(values) > 1 for values in same_time_values.values())
        local["conflicting_time_parameter_extra_values"] = sum(max(0, len(values) - 1) for values in same_time_values.values())
        local["primary_conflicting_time_parameter_keys"] = sum(
            len(values) > 1 for values in primary_same_time_values.values()
        )
        local["primary_conflicting_time_parameter_extra_values"] = sum(
            max(0, len(values) - 1) for values in primary_same_time_values.values()
        )
        local["static_conflict_parameters"] = sum(len(values) > 1 for values in static_values_at_zero.values())
        local["filename_recordid_mismatch"] = int(inside_record_ids != {record_id})
        for key in QUALITY_COLUMNS:
            local.setdefault(key, 0)
        for key, value in local.items():
            audit[key] += value
        quality_rows.append({"RecordID": record_id, **{key: int(local[key]) for key in QUALITY_COLUMNS}})

        primary = aggregate_patient(static_values, observations, config, include_48h=False)
        all_rows.append({"RecordID": record_id, **primary})
        if local["exact_48h_rows"]:
            alternate = aggregate_patient(static_values, observations, config, include_48h=True)
            sensitivity_rows.append({"RecordID": record_id, **alternate})

        if path_number % 2000 == 0:
            print(f"Processed {path_number:,}/{len(record_paths):,} record files", flush=True)

    feature_dictionary = build_feature_dictionary(config)
    expected_columns = feature_dictionary["feature_name"].tolist()
    feature_frame = pd.DataFrame(all_rows).reindex(columns=expected_columns).sort_values("RecordID").reset_index(drop=True)
    sensitivity_frame = pd.DataFrame(sensitivity_rows).reindex(columns=expected_columns).sort_values("RecordID").reset_index(drop=True)
    quality_frame = pd.DataFrame(quality_rows).fillna(0).sort_values("RecordID").reset_index(drop=True)
    feature_columns = expected_columns[1:]
    feature_frame[feature_columns] = feature_frame[feature_columns].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if not sensitivity_frame.empty:
        sensitivity_frame[feature_columns] = sensitivity_frame[feature_columns].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    sorted_outcomes = outcomes[["RecordID", "In-hospital_death"]].copy().sort_values("RecordID").reset_index(drop=True)
    split_assignment = deterministic_stratified_split(
        sorted_outcomes["RecordID"].to_numpy(), sorted_outcomes["In-hospital_death"].to_numpy(),
        int(config["random_seed"]), config["split_ratios"],
    )
    sorted_outcomes["split"] = sorted_outcomes["RecordID"].map(split_assignment)
    split_order = pd.CategoricalDtype(["train", "validation", "test"], ordered=True)
    sorted_outcomes["split"] = sorted_outcomes["split"].astype(split_order)

    merged = sorted_outcomes.merge(feature_frame, on="RecordID", how="inner", validate="one_to_one")
    train_mask = merged["split"].eq("train")
    fill_values: dict[str, float] = {}
    all_missing_in_train: list[str] = []
    for column in feature_columns:
        series = pd.to_numeric(merged.loc[train_mask, column], errors="coerce")
        if series.notna().any():
            fill_values[column] = float(series.median())
        else:
            fill_values[column] = 0.0
            all_missing_in_train.append(column)

    imputed = feature_frame.copy()
    imputed[feature_columns] = imputed[feature_columns].fillna(value=fill_values).astype(np.float32)
    if imputed[feature_columns].isna().any().any():
        raise AssertionError("Imputed feature matrix still contains missing values")
    if not np.isfinite(imputed[feature_columns].to_numpy(dtype=np.float64)).all():
        raise AssertionError("Imputed feature matrix contains non-finite values")

    split_outputs: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        ids = set(sorted_outcomes.loc[sorted_outcomes["split"].eq(split), "RecordID"].astype(int))
        split_outputs[split] = imputed[imputed["RecordID"].isin(ids)].sort_values("RecordID").reset_index(drop=True)

    leakage_terms = {"in_hospital_death", "survival", "length_of_stay", "sofa", "saps_i"}
    leakage_hits = [
        column for column in feature_columns
        if any(term in column.lower().replace("-", "_") for term in leakage_terms)
    ]
    if leakage_hits:
        raise AssertionError(f"Forbidden outcome-derived columns in X: {leakage_hits}")

    split_summary = (
        sorted_outcomes.groupby("split", observed=True)["In-hospital_death"]
        .agg(rows="size", deaths="sum", death_prevalence="mean")
        .reset_index()
    )
    split_summary["survivors"] = split_summary["rows"] - split_summary["deaths"]
    split_summary = split_summary[["split", "rows", "deaths", "survivors", "death_prevalence"]]

    missingness = pd.DataFrame({
        "feature_name": feature_columns,
        "missing_count_unimputed": [int(feature_frame[column].isna().sum()) for column in feature_columns],
        "missing_fraction_unimputed": [float(feature_frame[column].isna().mean()) for column in feature_columns],
        "fill_value_from_train": [fill_values[column] for column in feature_columns],
        "all_missing_in_train": [column in all_missing_in_train for column in feature_columns],
    }).sort_values(["missing_fraction_unimputed", "feature_name"], ascending=[False, True])

    parameter_distribution_rows = []
    for parameter in config["dynamic_parameters"]:
        values = np.asarray(parameter_valid_values.get(parameter, []), dtype=np.float64)
        # Filter out sentinel values (-1) before calculating statistics
        valid_values = values[values != config["missing_sentinel"]]
        parameter_distribution_rows.append({
            "Parameter": parameter,
            "valid_observations": int(valid_values.size),  # Count only non-sentinel values
            "patients_with_valid_value": len(parameter_valid_patients.get(parameter, set())),
            "patient_coverage": len(parameter_valid_patients.get(parameter, set())) / len(record_paths),
            "min": float(np.min(valid_values)) if valid_values.size else math.nan,
            "p01": float(np.quantile(valid_values, 0.01)) if valid_values.size else math.nan,
            "median": float(np.median(valid_values)) if valid_values.size else math.nan,
            "p99": float(np.quantile(valid_values, 0.99)) if valid_values.size else math.nan,
            "max": float(np.max(valid_values)) if valid_values.size else math.nan,
            "sentinel_minus1_rows": int(parameter_sentinel.get(parameter, 0)),
        })
    parameter_distribution = pd.DataFrame(parameter_distribution_rows).sort_values("Parameter")

    compresslevel = int(config["gzip_compresslevel"])
    paths = {
        "features_unimputed": out_dir / "patient_features_unimputed.csv.gz",
        "boundary_sensitivity": out_dir / "patient_features_48h_inclusive_affected_rows.csv.gz",
        "labels_splits": out_dir / "labels_and_splits.csv",
        "x_train": out_dir / "X_train_imputed.csv.gz",
        "x_validation": out_dir / "X_validation_imputed.csv.gz",
        "x_test": out_dir / "X_test_imputed.csv.gz",
        "feature_dictionary": out_dir / "feature_dictionary.csv",
        "quality_by_stay": out_dir / "quality_flags_by_stay.csv",
        "feature_missingness": out_dir / "feature_missingness.csv",
        "parameter_distributions": out_dir / "parameter_distributions.csv",
        "split_summary": out_dir / "split_summary.csv",
        "preprocessing_state": out_dir / "preprocessing_state.json",
    }
    write_csv_gzip(feature_frame, paths["features_unimputed"], compresslevel)
    write_csv_gzip(sensitivity_frame, paths["boundary_sensitivity"], compresslevel)
    sorted_outcomes.to_csv(paths["labels_splits"], index=False, lineterminator="\n")
    write_csv_gzip(split_outputs["train"], paths["x_train"], compresslevel)
    write_csv_gzip(split_outputs["validation"], paths["x_validation"], compresslevel)
    write_csv_gzip(split_outputs["test"], paths["x_test"], compresslevel)
    feature_dictionary.to_csv(paths["feature_dictionary"], index=False, lineterminator="\n")
    quality_frame.to_csv(paths["quality_by_stay"], index=False, lineterminator="\n")
    missingness.to_csv(paths["feature_missingness"], index=False, lineterminator="\n")
    parameter_distribution.to_csv(paths["parameter_distributions"], index=False, lineterminator="\n")
    split_summary.to_csv(paths["split_summary"], index=False, lineterminator="\n")

    preprocessing_state = {
        "version": "mortality_v1",
        "random_seed": config["random_seed"],
        "fit_scope": "train only",
        "imputation_method": config["imputation"],
        "scaling": config["scaling"],
        "feature_columns": feature_columns,
        "feature_schema_sha256": sha256_json(feature_columns),
        "fill_values": fill_values,
        "all_missing_in_train": all_missing_in_train,
        "drop_before_model_fit": ["RecordID"],
        "outcome_columns_not_used": ["SAPS-I", "SOFA", "Length_of_stay", "Survival"],
    }
    with paths["preprocessing_state"].open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(preprocessing_state, handle, ensure_ascii=False, indent=2, allow_nan=False)

    output_hashes = {
        _rel(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths.values()
    }
    source_manifest = add_source_manifest(config, record_paths)
    with (out_dir / "source_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(source_manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)

    split_sets = {
        split: set(sorted_outcomes.loc[sorted_outcomes["split"].eq(split), "RecordID"].astype(int))
        for split in ("train", "validation", "test")
    }
    split_disjoint = not (
        (split_sets["train"] & split_sets["validation"])
        or (split_sets["train"] & split_sets["test"])
        or (split_sets["validation"] & split_sets["test"])
    )
    summary = {
        "status": "PASS" if split_disjoint and len(feature_frame) == len(outcomes) and not leakage_hits else "FAIL",
        "decision": "GO_TO_MODELING_AFTER_TEAM_ACCEPTS_TASK_DEFINITION",
        "task_definition": config["task"],
        "primary_time_rule": config["time_rule_primary"],
        "source": {
            "record_files": len(record_paths),
            "outcome_rows": len(outcomes),
            "raw_record_rows": int(audit["raw_rows"]),
            "death_count": int(outcomes["In-hospital_death"].sum()),
            "death_prevalence": float(outcomes["In-hospital_death"].mean()),
        },
        "output": {
            "feature_rows": len(feature_frame),
            "model_feature_columns": len(feature_columns),
            "unimputed_missing_cells": int(feature_frame[feature_columns].isna().sum().sum()),
            "imputed_missing_cells": int(imputed[feature_columns].isna().sum().sum()),
            "all_missing_in_train_columns": len(all_missing_in_train),
            "boundary_sensitivity_rows": len(sensitivity_frame),
            "feature_schema_sha256": preprocessing_state["feature_schema_sha256"],
        },
        "audits": {key: int(value) for key, value in sorted(audit.items())},
        "unknown_parameters": dict(sorted(parameter_unknown.items())),
        "split_summary": split_summary.to_dict(orient="records"),
        "qa": {
            "feature_row_count_matches_outcomes": len(feature_frame) == len(outcomes),
            "record_id_unique": not feature_frame["RecordID"].duplicated().any(),
            "record_id_sets_match": set(feature_frame["RecordID"].astype(int)) == outcome_ids,
            "split_disjoint": split_disjoint,
            "split_union_complete": set().union(*split_sets.values()) == outcome_ids,
            "label_binary_complete": bool(
                not outcomes["In-hospital_death"].isna().any()
                and set(outcomes["In-hospital_death"].unique()).issubset({0, 1})
            ),
            "leakage_feature_hits": leakage_hits,
            "imputed_no_missing": not imputed[feature_columns].isna().any().any(),
            "imputed_all_finite": bool(np.isfinite(imputed[feature_columns].to_numpy(dtype=np.float64)).all()),
            "primary_excludes_48h": True,
        },
        "policies": {
            "missing": "-1 -> missing; absent summaries stay missing before train-only imputation",
            "duplicates": config["duplicate_policy"],
            "outliers": config["outlier_policy"],
            "scaling": config["scaling"],
            "leakage": "Only In-hospital_death is exported as y; all other outcome columns are excluded from X",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "outputs": output_hashes,
        "remaining_team_decisions": [
            "Confirm the landmark task: use first 48 hours to predict final in-hospital death.",
            "Have a medical reviewer approve the PHYSIOLOGICAL_RANGES bounds (out-of-range values are set to missing).",
            "Choose model-specific scaling/selection inside training folds; this preprocessing does not train a model.",
            "Patient-level repeat admissions cannot be checked because only stay-level RecordID is available.",
        ],
    }
    if not all(value for key, value in summary["qa"].items() if key != "leakage_feature_hits"):
        summary["status"] = "FAIL"
    if summary["qa"]["leakage_feature_hits"]:
        summary["status"] = "FAIL"
    with (out_dir / "preprocessing_summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    build_labels_and_baselines(out_dir)
    print(json.dumps({
        "status": summary["status"],
        "feature_rows": len(feature_frame),
        "features": len(feature_columns),
        "boundary_rows": len(sensitivity_frame),
        "elapsed_seconds": summary["runtime"]["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    return summary


def build_labels_and_baselines(out_dir: Path | str | None = None) -> pd.DataFrame:
    """labels_and_splits.csv + SAPS-I / SOFA, for the L2 (bedside score) baseline rung only.

    SAPS-I and SOFA are outcome-adjacent and stay OUT of the feature matrix; the L2 rung needs
    them as *prediction scores*. Exporting them here means modelling code never touches
    outcomes.csv, whose Length_of_stay / Survival columns would leak the outcome in one careless
    merge. Raw `-1` means "not computable": left numeric it would rank those patients as lowest
    risk, yet they are HIGHER risk (SAPS-I -1 rows: 20.7 % mortality vs 14.4 % overall), so it
    becomes NaN plus an explicit `<score>_available` flag.
    """
    out_dir = Path(out_dir) if out_dir is not None else OUTPUT_DIR
    labels = pd.read_csv(out_dir / "labels_and_splits.csv")
    outcomes = pd.read_csv(OUTCOMES_PATH)
    merged = labels.merge(
        outcomes[["RecordID", "SAPS-I", "SOFA", "In-hospital_death"]],
        on="RecordID", how="left", suffixes=("", "_src"), validate="one_to_one",
    )
    if len(merged) != len(labels):
        raise AssertionError(f"merge changed rowcount: {len(labels)} -> {len(merged)}")
    if not (merged["In-hospital_death"] == merged["In-hospital_death_src"]).all():
        raise AssertionError("label disagreement between splits file and outcomes.csv")
    merged = merged.drop(columns=["In-hospital_death_src"])
    for name, raw in (("saps_i", "SAPS-I"), ("sofa", "SOFA")):
        merged[f"{name}_available"] = (merged[raw] != -1).astype(int)
        merged[name] = merged[raw].where(merged[raw] != -1)
    ordered = merged[["RecordID", "In-hospital_death", "split",
                      "saps_i", "saps_i_available", "sofa", "sofa_available"]]
    leaked = {"Length_of_stay", "Survival"} & set(ordered.columns)
    if leaked:
        raise AssertionError(f"outcome-derived columns must not be exported: {leaked}")
    ordered.to_csv(out_dir / "labels_and_baselines.csv", index=False, lineterminator="\n")
    return ordered


if __name__ == "__main__":
    main()
