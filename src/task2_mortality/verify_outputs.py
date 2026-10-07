"""Independent end-to-end check of the task 2 pre-processing outputs.

Re-reads every artefact and re-derives what can be re-derived: row counts, label/split
consistency, schema hash, leakage guard, train-only median fill values, imputed matrices,
the 48:00 boundary file, one-hot integrity, and every output's SHA-256 against the run summary.
Any failed check raises AssertionError.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "preprocessed" / "task2_mortality"


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(output_dir: Path | str | None = None, write: bool = False) -> dict:
    OUTPUT_DIR = Path(output_dir) if output_dir is not None else globals()["OUTPUT_DIR"]
    with (OUTPUT_DIR / "preprocessing_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (OUTPUT_DIR / "preprocessing_state.json").open(encoding="utf-8") as handle:
        state = json.load(handle)

    labels = pd.read_csv(OUTPUT_DIR / "labels_and_splits.csv")
    features = pd.read_csv(OUTPUT_DIR / "patient_features_unimputed.csv.gz")
    boundary = pd.read_csv(OUTPUT_DIR / "patient_features_48h_inclusive_affected_rows.csv.gz")
    dictionary = pd.read_csv(OUTPUT_DIR / "feature_dictionary.csv")
    quality = pd.read_csv(OUTPUT_DIR / "quality_flags_by_stay.csv")
    split_frames = {
        "train": pd.read_csv(OUTPUT_DIR / "X_train_imputed.csv.gz"),
        "validation": pd.read_csv(OUTPUT_DIR / "X_validation_imputed.csv.gz"),
        "test": pd.read_csv(OUTPUT_DIR / "X_test_imputed.csv.gz"),
    }

    assert summary["status"] == "PASS"
    # Cohort size is derived from the run summary rather than hardcoded, so that
    # legitimate cohort changes (e.g. the Phase 1 negative-LOS exclusion) do not
    # silently rot this verifier. Cross-checks still pin every artifact together.
    expected_rows = int(summary["output"]["feature_rows"])
    assert len(labels) == len(features) == expected_rows
    assert int(summary["source"]["outcome_rows"]) == expected_rows
    assert labels["RecordID"].is_unique and features["RecordID"].is_unique
    assert set(labels["RecordID"]) == set(features["RecordID"])
    assert set(labels["In-hospital_death"]) == {0, 1}
    assert labels["In-hospital_death"].isna().sum() == 0
    assert int(labels["In-hospital_death"].sum()) == int(summary["source"]["death_count"])
    expected_split_rows = {row["split"]: int(row["rows"]) for row in summary["split_summary"]}
    assert labels["split"].value_counts().to_dict() == expected_split_rows
    assert sum(expected_split_rows.values()) == expected_rows

    expected_schema = state["feature_columns"]
    assert features.columns.tolist() == ["RecordID", *expected_schema]
    assert dictionary["feature_name"].tolist() == ["RecordID", *expected_schema]
    assert state["feature_schema_sha256"] == sha256_json(expected_schema)
    assert len(expected_schema) == len(set(expected_schema))
    assert len(expected_schema) == int(summary["output"]["model_feature_columns"])

    forbidden = ("in_hospital_death", "survival", "length_of_stay", "sofa", "saps_i")
    assert not [column for column in expected_schema if any(term in column.lower().replace("-", "_") for term in forbidden)]

    all_split_ids: set[int] = set()
    verified_imputed_cells = 0
    for split, frame in split_frames.items():
        expected_ids = set(labels.loc[labels["split"].eq(split), "RecordID"])
        assert set(frame["RecordID"]) == expected_ids
        assert frame.columns.tolist() == ["RecordID", *expected_schema]
        values = frame[expected_schema].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all()
        expected_values = (
            features[features["RecordID"].isin(expected_ids)]
            .sort_values("RecordID")
            .reset_index(drop=True)[expected_schema]
            .fillna(value=state["fill_values"])
            .to_numpy(dtype=np.float64)
        )
        assert np.allclose(values, expected_values, rtol=2e-6, atol=2e-6)
        verified_imputed_cells += int(values.size)
        assert not (all_split_ids & expected_ids)
        all_split_ids |= expected_ids
    assert all_split_ids == set(labels["RecordID"])

    train_ids = set(labels.loc[labels["split"].eq("train"), "RecordID"])
    train_unimputed = features[features["RecordID"].isin(train_ids)]
    mismatched_fill_values = []
    for column in expected_schema:
        clean = pd.to_numeric(train_unimputed[column], errors="coerce").dropna()
        expected_fill = float(clean.median()) if len(clean) else 0.0
        if not np.isclose(expected_fill, float(state["fill_values"][column]), rtol=2e-6, atol=2e-6):
            mismatched_fill_values.append(column)
    assert not mismatched_fill_values, mismatched_fill_values[:10]

    affected_ids = set(quality.loc[quality.get("exact_48h_rows", 0) > 0, "RecordID"])
    assert len(boundary) == len(affected_ids) == int(summary["output"]["boundary_sensitivity_rows"])
    assert set(boundary["RecordID"]) == affected_ids
    assert boundary.columns.tolist() == features.columns.tolist()

    gender_columns = ["static_gender_0", "static_gender_1", "static_gender_missing"]
    icu_columns = [
        "static_icutype_1", "static_icutype_2", "static_icutype_3",
        "static_icutype_4", "static_icutype_missing",
    ]
    assert (features[gender_columns].sum(axis=1) == 1).all()
    assert (features[icu_columns].sum(axis=1) == 1).all()

    for relative_path, metadata in summary["outputs"].items():
        artifact_path = OUTPUT_DIR / Path(relative_path).name
        assert artifact_path.exists()
        assert artifact_path.stat().st_size == metadata["bytes"]
        assert sha256_file(artifact_path) == metadata["sha256"]

    no_dynamic = int((features["record__0_48h__valid_observation_count"] == 0).sum())
    no_dynamic_excluding_descriptor_weight = int(
        (quality["valid_dynamic_rows_excluding_time0_weight"] == 0).sum()
    )
    verification = {
        "status": "PASS",
        "rows": len(features),
        "model_features": len(expected_schema),
        "split_rows": {split: len(frame) for split, frame in split_frames.items()},
        "split_death_prevalence": labels.groupby("split")["In-hospital_death"].mean().to_dict(),
        "stays_without_valid_dynamic_observations": no_dynamic,
        "stays_without_valid_dynamic_observations_excluding_time0_weight": no_dynamic_excluding_descriptor_weight,
        "boundary_affected_stays": len(boundary),
        "same_time_conflicting_keys": int(quality.get("conflicting_time_parameter_keys", pd.Series(dtype=int)).sum()),
        "same_time_conflicting_extra_values": int(quality["conflicting_time_parameter_extra_values"].sum()),
        "primary_same_time_conflicting_keys": int(quality["primary_conflicting_time_parameter_keys"].sum()),
        "primary_same_time_conflicting_extra_values": int(
            quality["primary_conflicting_time_parameter_extra_values"].sum()
        ),
        "exact_duplicate_rows": int(quality.get("exact_duplicate_rows", pd.Series(dtype=int)).sum()),
        "verified_train_only_fill_values": len(expected_schema),
        "verified_imputed_cells": verified_imputed_cells,
        "verified_output_hashes": len(summary["outputs"]),
        "imputed_outputs_all_finite": True,
        "feature_schema_sha256": state["feature_schema_sha256"],
    }
    if write:
        with (OUTPUT_DIR / "independent_verification.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(verification, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return verification


if __name__ == "__main__":
    print(json.dumps(main(write=True), ensure_ascii=False, indent=2))
