"""Unit tests for Phase 3 derived features: merged blood pressure and BMI.

Covers the two behaviour changes from preprocessing_fix_plan.md Phase 3:
  W1 - merged BP streams with per-window invasive preference + has_arterial_line
  W5 - static_bmi derived from the 00:00 Height/Weight descriptors

Also pins the schema contract: build_feature_dictionary() must describe exactly
the key set that aggregate_patient() emits. If those drift apart, the pipeline's
reindex(columns=...) silently drops undeclared keys or fabricates all-NaN
columns, which is the kind of failure that survives every downstream check.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.task2_mortality.preprocess import (  # noqa: E402
    aggregate_patient,
    build_feature_dictionary,
    derive_bmi,
    load_config,
)


STATICS = {"Age": 70.0, "Gender": 1.0, "Height": 170.0, "ICUType": 3.0, "Weight": 80.0}


class DeriveBmiTests(unittest.TestCase):
    def test_normal_case(self) -> None:
        self.assertAlmostEqual(derive_bmi(170.0, 80.0), 80.0 / 1.7**2, places=9)

    def test_missing_inputs_give_nan(self) -> None:
        self.assertTrue(math.isnan(derive_bmi(math.nan, 80.0)))
        self.assertTrue(math.isnan(derive_bmi(170.0, math.nan)))
        self.assertTrue(math.isnan(derive_bmi(math.nan, math.nan)))

    def test_non_positive_height_gives_nan_not_zero_division(self) -> None:
        self.assertTrue(math.isnan(derive_bmi(0.0, 80.0)))
        self.assertTrue(math.isnan(derive_bmi(-5.0, 80.0)))


class MergedBloodPressureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_invasive_preferred_when_present(self) -> None:
        observations = {
            "SysABP": [(10, 1, 120.0), (600, 2, 118.0)],
            "NISysABP": [(10, 3, 200.0)],
        }
        features = aggregate_patient(STATICS, observations, self.config, include_48h=False)
        self.assertEqual(features["bp_sys__0_24h__mean"], 119.0)
        self.assertEqual(features["bp__0_24h__has_arterial_line"], 1.0)

    def test_falls_back_to_noninvasive_when_invasive_absent(self) -> None:
        observations = {"NISysABP": [(10, 1, 125.0), (600, 2, 135.0)]}
        features = aggregate_patient(STATICS, observations, self.config, include_48h=False)
        self.assertEqual(features["bp_sys__0_24h__mean"], 130.0)
        self.assertEqual(features["bp__0_24h__has_arterial_line"], 0.0)

    def test_preference_is_decided_per_window(self) -> None:
        # Invasive only in 0-24h; non-invasive spans both windows.
        observations = {
            "SysABP": [(10, 1, 120.0)],
            "NISysABP": [(10, 2, 125.0), (2000, 3, 140.0)],
        }
        features = aggregate_patient(STATICS, observations, self.config, include_48h=False)
        self.assertEqual(features["bp_sys__0_24h__mean"], 120.0)
        self.assertEqual(features["bp__0_24h__has_arterial_line"], 1.0)
        self.assertEqual(features["bp_sys__24_48h__mean"], 140.0)
        self.assertEqual(features["bp__24_48h__has_arterial_line"], 0.0)

    def test_neither_stream_present_is_unmeasured(self) -> None:
        features = aggregate_patient(STATICS, {"HR": [(10, 1, 80.0)]}, self.config, include_48h=False)
        self.assertEqual(features["bp_sys__0_48h__measured"], 0.0)
        self.assertEqual(features["bp_sys__0_48h__count"], 0.0)
        self.assertTrue(math.isnan(features["bp_sys__0_48h__mean"]))
        self.assertEqual(features["bp__0_48h__has_arterial_line"], 0.0)

    def test_source_columns_are_retained_unchanged(self) -> None:
        observations = {
            "SysABP": [(10, 1, 120.0)],
            "NISysABP": [(10, 2, 125.0)],
        }
        features = aggregate_patient(STATICS, observations, self.config, include_48h=False)
        self.assertEqual(features["sysabp__0_24h__mean"], 120.0)
        self.assertEqual(features["nisysabp__0_24h__mean"], 125.0)

    def test_derived_streams_excluded_from_record_counters(self) -> None:
        # Two raw parameters observed once each -> distinct 2, total 2.
        # Merged BP streams must not inflate these counters.
        observations = {"SysABP": [(10, 1, 120.0)], "HR": [(10, 2, 80.0)]}
        features = aggregate_patient(STATICS, observations, self.config, include_48h=False)
        self.assertEqual(features["record__0_24h__distinct_parameter_count"], 2.0)
        self.assertEqual(features["record__0_24h__valid_observation_count"], 2.0)


class SchemaContractTests(unittest.TestCase):
    def test_dictionary_describes_exactly_what_aggregator_emits(self) -> None:
        config = load_config()
        declared = set(build_feature_dictionary(config)["feature_name"]) - {"RecordID"}
        observations = {
            "SysABP": [(10, 1, 120.0)],
            "NISysABP": [(10, 2, 125.0), (2000, 3, 130.0)],
            "Urine": [(10, 4, 0.0)],
            "MechVent": [(10, 5, 1.0)],
        }
        emitted = set(aggregate_patient(STATICS, observations, config, include_48h=False))
        self.assertEqual(
            emitted - declared, set(), "aggregator emits keys the dictionary omits; reindex would drop them"
        )
        self.assertEqual(
            declared - emitted, set(), "dictionary declares keys the aggregator never emits; they would be all-NaN"
        )

    def test_feature_dictionary_has_no_duplicate_names(self) -> None:
        names = build_feature_dictionary(load_config())["feature_name"].tolist()
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
