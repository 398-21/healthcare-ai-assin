from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.task2_mortality.preprocess import (  # noqa: E402
    Window,
    deterministic_stratified_split,
    parse_elapsed_minutes,
    summarize_observations,
)


class ParseElapsedMinutesTests(unittest.TestCase):
    def test_elapsed_hours_do_not_wrap(self) -> None:
        self.assertEqual(parse_elapsed_minutes("00:00"), 0)
        self.assertEqual(parse_elapsed_minutes("24:00"), 1440)
        self.assertEqual(parse_elapsed_minutes("47:59"), 2879)
        self.assertEqual(parse_elapsed_minutes("48:00"), 2880)

    def test_invalid_times(self) -> None:
        for value in ("", "bad", "12:60", "-1:00"):
            self.assertIsNone(parse_elapsed_minutes(value))


class AggregationTests(unittest.TestCase):
    def test_primary_excludes_48h_and_sensitivity_includes_it(self) -> None:
        observations = [(0, 1, 10.0), (2879, 2, 20.0), (2880, 3, 99.0)]
        window = Window("0_48h", 0, 2880)
        primary = summarize_observations(observations, window, include_48h=False)
        alternate = summarize_observations(observations, window, include_48h=True)
        self.assertEqual(primary["count"], 2)
        self.assertEqual(primary["last"], 20.0)
        self.assertEqual(alternate["count"], 3)
        self.assertEqual(alternate["last"], 99.0)

    def test_absent_variable_has_zero_count_and_missing_summaries(self) -> None:
        summary = summarize_observations([], Window("0_24h", 0, 1440))
        self.assertEqual(summary, {"measured": 0.0, "count": 0.0})

    def test_same_time_slope_is_missing(self) -> None:
        summary = summarize_observations([(10, 1, 1.0), (10, 2, 2.0)], Window("0_24h", 0, 1440))
        self.assertTrue(np.isnan(summary["slope_per_hour"]))

    def test_same_time_rule_is_deterministic(self) -> None:
        observations = [(10, 2, 8.0), (10, 1, 2.0), (20, 3, 10.0)]
        median = summarize_observations(observations, Window("0_24h", 0, 1440), same_time_rule="median")
        summed = summarize_observations(observations, Window("0_24h", 0, 1440), same_time_rule="sum")
        self.assertEqual(median["first"], 5.0)
        self.assertEqual(summed["first"], 10.0)
        self.assertEqual(median["count"], 3.0)


class SplitTests(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_disjoint(self) -> None:
        ids = np.arange(1000, 1100)
        labels = np.asarray([0] * 80 + [1] * 20)
        ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
        first = deterministic_stratified_split(ids, labels, 42, ratios)
        second = deterministic_stratified_split(ids, labels, 42, ratios)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(ids.tolist()))
        self.assertEqual(Counter(first.values()), Counter({"train": 70, "validation": 15, "test": 15}))


if __name__ == "__main__":
    unittest.main()
