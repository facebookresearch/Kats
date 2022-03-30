# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.compat.pandas import assert_frame_equal
from kats.utils import feature_engineering as fe


class FeatureEngineeringTest(TestCase):
    def assertDictAlmostEqual(
        self, expected: Dict[str, Any], features: Dict[str, Any], places: int = 4
    ) -> None:
        """Compares that two dictionaries are floating-point almost equal.

        Note: the dictionaries may or may not contain floating-point values.

        Args:
          expected: the expected dictionary of values.
          actual: the actual dictionary of values.
          places: the number of decimal places for floating point comparisons.
        """

        self.assertEqual(expected.keys(), features.keys())
        for k, v in expected.items():
            if isinstance(v, float):
                if np.isnan(v):
                    self.assertTrue(np.isnan(features[k]), msg=f"{k} differ")
                else:
                    self.assertAlmostEqual(v, features[k], places=4, msg=f"{k} differ")
            else:
                self.assertEqual(v, features[k], msg=f"{k} differ")

    def test_datetime_features(self) -> None:
        dates = pd.date_range("2021-01-22", "2021-01-31", tz="US/Pacific").tolist()
        dates += pd.date_range("2022-01-30", "2022-02-04", tz="US/Pacific").tolist()
        values = [10, 2, 11, 5, 13, 18, 4, 14, 17, 9] + [np.nan] * 6
        self.assertEqual(len(values), len(dates))
        expected = pd.DataFrame(
            {
                "val": values,
                "year": [2021] * 10 + [2022] * 6,
                "month": [1] * 12 + [2] * 4,
                "day": list(range(22, 32)) + [30, 31, 1, 2, 3, 4],
                "dayofweek": [4, 5, 6] + list(range(7)) + [6, 0, 1, 2, 3, 4],
                "dayofyear": list(range(22, 32)) + list(range(30, 36)),
                "quarter": [1] * 16,
                "season": [0] * 12 + [1] * 4,
                "weekofyear": [3] * 3 + [4] * 8 + [5] * 5,
                "weekofmonth": [4] * 3 + [5] * 8 + [6, 1, 1, 1, 1],
                "is_weekend": [False, True, True]
                + [False] * 5
                + [True] * 3
                + [False] * 5,
                "is_leap_year": [False] * 16,
                "is_leap_day": [False] * 16,
                "is_month_end": [False] * 9 + [True, False, True] + [False] * 4,
                "is_quarter_end": [False] * 16,
                "hour": [0] * 16,
                "minute": [0] * 16,
                "second": [0] * 16,
                "milliseconds": [0.0] * 16,
                "quarterhour": [1] * 16,
                "hourofweek": [96, 120, 144]
                + list(range(0, 168, 24))
                + [144, 0, 24, 48, 72, 96],
                "daytime": [0] * 16,
                "dst": [0.0] * 16,
                "utcoffset": [-28800.0] * 16,
            },
            index=dates,
        )
        result = fe.datetime_features(pd.Series(values, index=dates, name="val"))
        assert_frame_equal(expected, result)

    def test_date_features(self) -> None:
        dates = pd.date_range("2021-01-22", "2021-01-31", tz="US/Pacific").tolist()
        dates += pd.date_range("2022-01-30", "2022-02-04", tz="US/Pacific").tolist()
        values = [10, 2, 11, 5, 13, 18, 4, 14, 17, 9] + [np.nan] * 6
        self.assertEqual(len(values), len(dates))
        expected = pd.DataFrame(
            {
                "val": values,
                "year": [2021] * 10 + [2022] * 6,
                "month": [1] * 12 + [2] * 4,
                "day": list(range(22, 32)) + [30, 31, 1, 2, 3, 4],
                "dayofweek": [4, 5, 6] + list(range(7)) + [6, 0, 1, 2, 3, 4],
                "dayofyear": list(range(22, 32)) + list(range(30, 36)),
                "quarter": [1] * 16,
                "season": [0] * 12 + [1] * 4,
                "weekofyear": [3] * 3 + [4] * 8 + [5] * 5,
                "weekofmonth": [4] * 3 + [5] * 8 + [6, 1, 1, 1, 1],
                "is_weekend": [False, True, True]
                + [False] * 5
                + [True] * 3
                + [False] * 5,
                "is_leap_year": [False] * 16,
                "is_leap_day": [False] * 16,
                "is_month_end": [False] * 9 + [True, False, True] + [False] * 4,
                "is_quarter_end": [False] * 16,
            },
            index=dates,
        )
        result = fe.date_features(pd.Series(values, index=dates, name="val"))
        assert_frame_equal(expected, result)

    def test_time_features(self) -> None:
        dates = pd.date_range("2021-01-22", "2021-01-31", tz="US/Pacific").tolist()
        dates += pd.date_range("2022-01-30", "2022-02-04", tz="US/Pacific").tolist()
        values = [10, 2, 11, 5, 13, 18, 4, 14, 17, 9] + [np.nan] * 6
        self.assertEqual(len(values), len(dates))
        expected = pd.DataFrame(
            {
                "val": values,
                "hour": [0] * 16,
                "minute": [0] * 16,
                "second": [0] * 16,
                "milliseconds": [0.0] * 16,
                "quarterhour": [1] * 16,
                "hourofweek": [96, 120, 144]
                + list(range(0, 168, 24))
                + [144, 0, 24, 48, 72, 96],
                "daytime": [0] * 16,
                "dst": [0.0] * 16,
                "utcoffset": [-28800.0] * 16,
            },
            index=dates,
        )
        result = fe.time_features(pd.Series(values, index=dates, name="val"))
        assert_frame_equal(expected, result)

    def test_timestamp_time_features(self) -> None:
        t = pd.Timestamp("2021-01-01T02:03:04.5678", tz="US/Pacific")
        expected = {
            "year": 2021,
            "month": 1,
            "day": 1,
            "dayofweek": 4,
            "dayofyear": 1,
            "quarter": 1,
            "season": 0,
            "weekofyear": 53,
            "weekofmonth": 1,
            "is_weekend": False,
            "is_leap_year": False,
            "is_leap_day": False,
            "is_month_end": False,
            "is_quarter_end": False,
            "hour": 2,
            "minute": 3,
            "second": 4,
            "milliseconds": 567.8,
            "quarterhour": 1,
            "hourofweek": 98,
            "daytime": 0,
            "dst": 0.0,
            "utcoffset": -28800.0,
        }
        result = fe.timestamp_datetime_features(t)
        self.assertDictAlmostEqual(expected, result)

    def test_timestamp_time_features_notz(self) -> None:
        t = pd.Timestamp("2021-01-01T02:03:04.5678")
        expected = {
            "year": 2021,
            "month": 1,
            "day": 1,
            "dayofweek": 4,
            "dayofyear": 1,
            "quarter": 1,
            "season": 0,
            "weekofyear": 53,
            "weekofmonth": 1,
            "is_weekend": False,
            "is_leap_year": False,
            "is_leap_day": False,
            "is_month_end": False,
            "is_quarter_end": False,
            "hour": 2,
            "minute": 3,
            "second": 4,
            "milliseconds": 567.8,
            "quarterhour": 1,
            "hourofweek": 98,
            "daytime": 0,
            "dst": np.nan,
            "utcoffset": np.nan,
        }
        result = fe.timestamp_datetime_features(t)
        self.assertDictAlmostEqual(expected, result)

    def test_circle_encode(self) -> None:
        s32 = np.sqrt(3) / 2
        expected = pd.DataFrame(
            {
                "x_cos": [0.5, s32, 1.0, s32],
                "x_sin": [-s32, -0.5, 0.0, 0.5],
            }
        )
        result = fe.circle_encode(pd.DataFrame({"x": [10, 11, 12, 1]}), {"x": 12})
        assert_frame_equal(expected, result)

    def test_circle_encode_modulo(self) -> None:
        s32 = np.sqrt(3) / 2
        expected = pd.DataFrame(
            {
                "x_cos": [0.5, s32, 1.0, s32],
                "x_sin": [-s32, -0.5, 0.0, 0.5],
            }
        )
        result = fe.circle_encode(
            pd.DataFrame({"x": [10, 11, 12, 13]}), {"x": 12}, modulo=True
        )
        assert_frame_equal(expected, result)
