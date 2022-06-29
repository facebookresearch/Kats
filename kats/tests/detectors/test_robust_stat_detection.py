# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from operator import attrgetter
from typing import Sequence, Tuple
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.robust_stat_detection import (
    RobustStatChangePoint,
    RobustStatDetector,
)
from parameterized.parameterized import parameterized
from sklearn.datasets import make_spd_matrix


class RobustStatTest(TestCase):
    def setUp(self) -> None:
        self.random_seed = 10
        np.random.seed(self.random_seed)
        self.dates = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        # Initialize the change_points
        df_noregress = pd.DataFrame({"no_change": [math.sin(i) for i in range(60)]})

        df_increase = pd.DataFrame(
            {
                "increase": [
                    math.sin(i) if i < 41 else math.sin(i) + 17 for i in range(60)
                ]
            }
        )

        df_decrease = pd.DataFrame(
            {
                "decrease": [
                    math.sin(i) if i < 23 else math.sin(i) - 25 for i in range(60)
                ]
            }
        )

        df_spike_pos = pd.DataFrame(
            {"spike": [math.sin(i) if i != 27 else 30 * math.sin(i) for i in range(60)]}
        )

        df_spike_neg = pd.DataFrame(
            {
                "spike": [
                    math.sin(i) if i != 27 else -30 * math.sin(i) for i in range(60)
                ]
            }
        )

        def _create_change_pt(
            df: pd.DataFrame,
        ) -> Tuple[RobustStatDetector, Sequence[RobustStatChangePoint]]:
            df["time"] = self.dates
            timeseries = TimeSeriesData(df)
            detector = RobustStatDetector(timeseries)
            change_points = detector.detector()
            return detector, change_points

        self.d_noregress, self.change_points_noregress = _create_change_pt(df_noregress)
        self.d_increase, self.change_points_increase = _create_change_pt(df_increase)
        self.d_decrease, self.change_points_decrease = _create_change_pt(df_decrease)
        self.d_spike_pos, self.change_points_spike_pos = _create_change_pt(df_spike_pos)
        self.d_spike_neg, self.change_points_spike_neg = _create_change_pt(df_spike_neg)

        # Time series without name
        n = 10
        time = pd.Series(pd.date_range(start="2018-01-01", periods=n, freq="D"))
        value = pd.Series(np.arange(n))
        ts = TimeSeriesData(time=time, value=value)

        self.d_ts_without_name = RobustStatDetector(ts)
        self.change_points_ts_without_name = self.d_ts_without_name.detector()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["change_points_noregress",
    #  0], ["change_points_increase", 1], ["change_points_decrease", 1],
    #  ["change_points_spike_pos", 2], ["change_points_spike_neg", 2]])`.
    @parameterized.expand(
        [
            ["change_points_noregress", 0],
            ["change_points_increase", 1],
            ["change_points_decrease", 1],
            ["change_points_spike_pos", 2],
            ["change_points_spike_neg", 2],
        ]
    )
    def test_length(self, change_points: str, ans: int) -> None:
        self.assertEqual(len(attrgetter(change_points)(self)), ans)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ["d_noregress", "change_points_noregress"],
            ["d_increase", "change_points_increase"],
            ["d_decrease", "change_points_decrease"],
            ["d_spike_pos", "change_points_spike_pos"],
            ["d_spike_neg", "change_points_spike_neg"],
            ["d_ts_without_name", "change_points_ts_without_name"],
        ]
    )
    def test_plot(self, detector: str, change_points: str) -> None:
        attrgetter(detector)(self).plot(attrgetter(change_points)(self))

    def test_raise_error(self) -> None:
        D = 10
        mean1 = np.ones(D)
        mean2 = mean1 * 2
        sigma = make_spd_matrix(D, random_state=self.random_seed)

        df_increase = pd.DataFrame(
            np.concatenate(
                [
                    np.random.multivariate_normal(mean1, sigma, 60),
                    np.random.multivariate_normal(mean2, sigma, 30),
                ]
            )
        )

        df_increase["time"] = self.dates

        timeseries_multi = TimeSeriesData(df_increase)
        with self.assertRaises(ValueError):
            RobustStatDetector(timeseries_multi)
