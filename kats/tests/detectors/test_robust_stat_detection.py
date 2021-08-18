# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import math
import os
import pkgutil
import re
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.robust_stat_detection import RobustStatDetector
from sklearn.datasets import make_spd_matrix

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class RobustStatTest(TestCase):
    def test_no_change(self) -> None:
        np.random.seed(10)
        df_noregress = pd.DataFrame({"no_change": [math.sin(i) for i in range(60)]})

        df_noregress["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_noregress)
        detector = RobustStatDetector(timeseries)
        change_points = detector.detector()

        self.assertEqual(len(change_points), 0)
        detector.plot(change_points)

    def test_increasing_detection(self) -> None:
        np.random.seed(10)
        df_increase = pd.DataFrame(
            {
                "increase": [
                    math.sin(i) if i < 41 else math.sin(i) + 17 for i in range(60)
                ]
            }
        )

        df_increase["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_increase)
        detector = RobustStatDetector(timeseries)
        change_points = detector.detector()

        self.assertEqual(len(change_points), 1)
        detector.plot(change_points)

    def test_decreasing_detection(self) -> None:
        np.random.seed(10)
        df_decrease = pd.DataFrame(
            {
                "decrease": [
                    math.sin(i) if i < 23 else math.sin(i) - 25 for i in range(60)
                ]
            }
        )

        df_decrease["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_decrease)
        detector = RobustStatDetector(timeseries)
        change_points = detector.detector()

        self.assertEqual(len(change_points), 1)
        detector.plot(change_points)

    def test_spike_change_pos(self) -> None:
        np.random.seed(10)
        df_slope_change = pd.DataFrame(
            {"spike": [math.sin(i) if i != 27 else 30 * math.sin(i) for i in range(60)]}
        )

        df_slope_change["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_slope_change)
        detector = RobustStatDetector(timeseries)
        change_points = detector.detector()

        self.assertEqual(len(change_points), 2)
        detector.plot(change_points)

    def test_spike_change_neg(self) -> None:
        np.random.seed(10)
        df_slope_change = pd.DataFrame(
            {
                "spike": [
                    math.sin(i) if i != 27 else -30 * math.sin(i) for i in range(60)
                ]
            }
        )

        df_slope_change["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_slope_change)
        detector = RobustStatDetector(timeseries)
        change_points = detector.detector()

        self.assertEqual(len(change_points), 2)

    def test_rasie_error(self) -> None:
        D = 10
        random_state = 10
        np.random.seed(random_state)
        mean1 = np.ones(D)
        mean2 = mean1 * 2
        sigma = make_spd_matrix(D, random_state=random_state)

        df_increase = pd.DataFrame(
            np.concatenate(
                [
                    np.random.multivariate_normal(mean1, sigma, 60),
                    np.random.multivariate_normal(mean2, sigma, 30),
                ]
            )
        )

        df_increase["time"] = pd.Series(pd.date_range("2019-01-01", "2019-04-01"))

        timeseries_multi = TimeSeriesData(df_increase)
        with self.assertRaises(ValueError):
            RobustStatDetector(timeseries_multi)

    def test_ts_without_name(self) -> None:
        n = 10
        time = pd.Series(pd.date_range(start="2018-01-01", periods=n, freq="D"))
        value = pd.Series(np.arange(n))
        ts = TimeSeriesData(time=time, value=value)

        detector = RobustStatDetector(ts)
        change_points = detector.detector()
        detector.plot(change_points)
