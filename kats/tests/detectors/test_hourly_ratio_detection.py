# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.detectors.hourly_ratio_detection import HourlyRatioDetector


class HourlyRatioDectorTest(TestCase):
    def setUp(self) -> None:
        daily_data = load_data("peyton_manning.csv")
        daily_data.columns = ["time", "y"]
        self.ts_data_daily = TimeSeriesData(daily_data)

        self.TSData_empty = TimeSeriesData(pd.DataFrame([], columns=["time", "y"]))

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def data_generation(
        self, freq: str = "H", drop: bool = True, frac: float = 0.95
    ) -> TimeSeriesData:
        time = pd.date_range("2018-01-01", "2020-01-01", freq=freq)
        n = len(time)
        x = np.arange(n)
        values = np.abs(np.sin(30 * x) + 5 * x + 10 * x ** 2) + np.random.randn(n)
        df = pd.DataFrame(values, columns=["value"])
        df["time"] = time
        if drop:
            df = cast(pd.DataFrame, df.sample(frac=frac, replace=False))
        return TimeSeriesData(df)

    def test_detector(self) -> None:
        # test hourly data without missing vlaues
        ts = self.data_generation(drop=False)
        hr = HourlyRatioDetector(ts)
        hr._preprocess()
        hr.detector()
        hr.plot()

        # test hourly data with missing values
        ts = self.data_generation()
        hr = HourlyRatioDetector(ts, freq="H")
        hr._preprocess()
        hr.detector()
        hr.plot()

        # test minutely data with missing values
        ts = self.data_generation(freq="T")
        hr = HourlyRatioDetector(ts, freq="T", aggregate="max")
        hr._preprocess()
        hr.detector()
        hr.plot()

    def test_other(self) -> None:
        self.assertRaises(ValueError, HourlyRatioDetector, self.TSData_multi)

        self.assertRaises(ValueError, HourlyRatioDetector, self.ts_data_daily)

        self.assertRaises(ValueError, HourlyRatioDetector, self.TSData_empty)

        ts = self.data_generation(freq="T")
        self.assertRaises(ValueError, HourlyRatioDetector, data=ts)

        self.assertRaises(
            ValueError, HourlyRatioDetector, data=ts, aggregate="other_method"
        )

        hr = HourlyRatioDetector(ts, freq="T", aggregate="max")
        self.assertRaises(ValueError, hr.plot)
