# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Union
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.detectors.hourly_ratio_detection import HourlyRatioDetector
from parameterized import parameterized


class HourlyRatioDectorTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        daily_data = load_data("peyton_manning.csv")
        daily_data.columns = ["time", "y"]
        cls.invalid_daily_ts = TimeSeriesData(daily_data)
        cls.invalid_minutely_ts = cls.data_generation(freq="T")

        cls.invalid_empty_ts = TimeSeriesData(pd.DataFrame([], columns=["time", "y"]))

        multi_data = load_data("multivariate_anomaly_simulated_data.csv")
        cls.invalid_multi_ts = TimeSeriesData(multi_data)

        cls.valid_hourly_ts = cls.data_generation(drop=False)
        cls.valid_missing_ts = cls.data_generation()
        cls.valid_minutely_ts = cls.data_generation(freq="T")

        plot_before_detector_ts = cls.data_generation(freq="T")
        cls.plot_before_detector_hr = HourlyRatioDetector(
            plot_before_detector_ts, freq="T", aggregate="max"
        )

    @classmethod
    def data_generation(
        cls, freq: str = "H", drop: bool = True, frac: float = 0.95
    ) -> TimeSeriesData:
        time = pd.date_range("2018-01-01", "2020-01-01", freq=freq)
        n = len(time)
        x = np.arange(n)
        values = np.abs(np.sin(30 * x)) + 0.3 * np.random.randn(n)
        df = pd.DataFrame(values, columns=["value"])
        df["time"] = time
        if drop:
            df = cast(pd.DataFrame, df.sample(frac=frac, replace=False))
        return TimeSeriesData(df)

    # pyre-ignore[16] Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [  # name, freq, aggregate
            ["valid_hourly_ts", "H", None],
            ["valid_missing_ts", "H", None],
            ["valid_minutely_ts", "T", "max"],
        ]
    )
    def test_detector(self, name: str, freq: str, aggregate: Union[str, None]) -> None:
        # test hourly data without missing vlaues
        ts = getattr(self, name)
        hr = HourlyRatioDetector(ts, freq=freq, aggregate=aggregate)
        hr._preprocess()
        hr.detector()
        hr.plot()

    # pyre-ignore[16] Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["invalid_daily_ts"],
            ["invalid_minutely_ts"],
            ["invalid_multi_ts"],
            ["invalid_empty_ts"],
        ]
    )
    def test_invalid(self, name: str) -> None:
        ts = getattr(self, name)
        self.assertRaises(ValueError, HourlyRatioDetector, ts)

    def test_minutely_other_aggregate(self) -> None:
        ts = self.invalid_minutely_ts
        self.assertRaises(
            ValueError, HourlyRatioDetector, data=ts, aggregate="other_method"
        )

    def test_plot_before_detector(self) -> None:
        hr = self.plot_before_detector_hr
        self.assertRaises(ValueError, hr.plot)
