# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import re
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.hourly_ratio_detection import HourlyRatioDetector

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


class HourlyRatioDectorTest(TestCase):
    def setUp(self):
        daily_data = load_data("peyton_manning.csv")
        daily_data.columns = ["time", "y"]
        self.ts_data_daily = TimeSeriesData(daily_data)

        self.TSData_empty = TimeSeriesData(pd.DataFrame([], columns=["time", "y"]))

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def data_generation(self, freq="H", drop: bool = True, frac: float = 0.95):
        time = pd.date_range("2018-01-01", "2020-01-01", freq=freq)
        n = len(time)
        x = np.arange(n)
        values = np.abs(np.sin(30 * x) + 5 * x + 10 * x ** 2) + np.random.randn(n)
        df = pd.DataFrame(values, columns=["value"])
        df["time"] = time
        if drop:
            df = df.sample(frac=frac, replace=False)
        # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]` for 1st param
        #  but got `Union[pd.core.frame.DataFrame, pd.core.series.Series]`.
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
