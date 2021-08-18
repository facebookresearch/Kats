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
from kats.detectors.seasonality import ACFDetector, FFTDetector
from kats.models.harmonic_regression import HarmonicRegressionModel

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


class ACFDetectorTest(TestCase):
    def setUp(self):
        daily_data = load_data("peyton_manning.csv")
        daily_data.columns = ["time", "y"]
        self.ts_data_daily = TimeSeriesData(daily_data)

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def test_acf_detector(self) -> None:
        detector = ACFDetector(data=self.ts_data_daily)
        res = detector.detector(lags=None, diff=1, alpha=0.01)
        self.assertEqual(res["seasonality_presence"], True)
        detector.remover()
        detector.plot()

    def test_no_seasonality(self) -> None:
        np.random.seed(10)
        df_noregress = pd.DataFrame({"no_change": np.random.normal(1, 0.2, 60)})
        df_noregress["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))
        timeseries = TimeSeriesData(df_noregress)
        detector = ACFDetector(data=timeseries)
        res = detector.detector(lags=None, diff=1, alpha=0.01)
        self.assertEqual(res["seasonality_presence"], False)
        detector.remover()
        detector.plot()

    def test_logging(self) -> None:
        with self.assertRaises(ValueError):
            ACFDetector(data=self.TSData_multi)


class FFTDetectorTest(TestCase):
    def setUp(self):
        times = pd.to_datetime(
            np.arange(start=1576195200, stop=1577836801, step=60 * 60), unit="s"
        )
        self.series_times = pd.Series(times)
        harms = HarmonicRegressionModel.fourier_series(self.series_times, 24, 3)
        self.harms_sum = np.sum([1, 1, 1, 1, 1, 1] * harms, axis=1)
        self.data = TimeSeriesData(
            pd.DataFrame({"time": self.series_times, "values": self.harms_sum})
        )

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def test_detector(self) -> None:
        detector = FFTDetector(data=self.data)
        result = detector.detector()
        detector.plot(time_unit="Hour")
        detector.plot_fft(time_unit="Hour")
        self.assertTrue(result["seasonality_presence"])
        self.assertEqual(int(result["seasonalities"][0]), 24)

    def test_logging(self) -> None:
        with self.assertRaises(ValueError):
            FFTDetector(data=self.TSData_multi)
