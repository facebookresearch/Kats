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
from kats.detectors.trend_mk import MKDetector
from scipy.special import expit  # @manual

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


class MKDetectorTest(TestCase):
    def gen_no_trend_data_ndim(self, time: pd.Series, ndim: int = 1):
        n_days = len(time)
        data = np.ones((n_days, ndim)) * np.random.randint(1000, size=(1, ndim))
        no_trend_data = pd.DataFrame(data)
        no_trend_data["time"] = time

        return TimeSeriesData(no_trend_data)

    def gen_trend_data_ndim(
        self,
        time: pd.Series,
        seasonality: float = 0.00,
        change_smoothness: float = 5.0,
        ndim: int = 1,
    ):
        np.random.seed(20)

        n_days = len(time)
        ix = np.array([np.arange(n_days) for i in range(ndim)])
        initial = np.random.randint(9000.0, 10000.0, size=(ndim, 1))
        trend_change = -np.random.randint(60, size=(ndim, 1))
        trend = np.random.randint(2.0, 6.0, size=(ndim, 1))
        noise = np.array([1e-3] * ndim).reshape((ndim, 1))
        t_change = np.random.randint(
            int(0.4 * n_days), int(0.7 * n_days), size=(ndim, 1)
        )

        data = (
            (
                initial
                + trend * ix
                + trend_change * (ix - t_change) * expit((ix - t_change))
            )
            * (1 - seasonality * (ix % 7 >= 5))
            * np.array(
                [
                    np.cumprod(1 + noise[i] * np.random.randn(n_days))
                    for i in range(ndim)
                ]
            )
        )

        trend_data = pd.DataFrame(data.T)
        trend_data["time"] = time

        t_change = [t_change[i][0] for i in range(len(t_change))]

        return TimeSeriesData(trend_data), t_change

    def test_MKtest(self) -> None:
        window_size = 20
        time = pd.Series(pd.date_range(start="2020-01-01", end="2020-06-20", freq="1D"))

        # Check with no trend data
        no_trend_data = self.gen_no_trend_data_ndim(time=time)
        d = MKDetector(data=no_trend_data)
        detected_time_points = d.detector(window_size=window_size)
        d.plot(detected_time_points)
        self.assertEqual(len(detected_time_points), 0)

        # Check with univariate trend data
        # test whole time series
        trend_data, t_change = self.gen_trend_data_ndim(time=time)

        d = MKDetector(data=trend_data)
        detected_time_points = d.detector()
        d.plot(detected_time_points)
        metadata = detected_time_points[0][1]
        self.assertIsInstance(d, metadata.detector_type)
        self.assertFalse(metadata.is_multivariate)
        self.assertEqual(metadata.trend_direction, "increasing")
        self.assertIsInstance(metadata.Tau, float)
        print(metadata)

        results = d.get_MK_statistics()
        up_trend_detected = d.get_MK_results(results, direction="up")["ds"]
        down_trend_detected = d.get_MK_results(results, direction="down")["ds"]

        self.assertGreaterEqual(
            up_trend_detected.iloc[0],
            time[0],
            msg=f"The first {window_size}-days upward trend was not detected after it starts.",
        )
        self.assertLessEqual(
            up_trend_detected.iloc[-1],
            time[t_change[0] + window_size],
            msg=f"The last {window_size}-days upward trend was not detected before it ends.",
        )
        self.assertGreaterEqual(
            down_trend_detected.iloc[0],
            time[t_change[0]],
            msg=f"The first {window_size}-days downward trend was not detected after it starts.",
        )
        self.assertEqual(
            down_trend_detected.iloc[-1],
            time[len(time) - 1],
            msg=f"The last {window_size}-days downward trend was not detected before it ends.",
        )

        # test anchor point
        trend_data, t_change = self.gen_trend_data_ndim(time=time)

        d = MKDetector(data=trend_data)
        detected_time_points = d.detector(training_days=30)
        d.plot(detected_time_points)

        results = d.get_MK_statistics()
        up_trend_detected = d.get_MK_results(results, direction="up")["ds"]
        down_trend_detected = d.get_MK_results(results, direction="down")["ds"]

        self.assertEqual(
            down_trend_detected.iloc[-1],
            time[len(time) - 1],
            msg=f"The {window_size}-days downward trend at the anchor point was not detected.",
        )

        # Check with univariate trend data with seasonality
        # test whole time series
        trend_seas_data, t_change = self.gen_trend_data_ndim(
            time=time, seasonality=0.07
        )
        d = MKDetector(data=trend_seas_data)
        detected_time_points = d.detector(freq="weekly")
        d.plot(detected_time_points)

        results = d.get_MK_statistics()
        up_trend_detected = d.get_MK_results(results, direction="up")["ds"]
        down_trend_detected = d.get_MK_results(results, direction="down")["ds"]

        self.assertGreaterEqual(
            up_trend_detected.iloc[0],
            time[0],
            msg=f"The first {window_size}-days upward trend was not detected after it starts.",
        )
        self.assertLessEqual(
            up_trend_detected.iloc[-1],
            time[t_change[0] + window_size],
            msg=f"The last {window_size}-days upward trend was not detected before it ends.",
        )
        self.assertGreaterEqual(
            down_trend_detected.iloc[0],
            time[t_change[0]],
            msg=f"The first {window_size}-days downward trend was not detected after it starts.",
        )
        self.assertEqual(
            down_trend_detected.iloc[-1],
            time[len(time) - 1],
            msg=f"The last {window_size}-days downward trend was not detected before it ends.",
        )

        # test anchor point
        trend_data, t_change = self.gen_trend_data_ndim(time=time, seasonality=0.07)

        d = MKDetector(data=trend_data)
        detected_time_points = d.detector(training_days=30, freq="weekly")
        d.plot(detected_time_points)

        results = d.get_MK_statistics()
        up_trend_detected = d.get_MK_results(results, direction="up")["ds"]
        down_trend_detected = d.get_MK_results(results, direction="down")["ds"]

        self.assertEqual(
            down_trend_detected.iloc[-1],
            time[len(time) - 1],
            msg=f"The {window_size}-days downward trend at the anchor point not was detected.",
        )

    def test_multivariate_MKtest(self, ndim=5) -> None:
        window_size = 20
        time = pd.Series(pd.date_range(start="2020-01-01", end="2020-06-20", freq="1D"))

        # Check with no trend data
        no_trend_data = self.gen_no_trend_data_ndim(time=time, ndim=ndim)
        d = MKDetector(data=no_trend_data)
        detected_time_points = d.detector(window_size=window_size)
        d.plot(detected_time_points)
        d.plot_heat_map()
        self.assertEqual(len(detected_time_points), 0)

        # Check with multivariate trend data
        trend_data, t_change = self.gen_trend_data_ndim(time=time, ndim=ndim)
        d = MKDetector(data=trend_data, multivariate=True)
        detected_time_points = d.detector()
        d.plot(detected_time_points)

        results = d.get_MK_statistics()
        up_trend_detected = d.get_MK_results(results, direction="up")["ds"]
        down_trend_detected = d.get_MK_results(results, direction="down")["ds"]

        self.assertGreaterEqual(
            up_trend_detected.iloc[0],
            time[0],
            msg=f"The first {window_size}-days upward trend was not detected after it starts.",
        )
        self.assertLessEqual(
            up_trend_detected.iloc[-1],
            time[max(t_change) + window_size],
            msg=f"The last {window_size}-days upward trend was not detected before the it ends.",
        )
        self.assertGreaterEqual(
            down_trend_detected.iloc[0],
            time[max(t_change)],
            msg=f"The first {window_size}-days downward trend was not detected after it starts.",
        )
        self.assertEqual(
            down_trend_detected.iloc[-1],
            time[len(time) - 1],
            msg=f"The last {window_size}-days downward trend was not detected before it ends.",
        )

        # Check with multivariate trend data with seasonality
        trend_seas_data, t_change = self.gen_trend_data_ndim(
            time=time, seasonality=0.07, ndim=ndim
        )
        d = MKDetector(data=trend_seas_data, multivariate=True)
        detected_time_points = d.detector(freq="weekly")
        d.plot(detected_time_points)

        results = d.get_MK_statistics()
        up_trend_detected = d.get_MK_results(results, direction="up")["ds"]
        down_trend_detected = d.get_MK_results(results, direction="down")["ds"]

        self.assertGreaterEqual(
            up_trend_detected.iloc[0],
            time[0],
            msg=f"The first {window_size}-days upward trend was not detected after it starts.",
        )
        self.assertLessEqual(
            up_trend_detected.iloc[-1],
            time[max(t_change) + window_size],
            msg=f"The last {window_size}-days upward trend was not detected before the it ends.",
        )
        self.assertGreaterEqual(
            down_trend_detected.iloc[0],
            time[max(t_change)],
            msg=f"The first {window_size}-days downward trend was not detected after it starts.",
        )
        self.assertEqual(
            down_trend_detected.iloc[-1],
            time[len(time) - 1],
            msg=f"The last {window_size}-days downward trend was not detected before it ends.",
        )
