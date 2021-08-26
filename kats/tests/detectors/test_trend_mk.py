# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import re
from operator import attrgetter
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.trend_mk import MKDetector
from parameterized import parameterized
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


def gen_no_trend_data_ndim(time: pd.Series, ndim: int = 1):
    n_days = len(time)
    data = np.ones((n_days, ndim)) * np.random.randint(1000, size=(1, ndim))
    no_trend_data = pd.DataFrame(data)
    no_trend_data["time"] = time

    return TimeSeriesData(no_trend_data)


def gen_trend_data_ndim(
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
    t_change = np.random.randint(int(0.4 * n_days), int(0.7 * n_days), size=(ndim, 1))

    data = (
        (initial + trend * ix + trend_change * (ix - t_change) * expit((ix - t_change)))
        * (1 - seasonality * (ix % 7 >= 5))
        * np.array(
            [np.cumprod(1 + noise[i] * np.random.randn(n_days)) for i in range(ndim)]
        )
    )

    trend_data = pd.DataFrame(data.T)
    trend_data["time"] = time

    t_change = [t_change[i][0] for i in range(len(t_change))]

    return TimeSeriesData(trend_data), t_change


class UnivariateMKDetectorTest(TestCase):
    def setUp(self):
        self.window_size = 20
        self.time = pd.Series(
            pd.date_range(start="2020-01-01", end="2020-06-20", freq="1D")
        )
        no_trend_data = gen_no_trend_data_ndim(time=self.time)
        trend_data, self.t_change = gen_trend_data_ndim(time=self.time)
        trend_seas_data, self.t_change_seas = gen_trend_data_ndim(
            time=self.time, seasonality=0.07
        )

        # no trend data
        self.d_no_trend = MKDetector(data=no_trend_data)
        self.detected_time_points_no_trend = self.d_no_trend.detector(
            window_size=self.window_size
        )

        # trend data
        self.d_trend = MKDetector(data=trend_data)
        self.detected_time_points_trend = self.d_trend.detector(
            window_size=self.window_size
        )
        self.metadata_trend = self.detected_time_points_trend[0][1]
        results_trend = self.d_trend.get_MK_statistics()
        self.up_trend_detected_trend = self.d_trend.get_MK_results(
            results_trend, direction="up"
        )["ds"]
        self.down_trend_detected_trend = self.d_trend.get_MK_results(
            results_trend, direction="down"
        )["ds"]

        # trend data anchor point
        self.detected_time_points_trend2 = self.d_trend.detector(training_days=30)
        results_trend2 = self.d_trend.get_MK_statistics()
        self.up_trend_detected_trend2 = self.d_trend.get_MK_results(
            results_trend2, direction="up"
        )["ds"]
        self.down_trend_detected_trend2 = self.d_trend.get_MK_results(
            results_trend2, direction="down"
        )["ds"]

        # trend data with seasonality
        self.d_seas = MKDetector(data=trend_seas_data)
        self.detected_time_points_seas = self.d_seas.detector(freq="weekly")
        results_seas = self.d_seas.get_MK_statistics()
        self.up_trend_detected_seas = self.d_seas.get_MK_results(
            results_seas, direction="up"
        )["ds"]
        self.down_trend_detected_seas = self.d_seas.get_MK_results(
            results_seas, direction="down"
        )["ds"]

        # trend data with seasonality anchor point
        self.detected_time_points_seas2 = self.d_seas.detector(
            training_days=30, freq="weekly"
        )
        results_seas2 = self.d_seas.get_MK_statistics()
        self.up_trend_detected_seas2 = self.d_seas.get_MK_results(
            results_seas2, direction="up"
        )["ds"]
        self.down_trend_detected_seas2 = self.d_seas.get_MK_results(
            results_seas2, direction="down"
        )["ds"]

    # test for no trend data
    def test_no_trend_data(self) -> None:
        self.assertEqual(len(self.detected_time_points_no_trend), 0)

    # test for trend data
    def test_detector_type(self):
        self.assertIsInstance(self.d_trend, self.metadata_trend.detector_type)

    def test_tau(self):
        self.assertIsInstance(self.metadata_trend.Tau, float)

    def test_is_univariate(self):
        self.assertFalse(self.metadata_trend.is_multivariate)

    def test_incr_trend(self):
        self.assertEqual(self.metadata_trend.trend_direction, "increasing")

    @parameterized.expand([["up_trend_detected_trend"], ["up_trend_detected_seas"]])
    def test_upward_after_start(self, up_trend_detected):
        self.assertGreaterEqual(
            attrgetter(up_trend_detected)(self).iloc[0],
            self.time[0],
            msg=f"The first {self.window_size}-days upward trend was not detected after it starts.",
        )

    @parameterized.expand(
        [
            ["up_trend_detected_trend", "t_change"],
            ["up_trend_detected_seas", "t_change_seas"],
        ]
    )
    def test_upward_before_end(self, up_trend_detected, t_change):
        self.assertLessEqual(
            attrgetter(up_trend_detected)(self).iloc[-1],
            self.time[attrgetter(t_change)(self)[0] + self.window_size],
            msg=f"The last {self.window_size}-days upward trend was not detected before it ends.",
        )

    @parameterized.expand(
        [
            ["down_trend_detected_trend", "t_change"],
            ["down_trend_detected_seas", "t_change_seas"],
        ]
    )
    def test_downward_after_start(self, down_trend_detected, t_change):
        self.assertGreaterEqual(
            attrgetter(down_trend_detected)(self).iloc[0],
            self.time[attrgetter(t_change)(self)[0]],
            msg=f"The first {self.window_size}-days downward trend was not detected after it starts.",
        )

    @parameterized.expand(
        [
            ["down_trend_detected_trend"],
            ["down_trend_detected_trend2"],
            ["down_trend_detected_seas"],
            ["down_trend_detected_seas2"],
        ]
    )
    def test_downward_before_end(self, down_trend_detected):
        self.assertEqual(
            attrgetter(down_trend_detected)(self).iloc[-1],
            self.time[len(self.time) - 1],
            msg=f"The last {self.window_size}-days downward trend was not detected before it ends.",
        )

    @parameterized.expand(
        [
            ["d_no_trend", "detected_time_points_no_trend"],
            ["d_trend", "detected_time_points_trend"],
            ["d_trend", "detected_time_points_trend2"],
            ["d_seas", "detected_time_points_seas"],
            ["d_seas", "detected_time_points_seas2"],
        ]
    )
    def test_plot(self, detector, detected_time_points):
        attrgetter(detector)(self).plot(attrgetter(detected_time_points)(self))


class MultivariateMKDetectorTest(TestCase):
    def setUp(self):
        self.window_size = 20
        self.time = pd.Series(
            pd.date_range(start="2020-01-01", end="2020-06-20", freq="1D")
        )
        self.ndim = 5
        no_trend_data = gen_no_trend_data_ndim(time=self.time, ndim=self.ndim)
        trend_data, self.t_change = gen_trend_data_ndim(time=self.time, ndim=self.ndim)
        trend_seas_data, self.t_change_seas = gen_trend_data_ndim(
            time=self.time, seasonality=0.07, ndim=self.ndim
        )

        # no trend data
        self.d_no_trend = MKDetector(data=no_trend_data)
        self.detected_time_points_no_trend = self.d_no_trend.detector(
            window_size=self.window_size
        )

        # trend data
        self.d_trend = MKDetector(data=trend_data, multivariate=True)
        self.detected_time_points_trend = self.d_trend.detector()
        results_trend = self.d_trend.get_MK_statistics()
        self.up_trend_detected_trend = self.d_trend.get_MK_results(
            results_trend, direction="up"
        )["ds"]
        self.down_trend_detected_trend = self.d_trend.get_MK_results(
            results_trend, direction="down"
        )["ds"]

        # trend data with seasonality
        self.d_seas = MKDetector(data=trend_seas_data, multivariate=True)
        self.detected_time_points_seas = self.d_seas.detector(freq="weekly")
        results_seas = self.d_seas.get_MK_statistics()
        self.up_trend_detected_seas = self.d_seas.get_MK_results(
            results_seas, direction="up"
        )["ds"]
        self.down_trend_detected_seas = self.d_seas.get_MK_results(
            results_seas, direction="down"
        )["ds"]

    # test for no trend data
    def test_no_trend_data(self) -> None:
        self.assertEqual(len(self.detected_time_points_no_trend), 0)

    def test_heatmap(self) -> None:
        self.d_no_trend.plot_heat_map()

    @parameterized.expand([["up_trend_detected_trend"], ["up_trend_detected_seas"]])
    def test_upward_after_start(self, up_trend_detected):
        self.assertGreaterEqual(
            attrgetter(up_trend_detected)(self).iloc[0],
            self.time[0],
            msg=f"The first {self.window_size}-days upward trend was not detected after it starts.",
        )

    @parameterized.expand(
        [
            ["up_trend_detected_trend", "t_change"],
            ["up_trend_detected_seas", "t_change_seas"],
        ]
    )
    def test_upward_before_end(self, up_trend_detected, t_change):
        self.assertLessEqual(
            attrgetter(up_trend_detected)(self).iloc[-1],
            self.time[attrgetter(t_change)(self)[0] + self.window_size],
            msg=f"The last {self.window_size}-days upward trend was not detected before it ends.",
        )

    @parameterized.expand(
        [
            ["down_trend_detected_trend", "t_change"],
            ["down_trend_detected_seas", "t_change_seas"],
        ]
    )
    def test_downward_after_start(self, down_trend_detected, t_change):
        self.assertGreaterEqual(
            attrgetter(down_trend_detected)(self).iloc[0],
            self.time[attrgetter(t_change)(self)[0]],
            msg=f"The first {self.window_size}-days downward trend was not detected after it starts.",
        )

    @parameterized.expand([["down_trend_detected_trend"], ["down_trend_detected_seas"]])
    def test_downward_before_end(self, down_trend_detected):
        self.assertEqual(
            attrgetter(down_trend_detected)(self).iloc[-1],
            self.time[len(self.time) - 1],
            msg=f"The last {self.window_size}-days downward trend was not detected before it ends.",
        )

    @parameterized.expand(
        [
            ["d_no_trend", "detected_time_points_no_trend"],
            ["d_trend", "detected_time_points_trend"],
            ["d_seas", "detected_time_points_seas"],
        ]
    )
    def test_plot(self, detector, detected_time_points):
        attrgetter(detector)(self).plot(attrgetter(detected_time_points)(self))
