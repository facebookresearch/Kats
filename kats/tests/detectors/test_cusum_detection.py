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
from kats.detectors.cusum_detection import (
    CUSUMDetector,
    MultiCUSUMDetector,
)

# pyre-ignore[21]: Could not find name `chi2` in `scipy.stats`.
from scipy.stats import chi2  # @manual
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


class CUSUMDetectorTest(TestCase):
    def test_increasing_detection(self) -> None:
        np.random.seed(10)
        df_increase = pd.DataFrame(
            {
                "increase": np.concatenate(
                    [np.random.normal(1, 0.2, 30), np.random.normal(1.5, 0.2, 30)]
                )
            }
        )

        df_increase["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_increase)
        detector = CUSUMDetector(timeseries)
        change_points = detector.detector()
        detector.plot(change_points)

        self.assertEqual(len(change_points), 1)
        metadata = change_points[0][1]

        self.assertLessEqual(abs(metadata.cp_index - 29), 1)
        self.assertEqual(metadata.direction, "increase")
        self.assertLess(metadata.mu0, metadata.mu1)
        # pyre-fixme[6]: Expected `float` for 1st param but got `Union[float,
        #  np.ndarray]`.
        self.assertEqual(metadata.delta, metadata.mu1 - metadata.mu0)
        self.assertTrue(metadata.regression_detected)
        # pyre-ignore[16]: Module `stats` has no attribute `chi2`.
        self.assertEqual(metadata.p_value, 1 - chi2.cdf(metadata.llr, 2))
        self.assertTrue(np.isnan(metadata.p_value_int))
        self.assertEqual(metadata.llr_int, np.inf)
        self.assertTrue(metadata.stable_changepoint)

        print(metadata)

        # test on step change with no variance
        df_increase = pd.DataFrame(
            {
                "increase": np.concatenate(
                    [np.random.normal(1, 0, 30), np.random.normal(2, 0, 30)]
                )
            }
        )

        df_increase["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_increase)
        detector = CUSUMDetector(timeseries)
        change_points = detector.detector()
        detector.plot(change_points)

        self.assertEqual(len(change_points), 1)

    def test_decreasing_detection(self) -> None:
        np.random.seed(10)
        df_decrease = pd.DataFrame(
            {
                "decrease": np.concatenate(
                    [np.random.normal(1, 0.2, 50), np.random.normal(0.5, 0.2, 10)]
                )
            }
        )

        df_decrease["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_decrease)
        detector = CUSUMDetector(timeseries)
        change_points = detector.detector()
        detector.plot(change_points)

        self.assertEqual(len(change_points), 1)

        metadata = change_points[0][1]

        self.assertLessEqual(abs(metadata.cp_index - 49), 1)
        self.assertEqual(metadata.direction, "decrease")

    def test_noregression(self) -> None:
        np.random.seed(10)
        df_noregress = pd.DataFrame({"no_change": np.random.normal(1, 0.2, 60)})

        df_noregress["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_noregress)
        detector = CUSUMDetector(timeseries)
        change_points = detector.detector(start_point=20)
        detector.plot(change_points)

        self.assertEqual(len(change_points), 0)

    @staticmethod
    def simulate_seasonal_term(
        periodicity, total_cycles, noise_std=1.0, harmonics=None
    ):
        duration = periodicity * total_cycles
        assert duration == int(duration)
        duration = int(duration)
        harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

        lambda_p = 2 * np.pi / float(periodicity)

        gamma_jt = noise_std * np.random.randn((harmonics))
        gamma_star_jt = noise_std * np.random.randn((harmonics))

        total_timesteps = 100 * duration  # Pad for burn in
        series = np.zeros(total_timesteps)
        for t in range(total_timesteps):
            gamma_jtp1 = np.zeros_like(gamma_jt)
            gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
            for j in range(1, harmonics + 1):
                cos_j = np.cos(lambda_p * j)
                sin_j = np.sin(lambda_p * j)
                gamma_jtp1[j - 1] = (
                    gamma_jt[j - 1] * cos_j
                    + gamma_star_jt[j - 1] * sin_j
                    + noise_std * np.random.randn()
                )
                gamma_star_jtp1[j - 1] = (
                    -gamma_jt[j - 1] * sin_j
                    + gamma_star_jt[j - 1] * cos_j
                    + noise_std * np.random.randn()
                )
            series[t] = np.sum(gamma_jtp1)
            gamma_jt = gamma_jtp1
            gamma_star_jt = gamma_star_jtp1
        wanted_series = series[-duration:]  # Discard burn in

        return wanted_series

    def test_seasonality(self) -> None:

        np.random.seed(100)
        periodicity = 48
        total_cycles = 3
        harmonics = 2
        noise_std = 3

        seasonal_term = CUSUMDetectorTest.simulate_seasonal_term(
            periodicity, total_cycles, noise_std=noise_std, harmonics=harmonics
        )
        seasonal_term = seasonal_term / seasonal_term.std() * 2
        residual = np.random.normal(0, 1, periodicity * total_cycles)
        data = seasonal_term + residual
        data -= np.min(data)

        df_seasonality = pd.DataFrame(
            {
                "time": pd.date_range(
                    "2020-01-01", periods=periodicity * total_cycles, freq="30T"
                ),
                "seasonality": data,
            }
        )
        timeseries = TimeSeriesData(df_seasonality)
        detector = CUSUMDetector(timeseries)
        change_points = detector.detector(
            interest_window=[
                periodicity * (total_cycles - 1),
                periodicity * total_cycles - 1,
            ],
            magnitude_quantile=1,
            change_directions=["increase"],
            delta_std_ratio=0,
        )
        detector.plot(change_points)

        self.assertEqual(len(change_points), 0)

        # seasonality with increase trend
        trend_term = np.logspace(0, 1, periodicity * total_cycles)
        data = seasonal_term + residual + trend_term
        data -= np.min(data)

        df_seasonality = pd.DataFrame(
            {
                "time": pd.date_range(
                    "2020-01-01", periods=periodicity * total_cycles, freq="30T"
                ),
                "seasonality": data,
            }
        )
        timeseries = TimeSeriesData(df_seasonality)
        detector = CUSUMDetector(timeseries)
        change_points = detector.detector(
            interest_window=[
                periodicity * (total_cycles - 1),
                periodicity * total_cycles - 1,
            ],
            magnitude_quantile=1,
            change_directions=["increase", "decrease"],
            delta_std_ratio=0,
        )
        detector.plot(change_points)

        self.assertEqual(len(change_points), 1)
        change_meta = change_points[0][1]
        self.assertGreaterEqual(change_meta.cp_index, periodicity * (total_cycles - 1))
        # pyre-ignore[16]: Module `stats` has no attribute `chi2`.
        self.assertEqual(change_meta.p_value_int, 1 - chi2.cdf(change_meta.llr_int, 2))

    def test_logging(self) -> None:
        # test multivariate error
        np.random.seed(10)
        df_multi_var = pd.DataFrame(
            {
                "no_change": np.random.normal(1, 0.2, 60),
                "no_change2": np.random.normal(1, 0.2, 60),
            }
        )

        df_multi_var["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        with self.assertRaises(ValueError):
            timeseries = TimeSeriesData(df_multi_var)
            CUSUMDetector(timeseries)

        # test negative in magnitude
        np.random.seed(10)
        df_neg = pd.DataFrame({"no_change": -np.random.normal(1, 0.2, 60)})

        df_neg["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_neg)
        detector = CUSUMDetector(timeseries)
        with self.assertLogs(level="WARNING"):
            detector.detector(magnitude_quantile=0.9, interest_window=[40, 60])
        with self.assertLogs(level="DEBUG"):
            detector.detector(magnitude_quantile=None, interest_window=[40, 60])

    def test_ts_without_name(self) -> None:
        n = 10
        time = pd.Series(pd.date_range(start="2018-01-01", periods=n, freq="D"))
        value = pd.Series(np.arange(n))
        ts = TimeSeriesData(time=time, value=value)

        detector = CUSUMDetector(ts)
        change_points = detector.detector()
        detector.plot(change_points)


class MultiCUSUMDetectorTest(TestCase):
    def test_gaussian_increase(self) -> None:
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

        timeseries_increase = TimeSeriesData(df_increase)
        change_points = MultiCUSUMDetector(timeseries_increase).detector()
        self.assertEqual(len(change_points), 1)
        metadata = change_points[0][1]
        self.assertLessEqual(abs(metadata.cp_index - 59), 1)

        # pyre-fixme[6]: Expected `Iterable[Variable[_T1]]` for 1st param but got
        #  `Union[float, np.ndarray]`.
        for m1, m2 in zip(metadata.mu0, metadata.mu1):
            self.assertLess(m1, m2)
        # pyre-fixme[6]: Expected `Iterable[Variable[_T1]]` for 1st param but got
        #  `Union[float, np.ndarray]`.
        # pyre-fixme[6]: Expected `float` for 1st param but got `Union[float,
        #  np.ndarray]`.
        for d, diff in zip(metadata.delta, metadata.mu1 - metadata.mu0):
            self.assertEqual(d, diff)
        self.assertTrue(metadata.regression_detected)
        # pyre-ignore[16]: Module `stats` has no attribute `chi2`.
        self.assertEqual(metadata.p_value, 1 - chi2.cdf(metadata.llr, D + 1))
        self.assertTrue(np.isnan(metadata.p_value_int))
        self.assertEqual(metadata.llr_int, np.inf)
        self.assertTrue(metadata.stable_changepoint)

    def test_gaussian_decrease(self) -> None:
        D = 10
        random_state = 10
        np.random.seed(random_state)
        mean1 = np.ones(D)
        mean2 = mean1 * 2
        sigma = make_spd_matrix(D, random_state=random_state)
        df_decrease = pd.DataFrame(
            np.concatenate(
                [
                    np.random.multivariate_normal(mean2, sigma, 60),
                    np.random.multivariate_normal(mean1, sigma, 30),
                ]
            )
        )

        df_decrease["time"] = pd.Series(pd.date_range("2019-01-01", "2019-04-01"))

        timeseries_decrease = TimeSeriesData(df_decrease)
        change_points = MultiCUSUMDetector(timeseries_decrease).detector()
        self.assertEqual(len(change_points), 1)
        metadata = change_points[0][1]

        self.assertLessEqual(abs(metadata.cp_index - 59), 1)

        # pyre-fixme[6]: Expected `Iterable[Variable[_T1]]` for 1st param but got
        #  `Union[float, np.ndarray]`.
        for m1, m2 in zip(metadata.mu0, metadata.mu1):
            self.assertGreater(m1, m2)

        # pyre-fixme[6]: Expected `Iterable[Variable[_T1]]` for 1st param but got
        #  `Union[float, np.ndarray]`.
        # pyre-fixme[6]: Expected `float` for 1st param but got `Union[float,
        #  np.ndarray]`.
        for d, diff in zip(metadata.delta, metadata.mu1 - metadata.mu0):
            self.assertEqual(d, diff)
        self.assertTrue(metadata.regression_detected)
        # pyre-ignore[16]: Module `stats` has no attribute `chi2`.
        self.assertEqual(metadata.p_value, 1 - chi2.cdf(metadata.llr, D + 1))
        self.assertTrue(np.isnan(metadata.p_value_int))
        self.assertEqual(metadata.llr_int, np.inf)
        self.assertTrue(metadata.stable_changepoint)

    def test_no_changepoint(self) -> None:
        D = 10
        random_state = 10
        np.random.seed(random_state)
        mean = np.ones(D)
        sigma = make_spd_matrix(D, random_state=random_state)
        # Use the same mean for the entire series and there should be no changepoint
        df_no_change = pd.DataFrame(np.random.multivariate_normal(mean, sigma, 90))
        df_no_change["time"] = pd.Series(pd.date_range("2019-01-01", "2019-04-01"))

        timeseries_no_change = TimeSeriesData(df_no_change)
        change_points = MultiCUSUMDetector(timeseries_no_change).detector()
        self.assertEqual(len(change_points), 0)
