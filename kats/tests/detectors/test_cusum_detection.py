# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from operator import attrgetter
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import (
    CUSUMDetector,
    MultiCUSUMDetector,
)
from parameterized import parameterized
from scipy.stats import chi2  # @manual
from sklearn.datasets import make_spd_matrix

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


class CUSUMDetectorTest(TestCase):
    def setUp(self) -> None:
        np.random.seed(10)

        # increasing with variance detection setup
        df_increase = pd.DataFrame(
            {
                "increase": np.concatenate(
                    [np.random.normal(1, 0.2, 30), np.random.normal(1.5, 0.2, 30)]
                )
            }
        )

        df_increase["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))
        inc_timeseries = TimeSeriesData(df_increase)
        self.inc_detector = CUSUMDetector(inc_timeseries)
        self.inc_change_points = self.inc_detector.detector()
        self.inc_metadata = self.inc_change_points[0][1]

        # decreasing detection setup
        df_decrease = pd.DataFrame(
            {
                "decrease": np.concatenate(
                    [np.random.normal(1, 0.2, 50), np.random.normal(0.5, 0.2, 10)]
                )
            }
        )

        df_decrease["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        dec_timeseries = TimeSeriesData(df_decrease)
        self.dec_detector = CUSUMDetector(dec_timeseries)
        self.dec_change_points = self.dec_detector.detector()
        self.dec_metadata = self.dec_change_points[0][1]

        # seasonality setup
        self.periodicity = 48
        self.total_cycles = 3
        harmonics = 2
        noise_std = 3

        seasonal_term = CUSUMDetectorTest.simulate_seasonal_term(
            self.periodicity,
            self.total_cycles,
            noise_std=noise_std,
            harmonics=harmonics,
        )
        seasonal_term = seasonal_term / seasonal_term.std() * 2
        residual = np.random.normal(0, 1, self.periodicity * self.total_cycles)
        self.seasonal_data = seasonal_term + residual

        # seasonality with increase trend setup
        trend_term = np.logspace(0, 1, self.periodicity * self.total_cycles)
        data = self.seasonal_data + trend_term
        data -= np.min(data)

        df_seasonality = pd.DataFrame(
            {
                "time": pd.date_range(
                    "2020-01-01",
                    periods=self.periodicity * self.total_cycles,
                    freq="30T",
                ),
                "seasonality": data,
            }
        )
        timeseries = TimeSeriesData(df_seasonality)
        self.season_inc_trend_detector = CUSUMDetector(timeseries)
        self.season_inc_trend_change_points = self.season_inc_trend_detector.detector(
            interest_window=[
                self.periodicity * (self.total_cycles - 1),
                self.periodicity * self.total_cycles - 1,
            ],
            magnitude_quantile=1,
            change_directions=["increase", "decrease"],
            delta_std_ratio=0,
        )
        self.season_metadata = self.season_inc_trend_change_points[0][1]

        # test on step change with no variance
        df_increase_no_var = pd.DataFrame(
            {
                "increase": np.concatenate(
                    [np.random.normal(1, 0, 30), np.random.normal(2, 0, 30)]
                )
            }
        )

        df_increase_no_var["time"] = pd.Series(
            pd.date_range("2019-01-01", "2019-03-01")
        )

        no_var_timeseries = TimeSeriesData(df_increase_no_var)
        self.no_var_detector = CUSUMDetector(no_var_timeseries)
        self.no_var_change_points = self.no_var_detector.detector()

        # no seasonality setup
        data = self.seasonal_data
        data -= np.min(data)

        df_seasonality = pd.DataFrame(
            {
                "time": pd.date_range(
                    "2020-01-01",
                    periods=self.periodicity * self.total_cycles,
                    freq="30T",
                ),
                "seasonality": data,
            }
        )
        timeseries = TimeSeriesData(df_seasonality)
        self.no_season_detector = CUSUMDetector(timeseries)
        self.no_season_change_points = self.no_season_detector.detector(
            interest_window=[
                self.periodicity * (self.total_cycles - 1),
                self.periodicity * self.total_cycles - 1,
            ],
            magnitude_quantile=1,
            change_directions=["increase"],
            delta_std_ratio=0,
        )

        # no regression setup
        df_noregress = pd.DataFrame({"no_change": np.random.normal(1, 0.2, 60)})

        df_noregress["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_noregress)
        self.no_reg_detector = CUSUMDetector(timeseries)
        self.no_reg_change_points = self.no_reg_detector.detector(start_point=20)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_change_points", 1],
            ["dec_change_points", 1],
            ["season_inc_trend_change_points", 1],
            ["no_var_change_points", 1],
            ["no_reg_change_points", 0],
            ["no_season_change_points", 0],
        ]
    )
    def test_cp_len(self, cp_name, expected) -> None:
        self.assertEqual(len(attrgetter(cp_name)(self)), expected)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata", 29],
            ["dec_metadata", 49],
        ]
    )
    def test_cp_index(self, metadata_name, expected) -> None:
        self.assertLessEqual(
            abs(attrgetter(metadata_name)(self).cp_index - expected), 1
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata", "increase"],
            ["dec_metadata", "decrease"],
        ]
    )
    def test_direction(self, metadata_name, expected) -> None:
        self.assertEqual(attrgetter(metadata_name)(self).direction, expected)

    def test_increasing_mu(self) -> None:
        self.assertLess(self.inc_metadata.mu0, self.inc_metadata.mu1)

    def test_increasing_correct_delta(self) -> None:
        self.assertEqual(
            self.inc_metadata.delta, self.inc_metadata.mu1 - self.inc_metadata.mu0
        )

    def test_increasing_regression(self) -> None:
        self.assertTrue(self.inc_metadata.regression_detected)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["season_metadata.p_value_int", "season_metadata.llr_int"],
            ["inc_metadata.p_value", "inc_metadata.llr"],
        ]
    )
    def test_p_val(self, pval_name, llr_name) -> None:
        self.assertEqual(
            attrgetter(pval_name)(self),
            1 - chi2.cdf(attrgetter(llr_name)(self), 2),
        )

    def test_increasing_p_val_nan(self) -> None:
        self.assertTrue(np.isnan(self.inc_metadata.p_value_int))

    def test_increasing_llr_int(self) -> None:
        self.assertEqual(self.inc_metadata.llr_int, np.inf)

    def test_increasing_stable_changepoint(self) -> None:
        self.assertTrue(self.inc_metadata.stable_changepoint)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_detector", "inc_change_points"],
            ["dec_detector", "dec_change_points"],
            ["season_inc_trend_detector", "season_inc_trend_change_points"],
            ["no_var_detector", "no_var_change_points"],
            ["no_reg_detector", "no_reg_change_points"],
            ["no_season_detector", "no_season_change_points"],
        ]
    )
    def test_plot(self, detector_name, cp_name) -> None:
        attrgetter(detector_name)(self).plot(attrgetter(cp_name)(self))

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

    def test_seasonality_with_increasing_trend_cp_index(self) -> None:
        self.assertGreaterEqual(
            self.season_metadata.cp_index, self.periodicity * (self.total_cycles - 1)
        )

    def test_logging_multivariate_error(self) -> None:
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

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["WARNING", 0.9],
            ["DEBUG", None],
        ]
    )
    def test_logging_neg_magnitude(self, level, mag_q) -> None:
        # test logging setup - negative in magnitude
        np.random.seed(10)
        df_neg = pd.DataFrame({"no_change": -np.random.normal(1, 0.2, 60)})

        df_neg["time"] = pd.Series(pd.date_range("2019-01-01", "2019-03-01"))

        timeseries = TimeSeriesData(df_neg)
        logging_detector = CUSUMDetector(timeseries)

        with self.assertLogs(level=level):
            logging_detector.detector(
                magnitude_quantile=mag_q, interest_window=[40, 60]
            )

    def test_ts_without_name(self) -> None:
        n = 10
        time = pd.Series(pd.date_range(start="2018-01-01", periods=n, freq="D"))
        value = pd.Series(np.arange(n))
        ts = TimeSeriesData(time=time, value=value)

        detector = CUSUMDetector(ts)
        change_points = detector.detector()
        detector.plot(change_points)


class MultiCUSUMDetectorTest(TestCase):
    def setUp(self) -> None:
        # increasing setup
        self.D = 10
        random_state = 10
        np.random.seed(random_state)
        mean1 = np.ones(self.D)
        mean2 = mean1 * 2
        sigma = make_spd_matrix(self.D, random_state=random_state)

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
        self.inc_change_points = MultiCUSUMDetector(timeseries_increase).detector()
        self.inc_metadata = self.inc_change_points[0][1]

        # decreasing setup
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
        self.dec_change_points = MultiCUSUMDetector(timeseries_decrease).detector()
        self.dec_metadata = self.dec_change_points[0][1]

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_change_points"],
            ["dec_change_points"],
        ]
    )
    def test_cp_len(self, cp_name) -> None:
        self.assertEqual(len(attrgetter(cp_name)(self)), 1)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata"],
            ["dec_metadata"],
        ]
    )
    def test_cp_index(self, cp_name) -> None:
        self.assertLessEqual(abs(attrgetter(cp_name)(self).cp_index - 59), 1)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata.mu0", "inc_metadata.mu1"],
            ["dec_metadata.mu1", "dec_metadata.mu0"],
        ]
    )
    def test_mu(self, m1_name, m2_name) -> None:
        for m1, m2 in zip(attrgetter(m1_name)(self), attrgetter(m2_name)(self)):
            self.assertLess(m1, m2)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata", "inc_metadata.mu0", "inc_metadata.mu1"],
            ["dec_metadata", "dec_metadata.mu0", "dec_metadata.mu1"],
        ]
    )
    def test_correct_delta(self, metadata_name, mu0_name, mu1_name) -> None:
        for d, diff in zip(
            attrgetter(metadata_name)(self).delta,
            attrgetter(mu1_name)(self) - attrgetter(mu0_name)(self),
        ):
            self.assertEqual(d, diff)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata"],
            ["dec_metadata"],
        ]
    )
    def test_regression(self, metadata_name) -> None:
        self.assertTrue(attrgetter(metadata_name)(self).regression_detected)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata"],
            ["dec_metadata"],
        ]
    )
    def test_p_val(self, metadata_name) -> None:
        self.assertEqual(
            attrgetter(metadata_name)(self).p_value,
            1 - chi2.cdf(attrgetter(metadata_name)(self).llr, self.D + 1),
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata"],
            ["dec_metadata"],
        ]
    )
    def test_gaussian_increase_p_val_nan(self, metadata_name) -> None:
        self.assertTrue(np.isnan(attrgetter(metadata_name)(self).p_value_int))

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata"],
            ["dec_metadata"],
        ]
    )
    def test_gaussian_increase_llr_int(self, metadata_name) -> None:
        self.assertEqual(attrgetter(metadata_name)(self).llr_int, np.inf)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["inc_metadata"],
            ["dec_metadata"],
        ]
    )
    def test_gaussian_increase_stable_changepoint(self, metadata_name) -> None:
        self.assertTrue(attrgetter(metadata_name)(self).stable_changepoint)

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
