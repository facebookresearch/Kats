#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import random
import time
import unittest
from collections import Counter
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.bocpd import (
    BOCPDetector,
    BOCPDModelType,
    NormalKnownParameters,
    PoissonModelParameters,
    TrendChangeParameters,
)
from kats.detectors.bocpd_model import BocpdDetectorModel
from kats.detectors.changepoint_evaluator import (
    TuringEvaluator,
    Evaluation,
    EvalAggregate,
    f_measure,
)
from kats.detectors.cusum_detection import (
    CUSUMDetector,
    MultiCUSUMDetector,
)
from kats.detectors.cusum_model import (
    CUSUMDetectorModel,
    CusumScoreFunction,
)
from kats.detectors.detector_consts import (
    AnomalyResponse,
    ChangePointInterval,
    ConfidenceBand,
    MultiAnomalyResponse,
    MultiChangePointInterval,
    MultiPercentageChange,
    PercentageChange,
    SingleSpike,
)
from kats.detectors.hourly_ratio_detection import HourlyRatioDetector
from kats.detectors.outlier import (
    MultivariateAnomalyDetector,
    MultivariateAnomalyDetectorType,
    OutlierDetector,
)
from kats.detectors.prophet_detector import ProphetDetectorModel
from kats.detectors.robust_stat_detection import RobustStatDetector
from kats.detectors.seasonality import ACFDetector, FFTDetector
from kats.detectors.slow_drift_detector import (
    SlowDriftDetectorModel,
    time_series_to_data_points,
)
from kats.detectors.stat_sig_detector import (
    MultiStatSigDetectorModel,
    StatSigDetectorModel,
)
from kats.detectors.trend_mk import MKDetector
from kats.detectors.window_slo_detector import WindowSloDetectorModel
from kats.models.bayesian_var import BayesianVARParams
from kats.models.harmonic_regression import HarmonicRegressionModel
from kats.models.var import VARParams
from kats.utils.simulator import Simulator
from scipy.special import expit  # @manual

# pyre-fixme[21]: Could not find name `chi2` in `scipy.stats`.
from scipy.stats import chi2  # @manual
from sklearn.datasets import make_spd_matrix

if "kats/tests" in os.getcwd():
    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname("__file__"),
            "../",
            "data/air_passengers.csv"
            )
        )

    daily_data_path = os.path.abspath(
        os.path.join(
            os.path.dirname("__file__"),
            "../",
            "data/peyton_manning.csv"
            )
        )

    multi_data_path = os.path.abspath(
        os.path.join(
            os.path.dirname("__file__"),
            "../",
            "data/cdn_working_set.csv"
            )
        )
else:
    data_path = "kats/kats/data/air_passengers.csv"
    daily_data_path = "kats/kats/data/peyton_manning.csv"
    multi_data_path = "kats/kats/data/cdn_working_set.csv"

data = pd.read_csv(data_path)
data.columns = ["time", "y"]
ts_data = TimeSeriesData(data)
# generate muliple series
data_2 = data.copy()
data_2["y_2"] = data_2["y"]
ts_data_2 = TimeSeriesData(data_2)


daily_data = pd.read_csv(daily_data_path)
daily_data.columns = ["time", "y"]
ts_data_daily = TimeSeriesData(daily_data)

DATA_multi = pd.read_csv(multi_data_path)
TSData_multi = TimeSeriesData(DATA_multi)

TSData_empty = TimeSeriesData(pd.DataFrame([], columns=["time", "y"]))


# Anomaly detection tests
class OutlierDetectionTest(TestCase):
    def test_new_freq_outliers(self) -> None:
        def process_time(z):
            x0, x1 = z.split(" ")
            time = (
                "-".join(y.rjust(2, "0") for y in x0.split("/"))
                + "20 "
                + ":".join(y.rjust(2, "0") for y in x1.split(":"))
                + ":00"
            )

            return datetime.strptime(time, "%m-%d-%Y %H:%M:%S")

        df_15_min = DATA_multi

        df_15_min["ts"] = df_15_min["time"].apply(process_time)

        df_15_min_dict = {}

        for i in range(0, 4):

            df_15_min_temp = df_15_min.copy()
            df_15_min_temp["ts"] = [
                x + timedelta(minutes=15 * i) for x in df_15_min_temp["ts"]
            ]
            df_15_min_dict[i] = df_15_min_temp

        df_15_min_ts = pd.concat(df_15_min_dict.values()).sort_values(by="ts")[
            ["ts", "V1"]
        ]

        df_15_min_ts.columns = ["time", "y"]

        df_ts = TimeSeriesData(df_15_min_ts)

        m = OutlierDetector(df_ts, "additive")
        m.detector()
        m.remover(interpolate=True)

    def test_additive_overrides(self) -> None:
        m = OutlierDetector(ts_data, "additive")

        m.detector()
        outliers = m.remover(interpolate=True)

        m2 = OutlierDetector(ts_data, "logarithmic")

        m2.detector()
        outliers2 = m2.remover(interpolate=True)

        self.assertEqual(outliers.value.all(), outliers2.value.all())

    def test_outlier_detection_additive(self) -> None:
        m = OutlierDetector(ts_data, "additive")

        m.detector()
        m.remover(interpolate=True)

        m2 = OutlierDetector(ts_data_daily, "additive")
        m2.detector()
        m2.remover(interpolate=True)
        # test for multiple time series
        m3 = OutlierDetector(ts_data_2, "additive")
        m3.detector()
        m3.remover(interpolate=True)

    def test_outlier_detection_multiplicative(self) -> None:
        m = OutlierDetector(ts_data, "multiplicative")
        m.detector()
        m.remover(interpolate=True)

        m2 = OutlierDetector(ts_data_daily, "multiplicative")
        m2.detector()
        m2.remover(interpolate=True)
        # test for multiple time series
        m3 = OutlierDetector(ts_data_2, "additive")
        m3.detector()
        m3.remover(interpolate=True)

    def test_outlier_detector_exception(self) -> None:
        data_new = pd.concat([data, data])
        ts_data_new = TimeSeriesData(data_new)

        with self.assertLogs(level="ERROR"):
            m = OutlierDetector(ts_data_new)
            m.detector()


class MultivariateVARDetectorTest(TestCase):
    def test_var_detector(self) -> None:
        np.random.seed(10)

        params = VARParams(maxlags=3)
        d = MultivariateAnomalyDetector(TSData_multi, params, training_days=3)
        anomaly_score_df = d.detector()
        self.assertCountEqual(
            list(anomaly_score_df.columns),
            list(TSData_multi.value.columns) + ["overall_anomaly_score", "p_value"],
        )
        d.plot()
        alpha = 0.01
        anomalies = d.get_anomaly_timepoints(alpha)
        d.get_anomalous_metrics(anomalies[0], top_k=5)

    def test_bayesian_detector(self) -> None:
        np.random.seed(10)

        params = BayesianVARParams(p=3)
        d = MultivariateAnomalyDetector(
            TSData_multi[24 : -2 * 24],  # testing on shorter data
            params,
            training_days=3,
            model_type=MultivariateAnomalyDetectorType.BAYESIAN_VAR,
        )
        anomaly_score_df = d.detector()
        self.assertCountEqual(
            list(anomaly_score_df.columns),
            list(TSData_multi.value.columns) + ["overall_anomaly_score", "p_value"],
        )
        d.plot()
        alpha = 0.05
        anomalies = d.get_anomaly_timepoints(alpha)
        d.get_anomalous_metrics(anomalies[0], top_k=5)

    def test_runtime_errors(self) -> None:
        DATA_multi2 = pd.concat([DATA_multi, DATA_multi])
        TSData_multi2 = TimeSeriesData(DATA_multi2)
        params = VARParams(maxlags=3)

        with self.assertRaises(RuntimeError):
            d = MultivariateAnomalyDetector(TSData_multi2, params, training_days=3)
            d.detector()

        DATA_multi3 = pd.merge(
            DATA_multi, DATA_multi, how="inner", on="time", suffixes=("_1", "_2")
        )
        TSData_multi3 = TimeSeriesData(DATA_multi3)
        with self.assertRaises(RuntimeError):
            d2 = MultivariateAnomalyDetector(TSData_multi3, params, training_days=3)
            d2.detector()


# Change point (aka regression) detection tests
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
        self.assertEqual(metadata.delta, metadata.mu1 - metadata.mu0)
        self.assertTrue(metadata.regression_detected)
        # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
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
        # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
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

        for m1, m2 in zip(metadata.mu0, metadata.mu1):
            self.assertLess(m1, m2)

        for d, diff in zip(metadata.delta, metadata.mu1 - metadata.mu0):
            self.assertEqual(d, diff)
        self.assertTrue(metadata.regression_detected)
        # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
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

        for m1, m2 in zip(metadata.mu0, metadata.mu1):
            self.assertGreater(m1, m2)

        for d, diff in zip(metadata.delta, metadata.mu1 - metadata.mu0):
            self.assertEqual(d, diff)
        self.assertTrue(metadata.regression_detected)
        # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
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


class BOCPDTest(TestCase):
    first_cp_begin = 100
    first_cp_end = 200
    second_cp_begin = 350

    sigma = 0.05  # std. dev
    num_points = 450

    def setUp(self):
        self.sim = Simulator(n=450, start="2018-01-01")

        self.cp_array_input = [
            BOCPDTest.first_cp_begin,
            BOCPDTest.first_cp_end,
            BOCPDTest.second_cp_begin,
        ]

        self.level_arr = [1.35, 1.05, 1.35, 1.2]

    def test_normal(self) -> None:

        ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )
        bocpd_model = BOCPDetector(data=ts)

        cps = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=False,
            agg_cp=False,
        )
        bocpd_model.plot(cps)

        change_prob_dict = bocpd_model.get_change_prob()
        change_prob = list(change_prob_dict.values())[
            0
        ]  # dict only has a single element here
        self.assertEqual(change_prob.shape[0], len(ts))

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        cp_arr = np.concatenate(
            (
                ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        # TODO: this check only tests that all changepoints we find should be there
        #       but not the other way around, that we find all change points.
        for t in cps:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test the case where priors are chosen automatically
        cps2 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=True,
            agg_cp=False,
        )
        bocpd_model.plot(cps2)

        for t in cps2:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test the case where run-length posterior is aggregated
        cps3 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=False,
            agg_cp=True,
        )
        bocpd_model.plot(cps3)

        for t in cps3:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test the case where run-length posterior is aggregated and
        # automatically tuning prior
        cps4 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=True,
            agg_cp=True,
        )
        bocpd_model.plot(cps4)

        for t in cps4:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test the case where search method has been changed to grid
        # search
        cps5_params = NormalKnownParameters()
        cps5_params.search_method = "gridsearch"
        cps5 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            model_parameters=cps5_params,
            changepoint_prior=0.01,
            choose_priors=True,
        )
        bocpd_model.plot(cps5)

        for t in cps5:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test to see if agg_cp=True works
        cps6 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=True,
            agg_cp=True,
        )

        for t in cps6:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

    def test_normal_multivariate(self) -> None:

        ts = self.sim.level_shift_multivariate_indep_sim(
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.04,
            seasonal_period=7,
            seasonal_magnitude=0.0,
            dim=3,
        )

        bocpd_model = BOCPDetector(data=ts)
        cps = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            # pyre-fixme[6]: Expected `float` for 2nd param but got `ndarray`.
            changepoint_prior=np.array([0.01, 0.01, 1.0]),
            threshold=np.array([1.0, 0.5, 0.5]),
            choose_priors=False,
            agg_cp=False,
        )
        bocpd_model.plot(cps)

        change_prob_dict = bocpd_model.get_change_prob()
        change_prob_val = change_prob_dict.values()

        for prob_arr in change_prob_val:
            self.assertEqual(prob_arr.shape[0], len(ts))

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        cp_arr = np.concatenate(
            (
                ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        # We should have 3 change points per time series (of which there are 3)
        # However, we have set different change point priors, so we lose 3
        # and we set different thresholds, so we lose the other 3.
        self.assertEqual(len(cps), 3)

        counter = Counter()
        for t in cps:
            ts_name = t[1].ts_name
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)
            counter += Counter({ts_name: 1})

        # Check we have all the time series.
        self.assertEqual(counter, Counter(value2=3))

        # check if multivariate detection works with choosing priors
        cps2 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL, choose_priors=True, agg_cp=False
        )
        bocpd_model.plot(cps2)

        # check if multivariate detection works with aggregating run-length
        # posterior
        cps3 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL, choose_priors=False
        )
        bocpd_model.plot(cps3)

        # check if multivariate detection works with aggregating run-length
        # posterior and automated tuning prior
        cps4 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL, choose_priors=True
        )
        bocpd_model.plot(cps4)

        # check if multivariate detection works in detecting all changepoints
        cps5 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            # pyre-fixme[6]: Expected `float` for 2nd param but got `ndarray`.
            changepoint_prior=np.array([0.01, 0.01, 0.01]),
            threshold=np.array([0.85, 0.85, 0.85]),
            choose_priors=False,
        )
        bocpd_model.plot(cps5)

        change_prob_dict = bocpd_model.get_change_prob()
        change_prob_val = change_prob_dict.values()

        for prob_arr in change_prob_val:
            self.assertEqual(prob_arr.shape[0], len(ts))

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        cp_arr = np.concatenate(
            (
                ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        # With new algorithm, all changepoints should
        self.assertTrue(len(cps5) >= 9)

        counter = Counter()
        for t in cps5:
            ts_name = t[1].ts_name
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)
            counter += Counter({ts_name: 1})

        # Check we have all the time series.
        self.assertEqual(counter, Counter(value1=3, value2=3, value3=3))

    def test_trend(self) -> None:
        sim = Simulator(n=200, start="2018-01-01")
        ts = sim.trend_shift_sim(
            random_seed=15,
            cp_arr=[100],
            trend_arr=[3, 28],
            intercept=30,
            noise=30,
            seasonal_period=7,
            seasonal_magnitude=0,
        )
        threshold = 0.5
        detector = BOCPDetector(data=ts)
        cps = detector.detector(
            model=BOCPDModelType.TREND_CHANGE_MODEL,
            model_parameters=TrendChangeParameters(
                readjust_sigma_prior=True, num_points_prior=20
            ),
            debug=True,
            threshold=threshold,
            choose_priors=False,
            agg_cp=True,
        )
        detector.plot(cps)

        # expect only one cp
        # test if atleast one cp is in 90:110
        start_list = [cp[0].start_time for cp in cps]
        intersect = list(set(start_list) & set(ts.time.values[90:110]))
        self.assertGreaterEqual(len(intersect), 1)

        # check if confidence is greater than threshold
        self.assertGreaterEqual(
            cps[0][0].confidence,
            threshold,
            f"confidence should have been at least threshold {threshold}, but got {cps[0][0].confidence}",
        )

    def test_poisson(self) -> None:

        ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        bocpd_model = BOCPDetector(data=ts)
        cps = bocpd_model.detector(
            model=BOCPDModelType.POISSON_PROCESS_MODEL,
            changepoint_prior=0.01,
            model_parameters=PoissonModelParameters(beta_prior=0.01),
            choose_priors=False,
        )
        bocpd_model.plot(cps)

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        cp_arr = np.concatenate(
            (
                ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        # TODO: this check only tests that all changepoints we find should be there
        #       but not the other way around, that we find all change points.
        for t in cps:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)


class BocpdDetectorModelTest(TestCase):
    first_cp_begin = 100
    first_cp_end = 200
    second_cp_begin = 350

    def setUp(self):
        self.sim = Simulator(n=450, start="2018-01-01")

        self.cp_array_input = [
            BocpdDetectorModelTest.first_cp_begin,
            BocpdDetectorModelTest.first_cp_end,
            BocpdDetectorModelTest.second_cp_begin,
        ]

        self.ts_length = 450
        self.sigma = 0.05

        self.level_arr = [1.35, 1.05, 1.35, 1.2]

    def test_no_history(self) -> None:

        level_ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=self.sigma,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        bocpd_detector = BocpdDetectorModel()
        anom = bocpd_detector.fit_predict(data=level_ts)
        self.assertEqual(len(anom.scores), self.ts_length)
        threshold = 0.4

        # we have set changepoints at 100, 200, 350
        # we want to make sure those are detected
        # we set some slack for them be detected
        # 5 time points before/after
        self.assertTrue(np.max(anom.scores.value.values[95:105]) > threshold)
        self.assertTrue(np.max(anom.scores.value.values[195:205]) > threshold)
        self.assertTrue(np.max(anom.scores.value.values[345:355]) > threshold)

    def test_history(self) -> None:
        ts_length = 450
        ts_history_length = 100

        level_ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=self.sigma,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        level_ts_history = TimeSeriesData(
            time=level_ts.time.iloc[:ts_history_length],
            value=pd.Series(level_ts.value.iloc[:ts_history_length], name="value"),
        )

        level_ts_data = TimeSeriesData(
            time=level_ts.time.iloc[ts_history_length:],
            value=pd.Series(level_ts.value.iloc[ts_history_length:], name="value"),
        )

        bocpd_detector = BocpdDetectorModel()
        anom = bocpd_detector.fit_predict(
            historical_data=level_ts_history, data=level_ts_data
        )
        self.assertEqual(len(anom.scores), ts_length - ts_history_length)

        threshold = 0.4
        # same as above.
        # we test for the two changepoints in 200, 350, but shifted by 100
        # since that is the length of the history
        self.assertTrue(np.max(anom.scores.value.values[95:105]) > threshold)
        self.assertTrue(np.max(anom.scores.value.values[245:255]) > threshold)

    def test_slow_drift(self) -> None:
        ts_length = 200

        sim = Simulator(n=ts_length, start="2018-01-01")
        trend_ts = sim.trend_shift_sim(
            random_seed=15,
            cp_arr=[100],
            trend_arr=[3, 28],
            intercept=30,
            noise=30,
            seasonal_period=7,
            seasonal_magnitude=0,
        )
        bocpd_detector = BocpdDetectorModel(slow_drift=True)
        anom = bocpd_detector.fit_predict(data=trend_ts)
        self.assertEqual(len(anom.scores), ts_length)
        threshold = 0.4

        # we have set changepoints at 100
        # we want to make sure that is detected
        # we set some slack for them be detected
        # 5 time points before/after
        self.assertTrue(np.max(anom.scores.value.values[95:105]) > threshold)

    def test_serialize(self) -> None:

        level_ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=self.sigma,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        bocpd_detector = BocpdDetectorModel(slow_drift=False)
        ser_model = bocpd_detector.serialize()

        # check that it ignores the slow_drift parameter
        # and considers the serialized one instead
        bocpd_detector2 = BocpdDetectorModel(
            serialized_model=ser_model, slow_drift=True
        )
        self.assertEqual(bocpd_detector2.slow_drift, False)

        anom = bocpd_detector2.fit_predict(data=level_ts)
        self.assertEqual(len(anom.scores), self.ts_length)

    def test_missing_data(self) -> None:
        # this data is in the same format as OneDetection
        # it also crosses the daylight savings time
        history_time_list = (
            (
                pd.date_range(
                    "2020-03-01", "2020-03-10", tz="US/Pacific", freq="1d"
                ).astype(int)
                / 1e9
            )
            .astype(int)
            .to_list()
        )

        data_time_list = (
            (
                pd.date_range(
                    "2020-03-11", "2020-03-20", tz="US/Pacific", freq="1d"
                ).astype(int)
                / 1e9
            )
            .astype(int)
            .to_list()
        )

        history = TimeSeriesData(
            df=pd.DataFrame(
                {
                    "time": (history_time_list[:5] + history_time_list[6:]),
                    "value": np.random.randn(len(history_time_list) - 1),
                }
            ),
            use_unix_time=True,
            unix_time_units="s",
            tz="US/Pacific",
        )

        data = TimeSeriesData(
            df=pd.DataFrame(
                {
                    "time": (data_time_list[:5] + data_time_list[6:]),
                    "value": np.random.randn(len(data_time_list) - 1),
                }
            ),
            use_unix_time=True,
            unix_time_units="s",
            tz="US/Pacific",
        )
        bocpd_detector = BocpdDetectorModel()
        anom = bocpd_detector.fit_predict(historical_data=history, data=data)

        self.assertEqual(len(anom.scores), len(data))


# Other detection tests (seasonality, trend, etc)
class ACFDetectorTest(TestCase):
    def test_acf_detector(self) -> None:
        detector = ACFDetector(data=ts_data_daily)
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
            ACFDetector(data=TSData_multi)


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

        top_k_metrics = d.get_top_k_metrics(
            detected_time_points[0][0].start_time, top_k=5
        )

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

    def test_detector(self) -> None:
        detector = FFTDetector(data=self.data)
        result = detector.detector()
        detector.plot(time_unit="Hour")
        detector.plot_fft(time_unit="Hour")
        self.assertTrue(result["seasonality_presence"])
        self.assertEqual(int(result["seasonalities"][0]), 24)

    def test_logging(self) -> None:
        with self.assertRaises(ValueError):
            FFTDetector(data=TSData_multi)


class SingleSpikeTest(TestCase):
    def test_spike(self) -> None:
        spike_time_str = "2020-03-01"
        spike_time = datetime.strptime(spike_time_str, "%Y-%m-%d")
        spike = SingleSpike(time=spike_time, value=1.0, n_sigma=3.0)
        self.assertEqual(spike.time_str, spike_time_str)


class ChangePointIntervalTest(TestCase):
    def test_changepoint(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(15)]

        current_length = 10

        current_seq = [
            previous_seq[10] + timedelta(days=x) for x in range(current_length)
        ]
        previous_values = np.random.randn(len(previous_seq))
        current_values = np.random.randn(len(current_seq))

        # add a very large value to detect spikes
        current_values[0] = 100.0

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `previous`.
        self.previous = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": previous_values})
        )

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current`.
        self.current = TimeSeriesData(
            pd.DataFrame({"time": current_seq, "value": current_values})
        )

        previous_extend = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[9:], "value": previous_values[9:]})
        )

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `prev_start`.
        self.prev_start = previous_seq[0]
        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `prev_end`.
        self.prev_end = previous_seq[9]

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current_start`.
        self.current_start = current_seq[0]
        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current_end`.
        self.current_end = current_seq[-1] + timedelta(days=1)

        previous_int = ChangePointInterval(
            cp_start=self.prev_start, cp_end=self.prev_end
        )
        previous_int.data = self.previous

        # tests whether data is clipped property to start and end dates
        self.assertEqual(previous_int.data.tolist(), previous_values[0:9].tolist())

        # test extending the data
        # now the data is extended to include the whole sequence
        previous_int.end_time = previous_seq[-1] + timedelta(days=1)
        previous_int.extend_data(previous_extend)

        self.assertEqual(len(previous_int), len(previous_seq))

        current_int = ChangePointInterval(
            cp_start=self.current_start, cp_end=self.current_end
        )
        current_int.data = self.current
        current_int.previous_interval = previous_int

        # check all the properties
        self.assertEqual(current_int.start_time, self.current_start)
        self.assertEqual(current_int.end_time, self.current_end)
        self.assertEqual(
            current_int.start_time_str,
            datetime.strftime(self.current_start, "%Y-%m-%d"),
        )
        self.assertEqual(
            current_int.end_time_str, datetime.strftime(self.current_end, "%Y-%m-%d")
        )

        self.assertEqual(current_int.mean_val, np.mean(current_values))
        self.assertEqual(current_int.variance_val, np.var(current_values))
        self.assertEqual(len(current_int), current_length)
        self.assertEqual(current_int.previous_interval, previous_int)

        # check spike detection
        spike_list = current_int.spikes
        self.assertEqual(spike_list[0].value, 100.0)
        self.assertEqual(
            spike_list[0].time_str, datetime.strftime(self.current_start, "%Y-%m-%d")
        )


class PercentageChangeTest(TestCase):
    def test_perc_change(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(30)]

        current_length = 31
        # offset one to make the new interval start one day after the previous one ends
        current_seq = [
            previous_seq[-1] + timedelta(days=(x + 1)) for x in range(current_length)
        ]
        previous_values = 1.0 + 0.25 * np.random.randn(len(previous_seq))
        current_values = 10.0 + 0.25 * np.random.randn(len(current_seq))

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `previous`.
        self.previous = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": previous_values})
        )

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current`.
        self.current = TimeSeriesData(
            pd.DataFrame({"time": current_seq, "value": current_values})
        )

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `prev_start`.
        self.prev_start = previous_seq[0]
        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `prev_end`.
        self.prev_end = previous_seq[9]

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current_start`.
        self.current_start = current_seq[0]
        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current_end`.
        self.current_end = current_seq[-1]

        previous_int = ChangePointInterval(
            cp_start=previous_seq[0], cp_end=(previous_seq[-1] + timedelta(days=1))
        )
        previous_int.data = self.previous

        current_int = ChangePointInterval(
            cp_start=current_seq[0], cp_end=(current_seq[-1] + timedelta(days=1))
        )
        current_int.data = self.current
        current_int.previous_interval = previous_int

        perc_change_1 = PercentageChange(current=current_int, previous=previous_int)

        previous_mean = np.mean(previous_values)
        current_mean = np.mean(current_values)

        # test the ratios
        ratio_val = current_mean / previous_mean
        self.assertEqual(perc_change_1.ratio_estimate, ratio_val)
        self.assertAlmostEqual(perc_change_1.ratio_estimate, 10.0, 0)

        self.assertEqual(perc_change_1.perc_change, (ratio_val - 1) * 100)
        self.assertEqual(perc_change_1.direction, "up")
        self.assertEqual(perc_change_1.stat_sig, True)
        self.assertTrue(perc_change_1.p_value < 0.05)
        self.assertTrue(perc_change_1.score > 1.96)

        # test a detector with false stat sig
        second_values = 10.005 + 0.25 * np.random.randn(len(previous_seq))
        second = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": second_values})
        )

        second_int = ChangePointInterval(
            cp_start=previous_seq[0], cp_end=previous_seq[-1]
        )
        second_int.data = second

        perc_change_2 = PercentageChange(current=current_int, previous=second_int)
        self.assertEqual(perc_change_2.stat_sig, False)
        self.assertFalse(perc_change_2.p_value < 0.05)
        self.assertFalse(perc_change_2.score > 1.96)

        # test the edge case when one of the intervals
        # contains a single data point
        current_int_2 = ChangePointInterval(
            cp_start=current_seq[0], cp_end=current_seq[1]
        )

        current_int_2.data = self.current

        perc_change_3 = PercentageChange(current=current_int_2, previous=previous_int)
        self.assertTrue(perc_change_3.score > 1.96)

        # TODO delta method tests


class MultiChangePointIntervalTest(TestCase):
    def test_multichangepoint(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")

        previous_seq = [date_start + timedelta(days=x) for x in range(15)]

        current_length = 10

        current_seq = [
            previous_seq[10] + timedelta(days=x) for x in range(current_length)
        ]

        num_seq = 5
        previous_values = [np.random.randn(len(previous_seq)) for _ in range(num_seq)]
        current_values = [np.random.randn(len(current_seq)) for _ in range(num_seq)]

        # add a very large value to detect spikes
        for i in range(num_seq):
            current_values[i][0] = 100 * (i + 1)

        # pyre-fixme[16]: `MultiChangePointIntervalTest` has no attribute `previous`.
        self.previous = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": previous_values[i] for i in range(num_seq)},
                }
            )
        )

        # pyre-fixme[16]: `MultiChangePointIntervalTest` has no attribute `current`.
        self.current = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": current_seq},
                    **{f"value_{i}": current_values[i] for i in range(num_seq)},
                }
            )
        )

        previous_extend = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[10:]},
                    **{f"value_{i}": previous_values[i][10:] for i in range(num_seq)},
                }
            )
        )

        # pyre-fixme[16]: `MultiChangePointIntervalTest` has no attribute `prev_start`.
        self.prev_start = previous_seq[0]
        # pyre-fixme[16]: `MultiChangePointIntervalTest` has no attribute `prev_end`.
        self.prev_end = previous_seq[9]

        # pyre-fixme[16]: `MultiChangePointIntervalTest` has no attribute
        #  `current_start`.
        self.current_start = current_seq[0]
        # pyre-fixme[16]: `MultiChangePointIntervalTest` has no attribute `current_end`.
        self.current_end = current_seq[-1]

        previous_int = MultiChangePointInterval(
            cp_start=self.prev_start, cp_end=self.prev_end
        )
        previous_int.data = self.previous

        # tests whether data is clipped property to start and end dates
        for i in range(num_seq):
            self.assertEqual(
                previous_int.data[:, i].tolist(), previous_values[i][0:10].tolist()
            )

        # test extending the data
        # now the data is extended to include the whole sequence
        previous_int.end_time = previous_seq[-1]
        previous_int.extend_data(previous_extend)

        self.assertEqual(len(previous_int), len(previous_seq))

        current_int = MultiChangePointInterval(
            cp_start=self.current_start, cp_end=self.current_end
        )
        current_int.data = self.current
        current_int.previous_interval = previous_int

        # check all the properties
        self.assertEqual(current_int.start_time, self.current_start)
        self.assertEqual(current_int.end_time, self.current_end)
        self.assertEqual(
            current_int.start_time_str,
            datetime.strftime(self.current_start, "%Y-%m-%d"),
        )
        self.assertEqual(
            current_int.end_time_str, datetime.strftime(self.current_end, "%Y-%m-%d")
        )

        self.assertEqual(
            current_int.mean_val.tolist(),
            [np.mean(current_values[i]) for i in range(num_seq)],
        )
        self.assertEqual(
            current_int.variance_val.tolist(),
            [np.var(current_values[i]) for i in range(num_seq)],
        )
        self.assertEqual(len(current_int), current_length)
        self.assertEqual(current_int.previous_interval, previous_int)

        # check spike detection
        spike_array = current_int.spikes
        self.assertEqual(len(spike_array), num_seq)

        for i in range(num_seq):
            self.assertEqual(spike_array[i][0].value, 100 * (i + 1))
            self.assertEqual(
                spike_array[i][0].time_str,
                datetime.strftime(self.current_start, "%Y-%m-%d"),
            )


class MultiPercentageChangeTest(TestCase):
    def test_multi_perc_change(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(30)]

        current_length = 31
        # offset one to make the new interval start one day after the previous one ends
        current_seq = [
            previous_seq[-1] + timedelta(days=(x + 1)) for x in range(current_length)
        ]

        num_seq = 5

        previous_values = np.array(
            [1.0 + 0.0001 * np.random.randn(len(previous_seq)) for _ in range(num_seq)]
        )
        current_values = np.array(
            [10.0 + 0.0001 * np.random.randn(len(current_seq)) for _ in range(num_seq)]
        )

        # pyre-fixme[16]: `MultiPercentageChangeTest` has no attribute `previous`.
        self.previous = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": previous_values[i] for i in range(num_seq)},
                }
            )
        )

        # pyre-fixme[16]: `MultiPercentageChangeTest` has no attribute `current`.
        self.current = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": current_seq},
                    **{f"value_{i}": current_values[i] for i in range(num_seq)},
                }
            )
        )

        # pyre-fixme[16]: `MultiPercentageChangeTest` has no attribute `prev_start`.
        self.prev_start = previous_seq[0]
        # pyre-fixme[16]: `MultiPercentageChangeTest` has no attribute `prev_end`.
        self.prev_end = previous_seq[9]

        # pyre-fixme[16]: `MultiPercentageChangeTest` has no attribute `current_start`.
        self.current_start = current_seq[0]
        # pyre-fixme[16]: `MultiPercentageChangeTest` has no attribute `current_end`.
        self.current_end = current_seq[-1]

        previous_int = MultiChangePointInterval(
            cp_start=previous_seq[0], cp_end=previous_seq[-1]
        )
        previous_int.data = self.previous

        current_int = MultiChangePointInterval(
            cp_start=current_seq[0], cp_end=current_seq[-1]
        )
        current_int.data = self.current
        current_int.previous_interval = previous_int

        perc_change_1 = MultiPercentageChange(
            current=current_int, previous=previous_int
        )

        previous_mean = np.array([np.mean(previous_values[i]) for i in range(num_seq)])
        current_mean = np.array([np.mean(current_values[i]) for i in range(num_seq)])

        # test the ratios
        ratio_val = current_mean / previous_mean
        self.assertEqual(perc_change_1.ratio_estimate.tolist(), ratio_val.tolist())

        for r in perc_change_1.ratio_estimate:
            self.assertAlmostEqual(r, 10.0, 0)

        self.assertEqual(
            perc_change_1.perc_change.tolist(), ((ratio_val - 1) * 100).tolist()
        )
        self.assertEqual(perc_change_1.direction.tolist(), ["up"] * num_seq)
        self.assertEqual(perc_change_1.stat_sig.tolist(), [True] * num_seq)

        for p_value, score in zip(perc_change_1.p_value, perc_change_1.score):
            self.assertLess(p_value, 0.05)
            self.assertLess(1.96, score)

        # test a detector with false stat sig
        second_values = np.array(
            [10.005 + 0.25 * np.random.randn(len(previous_seq)) for _ in range(num_seq)]
        )

        second = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": second_values[i] for i in range(num_seq)},
                }
            )
        )

        second_int = MultiChangePointInterval(
            cp_start=previous_seq[0], cp_end=previous_seq[-1]
        )
        second_int.data = second

        perc_change_2 = MultiPercentageChange(current=current_int, previous=second_int)
        for stat_sig, p_value, score in zip(
            perc_change_2.stat_sig, perc_change_2.p_value, perc_change_2.score
        ):
            self.assertFalse(stat_sig)
            self.assertLess(0.05, p_value)
            self.assertLess(score, 1.96)

        # test a detector with a negative spike
        third_values = np.array(
            [
                1000.0 + 0.0001 * np.random.randn(len(previous_seq))
                for _ in range(num_seq)
            ]
        )

        third = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": third_values[i] for i in range(num_seq)},
                }
            )
        )

        third_int = MultiChangePointInterval(
            cp_start=previous_seq[0], cp_end=previous_seq[-1]
        )
        third_int.data = third

        perc_change_3 = MultiPercentageChange(current=current_int, previous=third_int)
        for p_value, score in zip(perc_change_3.p_value, perc_change_3.score):
            self.assertLess(p_value, 0.05)
            self.assertLess(score, -1.96)

        # test the edge case when one of the intervals
        # contains a single data point
        current_int_single_point = MultiChangePointInterval(
            cp_start=current_seq[0], cp_end=current_seq[1]
        )

        current_int_single_point.data = self.current

        perc_change_single_point = MultiPercentageChange(
            current=current_int_single_point, previous=previous_int
        )

        for p_value, score in zip(
            perc_change_single_point.p_value, perc_change_single_point.score
        ):
            self.assertLess(p_value, 0.05)
            self.assertLess(1.96, score)


class TestAnomalyResponse(TestCase):
    def test_response(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(30)]
        score_ts = TimeSeriesData(
            pd.DataFrame(
                {"time": previous_seq, "value": np.random.randn(len(previous_seq))}
            )
        )
        upper_values = 1.0 + np.random.randn(len(previous_seq))
        upper_ts = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": upper_values})
        )

        lower_ts = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": (upper_values - 0.1)})
        )

        conf_band = ConfidenceBand(upper=upper_ts, lower=lower_ts)

        pred_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": previous_seq,
                    "value": (10.0 + 0.25 * np.random.randn(len(previous_seq))),
                }
            )
        )

        mag_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": previous_seq,
                    "value": (10.0 + 0.25 * np.random.randn(len(previous_seq))),
                }
            )
        )

        stat_sig_ts = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": np.ones(len(previous_seq))})
        )

        response = AnomalyResponse(
            scores=score_ts,
            confidence_band=conf_band,
            predicted_ts=pred_ts,
            anomaly_magnitude_ts=mag_ts,
            stat_sig_ts=stat_sig_ts,
        )

        # test update
        new_date = previous_seq[-1] + timedelta(days=1)
        common_val = 1.23
        response.update(
            time=new_date,
            score=common_val,
            ci_upper=common_val,
            ci_lower=(common_val - 0.1),
            pred=common_val,
            anom_mag=common_val,
            stat_sig=0,
        )

        # assert that all the lengths of the time series are preserved
        N = len(previous_seq)
        self.assertEqual(len(response.scores), N)
        self.assertEqual(len(response.confidence_band.upper), N)
        self.assertEqual(len(response.confidence_band.lower), N)
        self.assertEqual(len(response.predicted_ts), N)
        self.assertEqual(len(response.anomaly_magnitude_ts), N)
        self.assertEqual(len(response.stat_sig_ts), N)

        # assert that each time series has moved one point forward
        self.assertEqual(response.scores.value[0], score_ts.value[1])
        self.assertEqual(
            response.confidence_band.upper.value[0], conf_band.upper.value[1]
        )
        self.assertEqual(
            response.confidence_band.lower.value[0], conf_band.lower.value[1]
        )
        self.assertEqual(response.predicted_ts.value[0], pred_ts.value[1])
        self.assertEqual(response.anomaly_magnitude_ts.value[0], mag_ts.value[1])
        self.assertEqual(response.stat_sig_ts.value[0], stat_sig_ts.value[1])

        # assert that a new point has been added to the end
        self.assertEqual(response.scores.value.values[-1], common_val)
        self.assertEqual(response.confidence_band.upper.value.values[-1], common_val)
        self.assertEqual(
            response.confidence_band.lower.value.values[-1], common_val - 0.1
        )
        self.assertEqual(response.predicted_ts.value.values[-1], common_val)
        self.assertEqual(response.anomaly_magnitude_ts.value.values[-1], common_val)
        self.assertEqual(response.stat_sig_ts.value.values[-1], 0.0)

        # assert that we return the last N values
        score_list = response.scores.value.values.tolist()

        n_val = 10
        response_last_n = response.get_last_n(n_val)
        self.assertEqual(len(response_last_n.scores), n_val)
        self.assertEqual(len(response_last_n.confidence_band.upper), n_val)
        self.assertEqual(len(response_last_n.confidence_band.lower), n_val)
        self.assertEqual(len(response_last_n.predicted_ts), n_val)
        self.assertEqual(len(response_last_n.anomaly_magnitude_ts), n_val)
        self.assertEqual(len(response_last_n.stat_sig_ts), n_val)

        self.assertEqual(
            response_last_n.scores.value.values.tolist(), score_list[-n_val:]
        )


class TestMultiAnomalyResponse(TestCase):
    def test_multi_response(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 5

        previous_seq = [date_start + timedelta(days=x) for x in range(30)]

        score_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": np.random.randn(len(previous_seq))
                        for i in range(num_seq)
                    },
                }
            )
        )

        upper_values = [
            1.0 + np.random.randn(len(previous_seq)) for _ in range(num_seq)
        ]

        upper_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": upper_values[i] for i in range(num_seq)},
                }
            )
        )

        lower_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": upper_values[i] - 0.1 for i in range(num_seq)},
                }
            )
        )

        conf_band = ConfidenceBand(upper=upper_ts, lower=lower_ts)

        pred_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": 10.0 + 0.25 * np.random.randn(len(previous_seq))
                        for i in range(num_seq)
                    },
                }
            )
        )

        mag_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": 10.0 + 0.25 * np.random.randn(len(previous_seq))
                        for i in range(num_seq)
                    },
                }
            )
        )

        stat_sig_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": np.ones(len(previous_seq)) for i in range(num_seq)
                    },
                }
            )
        )

        response = MultiAnomalyResponse(
            scores=score_ts,
            confidence_band=conf_band,
            predicted_ts=pred_ts,
            anomaly_magnitude_ts=mag_ts,
            stat_sig_ts=stat_sig_ts,
        )

        # test update
        new_date = previous_seq[-1] + timedelta(days=1)
        common_val = 1.23 * np.ones(num_seq)

        response.update(
            time=new_date,
            score=common_val,
            ci_upper=common_val,
            ci_lower=common_val - 0.1,
            pred=common_val,
            anom_mag=common_val,
            stat_sig=np.zeros(num_seq),
        )

        # assert that all the lengths of the time series are preserved

        for i in response.key_mapping:
            N = len(previous_seq)
            self.assertEqual(
                len(response.response_objects[response.key_mapping[i]].scores), N
            )
            self.assertEqual(
                len(
                    response.response_objects[
                        response.key_mapping[i]
                    ].confidence_band.upper
                ),
                N,
            )
            self.assertEqual(
                len(
                    response.response_objects[
                        response.key_mapping[i]
                    ].confidence_band.lower
                ),
                N,
            )
            self.assertEqual(
                len(response.response_objects[response.key_mapping[i]].predicted_ts), N
            )
            self.assertEqual(
                len(
                    response.response_objects[
                        response.key_mapping[i]
                    ].anomaly_magnitude_ts
                ),
                N,
            )
            self.assertEqual(
                len(response.response_objects[response.key_mapping[i]].stat_sig_ts), N
            )

            # assert that each time series has moved one point forward
            self.assertEqual(
                response.response_objects[response.key_mapping[i]]
                .scores.value[:-1]
                .tolist(),
                score_ts.value.iloc[1:, i].tolist(),
            )
            self.assertEqual(
                response.response_objects[response.key_mapping[i]]
                .confidence_band.upper.value[:-1]
                .tolist(),
                upper_ts.value.iloc[1:, i].tolist(),
            )
            self.assertEqual(
                response.response_objects[response.key_mapping[i]]
                .confidence_band.lower.value[:-1]
                .tolist(),
                lower_ts.value.iloc[1:, i].tolist(),
            )
            self.assertEqual(
                response.response_objects[response.key_mapping[i]]
                .predicted_ts.value[:-1]
                .tolist(),
                pred_ts.value.iloc[1:, i].tolist(),
            )
            self.assertEqual(
                response.response_objects[response.key_mapping[i]]
                .anomaly_magnitude_ts.value[:-1]
                .tolist(),
                mag_ts.value.iloc[1:, i].tolist(),
            )
            self.assertEqual(
                response.response_objects[response.key_mapping[i]]
                .stat_sig_ts.value[:-1]
                .tolist(),
                stat_sig_ts.value.iloc[1:, i].tolist(),
            )

            # assert that a new point has been added to the end
            self.assertEqual(
                response.response_objects[response.key_mapping[i]].scores.value.iloc[
                    -1
                ],
                # pyre-fixme[16]: `float` has no attribute `__getitem__`.
                common_val[i],
            )
            self.assertEqual(
                response.response_objects[
                    response.key_mapping[i]
                ].confidence_band.upper.value.iloc[-1],
                common_val[i],
            )
            self.assertEqual(
                response.response_objects[
                    response.key_mapping[i]
                ].confidence_band.lower.value.iloc[-1],
                common_val[i] - 0.1,
            )
            self.assertEqual(
                response.response_objects[
                    response.key_mapping[i]
                ].predicted_ts.value.iloc[-1],
                common_val[i],
            )
            self.assertEqual(
                response.response_objects[
                    response.key_mapping[i]
                ].anomaly_magnitude_ts.value.iloc[-1],
                common_val[i],
            )
            self.assertEqual(
                response.response_objects[
                    response.key_mapping[i]
                ].stat_sig_ts.value.iloc[-1],
                0,
            )

            # assert that we return the last N values
            score_list = response.response_objects[
                response.key_mapping[i]
            ].scores.value.values.tolist()

            n_val = 10
            response_last_n_object = response.get_last_n(n_val)
            response_last_n = response_last_n_object.response_objects[
                response_last_n_object.key_mapping[i]
            ]
            self.assertEqual(len(response_last_n.scores), n_val)
            self.assertEqual(len(response_last_n.confidence_band.upper), n_val)
            self.assertEqual(len(response_last_n.confidence_band.lower), n_val)
            self.assertEqual(len(response_last_n.predicted_ts), n_val)
            self.assertEqual(len(response_last_n.anomaly_magnitude_ts), n_val)
            self.assertEqual(len(response_last_n.stat_sig_ts), n_val)

            self.assertEqual(
                response_last_n.scores.value.values.tolist(), score_list[-n_val:]
            )


class TestStatSigDetector(TestCase):
    def test_detector(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = np.random.randn(len(previous_seq))
        ts_init = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[0:30], "value": values[0:30]})
        )

        ts_later = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[30:35], "value": values[30:35]})
        )

        ss_detect = StatSigDetectorModel(n_control=20, n_test=7)

        pred_later = ss_detect.fit_predict(historical_data=ts_init, data=ts_later)
        ss_detect.visualize()

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), len(ts_later))

    def test_pmm_use_case(self) -> None:
        random.seed(100)
        time_unit = 86400
        hist_data_time = [x * time_unit for x in range(0, 28)]
        data_time = [x * time_unit for x in range(28, 35)]

        hist_data_value = [random.normalvariate(100, 10) for _ in range(0, 28)]
        data_value = [random.normalvariate(130, 10) for _ in range(28, 35)]

        hist_ts = TimeSeriesData(
            time=pd.Series(hist_data_time),
            value=pd.Series(hist_data_value),
            use_unix_time=True,
            unix_time_units="s",
        )
        data_ts = TimeSeriesData(
            time=pd.Series(data_time),
            value=pd.Series(data_value),
            use_unix_time=True,
            unix_time_units="s",
        )

        ss_detect2 = StatSigDetectorModel(
            n_control=20 * time_unit, n_test=7 * time_unit, time_unit="S"
        )
        pred_later2 = ss_detect2.fit_predict(historical_data=hist_ts, data=data_ts)
        self.assertEqual(len(pred_later2.scores), len(data_ts))

        self.assertTrue(pred_later2.scores.value.values.max() > 2.0)

        # case with no history
        ss_detect3 = StatSigDetectorModel(
            n_control=10 * time_unit, n_test=10 * time_unit, time_unit="S"
        )
        pred_later3 = ss_detect3.fit_predict(data=hist_ts)
        self.assertEqual(len(pred_later3.scores), len(hist_ts))

    def test_no_historical_data(self) -> None:
        n = 35
        control_time = pd.date_range(start="2018-01-01", freq="D", periods=n)
        control_val = [random.normalvariate(100, 10) for _ in range(n)]
        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))

        n_control = 5
        n_test = 5

        ss_detect3 = StatSigDetectorModel(n_control=n_control, n_test=n_test)
        anom = ss_detect3.fit_predict(data=hist_ts)
        self.assertEqual(len(anom.scores), n)

        # for the first n_control + n_test  - 1 values, score is zero,
        # afterwards it is non zero once we reach (n_control + n_test) data points
        for i in range(n_control + n_test - 1):
            self.assertEqual(anom.scores.value.iloc[i], 0.0)

        self.assertNotEqual(anom.scores.value.iloc[n_control + n_test - 1], 0.0)

    def test_not_enough_historical_data(self) -> None:
        n_control = 12
        n_test = 8
        num_control = 8
        num_test = 12
        import random

        control_time = pd.date_range(start="2018-01-01", freq="D", periods=num_control)

        test_time = pd.date_range(start="2018-01-09", freq="D", periods=num_test)
        control_val = [random.normalvariate(100, 10) for _ in range(num_control)]
        test_val = [random.normalvariate(120, 10) for _ in range(num_test)]

        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
        data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))

        ss_detect = StatSigDetectorModel(n_control=n_control, n_test=n_test)
        anom = ss_detect.fit_predict(data=data_ts, historical_data=hist_ts)

        self.assertEqual(len(anom.scores), len(data_ts))
        # until we reach n_control + n_test, we get zeroes
        # non zero afterwards

        for i in range(n_control + n_test - num_control - 1):
            self.assertEqual(anom.scores.value.iloc[i], 0.0)

        self.assertNotEqual(anom.scores.value.iloc[-1], 0.0)

    def test_logging(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 3

        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = [np.random.randn(len(previous_seq)) for _ in range(num_seq)]

        ts_init = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:30]},
                    **{f"value_{i}": values[i][0:30] for i in range(num_seq)},
                }
            )
        )

        ts_later = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[30:35]},
                    **{f"value_{i}": values[i][30:35] for i in range(num_seq)},
                }
            )
        )

        ss_detect = StatSigDetectorModel(n_control=20, n_test=7)
        self.assertEqual(ss_detect.n_test, 7)
        with self.assertRaises(ValueError):
            ss_detect.fit_predict(historical_data=ts_init, data=ts_later)


class TestMultiStatSigDetector(TestCase):
    def test_multi_detector(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 3

        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = [np.random.randn(len(previous_seq)) for _ in range(num_seq)]

        ts_init = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:30]},
                    **{f"value_{i}": values[i][0:30] for i in range(num_seq)},
                }
            )
        )

        ts_later = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[30:35]},
                    **{f"value_{i}": values[i][30:35] for i in range(num_seq)},
                }
            )
        )

        ss_detect = MultiStatSigDetectorModel(n_control=20, n_test=7)
        self.assertEqual(ss_detect.n_test, 7)
        pred_later = ss_detect.fit_predict(historical_data=ts_init, data=ts_later)

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), len(ts_later))

        # rename the time series and make sure everthing still works as it did above
        ts_init_renamed = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:30]},
                    **{f"ts_{i}": values[i][0:30] for i in range(num_seq)},
                }
            )
        )

        ts_later_renamed = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[30:35]},
                    **{f"ts_{i}": values[i][30:35] for i in range(num_seq)},
                }
            )
        )

        ss_detect = MultiStatSigDetectorModel(n_control=20, n_test=7)
        self.assertEqual(ss_detect.n_test, 7)
        pred_later = ss_detect.fit_predict(
            historical_data=ts_init_renamed, data=ts_later_renamed
        )

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), len(ts_later_renamed))

    def test_no_historical_data(self) -> None:
        n = 35
        num_seq = 3
        control_time = pd.date_range(start="2018-01-01", freq="D", periods=n)
        control_val = [
            [random.normalvariate(100, 10) for _ in range(n)] for _ in range(num_seq)
        ]

        hist_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": control_time},
                    **{f"ts_{i}": control_val[i] for i in range(num_seq)},
                }
            )
        )

        n_control = 5
        n_test = 5

        ss_detect3 = MultiStatSigDetectorModel(n_control=n_control, n_test=n_test)
        anom = ss_detect3.fit_predict(data=hist_ts)
        self.assertEqual(len(anom.scores), n)

        # for the first n_control + n_test  - 1 values, score is zero,
        # afterwards it is non zero once we reach (n_control + n_test) data points
        for i in range(n_control + n_test - 1):
            self.assertEqual(
                anom.scores.value.iloc[i, :].tolist(), np.zeros(num_seq).tolist()
            )

        for j in range(anom.scores.value.shape[1]):
            self.assertNotEqual(anom.scores.value.iloc[n_control + n_test - 1, j], 0.0)

    def test_not_enough_historical_data(self) -> None:
        n_control = 12
        n_test = 8
        num_control = 8
        num_test = 12
        num_seq = 3
        import random

        control_time = pd.date_range(start="2018-01-01", freq="D", periods=num_control)

        test_time = pd.date_range(start="2018-01-09", freq="D", periods=num_test)
        control_val = [
            [random.normalvariate(100, 10) for _ in range(num_control)]
            for _ in range(num_seq)
        ]
        test_val = [
            [random.normalvariate(120, 10) for _ in range(num_test)]
            for _ in range(num_seq)
        ]

        hist_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": control_time},
                    **{f"ts_{i}": control_val[i] for i in range(num_seq)},
                }
            )
        )

        data_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": test_time},
                    **{f"ts_{i}": test_val[i] for i in range(num_seq)},
                }
            )
        )

        ss_detect = MultiStatSigDetectorModel(n_control=n_control, n_test=n_test)
        anom = ss_detect.fit_predict(data=data_ts, historical_data=hist_ts)

        self.assertEqual(len(anom.scores), len(data_ts))
        # until we reach n_control + n_test, we get zeroes
        # non zero afterwards

        for i in range(n_control + n_test - num_control - 1):
            self.assertEqual(
                anom.scores.value.iloc[i, :].tolist(), np.zeros(num_seq).tolist()
            )

        for j in range(anom.scores.value.shape[1]):
            self.assertNotEqual(anom.scores.value.iloc[-1, j], 0.0)

    def test_logging(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 1

        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = [np.random.randn(len(previous_seq)) for _ in range(num_seq)]

        ts_init = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:30]},
                    **{f"value_{i}": values[i][0:30] for i in range(num_seq)},
                }
            )
        )

        ts_later = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[30:35]},
                    **{f"value_{i}": values[i][30:35] for i in range(num_seq)},
                }
            )
        )

        ss_detect = MultiStatSigDetectorModel(n_control=20, n_test=7)
        self.assertEqual(ss_detect.n_test, 7)
        with self.assertRaises(ValueError):
            ss_detect.fit_predict(historical_data=ts_init, data=ts_later)


class HourlyRatioDectorTest(TestCase):
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
        self.assertRaises(ValueError, HourlyRatioDetector, TSData_multi)

        self.assertRaises(ValueError, HourlyRatioDetector, ts_data_daily)

        self.assertRaises(ValueError, HourlyRatioDetector, TSData_empty)

        ts = self.data_generation(freq="T")
        self.assertRaises(ValueError, HourlyRatioDetector, data=ts)

        self.assertRaises(
            ValueError, HourlyRatioDetector, data=ts, aggregate="other_method"
        )

        hr = HourlyRatioDetector(ts, freq="T", aggregate="max")
        self.assertRaises(ValueError, hr.plot)


class TestMultipleWindowSloDetector(TestCase):
    def test_multi_detector_data(self) -> None:
        date_start_str = "2020-08-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 10

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        values = [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [1] * 10]

        historical_data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:8]},
                    **{f"value_{i}": values[i][0:8] for i in range(num_seq)},
                }
            )
        )

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[8:10]},
                    **{f"value_{i}": values[i][8:10] for i in range(num_seq)},
                }
            )
        )

        window_size = (2, 5)
        error_budget = (2.0 / 5, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

        # First prediction should be True and second should be False
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertFalse(pred.scores.value.iloc[1])

        window_size = (2, 5)
        error_budget = (1.0 / 5, 2.0 / 5)
        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=True,
        )
        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)
        # All predictions should be False
        self.assertFalse(pred.scores.value.iloc[0])
        self.assertFalse(pred.scores.value.iloc[1])

        window_size = (3, 5)
        error_budget = (1.0 / 5, 2.0 / 5)
        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=True,
        )
        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)
        # All predictions should be False
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertFalse(pred.scores.value.iloc[1])

    def test_multi_detector_data_2(self) -> None:
        date_start_str = "2020-08-16"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 10

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        values = [[0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [1] * 10]

        historical_data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:8]},
                    **{f"value_{i}": values[i][0:8] for i in range(num_seq)},
                }
            )
        )

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[8:10]},
                    **{f"value_{i}": values[i][8:10] for i in range(num_seq)},
                }
            )
        )

        window_size = (2, 5)
        error_budget = (2.0 / 5, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

        # First and Second predictions should be True
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertTrue(pred.scores.value.iloc[1])

        window_size = (2, 5)
        error_budget = (1.0 / 2, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

        # First prediction should be False and second should be True
        self.assertFalse(pred.scores.value.iloc[0])
        self.assertTrue(pred.scores.value.iloc[1])

    def test_multi_detector_data_Long(self) -> None:
        date_start_str = "2020-08-16"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 20

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        violation_ts = [0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
        violation_ts.extend([0] * 10)
        self.assertEqual(len(violation_ts), 20)
        values = [violation_ts, [1] * len(violation_ts)]

        historical_data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:8]},
                    **{f"value_{i}": values[i][0:8] for i in range(num_seq)},
                }
            )
        )

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[8 : len(violation_ts)]},
                    **{
                        f"value_{i}": values[i][8 : len(violation_ts)]
                        for i in range(num_seq)
                    },
                }
            )
        )

        window_size = (2, 5)
        error_budget = (2.0 / 5, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

        # First and Second predictions should be True
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertTrue(pred.scores.value.iloc[1])

    def test_multi_detector_none_historical_data(self) -> None:
        date_start_str = "2020-08-16"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 20

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        values = [[0] * num_days, [1] * num_days]

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": values[i] for i in range(num_seq)},
                }
            )
        )

        window_size = (2, 5)
        error_budget = (2.0 / 5, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(data=data)

        # Since there are not enough historical_data, data will be divided
        # Prediction should returns scores of same length or original data
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

    def test_choose_model_parameters(self) -> None:
        sli_error_budget = 0.01
        error_budget = (1.0 * sli_error_budget, 0.1 * sli_error_budget)

        precision_ttd_tradeoff_factor = 0.8
        slo_detect = WindowSloDetectorModel(
            precision_ttd_tradeoff_factor=precision_ttd_tradeoff_factor,
            sli_error_budget=sli_error_budget,
            windows_have_same_end=False,
        )
        window_size = (120, 1440)
        self.assertEqual(slo_detect.window_size, window_size)
        self.assertEqual(slo_detect.error_budget, error_budget)

        precision_ttd_tradeoff_factor = 0.5
        slo_detect = WindowSloDetectorModel(
            precision_ttd_tradeoff_factor=precision_ttd_tradeoff_factor,
            sli_error_budget=sli_error_budget,
            windows_have_same_end=False,
        )
        window_size = (60, 720)
        self.assertEqual(slo_detect.window_size, window_size)
        self.assertEqual(slo_detect.error_budget, error_budget)

        precision_ttd_tradeoff_factor = 0.2
        slo_detect = WindowSloDetectorModel(
            precision_ttd_tradeoff_factor=precision_ttd_tradeoff_factor,
            sli_error_budget=sli_error_budget,
            windows_have_same_end=False,
        )
        window_size = (30, 360)
        self.assertEqual(slo_detect.window_size, window_size)
        self.assertEqual(slo_detect.error_budget, error_budget)


class TestSingleWindowSloDetector(TestCase):
    def test_multi_detector_data(self) -> None:
        date_start_str = "2020-08-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 10

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        values = [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [1] * 10]

        historical_data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:8]},
                    **{f"value_{i}": values[i][0:8] for i in range(num_seq)},
                }
            )
        )

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[8:10]},
                    **{f"value_{i}": values[i][8:10] for i in range(num_seq)},
                }
            )
        )

        window_size = 5
        error_budget = 2.0 / 5

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget[0], error_budget)
        self.assertEqual(slo_detect.window_size[0], window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))

        # First prediction should be True and second should be False
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertFalse(pred.scores.value.iloc[1])


class TestCUSUMDetectorModel(TestCase):
    def test_increase(self) -> None:
        np.random.seed(100)
        scan_window = 24 * 60 * 60  # in seconds
        historical_window = 3 * 24 * 60 * 60  # in seconds
        test_data_window = 16  # in hours
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 156), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_increase)

        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )
        score_tsd = model.fit_predict(
            data=tsd[-test_data_window:], historical_data=tsd[:-test_data_window]
        ).scores

        self.assertEqual(len(score_tsd), test_data_window)
        # make sure the time series name are the same
        self.assertTrue(score_tsd.value.name == tsd.value.name)
        # the regression is detected
        self.assertEqual((score_tsd.value > 0).sum(), 12)
        score_tsd = model._predict(
            data=tsd[-test_data_window:],
            score_func=CusumScoreFunction.percentage_change.value,
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value > 0).sum(), 12)
        score_tsd = model._predict(
            data=tsd[-test_data_window:], score_func=CusumScoreFunction.z_score.value
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value > 0).sum(), 12)

        serialized_model = model.serialize()
        self.assertIsInstance(serialized_model, bytes)
        model_new = CUSUMDetectorModel(serialized_model)
        self.assertEqual(model_new, model)
        self.assertNotEqual(
            model,
            CUSUMDetectorModel(
                scan_window=scan_window, historical_window=historical_window
            ),
        )

    def test_decrease(self) -> None:
        np.random.seed(100)
        scan_window = 24 * 60 * 60  # in seconds
        historical_window = 3 * 24 * 60 * 60  # in seconds
        test_data_window = 6  # in hours
        df_decrease = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(2, 0.2, 156), np.random.normal(1, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_decrease)

        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )
        score_tsd = model.fit_predict(
            data=tsd[-test_data_window:], historical_data=tsd[:-test_data_window]
        ).scores
        score_tsd = model._predict(
            data=tsd[-test_data_window:], score_func=CusumScoreFunction.change.value
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value < 0).sum(), test_data_window)
        score_tsd = model._predict(
            data=tsd[-test_data_window:],
            score_func=CusumScoreFunction.percentage_change.value,
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value < 0).sum(), test_data_window)
        score_tsd = model._predict(
            data=tsd[-test_data_window:], score_func=CusumScoreFunction.z_score.value
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value < 0).sum(), test_data_window)

    def test_adhoc(self) -> None:
        np.random.seed(100)
        historical_window = 48 * 60 * 60  # in seconds
        scan_window = 11 * 60 * 60 + 50  # in seconds
        n = 168
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [
                        np.random.normal(1, 0.2, 48),
                        np.random.normal(0.2, 0.1, 12),
                        np.random.normal(1, 0.2, 60),
                        np.random.normal(2, 0.2, 24),
                        np.random.normal(0.9, 0.2, 24),
                    ]
                ),
                "time": pd.date_range("2020-01-01", periods=n, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_increase)
        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )
        score_tsd = model.fit_predict(data=tsd).scores
        self.assertEqual(len(score_tsd), len(tsd))
        # the regression is went away
        # pyre-fixme[29]: `Series` is not a function.
        self.assertEqual(score_tsd.value[-6:].sum(), 0)
        # the increase regression is detected
        self.assertEqual((score_tsd.value > 0.5).sum(), 24)
        # the decrease regression is detected
        self.assertEqual((score_tsd.value < -0.45).sum(), 12)

        # test not enough data
        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )
        score_tsd = model.fit_predict(data=tsd[-4:], historical_data=tsd[-8:-4]).scores

        self.assertEqual(len(score_tsd), len(tsd[-4:]))
        self.assertEqual(score_tsd.value.sum(), 0)

        model = CUSUMDetectorModel(scan_window=scan_window, historical_window=3600)
        score_tsd = model.fit_predict(data=tsd[-8:]).scores

        self.assertEqual(len(score_tsd), len(tsd[-8:]))
        self.assertEqual(score_tsd.value.sum(), 0)

    def test_missing_data(self) -> None:
        df = pd.DataFrame(
            {
                "ts_value": [0] * 8,
                "time": [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-06",
                    "2020-01-08",
                    "2020-01-09",
                ],
            }
        )
        tsd = TimeSeriesData(df)
        # We also assume a bad input here
        model = CUSUMDetectorModel(
            scan_window=24 * 3600,
            historical_window=2 * 24 * 3600,
        )
        score_tsd = model.fit_predict(
            data=tsd,
        ).scores

        self.assertEqual(len(score_tsd), len(tsd))
        self.assertTrue((score_tsd.time.values == tsd.time.values).all())

    def test_streaming(self) -> None:
        np.random.seed(100)
        historical_window = 48 * 60 * 60  # in seconds
        scan_window = 12 * 60 * 60  # in seconds
        n = 72
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 60), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=n, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_increase)
        # Priming the model
        model = CUSUMDetectorModel(
            historical_window=historical_window, scan_window=scan_window
        )
        model.fit(data=tsd[:48])
        pre_serialized_model = model.serialize()

        anomaly_score = TimeSeriesData(
            time=pd.Series(), value=pd.Series([], name="ts_value")
        )
        # feeding 1 new data point a time
        for i in range(48, n):
            model = CUSUMDetectorModel(
                serialized_model=pre_serialized_model,
                historical_window=historical_window,
                scan_window=scan_window,
            )
            anomaly_score.extend(
                model.fit_predict(
                    data=tsd[i : i + 1], historical_data=tsd[i - 48 : i]
                ).scores,
                validate=False,
            )
            pre_serialized_model = model.serialize()
        anomaly_score.validate_data(validate_frequency=True, validate_dimension=False)
        self.assertEqual(len(anomaly_score), n - 48)
        self.assertTrue(8 <= (anomaly_score.value > 0).sum() <= 12)

    def test_decomposing_seasonality(self) -> None:
        np.random.seed(100)
        historical_window = 10 * 24 * 60 * 60  # in seconds
        scan_window = 12 * 60 * 60  # in seconds
        n = 480
        periodicity = 24

        df_sin = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=n, freq="H"),
                "ts_value": np.concatenate([20 * np.ones(n // 2), 21 * np.ones(n // 2)])
                + 4 * np.sin(2 * np.pi / periodicity * np.arange(0, n)),
            }
        )

        # removing a few data points to test the missing value handling as well
        tsd = TimeSeriesData(pd.concat([df_sin[:100], df_sin[103:]]))

        model = CUSUMDetectorModel(
            scan_window=scan_window,
            historical_window=historical_window,
            remove_seasonality=True,
            score_func=CusumScoreFunction.percentage_change,
        )
        score_tsd = model.fit_predict(
            data=tsd,
        ).scores

        self.assertEqual(len(score_tsd), len(tsd))
        # the scores set to zero after about 7 days
        # pyre-fixme[29]: `Series` is not a function.
        self.assertEqual(score_tsd.value[-72:].sum(), 0)
        # the increase regression is detected and is on for about 7 days
        self.assertEqual((score_tsd.value > 0.01).sum(), 162)
        # the decrease regression is detected
        self.assertTrue((score_tsd.time.values == tsd.time.values).all())
        # make sure the time series name are the same
        self.assertTrue(score_tsd.value.name == tsd.value.name)

    def test_raise(self) -> None:
        np.random.seed(100)
        historical_window = 48 * 60 * 60  # in seconds
        scan_window = 24 * 60 * 60  # in seconds
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 156), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )

        tsd = TimeSeriesData(df_increase)
        with self.assertRaisesRegex(
            ValueError,
            "Step window should smaller than scan window to ensure we have overlap for scan windows.",
        ):
            model = CUSUMDetectorModel(
                scan_window=scan_window,
                step_window=scan_window * 2,
                historical_window=historical_window,
            )

        with self.assertRaisesRegex(ValueError, "direction can only be right or left"):
            model = CUSUMDetectorModel(
                scan_window=scan_window, historical_window=historical_window
            )
            model._time2idx(tsd, tsd.time.iloc[0], "")

        with self.assertRaisesRegex(
            ValueError,
            "You must either provide serialized model or values for scan_window and historical_window.",
        ):
            model = CUSUMDetectorModel()

        with self.assertRaisesRegex(
            ValueError, "Not able to infer freqency of the time series"
        ):
            model = CUSUMDetectorModel(
                scan_window=scan_window, historical_window=historical_window
            )
            model.fit_predict(
                data=TimeSeriesData(
                    df=pd.DataFrame(
                        {
                            "value": [0] * 8,
                            "time": [
                                "2020-01-01",
                                "2020-01-02",
                                "2020-01-04",
                                "2020-01-05",
                                "2020-01-07",
                                "2020-01-08",
                                "2020-01-10",
                                "2020-01-11",
                            ],
                        }
                    )
                )
            )

        with self.assertRaisesRegex(
            ValueError, r"predict is not implemented, call fit_predict\(\) instead"
        ):
            model = CUSUMDetectorModel(
                scan_window=scan_window, historical_window=historical_window
            )
            model.predict(data=tsd)


class TestProphetDetector(TestCase):
    def create_random_ts(self, seed, length, magnitude, slope_factor):
        np.random.seed(seed)
        sim = Simulator(n=length, freq="1D", start=pd.to_datetime("2020-01-01"))

        sim.add_trend(magnitude=magnitude * np.random.rand() * slope_factor)
        sim.add_seasonality(
            magnitude * np.random.rand(),
            period=timedelta(days=7),
        )

        sim.add_noise(magnitude=0.1 * magnitude * np.random.rand())
        return sim.stl_sim()

    def add_smooth_anomaly(self, ts, seed, start_index, length, magnitude):
        # Add an anomaly that is half of a sine wave
        # start time and freq don't matter, since we only care about the values
        np.random.seed(seed)

        anomaly_sim = Simulator(n=length, freq="1D", start=pd.to_datetime("2020-01-01"))
        anomaly_sim.add_seasonality(magnitude, period=timedelta(days=2 * length))
        # anomaly_sim.add_noise(magnitude=0.3 * magnitude * np.random.rand())

        anomaly_ts = anomaly_sim.stl_sim()
        for i in range(0, length):
            ts.value.iloc[start_index + i] += anomaly_ts.value[i]

    def test_no_anomaly(self) -> None:
        # Prophet should not find any anomalies on a well formed synthetic time series
        for i in range(0, 5):
            ts = self.create_random_ts(i, 100, 10, 2)

            model = ProphetDetectorModel()
            model.fit(ts[:90])

            # alternate between using the current model and using serialized model
            if i % 2 == 0:
                serialized_model = model.serialize()
                model = ProphetDetectorModel(serialized_model=serialized_model)

            res = model.predict(ts[90:])
            self.assertEqual(len(res.scores), 10)
            anomaly_found = res.scores.min < -0.3 or res.scores.max > 0.3
            self.assertFalse(anomaly_found)

    def test_anomaly(self) -> None:
        # Prophet should find anomalies
        for i in range(0, 5):
            ts = self.create_random_ts(i, 100, 10, 2)
            self.add_smooth_anomaly(ts, i, 90, 10, 10)

            model = ProphetDetectorModel()
            model.fit(ts[:90])

            # alternate between using the current model and using serialized model
            if i % 2 == 0:
                serialized_model = model.serialize()
                model = ProphetDetectorModel(serialized_model=serialized_model)

            res = model.predict(ts[90:])
            self.assertEqual(len(res.scores), 10)
            anomaly_found = res.scores.min < -0.3 or res.scores.max > 0.3
            self.assertTrue(anomaly_found)

    def test_fit_predict(self) -> None:
        ts = self.create_random_ts(0, 100, 10, 2)
        self.add_smooth_anomaly(ts, 0, 90, 10, 10)

        model = ProphetDetectorModel()
        model.fit(ts[:90])
        res0 = model.predict(ts[90:])

        model = ProphetDetectorModel()
        res1 = model.fit_predict(data=ts[90:], historical_data=ts[:90])

        self.assertEqual(res0.scores.value.to_list(), res1.scores.value.to_list())


class TestSlowDriftDetector(TestCase):
    def generate_ts_with_slope_and_seasonality(
        self,
        num_points: int = 1000,
        end_time: int = 0,
        level: float = 10000.0,  # Initial Level
        slope: float = 20.0,  # Change per day
        seasonality: float = 0.9,  # Multiplier on weekends
    ) -> TimeSeriesData:
        granularity = 86400
        if end_time <= 0:
            end_time = int(time.time())
        start_time = end_time - num_points * granularity
        times = []
        values = []
        for i in range(num_points):
            timestamp = start_time + i * granularity
            value = level + i * slope
            if i % 7 >= 5:
                value *= seasonality
            times.append(timestamp)
            values.append(value)
        return TimeSeriesData(
            time=pd.Series(times),
            value=pd.Series(values),
            use_unix_time=True,
            unix_time_units="s",
        )

    def test_time_series_data_to_data_points(self) -> None:
        ts = TimeSeriesData(
            df=pd.DataFrame.from_dict(
                {
                    "time": [0, 1],
                    "value": [1.5, 3.0],
                }
            ),
            use_unix_time=True,
            unix_time_units="s",
        )

        result = time_series_to_data_points(ts)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].timestamp, 0)
        self.assertEqual(result[0].value, 1.5)
        self.assertEqual(result[1].timestamp, 1)
        self.assertEqual(result[1].value, 3.0)

    def test_constructor(self) -> None:
        model = SlowDriftDetectorModel(
            slow_drift_window=100,
            algorithm_version=2,
            seasonality_period=14,
            seasonality_num_points=12,
        )
        slow_drift_model = model.model
        model_data = slow_drift_model.get_model_data()
        self.assertEquals(model_data.slowDriftWindow, 100)
        self.assertEquals(model_data.algorithmVersion, 2)
        esp = slow_drift_model.get_parameters()
        self.assertEquals(esp.seasonalityPeriod, 14)
        self.assertEquals(esp.seasonalityNumPoints, 12)
        self.assertIsNone(slow_drift_model.get_ongoing_anomaly())

    def test_holt_winters(self) -> None:
        model = SlowDriftDetectorModel(
            slow_drift_window=28 * 86400,
            algorithm_version=3,
            seasonality_period=0,
            seasonality_num_points=0,
        )

        ts = self.generate_ts_with_slope_and_seasonality()
        result = model.fit_predict(ts)
        anomaly_scores = result.scores
        # Anomalies may happen near the beginning due to initialization.  Assert that
        # none occur after things have settled down.
        scores_after_t100 = anomaly_scores.value[100:]
        self.assertEquals(len(scores_after_t100[scores_after_t100 > 0]), 0)
        slow_drift_model = model.model
        slow_drift_model.pruneData()
        # By the end of the series, trend should converge and not have much noise
        # This expected value is due to the seasonal adjustment multiplier
        expected_slope = 20.0 * (5 * 1.0 + 2 * 0.9) / 7
        for trend_dp in slow_drift_model.get_trend_series():
            self.assertAlmostEqual(trend_dp.value, expected_slope, delta=0.001)
        # Level should be linear, with the expected slope as above
        level_series = slow_drift_model.get_level_series()
        for i, level_dp in enumerate(level_series):
            expected_level = level_series[0].value + i * expected_slope
            self.assertAlmostEqual(level_dp.value, expected_level, delta=0.001)
        # Seasonality state should be such that the weekday values are all equal
        # and weekend values are 0.9x the weekdays
        seasonality = slow_drift_model._model_data.recentData.seasonalitySeries
        self.assertAlmostEqual(seasonality[0].value, seasonality[1].value, delta=0.001)
        self.assertAlmostEqual(
            seasonality[5].value, 0.9 * seasonality[0].value, delta=0.001
        )

    def test_serialize(self) -> None:
        model = SlowDriftDetectorModel(
            slow_drift_window=28 * 86400,
            algorithm_version=3,
            seasonality_period=14,
            seasonality_num_points=12,
        )

        ts = self.generate_ts_with_slope_and_seasonality()
        model.fit(ts)

        serialized_model = model.serialize()
        self.assertIsInstance(serialized_model, bytes)
        loaded_model = SlowDriftDetectorModel(
            slow_drift_window=28 * 86400,
            algorithm_version=3,
            seasonality_period=14,
            seasonality_num_points=12,
            serialized_model=serialized_model,
        )
        self.assertEqual(model.model._model_data, loaded_model.model._model_data)


class TestChangepointEvaluator(TestCase):
    def test_eval_agg(self) -> None:
        eval_1 = Evaluation(dataset_name="eg_1", precision=0.3, recall=0.5, f_score=0.6)

        eval_2 = Evaluation(dataset_name="eg_2", precision=0.3, recall=0.5, f_score=0.7)

        eval_agg = EvalAggregate(eval_list=[eval_1, eval_2])
        avg_f_score = eval_agg.get_avg_f_score()
        self.assertAlmostEqual(avg_f_score, 0.65, places=4)

        eval_agg_1 = EvalAggregate(eval_list=[eval_1, eval_2])
        avg_precision = eval_agg_1.get_avg_precision()
        self.assertAlmostEqual(avg_precision, 0.3, places=4)

        eval_agg_2 = EvalAggregate(eval_list=[eval_1, eval_2])
        avg_recall = eval_agg_2.get_avg_recall()
        self.assertAlmostEqual(avg_recall, 0.5, places=4)

    def test_f_measure(self) -> None:
        """
        tests the correctness of f-measure, by comparing results with
        https://arxiv.org/pdf/2003.06222.pdf and TCPDBench github
        project
        """
        brent_spot_anno = {
            "6": [219, 230, 288],
            "8": [227, 381],
            "9": [86, 219, 230, 279, 375],
            "12": [169, 172, 217, 228, 287, 368, 382, 389, 409],
            "13": [170, 180, 219, 229, 246, 271, 286, 379, 409, 444, 483],
        }

        brent_spot_prophet_default_cploc = [259, 279, 299, 319, 339, 359]

        # scores are defined in
        # https://github.com/alan-turing-institute/TCPDBench/blob/master/analysis/output/summaries/summary_brent_spot.json

        f_brent_spot = f_measure(
            annotations=brent_spot_anno, predictions=brent_spot_prophet_default_cploc
        )
        self.assertAlmostEqual(f_brent_spot["f_score"], 0.2485875706214689, places=3)
        self.assertAlmostEqual(f_brent_spot["precision"], 0.2857142857142857, places=3)
        self.assertAlmostEqual(f_brent_spot["recall"], 0.21999999999999997, places=3)

    def test_evaluator(self) -> None:
        date_range = pd.date_range(start="2010-02-01", end="2020-03-31", freq="M")
        date_range_start = [x + timedelta(days=1) for x in date_range]
        y_m_d_str = [datetime.strftime(x, "%Y-%m-%d") for x in date_range_start]
        y_m_str = [datetime.strftime(x, "%Y-%m") for x in date_range_start]
        int_str = [str(x) for x in range(len(date_range_start))]
        int_val = list(range(len(date_range_start)))

        val = np.random.randn(len(date_range_start))

        eg_anno = {"1": [2, 6, 10], "2": [3, 6]}
        y_m_d_dict = {k: v for k, v in zip(y_m_d_str, val)}
        y_m_dict = {k: v for k, v in zip(y_m_str, val)}
        int_dict = {k: v for k, v in zip(int_str, val)}
        int_val_dict = {k: v for k, v in zip(int_val, val)}

        eg_df = pd.DataFrame(
            [
                {
                    "dataset_name": "eg_1",
                    "time_series": str(y_m_d_dict),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_2",
                    "time_series": str(y_m_dict),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_3",
                    "time_series": str(int_dict),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_4",
                    "time_series": str(int_val_dict),
                    "annotation": str(eg_anno),
                },
            ]
        )

        model_params = {"p_value_cutoff": 5e-3, "comparison_window": 2}

        # Test RobustStatDetector
        # pyre-fixme[6]: Expected `Detector` for 1st param but got
        #  `Type[RobustStatDetector]`.
        turing_2 = TuringEvaluator(detector=RobustStatDetector)
        eval_agg_2_df = turing_2.evaluate(data=eg_df, model_params=model_params)
        self.assertEqual(eval_agg_2_df.shape[0], eg_df.shape[0])

        # Test CUSUMDetector
        # pyre-fixme[6]: Expected `Detector` for 1st param but got
        #  `Type[CUSUMDetector]`.
        turing_3 = TuringEvaluator(detector=CUSUMDetector)
        eval_agg_3_df = turing_3.evaluate(data=eg_df)
        self.assertEqual(eval_agg_3_df.shape[0], eg_df.shape[0])

        # Test BOCPDDetector
        # pyre-fixme[6]: Expected `Detector` for 1st param but got `Type[BOCPDetector]`.
        turing_4 = TuringEvaluator(detector=BOCPDetector)
        eval_agg_4_df = turing_4.evaluate(data=eg_df)
        self.assertEqual(eval_agg_4_df.shape[0], eg_df.shape[0])

        # test the eval_agg
        eval_agg_4 = turing_4.get_eval_aggregate()
        eval_agg_df = eval_agg_4.get_eval_dataframe()
        self.assertEqual(eval_agg_df.shape[0], eg_df.shape[0])
        avg_precision = eval_agg_4.get_avg_precision()
        avg_recall = eval_agg_4.get_avg_recall()
        avg_f_score = eval_agg_4.get_avg_f_score()
        self.assertTrue(0.0 <= avg_precision <= 1.0)
        self.assertTrue(0.0 <= avg_recall <= 1.0)
        self.assertTrue(0.0 <= avg_f_score <= 1.0)

        # test load data
        # pyre-fixme[6]: Expected `Detector` for 1st param but got
        #  `Type[RobustStatDetector]`.
        turing_5 = TuringEvaluator(detector=RobustStatDetector)
        # pyre-fixme[6]: Expected `DataFrame` for 1st param but got `None`.
        eval_agg_5_df = turing_5.evaluate(data=None, model_params=model_params)
        self.assertTrue(eval_agg_5_df.shape[0] > 0)

        # test ignore list
        # pyre-fixme[6]: Expected `Detector` for 1st param but got
        #  `Type[RobustStatDetector]`.
        turing_6 = TuringEvaluator(detector=RobustStatDetector)
        eval_agg_6_df = turing_6.evaluate(
            data=eg_df, model_params=model_params, ignore_list=["eg_2"]
        )
        self.assertEqual(eval_agg_6_df.shape[0], eg_df.shape[0] - 1)

        # test the detectormodels

        # test BOCPD
        # pyre-fixme[6]: Expected `Detector` for 1st param but got
        #  `Type[BocpdDetectorModel]`.
        turing_7 = TuringEvaluator(detector=BocpdDetectorModel, is_detector_model=True)
        # pyre-fixme[6]: Expected `Dict[str, float]` for 2nd param but got `None`.
        eval_agg_7_df = turing_7.evaluate(data=eg_df, model_params=None)
        self.assertEqual(eval_agg_7_df.shape[0], eg_df.shape[0])

        # test Statsig
        num_secs_in_month = 86400 * 30
        statsig_model_params = {"n_control": 7 * num_secs_in_month,
                                "n_test": 7 * num_secs_in_month,
                                "time_unit": "sec"}

        # pyre-fixme[6]: Expected `Detector` for 1st param but got
        #  `Type[StatSigDetectorModel]`.
        turing_8 = TuringEvaluator(detector=StatSigDetectorModel, is_detector_model=True)
        # pyre-fixme[6]: Expected `Dict[str, float]` for 2nd param but got
        #  `Dict[str, typing.Union[int, str]]`.
        eval_agg_8_df = turing_8.evaluate(data=eg_df, model_params=statsig_model_params,
                                          alert_style_cp=False, threshold_low=-5.0,
                                          threshold_high=5.0)

        self.assertEqual(eval_agg_8_df.shape[0], eg_df.shape[0])

        # test CUSUM
        # since CUSUM needs daily data, constructing another eg_df
        eg_start_unix_time = 1613764800
        num_secs_in_day = 3600 * 24

        date_range_daily = pd.date_range(start="2020-03-01", end="2020-03-31", freq="D")
        date_range_start_daily = [x + timedelta(days=1) for x in date_range_daily]
        y_m_d_str_daily = [datetime.strftime(x, "%Y-%m-%d") for x in date_range_start_daily]
        int_daily = [(eg_start_unix_time + x * num_secs_in_day) for x in range(len(date_range_start_daily))]
        int_str_daily = [str(x) for x in int_daily]

        val_daily = np.random.randn(len(date_range_start_daily))

        y_m_d_dict_daily = {k: v for k, v in zip(y_m_d_str_daily, val_daily)}
        int_dict_daily = {k: v for k, v in zip(int_daily, val_daily)}
        int_str_dict_daily = {k: v for k, v in zip(int_str_daily, val_daily)}

        eg_df_daily = pd.DataFrame(
            [
                {
                    "dataset_name": "eg_1",
                    "time_series": str(y_m_d_dict_daily),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_3",
                    "time_series": str(int_dict_daily),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_4",
                    "time_series": str(int_str_dict_daily),
                    "annotation": str(eg_anno),
                },
            ]
        )

        cusum_model_params = {"scan_window": 8 * num_secs_in_day,
                              "historical_window": 8 * num_secs_in_day,
                              "threshold": 0.01,
                              "delta_std_ratio": 1.0,
                              "change_directions": ["increase", "decrease"],
                              "score_func": CusumScoreFunction.percentage_change,
                              "remove_seasonality": False}

        # pyre-fixme[6]: Expected `Detector` for 1st param but got
        #  `Type[CUSUMDetectorModel]`.
        turing_9 = TuringEvaluator(detector=CUSUMDetectorModel, is_detector_model=True)
        # pyre-fixme[6]: Expected `Dict[str, float]` for 2nd param but got
        #  `Dict[str, typing.Union[typing.List[str], CusumScoreFunction, float]]`.
        eval_agg_9_df = turing_9.evaluate(data=eg_df_daily, model_params=cusum_model_params,
                                          alert_style_cp=True, threshold_low=-0.1,
                                          threshold_high=0.1)

        self.assertEqual(eval_agg_9_df.shape[0], eg_df_daily.shape[0])


if __name__ == "__main__":
    unittest.main()
