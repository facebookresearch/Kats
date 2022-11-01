# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta
from operator import attrgetter
from typing import List, Tuple, Type, Union
from unittest import TestCase

import numpy as np
import pandas as pd

from kats.consts import TimeSeriesData
from kats.detectors.interval_detector import ABDetectorModel, TestStatistic
from parameterized.parameterized import parameterized
from scipy.stats import norm


_SERIALIZED = b'{"alpha": 0.1, "duration": 1, "test_direction": "b_greater", "distribution": "binomial", "test_statistic": "absolute_difference"}'

_Z_SCORE: float = 1.6448536269514722
_P_VALUE: float = 0.05


def _dp_solve(p: float, n: int, m: int) -> np.ndarray:
    """dp solution used to validate `probability_of_at_least_one_m_run_in_n_trials`."""
    a = np.zeros(n)
    p_m = p**m
    q = 1 - p
    # [0, ...(m - 2)..., p ** m, 0, .., 0] -> (n, )
    a[m - 1] = p_m
    # [p^(m - 1), p^(m - 2), ..., p^0] -> (m, )
    f = np.power(p, np.arange(start=m - 1, stop=-1, step=-1))
    # [(1 - p), (1 - p), ..., (1 - p)] -> (m, )
    f *= np.array([q] * m)
    # Order m recursion - O(n) complexity.
    for i in range(n - m):
        a[i + m] = np.dot(f, a[i : i + m]) + p_m
    return a


class TestABDetectorModel(TestCase):
    def setUp(self) -> None:
        date_start = datetime.strptime("2020-03-01", "%Y-%m-%d")
        self.time = [date_start + timedelta(hours=x) for x in range(60)]
        self.value_a = np.array([0.05] * len(self.time))
        self.value_b = np.array([0.08] * len(self.time))
        self.variance_a = np.array([1] * len(self.time))
        self.variance_b = np.array([1] * len(self.time))
        self.sample_count_a = np.array([100] * len(self.time))
        self.sample_count_b = np.array([100] * len(self.time))
        self.effect_size = np.array([0.02] * len(self.time))
        self.df = pd.DataFrame(
            {
                "time": self.time,
                "value_a": self.value_a,
                "value_b": self.value_b,
                "variance_a": self.variance_a,
                "variance_b": self.variance_b,
                "sample_count_a": self.sample_count_a,
                "sample_count_b": self.sample_count_b,
                "effect_size": self.effect_size,
            }
        )
        self.interval_detector = ABDetectorModel(alpha=0.05, duration=1)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @parameterized.expand(
        [
            ["alpha", 0.1],
            ["duration", 1],
        ]
    )
    def test_load_from_serialized(self, attribute: str, expected: object) -> None:
        detector = ABDetectorModel(serialized_model=_SERIALIZED)
        self.assertEqual(attrgetter(attribute)(detector), expected)

    def test_serialize(self) -> None:
        detector = ABDetectorModel(alpha=0.1, duration=1)
        self.assertEqual(_SERIALIZED, detector.serialize())

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @parameterized.expand(
        [
            [None, ValueError],
            [-0.1, ValueError],
            [5, ValueError],
        ]
    )
    def test_incorrect_alpha(
        self, alpha: Union[None, float, int], expected: Type[Exception]
    ) -> None:
        with self.assertRaises(expected):
            ABDetectorModel(alpha=alpha, duration=1)

    def test_negative_duration(self) -> None:
        with self.assertRaises(ValueError):
            ABDetectorModel(alpha=0.05, duration=-1)

    def test_no_test_direction(self) -> None:
        with self.assertRaises(ValueError):
            ABDetectorModel(alpha=0.05, test_direction=None)

    def test_column_names(self) -> None:
        df = self.df.copy()
        # valid df
        self.interval_detector._validate_columns_name(df)
        df.drop(columns=["variance_a"], inplace=True)
        with self.assertRaises(ValueError):
            self.interval_detector._validate_columns_name(df)

    def test_non_negative_columns(self) -> None:
        df = self.df.copy()
        # valid df
        self.interval_detector._validate_data(df)
        df["variance_a"].iloc[10] = -0.1
        with self.assertRaises(ValueError):
            self.interval_detector._validate_data(df)

    def test_positive_columns(self) -> None:
        df = self.df.copy()
        # valid df
        self.interval_detector._validate_data(df)
        df["effect_size"].iloc[10] = -0.1
        df["sample_count_a"].iloc[2] = 0.0
        with self.assertRaises(ValueError):
            self.interval_detector._validate_data(df)

    def test_integer_columns(self) -> None:
        df = self.df.copy()
        # valid df
        self.interval_detector._validate_data(df)
        df["sample_count_a"].iloc[10] = 1.5
        with self.assertRaises(ValueError):
            self.interval_detector._validate_data(df)

    def test_na_in_data(self) -> None:
        df = self.df.copy()
        # valid df
        self.interval_detector._validate_data(df)
        df["effect_size"].iloc[4] = None
        with self.assertRaises(ValueError):
            self.interval_detector._validate_data(df)
        df["effect_size"].iloc[6] = 1
        df["value_a"].iloc[10] = None
        with self.assertRaises(ValueError):
            self.interval_detector._validate_data(df)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @parameterized.expand(
        [
            [[True, True, False, True], ([0, 3], [1, 3])],
            [[False, True, True, False], ([1], [2])],
            [[False, False, False, False], ([], [])],
        ]
    )
    def test_get_true_run_indices(
        self, sequence: List[bool], expected: Tuple[List[int], List[int]]
    ) -> None:
        starts, ends = self.interval_detector._get_true_run_indices(np.array(sequence))
        expected_starts, expected_ends = expected
        assert starts.tolist() == expected_starts
        assert ends.tolist() == expected_ends

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @parameterized.expand(
        [
            [0.01],
            [0.05],
            [0.1],
        ]
    )
    def test_get_critical_value_custom_duration(self, p_goal: float) -> None:
        lowest_p = self.interval_detector._get_lowest_p(
            m=3, n=100, p_goal=p_goal, r_tol=1e-3
        )
        assert np.isclose(lowest_p.p_global, p_goal, rtol=1e-3)

    def test_absolute_difference_test_statistic(self) -> None:
        df = self.df.copy()
        self.interval_detector.critical_value = (
            self.interval_detector._get_critical_value(1, 1e-5)
        )
        test_statistic = self.interval_detector.get_test_statistic(df)
        # "manually" compute z-scores
        diff = self.value_b - self.value_a - self.effect_size
        std_error = np.sqrt(
            self.value_a * (1 - self.value_a) / self.sample_count_a
            + self.value_b * (1 - self.value_b) / self.sample_count_b
        )
        z_score = diff / std_error
        upper = diff + _Z_SCORE * std_error
        lower = diff - _Z_SCORE * std_error
        assert all(np.isclose(test_statistic.test_statistic.values, z_score))
        assert all(np.isclose(test_statistic.stat_sig.values, norm.sf(z_score)))
        assert all(np.isclose(test_statistic.upper, upper))
        assert all(np.isclose(test_statistic.lower, lower))

    def test_relative_difference_test_statistic(self) -> None:
        df = self.df.copy()
        self.interval_detector.critical_value = (
            self.interval_detector._get_critical_value(1, 1e-5)
        )
        self.interval_detector.test_statistic = TestStatistic.RELATIVE_DIFFERENCE
        test_statistic = self.interval_detector.get_test_statistic(df)
        # "manually" compute z-scores
        diff = (
            np.log(self.value_b) - np.log(self.value_a) - np.log(1 + self.effect_size)
        )
        std_error = np.sqrt(
            self.value_a
            * (1 - self.value_a)
            / self.sample_count_a
            / (self.value_a**2)
            + self.value_b
            * (1 - self.value_b)
            / self.sample_count_b
            / (self.value_b**2)
        )
        z_score = diff / std_error
        upper = np.exp(diff + _Z_SCORE * std_error)
        lower = np.exp(diff - _Z_SCORE * std_error)
        assert all(np.isclose(test_statistic.test_statistic.values, z_score))
        assert all(np.isclose(test_statistic.stat_sig.values, norm.sf(z_score)))
        assert all(np.isclose(test_statistic.upper, upper))
        assert all(np.isclose(test_statistic.lower, lower))

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @parameterized.expand(
        [
            [True],
            [False],
        ]
    )
    def test_blatent_anomaly(self, consolidate_into_intervals: bool) -> None:
        """E2E test of an apparent anomaly."""
        df = self.df.copy()
        df.value_b.iloc[10] = 1.0
        detector = ABDetectorModel(serialized_model=_SERIALIZED)
        anomaly_response = detector.fit_predict(
            TimeSeriesData(df), consolidate_into_intervals=consolidate_into_intervals
        )
        assert anomaly_response.predicted_ts is not None
        assert anomaly_response.stat_sig_ts is not None
        assert anomaly_response.confidence_band is not None
        _predicted_ds: TimeSeriesData = anomaly_response.predicted_ts
        _stat_sig_ts: TimeSeriesData = anomaly_response.stat_sig_ts
        _upper: TimeSeriesData = anomaly_response.confidence_band.upper
        _lower: TimeSeriesData = anomaly_response.confidence_band.lower
        assert np.isclose(_stat_sig_ts.value.iloc[10], 0.0)
        if consolidate_into_intervals:
            assert _predicted_ds.value.iloc[10]
            assert (
                _predicted_ds.value.iloc[10] > _upper.value.iloc[10]
                or _predicted_ds.value.iloc[10] < _lower.value.iloc[10]
            )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @parameterized.expand(
        [
            [TestStatistic.ABSOLUTE_DIFFERENCE],
            [TestStatistic.RELATIVE_DIFFERENCE],
        ]
    )
    def test_blatent_anomalies(self, test_statistic: TestStatistic) -> None:
        """E2E test of apparent anomalies."""
        df = self.df.copy()
        df.value_b.iloc[10:15] = 1.0
        df.value_b.iloc[40:45] = 1.0
        detector = ABDetectorModel(serialized_model=_SERIALIZED)
        detector.test_statistic = test_statistic
        anomaly_response = detector.fit_predict(TimeSeriesData(df))
        assert anomaly_response.predicted_ts is not None
        assert anomaly_response.stat_sig_ts is not None
        _predicted_ds: TimeSeriesData = anomaly_response.predicted_ts
        _stat_sig_ts: TimeSeriesData = anomaly_response.stat_sig_ts
        assert _predicted_ds.value.iloc[10:15].all()
        assert np.isclose(_stat_sig_ts.value.iloc[10:15].values, 0.0).all()
        assert _predicted_ds.value.iloc[40:45].all()
        assert np.isclose(_stat_sig_ts.value.iloc[40:45].values, 0.0).all()

    def test_duration(self) -> None:
        """E2E test of the duration parameter."""
        df = self.df.copy()
        df.value_b.iloc[10] = 1.0
        df.value_b.iloc[20:22] = 1.0
        df.value_b.iloc[30:33] = 1.0
        detector = ABDetectorModel(serialized_model=_SERIALIZED)
        detector.duration = 2
        anomaly_response = detector.fit_predict(TimeSeriesData(df))
        assert anomaly_response.predicted_ts is not None
        assert anomaly_response.stat_sig_ts is not None
        _predicted_ds: TimeSeriesData = anomaly_response.predicted_ts
        assert ~_predicted_ds.value.iloc[10].all()
        assert _predicted_ds.value.iloc[20:22].all()
        assert _predicted_ds.value.iloc[30:33].all()

    def test_historical_data(self) -> None:
        """E2E test of the duration parameter."""
        historical_data = self.df.copy()
        date_start = historical_data.time.iloc[-1]
        time = [date_start + timedelta(hours=x) for x in range(1, 61)]
        data = self.df.copy()
        data.time = time
        data.value_b.iloc[30] = 1.0
        anomaly_response = self.interval_detector.fit_predict(
            TimeSeriesData(data), historical_data=TimeSeriesData(historical_data)
        )
        assert self.interval_detector.data is not None
        assert len(self.interval_detector.data) == 120
        assert anomaly_response.predicted_ts is not None
        assert anomaly_response.stat_sig_ts is not None
        _predicted_ds: TimeSeriesData = anomaly_response.predicted_ts
        assert _predicted_ds.value.iloc[90].all()

    def test_dp_solve(self) -> None:
        """Test the _dp_solve helper function."""
        _precomputed = [
            0.0,
            0.0025000000000000005,
            0.004875000000000001,
            0.007250000000000001,
            0.009619062500000001,
            0.011982484375,
            0.014340265625,
            0.0166924203515625,
            0.019038961951171877,
            0.021379903820312504,
            0.023715259321977544,
            0.026045041787343508,
            0.028369264515770265,
            0.030687940774880566,
            0.03300108380063563,
            0.035308706797410674,
            0.03761082293807033,
            0.03990744536404382,
            0.042198587185399976,
            0.04448426148092206,
            0.04676448129818246,
            0.049039259653617134,
            0.05130860953259994,
            0.05357254388951676,
            0.055831075647839415,
        ]
        p = 0.05
        q = 1 - p
        run_1 = _dp_solve(p=p, n=25, m=1)
        run_2 = _dp_solve(p=p, n=25, m=2)
        run_3 = _dp_solve(p=p, n=25, m=3)
        run_25 = _dp_solve(p=p, n=25, m=25)
        # Verify simple cases solved by hand.
        assert np.isclose(run_1[-1], 1 - 0.95**25)
        assert np.isclose(run_2[2], p**3 + 2 * p**2 * q)
        assert np.isclose(run_2[3], p**4 + 4 * p**3 * q + 3 * p**2 * q**2)
        assert np.isclose(run_3[3], p**4 + 2 * p**3 * q)
        assert np.isclose(run_3[4], p**5 + 3 * p**3 * q**2 + 4 * p**4 * q)
        assert np.isclose(run_25[-1], p**25)
        # Verify the entire precomputed sequence for a run of 2.
        assert np.all(np.isclose(run_2, _precomputed))

    def test_probability_of_at_least_one_m_run_in_n_trials(self) -> None:
        """Compare vectorized solution against a known and correct dp solution."""
        p = 0.05
        for m in range(1, 100):
            assert np.isclose(
                _dp_solve(p=p, n=100, m=m)[-1],
                self.interval_detector._probability_of_at_least_one_m_run_in_n_trials(
                    p=p, n=100, m=m
                ),
            )
        for n in range(1, 100):
            assert np.isclose(
                _dp_solve(p=p, n=n, m=n)[-1],
                self.interval_detector._probability_of_at_least_one_m_run_in_n_trials(
                    p=p, n=n, m=n
                ),
            )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @parameterized.expand(
        [
            [0.05, 0.0],
            [0.05, 0.1],
            [0.1, 0.0],
            [0.1, 0.1],
        ]
    )
    def test_get_lowest_m(self, p: float, r_tol: float) -> None:
        """Test user-facing automatic duration method."""
        for n in range(1, 100):
            lowest_m = self.interval_detector._get_lowest_m(p, n, r_tol=r_tol)
            assert lowest_m.p <= p * (1 + r_tol)
            assert lowest_m.m > 0

    def test_automatic_duration(self) -> None:
        """E2E test of the automatic duration parameter."""
        df = self.df.copy()
        df.value_b.iloc[10] = 1.0
        df.value_b.iloc[20:22] = 1.0
        df.value_b.iloc[30:33] = 1.0
        # Set a detector without a duration
        detector = ABDetectorModel(alpha=0.05)
        anomaly_response = detector.fit_predict(TimeSeriesData(df))
        assert anomaly_response.predicted_ts is not None
        assert anomaly_response.stat_sig_ts is not None
        _predicted_ds: TimeSeriesData = anomaly_response.predicted_ts
        assert detector.duration == 3
        assert np.isclose(detector.corrected_alpha, 0.006872806179441522)
        assert ~_predicted_ds.value.iloc[10].all()
        assert ~_predicted_ds.value.iloc[20:22].all()
        assert _predicted_ds.value.iloc[30:33].all()

    def test_plot(self) -> None:
        df = self.df.copy()
        detector = ABDetectorModel(alpha=0.05)
        detector.fit_predict(TimeSeriesData(df))
        detector.plot()
