# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Conducts an AB test across two concurrent time series. This would be useful
when an experiment is ran between two versions that are logged over time, and there is an
interest when one metric is signficantly different from another.

This implementation supports the following key features:

    1. Rejection Intervals: Sequential rejections are consolidated into contiguous intervals.

    2. Multiple Hypothesis Testing: Multiple Hypothesis Testing occurs when a set of statistical
        inferences occur simultaneously. This is controlled by setting the `duration` parameter
        that only accepts Rejection Intervals of a certain length.

    3. Multiple Distributions: Normal, Binomial, and Poisson likelihoods are available.

    4. One sample and Two Sample tests. For Two Sample tests we support,
        tests of absolute differences (b - a) or relative differences (b / a).

    5. One sided (lower and upper) and two sided tests.

Typical usage example:

>>> timeseries = TimeSeriesData(...)
>>> # Any extension of IntervalDetectorModel
>>> detector = TwoSampleProportion()
>>> # Run detector
>>> ab_test_results = detector.fit_predict(data=timeseries)
>>> # Plot the results
>>> detector.plot()
"""

from __future__ import annotations

import itertools

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from kats.consts import IntervalAnomaly, TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse, ConfidenceBand
from matplotlib import pyplot as plt

from numpy.linalg import matrix_power
from scipy.linalg import toeplitz
from scipy.stats import beta, binom, mvn, norm
from scipy.stats._multivariate import _PSD, multivariate_normal_gen
from statsmodels.tsa.arima_process import ArmaProcess

DEFAULT_FIGSIZE = (10, 12)


class ListEnum(Enum):
    @classmethod
    def __list__(cls) -> List[Enum]:
        """Converts cls members to a list.

        References:
            - https://github.com/python/cpython/blob/3.11/Lib/enum.py#L775.
        """
        return [cls._member_map_[name] for name in cls._member_names_]


@unique
class OneSampleColumns(ListEnum):
    VALUE = "value"
    VARIANCE = "variance"
    SAMPLE_COUNT = "sample_count"
    EFFECT_SIZE = "effect_size"


@unique
class TwoSampleColumns(ListEnum):
    VALUE_A = "value_a"
    VALUE_B = "value_b"
    VARIANCE_A = "variance_a"
    VARIANCE_B = "variance_b"
    SAMPLE_COUNT_A = "sample_count_a"
    SAMPLE_COUNT_B = "sample_count_b"
    EFFECT_SIZE = "effect_size"


class Schema(ABC):
    enum_cls = ListEnum

    @property
    def columns(self) -> List[Enum]:
        return self.enum_cls.__list__()

    @property
    @abstractmethod
    def non_negative_columns(self) -> List[Enum]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def positive_columns(self) -> List[Enum]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def integer_columns(self) -> List[Enum]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def proportion_columns(self) -> List[Enum]:
        raise NotImplementedError()

    def _validate_data(self, df: pd.core.frame.DataFrame) -> None:
        """Validates the data according to a schema.

        Notes:
            Columns must be non-nullable.
        """
        self._validate_names(df)
        self._validate_nullable(df)
        self._validate_postitive(df, self.positive_columns)
        self._validate_non_negative(df, self.non_negative_columns)
        self._validate_count(df, self.integer_columns)

    def _validate_names(self, df: pd.core.frame.DataFrame) -> None:
        for columns in self.columns:
            if columns.value not in df.columns:
                raise ValueError(f"{columns.value} must be provided")

    def _validate_nullable(self, df: pd.core.frame.DataFrame) -> None:
        if df.isnull().values.any():
            raise ValueError(
                "All entries must be specified but na's were found in data.value."
            )

    def _validate_postitive(
        self, df: pd.core.frame.DataFrame, columns: List[Enum]
    ) -> None:
        for column in columns:
            if df[column.value].le(0.0).any():
                raise ValueError(
                    f"{column.value} must be > 0 for each index. Found: \n {df[column.value]}."
                )

    def _validate_non_negative(
        self, df: pd.core.frame.DataFrame, columns: List[Enum]
    ) -> None:
        for column in columns:
            if df[column.value].lt(0.0).any():
                raise ValueError(
                    f"{column.value} must be >= 0 for each index. Found: \n {df[column.value]}."
                )

    def _validate_count(self, df: pd.core.frame.DataFrame, columns: List[Enum]) -> None:
        for column in columns:
            if df[column.value].dtype != np.dtype("int64"):
                raise ValueError(
                    f"{column.value} must be of type int64 for each index. Found: \n {df[column.value].dtype}."
                )

    def _validate_proportion(
        self, df: pd.core.frame.DataFrame, columns: List[Enum]
    ) -> None:
        for column in columns:
            if df[column.value].lt(0.0).any() or df[column.value].gt(1.0).any():
                raise ValueError(
                    f"{column.value} must be >= 0.0 and <= 1.0 for each index. Found: \n {df[column.value]}."
                )


class TwoSampleSchema(Schema):
    enum_cls = TwoSampleColumns

    @property
    def non_negative_columns(self) -> List[Enum]:
        return [
            self.enum_cls.VARIANCE_A,
            self.enum_cls.VARIANCE_B,
        ]

    @property
    def positive_columns(self) -> List[Enum]:
        return [
            self.enum_cls.SAMPLE_COUNT_A,
            self.enum_cls.SAMPLE_COUNT_B,
        ]

    @property
    def integer_columns(self) -> List[Enum]:
        return [
            self.enum_cls.SAMPLE_COUNT_A,
            self.enum_cls.SAMPLE_COUNT_B,
        ]

    @property
    def proportion_columns(self) -> List[Enum]:
        return [
            self.enum_cls.VALUE_A,
            self.enum_cls.VALUE_B,
        ]


class OneSampleSchema(Schema):
    enum_cls = OneSampleColumns

    @property
    def non_negative_columns(self) -> List[Enum]:
        return [
            self.enum_cls.VARIANCE,
        ]

    @property
    def positive_columns(self) -> List[Enum]:
        return [
            self.enum_cls.SAMPLE_COUNT,
        ]

    @property
    def integer_columns(self) -> List[Enum]:
        return [
            self.enum_cls.SAMPLE_COUNT,
        ]

    @property
    def proportion_columns(self) -> List[Enum]:
        return [
            self.enum_cls.VALUE,
        ]


@unique
class ABIntervalType(Enum):
    # Camel case since these are used as ABInterval class names
    FAIL_TO_REJECT = "FailToReject"
    REJECT = "Reject"


@unique
class TestStatistic(Enum):
    ABSOLUTE_DIFFERENCE = "absolute_difference"
    RELATIVE_DIFFERENCE = "relative_difference"


@unique
class TestType(Enum):
    ONE_SIDED_LOWER = "one_sided_lower"
    ONE_SIDED_UPPER = "one_sided_upper"
    TWO_SIDED = "two_sided"


@dataclass
class ABTestResult:
    test_statistic: pd.Series
    stat_sig: pd.Series
    upper: pd.Series
    lower: pd.Series


@dataclass
class CriticalValue:
    lower: Optional[pd.Series]
    upper: Optional[pd.Series]

    @property
    def absolute_critical_value(self) -> pd.Series:
        if self.lower is not None:
            return np.absolute(self.lower)
        elif self.upper is not None:
            return np.absolute(self.upper)
        else:
            raise ValueError(
                "Expecting either lower or upper to be specified. Found None."
            )

    @property
    def non_nullable_lower(self) -> pd.Series:
        assert self.lower is not None
        return self.lower

    @property
    def non_nullable_upper(self) -> pd.Series:
        assert self.upper is not None
        return self.upper


class ABInterval(IntervalAnomaly):
    """Extension of IntervalAnomaly that stores IntervalDetectorModel metadata.

    Used internally to consolidate sequential predictions of the IntervalDetectorModel
    before returning an AnomalyResponse to the user.
    """

    def __init__(
        self,
        interval_type: ABIntervalType,
        # pyre-fixme[11]: Annotation `Timestamp` is not defined as a type.
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        super().__init__(start=start, end=end)
        self.interval_type: ABIntervalType = interval_type
        self.__start_idx: Optional[int] = None
        self.__end_idx: Optional[int] = None

    @property
    def start_idx(self) -> Optional[int]:
        """Returns the start index wrt to the original TimeSeriesData."""
        return self.__start_idx

    @start_idx.setter
    def start_idx(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise ValueError(
                f"Expecting start_idx to be of type int. Found {type(idx)}"
            )
        self.__start_idx = idx

    @property
    def end_idx(self) -> Optional[int]:
        """Returns the end index wrt to the original TimeSeriesData."""
        return self.__end_idx

    @end_idx.setter
    def end_idx(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise ValueError(f"Expecting end_idx to be of type int. Found {type(idx)}")
        self.__end_idx = idx

    @property
    def indices(self) -> List[int]:
        """Returns the (inclusive) indices of the interval wrt to the original TimeSeriesData."""
        assert self.start_idx is not None
        assert self.end_idx is not None
        return list(range(self.start_idx, self.end_idx + 1))

    def __repr__(self) -> str:
        _indent = "\n   "
        _repr = f"{self.interval_type.value}Interval("
        _repr += _indent + f"start={self.start},"
        _repr += _indent + f"end={self.end},"
        if self.start_idx is not None and self.end_idx is not None:
            _repr += _indent + f"length={len(self.indices)},"
        _repr += "\n)"
        return _repr


def arma_p_q(ar: List[float], ma: List[float], n: int) -> npt.NDArray:
    """Returns the autocorrelation matrix of an ARMA(p, q) Process."""
    # (1 - \phi_1 * L - ... - \phi_p * L^p) * y_t =
    #       (1 + \theta_1 * L + ... + \theta_q * L^q) * e_t
    return toeplitz(
        ArmaProcess(ar=np.r_[1, -np.array(ar)], ma=np.r_[1, np.array(ma)]).acf(lags=n)
    )


def ar_1(rho: float, n: int) -> npt.NDArray:
    """Returns the autocorrelation matrix of an AR(1) Process."""
    return arma_p_q(ar=[rho], ma=[], n=n)


class IntervalDetectorModel(DetectorModel, ABC):
    """Abstract Base Class for conducting statistical tests on time series data.

    IntervalDetectorModel runs a hypothesis test for each time index.
    Multiple Hypothesis Testing is then mitigated by (optionally) applying a duration
    parameter that consolidates sequential predictions into contiguous intervals.

    Notes:
        - If a duration parameter is not specified, it will be calculated from
        the length of the time series and the overall requested Type I error (alpha).
        - If a duration parameter is specified, just use the supplied duration and alpha
        that is assigned.

    Properties:
        alpha: Overall Type-I error of statistical test. Between 0 and 1 (inclusive).
        corrected_alpha: Corrected Type-I error of the statistical test taking the `duration`
            parameter into consideration. Between 0 and 1 (inclusive).
        duration: length of consecutive predictions considered to be significant.
        test_type: The type of test. One of the following:
            - For testing b - a < 0 or b / a < 1, use TestType.ONE_SIDED_LOWER.
            - For testing b - a > 0 or b / a > 1, use TestType.ONE_SIDED_UPPER.
            - For testing b - a ≠ 0 or b / a ≠ 1, use TestType.TWO_SIDED.
        anomaly_intervals: A list of rejection intervals that meet or exceed the minimal `duration`.
        caution_intervals: Similar to `anomaly_intervals`, but don't meet the minimal `duration`.

    Attributes:
        data: Time series data passed to fit_predict. Must adhere to self.schema.
        critical_value: Critical value used to convert test_statistic(s) to decisions.
        test_result: Results of the AB test.
        fail_to_reject_intervals: List of intervals describing where the test has accepted the null hypothesis.
        reject_intervals: List of intervals describing where the test has rejected the null hypothesis.
    """

    _alpha: Optional[float] = None
    _corrected_alpha: Optional[float] = None
    _duration: Optional[int] = None
    _test_statistic: Optional[TestStatistic] = None
    _test_type: Optional[TestType] = None
    data: Optional[TimeSeriesData] = None
    critical_value: CriticalValue = CriticalValue(lower=None, upper=None)
    test_result: Optional[ABTestResult] = None
    fail_to_reject_intervals: Optional[List[ABInterval]] = None
    reject_intervals: Optional[List[ABInterval]] = None

    def __init__(
        self,
        alpha: Optional[float] = 0.05,
        duration: Optional[int] = None,
        serialized_model: Optional[bytes] = None,
        test_type: Optional[TestType] = TestType.ONE_SIDED_UPPER,
    ) -> None:
        if serialized_model:
            model_dict = json.loads(serialized_model)
            self.alpha = model_dict["alpha"]
            self.duration = model_dict["duration"]
            # deserialize value
            self.test_type = TestType(model_dict["test_type"])
        else:
            self.alpha = alpha
            self.duration = duration
            self.test_type = test_type

    def __repr__(self) -> str:
        _indent = "\n   "
        _repr = self.__class__.__name__
        _repr += "("
        for key, value in self.json.items():
            _repr += _indent + f"{key}={value}"
        _repr += "\n)"
        return _repr

    @property
    def alpha(self) -> float:
        if self._alpha is None:
            raise ValueError("alpha is not initialized.")
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: Optional[float]) -> None:
        if alpha is None:
            raise ValueError(f"alpha must be specified. Found {alpha}.")
        # alpha must be specified and between 0 and 1.
        elif alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must be between 0 and 1 (inclusive). Found {alpha}"
            )
        self._alpha = alpha

    @property
    def corrected_alpha(self) -> float:
        if self._corrected_alpha is None:
            raise ValueError("corrected_alpha is not initialized.")
        return self._corrected_alpha

    @corrected_alpha.setter
    def corrected_alpha(self, alpha: Optional[float]) -> None:
        if alpha is None:
            raise ValueError(f"corrected_alpha must be specified. Found {alpha}.")
        # alpha must be specified and between 0 and 1.
        elif alpha < 0 or alpha > 1:
            raise ValueError(
                f"corrected_alpha must be between 0 and 1 (inclusive). Found {alpha}"
            )
        self._corrected_alpha = alpha

    @property
    def test_type(self) -> TestType:
        if self._test_type is None:
            raise ValueError("test_type is not initialized.")
        return self._test_type

    @test_type.setter
    def test_type(self, test_type: Optional[TestType]) -> None:
        if test_type is None:
            raise ValueError(f"test_type must be specified. Found {test_type}.")
        elif not isinstance(test_type, TestType):
            raise TypeError(
                f"test_type must be of type TestType. Found {type(test_type)}."
            )
        self._test_type = test_type

    @property
    def duration(self) -> Optional[int]:
        return self._duration

    @duration.setter
    def duration(self, duration: Optional[int]) -> None:
        # If duration is None, IntervalDetectorModel will assign a value.
        # Otherwise, it needs to be a positive value.
        if duration is not None and duration <= 0:
            raise ValueError(f"duration must be > 0. Found {duration}.")
        self._duration = duration

    @property
    def anomaly_intervals(self) -> List[ABInterval]:
        _reject_intervals, _duration = self._get_rejection_intervals_and_duration()
        return [
            interval
            for interval in _reject_intervals
            if len(interval.indices) >= _duration
        ]

    @property
    def caution_intervals(self) -> List[ABInterval]:
        _reject_intervals, _duration = self._get_rejection_intervals_and_duration()
        return [
            interval
            for interval in _reject_intervals
            if len(interval.indices) < _duration
        ]

    def _get_rejection_intervals_and_duration(self) -> Tuple[List[ABInterval], int]:
        """Retrieve rejection intervals and minimal duration for post-processing."""
        # reject_intervals and duration must not be None at this point
        if self.reject_intervals is None:
            raise ValueError("reject_intervals are required for anomaly_intervals.")
        _reject_intervals: List[ABInterval] = self.reject_intervals

        if self.duration is None:
            raise ValueError("duration is required for anomaly_intervals.")
        _duration: int = self.duration
        return _reject_intervals, _duration

    @property
    @abstractmethod
    def schema(self) -> Schema:
        """Column Schema for the `fit_predict()` method."""
        raise NotImplementedError

    @property
    def json(self) -> Dict[str, str]:
        return {
            **{
                "alpha": self.alpha,
                "duration": self.duration,
                # serialize value
                "test_type": self.test_type.value,
            },
            **self._json,
        }

    @property
    def _json(self) -> Dict[str, str]:
        """Hook for extensions to add attributes to serialization."""
        return {}

    def serialize(self) -> bytes:
        """Serializes a model into a json representation."""
        return json.dumps(self.json).encode("utf-8")

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        consolidate_into_intervals: bool = True,
        interval_padding: int = 30,
        interval_units: str = "m",
        r_tol: float = 0.1,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """Fit and predict on a Interval based AB test on time series data.

        Notes:
            - All entries of data and historical_data must be specified (no na values).

        Args:
            data: Time series containing columns specified in `self.schema`.
            historical_data: Data that will be prepended to `data`.
                Used in an online setting when data is observed data
                in addition to the previous historic observations.
            consolidate_into_intervals: Whether to consolidate sequential predictions
                into contiguous intervals.
            interval_padding: For the case of two sequential anomalies, a ABInterval is well defined.
                However, for a single non-adjacent anomaly, we need to pad the time index to form
                a ABInterval. interval_padding decides how much time to use in this padding.
            interval_units: The units of interval_padding. Includes:
                - D (day)
                - h (hour)
                - m (minute)
                - s (second)
                For less common units and a complete listing, see:
                https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
            r_tol: Relative tolerance used for automatic assignment to duration property.
                If duration is `None`, then a value is automatically assigned such that
                alpha is corrected to be no greater than alpha * (1 + r_tol).

        Returns:
            The results of the Interval based AB test. Including:
                - scores: Raw test statistic.
                - predicted_ts: Boolean array of predictions that are formed from contiguous intervals.
                - stat_sig: Statistical significance of `scores`.
                - upper: Upper limit in the (1 - alpha) confidence interval.
                - lower: Lower limit in the (1 - alpha) confidence interval.
        """
        self.data = None
        self.schema._validate_data(pd.DataFrame(data.value))
        if historical_data is None:
            self.data = data
        else:
            historical_data.extend(data, validate=False)
            self.data = historical_data
        assert self.data is not None
        _data: TimeSeriesData = self.data
        n = len(_data)
        len_data_zeros = pd.Series(np.zeros(n))

        # Step 1: Determine critical_value of test_statistic.
        self.critical_value = self._get_critical_value(length=n, r_tol=r_tol)

        # Step 2: Get a test_statistic for each time value.
        self.test_result = self.get_test_statistic(pd.DataFrame(_data.value))

        # Step 3 (Optional): Consolidate sequential predictions into contiguous intervals.
        #   At this point, we have a computed test statistics for each time index.
        #   To convert these to actual decisions we need to apply:
        #       (1) critical_value
        #       (2) duration
        #   If the user wishes to do this manually, this step can also be bypassed,
        #   returning just the raw test statistics without any critical_value or duration applied.
        if consolidate_into_intervals:
            self.reject_intervals = self._get_intervals(
                time=_data.time,
                interval_type=ABIntervalType.REJECT,
                interval_padding=interval_padding,
                interval_units=interval_units,
            )
            self.fail_to_reject_intervals = self._get_intervals(
                time=_data.time,
                interval_type=ABIntervalType.FAIL_TO_REJECT,
                interval_padding=interval_padding,
                interval_units=interval_units,
            )

        # Step 4: Return to user.
        if self.reject_intervals is not None:
            # If we consolidate_into_intervals, then apply to predicted_ts
            _reject_intervals: List[ABInterval] = self.reject_intervals
            _predicted_ts = self._convert_intervals_to_predictions(
                time=_data.time, intervals=_reject_intervals
            )
        else:
            # Otherwise, return a placeholder
            _predicted_ts = TimeSeriesData(time=_data.time, value=len_data_zeros)

        assert self.test_result is not None
        _stat_sig: pd.Series = self.test_result.stat_sig
        _upper: pd.Series = self.test_result.upper
        _lower: pd.Series = self.test_result.lower
        return AnomalyResponse(
            scores=TimeSeriesData(
                time=_data.time, value=self.test_result.test_statistic
            ),
            confidence_band=ConfidenceBand(
                upper=TimeSeriesData(
                    time=_data.time,
                    value=_upper,
                ),
                lower=TimeSeriesData(
                    time=_data.time,
                    value=_lower,
                ),
            ),
            predicted_ts=_predicted_ts,
            anomaly_magnitude_ts=TimeSeriesData(time=_data.time, value=len_data_zeros),
            stat_sig_ts=TimeSeriesData(time=_data.time, value=_stat_sig),
        )

    def _get_critical_value(self, length: int, r_tol: float) -> CriticalValue:
        """Determine a critical value for a statistical test.

        Notes:
            If the user has passed a duration, then they override the automatic assignment.
            In this case, the critical value is directly applied to alpha without consideration of `length`.
            Otherwise, we treat self.alpha initially as the global type-I error, and correct this based
            off `length`, producing an automated assignment for duration and a corrected type-I error.

        Args:
            length: Length of the time series that is being tested.
            r_tol: See `fit_predict()`.

        Returns:
            Critical value used for determing whether to reject or fail to reject the null hypothesis.
        """
        if self.duration is None:
            # Use self.alpha and length to determine best duration
            # and also adjust alpha for the critical_value calculation.
            lowest_m = self._get_lowest_m(
                p=self.alpha, n=length, test_type=self.test_type, r_tol=r_tol
            )
            logging.warning(
                f"Automatic duration with {length} data points:"
                + f"\nduration set to {lowest_m.m}"
                + f"\ncorrected_alpha set to {lowest_m.p}"
            )
            self.duration = lowest_m.m
            self.corrected_alpha = lowest_m.p
        else:
            # With duration and length fixed, determine an adjusted Type-I error
            # such that the global Type-I error still remains within a
            # relative threshold of self.alpha.
            lowest_p = self._get_lowest_p(
                m=self.duration,
                n=length,
                p_goal=self.alpha,
                test_type=self.test_type,
                r_tol=r_tol,
            )
            logging.warning(
                f"Type-I Adjustment with {length} data points:"
                + f"\nduration set to {self.duration}"
                + f"\ncorrected_alpha set to {lowest_p.p_corrected}"
                + f"\nType-I Error adjusted to {lowest_p.p_global} from {self.alpha}"
            )
            self.corrected_alpha = lowest_p.p_corrected
        if self.test_type == TestType.ONE_SIDED_LOWER:
            return CriticalValue(
                lower=self._convert_alpha_to_critical_value(
                    self.corrected_alpha, length
                ),
                upper=None,
            )
        elif self.test_type == TestType.ONE_SIDED_UPPER:
            return CriticalValue(
                lower=None,
                upper=self._convert_alpha_to_critical_value(
                    1.0 - self.corrected_alpha, length
                ),
            )
        elif self.test_type == TestType.TWO_SIDED:
            return CriticalValue(
                lower=self._convert_alpha_to_critical_value(
                    self.corrected_alpha / 2, length
                ),
                upper=self._convert_alpha_to_critical_value(
                    1.0 - self.corrected_alpha / 2, length
                ),
            )
        else:
            raise ValueError(
                f"Expected test_type to be of TestType. Found {self.test_type}"
            )

    @abstractmethod
    def _convert_alpha_to_critical_value(self, alpha: float, length: int) -> pd.Series:
        raise NotImplementedError()

    def get_test_statistic(self, df: pd.core.frame.DataFrame) -> ABTestResult:
        self._get_test_statistic_hook(df)
        return self._get_test_statistic(df)

    def _get_test_statistic_hook(self, df: pd.core.frame.DataFrame) -> None:
        pass

    @abstractmethod
    def _get_test_statistic(self, df: pd.core.frame.DataFrame) -> ABTestResult:
        raise NotImplementedError()

    def _convert_intervals_to_predictions(
        self, time: pd.Series, intervals: List[ABInterval]
    ) -> TimeSeriesData:
        # Initialize with all non-predictions
        values = [False] * len(time)
        # At this point, we need to have a duration specified.
        assert self.duration is not None
        _duration: int = self.duration
        # Replace all values for the indices of the ABIntervals
        # if the interval is at least as large as the duration parameter.
        for interval in intervals:
            if len(interval.indices) >= _duration:
                for idx in interval.indices:
                    values[idx] = True
        return TimeSeriesData(time=time, value=pd.Series(values))

    @dataclass
    class LowestM:
        m: int
        p: float

    @staticmethod
    def _get_lowest_m(
        p: float, n: int, r_tol: float, test_type: TestType, max_iter: int = 1000
    ) -> LowestM:
        """Finds lowest m such that the corrected probability is still less than p in n trials.

        Notes:
            This method only bounds the resulting type-I error, it does not gurantee that the
            corrected type-I error is preserved.

        Args:
            p: Overall Type-I error of an experiment.
            n: Total number of trials.
            r_tol: Relative tolerance applied to p.
                i.e. p = 0.5, r_tol = 0.2, p * (r_tol + 1) = 0.6
            max_iter: Total number of iterations before stopping.

        Returns:
            Lowest m and the corrected Type-I error.
        """
        p_global = p
        m = 1
        while m < max_iter and p >= p_global and m <= n:
            # Correct p based off current iter and p_global.
            p = IntervalDetectorModel._probability_of_at_least_one_m_run_in_n_trials(
                p_global, n=n, m=m, test_type=test_type
            )
            if p <= p_global * (r_tol + 1):
                # We have found a solution meeting r_tol.
                return IntervalDetectorModel.LowestM(m=m, p=p)
            # Otherwise, try the next m.
            m += 1
        raise Exception(
            "Automatic duration did not converge. Please explicitly pass duration or revise alpha."
        )

    @dataclass
    class LowestP:
        p_corrected: float
        p_global: float

    @staticmethod
    def _get_lowest_p(
        m: int,
        n: int,
        p_goal: float,
        r_tol: float,
        test_type: TestType,
        max_iter: int = 1000,
    ) -> LowestP:
        """Finds a p so that the corrected p is with `r_tol` of `p_global` with n trials and m run size.

        Notes:
            A binary search is performed to find an approximate solution. In cases where this does not
            converge, raising either `max_iter` or `r_tol` will prevent an exception being raised.

        Args:
            m: number of consecutive 1's.
            n: total number of trials.
            p_goal: desired probability of seeing at least one run of m in n trials.
            r_tol: Relative tolerance applied to p_goal, >=1e-9.
                    i.e. p = 0.5, r_tol = 0.2, p * (1 - r_tol) = 0.4 & p * (r_tol + 1) = 0.6
            max_iter: Total number of iterations before stopping.

        Returns:
            A corrected p such that the global p is still within a relative tolerance of `p_goal`.
        """
        if m > n:
            raise ValueError(f"m must be <= n. Found n={n} and m={m}.")

        # p_goal = p ** n
        if n == m:
            return IntervalDetectorModel.LowestP(
                p_corrected=p_goal ** (1 / m), p_global=p_goal
            )

        if r_tol < 1e-9:
            raise ValueError(
                f"r_tol=1e-9 is the smallest supported value, found: {r_tol}"
            )

        i: int = 0
        # p_goal = U_{i=1}^{n-m} P(run size of m starting at position i)
        #   ≤ ∑_{i=1}^{n-m} P(run size of m starting at position i)
        #   ≤ p ** m * n
        p_low: float = (p_goal / n) ** (1 / m)
        # p_goal ≥ Binomial(k > 0; n//m, p ** m)
        #   = 1 - Binomial(k = 0; n//m, p ** m)
        #   = 1 - (1 - p ** m) ** (n // m)
        p_high: float = (1 - (1 - p_goal) ** (1 / (n // m))) ** (1 / m)
        while p_low <= p_high and i <= max_iter:
            p_corrected = (p_high + p_low) / 2.0
            p_global = (
                IntervalDetectorModel._probability_of_at_least_one_m_run_in_n_trials(
                    p_corrected, n=n, m=m, test_type=test_type
                )
            )
            # Return if the corrected Type-I error is within our relative tolerance.
            if p_global <= p_goal * (r_tol + 1) and p_global >= p_goal * (1 - r_tol):
                return IntervalDetectorModel.LowestP(
                    p_corrected=p_corrected, p_global=p_global
                )
            # Otherwise search higher
            elif p_global < p_goal:
                p_low = p_corrected
            # Otherwise search lower
            elif p_global > p_goal:
                p_high = p_corrected
            i += 1
        raise Exception(
            f"max_iter={max_iter} exceeded while adjusting a goal of {p_goal} with r_tol={r_tol}. Raise max_iter or r_tol."
        )

    @staticmethod
    def _mvn_mvnun(
        lower: npt.NDArray,
        upper: npt.NDArray,
        mean: Optional[np.ndarray] = None,
        cov: Union[int, np.ndarray] = 1,
        allow_singular: bool = False,
        maxpts: Optional[int] = None,
        abseps: float = 1e-6,
        releps: float = 1e-6,
    ) -> float:
        """Wrapper on scipy mvnun function enabling definite integrals.

        References:
            https://github.com/scipy/scipy/blob/main/scipy/stats/mvndst.f

        Args:
            lower: Lower limit of integration
            upper: Upper limit of integration
            mean: (N,) dimensional mean vector of MVN distribution.
            cov: (N, N) dimensional covariance matrix of MVN distribution.
            maxpts: The maximum number of points to use for integration
            abseps: Absolute error tolerance
            releps: Relative error tolerance
        """
        # Follow preprocessing from:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
        _multivariate_normal_gen = multivariate_normal_gen()
        # pyre-ignore
        dim, mean, cov = _multivariate_normal_gen._process_parameters(None, mean, cov)
        # pyre-ignore
        lower = _multivariate_normal_gen._process_quantiles(lower, dim)
        upper = _multivariate_normal_gen._process_quantiles(upper, dim)
        _PSD(cov, allow_singular=allow_singular)
        return mvn.mvnun(
            lower=lower,
            upper=upper,
            means=mean,
            covar=cov,
            maxpts=1_000_000 * dim if not maxpts else maxpts,
            abseps=abseps,
            releps=releps,
        )[0]

    @staticmethod
    def _w_independent(m: int, p: float) -> npt.NDArray:
        return np.power(p, np.arange(m + 1)) * np.array([(1 - p)] * m + [1])

    @staticmethod
    def _w_one_tailed(
        m: int, p: float, test_type: TestType, cov: npt.NDArray
    ) -> npt.NDArray:
        z_crit = norm().ppf(1.0 - p)
        if test_type == TestType.ONE_SIDED_UPPER:
            fail_to_reject_lower = np.array([z_crit] * (m - 1) + [-np.inf])
            fail_to_reject_upper = np.array([np.inf] * (m - 1) + [z_crit])
            reject_lower = np.full(m, z_crit)
            reject_upper = np.full(m, np.inf)
        elif test_type == TestType.ONE_SIDED_LOWER:
            fail_to_reject_lower = np.array([-np.inf] * (m - 1) + [-z_crit])
            fail_to_reject_upper = np.array([-z_crit] * (m - 1) + [np.inf])
            reject_lower = np.full(m, -np.inf)
            reject_upper = np.full(m, -z_crit)
        else:
            raise ValueError(f"Expected test_type to be of TestType. Found {test_type}")
        res = [
            IntervalDetectorModel._mvn_mvnun(
                lower=fail_to_reject_lower[k:],
                upper=fail_to_reject_upper[k:],
                cov=cov[k:, k:],
            )
            for k in range(m - 1, -1, -1)
        ]
        res += [
            IntervalDetectorModel._mvn_mvnun(
                lower=reject_lower, upper=reject_upper, cov=cov
            )
        ]
        return np.array(res)

    @staticmethod
    def _w_two_tailed(m: int, p: float, cov: npt.NDArray) -> npt.NDArray:
        _m_warn = 8
        if m > _m_warn:
            logging.warning(
                f"Non-spherical covariance is unstable for m > {_m_warn}"
                + f" and requires {2 ** m} simulations / evaluation."
            )
        z_crit = norm().ppf(1.0 - p / 2)
        # Integration region is an AND conjunction of several OR clauses.
        # Instead of pre-specifying the integration regions such as in
        # _w_one_tailed, we compute these on the fly.
        res = [
            sum(
                [
                    IntervalDetectorModel._mvn_mvnun(
                        lower=np.array(
                            [-np.inf if r == "l" else z_crit for r in regions]
                            + [-z_crit]
                        ),
                        upper=np.array(
                            [-z_crit if r == "l" else np.inf for r in regions]
                            + [z_crit]
                        ),
                        cov=cov[k:, k:],
                    )
                    for regions in itertools.product(*["lh"], repeat=m - 1 - k)
                ]
            )
            for k in range(m - 1, -1, -1)
        ]
        res += [
            sum(
                [
                    IntervalDetectorModel._mvn_mvnun(
                        lower=np.array(
                            [-np.inf if r == "l" else z_crit for r in regions]
                        ),
                        upper=np.array(
                            [-z_crit if r == "l" else np.inf for r in regions]
                        ),
                        cov=cov,
                    )
                    for regions in itertools.product(*["lh"], repeat=m)
                ]
            )
        ]
        return np.array(res)

    @staticmethod
    def _w(
        m: int, p: float, test_type: TestType, cov: Optional[np.ndarray] = None
    ) -> npt.NDArray:
        if cov is None:
            return IntervalDetectorModel._w_independent(m=m, p=p)
        elif TestType(test_type.value) == TestType.TWO_SIDED:
            return IntervalDetectorModel._w_two_tailed(m=m, p=p, cov=cov)
        else:
            return IntervalDetectorModel._w_one_tailed(
                m=m, p=p, test_type=TestType(test_type.value), cov=cov
            )

    @staticmethod
    def _probability_of_at_least_one_m_run_in_n_trials(
        p: float,
        n: int,
        m: int,
        test_type: TestType = TestType.ONE_SIDED_UPPER,
        cov: Optional[np.ndarray] = None,
    ) -> float:
        """P(at least 1 run of m consecutive rejections) in a vectorized formulation.

        Notes:
            - Passing cov=None will default to using an independence assumption
            in _vec_solve. Otherwise, _mvn_mvnun is used to estimate weights.
            - TestType.TWO_SIDED is significantly more intensive to compute for
            non-spherical covariances. It is not recommended to use above m=8
            in this setting.

        Args:
            p: P(reject H_i) ∀ i
            n: total number of tests.
            m: required number of consecutive rejections.
            test_type: Tail(s) of the test that rejects.
            cov: Covariance matrix of P(reject H_i ∩ reject H_j) for i ≠ j.

        Returns:
            P(at least 1 run of m consecutive rejections).
        """

        def _check_args(n: int, m: int) -> None:
            if m <= 0:
                raise ValueError(f"m must be > 0. Found m={m}.")
            if n <= 0:
                raise ValueError(f"n must be > 0. Found n={n}.")
            if m > n:
                raise ValueError(f"m must be <= n. Found n={n} and m={m}.")

        def _A(m: int, w: npt.NDArray) -> npt.NDArray:
            A = np.diag(v=[1.0] * m, k=1)
            A[:, 0] = w
            A[-2, -1] = 0.0
            A[-1, -1] = 1.0
            return A

        def _vec_solve(w: npt.NDArray, n: int, m: int) -> npt.NDArray:
            r = np.array([0] * m + [1])
            A = _A(m=m, w=w)
            return r @ np.linalg.matrix_power(A, n - m + 1)

        # By default, return where i=n from the full state space.
        _check_args(n=n, m=m)
        w = IntervalDetectorModel._w(m=m, p=p, test_type=test_type, cov=cov)
        return _vec_solve(w=w, n=n, m=m)[0]

    def _get_intervals(
        self,
        time: pd.Series,
        interval_type: ABIntervalType,
        interval_padding: int,
        interval_units: str,
    ) -> List[ABInterval]:
        """Consolidates test_result's sequential predictions into contiguous intervals.

        Args:
            time: Time column of the original dataframe.
            interval_type: Name of the interval. One of "accept" or "reject".
            interval_padding: See `fit_predict` for a description.
            interval_units: See `fit_predict` for a description.

        Returns:
            Contiguous intervals made from the sequential predictions.
        """
        _mask = self._get_test_decision(interval_type)
        starts, ends = self._get_true_run_indices(_mask)
        intervals: List[ABInterval] = []

        # Loop thru consecutive runs and create Interval objects.
        for start, end in zip(starts, ends):
            start_time = time[start]
            end_time = time[end]
            # Pad the intervals so that single points will have a notion of length.
            start_time -= np.timedelta64(interval_padding, interval_units)
            end_time += np.timedelta64(interval_padding, interval_units)
            interval = ABInterval(
                interval_type=interval_type, start=start_time, end=end_time
            )
            # Set attributes of the Interval object. These correspond to the indices
            # of the original TimeSeriesData passed in `fit_predict`.
            interval.start_idx = int(start)
            interval.end_idx = int(end)
            intervals.append(interval)
        return intervals

    def _get_test_decision(self, interval_type: ABIntervalType) -> npt.NDArray:
        """Converts a critical value and test statistic into a boolean decision.

        Args:
            interval_type: The type of interval for a given decision.

        Returns:
            A one-dimensional boolean array of test decisions.
        """
        if self.test_result is None:
            raise ValueError("test result is None. Call fit_predict() first")
        else:
            test_result: ABTestResult = self.test_result

        def _check_shapes(x: pd.Series) -> None:
            if test_result.test_statistic.shape != x.shape:
                raise ValueError(
                    f"test_statistic and critical_value have mismatching shapes. "
                    f"Found {test_result.test_statistic.shape} and {x.shape}."
                )

        # Compute _mask for interval_type.REJECT case.
        if self.test_type == TestType.ONE_SIDED_LOWER:
            _lower: pd.Series = self.critical_value.non_nullable_lower
            _check_shapes(_lower)
            _mask: npt.NDArray = (
                test_result.test_statistic.to_numpy() <= _lower.to_numpy()
            )
        elif self.test_type == TestType.ONE_SIDED_UPPER:
            _upper: pd.Series = self.critical_value.non_nullable_upper
            _check_shapes(_upper)
            _mask: npt.NDArray = (
                test_result.test_statistic.to_numpy() >= _upper.to_numpy()
            )
        elif self.test_type == TestType.TWO_SIDED:
            _lower: pd.Series = self.critical_value.non_nullable_lower
            _upper: pd.Series = self.critical_value.non_nullable_upper
            _check_shapes(_lower)
            _check_shapes(_upper)
            _mask: npt.NDArray = np.logical_or(
                test_result.test_statistic.to_numpy() <= _lower.to_numpy(),
                test_result.test_statistic.to_numpy() >= _upper.to_numpy(),
            )
        else:
            raise ValueError(
                f"Expected test_type to be of TestType. Found {self.test_type}"
            )
        # Check the return object.
        if _mask.ndim != 1:
            raise ValueError(f"Expecting a 1D array. Found {_mask.ndim} dimensions.")
        elif _mask.shape[0] == 0:
            raise ValueError(
                f"Expecting an array with length > 0. Found {_mask.shape[0]}."
            )
        elif _mask.dtype != np.dtype("bool"):
            raise ValueError(
                f"x must have x.dtype == np.dtype('bool'). Found {_mask.dtype}."
            )
        # Invert _mask for interval_type.FAIL_TO_REJECT, otherwise just _mask.
        if interval_type == ABIntervalType.REJECT:
            return _mask
        if interval_type == ABIntervalType.FAIL_TO_REJECT:
            return np.invert(_mask)
        else:
            raise ValueError(
                f"Expected interval_type to be of ABIntervalType. Found {ABIntervalType}."
            )

    @staticmethod
    def _get_true_run_indices(x: npt.NDArray) -> Tuple[np.ndarray, np.ndarray]:
        """Helper function that finds consecutive runs of `True` values.

        Example:
            >>> run_indices = _get_true_run_indices([True, True, False, True])
            >>> print([i.tolist() for i in run_indices])
            >>> [[0, 3], [1, 3]]

        Args:
            x: numpy array of boolean values.

        Returns:
            Tuple of starting & ending index arrays for all consecutive `True` runs in `x`.
        """
        n = x.shape[0]
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        run_values = x[loc_run_start]
        run_lengths = np.diff(np.append(run_starts, n))
        return run_starts[run_values], (run_starts + run_lengths - 1)[run_values]

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """Runs `fit_predict`. See `fit_predict` for details."""
        return self.fit_predict(
            data,
            historical_data,
            **kwargs,
        )

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> None:
        """Not implemented for this class."""
        NotImplementedError("fit() not implemented. See fit_predict for usage.")

    def plot(
        self,
        figsize: Optional[Tuple[int, int]] = DEFAULT_FIGSIZE,
        interval_units: str = "m",
    ) -> Tuple[plt.Axes, plt.Axes]:
        """Plot the ABTestResult.

        Warnings:
            This method can only be ran after `fit_predict`.

        Notes:
            To be serializable, the x-axis is plotted in `interval_units` since the start time.

        Args:
            figsize: Figure size.
            interval_units: Units of the intervals. See `fit_predict` for more details.

        Returns:
            Plot of the values & their uncertainty and the test_statistic with overlayed intervals.
        """
        ALPHA = 0.3

        # Data validation
        if self.data is None:
            raise ValueError("Error: data is None. Call fit_predict() before plot()")
        data: Optional[TimeSeriesData] = self.data
        if self.test_result is None:
            raise ValueError(
                "Error: test_result is None. Call fit_predict() before plot()"
            )
        test_result: Optional[ABTestResult] = self.test_result
        assert self.fail_to_reject_intervals is not None
        fail_to_reject_intervals = self.fail_to_reject_intervals

        # Setup axes
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # X-axis
        x_axis = (data.time - data.time.min()) / np.timedelta64(1, interval_units)

        # Abstract plot for values and uncertainty.
        self._plot(x_axis=x_axis, data=data, axis=ax1, alpha=ALPHA)
        ax1.set_ylabel("Values")
        ax1.set_xlabel(f"Elapsed time ({interval_units}) from {data.time.min()}")
        ax1.legend()

        # Plot the test statistic and intervals
        ax2.plot(x_axis, test_result.test_statistic, label="test statistic")

        # Define two helper functions to extract information from a ABInterval.
        def get_grid(interval: ABInterval) -> pd.Series:
            """Helper function creating a grid of Timestamp.timestamp float values."""
            assert data is not None
            assert interval.end_idx is not None
            end_idx = interval.end_idx
            return (
                pd.concat(
                    [
                        pd.Series(interval.start),
                        pd.Series(data.time[interval.start_idx : end_idx + 1]),
                        pd.Series(interval.end),
                    ]
                )
                - data.time.min()
            ) / np.timedelta64(1, interval_units)

        def get_values(values: pd.Series, interval: ABInterval) -> pd.Series:
            assert interval.end_idx is not None
            end_idx = interval.end_idx
            # pyre-fixme[7]: Expected `Series` but got `Union[DataFrame, Series]`.
            return pd.concat(
                [
                    pd.Series(values[interval.start_idx]),
                    pd.Series(values[interval.start_idx : end_idx + 1]),
                    pd.Series(values[end_idx]),
                ]
            )

        if self.test_type == TestType.ONE_SIDED_LOWER:
            critical_value = self.critical_value.non_nullable_lower
        elif self.test_type == TestType.ONE_SIDED_UPPER:
            critical_value = self.critical_value.non_nullable_upper
        elif self.test_type == TestType.TWO_SIDED:
            # TODO: Figure out a better way to plot two sided test.
            return ax1, ax2
        else:
            raise ValueError(
                f"Expected test_type to be of TestType. Found {self.test_type}"
            )

        # Plot the critical value
        ax2.plot(x_axis, critical_value, label="reject", color="r", ls="--")

        # Plot the fail to reject intervals
        for i, interval in enumerate(fail_to_reject_intervals):
            ax2.fill_between(
                x=get_grid(interval),
                y1=get_values(critical_value, interval),
                y2=get_values(test_result.test_statistic, interval),
                alpha=ALPHA,
                color="green",
                label="Fail to Reject" if i == 0 else None,
            )

        # Plot the reject intervals less than duration
        for i, interval in enumerate(self.caution_intervals):
            ax2.fill_between(
                x=get_grid(interval),
                y1=get_values(critical_value, interval),
                y2=get_values(test_result.test_statistic, interval),
                alpha=ALPHA,
                color="yellow",
                label="Reject < Duration" if i == 0 else None,
            )

        # Plot the anomaly intervals more than or equal duration
        for i, interval in enumerate(self.anomaly_intervals):
            ax2.fill_between(
                x=get_grid(interval),
                y1=get_values(critical_value, interval),
                y2=get_values(test_result.test_statistic, interval),
                alpha=ALPHA,
                color="red",
                label="Reject >= Duration" if i == 0 else None,
            )
        ax2.set_ylabel("Test Statistic")
        ax2.set_xlabel(f"Elapsed time ({interval_units}) from {data.time.min()}")
        ax2.title.set_text(f"Test Statistic w/ duration: {self.duration}")
        ax2.legend()
        return ax1, ax2

    @abstractmethod
    def _plot(
        self, x_axis: pd.Series, data: TimeSeriesData, axis: plt.Axes, alpha: float
    ) -> None:
        """Method for plotting extension specific data values and uncertainty."""
        raise NotImplementedError()


class TwoSampleIntervalDetectorModel(IntervalDetectorModel, ABC):
    """Abstract Base Class that considers two samples at each time index.

    This class interprets `effect_size` as either the difference or ratio of
    the two samples. This is determined by the `test_statistic` property as
    described below.

    Properties:
        test_statistic: The type of test statistic to compute for each time index.
            - For value_b - value_a, use TestStatistic.ABSOLUTE_DIFFERENCE
            - For value_b / value_a, use TestStatistic.RELATIVE_DIFFERENCE
            Internally, TestStatistic.RELATIVE_DIFFERENCE is tested in log-space and the
            delta method is applied to compute the standard error.

    Notes:
        - This class relies on a normal approximation for the difference or
            ratio of two distributions.
        - In the case of TestStatistic.RELATIVE_DIFFERENCE, (1 + `effect_size`)
            will be used as the ratio to test value_b / value_a.
        - In the case of TestStatistic.RELATIVE_DIFFERENCE, lower and upper are
            reported on the relative risk scale (exponentiated from log-space).
    """

    def __init__(
        self,
        alpha: Optional[float] = 0.05,
        duration: Optional[int] = None,
        serialized_model: Optional[bytes] = None,
        test_statistic: Optional[TestStatistic] = TestStatistic.ABSOLUTE_DIFFERENCE,
        test_type: Optional[TestType] = TestType.ONE_SIDED_UPPER,
    ) -> None:
        super().__init__(
            alpha=alpha,
            duration=duration,
            serialized_model=serialized_model,
            test_type=test_type,
        )
        if serialized_model:
            model_dict = json.loads(serialized_model)
            self.test_statistic = TestStatistic(model_dict["test_statistic"])
        else:
            self.test_statistic = test_statistic

    @property
    def test_statistic(self) -> TestStatistic:
        if self._test_statistic is None:
            raise ValueError("test_statistic is not initialized.")
        return self._test_statistic

    @test_statistic.setter
    def test_statistic(self, test_statistic: Optional[TestStatistic]) -> None:
        if test_statistic is None:
            raise ValueError(
                f"test_statistic must be specified. Found {test_statistic}."
            )
        elif not isinstance(test_statistic, TestStatistic):
            raise TypeError(
                f"test_statistic must be of type TestStatistic. Found {type(test_statistic)}."
            )
        self._test_statistic = test_statistic

    @property
    def _json(self) -> Dict[str, str]:
        return {"test_statistic": self.test_statistic.value}

    @property
    def schema(self) -> Schema:
        return TwoSampleSchema()

    def _convert_alpha_to_critical_value(self, alpha: float, length: int) -> pd.Series:
        return pd.Series([norm.ppf(alpha)] * length)

    def _get_test_statistic(self, df: pd.core.frame.DataFrame) -> ABTestResult:
        if self.test_statistic == TestStatistic.ABSOLUTE_DIFFERENCE:
            _fn = self._absolute_difference_test_statistic
        elif self.test_statistic == TestStatistic.RELATIVE_DIFFERENCE:
            _fn = self._relative_difference_test_statistic
        else:
            raise ValueError(
                f"test_statistic was incorrectly specified. Found {self.test_statistic}"
            )
        return _fn(
            value_a=df.value_a,
            value_b=df.value_b,
            effect_size=df.effect_size,
            variance_a=df.variance_a,
            variance_b=df.variance_b,
            sample_count_a=df.sample_count_a,
            sample_count_b=df.sample_count_b,
        )

    def _absolute_difference_test_statistic(
        self,
        value_a: pd.Series,
        value_b: pd.Series,
        effect_size: pd.Series,
        variance_a: pd.Series,
        variance_b: pd.Series,
        sample_count_a: pd.Series,
        sample_count_b: pd.Series,
    ) -> ABTestResult:
        _variance_a, _variance_b = self._get_variance(
            value_a=value_a,
            value_b=value_b,
            effect_size=effect_size,
            variance_a=variance_a,
            variance_b=variance_b,
            sample_count_a=sample_count_a,
            sample_count_b=sample_count_b,
        )

        # Cache the variance for plotting.
        if self.data is not None:
            self.data.value.variance_a = _variance_a
            self.data.value.variance_b = _variance_b

        difference = value_b - value_a
        difference_mean = difference - effect_size
        difference_std_error = np.sqrt(_variance_a + _variance_b)
        test_statistic = pd.Series(difference_mean / difference_std_error)

        # Use symmetry of Normal distribution for a single critical value.
        critical_value: pd.Series = self.critical_value.absolute_critical_value

        # -z < (x - mu) / sigma < z
        # -z * sigma - x < -mu < z * sigma - x
        #    ONE_SIDED_UPPER        ONE_SIDED_LOWER
        # => x - z * sigma <= mu <= z * sigma + x
        upper = difference + critical_value * difference_std_error
        lower = difference - critical_value * difference_std_error
        if self.test_type == TestType.ONE_SIDED_LOWER:
            stat_sig = pd.Series(norm.cdf(test_statistic))
            lower = pd.Series([-np.inf] * len(difference))
        elif self.test_type == TestType.ONE_SIDED_UPPER:
            stat_sig = pd.Series(norm.sf(test_statistic))
            upper = pd.Series([np.inf] * len(difference))
        elif self.test_type == TestType.TWO_SIDED:
            stat_sig = pd.Series(
                np.minimum(norm.cdf(test_statistic), norm.sf(test_statistic))
            )
        else:
            raise ValueError(
                f"Expected test_type to be of TestType. Found {self.test_type}"
            )
        return ABTestResult(
            test_statistic=test_statistic, stat_sig=stat_sig, upper=upper, lower=lower
        )

    def _relative_difference_test_statistic(
        self,
        value_a: pd.Series,
        value_b: pd.Series,
        effect_size: pd.Series,
        variance_a: pd.Series,
        variance_b: pd.Series,
        sample_count_a: pd.Series,
        sample_count_b: pd.Series,
    ) -> ABTestResult:
        _EPS = 1e-9
        _EPS_2 = _EPS**2
        _variance_a, _variance_b = self._get_variance(
            value_a=value_a,
            value_b=value_b,
            effect_size=effect_size,
            variance_a=variance_a,
            variance_b=variance_b,
            sample_count_a=sample_count_a,
            sample_count_b=sample_count_b,
        )

        # Cache the variance for plotting.
        if self.data is not None:
            self.data.value.variance_a = _variance_a
            self.data.value.variance_b = _variance_b

        # Convert value_a / value_b, consider difference of logs.
        difference = np.log(np.maximum(value_b, _EPS))
        difference -= np.log(np.maximum(value_a, _EPS))
        # pyre-fixme[58]: `+` is not supported for operand types `int` and `Series`.
        difference_mean = difference - np.log(1 + effect_size)
        # Apply a delta method for the variance of the log by scaling
        # by a g'(𝜽) ** 2 term. In the case of g = log, g'(𝜽) = 1 / 𝜽.
        # See https://www.stata.com/support/faqs/statistics/delta-method/ for more details.
        difference_std_error = np.sqrt(
            _variance_a / np.maximum(value_a**2, _EPS_2)
            + _variance_b / np.maximum(value_b**2, _EPS_2)
        )
        test_statistic = pd.Series(difference_mean / difference_std_error)

        # Use symmetry of Normal distribution for a single critical value.
        critical_value: pd.Series = self.critical_value.absolute_critical_value

        # b / a >= 1 + r
        # log(b) - log(a) >= log(1 + r)
        # log(b) - log(a) - log(1 + r) >= 0
        # log(b) - log(a) = r'
        # -z < (r' - log(1 + r)) / sigma < z
        #    ONE_SIDED_UPPER                 ONE_SIDED_LOWER
        # => exp[r' - z * sigma] <= 1 + r <= exp[r' + z * sigma]
        upper = np.exp(difference + critical_value * difference_std_error)
        lower = np.exp(difference - critical_value * difference_std_error)
        if self.test_type == TestType.ONE_SIDED_LOWER:
            stat_sig = pd.Series(norm.cdf(test_statistic))
            lower = pd.Series([-np.inf] * len(upper))
        elif self.test_type == TestType.ONE_SIDED_UPPER:
            stat_sig = pd.Series(norm.sf(test_statistic))
            upper = pd.Series([np.inf] * len(lower))
        elif self.test_type == TestType.TWO_SIDED:
            stat_sig = pd.Series(
                np.minimum(norm.cdf(test_statistic), norm.sf(test_statistic))
            )

        else:
            raise ValueError(
                f"Expected test_type to be of TestType. Found {self.test_type}"
            )
        return ABTestResult(
            test_statistic=test_statistic, stat_sig=stat_sig, upper=upper, lower=lower
        )

    @abstractmethod
    def _get_variance(
        self,
        value_a: pd.Series,
        value_b: pd.Series,
        effect_size: pd.Series,
        variance_a: pd.Series,
        variance_b: pd.Series,
        sample_count_a: pd.Series,
        sample_count_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Extension specific variance."""
        raise NotImplementedError()

    def _plot(
        self, x_axis: pd.Series, data: TimeSeriesData, axis: plt.Axes, alpha: float
    ) -> None:
        # Plot the original time series with the respective variance.
        axis.plot(x_axis, data.value.value_b, label="value_b", ls="-", color="blue")
        axis.plot(x_axis, data.value.value_a, label="value_a", ls="-", color="teal")

        if self.test_statistic == TestStatistic.ABSOLUTE_DIFFERENCE:
            axis.plot(
                x_axis,
                data.value.value_a + data.value.effect_size,
                label="threshold",
                ls="--",
                color="red",
            )
        elif self.test_statistic == TestStatistic.RELATIVE_DIFFERENCE:
            axis.plot(
                x_axis,
                data.value.value_a * (1 + data.value.effect_size),
                label="threshold",
                ls="--",
                color="red",
            )
        a_se = np.sqrt(data.value.variance_a)
        axis.fill_between(
            x=x_axis,
            y1=data.value.value_a - 1.96 * a_se,
            y2=data.value.value_a + 1.96 * a_se,
            alpha=alpha,
            color="teal",
            label="value_a SE",
        )
        b_se = np.sqrt(data.value.variance_b)
        axis.fill_between(
            x=x_axis,
            y1=data.value.value_b - 1.96 * b_se,
            y2=data.value.value_b + 1.96 * b_se,
            alpha=alpha,
            color="blue",
            label="value_b SE",
        )
        axis.title.set_text("Value for A and B")


class TwoSampleRealValuedIntervalDetectorModel(TwoSampleIntervalDetectorModel):
    """An extension that considers two real values at each time index."""

    def _get_variance(
        self,
        value_a: pd.Series,
        value_b: pd.Series,
        effect_size: pd.Series,
        variance_a: pd.Series,
        variance_b: pd.Series,
        sample_count_a: pd.Series,
        sample_count_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        _variance_a = variance_a / sample_count_a
        _variance_b = variance_b / sample_count_b
        return _variance_a, _variance_b


class TwoSampleProportionIntervalDetectorModel(TwoSampleIntervalDetectorModel):
    """An extension that considers two proportion values at each time index."""

    def _get_test_statistic_hook(self, df: pd.core.frame.DataFrame) -> None:
        self.schema._validate_proportion(df, self.schema.proportion_columns)

    def _get_variance(
        self,
        value_a: pd.Series,
        value_b: pd.Series,
        effect_size: pd.Series,
        variance_a: pd.Series,
        variance_b: pd.Series,
        sample_count_a: pd.Series,
        sample_count_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """A Normal approximation to the Binomial distribution.

        X ~ Binomial(n, p), then X ≈ Normal(μ=np, σ=√np(1 - p))
        provided np ≥ 5 and n(1-p) ≥ 5.

        References:
            https://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation
        """
        # pyre-fixme[58]: `-` is not supported for operand types `int` and `Series`.
        _variance_a = value_a * (1 - value_a) / sample_count_a
        # pyre-fixme[58]: `-` is not supported for operand types `int` and `Series`.
        _variance_b = value_b * (1 - value_b) / sample_count_b
        return _variance_a, _variance_b


class TwoSampleCountIntervalDetectorModel(TwoSampleIntervalDetectorModel):
    """An extension that considers two count values at each time index."""

    def _get_test_statistic_hook(self, df: pd.core.frame.DataFrame) -> None:
        self.schema._validate_count(
            df, [TwoSampleColumns.VALUE_A, TwoSampleColumns.VALUE_B]
        )

    def _get_variance(
        self,
        value_a: pd.Series,
        value_b: pd.Series,
        effect_size: pd.Series,
        variance_a: pd.Series,
        variance_b: pd.Series,
        sample_count_a: pd.Series,
        sample_count_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """A Normal approximation to the Poisson distribution.

        X ~ Poisson(𝜆), then X ≈ Normal(μ=𝜆, σ=√𝜆) provided 𝜆≫10

        References:
            https://en.wikipedia.org/wiki/Poisson_distribution#General
        """
        _variance_a = value_a / sample_count_a
        _variance_b = value_b / sample_count_b
        return _variance_a, _variance_b


class TwoSampleArrivalTimeIntervalDetectorModel(TwoSampleIntervalDetectorModel):
    """An extension that considers two arrival time values at each time index."""

    def _get_test_statistic_hook(self, df: pd.core.frame.DataFrame) -> None:
        self.schema._validate_postitive(
            df, [TwoSampleColumns.VALUE_A, TwoSampleColumns.VALUE_B]
        )

    def _get_variance(
        self,
        value_a: pd.Series,
        value_b: pd.Series,
        effect_size: pd.Series,
        variance_a: pd.Series,
        variance_b: pd.Series,
        sample_count_a: pd.Series,
        sample_count_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """A Normal approximation to the Gamma distribution.

        X ~ Exponential(𝜆), then X_bar ~ Gamma(α=n, β=1/(𝜆n)).

        And,

        Gamma(α, β) -> Normal(μ=αβ, σ=√αβ) provided α≫10

        So,
        X_bar ≈ Normal(μ=1/𝜆, σ=1/(𝜆√n))

        References:
            http://webhome.auburn.edu/~carpedm/courses/stat3610b/documents/Exponential_sampling.pdf
        """
        _variance_a = value_a**2 / sample_count_a
        _variance_b = value_b**2 / sample_count_b
        return _variance_a, _variance_b


class OneSampleIntervalDetectorModel(IntervalDetectorModel, ABC):
    """An extension that considers one sample at each time index."""

    @property
    def schema(self) -> Schema:
        return OneSampleSchema()

    def _get_test_statistic(self, df: pd.core.frame.DataFrame) -> ABTestResult:
        return self._one_sample_test_statistic(
            value=df.value,
            effect_size=df.effect_size,
            variance=df.effect_size,
            sample_count=df.sample_count,
        )

    @abstractmethod
    def _one_sample_test_statistic(
        self,
        value: pd.Series,
        effect_size: pd.Series,
        variance: pd.Series,
        sample_count: pd.Series,
    ) -> ABTestResult:
        raise NotImplementedError


class OneSampleProportionIntervalDetectorModel(OneSampleIntervalDetectorModel):
    def _convert_alpha_to_critical_value(self, alpha: float, length: int) -> pd.Series:
        if self.data is None:
            raise ValueError("data cannot be None. Call `fit_predict` first.")
        data = self.data
        effect_size = data.value[OneSampleColumns.EFFECT_SIZE.value]
        n = data.value[OneSampleColumns.SAMPLE_COUNT.value]
        return pd.Series(
            binom.ppf(
                alpha,
                n=n,
                p=effect_size,
            )
            / n
        )

    def _get_test_statistic_hook(self, df: pd.core.frame.DataFrame) -> None:
        self.schema._validate_proportion(
            df, self.schema.proportion_columns + [OneSampleColumns.EFFECT_SIZE]
        )

    def _one_sample_test_statistic(
        self,
        value: pd.Series,
        effect_size: pd.Series,
        variance: pd.Series,
        sample_count: pd.Series,
    ) -> ABTestResult:
        """Computes a One Sample Test Statistic for a Sample Proportion.

        References:
            Clopper-Pearson Interval for Binomial Confidence Interval -
            https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
        """
        k = value * sample_count
        lower = pd.Series(
            beta.ppf(
                q=self.corrected_alpha,
                a=k,
                b=sample_count - k + 1,
            )
        )
        upper = pd.Series(
            beta.ppf(
                q=1 - self.corrected_alpha,
                a=k + 1,
                b=sample_count - k,
            )
        )
        if self.test_type == TestType.ONE_SIDED_LOWER:
            stat_sig = pd.Series(binom.cdf(k=k, n=sample_count, p=effect_size))
            lower = pd.Series([0.0] * len(upper))
        elif self.test_type == TestType.ONE_SIDED_UPPER:
            stat_sig = pd.Series(binom.sf(k=k, n=sample_count, p=effect_size))
            upper = pd.Series([1.0] * len(lower))
        elif self.test_type == TestType.TWO_SIDED:
            stat_sig = pd.Series(
                np.minimum(
                    binom.cdf(k=k, n=sample_count, p=effect_size),
                    binom.sf(k=k, n=sample_count, p=effect_size),
                )
            )
        else:
            raise ValueError(
                f"Expected test_type to be of TestType. Found {self.test_type}"
            )
        return ABTestResult(
            test_statistic=value,
            stat_sig=stat_sig,
            lower=lower,
            upper=upper,
        )

    def _plot(
        self, x_axis: pd.Series, data: TimeSeriesData, axis: plt.Axes, alpha: float
    ) -> None:
        n = data.value[OneSampleColumns.SAMPLE_COUNT.value]
        axis.plot(x_axis, data.value.value, label="value", ls="-", color="blue")
        axis.plot(
            x_axis,
            data.value.effect_size,
            label="threshold",
            ls="--",
            color="red",
        )
        axis.fill_between(
            x=x_axis,
            y1=binom.ppf(q=self.alpha, n=n, p=data.value.value) / n,
            y2=binom.ppf(q=1 - self.alpha, n=n, p=data.value.value) / n,
            alpha=alpha,
            color="blue",
            label="value SE",
        )
        axis.title.set_text("Values")
