# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module implements a simulator for generating synthetic time series data.
"""

from __future__ import annotations

import copy
from datetime import timedelta
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from pandas.tseries.frequencies import to_offset
from scipy.stats import norm  # @manual

# A TimedeltaLike object represents a time offset.
# E.g., length of 1 day can be represented by timedelta(days=1),or by
# a timestamp offset 86400, or by the offset alias "1D" defined in pandas
TimedeltaLike = Union[timedelta, float, str]


def moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    """Moving average of a numpy array.

    Args:
        a: numpy array.
        n: number of points over which we take moving avg.

    Returns:
        Array of moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


class Simulator:
    """TimeSeriesData simulator, to generate synthetic timeseries data.

    The Simulator currently supports generating synthetic time series
    using the STL, ARIMA models and also adds level and trend changepoints.

    Attributes:
        n: length of the time series.
        freq: desired frequency (e.g. daily, weekly) of a time series.
        start: start date of the time series.
    """

    def __init__(
        self,
        n: int = 100,
        freq: str = "D",
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        start: Optional[Any] = None,
    ) -> None:
        self.n = n
        self.freq = freq
        self.start = start

        # create time
        # pyre-fixme[4]: Attribute must be annotated.
        self.time = pd.date_range(
            start=start,
            freq=freq,
            periods=n,
        )

        # create the simulated time series
        # pyre-fixme[4]: Attribute must be annotated.
        self.timeseries = np.zeros(self.n)

    def arima_sim(
        self,
        ar: List[float],
        ma: List[float],
        mu: float = 0,
        sigma: float = 1,
        burnin: int = 10,
        d: int = 0,
        t: int = 0,
    ) -> TimeSeriesData:
        """Simulate data from ARIMA model.

        Data generation includes two steps:
        (1). Simulate data from ARMA(p', q) model

        The configuration of ARMA(p', q) model is:
        X_t = alpha_1 * X_{t-1} + ... + alpha_p * X_{t-p'}
                + 1 * epsilon_t + theta_1 * epsilon_{t-1} + ... + theta_q * epsilon_{t-q}

        (2). Add drift d
        d is the order of differencing
        p = p' - d for ARIMA(p, d, q)

        Args:
            ar: [alpha_1, ..., alpha_p'], coefficients of AR parameters.
                   p = len(alpha)
            ma: [theta_1, ..., theta_q], coefficients of MA parameters.
                   q = len(theta)
            epsilon: error terms follows normal distribution(mu, sigma).
            mu: mean of normal distribution for epsilon.
            sigma: standard dev of normal distribution for epsilon.
            burnin: number of data that will be dropped because lack of lagged data in the beginning.
            d: number of unit roots for non-stationary data.
            t: linear trend constant.

        Returns:
            ts: TimeSeries generated.
        Examples:
        >>> sim = Simulator(n=100, freq="MS", start = pd.to_datetime("2011-01-01 00:00:00"))
        >>> np.random.seed(100)
        >>> ts = sim.arima_sim(ar=[0.1, 0.05], ma = [0.04, 0.1], d = 1)
        """

        # validate params
        if sigma < 0 or sigma == 0:
            raise ValueError(
                "Standard deviation of normal distribution must be positive"
            )

        # Step 1: simulate ARMA(p, q)
        # get the max order of p' and q
        p_max = len(ar)

        # add theta_0=1 to ma coefficients
        ma = np.append(1, ma)
        q_max = len(ma)

        # simulate epsilon
        pq_max = max(p_max, q_max)
        epsilon = np.random.normal(mu, sigma, (self.n + pq_max + burnin, 1))

        # simulated data
        x = np.zeros((self.n + pq_max + burnin, 1))

        # initialization
        x[0] = epsilon[0]

        for i in range(1, x.shape[0]):
            AR = np.dot(ar[0 : min(i, p_max)], np.flip(x[i - min(i, p_max) : i], 0))
            MA = np.dot(
                ma[0 : min(i + 1, q_max)],
                np.flip(epsilon[i - min(i, q_max - 1) : i + 1], 0),
            )
            x[i] = AR + MA + t

        # Step 2: add drift, unit roots
        if d != 0:
            ARMA = x[-self.n :]
            m = ARMA.shape[0]
            temp = np.zeros((m + 1, 1))

            for _ in range(d):
                for j in range(m):
                    temp[j + 1] = ARMA[j] + temp[j]
                ARMA = temp[1:]

            # replacing data
            x[-self.n :] = temp[1:]

        # generate time
        time = pd.date_range(
            start=self.start,
            freq=self.freq,
            periods=self.n,
        )

        ts = TimeSeriesData(
            time=time, value=pd.Series(x[-self.n :].reshape(-1), copy=False)
        )
        return ts

    def add_trend(
        self, magnitude: float, trend_type: str = "linear", multiply: bool = False
    ) -> Simulator:
        """Add a trend component to the target time series for STL-based simulator.

        trend_type -  shape of the trend. {"linear","sigmoid"}

        Args:
            magnitude: slope of the trend, float.
            trend_type: linear or sigmoid, string.
            multiply: True if the trend is multiplicative, otherwise additive.

        Returns:
            The timeseries generated.
        """

        def component_gen(timepoints: np.ndarray) -> float:
            if trend_type == "sigmoid" or trend_type == "S":
                return magnitude * self.sigmoid(timepoints - 0.5)
            else:  # 'linear' trend by default
                return magnitude * timepoints

        return self._add_component(component_gen, multiply)

    def add_noise(
        self,
        magnitude: float = 1.0,
        multiply: bool = False,
    ) -> Simulator:

        """Add noise to the generated time series for STL-based simulator.

        Noise type is normal - noise will be generated from iid normal distribution;
        may consider adding noise generated by ARMA in the future if there're use cases.

        Args:
            magnitude: float.
            multiply: True if the noise is multiplicative, otherwise additive.

        Returns:
            Generated timeseries.
        """

        def component_gen(timepoints: np.ndarray) -> float:
            return magnitude * np.random.randn(len(timepoints))

        return self._add_component(component_gen, multiply)

    def add_seasonality(
        self,
        magnitude: float = 0.0,
        period: TimedeltaLike = "1D",
        multiply: bool = False,
    ) -> Simulator:
        """Add a seasonality component to the time series for STL-based simulator.

        Args:
            magnitude: slope of the trend, float.
            period: period of seasonality, timedelta.
            multiply: True if the seasonality is multiplicative, otherwise additive.

        Returns:
            Generated timeseries.
        """

        # pyre-fixme[6]: For 1st param expected `str` but got `Union[float, str,
        #  timedelta]`.
        period = self._convert_period(period)

        def component_gen(timepoints: np.ndarray) -> float:
            return magnitude * np.sin(2 * np.pi * timepoints)

        return self._add_component(component_gen, multiply, time_scale=period)

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-10 * x))

    @staticmethod
    def _convert_period(period: str) -> float:
        """
        Convert TimedeltaLike object to time offset in seconds
        """
        # pyre-fixme[16]: `Optional` has no attribute `nanos`.
        return to_offset(period).nanos / 1e9

    def _add_component(
        self,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        component_gen: Callable,
        multiply: bool,
        time_scale: Optional[float] = None,
    ) -> Simulator:
        """
        Add a new component to the target time series.
        The component is defined by component_gen.
        """
        timestamps = self.time.values.astype(np.float64) / 1e9
        if time_scale is None:
            time_scale = timestamps[-1] - timestamps[0] + np.finfo(float).eps
        timepoints = (timestamps - timestamps[0]) / time_scale
        component = component_gen(timepoints)

        if multiply:
            self.timeseries *= 1 + component
        else:
            self.timeseries += component

        return self

    def stl_sim(self) -> TimeSeriesData:
        """Simulate time series data with seasonality, trend, and noise.

        Args:
            None.

        Returns:
            Generated timeseries.

        Example usage:
        >>> sim = Simulator(n=100, freq="1D", start = pd.to_datetime("2011-01-01"))
        >>> sim.add_trend(magnitude=10)
        >>> sim.add_seasonality(5, period=timedelta(days=7))
        >>> sim.add_noise(magnitude=2)
        >>> sim_ts = sim.stl_sim()
        """

        ts = TimeSeriesData(
            time=self.time, value=pd.Series(self.timeseries, copy=False)
        )
        return ts

    def _adjust_length(self, length: int) -> None:
        """
        if given length is not compatible with other parameters, adjust the length
        """

        self.n = length
        self.time = pd.date_range(
            start=self.start,
            freq=self.freq,
            periods=self.n,
        )

    def _get_level_shift_y_val(
        self,
        random_seed: int = 100,
        cp_arr: Optional[List[int]] = None,
        level_arr: Optional[List[float]] = None,
        noise: float = 30,
        seasonal_period: int = 7,
        seasonal_magnitude: float = 3.0,
        anomaly_arr: Optional[List[int]] = None,
        z_score_arr: Optional[List[float]] = None,
    ) -> pd.Series:
        """
        generates values of a time series with multiple level shifts
        """
        if cp_arr is None:
            cp_arr = [100, 200, 350]
        if level_arr is None:
            level_arr = [1.35, 1.05, 1.35, 1.2]
        if anomaly_arr is None:
            anomaly_arr = []
        if z_score_arr is None:
            z_score_arr = []

        np.random.seed(seed=random_seed)

        # if cp_arr is not sorted, sort it
        cp_arr = sorted(cp_arr)

        # length of trend array should be one larger than cp array
        # so that there is a trend corresponding to every segment
        if len(level_arr) - len(cp_arr) != 1:
            raise ValueError(
                f"""
                Length of level array should be one greater than
                cp array. But we got
                cp_arr: {len(cp_arr)},
                level_arr: {len(level_arr)}
                """
            )

        if len(cp_arr) > 0 and cp_arr[-1] >= self.n:
            raise ValueError(f"Last cp {cp_arr[-1]} is greater than length {self.n}")

        cp_arr.append(self.n)
        cp_arr.insert(0, 0)

        y_val = np.ones(self.n)

        for i in range(len(cp_arr) - 1):
            cp_begin = cp_arr[i]
            cp_end = cp_arr[i + 1]

            y_val[cp_begin:cp_end] = norm.rvs(
                loc=level_arr[i], scale=noise, size=(cp_end - cp_begin)
            )

        # add anomalies
        if len(anomaly_arr) != len(z_score_arr):
            raise ValueError(
                f"""
                Length of anomaly array should be equal to z_score array. But we got
                anomaly_arr: {len(anomaly_arr)},
                z_score_arr: {len(z_score_arr)}
                """
            )
        for arr_idx, y_idx in enumerate(anomaly_arr):
            if y_idx < 0 or y_idx >= self.n:
                raise ValueError(f"Anamaly point {y_idx} is out of range")
            # find out the mean value of this segment
            seg_idx = 0
            while y_idx > cp_arr[seg_idx]:
                seg_idx += 1
            y_val[y_idx] = level_arr[seg_idx - 1] + z_score_arr[arr_idx] * noise

        # add seasonality
        y_val += seasonal_magnitude * np.sin(
            (np.pi / seasonal_period) * np.arange(self.n)
        )

        return pd.Series(y_val, copy=False)

    def level_shift_sim(
        self,
        random_seed: int = 100,
        cp_arr: Optional[List[int]] = None,
        level_arr: Optional[List[float]] = None,
        noise: float = 30.0,
        seasonal_period: int = 7,
        seasonal_magnitude: float = 3.0,
        anomaly_arr: Optional[List[int]] = None,
        z_score_arr: Optional[List[float]] = None,
    ) -> TimeSeriesData:
        """Produces a time series with level shifts.

        The positions of the level shifts are indicated by the beginning and
        end changepoints.
        the duration of the first change is [first_cp_begin, first_cp_end],
        the duration of the second change point is [second_cp_begin, self.n]

        Args:
            cp_arr: Array of changepoint locations.
            level_arr: Array containing levels for each segment. Since the
                number of segments is one more than the number of changepoints,
                hence, the level arr should be longer than the cp_arr by one.
            noise: std. dev of random Gaussian noise added.
            seasonal_period: periodicity of the time series.
            seasonal_magnitude: amplitude of the seasonality. Set this to 0, if
                you want a time series without seasonality.
            anomaly_arr: locations where we introduce an anomalous point.
            z_score_arr: same length as anomaly arr. This is the z-score of the
                anomaly introduced at the location indicated by the anomaly_arr.

        Returns:
            Generated timeseries.

        Example Usage:
        >>> sim2 = Simulator(n=450, start="2018-01-01")
        >>> ts2 = sim2.level_shift_sim(
                cp_arr=[100, 200],
                level_arr=[3, 20, 2],
                noise=3,
                seasonal_period=7,
                seasonal_magnitude=3,
                anomaly_arr = [50, 150, 250],
                z_score_arr = [10, -10, 20],
        )
        """

        if cp_arr is None:
            cp_arr = [100, 200, 350]
        if level_arr is None:
            level_arr = [1.35, 1.05, 1.35, 1.2]
        if anomaly_arr is None:
            anomaly_arr = []
        if z_score_arr is None:
            z_score_arr = []

        # we need atleast points till second cp
        if len(cp_arr) > 0 and self.n < np.max(cp_arr):
            self._adjust_length(np.max(cp_arr))

        ts = TimeSeriesData(
            time=self.time,
            value=self._get_level_shift_y_val(
                random_seed,
                cp_arr,
                level_arr,
                noise,
                seasonal_period,
                seasonal_magnitude,
                anomaly_arr,
                z_score_arr,
            ),
        )

        return ts

    def level_shift_multivariate_indep_sim(
        self,
        cp_arr: Optional[List[int]] = None,
        level_arr: Optional[List[float]] = None,
        noise: float = 30.0,
        seasonal_period: int = 7,
        seasonal_magnitude: float = 3.0,
        anomaly_arr: Optional[List[int]] = None,
        z_score_arr: Optional[List[float]] = None,
        dim: int = 3,
    ) -> TimeSeriesData:
        """Produces a multivariate time series with level shifts.

        The positions of the level shifts are indicated by the beginning and
        end changepoints.
        the duration of the first change is [first_cp_begin, first_cp_end],
        the duration of the second change point is [second_cp_begin, self.n]
        The number of dimensions are indicated by dim.

        Args:
            cp_arr: Array of changepoint locations.
            level_arr: Array containing levels for each segment. Since the
                number of segments is one more than the number of changepoints,
                hence, the level arr should be longer than the cp_arr by one.
            noise: std. dev of random Gaussian noise added.
            seasonal_period: periodicity of the time series.
            seasonal_magnitude: amplitude of the seasonality. Set this to 0, if
                you want a time series without seasonality.
            anomaly_arr: locations where we introduce an anomalous point.
            z_score_arr: same length as anomaly arr. This is the z-score of the
                anomaly introduced at the location indicated by the anomaly_arr.
            dim: number of dimensions of the timeseries.

        Returns:
            Generated timeseries.
        """

        if cp_arr is None:
            cp_arr = [100, 200, 350]
        if level_arr is None:
            level_arr = [1.35, 1.05, 1.35, 1.2]
        if anomaly_arr is None:
            anomaly_arr = []
        if z_score_arr is None:
            z_score_arr = []

        # we need atleast points till second cp
        if self.n < np.max(cp_arr):
            self._adjust_length(np.max(cp_arr))

        df_dict = {"time": self.time}

        for i in range(dim):
            yval = self._get_level_shift_y_val(
                100 * (i + 1),
                cp_arr,
                level_arr,
                noise,
                seasonal_period,
                seasonal_magnitude,
                anomaly_arr,
                z_score_arr,
            )
            df_dict[f"value{i+1}"] = yval

        ts_df = pd.DataFrame(df_dict, copy=False)

        return TimeSeriesData(ts_df)

    def trend_shift_sim(
        self,
        random_seed: int = 15,
        cp_arr: Optional[List[int]] = None,
        trend_arr: Optional[List[float]] = None,
        intercept: float = 100.0,
        noise: float = 30.0,
        seasonal_period: int = 7,
        seasonal_magnitude: float = 3.0,
        anomaly_arr: Optional[List[int]] = None,
        z_score_arr: Optional[List[int]] = None,
    ) -> TimeSeriesData:
        """Produces a time series with multiple trend shifts and seasonality.

        This can be used as synthetic data to test trend changepoints
        first_cp_begin is where the trend change begins, and continues till the
        end.

        Args:
            random_seed: Seed, to reproduce the same time series.
            cp_arr: Array of changepoint locations.
            trend_arr: Array containing trends for each segment. Since the
                number of segments is one more than the number of changepoints,
                hence, the trend arr should be longer than the cp_arr by one.
            noise: std. dev of random Gaussian noise added.
            seasonal_period: periodicity of the time series.
            seasonal_magnitude: amplitude of the seasonality. Set this to 0, if
                you want a time series without seasonality.
            anomaly_arr: locations where we introduce an anomalous point.
            z_score_arr: same length as anomaly arr. This is the z-score of the
                anomaly introduced at the location indicated by the anomaly_arr.

        Returns:
            Generated timeseries.

        Example usage:
        >>> sim2 = Simulator(n=450, start="2018-01-01")
        >>> ts2 = sim2.trend_shift_sim(
                cp_arr=[100, 200],
                trend_arr=[3, 20, 2],
                intercept=30,
                noise=30,
                seasonal_period=7,
                seasonal_magnitude=3,
                anomaly_arr = [50, 150, 250],
                z_score_arr = [10, -10, 20],
        )
        """

        # initializing the lists inside the function since
        # mutable lists as defaults is bad practice that linter flags
        if cp_arr is None:
            cp_arr = [100]
        if trend_arr is None:
            trend_arr = [3.0, 30.0]
        if anomaly_arr is None:
            anomaly_arr = []
        if z_score_arr is None:
            z_score_arr = []

        # if cp_arr is not sorted, sort it
        cp_arr = sorted(cp_arr)

        # length of trend array should be one larger than cp array
        # so that there is a trend corresponding to every segment
        if len(trend_arr) - len(cp_arr) != 1:
            raise ValueError(
                f"""
                Length of trend array should be one greater than
                cp array. But we got
                cp_arr: {len(cp_arr)},
                trend_arr: {len(trend_arr)}
                """
            )

        if len(cp_arr) > 0 and cp_arr[-1] >= self.n:
            raise ValueError(f"Last cp {cp_arr[-1]} is greater than length {self.n}")

        cp_arr.append(self.n)
        cp_arr.insert(0, 0)

        y_val = np.full(self.n, intercept, dtype=float)

        for i in range(len(cp_arr) - 1):
            cp_begin = cp_arr[i]
            cp_end = cp_arr[i + 1]

            y_val[cp_begin:cp_end] = y_val[cp_begin:cp_end] + trend_arr[i] * np.arange(
                cp_begin, cp_end
            )

            if i > 0:
                delta_val = y_val[cp_begin] - y_val[cp_begin - 1]
                y_val[cp_begin:cp_end] -= delta_val

        # add seasonality
        y_val += seasonal_magnitude * np.sin(
            (np.pi / seasonal_period) * np.arange(self.n)
        )

        # add noise and anomalies
        noise_arr = norm.rvs(loc=0, scale=noise, size=self.n)
        if len(anomaly_arr) != len(z_score_arr):
            raise ValueError(
                f"""
                Length of anomaly array should be equal to z_score array. But we got
                anomaly_arr: {len(anomaly_arr)},
                z_score_arr: {len(z_score_arr)}
                """
            )
        for arr_idx, y_idx in enumerate(anomaly_arr):
            if y_idx < 0 or y_idx >= self.n:
                raise ValueError(f"Anomaly point {y_idx} is out of range")
            # pyre-fixme[16]: `Sequence` has no attribute `__setitem__`.
            noise_arr[y_idx] = z_score_arr[arr_idx] * noise

        y_val += noise_arr

        ts = TimeSeriesData(time=self.time, value=pd.Series(y_val, copy=False))

        return ts

    def _check_arr_inputs(
        self,
        ts: TimeSeriesData,
        loc_arr: Optional[List[int]],
        param_arr: Optional[List[float]],
        loc_param_diff: int = 1,
    ) -> None:
        """Private method, to check whether arguments to injected anomaly functions are valid.

        Args:
            ts: Given time series.
            loc_arr: Locations of anomaly/change point.
            param_arr: Parameters for each segment.
            loc_param_diff: Expected difference in length of loc and param arrays.
        """

        if loc_arr is None or len(loc_arr) == 0:
            raise ValueError("Changepoint Array cannot be empty")

        if param_arr is None or len(param_arr) == 0:
            raise ValueError("Parameter Array cannot be empty")

        len_ts = len(ts)

        # check if cp_arr is sorted
        if sorted(loc_arr) != loc_arr:
            raise ValueError("cp_arr must be sorted")

        # check if last cp is less than length of ts
        if loc_arr[-1] >= len_ts:
            raise ValueError(
                f"Last CP is at {loc_arr[-1]}, larger than length of ts, {len_ts}"
            )

        if (len(loc_arr) - len(param_arr)) != loc_param_diff:
            raise ValueError(
                f"""
                Parameter array should be {loc_param_diff} longer than cp_arr, but
                cp_arr is {len(loc_arr)} long and parameter array is
                {len(param_arr)} long
                """
            )

    def inject_level_shift(
        self,
        ts_input: TimeSeriesData,
        cp_arr: Optional[List[int]] = None,
        level_arr: Optional[List[float]] = None,
    ) -> TimeSeriesData:
        """Produces Time Series with injected level shifts

        Args:
            ts_input: Input time series data.
            cp_arr: Array of changepoint locations.
            level_arr: Array containing levels for each segment. Since the
                number of segments is one more than the number of changepoints,
                hence, the level arr should be longer than the cp_arr by one.

        Returns:
            Timeseries with injected anomalies
        """

        if cp_arr is None or level_arr is None:
            return ts_input

        ts = copy.deepcopy(ts_input)
        self._check_arr_inputs(ts=ts, loc_arr=cp_arr, param_arr=level_arr)

        # check if two array sizes match, cp_arr is sorted, and less than length
        for i in range(len(cp_arr) - 1):
            start = cp_arr[i]
            end = cp_arr[i + 1]
            ts.value.values[start:end] += level_arr[i]

        return ts

    def inject_trend_shift(
        self,
        ts_input: TimeSeriesData,
        cp_arr: Optional[List[int]] = None,
        trend_arr: Optional[List[float]] = None,
    ) -> TimeSeriesData:
        """Produces Time Series with injected level shifts

        Args:
            ts_input: Input time series data.
            cp_arr: Array of changepoint locations.
            trend_arr: Array containing trends for each segment. Since the
                number of segments is one more than the number of changepoints,
                hence, the trend arr should be longer than the cp_arr by one.

        Returns:
            Timeseries with injected anomalies
        """

        if cp_arr is None or trend_arr is None:
            return ts_input

        ts = copy.deepcopy(ts_input)
        self._check_arr_inputs(ts=ts, loc_arr=cp_arr, param_arr=trend_arr)

        for i in range(len(cp_arr) - 1):
            start = cp_arr[i]
            end = cp_arr[i + 1]
            add_arr = trend_arr[i] * np.arange(end - start)
            ts.value.values[start:end] += add_arr

            # pyre-ignore Undefined attribute [16]: `float` has no attribute `__getitem__
            ts.value.values[end:] += add_arr[-1]

        return ts

    def inject_spikes(
        self,
        ts_input: TimeSeriesData,
        anomaly_arr: Optional[List[int]] = None,
        z_score_arr: Optional[List[float]] = None,
        epsilon_std_dev: float = 0,
    ) -> TimeSeriesData:
        """Produces time series with injected spikes.

        Args:
            ts_input: Input time series data.
            anomaly_arr: locations where we introduce an anomalous point.
            z_score_arr: same length as anomaly arr. This is the z-score of the
                anomaly introduced at the location indicated by the anomaly_arr.
            epsilon_std_dev: A small value of standard deviation, to produce anomalies
                even when the variance of the original time series is close to zero.
                This is useful for simulating anomalies in time series of nearly constant
                data.

        Returns:
            Timeseries with injected anomalies
        """

        if anomaly_arr is None or z_score_arr is None:
            return ts_input
        ts = copy.deepcopy(ts_input)
        self._check_arr_inputs(
            ts=ts, loc_arr=anomaly_arr, param_arr=z_score_arr, loc_param_diff=0
        )

        # get the std. dev of the time series
        ts_val = ts.value.values
        mov_avg = moving_average(ts_val, 3)
        std_dev = np.std(ts_val[2:] - mov_avg)

        for a_index, z_score in zip(anomaly_arr, z_score_arr):
            ts.value.values[a_index] += z_score * (std_dev + epsilon_std_dev)

        return ts
