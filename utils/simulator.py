#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from typing import List, Any, Union, Callable
from datetime import timedelta
from pandas.tseries.frequencies import to_offset

from infrastrategy.kats.consts import TimeSeriesData
from scipy.stats import norm  # @manual

# A TimedeltaLike object represents a time offset.
# E.g., length of 1 day can be represented by timedelta(days=1),or by
# a timestamp offset 86400, or by the offset alias "1D" defined in pandas
TimedeltaLike = Union[timedelta, float, str]


class Simulator:
    """
    TimeSeriesData simulator, currently support:

    - arima_sim
    - stl_sim
    - level_shift_sim
    """

    def __init__(self,
                 n: int = 100,
                 freq: str = 'D',
                 start: Any = None,
                ) -> None:
        self.n = n
        self.freq = freq
        self.start = start

        # create time
        self.time = pd.date_range(
            start=start,
            freq=freq,
            periods=n,
        )

        # create the simulated time series
        self.timeseries = np.zeros(self.n)

    def arima_sim(self,
                  ar: List[float],
                  ma: List[float],
                  mu: float = 0,
                  sigma: float = 1,
                  burnin: int = 10,
                  d: int = 0,
                  t: int = 0,
                 ) -> TimeSeriesData:
        """
        Simulate data from ARIMA model, which includes two steps:
        (1). Simulate data from ARMA(p', q) model

        The configuration of ARMA(p', q) model is:
        X_t = alpha_1 * X_{t-1} + ... + alpha_p * X_{t-p'}
                + 1 * epsilon_t + theta_1 * epsilon_{t-1} + ... + theta_q * epsilon_{t-q}

        (2). Add drift d
        d is the order of differencing
        p = p' - d for ARIMA(p, d, q)

        Parameters:
            ar: [alpha_1, ..., alpha_p'], coefficients of AR parameters
                   p = len(alpha)
            ma: [theta_1, ..., theta_q], coefficients of MA parameters
                   q = len(theta)
            epsilon: error terms follows normal distribution(mu, sigma)
            mu: mean of normal distribution for epsilon
            sigma: standard dev of normal distribution for epsilon
            burnin: number of data that will be dropped because lack of lagged data in the beginning
            d: number of unit roots for non-stationary data
            t: linear trend constant

        Examples:
        >>> sim = Simulator(n=100, freq="MS", start = pd.to_datetime("2011-01-01 00:00:00"))
        >>> np.random.seed(100)
        >>> ts = sim.arima_sim(ar=[0.1, 0.05], ma = [0.04, 0.1], d = 1)
        """
        # validate params
        if sigma < 0 or sigma == 0:
            raise ValueError("Standard deviation of normal distribution must be positive")

        # Step 1: simulate ARMA(p, q)
        # get the max order of p' and q
        p_max = len(ar)

        # add theta_0=1 to ma coefficients
        ma = np.append(1, ma)
        q_max = len(ma)

        # simulate epsilon
        pq_max = max(p_max, q_max)
        epsilon = np.random.normal(mu, sigma,
                                   (self.n + pq_max + burnin,
                                    1))

        # simulated data
        x = np.zeros((self.n + pq_max + burnin, 1))

        # initialization
        x[0] = epsilon[0]

        for i in range(1, x.shape[0]):
            AR = np.dot(
                ar[0 : min(i, p_max)],
                np.flip(x[i - min(i, p_max) : i], 0)
            )
            MA = np.dot(
                ma[0 : min(i + 1, q_max)],
                np.flip(epsilon[i - min(i, q_max - 1) : i + 1], 0)
            )
            x[i] = AR + MA + t

        # Step 2: add drift, unit roots
        if d != 0:
            ARMA = x[-self.n: ]
            m = ARMA.shape[0]
            temp = np.zeros((m + 1, 1))

            for _ in range(d):
                for j in range(m):
                    temp[j + 1] = ARMA[j] + temp[j]
                ARMA = temp[1: ]

            # replacing data
            x[-self.n: ] = temp[1: ]

        # generate time
        time = pd.date_range(
            start=self.start,
            freq=self.freq,
            periods=self.n,
        )

        ts = TimeSeriesData(time = time, value = pd.Series(x[-self.n: ].reshape(-1)))
        return ts

    def add_trend(self,
                   magnitude: float,
                   trend_type: str = "linear",
                   multiply: bool = False):
        """
        Add a trend component to the target time series for STL-based simulator
        trend_type -  shape of the trend. {"linear","sigmoid"}

        Parameters:
        - magnitude: slope of the trend, float
        - trend_type: linear or sigmoid, string
        - multiply: True if the trend is multiplicative, otherwise additive
        """

        def component_gen(timepoints):
            if trend_type == "sigmoid" or trend_type == "S":
                return magnitude * self.sigmoid(timepoints - 0.5)
            else: # 'linear' trend by default
                return magnitude * timepoints

        return self._add_component(
            component_gen,
            multiply
        )


    def add_noise(self,
                   magnitude: float = 1.0,
                   multiply: bool = False,
                   ):

        """
        Add noise to the generated time series for STL-based simulator
        noise type is normal - noise will be generated from iid normal distribution;
        may consider adding noise generated by ARMA in the future if there're use cases.

        Parameters:
        - magnitude: float
        - multiply: True if the noise is multiplicative, otherwise additive
        """

        def component_gen(timepoints):
            return magnitude * np.random.randn(len(timepoints))

        return self._add_component(
            component_gen,
            multiply
        )

    def add_seasonality(self,
                        magnitude: float = 0.0,
                        period: TimedeltaLike = "1D",
                        multiply: bool = False):
        """
        Add a seasonality component to the time series for STL-based simulator

        Parameters:
        - magnitude: slope of the trend, float
        - period: period of seasonality, timedelta
        - multiply: True if the seasonality is multiplicative, otherwise additive
        """

        period = self._convert_period(period)

        def component_gen(timepoints):
            return magnitude * np.sin(2 * np.pi * timepoints)

        return self._add_component(
            component_gen,
            multiply,
            time_scale = period
        )

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-10 * x))

    @staticmethod
    def _convert_period(period):
        """
        Convert TimedeltaLike object to time offset in seconds
        """
        return to_offset(period).nanos / 1e9

    def _add_component(self,
                       component_gen: Callable,
                       multiply: bool,
                       time_scale: float = None):
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
        """
        Simulate time series data with seasonality, trend, and noise.

        Example usage:
        sim = Simulator(n=100, freq="1D", start = pd.to_datetime("2011-01-01"))
        sim.add_trend(magnitude=10)
        sim.add_seasonality(5, period=timedelta(days=7))
        sim.add_noise(magnitude=2)
        sim_ts = sim.stl_sim()
        """
        ts = TimeSeriesData(time = self.time, value = pd.Series(self.timeseries))
        return ts

    def _adjust_length(self, length: int):
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
        cp_arr: List[int] = None,
        level_arr: List[int] = None,
        noise: int = 30,
        seasonal_period: int = 7,
        seasonal_magnitude: int = 3,
    ) -> pd.Series:
        """
        generates values of a time series with multiple level shifts

        """
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

        if cp_arr[-1] >= self.n:
            raise ValueError(f'Last cp {cp_arr[-1]} is greater than length {self.n}')

        cp_arr.append(self.n)
        cp_arr.insert(0, 0)

        y_val = np.ones(self.n)

        for i in range(len(cp_arr) - 1):
            cp_begin = cp_arr[i]
            cp_end = cp_arr[i + 1]

            y_val[cp_begin : cp_end] = norm.rvs(
                loc=level_arr[i], scale=noise, size=(cp_end - cp_begin)
            )

        # add seasonality
        y_val += seasonal_magnitude * np.sin((np.pi / seasonal_period) * np.arange(self.n))

        return pd.Series(y_val)



    def level_shift_sim(self, random_seed: int = 100, cp_arr: List[int] = None,
                        level_arr: List[int] = None,
                        noise: int = 30, seasonal_period: int = 7,
                        seasonal_magnitude: int = 3) -> TimeSeriesData:
        """
        Produces a time series with 3 level shifts.
        The positions of the level shifts are indicated by the beginning and
        end changepoints.
        the duration of the first change is [first_cp_begin, first_cp_end],
        the duration of the second change point is [second_cp_begin, self.n]

        """

        if cp_arr is None:
            cp_arr = [100, 200, 350]
        if level_arr is None:
            level_arr = [1.35, 1.05, 1.35, 1.2]

        # we need atleast points till second cp
        if self.n < np.max(cp_arr):
            self._adjust_length(np.max(cp_arr))

        ts = TimeSeriesData(
            time=self.time,
            value=self._get_level_shift_y_val(random_seed, cp_arr, level_arr,
                                              noise, seasonal_period,
                                              seasonal_magnitude)
        )

        return ts

    def level_shift_multivariate_indep_sim(self, cp_arr: List[int] = None,
                                           level_arr: List[int] = None,
                                           noise: int = 30, seasonal_period: int = 7,
                                           seasonal_magnitude: int = 3,
                                           dim: int = 3) -> TimeSeriesData:
        """
        Produces a multivariate time series with 3 level shifts.
        The positions of the level shifts are indicated by the beginning and
        end changepoints.
        the duration of the first change is [first_cp_begin, first_cp_end],
        the duration of the second change point is [second_cp_begin, self.n]
        The number of dimensions are indicated by dim
        """

        if cp_arr is None:
            cp_arr = [100, 200, 350]
        if level_arr is None:
            level_arr = [1.35, 1.05, 1.35, 1.2]

        # we need atleast points till second cp
        if self.n < np.max(cp_arr):
            self._adjust_length(np.max(cp_arr))

        df_dict = {'time': self.time}

        for i in range(dim):
            yval = self._get_level_shift_y_val(100 * (i + 1), cp_arr, level_arr,
                                              noise, seasonal_period,
                                              seasonal_magnitude)
            df_dict[f'value{i+1}'] = yval

        ts_df = pd.DataFrame(df_dict)

        return TimeSeriesData(ts_df)

    def trend_shift_sim(
        self,
        random_seed: int = 15,
        cp_arr: List[int] = None,
        trend_arr: List[int] = None,
        intercept: int = 100,
        noise: int = 30, seasonal_period: int = 7,
        seasonal_magnitude: int = 3,
        ) -> TimeSeriesData:
        """
        Produces a time series with multiple trend shifts and seasonality.
        This can be used as synthetic data to test trend changepoints
        first_cp_begin is where the trend change begins, and continues till the
        end
        """
        # initializing the lists inside the function since
        # mutable lists as defaults is bad practice that linter flags
        if cp_arr is None:
            cp_arr = [100]
        if trend_arr is None:
            trend_arr = [3, 30]

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

        if cp_arr[-1] >= self.n:
            raise ValueError(f'Last cp {cp_arr[-1]} is greater than length {self.n}')

        cp_arr.append(self.n)
        cp_arr.insert(0, 0)

        y_val = intercept * np.ones(self.n)

        for i in range(len(cp_arr) - 1):
            cp_begin = cp_arr[i]
            cp_end = cp_arr[i + 1]

            y_val[cp_begin : cp_end] = (
                y_val[cp_begin: cp_end]
                + trend_arr[i] * np.arange(cp_begin, cp_end)
            )

            if i > 0:
                delta_val = y_val[cp_begin] - y_val[cp_begin - 1]
                y_val[cp_begin: cp_end] -= delta_val

        # add seasonality
        y_val += seasonal_magnitude * np.sin((np.pi / seasonal_period) * np.arange(self.n))

        # add noise
        y_val += norm.rvs(loc=0, scale=noise, size=self.n)

        ts = TimeSeriesData(
            time=self.time,
            value=pd.Series(y_val)
        )

        return ts
