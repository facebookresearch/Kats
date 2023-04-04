# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kats.consts import (
    DataError,
    DataIrregularGranularityError,
    IRREGULAR_GRANULARITY_ERROR,
    ParameterError,
    TimeSeriesData,
)
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# from numpy.typing import ArrayLike
ArrayLike = Union[np.ndarray, Sequence[float]]
Figsize = Tuple[int, int]

_log: logging.Logger = logging.getLogger(__name__)


def _identity(x: ArrayLike) -> ArrayLike:
    return x


class TimeSeriesDecomposition:
    """Model class for Time Series Decomposition.

    This class provides utilities to decompose an input time series

    Pass specific arguments to seasonal_decompose and STL functions via kwargs.

    Attributes:
        data: the input time series data as `TimeSeriesData`
        decomposition: `additive` or `multiplicative` decomposition
        method: `STL decompostion` or `seasonal_decompose`
    """

    freq: Optional[Union[str, pd.Timedelta]] = None
    results: Optional[Dict[str, TimeSeriesData]] = None
    decomposition: str
    method: Callable[[pd.DataFrame], Dict[str, pd.DataFrame]]
    period: Optional[int]
    seasonal: int
    trend: Optional[int]
    low_pass: Optional[int]
    seasonal_deg: int
    trend_deg: int
    low_pass_deg: int
    robust: bool
    seasonal_jump: int
    trend_jump: int
    low_pass_jump: int

    def __init__(
        self,
        data: TimeSeriesData,
        decomposition: str = "additive",
        method: str = "STL",
        **kwargs: Any,
    ) -> None:
        if not isinstance(data.value, pd.Series) and method != "seasonal_decompose":
            msg = f"Only support univariate time series, but got {type(data.value)}. \
                For multivariate, use method='seasonal_decompose'."
            logging.error(msg)
            raise ValueError(msg)

        self.data = data
        if decomposition in ("additive", "multiplicative"):
            self.decomposition = decomposition
        else:
            logging.info(
                "Invalid decomposition setting specified; "
                "defaulting to Additive Decomposition."
            )
            self.decomposition = "additive"
        if method == "seasonal_decompose":
            self.method = self.__decompose_seasonal
        else:
            if method != "STL":
                logging.info(
                    f"""Invalid decomposition setting {method} specified.
                    Possible Values: STL, seasonal_decompose.
                    Defaulting to STL."""
                )
            self.method = self.__decompose_STL

        ## The following are params for the STL Module
        self.period = kwargs.get("period", None)
        self.seasonal = kwargs.get("seasonal", 7)
        self.trend = kwargs.get("trend", None)
        self.low_pass = kwargs.get("low_pass", None)
        self.seasonal_deg = kwargs.get("seasonal_deg", 1)
        self.trend_deg = kwargs.get("trend_deg", 1)
        self.low_pass_deg = kwargs.get("low_pass_deg", 1)
        self.robust = kwargs.get("robust", False)
        self.seasonal_jump = kwargs.get("seasonal_jump", 1)
        self.trend_jump = kwargs.get("trend_jump", 1)
        self.low_pass_jump = kwargs.get("low_pass_jump", 1)

    def __clean_ts(self) -> pd.DataFrame:
        """Internal function to clean the time series.

        Internal function to interpolate time series and infer frequency of
        time series required for decomposition.
        """
        original = self.data.to_dataframe()
        original.set_index(self.data.time_col_name, inplace=True)

        if self.data.infer_freq_robust() is None:
            original = original.asfreq("D")
            logging.info("Setting frequency to Daily since it cannot be inferred")
            self.freq = pd.infer_freq(original.index)
        else:
            self.freq = self.data.infer_freq_robust()

        original.interpolate(
            method="polynomial", limit_direction="both", order=3, inplace=True
        )

        ## This is a hack since polynomial interpolation is not working here
        if any(original.isna()):
            original.interpolate(method="linear", limit_direction="both", inplace=True)

        # pyre-ignore[7]: Expected `DataFrame` but got
        #  `Union[pd.core.frame.DataFrame, pd.core.series.Series]`.
        return original

    def _get_period(self) -> Optional[int]:
        period = self.period
        freq = self.freq
        if period is None:
            if freq is not None and isinstance(freq, str) and "T" in freq:
                logging.warning(
                    """Seasonal Decompose cannot handle sub day level granularity.
                    Please consider setting period yourself based on the input data.
                    Defaulting to a period of 2."""
                )
                period = 2
            elif freq is not None and isinstance(freq, pd.Timedelta) and freq.days == 0:
                logging.warning(
                    """Seasonal Decompose cannot handle sub day level granularity.
                    Please consider setting period yourself based on the input data.
                    Defaulting to a period of 2."""
                )
                period = 2
        return period

    def __decompose_seasonal(self, original: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Internal function to call seasonal_decompose to do the decomposition."""
        period = self._get_period()
        result = seasonal_decompose(original, model=self.decomposition, period=period)

        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "rem": result.resid,
        }

    def __decompose_STL(self, original: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Internal function to call STL to do the decomposition.

        The arguments to STL can be passed in the class via kwargs
        """
        self.period = period = self._get_period()

        if self.decomposition == "additive":
            data = original
            post_transform = _identity
        else:
            if np.any(original <= 0):
                logging.error(
                    "Multiplicative seasonality is not appropriate "
                    "for zero and negative values."
                )
            data = np.log(original)
            post_transform = np.exp

        result = STL(
            data,
            period=period,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            robust=self.robust,
            seasonal_jump=self.seasonal_jump,
            trend_jump=self.trend_jump,
            low_pass_jump=self.low_pass_jump,
        ).fit()

        return {
            "trend": post_transform(result.trend),
            "seasonal": post_transform(result.seasonal),
            "rem": post_transform(result.resid),
        }

    def __decompose(self, original: pd.DataFrame) -> Dict[str, TimeSeriesData]:
        output = self.method(original)
        ret = {}
        for name, ts in output.items():
            tmp = pd.DataFrame(ts)
            if original.shape[1] > 1:
                tmp.columns = original.columns
            ret[name] = TimeSeriesData(value=tmp, time=original.index)
        return ret

    def decomposer(self) -> Dict[str, TimeSeriesData]:
        """Decompose the time series.

        Returns:
            A dictionary with three time series for the three components:
            `trend` : Trend
            `seasonal` : Seasonality, and
            `rem` : Residual
        """
        original = self.__clean_ts()
        self.results = result = self.__decompose(original)
        return result

    def plot(
        self,
        figsize: Optional[Figsize] = None,
        linewidth: int = 3,
        xlabel: str = "Time",
        original_title: str = "Original Time Series",
        trend_title: str = "Trend",
        seasonality_title: str = "Seasonality",
        residual_title: str = "Residual",
        subplot_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
        """Plot the original time series and the three decomposed components."""
        results = self.results
        if results is None:
            raise ValueError("Call decomposer() before plot().")

        if figsize is None:
            figsize = (20, 10)
        if subplot_kwargs is None:
            subplot_kwargs = {"hspace": 0.2}

        sharex = kwargs.pop("sharex", True)
        fig, axs = plt.subplots(
            nrows=4, ncols=1, figsize=figsize, sharex=sharex, **kwargs
        )

        axs[0].plot(
            self.data.time.values,
            self.data.value.values,
            linewidth=linewidth,
        )
        axs[0].set_title(original_title)

        titles = [trend_title, seasonality_title, residual_title]
        parts = ["trend", "seasonal", "rem"]
        for part, ax, title in zip(parts, axs[1:], titles):
            ts: TimeSeriesData = results[part]
            ax.plot(ts.time.values, ts.value.values, linewidth=linewidth)
            ax.set_title(title)

        axs[3].set_xlabel(xlabel)
        plt.subplots_adjust(**subplot_kwargs)
        return (axs[0], axs[1], axs[2], axs[3])


class SeasonalityHandler:
    """
    SeasonalityHandler is a class that do timeseries STL decomposition for detecors
    Attributes:
        data: TimeSeriesData that need to be decomposed
        seasonal_period: str, default value is 'daily'. Other possible values: 'hourly', 'weekly', 'biweekly', 'monthly', 'yearly'

    >>> # Example usage:
    >>> from kats.utils.simulator import Simulator
    >>> sim = Simulator(n=120, start='2018-01-01')
    >>> ts = sim.level_shift_sim(cp_arr = [60], level_arr=[1.35, 1.05], noise=0.05, seasonal_period=7, seasonal_magnitude=0.575)
    >>> sh = SeasonalityHandler(data=ts, seasonal_period='weekly')
    >>> sh.get_seasonality()
    >>> sh.remove_seasonality()
    """

    def __init__(
        self,
        data: TimeSeriesData,
        seasonal_period: str = "daily",
        ignore_irregular_freq: bool = False,
        **kwargs: Any,
    ) -> None:
        if len(data) < 7:
            msg = "Input data for SeasonalityHandler must have at least 7 data points."
            _log.error(msg)
            raise DataError(msg)

        self.data = data

        _map = {
            "hourly": 1,
            "daily": 24,
            "weekly": 7 * 24,
            "biweekly": 14 * 24,
            "monthly": 30 * 24,
            "yearly": 365 * 24,
        }
        if seasonal_period not in _map:
            msg = "Invalid seasonal_period, possible values are 'hourly', 'daily', 'weekly', 'biweekly', 'monthly', and 'yearly'"
            logging.error(msg)
            raise ParameterError(msg)
        self.seasonal_period: int = _map[seasonal_period]

        self.low_pass_jump_factor: float = kwargs.get("lpj_factor", 0.15)
        self.trend_jump_factor: float = kwargs.get("tj_factor", 0.15)

        if ignore_irregular_freq:
            self.frequency: pd.Timedelta = self.data.infer_freq_robust()

        else:
            self.frequency: pd.Timedelta = self.data.freq_to_timedelta()
            if self.frequency is None or self.frequency is pd.NaT:
                # Use the top frequency if any, when not able to infer from data.
                freq_counts = (
                    self.data.time.diff().value_counts().sort_values(ascending=False)
                )
                if freq_counts.iloc[0] >= int(len(self.data)) * 0.5 - 1:
                    self.frequency = freq_counts.index[0]
                else:
                    _log.debug(f"freq_counts: {freq_counts}")
                    raise DataIrregularGranularityError(IRREGULAR_GRANULARITY_ERROR)

        self.frequency_sec: int = int(self.frequency.total_seconds())
        self.frequency_sec_str: str = str(self.frequency_sec) + "s"

        # calculate resample base in second level
        time0 = pd.to_datetime(self.data.time[0])
        # calculate remainder as resampling base
        resample_base_sec = (
            time0.day * 24 * 60 * 60
            + time0.hour * 60 * 60
            + time0.minute * 60
            + time0.second
        ) % self.frequency_sec

        self.decomposer_input: TimeSeriesData = self.data.interpolate(
            freq=self.frequency_sec_str,
            base=resample_base_sec,
        )

        data_time_idx = self.decomposer_input.time.isin(self.data.time)
        if len(self.decomposer_input.time[data_time_idx]) != len(self.data):
            raise DataIrregularGranularityError(IRREGULAR_GRANULARITY_ERROR)

        self.period: int = min(
            int(self.seasonal_period * 60 * 60 / self.frequency.total_seconds()),
            len(self.data) // 2,
        )

        if self.period < 2:
            _log.info(f"The period {self.period} is less than 2. Setting to 7.")
            self.period = 7

        self.decomp: Optional[dict[str, Any]] = None

        self.ifmulti: bool = False
        # for multi-variate TS
        if len(self.data.value.values.shape) != 1:
            self.ifmulti = True
            self.num_seq: int = self.data.value.values.shape[1]

        self.data_season = TimeSeriesData(time=self.data.time, value=self.data.value)
        self.data_nonseason = TimeSeriesData(time=self.data.time, value=self.data.value)

    def _decompose(self) -> None:
        if not self.ifmulti:
            decomposer = TimeSeriesDecomposition(
                self.decomposer_input,
                period=max(self.period, 2),
                robust=True,
                seasonal_deg=0,
                trend_deg=1,
                low_pass_deg=1,
                low_pass_jump=max(
                    int((self.period + 1) * self.low_pass_jump_factor), 1
                ),
                seasonal_jump=1,
                trend_jump=max(int((self.period + 1) * self.trend_jump_factor), 1),
            )

            self.decomp = decomposer.decomposer()
            return

        self._decompose_multi()

    def _decompose_multi(self) -> None:
        self.decomp = {}
        for i in range(self.num_seq):
            temp_ts = TimeSeriesData(
                time=self.decomposer_input.time,
                value=pd.Series(self.decomposer_input.value.values[:, i], copy=False),
            )
            decomposer = TimeSeriesDecomposition(
                temp_ts,
                period=max(self.period, 2),
                robust=True,
                seasonal_deg=0,
                trend_deg=1,
                low_pass_deg=1,
                low_pass_jump=max(
                    int((self.period + 1) * self.low_pass_jump_factor), 1
                ),
                seasonal_jump=1,
                trend_jump=max(int((self.period + 1) * self.trend_jump_factor), 1),
            )
            assert self.decomp is not None
            self.decomp[str(i)] = decomposer.decomposer()

    def remove_seasonality(self) -> TimeSeriesData:
        if self.decomp is None:
            self._decompose()
        if not self.ifmulti:
            decomp = self.decomp
            assert decomp is not None
            data_time_idx = decomp["rem"].time.isin(self.data_nonseason.time)

            self.data_nonseason.value = pd.Series(
                decomp["rem"][data_time_idx].value
                + decomp["trend"][data_time_idx].value,
                name=self.data_nonseason.value.name,
                copy=False,
            )
            return self.data_nonseason
        decomp = self.decomp
        assert decomp is not None
        data_time_idx = decomp[str(0)]["rem"].time.isin(self.data_nonseason.time)
        for i in range(self.num_seq):
            self.data_nonseason.value.iloc[:, i] = pd.Series(
                decomp[str(i)]["rem"][data_time_idx].value
                + decomp[str(i)]["trend"][data_time_idx].value,
                name=self.data_nonseason.value.iloc[:, i].name,
                copy=False,
            )

        return self.data_nonseason

    def get_seasonality(self) -> TimeSeriesData:
        if self.decomp is None:
            self._decompose()
        decomp = self.decomp
        assert decomp is not None
        if not self.ifmulti:
            data_time_idx = decomp["seasonal"].time.isin(self.data_season.time)
            self.data_season.value = pd.Series(
                decomp["seasonal"][data_time_idx].value,
                name=self.data_season.value.name,
                copy=False,
            )
            return self.data_season

        data_time_idx = decomp[str(0)]["seasonal"].time.isin(self.data_season.time)
        for i in range(self.num_seq):
            self.data_season.value.iloc[:, i] = pd.Series(
                decomp[str(i)]["seasonal"][data_time_idx].value,
                name=self.data_season.value.iloc[:, i].name,
                copy=False,
            )

        return self.data_season
